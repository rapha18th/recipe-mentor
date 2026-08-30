"""
The deterministic backbone behind the autonomous agent's five tools.

AgentToolkit holds all real state (dataset paths, detected task type,
hyperparameters, trained model paths) in `self.state` -- the LLM never sees
or passes any of it directly, only small bounded scalars (owner_slug,
epochs, batch_size, lr). Every method does real work (a real Kaggle
download, real training, real ONNX export/static-INT8 quantize/verify via
common_quant.py) and returns a small JSON-safe dict, including a
`step_results` mapping of {recipe_step_number: (status, detail)} that the
caller (agent_runner.py) turns into passport writes -- the actual proof
that a tool call happened, keyed off what the tool did, never off what the
model said.

Out-of-order calls return a structured error dict rather than raising --
ADK feeds that back to the model as the function's own response, so the
agent can see the mistake and correct it. That's real (and good demo
material) agentic error recovery, not a crash.

Exercise this class directly with a plain script, no ADK, no LLM, before
wiring any agent on top -- it's the majority of the actual work, and the
part that has to be right independent of whether the model calls it in
the expected order.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pipelines import generic_audio_anomaly_train, generic_image_train
from ..pipelines.kaggle_fetch import fetch_dataset
from .bounds import clamp
from .detect import detect_task_type as _detect_task_type

AGENT_DATA_ROOT = Path(__file__).parent.parent / "data" / "agent_runs"


def _describe_dataset(info: dict) -> str:
    if info["task_type"] == "image_classification":
        return f"Detected image classification: {len(info['classes'])} classes ({', '.join(info['classes'])})."
    return f"Detected audio anomaly detection: {info['n_normal']} normal, {info['n_abnormal']} abnormal clips."


@dataclass
class AgentToolkit:
    #: All state a tool call needs, never round-tripped through the model.
    state: dict[str, Any] = field(default_factory=dict)

    # -- 1: fetch -------------------------------------------------------------

    def fetch_kaggle_dataset(self, owner_slug: str) -> dict:
        """Download a Kaggle dataset by 'owner/slug' and make it available
        for the rest of the pipeline. Call this first."""
        result = fetch_dataset(owner_slug)
        self.state["dataset_ref"] = owner_slug
        self.state["dataset_path"] = result["path"]
        detail = (
            f"Fetched {owner_slug}: {result['n_files']} files, "
            f"{result['total_bytes'] / 1e6:.1f} MB"
            + (" (already cached locally)" if result["already_cached"] else "")
        )
        return {**result, "step_results": {1: ("done", detail)}}

    # -- 2: detect --------------------------------------------------------------

    def detect_task_type(self) -> dict:
        """Inspect the fetched dataset and identify its task type
        (image_classification or audio_anomaly) and shape. Call this after
        fetch_kaggle_dataset. Returns an error if the dataset doesn't match
        either supported shape."""
        if "dataset_path" not in self.state:
            return {"error": "call fetch_kaggle_dataset first"}

        info = _detect_task_type(Path(self.state["dataset_path"]))
        if "error" in info:
            return info

        self.state["task_type"] = info["task_type"]
        self.state["dataset_info"] = info
        feature_note = (
            "Pure NumPy + soundfile mel-spectrogram, no librosa."
            if info["task_type"] == "audio_anomaly"
            else "PIL resize + NumPy normalize, no torchvision transform pipeline."
        )
        return {**info, "step_results": {2: ("done", f"{_describe_dataset(info)} Feature pipeline: {feature_note}")}}

    # -- 3: configure -----------------------------------------------------------

    def configure_run(self, epochs: int, batch_size: int, lr: float) -> dict:
        """Propose training hyperparameters for the detected task type.
        Values outside a safe range are clamped and reported back. Call
        this after detect_task_type."""
        if "task_type" not in self.state:
            return {"error": "call detect_task_type first"}

        clamped, notes = clamp(self.state["task_type"], epochs, batch_size, lr)
        self.state["hparams"] = clamped
        detail = f"epochs={clamped['epochs']} batch_size={clamped['batch_size']} lr={clamped['lr']}"
        if notes:
            detail += " (clamped: " + "; ".join(notes.values()) + ")"
        return {"hyperparameters": clamped, "clamped": notes, "step_results": {}}

    # -- 4: train + quantize + verify (inseparable) ------------------------------

    def train_and_verify(self) -> dict:
        """Run the full training pipeline for the detected task type:
        train, export to ONNX, quantize to static INT8, and verify
        FP32-vs-INT8 accuracy on a held-out split. Verification always
        runs -- there is no tool that would let you skip it. Call this
        after configure_run."""
        if "hparams" not in self.state:
            return {"error": "call configure_run first"}

        info = self.state["dataset_info"]
        hp = self.state["hparams"]
        task_type = self.state["task_type"]
        out_dir = AGENT_DATA_ROOT / "_run" / self.state["dataset_ref"].replace("/", "__")

        if task_type == "image_classification":
            report = generic_image_train.run(
                Path(info["root"]), info["classes"], out_dir,
                epochs=hp["epochs"], batch_size=hp["batch_size"], lr=hp["lr"],
            )
            fp32_metric, int8_metric, metric_name = report.fp32_accuracy, report.int8_accuracy, "accuracy"
            split_note = (
                "done",
                "File-level split, stratified by class only -- no photo/source metadata "
                "available in this dataset (the same known gap the maize pipeline's own "
                "local-gap note names).",
            )
        else:
            report = generic_audio_anomaly_train.run(
                [Path(d) for d in info["normal_dirs"]], [Path(d) for d in info["abnormal_dirs"]], out_dir,
                epochs=hp["epochs"], batch_size=hp["batch_size"], lr=hp["lr"],
            )
            fp32_metric, int8_metric, metric_name = report.fp32_auc, report.int8_auc, "auc"
            split_note = (
                ("done", "Split by source subfolder (train/val use disjoint sources), mirroring "
                         "the MIMII id_00/id_02 pattern.")
                if report.split_by_source else
                ("corrected", "No source subfolders found under the normal/ directory; fell back "
                              "to a file-level split, stratified only by class -- a real gap, "
                              "recorded rather than hidden.")
            )

        self.state["report"] = report
        self.state["report_dict"] = dataclasses.asdict(report)

        step_results = {
            3: split_note,
            4: ("done", f"Reused the {task_type.replace('_', ' ')} convolutional architecture already "
                        f"proven to show a genuine static-INT8 win on this pipeline family "
                        f"({report.n_params} params)."),
            5: ("done", f"Trained {hp['epochs']} epochs; best-validation checkpoint restored. "
                        f"FP32 {metric_name}={fp32_metric:.4f}."),
            6: ("done", "Quantized statically (QDQ INT8, calibrated on real training examples), "
                        "not dynamically."),
            7: ("done", f"FP32 {metric_name}={fp32_metric:.4f}, INT8 {metric_name}={int8_metric:.4f}, "
                        f"verdict: {report.verdict}."),
        }
        if hasattr(report, "clean_below"):
            step_results[11] = (
                "done",
                f"Confidence bands read from the validation-normal distribution: "
                f"clean<{report.clean_below:.4f}, uncertain<{report.uncertain_below:.4f}.",
            )

        return {
            "metric_name": metric_name, "fp32": fp32_metric, "int8": int8_metric,
            "verdict": report.verdict, "step_results": step_results,
        }

    # -- 5: record ----------------------------------------------------------------

    def record_results(self) -> dict:
        """Finalize this run: mark the on-device steps (8-10) as not
        attempted, since this run never touches a physical device, and
        report what's not yet done. Call this last."""
        if "report_dict" not in self.state:
            return {"error": "call train_and_verify first"}

        step_results = {
            8: ("not_attempted", "No physical device in this autonomous run."),
            9: ("not_attempted", "No physical device in this autonomous run."),
            10: ("not_attempted", "No physical device in this autonomous run."),
            12: ("done", "Export/quantize/verify confirmed on this machine only; "
                         "on-device steps 8-10 remain."),
        }
        self.state["finalized"] = True
        return {
            "report": self.state["report_dict"],
            "dataset_ref": self.state["dataset_ref"],
            "task_type": self.state["task_type"],
            "step_results": step_results,
        }
