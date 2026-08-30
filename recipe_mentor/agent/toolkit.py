"""
The deterministic backbone behind the autonomous agent's tools: fetch,
detect, explore, ask_user, configure_run, train_and_verify, record_results.

AgentToolkit holds all real state (dataset paths, detected task type,
exploration findings, hyperparameters, trained model paths) in
`self.state` -- the LLM never sees or passes any of it directly, only
small bounded scalars (owner_slug, epochs, batch_size, lr, a question and
its answer). Every method does real work (a real Kaggle download, real
dataset sampling, real training, real ONNX export/static-INT8
quantize/verify via common_quant.py) and returns a small JSON-safe dict,
including a `step_results` mapping of {recipe_step_number: (status,
detail)} that the caller (agent_runner.py) turns into passport writes --
the actual proof that a tool call happened, keyed off what the tool did,
never off what the model said.

`ask_user()` is the one tool that genuinely waits: it's async, and it
blocks on a real `asyncio.Future` until a human answer arrives from
outside (agent_runner.py's `input()`, or web/app.py's `/choice`
endpoint), via `answer_pending()`. This is what makes the agent actually
collaborative rather than a one-shot batch job with a status feed --
explore_dataset's real findings, plus whatever cross_project_recall
surfaced at kickoff, are what a call to ask_user should be grounded in,
per the system instruction in agent/core.py.

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

import asyncio
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..pipelines import generic_audio_anomaly_train, generic_image_train
from ..pipelines.kaggle_fetch import fetch_dataset
from .bounds import clamp
from .detect import detect_task_type as _detect_task_type
from .explore import explore_audio_anomaly, explore_image_classification

AGENT_DATA_ROOT = Path(__file__).parent.parent / "data" / "agent_runs"


def _describe_dataset(info: dict) -> str:
    if info["task_type"] == "image_classification":
        return f"Detected image classification: {len(info['classes'])} classes ({', '.join(info['classes'])})."
    return f"Detected audio anomaly detection: {info['n_normal']} normal, {info['n_abnormal']} abnormal clips."


@dataclass
class AgentToolkit:
    #: All state a tool call needs, never round-tripped through the model.
    state: dict[str, Any] = field(default_factory=dict)
    #: Whatever ask_user() is currently waiting on -- resolved from outside
    #: (agent_runner.py's input(), or web/app.py's /choice endpoint) via
    #: answer_pending(), never by the model itself.
    _pending: "asyncio.Future | None" = field(default=None, init=False, repr=False, compare=False)
    #: Fired synchronously the instant ask_user() actually starts waiting
    #: (not when the model merely decides to call it -- those are two
    #: different moments; see agent/core.py::run_agent's own comment on
    #: why that distinction is load-bearing, not pedantic).
    on_awaiting_choice: "Callable[[str, list[str]], None] | None" = field(
        default=None, repr=False, compare=False,
    )

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

    # -- 3: explore ---------------------------------------------------------------

    def explore_dataset(self) -> dict:
        """Sample the fetched dataset and report real, empirical findings
        -- class balance, sample-rate consistency, corrupt files, image
        dimension spread -- not just its shape. Call this after
        detect_task_type. Ground your next ask_user() call in what this
        actually finds about THIS dataset, not a generic question."""
        if "task_type" not in self.state:
            return {"error": "call detect_task_type first"}

        info = self.state["dataset_info"]
        if self.state["task_type"] == "image_classification":
            findings = explore_image_classification(Path(info["root"]), info["classes"], info["class_counts"])
        else:
            findings = explore_audio_anomaly(
                [Path(d) for d in info["normal_dirs"]], [Path(d) for d in info["abnormal_dirs"]],
            )
        self.state["exploration"] = findings
        return {**findings, "step_results": {}}

    # -- ask: a real, waited-on question, not a rhetorical one --------------------

    async def ask_user(self, question: str, options: list[str] | None = None) -> dict:
        """Ask the person running this a real, specific question -- a
        preprocessing or configuration decision grounded in what
        explore_dataset found and what past projects' recorded lessons
        (given to you at the start of this run) suggest. This call
        genuinely waits for a real answer before returning -- use it
        whenever there's an actual choice worth surfacing, rather than
        deciding silently and only reporting the outcome. `options` is a
        short list of concrete choices when there are natural ones; leave
        it empty for an open question."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending = fut
        # Fire the notification only now that self._pending is genuinely
        # set -- not when the model decided to call this tool (that
        # happens one ADK event earlier, before this coroutine body has
        # even started running). Notifying too early would let a harness
        # "answer" a future that doesn't exist yet.
        if self.on_awaiting_choice:
            self.on_awaiting_choice(question, list(options or []))
        answer = await fut
        self._pending = None
        return {"answer": answer, "step_results": {}}

    def answer_pending(self, answer: str) -> bool:
        """Called by the harness (CLI input(), or the web /choice
        endpoint) once a real human answer arrives -- resolves whatever
        ask_user() call is currently waiting. Returns False if nothing
        was actually waiting (a late or duplicate answer)."""
        if self._pending is not None and not self._pending.done():
            self._pending.set_result(answer)
            return True
        return False

    # -- 4: configure -----------------------------------------------------------

    def configure_run(self, epochs: int, batch_size: int, lr: float, image_size: int | None = None) -> dict:
        """Propose training hyperparameters for the detected task type.
        Values outside a safe range are clamped and reported back.
        `image_size` is only meaningful for image_classification (the
        input resolution images get resized to before training -- ignored
        for audio_anomaly); pass it when the user's ask_user() answer
        actually concerned resolution. Call this after detect_task_type,
        and after ask_user if you called it."""
        if "task_type" not in self.state:
            return {"error": "call detect_task_type first"}

        extra = {"image_size": image_size} if image_size is not None else {}
        clamped, notes = clamp(self.state["task_type"], epochs, batch_size, lr, **extra)
        self.state["hparams"] = clamped
        detail = f"epochs={clamped['epochs']} batch_size={clamped['batch_size']} lr={clamped['lr']}"
        if "image_size" in clamped:
            detail += f" image_size={clamped['image_size']}"
        if notes:
            detail += " (clamped: " + "; ".join(notes.values()) + ")"
        return {"hyperparameters": clamped, "clamped": notes, "step_results": {}}

    # -- 5: train + quantize + verify (inseparable) ------------------------------

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
            image_kwargs = {"image_size": hp["image_size"]} if "image_size" in hp else {}
            report = generic_image_train.run(
                Path(info["root"]), info["classes"], out_dir,
                epochs=hp["epochs"], batch_size=hp["batch_size"], lr=hp["lr"], **image_kwargs,
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
                        f"({report.n_params} params)"
                        + (f", input resolution {hp['image_size']}x{hp['image_size']} as decided with the "
                           f"user." if "image_size" in hp else ".")),
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

    # -- 6: record ----------------------------------------------------------------

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
