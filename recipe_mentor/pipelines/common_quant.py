"""
Shared ONNX export / static QDQ INT8 quantization / FP32-vs-INT8
verification / confidence-band module -- recipe steps 5, 6, 7, and 11.

Used unchanged by both mimii_anomaly.py and maize_train.py. That reuse is
the literal proof behind Full Spectrum's own claim that a project card
"reuses the [prior project's] pipeline, unchanged, retargeted" -- this
module is the part that's actually unchanged, not just described as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import onnxruntime.quantization  # noqa: F401 -- not auto-imported by `import onnxruntime`
import torch


def export_onnx(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    path: Path,
    *,
    input_name: str = "input",
    output_name: str = "output",
) -> Path:
    """Recipe step 5: export with a dynamic batch axis so the same graph
    serves training-time batches and single-example on-device inference.

    `dynamo=False` is deliberate: torch 2.13's default dynamo-based
    exporter (a) warns that `dynamic_axes` is unsupported under it and
    wants `dynamic_shapes` instead, and (b) crashes outright on Windows --
    its verbose success message prints a checkmark emoji that Windows'
    default cp1252 console encoding can't encode, raising
    UnicodeEncodeError before export even completes. The legacy
    TorchScript-based exporter (`dynamo=False`) has neither problem and is
    what this module's `dynamic_axes`-based API already assumes.
    """
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample_input,
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


class _CalibrationReader(ort.quantization.CalibrationDataReader):
    """Feeds real calibration examples to static quantization -- recipe
    step 6 requires calibration on real data, not a synthetic distribution."""

    def __init__(self, samples: list[np.ndarray], input_name: str):
        self._iter = iter(samples)
        self._input_name = input_name

    def get_next(self):
        sample = next(self._iter, None)
        return None if sample is None else {self._input_name: sample}


def quantize_static_int8(
    fp32_path: Path,
    int8_path: Path,
    calibration_samples: list[np.ndarray],
    *,
    input_name: str = "input",
) -> Path:
    """
    Recipe step 6: static QDQ INT8, not dynamic. ONNX Runtime's dynamic
    quantization only touches MatMul/Gemm by default and leaves a
    convolutional model almost entirely in FP32 -- static QDQ, calibrated on
    real examples, is what actually quantizes the convolution layers, and
    what an INT8 dot-product path is built to accelerate.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    reader = _CalibrationReader(calibration_samples, input_name)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    return int8_path


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    ROC-AUC via the Mann-Whitney U statistic -- no sklearn dependency for
    one metric. `labels`: 1 = anomalous, 0 = normal. Higher `scores` means
    more anomalous (reconstruction error, for an autoencoder).
    """
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([neg, pos])
    ranks = np.argsort(np.argsort(combined)) + 1
    rank_pos_sum = ranks[len(neg):].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class VerifyResult:
    fp32_scores: np.ndarray
    int8_scores: np.ndarray
    fp32_metric: float
    int8_metric: float
    n_eval: int
    verdict: str


def _run_onnx(path: Path, input_name: str, batches: list[np.ndarray]) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, {input_name: b})[0] for b in batches], axis=0)


def verify_fp32_vs_int8(
    fp32_path: Path,
    int8_path: Path,
    eval_batches: list[np.ndarray],
    labels: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc,
    *,
    input_name: str = "input",
    noise_floor_n: int = 200,
) -> VerifyResult:
    """
    Recipe step 7: compare FP32 against INT8 on the SAME held-out data.
    `score_fn(raw_model_output) -> per_example_score` turns model output
    into one anomaly score per example (e.g. reconstruction error);
    `metric_fn(scores, labels) -> scalar` turns that into one number
    (default: ROC-AUC).

    A held-out set below `noise_floor_n` gets its verdict downgraded to
    "no_loss" even on a nominal INT8 win -- the paper's own rule: "a
    validation set small enough that INT8 edges out FP32 by noise is a
    validation set too small to claim a win from either direction; report
    it as no loss, not as an improvement."
    """
    fp32_scores = score_fn(_run_onnx(fp32_path, input_name, eval_batches))
    int8_scores = score_fn(_run_onnx(int8_path, input_name, eval_batches))

    fp32_metric = metric_fn(fp32_scores, labels)
    int8_metric = metric_fn(int8_scores, labels)
    n_eval = len(labels)
    delta = int8_metric - fp32_metric

    if n_eval < noise_floor_n:
        verdict = "no_loss (n too small to distinguish signal from noise)"
    elif delta < -0.01:
        verdict = "regressed"
    elif delta > 0.01:
        verdict = "improved (re-verify on a larger split before trusting this)"
    else:
        verdict = "no_loss"

    return VerifyResult(fp32_scores, int8_scores, fp32_metric, int8_metric, n_eval, verdict)


@dataclass
class ConfidenceBands:
    clean_below: float
    uncertain_below: float
    #: percentile of NORMAL validation scores each band is read from
    p_clean: float
    p_uncertain: float


def confidence_bands_from_validation(
    normal_scores: np.ndarray, *, p_clean: float = 0.90, p_uncertain: float = 0.99
) -> ConfidenceBands:
    """
    Recipe step 11: read confidence bands from where the validation-normal
    score distribution actually stops separating cleanly, not an assumed
    50% cutoff. Mirrors SiloSense's own three-tier read-out (likely clean /
    uncertain / likely anomalous).
    """
    return ConfidenceBands(
        clean_below=float(np.percentile(normal_scores, p_clean * 100)),
        uncertain_below=float(np.percentile(normal_scores, p_uncertain * 100)),
        p_clean=p_clean, p_uncertain=p_uncertain,
    )
