"""
Generic conv-autoencoder anomaly detector, generalizing mimii_anomaly.py to
any Kaggle dataset the agent's detect_task_type() tool recognizes as
normal/abnormal-labeled audio. Same architecture, same anomaly-only
training paradigm, same common_quant.py tail. Reuses features.py's mel math
unchanged; only wraps the file-loading step to tolerate a sample rate other
than MIMII's native 16kHz, via plain-NumPy linear-interpolation resampling
-- no new dependency, consistent with recipe step 2's "portable math" rule.

Split discipline (recipe step 3): if the discovered normal/ directory
itself contains source-id subfolders (mirroring MIMII's own id_00/id_02
layout), train and validation are split by whole subfolder, never mixing
one source's clips across the split. Otherwise -- a flat folder with no
source metadata, the same honest gap maize_train.py's own local-gap note
already names for images -- this falls back to a file-level split, recorded
as such rather than silently presented as source-level.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from .common_quant import (
    confidence_bands_from_validation, export_onnx, quantize_static_int8,
    roc_auc, verify_fp32_vs_int8,
)
from .features import CLIP_SAMPLES, SAMPLE_RATE, feature_shape, log_mel_spectrogram

INPUT_NAME = "log_mel"
MAX_TEST_ABNORMAL = 150
VAL_NORMAL_FRACTION = 0.15


class ConvAutoencoder(nn.Module):
    """Identical to mimii_anomaly.py's own architecture -- three stride-2
    downsampling convs, mirrored back up. Recipe step 4: sized to give
    INT8 a genuine amount of real convolutional compute to accelerate."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _load_mono_resampled(path: Path) -> np.ndarray:
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Deliberate simplification, not a mastering-grade resampler: plain
        # linear interpolation is adequate for mel-spectrogram features at
        # this scale, and keeps this module free of a librosa/scipy
        # dependency the same way features.py already is.
        n_target = max(1, int(round(len(mono) * SAMPLE_RATE / sr)))
        mono = np.interp(
            np.linspace(0, len(mono) - 1, n_target), np.arange(len(mono)), mono,
        ).astype(np.float32)
    if len(mono) < CLIP_SAMPLES:
        mono = np.pad(mono, (0, CLIP_SAMPLES - len(mono)))
    else:
        mono = mono[:CLIP_SAMPLES]
    return mono


def _extract(path: Path) -> np.ndarray:
    return log_mel_spectrogram(_load_mono_resampled(path))


def _features_for(paths: list[Path]) -> np.ndarray:
    n_mels, n_frames = feature_shape()
    out = np.empty((len(paths), 1, n_mels, n_frames), dtype=np.float32)
    for i, p in enumerate(paths):
        out[i, 0] = _extract(p)
    return out


def _normalize(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def _discover_source_split(normal_dirs: list[Path], seed: int = 0) -> tuple[list[Path], list[Path], bool]:
    """Returns (train_files, held_out_files, split_by_source)."""
    rng = random.Random(seed)
    source_groups: dict[str, list[Path]] = {}
    flat_files: list[Path] = []
    for d in normal_dirs:
        for child in sorted(d.iterdir()):
            if child.is_dir():
                wavs = sorted(child.rglob("*.wav"))
                if wavs:
                    source_groups.setdefault(child.name, []).extend(wavs)
            elif child.suffix.lower() == ".wav":
                flat_files.append(child)

    if len(source_groups) >= 2:
        keys = sorted(source_groups)
        rng.shuffle(keys)
        n_held_out_sources = max(1, len(keys) // 4)
        held_out_keys, train_keys = keys[:n_held_out_sources], keys[n_held_out_sources:]
        train_files = [f for k in train_keys for f in source_groups[k]]
        held_out_files = [f for k in held_out_keys for f in source_groups[k]]
        return train_files, held_out_files, True

    all_normal = flat_files + [f for group in source_groups.values() for f in group]
    rng.shuffle(all_normal)
    n_held_out = max(2, int(len(all_normal) * VAL_NORMAL_FRACTION * 2))
    return all_normal[n_held_out:], all_normal[:n_held_out], False


@dataclass
class RunReport:
    n_params: int
    split_by_source: bool
    n_train: int
    n_val: int
    n_test_normal: int
    n_test_abnormal: int
    train_loss_final: float
    fp32_auc: float
    int8_auc: float
    verdict: str
    clean_below: float
    uncertain_below: float
    fp32_onnx_bytes: int
    int8_onnx_bytes: int


def run(
    normal_dirs: list[Path], abnormal_dirs: list[Path], out_dir: Path,
    *, epochs: int = 150, batch_size: int = 16, lr: float = 1e-3,
) -> RunReport:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, held_out_normal, split_by_source = _discover_source_split(normal_dirs)
    rng = random.Random(1)
    rng.shuffle(held_out_normal)
    half = max(1, len(held_out_normal) // 2)
    test_normal_files = held_out_normal[:half]
    val_files = held_out_normal[half:] or held_out_normal[:1]

    abnormal_files = sorted({p for d in abnormal_dirs for p in d.rglob("*.wav")})
    rng.shuffle(abnormal_files)
    test_abnormal_files = abnormal_files[:MAX_TEST_ABNORMAL]

    print(
        f"train={len(train_files)} val={len(val_files)} "
        f"test_normal={len(test_normal_files)} test_abnormal={len(test_abnormal_files)} "
        f"split_by_source={split_by_source}"
    )
    if not train_files:
        raise SystemExit("no normal training files found")

    print("Extracting features (pure NumPy + soundfile, no librosa)...")
    x_train = _features_for(train_files)
    x_val = _features_for(val_files) if val_files else x_train[:0]
    x_test_normal = _features_for(test_normal_files)
    x_test_abnormal = _features_for(test_abnormal_files)

    # Normalize using TRAIN statistics only -- val/test never leak into this.
    mean, std = float(x_train.mean()), float(x_train.std())
    x_train_n = _normalize(x_train, mean, std)
    x_val_n = _normalize(x_val, mean, std) if len(x_val) else x_val
    x_test_normal_n = _normalize(x_test_normal, mean, std)
    x_test_abnormal_n = _normalize(x_test_abnormal, mean, std)

    torch.manual_seed(0)
    model = ConvAutoencoder()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_t = torch.from_numpy(x_train_n)
    val_t = torch.from_numpy(x_val_n) if len(x_val_n) else None

    print(f"Training conv autoencoder, anomaly-only, {n_params} params...")
    final_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_t))
        epoch_loss = 0.0
        for i in range(0, len(train_t), batch_size):
            idx = perm[i:i + batch_size]
            batch = train_t[idx]
            opt.zero_grad()
            out = model(batch)
            loss = ((out - batch) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= len(train_t)
        final_loss = epoch_loss
        if val_t is not None and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = ((model(val_t) - val_t) ** 2).mean().item()
            print(f"  epoch {epoch + 1:3d}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}")

    fp32_path, int8_path = out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx"
    export_onnx(model, torch.randn(1, 1, *feature_shape()), fp32_path, input_name=INPUT_NAME)
    calib = [x_train_n[i:i + 1] for i in range(min(50, len(x_train_n)))]
    quantize_static_int8(fp32_path, int8_path, calib, input_name=INPUT_NAME)

    eval_x = np.concatenate([x_test_normal_n, x_test_abnormal_n], axis=0)
    labels = np.array([0] * len(x_test_normal_n) + [1] * len(x_test_abnormal_n))
    batches = [eval_x[i:i + 1] for i in range(len(eval_x))]

    def score_fn(model_out: np.ndarray) -> np.ndarray:
        return ((model_out - eval_x) ** 2).mean(axis=(1, 2, 3))

    result = verify_fp32_vs_int8(fp32_path, int8_path, batches, labels, score_fn, roc_auc, input_name=INPUT_NAME)
    print(f"FP32 AUC={result.fp32_metric:.4f}  INT8 AUC={result.int8_metric:.4f}  verdict={result.verdict}")

    normal_scores_fp32 = result.fp32_scores[labels == 0]
    bands = confidence_bands_from_validation(normal_scores_fp32)

    report = RunReport(
        n_params=n_params, split_by_source=split_by_source,
        n_train=len(train_files), n_val=len(val_files),
        n_test_normal=len(test_normal_files), n_test_abnormal=len(test_abnormal_files),
        train_loss_final=final_loss,
        fp32_auc=result.fp32_metric, int8_auc=result.int8_metric, verdict=result.verdict,
        clean_below=bands.clean_below, uncertain_below=bands.uncertain_below,
        fp32_onnx_bytes=fp32_path.stat().st_size, int8_onnx_bytes=int8_path.stat().st_size,
    )
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\nWrote report -> {out_dir / 'report.json'}")
    return report
