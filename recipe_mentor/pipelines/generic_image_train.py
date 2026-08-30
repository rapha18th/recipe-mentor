"""
Generic small-CNN image classifier, generalizing maize_train.py's pipeline
to any class-labeled image folder the agent's detect_task_type() tool
finds. Same architecture, same common_quant.py tail (export -> static QDQ
INT8 -> FP32-vs-INT8 verify) -- the literal "reuses the pipeline, unchanged,
retargeted" claim Full Spectrum's own project cards make, now proven for a
dataset picked at run time, not one hand-selected in advance.

Per-class split sizes scale to the smallest usable class's own count
(bounded), rather than maize_train.py's fixed 120/20/20 -- a dataset with
fewer than 160 images per class would otherwise hard-fail here.

Known local-gap honesty note, same one maize_train.py's own docstring
names: most Kaggle image sets carry no per-photo source/subject identifier,
so recipe step 3's "split by source, never by window" is honored only at
the file level -- stratified by class, no duplicate files across splits --
because there is no metadata to split by a stronger unit than that.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .common_quant import export_onnx, quantize_static_int8, verify_fp32_vs_int8

IMG_SIZE = 64
MIN_PER_CLASS = 20
MAX_TRAIN_PER_CLASS = 150
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class SmallCNN(nn.Module):
    """Same shape as maize_train.py's own SmallCNN -- enough real
    convolutional compute for INT8 to show a genuine win (recipe step 4)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _list_images(class_dir: Path) -> list[Path]:
    # Case-normalized suffix check, not Path.glob("*.jpg") + glob("*.JPG")
    # concatenated -- Windows' case-insensitive filesystem would otherwise
    # return every file twice (the exact bug maize_train.py's own docstring
    # documents catching).
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # HWC -> CHW


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray


def _split_sizes(n_available: int) -> tuple[int, int, int]:
    n_available = min(
        n_available,
        MAX_TRAIN_PER_CLASS + int(MAX_TRAIN_PER_CLASS * (VAL_FRACTION + TEST_FRACTION) / (1 - VAL_FRACTION - TEST_FRACTION)),
    )
    n_val = max(1, int(n_available * VAL_FRACTION))
    n_test = max(1, int(n_available * TEST_FRACTION))
    n_train = n_available - n_val - n_test
    return n_train, n_val, n_test


def _build_splits(root: Path, classes: list[str], seed: int = 0):
    rng = random.Random(seed)
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    class_counts: dict[str, int] = {}

    for label, cls in enumerate(classes):
        files = _list_images(root / cls)
        class_counts[cls] = len(files)
        if len(files) < MIN_PER_CLASS:
            raise SystemExit(f"{cls}: only {len(files)} images, need >= {MIN_PER_CLASS}")
        rng.shuffle(files)
        n_train, n_val, n_test = _split_sizes(len(files))

        for f in files[:n_train]:
            train_x.append(_load_image(f)); train_y.append(label)
        for f in files[n_train:n_train + n_val]:
            val_x.append(_load_image(f)); val_y.append(label)
        for f in files[n_train + n_val:n_train + n_val + n_test]:
            test_x.append(_load_image(f)); test_y.append(label)

    def stack(xs, ys):
        return Split(np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64))

    return stack(train_x, train_y), stack(val_x, val_y), stack(test_x, test_y), class_counts


@dataclass
class RunReport:
    classes: list
    n_params: int
    n_train: int
    n_val: int
    n_test: int
    class_counts_full_corpus: dict
    class_weights_used: list
    train_loss_final: float
    fp32_accuracy: float
    int8_accuracy: float
    verdict: str
    fp32_onnx_bytes: int
    int8_onnx_bytes: int


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def run(
    root: Path, classes: list[str], out_dir: Path, *, epochs: int = 15, batch_size: int = 16, lr: float = 1e-3,
) -> RunReport:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {root} ({len(classes)} classes)...")
    train, val, test, class_counts = _build_splits(root, classes)
    print(f"train={len(train.y)} val={len(val.y)} test={len(test.y)}  class counts={class_counts}")

    counts = np.array([class_counts[c] for c in classes], dtype=np.float32)
    weights = counts.sum() / (len(classes) * counts)
    print(f"Class weights (inverse frequency): {dict(zip(classes, weights.round(3).tolist()))}")

    torch.manual_seed(0)
    model = SmallCNN(len(classes))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))

    train_x_t, train_y_t = torch.from_numpy(train.x), torch.from_numpy(train.y)
    val_x_t, val_y_t = torch.from_numpy(val.x), torch.from_numpy(val.y)

    print(f"Training CNN, {n_params} params...")
    final_loss = float("nan")
    best_state, best_val_acc = None, -1.0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x_t))
        epoch_loss = 0.0
        for i in range(0, len(train_x_t), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(train_x_t[idx])
            loss = loss_fn(out, train_y_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= len(train_x_t)
        final_loss = epoch_loss

        model.eval()
        with torch.no_grad():
            val_acc = _accuracy(model(val_x_t).numpy(), val.y)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 3 == 0:
            print(f"  epoch {epoch + 1:3d}  train_loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"Restored best-validation checkpoint (val_acc={best_val_acc:.4f})")

    fp32_path, int8_path = out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx"
    export_onnx(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), fp32_path, input_name="image")
    calib = [train.x[i:i + 1] for i in range(min(50, len(train.x)))]
    quantize_static_int8(fp32_path, int8_path, calib, input_name="image")

    test_batches = [test.x[i:i + 1] for i in range(len(test.x))]
    result = verify_fp32_vs_int8(
        fp32_path, int8_path, test_batches, test.y,
        score_fn=lambda out: out, metric_fn=_accuracy, input_name="image",
    )
    print(f"FP32 accuracy={result.fp32_metric:.4f}  INT8 accuracy={result.int8_metric:.4f}  verdict={result.verdict}")

    report = RunReport(
        classes=classes, n_params=n_params,
        n_train=len(train.y), n_val=len(val.y), n_test=len(test.y),
        class_counts_full_corpus=class_counts, class_weights_used=weights.round(4).tolist(),
        train_loss_final=final_loss, fp32_accuracy=result.fp32_metric, int8_accuracy=result.int8_metric,
        verdict=result.verdict,
        fp32_onnx_bytes=fp32_path.stat().st_size, int8_onnx_bytes=int8_path.stat().st_size,
    )
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\nWrote report -> {out_dir / 'report.json'}")
    return report
