"""
The maize project's real pipeline: a small CNN classifier over four leaf
classes (Blight, Common_Rust, Gray_Leaf_Spot, Healthy), trained on a
bounded Kaggle corpus, reusing common_quant.py unchanged for export,
quantization, and FP32-vs-INT8 verification -- steps 5 through 7 of the
Full Spectrum recipe, the same steps mimii_anomaly.py exercises on audio.

Dataset: smaranjitghose/corn-or-maize-leaf-disease-dataset (Kaggle),
downloaded whole (161MB, small enough that a bounded partial-fetch trick
like MIMII's wasn't needed). Genuinely imbalanced across classes
(Gray_Leaf_Spot: 574 vs Common_Rust: 1306) -- exercises recipe step 5's
class-weighting note for real, not hypothetically.

**Known local-gap honesty note, not swept under the rug**: this dataset
carries no per-photo source/plant/field identifier, so recipe step 3's
"split by source, never by window" is honored only at the file level here
(stratified by class, no duplicate files across splits) -- there is no
metadata to split by a stronger unit than that. This is exactly the kind of
gap Full Spectrum's own project card for maize already names: the real local
work is a corpus tied to an agronomist's own treatment call, with real
provenance, not a downloaded Kaggle set.

Usage:
    python -m recipe_mentor.pipelines.maize_train
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

DATA_ROOT = Path(__file__).parent.parent / "data" / "maize_raw" / "data"
OUT_DIR = Path(__file__).parent.parent / "data" / "maize_run"
INPUT_NAME = "image"
IMG_SIZE = 64
CLASSES = ("Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy")
#: Bounded per-class caps -- real training data, not the full ~4200 images,
#: kept small enough for a fast CPU training loop in a weekend build.
N_TRAIN_PER_CLASS = 120
N_VAL_PER_CLASS = 20
N_TEST_PER_CLASS = 20


class SmallCNN(nn.Module):
    """Sized the same way as the audio autoencoder: enough real
    convolutional compute for INT8 to show a genuine win (recipe step 4)."""

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),   # 64 -> 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32 -> 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _list_images(cls: str) -> list[Path]:
    """
    List every image file for one class, exactly once.

    Not `Path.glob("*.jpg")` + `Path.glob("*.JPG")` concatenated: Windows'
    filesystem is case-insensitive, so both patterns match the identical
    set of files and every image is returned twice. That bug was caught by
    testing, not by inspection -- it doubled the reported class counts
    (which happened not to matter, since it doubled all four uniformly and
    class weights are ratios) but also meant the later `rng.shuffle(files)`
    was shuffling a list where each real file appears twice, with a real
    chance both copies of one file landed in different splits -- a literal
    train/val leak, in the exact step (recipe step 3) this demo exists to
    enforce. A single case-normalized suffix check has no such duplication.
    """
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in (DATA_ROOT / cls).iterdir() if p.suffix.lower() in exts)


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # HWC -> CHW


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray


def _build_splits(seed: int = 0) -> tuple[Split, Split, Split, dict[str, int]]:
    rng = random.Random(seed)
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    class_counts: dict[str, int] = {}

    for label, cls in enumerate(CLASSES):
        files = _list_images(cls)
        class_counts[cls] = len(files)
        rng.shuffle(files)
        need = N_TRAIN_PER_CLASS + N_VAL_PER_CLASS + N_TEST_PER_CLASS
        if len(files) < need:
            raise SystemExit(f"{cls}: only {len(files)} images, need {need}")

        train_files = files[:N_TRAIN_PER_CLASS]
        val_files = files[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS]
        test_files = files[N_TRAIN_PER_CLASS + N_VAL_PER_CLASS:need]

        for f in train_files:
            train_x.append(_load_image(f)); train_y.append(label)
        for f in val_files:
            val_x.append(_load_image(f)); val_y.append(label)
        for f in test_files:
            test_x.append(_load_image(f)); test_y.append(label)

    def stack(xs, ys):
        return Split(np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64))

    return stack(train_x, train_y), stack(val_x, val_y), stack(test_x, test_y), class_counts


@dataclass
class RunReport:
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
    preds = logits.argmax(axis=1)
    return float((preds == labels).mean())


def run(*, epochs: int = 15, batch_size: int = 16, lr: float = 1e-3) -> RunReport:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and resizing images (PIL, no torchvision)...")
    train, val, test, class_counts = _build_splits()
    print(f"train={len(train.y)} val={len(val.y)} test={len(test.y)}  "
          f"full-corpus class counts={class_counts}")

    # Recipe step 5: class-weight the loss since the raw data is imbalanced
    # (Gray_Leaf_Spot has roughly half Common_Rust's example count).
    counts = np.array([class_counts[c] for c in CLASSES], dtype=np.float32)
    weights = (counts.sum() / (len(CLASSES) * counts))
    print(f"Class weights (inverse frequency, full corpus): "
          f"{dict(zip(CLASSES, weights.round(3).tolist()))}")

    torch.manual_seed(0)
    model = SmallCNN(len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))

    train_x_t = torch.from_numpy(train.x)
    train_y_t = torch.from_numpy(train.y)
    val_x_t = torch.from_numpy(val.x)
    val_y_t = torch.from_numpy(val.y)

    print(f"Training CNN, {sum(p.numel() for p in model.parameters())} params...")
    final_loss = float("nan")
    best_val_state = None
    best_val_acc = -1.0
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
            val_logits = model(val_x_t).numpy()
        val_acc = _accuracy(val_logits, val.y)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 3 == 0:
            print(f"  epoch {epoch+1:3d}  train_loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")

    # Recipe step 5: keep the best-validation checkpoint, not the last epoch.
    model.load_state_dict(best_val_state)
    print(f"Restored best-validation checkpoint (val_acc={best_val_acc:.4f})")

    fp32_path = OUT_DIR / "model_fp32.onnx"
    int8_path = OUT_DIR / "model_int8.onnx"
    export_onnx(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), fp32_path, input_name=INPUT_NAME)
    print(f"Exported ONNX (dynamic batch axis) -> {fp32_path}")

    calib = [train.x[i:i + 1] for i in range(min(50, len(train.x)))]
    quantize_static_int8(fp32_path, int8_path, calib, input_name=INPUT_NAME)
    print(f"Quantized statically (QDQ INT8, calibrated on {len(calib)} real train examples) -> {int8_path}")

    test_batches = [test.x[i:i + 1] for i in range(len(test.x))]
    result = verify_fp32_vs_int8(
        fp32_path, int8_path, test_batches, test.y,
        score_fn=lambda out: out,      # raw logits per example
        metric_fn=_accuracy,
        input_name=INPUT_NAME,
    )
    print(f"FP32 accuracy={result.fp32_metric:.4f}  INT8 accuracy={result.int8_metric:.4f}  "
          f"verdict={result.verdict}")

    report = RunReport(
        n_train=len(train.y), n_val=len(val.y), n_test=len(test.y),
        class_counts_full_corpus=class_counts,
        class_weights_used=weights.round(4).tolist(),
        train_loss_final=final_loss,
        fp32_accuracy=result.fp32_metric, int8_accuracy=result.int8_metric,
        verdict=result.verdict,
        fp32_onnx_bytes=fp32_path.stat().st_size, int8_onnx_bytes=int8_path.stat().st_size,
    )
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\nWrote report -> {report_path}")
    print(f"Model size: FP32 {report.fp32_onnx_bytes/1024:.1f} KB -> "
          f"INT8 {report.int8_onnx_bytes/1024:.1f} KB "
          f"({report.fp32_onnx_bytes/max(report.int8_onnx_bytes,1):.2f}x smaller)")
    return report


if __name__ == "__main__":
    run()
