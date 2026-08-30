"""
Deterministic dataset-shape detection -- pure filesystem logic, no model,
no network. Given a fetched Kaggle dataset's local folder, decides which of
the two supported task types it matches, and returns the concrete paths the
matching pipeline needs. An unrecognized shape returns a plain error dict
rather than guessing -- the agent's train_and_verify() tool has exactly two
real pipelines behind it (generic_image_train.py, generic_audio_anomaly_
train.py), and running either against data it wasn't built for would be a
silent correctness bug, not a shortcut worth taking under deadline.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTS = {".wav"}
NORMAL_ALIASES = {"normal", "ok", "good"}
ABNORMAL_ALIASES = {"abnormal", "anomaly", "anomalous", "ng", "bad", "fault", "faulty"}
MIN_IMAGES_PER_CLASS = 10
MIN_CLASSES = 2
DOMINANT_FRACTION = 0.5


def _find_binary_audio_dirs(root: Path) -> tuple[list[Path], list[Path]]:
    normal, abnormal = [], []
    for d in (root, *[p for p in root.rglob("*") if p.is_dir()]):
        name = d.name.lower()
        if name not in NORMAL_ALIASES and name not in ABNORMAL_ALIASES:
            continue
        has_wavs = any(p.suffix.lower() == ".wav" for p in d.rglob("*") if p.is_file())
        if not has_wavs:
            continue
        (normal if name in NORMAL_ALIASES else abnormal).append(d)
    return normal, abnormal


def _find_class_folders(root: Path) -> tuple[Path | None, list[str]]:
    """Find the directory whose direct subdirectories are each full of
    images -- that's the class-folder root. Prefers the match with the
    most usable classes, then the shallowest path."""
    candidates: dict[Path, list[str]] = {}
    for d in (root, *[p for p in root.rglob("*") if p.is_dir()]):
        subdirs = [c for c in d.iterdir() if c.is_dir()]
        if len(subdirs) < MIN_CLASSES:
            continue
        image_subdirs = [
            c for c in subdirs
            if any(p.suffix.lower() in IMAGE_EXTS for p in c.iterdir() if p.is_file())
        ]
        if len(image_subdirs) >= MIN_CLASSES:
            candidates[d] = sorted(c.name for c in image_subdirs)
    if not candidates:
        return None, []
    best = max(candidates, key=lambda d: (len(candidates[d]), -len(d.parts)))
    return best, candidates[best]


def detect_task_type(root: Path) -> dict:
    """
    Inspect a fetched dataset and identify its task type and shape.

    Returns either:
      {"task_type": "audio_anomaly", "root": ..., "normal_dirs": [...],
       "abnormal_dirs": [...], "n_normal": N, "n_abnormal": M}
      {"task_type": "image_classification", "root": ..., "classes": [...],
       "class_counts": {...}}
      {"error": "..."}
    """
    all_files = [p for p in root.rglob("*") if p.is_file()]
    if not all_files:
        return {"error": f"no files found under {root}"}

    ext_counts = Counter(p.suffix.lower() for p in all_files)
    n_total = len(all_files)
    n_audio = ext_counts.get(".wav", 0)
    n_image = sum(ext_counts.get(e, 0) for e in IMAGE_EXTS)

    if n_audio > 0 and n_audio >= n_total * DOMINANT_FRACTION:
        normal_dirs, abnormal_dirs = _find_binary_audio_dirs(root)
        if normal_dirs and abnormal_dirs:
            n_normal = sum(1 for d in normal_dirs for p in d.rglob("*.wav") if p.is_file())
            n_abnormal = sum(1 for d in abnormal_dirs for p in d.rglob("*.wav") if p.is_file())
            return {
                "task_type": "audio_anomaly",
                "root": str(root),
                "normal_dirs": [str(d) for d in normal_dirs],
                "abnormal_dirs": [str(d) for d in abnormal_dirs],
                "n_normal": n_normal,
                "n_abnormal": n_abnormal,
            }
        return {
            "error": "found .wav files but no normal/abnormal-style folder split "
                     "(expected subdirectories named e.g. 'normal' and 'abnormal')"
        }

    if n_image > 0 and n_image >= n_total * DOMINANT_FRACTION:
        class_root, classes = _find_class_folders(root)
        if class_root is not None:
            counts = {
                c: len([p for p in (class_root / c).iterdir() if p.suffix.lower() in IMAGE_EXTS])
                for c in classes
            }
            usable = sorted(c for c, n in counts.items() if n >= MIN_IMAGES_PER_CLASS)
            if len(usable) >= MIN_CLASSES:
                return {
                    "task_type": "image_classification",
                    "root": str(class_root),
                    "classes": usable,
                    "class_counts": {c: counts[c] for c in usable},
                }
        return {
            "error": "found image files but no usable class-labeled subfolder structure "
                     f"(need >= {MIN_CLASSES} class folders with >= {MIN_IMAGES_PER_CLASS} images each)"
        }

    top_ext, top_n = ext_counts.most_common(1)[0]
    return {"error": f"unsupported dataset shape (dominant file type: {top_ext or '(none)'}, {top_n}/{n_total} files)"}
