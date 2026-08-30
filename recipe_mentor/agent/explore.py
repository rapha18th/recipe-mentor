"""
Real empirical exploration of the fetched dataset -- the step between
"what shape is this" (detect.py, pure filesystem logic) and "how should
this be configured" (configure_run). Where detect.py answers "is this
audio_anomaly or image_classification," this answers "what does *this*
dataset actually look like, and what should a builder know before
training on it" -- class balance, sample-rate consistency, corrupt files,
image dimension spread. Cheap, bounded sampling (a handful of files, not
the whole corpus) so it stays fast enough for a live run.

This is deliberately NOT where the decision gets made. agent/toolkit.py's
ask_user() is where a real choice, grounded in these findings plus
whatever cross_project_recall surfaced at kickoff, gets put to the person
running this -- explore_dataset's job is only to make that choice
grounded in something real about this specific dataset, not a generic
prompt.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, UnidentifiedImageError

_IMAGE_SAMPLE_PER_CLASS = 6
_AUDIO_SAMPLE_PER_GROUP = 6


def explore_image_classification(root: Path, classes: list[str], class_counts: dict[str, int]) -> dict:
    rng = random.Random(0)
    dims: list[tuple[int, int]] = []
    modes: set[str] = set()
    corrupt = 0
    n_checked = 0

    for cls in classes:
        files = sorted((root / cls).iterdir())
        sample = rng.sample(files, min(_IMAGE_SAMPLE_PER_CLASS, len(files)))
        for f in sample:
            n_checked += 1
            try:
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    dims.append(img.size)
                    modes.add(img.mode)
            except (UnidentifiedImageError, OSError):
                corrupt += 1

    counts = [class_counts[c] for c in classes]
    imbalance_ratio = round(max(counts) / max(min(counts), 1), 2)
    widths = [d[0] for d in dims]
    heights = [d[1] for d in dims]

    return {
        "n_sampled": n_checked,
        "corrupt_in_sample": corrupt,
        "class_imbalance_ratio": imbalance_ratio,
        "class_counts": class_counts,
        "image_modes_seen": sorted(modes),
        "width_range": [min(widths), max(widths)] if widths else None,
        "height_range": [min(heights), max(heights)] if heights else None,
        "uniform_dimensions": len(set(dims)) <= 1 if dims else None,
    }


def explore_audio_anomaly(normal_dirs: list[Path], abnormal_dirs: list[Path]) -> dict:
    rng = random.Random(0)

    def sample_files(dirs: list[Path]) -> list[Path]:
        all_files = sorted({p for d in dirs for p in d.rglob("*.wav")})
        return rng.sample(all_files, min(_AUDIO_SAMPLE_PER_GROUP, len(all_files)))

    normal_sample = sample_files(normal_dirs)
    abnormal_sample = sample_files(abnormal_dirs)
    all_sample = normal_sample + abnormal_sample

    sample_rates: set[int] = set()
    channel_counts: set[int] = set()
    durations: list[float] = []
    peak_amplitudes: list[float] = []
    unreadable = 0

    for f in all_sample:
        try:
            info = sf.info(str(f))
            sample_rates.add(info.samplerate)
            channel_counts.add(info.channels)
            durations.append(info.frames / info.samplerate)
        except Exception:
            unreadable += 1
            continue

    # Peak-amplitude / clipping check on a smaller sub-sample -- this one
    # actually decodes the audio, so keep it tighter than the metadata pass.
    for f in all_sample[:4]:
        try:
            data, _ = sf.read(str(f), dtype="float32")
            peak_amplitudes.append(float(np.abs(data).max()))
        except Exception:
            continue

    n_normal = sum(1 for d in normal_dirs for _ in d.rglob("*.wav"))
    n_abnormal = sum(1 for d in abnormal_dirs for _ in d.rglob("*.wav"))
    imbalance_ratio = round(max(n_normal, n_abnormal) / max(min(n_normal, n_abnormal), 1), 2)

    return {
        "n_sampled": len(all_sample),
        "unreadable_in_sample": unreadable,
        "class_imbalance_ratio": imbalance_ratio,
        "n_normal": n_normal,
        "n_abnormal": n_abnormal,
        "sample_rates_seen": sorted(sample_rates),
        "resample_needed": len(sample_rates) > 1 or (sample_rates and 16000 not in sample_rates),
        "channel_counts_seen": sorted(channel_counts),
        "duration_range_seconds": [round(min(durations), 2), round(max(durations), 2)] if durations else None,
        "possible_clipping": any(p >= 0.999 for p in peak_amplitudes) if peak_amplitudes else None,
    }
