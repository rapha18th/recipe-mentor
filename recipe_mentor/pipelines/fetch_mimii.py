"""
Pull a bounded, source-split MIMII subset from Zenodo without downloading
the full 6.9GB zip.

MIMII's own record (Zenodo 3384388) publishes one zip per machine
type/SNR -- even the smallest single file (6_dB_valve.zip) is 6.9GB,
containing all four physical valve units (id_00, id_02, id_04, id_06).
Zenodo honors HTTP Range requests (confirmed live: a `Range: bytes=0-1023`
request against the file returns 206, not 200), so `remotezip` fetches only
the central directory plus the specific entries this script asks for,
without touching the other ~10,000 files in the archive.

Recipe step 3 ("split by source, never by window") is honored at the
strongest available grain: id_00 (a physically distinct valve unit) is used
for training and a held-out-normal test slice; id_02 (a different physical
unit entirely) is used for validation and a cross-unit anomaly test slice.
This is a stronger split than holding out files *within* one unit -- it
answers "does this generalize to a valve the model has never heard" rather
than just "does this generalize to a different ten seconds of the same
valve."

Usage:
    python -m recipe_mentor.pipelines.fetch_mimii
    python -m recipe_mentor.pipelines.fetch_mimii --dry-run
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

MIMII_ZIP_URL = "https://zenodo.org/records/3384388/files/6_dB_valve.zip"
DATA_ROOT = Path(__file__).parent.parent / "data" / "mimii" / "valve"

#: Bounded subset sizes, chosen to keep the download in the hundreds-of-MB
#: range rather than the multi-GB source file, while staying large enough
#: for a real (if small) anomaly-only training run.
N_TRAIN = 100          # id_00 normal -> train
N_HELDOUT_NORMAL = 30  # id_00 normal, disjoint from train -> test (normal)
N_TEST_ABNORMAL_ID00 = 40   # id_00 abnormal -> test
N_VAL_NORMAL = 30      # id_02 normal -> validation (different physical unit)
N_TEST_ABNORMAL_ID02 = 40   # id_02 abnormal -> test, cross-unit


@dataclass(frozen=True)
class FetchPlan:
    zip_path: str      # path inside the remote zip
    local_path: Path    # where to write it
    split: str           # train / val / test_normal / test_abnormal
    source_id: str        # id_00 / id_02 -- the "source" for step-3 purposes


def _build_plan() -> list[FetchPlan]:
    plan: list[FetchPlan] = []

    def add(id_: str, kind: str, indices: range, split: str) -> None:
        for i in indices:
            name = f"{i:08d}.wav"
            zip_path = f"valve/{id_}/{kind}/{name}"
            local = DATA_ROOT / split / id_ / name
            plan.append(FetchPlan(zip_path, local, split, id_))

    add("id_00", "normal", range(0, N_TRAIN), "train")
    add("id_00", "normal", range(N_TRAIN, N_TRAIN + N_HELDOUT_NORMAL), "test_normal")
    add("id_00", "abnormal", range(0, N_TEST_ABNORMAL_ID00), "test_abnormal")
    add("id_02", "normal", range(0, N_VAL_NORMAL), "val")
    add("id_02", "abnormal", range(0, N_TEST_ABNORMAL_ID02), "test_abnormal")

    return plan


def fetch(*, dry_run: bool = False) -> dict:
    plan = _build_plan()
    manifest = {
        "source": MIMII_ZIP_URL,
        "source_dataset": "MIMII (Purohit et al. 2019), Zenodo record 3384388",
        "machine_type": "valve",
        "snr_db": 6,
        "split_by_source_note": (
            "train/test_normal/test_abnormal(id_00) drawn from physical unit "
            "id_00; val/test_abnormal(id_02) drawn from physical unit id_02 -- "
            "a different machine, not just different files, per recipe step 3."
        ),
        "counts": {},
        "files": [],
    }
    counts: dict[str, int] = {}
    for item in plan:
        counts[item.split] = counts.get(item.split, 0) + 1
        manifest["files"].append({
            "zip_path": item.zip_path,
            "local_path": str(item.local_path.relative_to(DATA_ROOT.parent.parent)),
            "split": item.split,
            "source_id": item.source_id,
        })
    manifest["counts"] = counts

    if dry_run:
        print(json.dumps({"counts": counts, "total": len(plan)}, indent=2))
        return manifest

    from remotezip import RemoteZip

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    done = 0
    with RemoteZip(MIMII_ZIP_URL) as z:
        for item in plan:
            if item.local_path.exists():
                done += 1
                continue
            item.local_path.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(item.zip_path)
            item.local_path.write_bytes(data)
            done += 1
            if done % 25 == 0 or done == len(plan):
                print(f"[{done}/{len(plan)}] {item.split}/{item.source_id}")

    manifest_path = DATA_ROOT.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest -> {manifest_path}")
    print(f"Total files: {len(plan)} ({sum(counts.values())})")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, fetch nothing.")
    args = ap.parse_args(argv)
    fetch(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
