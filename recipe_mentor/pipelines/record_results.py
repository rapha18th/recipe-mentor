"""
Read each pipeline's report.json and write its real numbers into the shared
passport as Facts, via the same MentorSession.record_metric/
record_licence_check path a live session would use. Keeps the pipeline
scripts themselves free of any SozoGraph/passport concern -- they just
write a plain report.json, this is the only place that turns that into
passport state.

Usage:
    python -m recipe_mentor.pipelines.record_results
"""
from __future__ import annotations

import json
from pathlib import Path

from ..mentor import MentorSession, PASSPORT_PATH, _load_passport, _save_passport
from ..projects.diesel_generator import DIESEL_GENERATOR
from ..projects.maize import MAIZE

MAIZE_REPORT = Path(__file__).parent.parent / "data" / "maize_run" / "report.json"
MIMII_REPORT = Path(__file__).parent.parent / "data" / "mimii_run" / "report.json"


def record_maize(session: MentorSession) -> None:
    if not MAIZE_REPORT.exists():
        print(f"skip maize: no report at {MAIZE_REPORT}")
        return
    r = json.loads(MAIZE_REPORT.read_text(encoding="utf-8"))
    session.record_metric("fp32_accuracy", r["fp32_accuracy"])
    session.record_metric("int8_accuracy", r["int8_accuracy"])
    session.record_metric("verdict", r["verdict"])
    session.record_metric("fp32_size_kb", round(r["fp32_onnx_bytes"] / 1024, 1))
    session.record_metric("int8_size_kb", round(r["int8_onnx_bytes"] / 1024, 1))
    session.record_metric("compression_ratio", round(r["fp32_onnx_bytes"] / max(r["int8_onnx_bytes"], 1), 2))
    session.record_metric("n_train", r["n_train"])
    session.record_metric("n_test", r["n_test"])
    session.record_licence_check(
        "corn_or_maize_leaf_disease_dataset", "Kaggle",
        "copyright-authors (see dataset page for terms)",
    )
    print(f"Recorded maize metrics: fp32_acc={r['fp32_accuracy']:.4f} "
          f"int8_acc={r['int8_accuracy']:.4f} verdict={r['verdict']}")


def record_diesel_generator(session: MentorSession) -> None:
    if not MIMII_REPORT.exists():
        print(f"skip diesel_generator: no report at {MIMII_REPORT}")
        return
    r = json.loads(MIMII_REPORT.read_text(encoding="utf-8"))
    session.record_metric("fp32_auc", r["fp32_auc"])
    session.record_metric("int8_auc", r["int8_auc"])
    session.record_metric("verdict", r["verdict"])
    session.record_metric("fp32_size_kb", round(r["fp32_onnx_bytes"] / 1024, 1))
    session.record_metric("int8_size_kb", round(r["int8_onnx_bytes"] / 1024, 1))
    session.record_metric("compression_ratio", round(r["fp32_onnx_bytes"] / max(r["int8_onnx_bytes"], 1), 2))
    session.record_metric("confidence_band_clean_below", round(r["clean_below"], 4))
    session.record_metric("confidence_band_uncertain_below", round(r["uncertain_below"], 4))
    session.record_metric("n_train", r["n_train"])
    session.record_metric("n_test_normal", r["n_test_normal"])
    session.record_metric("n_test_abnormal", r["n_test_abnormal"])
    session.record_licence_check(
        "mimii", "Zenodo (record 3384388)",
        "CC BY-SA 4.0 (proxy dataset -- see the diesel_generator project card's local-gap note)",
    )
    print(f"Recorded diesel_generator metrics: fp32_auc={r['fp32_auc']:.4f} "
          f"int8_auc={r['int8_auc']:.4f} verdict={r['verdict']}")


def main() -> None:
    passport = _load_passport()
    record_maize(MentorSession(passport, MAIZE, source="pipeline:maize_train"))
    record_diesel_generator(MentorSession(passport, DIESEL_GENERATOR, source="pipeline:mimii_anomaly"))
    _save_passport(passport)
    print(f"\nSaved -> {PASSPORT_PATH}")


if __name__ == "__main__":
    main()
