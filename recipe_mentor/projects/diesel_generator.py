"""
Diesel and mini-grid generator health monitor -- audio, session 2, the
flagship, priority build.

Full Spectrum names this project's pipeline as "the acoustic mill monitor's
pipeline, unchanged, retargeted a second time" -- an anomaly-only model
trained on healthy machine sound, flagging deviation, with no fault examples
needed to start. Deeply Zimbabwe-relevant given chronic load-shedding and
diesel-backup dependence, and genuinely new work for this build (the
small-mill card itself is deliberately not rebuilt here).
"""
from __future__ import annotations

from . import ProjectCard

DIESEL_GENERATOR = ProjectCard(
    key="diesel_generator",
    concept=(
        "A clip-on contact microphone on a backup generator -- at a telecom "
        "tower, mini-grid site, or factory -- flags injector wear, fuel "
        "dilution, and developing mechanical faults from its running sound, "
        "and cross-checks the operator's own logged runtime hours against the "
        "engine's actual acoustic duty cycle."
    ),
    sensors="Phone or clip-on contact microphone.",
    baseline_dataset=(
        "No clean public dataset exists for diesel-generator fault or "
        "fuel-adulteration sound specifically. The nearest usable proxy is "
        "MIMII (Hitachi's industrial machine-sound corpus, Zenodo -- valves, "
        "pumps, fans, slide rails, recorded under normal and induced-fault "
        "conditions; also the reference dataset behind the DCASE "
        "anomalous-sound-detection challenge)."
    ),
    why_good_baseline=(
        "It validates the same anomaly-only training approach and the same "
        "contact-microphone technique already proven on a different rotating "
        "machine, on a genuinely different one."
    ),
    local_gap=(
        "The entire fault and fuel-integrity corpus. Telecom-tower operators "
        "and mini-grid developers already track generator runtime hours and "
        "fuel deliveries as a matter of financial control -- attach the "
        "sensor to a process that already measures the answer."
    ),
    path_to_production=(
        "This is the small-mill monitor's pipeline, unchanged, retargeted a "
        "second time -- steps 3, 5-7, and 11 (a source-stratified split "
        "across recording sessions, anomaly-only training, static QDQ "
        "quantization, FP32-vs-INT8 verification, confidence bands from the "
        "real validation distribution) all carry over directly."
    ),
    first_90_days=(
        "Partner with one telecom-tower operator or mini-grid developer "
        "already tracking generator runtime and fuel spend, and instrument "
        "two or three units."
    ),
    risk_steps=(1, 3, 6, 7, 11),
    reuses_pipeline_from=("small_mill_mimii",),
)
