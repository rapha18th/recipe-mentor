"""
The Full Spectrum production recipe, as data.

Sozo Analytics Lab's internal working paper ("Full Spectrum," 19 Aug 2026)
restates SiloSense's own shipped pipeline as a reusable twelve-step recipe.
Every project card in the paper names which of these twelve steps carries the
real risk for that specific build. This module is the single source of truth
for step numbers, titles, and the risk note the mentor cites when it corrects
a user, so the recipe text lives in exactly one place rather than being
re-typed into every prompt.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecipeStep:
    number: int
    title: str
    #: What SiloSense itself measured, restated as the rule the mentor enforces.
    risk_note: str
    #: The Socratic opening question the mentor asks before revealing the rule.
    prompt: str


STEPS: tuple[RecipeStep, ...] = (
    RecipeStep(
        1, "Find or assemble a baseline dataset",
        "A wrong join silently returns unlabelled files. Getting the join right "
        "is the first milestone, not an afterthought.",
        "Where's the baseline data coming from, and how do you know the labels "
        "actually line up with the files?",
    ),
    RecipeStep(
        2, "Ship the feature pipeline in pure, portable math",
        "Compute features in plain NumPy, not a library with no reliable Arm "
        "build (SiloSense avoided librosa for exactly this reason).",
        "What are you computing features with, and will it compile on-device?",
    ),
    RecipeStep(
        3, "Split by source, never by window",
        "Train/validation must split at the file or subject level, stratified "
        "by class and by source. Splitting inside one recording or one image "
        "crop inflates validation accuracy without meaning anything.",
        "How are you splitting train and validation?",
    ),
    RecipeStep(
        4, "Size the model to show a genuine win",
        "Too small, and fixed kernel-launch overhead makes quantization look "
        "slower than it is. Size up until there's enough real compute for INT8 "
        "to show a measurable difference.",
        "How big is the model, and why that size?",
    ),
    RecipeStep(
        5, "Train, then export to ONNX with a dynamic batch axis",
        "Keep the best-validation checkpoint, not the last epoch. Class-weight "
        "the loss if the raw data is imbalanced.",
        "Which checkpoint are you exporting, and is the data balanced?",
    ),
    RecipeStep(
        6, "Quantize statically, not dynamically",
        "Dynamic quantization only touches MatMul/Gemm by default and leaves a "
        "convolutional model almost entirely in FP32. Static QDQ INT8, "
        "calibrated on real examples, is what actually quantizes the "
        "convolution layers.",
        "Static or dynamic quantization, and why?",
    ),
    RecipeStep(
        7, "Verify accuracy held, on the same held-out split",
        "Compare FP32 against INT8 on identical data. A validation set too "
        "small to tell noise from signal gets reported as 'no loss,' not as "
        "an improvement.",
        "How are you confirming INT8 didn't cost you accuracy?",
    ),
    RecipeStep(
        8, "Port the feature pipeline to the device language, and diff-test it",
        "SiloSense's Kotlin port matched its Python reference to 4.2e-7, "
        "verified with an automated test, not assumed.",
        "How will you verify the on-device port matches the training pipeline "
        "numerically?",
    ),
    RecipeStep(
        9, "Configure the runtime to try the fastest path, and profile which "
        "one actually ran",
        "Registering successfully is not the same as executing a single node. "
        "Profile, don't trust the registration log.",
        "Which execution provider are you targeting, and how will you confirm "
        "it actually ran?",
    ),
    RecipeStep(
        10, "Benchmark like a matched pair, on the real device",
        "Both models loaded and timed in the same run, warm-up runs "
        "discarded, reported as one phone in one session, not a universal "
        "constant.",
        "How are you structuring the FP32-vs-INT8 benchmark?",
    ),
    RecipeStep(
        11, "Set confidence bands from the validation distribution",
        "Not an assumed cutoff. Read the bands from where the validation data "
        "actually stops separating cleanly.",
        "Where do your confidence thresholds come from?",
    ),
    RecipeStep(
        12, "Measure what the deployment context actually needs, and report "
        "what is not yet done",
        "Battery draw, cold-start time, memory, thermal state. Then a plain "
        "list of what's verified against what isn't, so the next reader "
        "inherits an honest starting point.",
        "What deployment-side numbers do you still need to measure?",
    ),
)


def step(number: int) -> RecipeStep:
    for s in STEPS:
        if s.number == number:
            return s
    raise KeyError(f"no recipe step {number}")
