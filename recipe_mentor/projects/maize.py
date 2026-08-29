"""
Maize leaf disease diagnosis — vision, session 1 (lighter build).

Swapped in for Full Spectrum's cassava card: Zimbabwe grows very little
cassava, and maize dominates local smallholder production instead. The paper
names the honest local gap itself -- PlantVillage's maize classes are
lab-photographed, not field-photographed the way the Cassava Leaf Disease set
is -- which becomes a real talking point in the demo rather than a weakness
to paper over.
"""
from __future__ import annotations

from . import ProjectCard

MAIZE = ProjectCard(
    key="maize",
    concept=(
        "An offline phone app that identifies common maize leaf disease and "
        "pest damage from a photograph, and returns a treatment "
        "recommendation."
    ),
    sensors="Rear camera only.",
    baseline_dataset=(
        "PlantVillage's maize classes (widely mirrored on Kaggle and Hugging "
        "Face) for broad disease coverage."
    ),
    why_good_baseline=(
        "PlantVillage is large, clean, and well-labelled -- enough to "
        "validate the whole feature/train/quantize pipeline before any local "
        "photography exists."
    ),
    local_gap=(
        "PlantVillage's maize images are lab-photographed against a plain "
        "background, not field-photographed under real leaf backgrounds and "
        "lighting. A locally photographed maize and small-grains corpus, tied "
        "to an agronomist's own treatment call, is the real local work -- "
        "stated honestly, per the paper's own discipline, not hidden behind a "
        "clean validation number."
    ),
    path_to_production=(
        "Recipe steps 1-4 carry the real risk: dataset assembly, a portable "
        "feature pipeline, a source-stratified split, and sizing a model that "
        "stays useful on a low-end phone under real-world (not studio) "
        "conditions."
    ),
    first_90_days=(
        "Train a baseline on PlantVillage, quantize it with the static QDQ "
        "recipe, and field-test it against real Zimbabwean maize photographs "
        "before writing a single treatment recommendation."
    ),
    risk_steps=(1, 2, 3, 4),
    reuses_pipeline_from=(),
)
