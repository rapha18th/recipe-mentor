"""
Deterministic project-identity authoring for the autonomous agent path.

A new agent-run project doesn't get a hand-written ProjectCard the way
maize/diesel_generator do -- but it still needs one, since recall.py, the
dashboard, and every passport-write convention in mentor.py are all keyed
off ProjectCard.fact_prefix()/risk_steps. This module builds one
deterministically from the problem statement, dataset ref, and detected
task type: templated prose, not a second LLM call. These fields don't
affect pipeline correctness, so spending a model call authoring them would
just be one more thing that can go wrong on demo day for no functional
benefit -- the agent's real, model-driven decisions are in
configure_run()'s hyperparameters and its own closing narration, not here.
"""
from __future__ import annotations

import hashlib
import re

from . import ProjectCard

#: Which recipe steps carry the real risk for each task type -- mirrors how
#: MAIZE/DIESEL_GENERATOR's own risk_steps were hand-picked, generalized to
#: apply to any project of that type rather than one specific dataset.
#: Step 12 is included for both so a run's closing honesty note (what's not
#: yet done) and any captured user feedback resurface for the next
#: same-task-type project, not just the recipe's discipline steps.
TASK_TYPE_RISK_STEPS: dict[str, tuple[int, ...]] = {
    "image_classification": (1, 2, 3, 4, 6, 12),
    "audio_anomaly": (1, 3, 6, 7, 11, 12),
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_project_key(problem_statement: str, dataset_ref: str) -> str:
    """
    A safe, underscore-only project key, computed up front rather than left
    to sozograph.utils.normalize_key's own silent collapse -- the exact
    colon-vs-underscore bug recall.py's docstring already documents,
    avoided here by never producing a key that would collapse into
    something else. Deterministic in the dataset ref alone (via a short
    hash suffix) so two runs against the same problem/dataset pair land on
    the same project key rather than fragmenting history.
    """
    base = _SLUG_RE.sub("_", problem_statement.lower()).strip("_")[:40]
    suffix = hashlib.sha1(dataset_ref.encode("utf-8")).hexdigest()[:6]
    return f"{base}_{suffix}" if base else f"project_{suffix}"


def author_project_card(
    *, problem_statement: str, dataset_ref: str, task_type: str, dataset_summary: str,
) -> ProjectCard:
    key = slugify_project_key(problem_statement, dataset_ref)
    risk_steps = TASK_TYPE_RISK_STEPS.get(task_type, (1, 2, 3, 4, 6, 7, 12))
    readable_type = task_type.replace("_", " ")
    return ProjectCard(
        key=key,
        concept=problem_statement.strip(),
        sensors="Unspecified -- an agent-run project, not hand-authored.",
        baseline_dataset=f"Kaggle: {dataset_ref}. {dataset_summary}",
        why_good_baseline=(
            f"Named directly by the user as the dataset to build this project on; the "
            f"agent verified it fits the {readable_type} pipeline contract before training."
        ),
        local_gap=(
            "Not yet assessed -- this run trained and verified a pipeline against a public "
            "Kaggle dataset only. Whether it holds on real local data is still open."
        ),
        path_to_production=(
            f"Recipe steps {', '.join(str(s) for s in risk_steps)} carry the real risk for a "
            f"{readable_type} project; see the step grid for what this run actually verified."
        ),
        first_90_days="Not yet planned -- this run establishes a verified baseline pipeline only.",
        risk_steps=risk_steps,
        reuses_pipeline_from=(
            ("generic_image_train",) if task_type == "image_classification"
            else ("generic_audio_anomaly_train",) if task_type == "audio_anomaly"
            else ()
        ),
    )
