from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProjectCard:
    """
    One Full Spectrum project card, verbatim from the paper's seven fields.

    `key` is the namespace prefix used for every passport record this project
    writes (`project:{key}:step:NN:status`, `project:{key}:metric:*`, ...).
    """

    key: str
    concept: str
    sensors: str
    baseline_dataset: str
    why_good_baseline: str
    local_gap: str
    path_to_production: str
    first_90_days: str
    #: Recipe step numbers this project's own card flags as carrying real risk.
    risk_steps: tuple[int, ...] = field(default_factory=tuple)
    #: Other project keys sharing this pipeline, per the paper's own text.
    reuses_pipeline_from: tuple[str, ...] = field(default_factory=tuple)

    def fact_prefix(self) -> str:
        # sozograph.resolver normalizes every Fact.key through normalize_key(),
        # which collapses any run of non a-z0-9 characters (colons included)
        # into a single underscore -- so the convention has to be
        # underscore-delimited from the start, or the written key silently
        # stops matching the one we think we wrote.
        return f"project_{self.key}"
