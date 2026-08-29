"""
The one enhancement layer this build adds on top of SozoGraph.

Plain BM25 + entity-expansion retrieval (sozograph.retrieve) ranks records
against the *current utterance's vocabulary*. That's the wrong tool here: a
maize/vision lesson about splitting by source and a diesel-generator/audio
session share no vocabulary and no named entity, so BM25 will never lift the
maize lesson into the diesel-generator session's context on its own.

But the Full Spectrum recipe is a fixed, closed taxonomy -- twelve numbered
steps, known at build time -- so the fix isn't semantic search. It's a plain,
deterministic filter over the passport's own facts and observations by a
`step:NN` tag, independent of relevance-to-current-query. This module is that
filter. Nothing here calls a model.

Tagging convention (see mentor.py for where these are written):
  - Fact.key = "project_{project_key}_step_{NN:02d}_status" (and _metric_*,
    _licence_* variants). Underscore-delimited, not colon-delimited:
    sozograph.resolver normalizes every Fact.key through normalize_key(),
    which collapses any run of non a-z0-9 characters -- colons included --
    into a single underscore, so a colon convention silently stops matching
    the moment it's written. Facts get this treatment; Observations don't
    (see below), which is why the two conventions differ.
  - Observation.source = "project:{project_key}:step:{NN:02d}" for any lesson
    tied to a specific recipe step. Observation.source is never normalized
    by the resolver, so colons are safe here and kept for readability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sozograph.schema import Fact, Observation, Passport

_SOURCE_RE = re.compile(r"^project:(?P<project>[a-z0-9_]+):step:(?P<step>\d{2})$")


@dataclass(frozen=True)
class StepRecord:
    project_key: str
    step: int
    field: str
    fact: Fact


@dataclass(frozen=True)
class StepLesson:
    project_key: str
    step: int
    observation: Observation


def facts_for_step(passport: Passport, project_key: str, step: int) -> list[StepRecord]:
    """Every fact this project wrote for this recipe step."""
    prefix = f"project_{project_key}_step_{step:02d}_"
    out = []
    for f in passport.facts:
        if f.key.startswith(prefix):
            out.append(StepRecord(project_key, step, f.key[len(prefix):], f))
    return out


def lessons_for_step(
    passport: Passport,
    step: int,
    *,
    exclude_project: str | None = None,
) -> list[StepLesson]:
    """
    Every dated observation tagged to this recipe step, across ALL projects
    (unless `exclude_project` filters one out). This is what a session's
    opening turn queries to recall a lesson from a different project.
    """
    out = []
    for obs in passport.observations:
        m = _SOURCE_RE.match(obs.source)
        if not m or int(m.group("step")) != step:
            continue
        project = m.group("project")
        if exclude_project and project == exclude_project:
            continue
        out.append(StepLesson(project, step, obs))
    # Most recent first -- if a step's lesson was corrected more than once,
    # the newest is the one worth citing.
    out.sort(key=lambda sl: sl.observation.ts, reverse=True)
    return out


def cross_project_recall(
    passport: Passport,
    current_project_key: str,
    risk_steps: tuple[int, ...],
) -> dict[int, list[StepLesson]]:
    """
    For a project about to start (identified by its own card's risk_steps),
    pull every lesson recorded against those same step numbers by any OTHER
    project. This is the exact call the diesel-generator session's opening
    turn makes before the user has typed anything.
    """
    recalled: dict[int, list[StepLesson]] = {}
    for step in risk_steps:
        found = lessons_for_step(passport, step, exclude_project=current_project_key)
        if found:
            recalled[step] = found
    return recalled


def step_status(passport: Passport, project_key: str, step: int) -> str | None:
    for rec in facts_for_step(passport, project_key, step):
        if rec.field == "status":
            return str(rec.fact.value)
    return None


def project_progress(passport: Passport, project_key: str, total_steps: int = 12) -> dict[int, str]:
    """{step_number: status} for every step this project has touched."""
    progress: dict[int, str] = {}
    for step in range(1, total_steps + 1):
        status = step_status(passport, project_key, step)
        if status is not None:
            progress[step] = status
    return progress
