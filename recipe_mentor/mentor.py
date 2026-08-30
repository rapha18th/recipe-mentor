"""
The session orchestrator: a Socratic walk through the Full Spectrum recipe
for one project card, writing deterministic state into a shared SozoGraph
passport as it goes.

Two judging modes:
  - offline (default, zero dependencies): a small keyword heuristic per step
    catches the specific mistakes this demo is built around (a random split
    instead of a source split, dynamic instead of static quantization) so the
    whole flow is runnable and testable today, before any GCP/ADK setup.
  - adk (--backend adk): a real Gemini 3.5+ call through Google's Agent
    Development Kit (see llm.py -- one ADK agent + session per mentor
    session) judges the free-text answer against the step's rule and writes
    its own explanation as the lesson text. State transitions still go
    through the same deterministic merge_passport_update() path either way
    -- the write side never depends on the model being right, only the
    judgment and the wording do.

Two passport stores, independent of the judge backend:
  - local (default): passports/lab_demo.json, plain JSON, diffable, the
    format SozoGraph itself centers on.
  - firestore (--store firestore): the same passport, persisted in a
    dedicated Firestore database (see firestore_store.py). Always mirrored
    to the local file too, so the dashboard and record_results.py keep
    working unchanged regardless of which store a session used.

Usage:
    python -m recipe_mentor.mentor --project maize --show-recipe
    python -m recipe_mentor.mentor --project maize
    python -m recipe_mentor.mentor --project diesel_generator
    python -m recipe_mentor.mentor --project diesel_generator --backend adk --store firestore
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sozograph.resolver import merge_passport_update
from sozograph.schema import Fact, Observation, Passport

from .recall import cross_project_recall, project_progress
from .recipe import STEPS, RecipeStep, step as recipe_step

from .projects import ProjectCard
from .projects.diesel_generator import DIESEL_GENERATOR
from .projects.maize import MAIZE

DEFAULT_USER_KEY = "sozo_lab_demo"

PASSPORT_PATH = Path(__file__).parent / "passports" / "lab_demo.json"
PROJECTS: dict[str, ProjectCard] = {p.key: p for p in (MAIZE, DIESEL_GENERATOR)}

#: Deterministic offline judge. Keyed by step number -> (must_contain_any,
#: must_not_contain_any). Correct only if a "good" keyword is present and no
#: "bad" keyword is. This is intentionally narrow -- it exists to make the
#: demo's two scripted mistakes (step 3, step 6) reproducible without an API
#: key, not to be a general-purpose grader.
_OFFLINE_JUDGE: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {
    3: (("source", "subject", "site", "machine", "speaker"), ("random", "shuffle")),
    6: (("static", "qdq"), ("dynamic",)),
}


def _load_passport(store: str = "local") -> Passport:
    if store == "firestore":
        from .firestore_store import load as fs_load
        p = fs_load(DEFAULT_USER_KEY)
        if p is not None:
            return p
        # No document yet -- fresh passport, not an error.
    elif PASSPORT_PATH.exists():
        return Passport.load(PASSPORT_PATH)
    p = Passport.new()
    p.user_key = DEFAULT_USER_KEY
    return p


def _save_passport(p: Passport, store: str = "local") -> None:
    p.save(PASSPORT_PATH)  # always mirrored locally, regardless of store
    if store == "firestore":
        from .firestore_store import save as fs_save
        fs_save(p)


def _judge_offline(step_num: int, answer: str) -> bool:
    rule = _OFFLINE_JUDGE.get(step_num)
    if rule is None:
        # No scripted mistake defined for this step -- accept anything
        # non-empty, since offline mode's job is reproducing the two demo
        # corrections, not grading the full recipe.
        return bool(answer.strip())
    good, bad = rule
    low = answer.lower()
    if any(b in low for b in bad):
        return False
    return any(g in low for g in good)


_JUDGE_PROMPT = """You are reviewing one step of a 12-step ML production recipe, \
already proven end to end on a prior shipped project, for the project "{project_key}": \
{concept}

Step {num}: {title}
The rule this step enforces: {risk_note}

The builder's answer to "{prompt}":
"{answer}"

Judge only whether this answer satisfies the rule above. Reply in exactly this \
format, nothing else:
CORRECT or INCORRECT
<one or two sentences, in your own words, speaking directly to the builder>
"""


def _judge_adk(judge: Any, project: ProjectCard, s: RecipeStep, answer: str) -> tuple[bool, str]:
    """
    Real judgment via an ADK agent (llm.AdkJudge). Deliberately plain-text,
    not structured output -- a hackathon-weekend amount of plumbing for a
    two-line response, easy to parse, easy to debug on stage if the model
    wanders off-format. ADK's event text is already a plain string (unlike
    the LangChain integration this replaced), so no block-parsing needed.
    """
    prompt = _JUDGE_PROMPT.format(
        project_key=project.key, concept=project.concept,
        num=s.number, title=s.title, risk_note=s.risk_note,
        prompt=s.prompt, answer=answer,
    )
    text = (judge.ask(prompt) or "").strip()
    first_line, _, rest = text.partition("\n")
    correct = first_line.strip().upper().startswith("CORRECT")
    feedback = rest.strip() or text
    return correct, feedback


class MentorSession:
    def __init__(
        self,
        passport: Passport,
        project: ProjectCard,
        *,
        source: str = "mentor:cli",
        judge: Any = None,
    ):
        self.passport = passport
        self.project = project
        self.source = source
        #: An llm.AdkJudge instance, or None for the offline heuristic judge.
        self.judge = judge

    # -- opening: the cross-project recall proof --------------------------

    def opening_message(self) -> str:
        recalled = cross_project_recall(self.passport, self.project.key, self.project.risk_steps)
        lines = [f"[{self.project.key}] {self.project.concept}"]
        if not recalled:
            lines.append("\nNo prior lessons tagged to this project's risk steps yet.")
            return "\n".join(lines)

        lines.append("")
        for step_num in sorted(recalled):
            lesson = recalled[step_num][0]  # most recent
            title = recipe_step(step_num).title
            when = lesson.observation.when or lesson.observation.ts.date().isoformat()
            lines.append(
                f"Step {step_num} ({title}) -- {lesson.observation.text} "
                f"[from '{lesson.project_key}', {when}]"
            )
        lines.append("\nApplying those by default this time, unless you say otherwise.")
        return "\n".join(lines)

    # -- per-step turn ------------------------------------------------------

    def run_step(self, step_num: int, answer: str) -> str:
        """Returns the mentor's response text. Writes state deterministically
        regardless of which judge decided correctness."""
        s = recipe_step(step_num)

        if self.judge is not None:
            correct, feedback = _judge_adk(self.judge, self.project, s, answer)
        else:
            correct = _judge_offline(step_num, answer)
            feedback = s.risk_note

        status = "done" if correct else "corrected"
        self.write_status(step_num, status, detail=answer[:200])

        if correct:
            return f"Good -- step {step_num} ({s.title}) checks out."

        lesson_text = (
            f"On {self.project.key}, user proposed '{answer.strip()}' for step {step_num} "
            f"({s.title}); mentor corrected: {feedback}"
        )
        self.write_lesson(step_num, lesson_text)
        return f"Careful -- step {step_num}: {feedback}"

    def record_metric(self, name: str, value: float | int | str) -> None:
        key = f"{self.project.fact_prefix()}_metric_{name}"
        merge_passport_update(
            self.passport,
            facts=[Fact(key=key, value=value, source=self.source, confidence=0.95)],
        )

    def record_licence_check(self, dataset: str, host: str, terms: str) -> None:
        key = f"{self.project.fact_prefix()}_licence_{dataset}"
        merge_passport_update(
            self.passport,
            facts=[Fact(
                key=key,
                value={"host": host, "terms": terms, "verified_on": date.today().isoformat()},
                source=self.source,
                confidence=1.0,
            )],
        )

    # -- public writes, also used by the autonomous agent path --------------

    def write_status(self, step_num: int, status: str, *, detail: str = "") -> None:
        key = f"{self.project.fact_prefix()}_step_{step_num:02d}_status"
        merge_passport_update(
            self.passport,
            facts=[Fact(key=key, value=status, source=self.source, confidence=0.9)],
        )

    def write_lesson(self, step_num: int, text: str) -> None:
        obs = Observation(
            text=text,
            when=date.today().isoformat(),
            source=f"project:{self.project.key}:step:{step_num:02d}",
            participants=["user", "recipe_mentor"],
        )
        merge_passport_update(self.passport, observations=[obs])

    def write_step_note(self, step_num: int, status: str, text: str) -> None:
        """Used by the autonomous agent path (agent_runner.py): every tool
        call's real outcome becomes both a status Fact and a dated
        Observation, unlike the Socratic path (which only writes a lesson
        on a correction) -- an agent run's "lesson" is closer to "what
        actually happened," and future cross-project recall benefits from
        having real content to cite, not just corrections."""
        self.write_status(step_num, status, detail=text[:200])
        self.write_lesson(step_num, text)

    # -- back-compat aliases -- write_status/write_lesson are the public
    # surface; kept under the old private names too in case anything else
    # in a fork of this repo still calls them directly.
    _write_status = write_status
    _write_lesson = write_lesson


def _print_progress(passport: Passport, project: ProjectCard) -> None:
    progress = project_progress(passport, project.key)
    print(f"\n[{project.key}] progress: " + ", ".join(
        f"{n}:{progress.get(n, '-')}" for n in range(1, 13)
    ))


def show_recipe(project: ProjectCard) -> None:
    print(f"=== {project.key} ===")
    print(project.concept)
    print(f"\nSensors: {project.sensors}")
    print(f"Baseline dataset: {project.baseline_dataset}")
    print(f"Local gap: {project.local_gap}")
    print(f"Path to production: {project.path_to_production}")
    print(f"First 90 days: {project.first_90_days}")
    print(f"\nRisk steps: {project.risk_steps}")
    print("\n12-step recipe:")
    for s in STEPS:
        flag = " <-- risk step for this project" if s.number in project.risk_steps else ""
        print(f"  {s.number:2d}. {s.title}{flag}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Recipe Mentor session")
    ap.add_argument("--project", required=True, choices=sorted(PROJECTS))
    ap.add_argument("--show-recipe", action="store_true")
    ap.add_argument("--backend", choices=("offline", "adk"), default="offline")
    ap.add_argument("--store", choices=("local", "firestore"), default="local")
    args = ap.parse_args(argv)

    project = PROJECTS[args.project]
    if args.show_recipe:
        show_recipe(project)
        return

    judge = None
    if args.backend == "adk":
        from .llm import build_judge
        judge = build_judge()
        print(f"[backend: adk, model: {judge.model}]")
    if args.store == "firestore":
        print(f"[store: firestore, project: neofix-676da, database: recipe-mentor]")
    print()

    passport = _load_passport(store=args.store)
    session = MentorSession(passport, project, judge=judge)

    print(session.opening_message())
    print()

    for s in STEPS:
        try:
            answer = input(f"Step {s.number}. {s.prompt}\n> ")
        except EOFError:
            break
        if not answer.strip():
            continue
        print(session.run_step(s.number, answer))
        print()

    _print_progress(passport, project)
    _save_passport(passport, store=args.store)
    print(f"\nSaved -> {PASSPORT_PATH}" + (" (+ Firestore)" if args.store == "firestore" else ""))


if __name__ == "__main__":
    main()
