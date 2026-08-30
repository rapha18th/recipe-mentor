"""
CLI entry point for the autonomous agent path: given a problem statement and
a named Kaggle dataset, a real Google ADK agent with tool-calling decides
how to fetch, detect, configure, train, and record a full ML pipeline run.
Each tool call is logged live and written into the shared SozoGraph
passport as it happens, via the same deterministic merge_passport_update()
path the Socratic mentor.py session uses -- the write side never depends on
the model being right, only on what its tools actually did.

Two real human check-ins bracket the autonomous core, per the Collaborative
Partner track's own requirement to ask clarifying questions, guide the user
step by step, and capture feedback that adapts future runs: one question
before the run starts (a speed-vs-accuracy priority, folded into the
agent's kickoff prompt), and one feedback prompt after it finishes (written
via the same step-12 mechanism as the run's own honesty note, so it
resurfaces for the next same-task-type project through cross_project_recall).
These are asked by this module, not phrased live by the LLM mid-tool-loop --
a deliberate reliability choice, consistent with keeping the model's
freedom bounded everywhere else in this design (see agent/toolkit.py).

Usage:
    python -m recipe_mentor.agent_runner \\
        --problem "Detect diesel generator bearing faults from acoustic recordings" \\
        --dataset "owner/slug"
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import time

from sozograph.resolver import merge_passport_update
from sozograph.schema import Entity, Passport

from .agent.toolkit import AgentToolkit
from .dashboard.render_dashboard import render as render_dashboard
from .llm import build_agent_runner
from .mentor import (
    DEFAULT_USER_KEY, PASSPORT_PATH, MentorSession, _load_passport, _print_progress, _save_passport,
)
from .projects import ProjectCard
from .projects.dynamic import TASK_TYPE_RISK_STEPS, author_project_card, slugify_project_key
from .recall import cross_project_recall

MAX_EVENTS = 60
MAX_SECONDS = 300

_AGENT_INSTRUCTION = (
    "You are an autonomous ML engineer following a proven twelve-step production "
    "recipe. Given a problem statement and a Kaggle dataset reference, call your "
    "tools in this order: fetch_kaggle_dataset, detect_task_type, configure_run, "
    "train_and_verify, record_results. Each tool enforces the recipe's discipline "
    "itself -- you cannot skip quantization or verification, only choose "
    "hyperparameters within the range the tool accepts. If a tool returns an "
    "error, read it and correct your next call; do not repeat the same call "
    "unchanged. If the user stated a priority for speed vs. accuracy, let that "
    "guide your epoch count choice within the allowed range. After "
    "record_results succeeds, write one short paragraph summarizing what you "
    "found, in your own words, speaking directly to the person who asked for "
    "this project -- then stop."
)


def _build_kickoff_prompt(problem: str, dataset_ref: str, passport: Passport, priority: str) -> str:
    lines = [f"Problem: {problem}", f"Kaggle dataset: {dataset_ref}"]
    if priority:
        lines.append(f"User priority for this run: {priority}")

    recall_lines: list[str] = []
    for task_type, risk_steps in TASK_TYPE_RISK_STEPS.items():
        recalled = cross_project_recall(passport, "_pending_", risk_steps, task_type=task_type)
        if not recalled:
            continue
        recall_lines.append(f"\n[if this turns out to be a {task_type} project]")
        for step_num in sorted(recalled):
            lesson = recalled[step_num][0]
            when = lesson.observation.when or lesson.observation.ts.date().isoformat()
            recall_lines.append(f"  step {step_num}: {lesson.observation.text} [from '{lesson.project_key}', {when}]")
    if recall_lines:
        lines.append(
            "\nBefore you start: here is what past projects in this lab learned, by "
            "task type. Apply whatever turns out to be relevant once you know which "
            "type this dataset is." + "\n".join(recall_lines)
        )

    lines.append("\nNow begin: call fetch_kaggle_dataset, then proceed through the tools in order.")
    return "\n".join(lines)


def _print_tool_call(name: str, args: dict) -> None:
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"\n[tool call] {name}({shown})")


def _print_tool_result(name: str, result: dict) -> None:
    shown = {k: v for k, v in result.items() if k != "step_results"}
    print(f"[tool result] {name} -> {shown}")


def _apply_step_results(session: MentorSession, result: dict) -> None:
    for step_num, (status, detail) in result.get("step_results", {}).items():
        session.write_step_note(step_num, status, detail)


def _summarize_dataset(info: dict) -> str:
    if info.get("task_type") == "image_classification":
        return f"{len(info.get('classes', []))} classes: {', '.join(info.get('classes', []))}."
    return f"{info.get('n_normal', '?')} normal / {info.get('n_abnormal', '?')} abnormal clips."


def _register_project(passport: Passport, card: ProjectCard, task_type: str, dataset_ref: str) -> None:
    passport.meta.setdefault("projects", {})[card.key] = {
        **dataclasses.asdict(card), "task_type": task_type, "dataset_ref": dataset_ref,
    }
    merge_passport_update(passport, entities=[
        Entity(name=card.key, type="project", aliases=[dataset_ref, task_type]),
    ])


def _record_metrics_and_licence(session: MentorSession, toolkit: AgentToolkit) -> None:
    if toolkit.state.get("_metrics_recorded"):
        return
    for name, value in toolkit.state.get("report_dict", {}).items():
        if isinstance(value, (int, float, str, bool)):
            session.record_metric(name, value)
    session.record_licence_check(
        toolkit.state["dataset_ref"].replace("/", "_"), "Kaggle",
        "see dataset page for terms (fetched by an autonomous agent run)",
    )
    toolkit.state["_metrics_recorded"] = True


async def _drive(
    toolkit: AgentToolkit, session: MentorSession, passport: Passport,
    problem: str, dataset_ref: str, prompt: str,
) -> str:
    tools = [
        toolkit.fetch_kaggle_dataset, toolkit.detect_task_type,
        toolkit.configure_run, toolkit.train_and_verify, toolkit.record_results,
    ]
    runner = build_agent_runner(tools, _AGENT_INSTRUCTION)

    final_text = ""
    n_events = 0
    start = time.monotonic()
    async for event in runner.run(prompt):
        n_events += 1
        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    _print_tool_call(fc.name, dict(fc.args or {}))
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    result = fr.response or {}
                    _print_tool_result(fr.name, result)
                    _apply_step_results(session, result)
                    if fr.name == "detect_task_type" and "task_type" in result:
                        full_card = author_project_card(
                            problem_statement=problem, dataset_ref=dataset_ref,
                            task_type=result["task_type"], dataset_summary=_summarize_dataset(result),
                        )
                        session.project = full_card
                        _register_project(passport, full_card, result["task_type"], dataset_ref)
            if event.is_final_response() and event.content.parts:
                text = event.content.parts[0].text
                if text:
                    final_text = text
        if n_events >= MAX_EVENTS or (time.monotonic() - start) > MAX_SECONDS:
            print("\n[agent_runner] event/time budget reached, stopping the loop.")
            break

    # Deterministic fallback: if training finished but the agent never
    # called record_results, finish it ourselves so the demo always
    # produces a complete run, through the exact same write path.
    if toolkit.state.get("report_dict") and not toolkit.state.get("finalized"):
        print("\n[agent_runner] agent finished without calling record_results -- finishing deterministically.")
        result = toolkit.record_results()
        _apply_step_results(session, result)

    if toolkit.state.get("finalized"):
        _record_metrics_and_licence(session, toolkit)

    return final_text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Recipe Mentor autonomous agent run")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--dataset", required=True, help="Kaggle owner/slug")
    ap.add_argument("--user-key", default=DEFAULT_USER_KEY)
    ap.add_argument("--store", choices=("local", "firestore"), default="local")
    args = ap.parse_args(argv)

    print(f"[problem] {args.problem}")
    print(f"[dataset] {args.dataset}\n")

    priority = ""
    try:
        priority = input(
            "One clarifying question before I start: any priority I should know about "
            "-- favor faster training, or better accuracy? (Enter to skip)\n> "
        ).strip()
    except EOFError:
        priority = ""

    passport = _load_passport(store=args.store)
    key = slugify_project_key(args.problem, args.dataset)
    provisional = ProjectCard(
        key=key, concept=args.problem, sensors="(agent-run project)",
        baseline_dataset=f"Kaggle: {args.dataset}", why_good_baseline="",
        local_gap="", path_to_production="", first_90_days="",
    )
    session = MentorSession(passport, provisional, source="agent:cli")
    toolkit = AgentToolkit()

    kickoff = _build_kickoff_prompt(args.problem, args.dataset, passport, priority)
    print()
    final_text = asyncio.run(_drive(toolkit, session, passport, args.problem, args.dataset, kickoff))

    print("\n" + "=" * 60)
    if final_text:
        print(final_text)
    _print_progress(passport, session.project)

    feedback = ""
    try:
        feedback = input(
            "\nAny feedback on this run? Something to remember for next time? (Enter to skip)\n> "
        ).strip()
    except EOFError:
        feedback = ""
    if feedback:
        session.write_step_note(12, "noted", f"User feedback on this run: {feedback}")
        print("Noted -- recorded, and future same-task-type projects will see it.")

    _save_passport(passport, store=args.store)
    print(f"\nSaved -> {PASSPORT_PATH}" + (" (+ Firestore)" if args.store == "firestore" else ""))

    out = render_dashboard(passport)
    print(f"Dashboard -> {out}")


if __name__ == "__main__":
    main()
