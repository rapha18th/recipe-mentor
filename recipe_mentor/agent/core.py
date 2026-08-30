"""
The autonomous agent's driving loop, shared by the CLI (agent_runner.py)
and the hosted web interface (web/app.py) -- one place that builds the
kickoff prompt, drives the ADK tool-calling loop, and writes every real
outcome into the passport, so the two front ends can never drift apart on
what actually happens during a run. Neither front end reimplements any of
this; each only decides how to *display* the event stream this module
yields.

`run_agent()` is an async generator, not a return-a-result function,
because both front ends need to show a run happening live -- the CLI
prints each event as it arrives, the web interface forwards each one over
Server-Sent Events. The passport writes and project bookkeeping happen
inline as the generator advances, not after collecting a list, so a
caller that stops consuming early (a dropped web connection, a CLI
Ctrl-C) still leaves the passport in a consistent, already-recorded state
for whatever completed.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, AsyncIterator

from sozograph.resolver import merge_passport_update
from sozograph.schema import Entity, Passport

from .toolkit import AgentToolkit
from ..llm import build_agent_runner
from ..mentor import MentorSession
from ..projects import ProjectCard
from ..projects.dynamic import TASK_TYPE_RISK_STEPS, author_project_card, slugify_project_key
from ..recall import cross_project_recall

MAX_EVENTS = 60
MAX_SECONDS = 300

AGENT_INSTRUCTION = (
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


def provisional_project_card(problem: str, dataset_ref: str) -> ProjectCard:
    """A minimal ProjectCard, keyed correctly from the first write, before
    the task type (and the fuller, task-type-aware card) is known -- see
    projects/dynamic.py's slugify_project_key for why the key alone is
    computable up front."""
    return ProjectCard(
        key=slugify_project_key(problem, dataset_ref), concept=problem,
        sensors="(agent-run project)", baseline_dataset=f"Kaggle: {dataset_ref}",
        why_good_baseline="", local_gap="", path_to_production="", first_90_days="",
    )


def build_kickoff_prompt(problem: str, dataset_ref: str, passport: Passport, priority: str) -> str:
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


def _summarize_dataset(info: dict) -> str:
    if info.get("task_type") == "image_classification":
        return f"{len(info.get('classes', []))} classes: {', '.join(info.get('classes', []))}."
    return f"{info.get('n_normal', '?')} normal / {info.get('n_abnormal', '?')} abnormal clips."


def _apply_step_results(session: MentorSession, result: dict) -> None:
    for step_num, (status, detail) in result.get("step_results", {}).items():
        session.write_step_note(step_num, status, detail)


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


async def run_agent(
    toolkit: AgentToolkit, session: MentorSession, passport: Passport,
    problem: str, dataset_ref: str, prompt: str,
    *, max_events: int = MAX_EVENTS, max_seconds: float = MAX_SECONDS,
) -> AsyncIterator[dict[str, Any]]:
    """Drives one full agent run, yielding an event per thing that happens:
    {"type": "tool_call", "name", "args"}
    {"type": "tool_result", "name", "result"}   -- result includes step_results
    {"type": "project_identified", "task_type", "project_key"}
    {"type": "final_text", "text"}
    {"type": "budget_reached"}
    {"type": "fallback_finalized"}
    {"type": "done", "final_text"}
    Every passport write happens inline as these are yielded, not after."""
    tools = [
        toolkit.fetch_kaggle_dataset, toolkit.detect_task_type,
        toolkit.configure_run, toolkit.train_and_verify, toolkit.record_results,
    ]
    runner = build_agent_runner(tools, AGENT_INSTRUCTION)

    final_text = ""
    n_events = 0
    start = time.monotonic()
    async for event in runner.run(prompt):
        n_events += 1
        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    yield {"type": "tool_call", "name": fc.name, "args": dict(fc.args or {})}
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    result = fr.response or {}
                    yield {"type": "tool_result", "name": fr.name, "result": result}
                    _apply_step_results(session, result)
                    if fr.name == "detect_task_type" and "task_type" in result:
                        full_card = author_project_card(
                            problem_statement=problem, dataset_ref=dataset_ref,
                            task_type=result["task_type"], dataset_summary=_summarize_dataset(result),
                        )
                        session.project = full_card
                        _register_project(passport, full_card, result["task_type"], dataset_ref)
                        yield {
                            "type": "project_identified",
                            "task_type": result["task_type"], "project_key": full_card.key,
                        }
            if event.is_final_response() and event.content.parts:
                text = event.content.parts[0].text
                if text:
                    final_text = text
                    yield {"type": "final_text", "text": text}
        if n_events >= max_events or (time.monotonic() - start) > max_seconds:
            yield {"type": "budget_reached"}
            break

    # Deterministic fallback: if training finished but the agent never
    # called record_results, finish it ourselves so a run always completes,
    # through the exact same write path.
    if toolkit.state.get("report_dict") and not toolkit.state.get("finalized"):
        result = toolkit.record_results()
        _apply_step_results(session, result)
        yield {"type": "fallback_finalized"}

    if toolkit.state.get("finalized"):
        _record_metrics_and_licence(session, toolkit)

    yield {"type": "done", "final_text": final_text}


def format_event(event: dict[str, Any]) -> str | None:
    """One human-readable console line per event, or None to skip it --
    shared by agent_runner.py's CLI and web/app.py's server-side console
    mirror, so a terminal watching either front end shows the exact same
    real, live evidence of Google services being called: the ADK tool
    calls, in the order they actually happened, not a paraphrase."""
    kind = event["type"]
    if kind == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
        return f"[tool call] {event['name']}({args})"
    if kind == "tool_result":
        shown = {k: v for k, v in event["result"].items() if k != "step_results"}
        return f"[tool result] {event['name']} -> {shown}"
    if kind == "project_identified":
        return f"[identified] {event['task_type']} project -> {event['project_key']}"
    if kind == "final_text":
        return f"[agent] {event['text']}"
    if kind == "budget_reached":
        return "[note] event/time budget reached, stopping the loop"
    if kind == "fallback_finalized":
        return "[note] agent finished without calling record_results -- finishing deterministically"
    return None
