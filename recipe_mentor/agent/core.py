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

import asyncio
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

MAX_EVENTS = 80
#: Generous on purpose: ask_user() genuinely pauses for a real human
#: answer, and that wait time counts against wall-clock elapsed even
#: though the loop itself is idle (see run_agent's docstring on why the
#: budget check can't fire mid-wait, only after -- this just makes sure a
#: minute or two of real deliberation on-camera doesn't itself trip the
#: budget once the agent resumes).
MAX_SECONDS = 1200

AGENT_INSTRUCTION = (
    "You are an autonomous ML engineer, and a collaborative partner -- not a "
    "silent batch job. Work through these tools: fetch_kaggle_dataset, "
    "detect_task_type, explore_dataset, ask_user, configure_run, "
    "train_and_verify, record_results.\n\n"
    "After detect_task_type and explore_dataset, you will know real, specific "
    "things about this dataset -- class balance, sample-rate consistency, "
    "corrupt files, image dimensions -- and you were told, at the start of this "
    "conversation, what past projects in this lab learned the hard way. Before "
    "calling configure_run, call ask_user with ONE concrete, specific question "
    "about a real preprocessing or configuration decision this dataset actually "
    "presents, grounded explicitly in what explore_dataset found and in any "
    "relevant past lesson -- name 2-4 real options with a one-line tradeoff "
    "each, and say which you'd recommend and why. Do not ask a vague or generic "
    "question; if explore_dataset found nothing decision-worthy, say so plainly "
    "and proceed without asking. You may call ask_user again if a second real "
    "decision comes up later (for instance, if train_and_verify's result is "
    "surprising and there's a genuine choice about how to proceed) -- but only "
    "when there's an actual decision, not to seem collaborative.\n\n"
    "Each tool enforces the recipe's discipline itself -- you cannot skip "
    "quantization or verification, only choose hyperparameters within the "
    "range the tool accepts. For image_classification, configure_run also "
    "accepts image_size (the resize resolution training actually uses) -- if "
    "your ask_user question was about resolution or image dimension spread, "
    "pass the resulting choice here, don't just narrate it. configure_run's "
    "hyperparameters should reflect whatever the user actually chose in "
    "ask_user, not be decided independently of it. If a tool returns an "
    "error, read it and correct your "
    "next call; do not repeat the same call unchanged. If the user stated a "
    "priority for speed vs. accuracy at the very start, let that guide your "
    "epoch count choice too. After record_results succeeds, write one short "
    "paragraph summarizing what you found and what was decided together, in "
    "your own words, speaking directly to the person who asked for this "
    "project -- then stop."
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
    {"type": "awaiting_choice", "question", "options"}  -- ask_user is now
        genuinely blocked; the caller must get a real answer and call
        toolkit.answer_pending(answer) before this generator advances again
    {"type": "project_identified", "task_type", "project_key"}
    {"type": "final_text", "text"}
    {"type": "budget_reached"}
    {"type": "fallback_finalized"}
    {"type": "done", "final_text"}
    Every passport write happens inline as these are yielded, not after.

    "awaiting_choice" is NOT inferred from the ADK function_call event for
    ask_user -- that was tried and produced a real, reproducible hang.
    The function_call event announces the model's *decision* to call
    ask_user; the tool's own coroutine (and the asyncio.Future it awaits)
    doesn't exist until ADK actually invokes it, one step later. Answering
    on the function_call event resolves a future that isn't there yet, and
    the real one -- created moments after -- waits forever. Instead,
    toolkit.ask_user() fires a synchronous callback the instant it
    genuinely starts waiting, bridged into this generator's own yield
    stream via an asyncio.Event, concurrently merged against the ADK
    event stream below.

    That merge drives `runner.run(prompt)` from exactly one persistent
    background task (`_pump`), not a fresh `asyncio.ensure_future(anext(...))`
    per event -- also tried, also a real bug: ADK's own tracing keeps a
    contextvar token across an operation, and re-wrapping `__anext__()` in
    a new Task each iteration silently split that operation across two
    different task contexts, raising `ValueError: ... was created in a
    different Context` the first time a tool call and its result landed in
    separate iterations. One task owns the whole ADK stream for its entire
    lifetime; only the queue reads it feeds are re-awaited per iteration."""
    tools = [
        toolkit.fetch_kaggle_dataset, toolkit.detect_task_type, toolkit.explore_dataset,
        toolkit.ask_user, toolkit.configure_run, toolkit.train_and_verify, toolkit.record_results,
    ]
    runner = build_agent_runner(tools, AGENT_INSTRUCTION)

    choice_ready = asyncio.Event()
    pending_choice: dict[str, Any] = {}

    def _on_awaiting_choice(question: str, options: list[str]) -> None:
        pending_choice["question"] = question
        pending_choice["options"] = options
        choice_ready.set()

    toolkit.on_awaiting_choice = _on_awaiting_choice

    final_text = ""
    n_events = 0
    start = time.monotonic()

    queue: asyncio.Queue = asyncio.Queue()

    async def _pump() -> None:
        try:
            async for ev in runner.run(prompt):
                await queue.put(("event", ev))
        except Exception as exc:  # surfaced to the caller below, not swallowed
            await queue.put(("error", exc))
        finally:
            await queue.put(("done", None))

    pump_task = asyncio.ensure_future(_pump())
    queue_task = asyncio.ensure_future(queue.get())
    choice_task = asyncio.ensure_future(choice_ready.wait())

    try:
        while True:
            done, _ = await asyncio.wait({queue_task, choice_task}, return_when=asyncio.FIRST_COMPLETED)

            if choice_task in done:
                choice_ready.clear()
                yield {
                    "type": "awaiting_choice",
                    "question": pending_choice.get("question", ""),
                    "options": list(pending_choice.get("options", [])),
                }
                choice_task = asyncio.ensure_future(choice_ready.wait())
                continue  # queue_task, if also ready, gets picked up next iteration

            kind, payload = queue_task.result()
            queue_task = asyncio.ensure_future(queue.get())

            if kind == "done":
                break
            if kind == "error":
                raise payload
            event = payload

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
    finally:
        toolkit.on_awaiting_choice = None
        for t in (pump_task, queue_task, choice_task):
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass

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
    if kind == "awaiting_choice":
        opts = event.get("options") or []
        suffix = f" Options: {', '.join(opts)}" if opts else ""
        return f"[awaiting your answer] {event['question']}{suffix}"
    if kind == "project_identified":
        return f"[identified] {event['task_type']} project -> {event['project_key']}"
    if kind == "final_text":
        return f"[agent] {event['text']}"
    if kind == "budget_reached":
        return "[note] event/time budget reached, stopping the loop"
    if kind == "fallback_finalized":
        return "[note] agent finished without calling record_results -- finishing deterministically"
    return None
