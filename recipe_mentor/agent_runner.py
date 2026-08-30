"""
CLI entry point for the autonomous agent path: given a problem statement and
a named Kaggle dataset, a real Google ADK agent with tool-calling decides
how to fetch, detect, configure, train, and record a full ML pipeline run.
Each tool call is logged live and written into the shared SozoGraph
passport as it happens, via the same deterministic merge_passport_update()
path the Socratic mentor.py session uses -- the write side never depends on
the model being right, only on what its tools actually did.

The actual driving loop lives in agent/core.py::run_agent(), an async
generator shared with the hosted web interface (web/app.py) -- this module
is just the CLI's presentation of the same event stream, plus the two
human check-ins (see below).

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

from .agent.core import build_kickoff_prompt, format_event, provisional_project_card, run_agent
from .agent.toolkit import AgentToolkit
from .dashboard.render_dashboard import render as render_dashboard
from .mentor import DEFAULT_USER_KEY, PASSPORT_PATH, MentorSession, _load_passport, _print_progress, _save_passport


async def _drive_and_print(toolkit: AgentToolkit, session, passport, problem: str, dataset_ref: str, prompt: str) -> str:
    final_text = ""
    async for event in run_agent(toolkit, session, passport, problem, dataset_ref, prompt):
        # final_text prints once, after the "====" separator below, not here.
        if event["type"] != "final_text":
            line = format_event(event)
            if line is not None:
                print(("\n" if event["type"] in ("tool_call", "budget_reached", "fallback_finalized") else "") + line)
        if event["type"] == "done":
            final_text = event["final_text"]
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
    session = MentorSession(passport, provisional_project_card(args.problem, args.dataset), source="agent:cli")
    toolkit = AgentToolkit()

    kickoff = build_kickoff_prompt(args.problem, args.dataset, passport, priority)
    print()
    final_text = asyncio.run(_drive_and_print(toolkit, session, passport, args.problem, args.dataset, kickoff))

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
