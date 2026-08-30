"""
A lightweight reflective agent: no tools, no autonomy -- just a real
Gemini conversation grounded in the passport's own history, for the
question every builder eventually asks: what have we actually done so
far, and what's worth trying next. Reuses llm.py's AdkJudge session
plumbing verbatim (one ADK agent + one session, reused across turns) --
the only thing specific to this feature is the system instruction and
the real passport summary it's grounded in.
"""
from __future__ import annotations

from sozograph.schema import Passport

from ..llm import AdkJudge
from ..recall import all_project_keys

CHAT_INSTRUCTION_TEMPLATE = """You are a thoughtful collaborator reflecting on a lab's ML project \
history with the person who built it. Below is the real, current state of every project recorded \
in their SozoGraph passport -- not a summary you're inferring, the actual facts and observations.

{summary}

Answer questions about what's been done, honestly -- including weak results and known gaps, don't \
paper over them. When asked what to explore next, suggest specific, concrete next experiments \
grounded in the actual gaps and lessons above (an unaddressed local-gap note, a task type never \
tried, a weak metric worth a real second attempt, a dataset property flagged during exploration \
but never acted on) -- not generic ML advice a stranger to this project history could have given. \
Keep answers to a few sentences unless asked for more detail."""


def build_passport_summary(passport: Passport, *, max_observations: int = 30) -> str:
    """A compact, real text summary of the whole passport -- every
    project's status and verified metrics, plus the most recent lessons
    -- grounding the chat in what's actually recorded, not what the model
    might otherwise guess or hallucinate."""
    all_keys = all_project_keys(passport)
    if not all_keys:
        return "No projects recorded yet -- this is a fresh passport."

    lines: list[str] = []
    for key in all_keys:
        prefix = f"project_{key}_"
        statuses = {
            f.key[len(prefix):-len("_status")]: f.value
            for f in passport.facts
            if f.key.startswith(prefix) and f.key.endswith("_status")
        }
        metrics = {
            f.key[len(prefix) + len("metric_"):]: f.value
            for f in passport.facts
            if f.key.startswith(prefix + "metric_")
        }
        done = sum(1 for v in statuses.values() if v == "done")
        lines.append(
            f"- {key}: {done}/{len(statuses) or 12} recipe steps done. "
            f"Metrics: {metrics if metrics else 'none recorded yet'}."
        )

    lines.append("\nRecent lessons (most recent first):")
    recent = sorted(passport.observations, key=lambda o: o.ts, reverse=True)[:max_observations]
    for obs in recent:
        when = obs.when or obs.ts.date().isoformat()
        lines.append(f"- [{when}] ({obs.source}) {obs.text}")

    if passport.contradictions:
        lines.append("\nReversed decisions:")
        for c in passport.contradictions[-10:]:
            lines.append(f"- {c.key}: {c.old!r} -> {c.new!r}")

    return "\n".join(lines)


def build_history_chat(passport: Passport, **kwargs) -> AdkJudge:
    """One ADK agent + session, grounded in the passport's state at the
    moment the chat starts. Reuses AdkJudge's exact plumbing -- this
    feature needed a different instruction, not different machinery."""
    summary = build_passport_summary(passport)
    instruction = CHAT_INSTRUCTION_TEMPLATE.format(summary=summary)
    return AdkJudge(name="recipe_mentor_history_chat", instruction=instruction, **kwargs)
