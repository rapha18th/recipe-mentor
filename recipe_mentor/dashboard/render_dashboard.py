"""
Renders a static dashboard.html from a saved Passport -- no server, no
framework. Regenerate after each session and reload the browser tab.

Usage:
    python -m recipe_mentor.dashboard.render_dashboard
    python -m recipe_mentor.dashboard.render_dashboard --out somewhere.html
"""
from __future__ import annotations

import argparse
import dataclasses
import html
from pathlib import Path

from sozograph.schema import Passport

from ..projects import ProjectCard
from ..projects.diesel_generator import DIESEL_GENERATOR
from ..projects.maize import MAIZE
from ..recall import project_progress
from ..recipe import STEPS

_HARDCODED_PROJECTS: tuple[ProjectCard, ...] = (MAIZE, DIESEL_GENERATOR)
PASSPORT_PATH = Path(__file__).parent.parent / "passports" / "lab_demo.json"
_CARD_FIELDS = {f.name for f in dataclasses.fields(ProjectCard)}
TEMPLATE_PATH = Path(__file__).parent / "template.html"
DEFAULT_OUT = Path(__file__).parent.parent / "dashboard.html"

_STATUS_CLASS = {
    "done": "status-done",
    "corrected": "status-corrected",
    "blocked": "status-blocked",
}


def _step_grid(passport: Passport, project: ProjectCard) -> str:
    progress = project_progress(passport, project.key)
    rows = []
    for s in STEPS:
        status = progress.get(s.number, "pending")
        css = _STATUS_CLASS.get(status, "status-pending")
        risk = "risk" if s.number in project.risk_steps else ""
        rows.append(
            f'<div class="step {css} {risk}">'
            f'<span class="step-num">{s.number}</span>'
            f'<span class="step-title">{html.escape(s.title)}</span>'
            f'<span class="step-status">{html.escape(status)}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def _metrics_table(passport: Passport, project: ProjectCard) -> str:
    prefix = f"project_{project.key}_metric_"
    rows = []
    for f in passport.facts:
        if f.key.startswith(prefix):
            name = f.key[len(prefix):]
            rows.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(f.value))}</td></tr>")
    if not rows:
        return '<tr><td colspan="2" class="empty">No verified metrics yet.</td></tr>'
    return "\n".join(rows)


def _lessons_timeline(passport: Passport) -> str:
    lessons = [o for o in passport.observations if o.source.startswith("project:")]
    lessons.sort(key=lambda o: o.ts, reverse=True)
    if not lessons:
        return '<li class="empty">No lessons recorded yet.</li>'
    items = []
    for o in lessons:
        when = o.when or o.ts.date().isoformat()
        items.append(
            f'<li><span class="lesson-date">{html.escape(when)}</span> '
            f'<span class="lesson-source">{html.escape(o.source)}</span><br>'
            f"{html.escape(o.text)}</li>"
        )
    return "\n".join(items)


def _contradictions_table(passport: Passport) -> str:
    if not passport.contradictions:
        return '<tr><td colspan="4" class="empty">No reversed decisions yet.</td></tr>'
    rows = []
    for c in passport.contradictions:
        rows.append(
            f"<tr><td>{html.escape(c.key)}</td>"
            f"<td>{html.escape(str(c.old))} <span class='ts'>({c.ts_old.date().isoformat()})</span></td>"
            f"<td>{html.escape(str(c.new))} <span class='ts'>({c.ts_new.date().isoformat()})</span></td>"
            f"<td>{html.escape(c.source_old)} &rarr; {html.escape(c.source_new)}</td></tr>"
        )
    return "\n".join(rows)


def _project_section(passport: Passport, project: ProjectCard) -> str:
    reuse = ""
    if project.reuses_pipeline_from:
        reuse = (
            '<p class="reuse-note">Reuses pipeline from: '
            + ", ".join(html.escape(r) for r in project.reuses_pipeline_from)
            + "</p>"
        )
    return f"""
    <section class="project-card">
      <h2>{html.escape(project.key)}</h2>
      <p class="concept">{html.escape(project.concept)}</p>
      {reuse}
      <div class="step-grid">{_step_grid(passport, project)}</div>
      <h3>Verified metrics</h3>
      <table class="metrics"><tbody>{_metrics_table(passport, project)}</tbody></table>
    </section>
    """


def _discover_projects(passport: Passport) -> tuple[ProjectCard, ...]:
    """The two hand-authored cards, plus one ProjectCard per entry the
    autonomous agent path registered in passport.meta["projects"]
    (projects/dynamic.py) -- so the dashboard grows with every real agent
    run, not just the two demo projects."""
    cards: dict[str, ProjectCard] = {p.key: p for p in _HARDCODED_PROJECTS}
    for key, data in passport.meta.get("projects", {}).items():
        if key in cards:
            continue
        kwargs = {k: v for k, v in data.items() if k in _CARD_FIELDS}
        if isinstance(kwargs.get("risk_steps"), list):
            kwargs["risk_steps"] = tuple(kwargs["risk_steps"])
        if isinstance(kwargs.get("reuses_pipeline_from"), list):
            kwargs["reuses_pipeline_from"] = tuple(kwargs["reuses_pipeline_from"])
        cards[key] = ProjectCard(**kwargs)
    return tuple(cards.values())


def render(passport: Passport, out_path: Path = DEFAULT_OUT) -> Path:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    body = "\n".join(_project_section(passport, p) for p in _discover_projects(passport))
    html_out = (
        template
        .replace("{{PROJECTS}}", body)
        .replace("{{LESSONS}}", _lessons_timeline(passport))
        .replace("{{CONTRADICTIONS}}", _contradictions_table(passport))
        .replace("{{UPDATED_AT}}", html.escape(passport.updated_at.isoformat()))
    )
    out_path.write_text(html_out, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passport", type=Path, default=PASSPORT_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.passport.exists():
        raise SystemExit(f"No passport at {args.passport} -- run a mentor session first.")
    passport = Passport.load(args.passport)
    out = render(passport, args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
