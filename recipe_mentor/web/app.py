"""
The hosted interface: a FastAPI app wrapping the same autonomous agent loop
agent_runner.py drives from a terminal, for a browser instead of a shell.
The driving logic itself lives in agent/core.py::run_agent() -- this module
only turns that event stream into Server-Sent Events and handles the two
human check-ins (priority before, feedback after) as HTTP requests instead
of input().

Auth split, deliberately: Vertex/Gemini access comes from the Cloud Run
service's own service account -- the same RECIPE_MENTOR_GCP_* environment
variables agent_runner.py already reads, never anything sent by the
browser. Kaggle credentials come from each visitor, per request -- Kaggle's
own terms don't allow sharing one person's token to download datasets on
another's behalf, so this has to be bring-your-own. A visitor's
KAGGLE_USERNAME/KAGGLE_KEY live in this process's environment only for the
duration of their run, restored (or cleared) in a `finally` block the
instant it ends, never written to disk, never logged.

One run at a time: an in-process lock rejects a second run with a plain
409 while one is active, backed up by the Cloud Run deploy's own
--concurrency=1 --max-instances=1 (see docs/DEPLOY.md) so two browser tabs
hitting two different container instances can't both hold Kaggle
credentials in environment variables at once.

Local demo mode (RECIPE_MENTOR_LOCAL_DEMO=1, never set in the Cloud Run
deploy): for recording a screen capture on your own machine, where the
Kaggle bring-your-own-token requirement above only exists to keep a
*public* service honest about whose dataset access it's using. Running
this yourself, on your own machine, under your own Kaggle account, isn't
that case -- so this mode lets the Kaggle fields go empty and falls back
to the same ~/.kaggle/kaggle.json file kaggle_fetch.py has always read,
and it prefills the problem/dataset fields so a recording can start
without typing a credential on camera. See docs/DEPLOY.md.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..agent.core import build_kickoff_prompt, format_event, provisional_project_card, run_agent
from ..agent.toolkit import AgentToolkit
from ..dashboard.render_dashboard import render as render_dashboard
from ..mentor import MentorSession, _load_passport, _save_passport
from ..recall import project_progress

STORE = os.environ.get("RECIPE_MENTOR_STORE", "local")
STATIC_DIR = Path(__file__).parent / "static"
DASHBOARD_OUT = Path(tempfile.gettempdir()) / "recipe_mentor_dashboard.html"

LOCAL_DEMO = os.environ.get("RECIPE_MENTOR_LOCAL_DEMO", "").lower() in ("1", "true", "yes")
#: The diesel-generator project's real proxy dataset (see
#: docs/ADR_Recipe_Mentor_Pipelines_2026-08-29.md for why MIMII-family
#: acoustic data stands in for a real generator corpus) -- already fetched
#: and cached locally as of this build, so a recording's first tool call
#: returns instantly instead of a live multi-minute download.
LOCAL_DEMO_PROBLEM = "Detect diesel generator bearing faults from acoustic recordings"
LOCAL_DEMO_DATASET = "senaca/mimii-pump-sound-dataset"

app = FastAPI(title="Recipe Mentor")

_lock = asyncio.Lock()
_runs: dict[str, dict[str, Any]] = {}


class RunRequest(BaseModel):
    problem: str
    dataset: str
    kaggle_username: str = ""
    kaggle_key: str = ""
    priority: str = ""


class FeedbackRequest(BaseModel):
    feedback: str = ""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    passport = _load_passport(store=STORE)
    out = render_dashboard(passport, DASHBOARD_OUT)
    return out.read_text(encoding="utf-8")


@app.post("/api/run")
async def start_run(req: RunRequest) -> dict:
    if _lock.locked():
        raise HTTPException(status_code=409, detail="A run is already in progress. Try again shortly.")
    problem, dataset = req.problem.strip(), req.dataset.strip()
    if not problem or not dataset:
        raise HTTPException(status_code=400, detail="A problem statement and a Kaggle dataset are both required.")
    if "/" not in dataset:
        raise HTTPException(status_code=400, detail="Dataset should look like owner/slug.")
    if not LOCAL_DEMO and (not req.kaggle_username.strip() or not req.kaggle_key.strip()):
        raise HTTPException(
            status_code=400,
            detail="Kaggle username and key are both required -- this run downloads a real dataset under your Kaggle identity.",
        )

    run_id = uuid.uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    _runs[run_id] = {"queue": queue, "status": "running", "session": None, "passport": None}
    asyncio.create_task(_execute_run(run_id, req, queue))
    return {"run_id": run_id}


async def _execute_run(run_id: str, req: RunRequest, queue: asyncio.Queue) -> None:
    await _lock.acquire()
    prev_user, prev_key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    #: In local demo mode with no credentials typed, leave the environment
    #: untouched -- KaggleApi.authenticate() falls back to
    #: ~/.kaggle/kaggle.json on its own, the same file-based auth this repo
    #: has documented from the start.
    set_env = bool(req.kaggle_username.strip() and req.kaggle_key.strip())
    try:
        if set_env:
            os.environ["KAGGLE_USERNAME"] = req.kaggle_username.strip()
            os.environ["KAGGLE_KEY"] = req.kaggle_key.strip()

        passport = _load_passport(store=STORE)
        session = MentorSession(
            passport, provisional_project_card(req.problem.strip(), req.dataset.strip()), source="agent:web",
        )
        toolkit = AgentToolkit()
        _runs[run_id]["session"] = session
        _runs[run_id]["passport"] = passport

        kickoff = build_kickoff_prompt(req.problem.strip(), req.dataset.strip(), passport, req.priority.strip())
        print(f"\n===== run {run_id} started -- Google ADK agent, Gemini 3.5 via Vertex AI =====", flush=True)
        async for event in run_agent(toolkit, session, passport, req.problem.strip(), req.dataset.strip(), kickoff):
            await queue.put(event)
            line = format_event(event)
            if line is not None:
                print(f"[{run_id}] {line}", flush=True)
        print(f"===== run {run_id} finished =====\n", flush=True)

        _save_passport(passport, store=STORE)
        await queue.put({"type": "progress", "progress": project_progress(passport, session.project.key)})
        _runs[run_id]["status"] = "awaiting_feedback"
    except Exception as exc:  # a run's own failure is real information, not a 500 the visitor never sees
        await queue.put({"type": "error", "message": str(exc)})
        _runs[run_id]["status"] = "error"
    finally:
        if set_env:
            if prev_user is not None:
                os.environ["KAGGLE_USERNAME"] = prev_user
            else:
                os.environ.pop("KAGGLE_USERNAME", None)
            if prev_key is not None:
                os.environ["KAGGLE_KEY"] = prev_key
            else:
                os.environ.pop("KAGGLE_KEY", None)
        await queue.put(None)  # sentinel: stream end
        _lock.release()


@app.get("/api/run/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run.")

    async def gen():
        queue: asyncio.Queue = run["queue"]
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/run/{run_id}/feedback")
async def submit_feedback(run_id: str, req: FeedbackRequest) -> dict:
    run = _runs.get(run_id)
    if run is None or run.get("session") is None:
        raise HTTPException(status_code=404, detail="Unknown or unfinished run.")
    session: MentorSession = run["session"]
    passport = run["passport"]
    if req.feedback.strip():
        session.write_step_note(12, "noted", f"User feedback on this run: {req.feedback.strip()}")
        _save_passport(passport, store=STORE)
    run["status"] = "done"
    return {"ok": True}


@app.get("/api/local-demo-defaults")
async def local_demo_defaults() -> dict:
    """Only ever returns something when RECIPE_MENTOR_LOCAL_DEMO is set --
    never in the Cloud Run deploy. Deliberately never includes a Kaggle
    credential, even locally: the frontend just hides the credential
    fields and submits blank ones, and the backend's own kaggle_fetch call
    falls back to ~/.kaggle/kaggle.json on the machine it's running on."""
    if not LOCAL_DEMO:
        raise HTTPException(status_code=404, detail="Not in local demo mode.")
    return {"problem": LOCAL_DEMO_PROBLEM, "dataset": LOCAL_DEMO_DATASET}


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "store": STORE, "local_demo": LOCAL_DEMO}
