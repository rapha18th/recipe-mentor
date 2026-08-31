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
import io
import json
import os
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..agent.chat import build_history_chat
from ..agent.core import build_kickoff_prompt, format_event, provisional_project_card, run_agent
from ..agent.toolkit import AGENT_DATA_ROOT, AgentToolkit
from ..dashboard.render_dashboard import render as render_dashboard
from ..mentor import MentorSession, _load_passport, _save_passport
from ..recall import project_progress

#: The three files each real pipeline run writes -- see
#: pipelines/generic_image_train.py and generic_audio_anomaly_train.py.
#: Not every run has all three (a run that errored before training has
#: none), so the export endpoint below includes only what actually exists.
ARTIFACT_FILES = ("model_fp32.onnx", "model_int8.onnx", "report.json")

STORE = os.environ.get("RECIPE_MENTOR_STORE", "local")
STATIC_DIR = Path(__file__).parent / "static"
DASHBOARD_OUT = Path(tempfile.gettempdir()) / "recipe_mentor_dashboard.html"

LOCAL_DEMO = os.environ.get("RECIPE_MENTOR_LOCAL_DEMO", "").lower() in ("1", "true", "yes")
#: Deliberately not maize or the diesel generator -- both already have a
#: full history in the passport, so a recorded run on either would skip
#: straight to recall instead of showing the agent explore a dataset it has
#: never seen and ask a real, first-time question about it. Also
#: deliberately not a generic object-recognition set (cats, cars, flowers)
#: -- the lab's whole premise is models shipped into real field and
#: production settings, so the demo dataset should read the same way.
#: Casting product image data for quality inspection
#: (ravirajsinh45/real-life-industrial-dataset-of-casting-product on
#: Kaggle): 7,348 grayscale top-view photos of a submersible pump
#: impeller, from a real foundry's own inspection line, labeled
#: def_front / ok_front -- an image-side sibling of the diesel generator's
#: audio-side anomaly detection, same domain, different sensor.
LOCAL_DEMO_PROBLEM = "Flag defective castings on an automated visual inspection line"
LOCAL_DEMO_DATASET = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"

app = FastAPI(title="Recipe Mentor")

_lock = asyncio.Lock()
_runs: dict[str, dict[str, Any]] = {}
#: One reflective chat, grounded in the passport at the moment it started.
#: Rebuilt on "reset" (a fresh conversation) or if a run has happened
#: since it was built, so it's never grounded in stale history.
_history_chat: Any = None
_history_chat_built_at: str | None = None


class RunRequest(BaseModel):
    problem: str
    dataset: str
    kaggle_username: str = ""
    kaggle_key: str = ""
    priority: str = ""


class FeedbackRequest(BaseModel):
    feedback: str = ""


class ChoiceRequest(BaseModel):
    answer: str = ""


class ChatRequest(BaseModel):
    message: str
    reset: bool = False


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
    #: out_dir is deterministic from the dataset ref alone (see
    #: agent/toolkit.py::train_and_verify), so it's known before the run
    #: even starts -- the export endpoint just checks what's actually
    #: there once training's had a chance to write it.
    out_dir = AGENT_DATA_ROOT / "_run" / dataset.replace("/", "__")
    _runs[run_id] = {
        "queue": queue, "status": "running", "session": None, "passport": None,
        "toolkit": None, "dataset": dataset, "out_dir": out_dir,
    }
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
        _runs[run_id]["toolkit"] = toolkit

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


@app.get("/api/run/{run_id}/artifacts")
async def download_artifacts(run_id: str) -> StreamingResponse:
    """Zips up whatever this run actually wrote (model_fp32.onnx,
    model_int8.onnx, report.json) and streams it down. This exists because
    Cloud Run's filesystem is ephemeral -- those files live only as long as
    the container instance does, and nothing in this repo uploads them
    anywhere durable. The passport survives in Firestore; the model files
    only survive if whoever ran this downloads them, here, before the
    container recycles."""
    run = _runs.get(run_id)
    if run is None or not run.get("out_dir"):
        raise HTTPException(status_code=404, detail="Unknown run.")

    out_dir: Path = run["out_dir"]
    present = [f for f in ARTIFACT_FILES if (out_dir / f).exists()]
    if not present:
        raise HTTPException(
            status_code=404,
            detail="No artifact files yet for this run. Training may still be in progress, or the run errored before it got there.",
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in present:
            zf.write(out_dir / name, arcname=name)
    buf.seek(0)

    dataset = run.get("dataset", run_id)
    filename = f"recipe_mentor_{dataset.replace('/', '__')}.zip"
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/run/{run_id}/choice")
async def submit_choice(run_id: str, req: ChoiceRequest) -> dict:
    """Answers a pending ask_user() call -- the agent is genuinely
    blocked inside its own tool call until this arrives; nothing else in
    the run advances until a real answer lands here."""
    run = _runs.get(run_id)
    if run is None or run.get("toolkit") is None:
        raise HTTPException(status_code=404, detail="Unknown or unfinished run.")
    toolkit: AgentToolkit = run["toolkit"]
    if not toolkit.answer_pending(req.answer.strip()):
        raise HTTPException(status_code=409, detail="Nothing is currently waiting on an answer.")
    return {"ok": True}


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


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    """A real Gemini conversation grounded in the passport's own history --
    no tools, no autonomy, just the easy complement to the autonomous
    agent above: ask what's been done, get suggestions for what to try
    next, grounded in the actual recorded gaps and lessons, not generic
    advice."""
    global _history_chat, _history_chat_built_at
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="A message is required.")

    passport = _load_passport(store=STORE)
    stamp = passport.updated_at.isoformat()
    if req.reset or _history_chat is None or _history_chat_built_at != stamp:
        _history_chat = build_history_chat(passport)
        _history_chat_built_at = stamp

    # Not .ask() -- that wraps asyncio.run(), which can't be called from
    # inside a route handler already running in an event loop. This route
    # is already async, so it awaits the same underlying coroutine directly.
    reply = await _history_chat._ask_async(req.message.strip())
    return {"reply": reply}


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
