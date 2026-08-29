# ADR: Recipe Mentor — ADK Backend and Firestore Store (Compliance Fixes)

Status: Implemented, verified live.
Scope: `recipe_mentor/llm.py` (rewritten), `firestore_store.py` (new),
`mentor.py` (backend/store wiring).
Covers: 2026-08-29, continuing `ADR_Recipe_Mentor_Vertex_Backend_2026-08-29.md`
and `ADR_Recipe_Mentor_Pipelines_2026-08-29.md`.

## Objective

A compliance re-read against the hackathon's actual submission rules (not
just the track description) found two of three mandatory technology
requirements unmet:

| Requirement | Status before this ADR |
|---|---|
| Gemini 3.5+ via Gemini API or Vertex AI | Met |
| A Google Agent Framework (ADK, GenAI SDK, Antigravity SDK, GenKit) | **Not met** — routing was through LangChain's `ChatVertexAI`, which isn't on the accepted list |
| A Google Cloud infra service (Cloud Run, Cloud SQL, Firestore, GKE, Pub/Sub) | **Not met** — Vertex AI (the model API) was the only Cloud usage |

Both are first-stage pass/fail gates before scoring even starts. This ADR
records the fixes.

## Decision: ADK replaces LangChain entirely, not alongside it

The user asked specifically for an ADK version. Rather than add ADK as a
third parallel backend next to the existing offline and LangChain-Vertex
paths, LangChain was dropped and ADK took its place directly — three
reasons, not just "the user asked":

1. **It's the compliance fix, not an addition to justify.** LangChain
   wasn't accepted; ADK is. Keeping both would mean shipping a backend that
   still doesn't count toward the requirement.
2. **ADK is architecturally the better fit anyway.** It's a purpose-built
   agent framework (sessions, runners, structured events) rather than a raw
   chat-model wrapper, which is a stronger story for the "Architectural
   Discipline" judging criterion (30%) than a thin LangChain shim was.
3. **It uses the same underlying GenAI SDK**, so Vertex AI routing,
   project, and model selection didn't need to change — only the client
   layer above it did.

Confirmed live (not assumed from docs, which disagreed with each other in
places): `google-adk`'s `LlmAgent` + `Runner` + `InMemorySessionService`
against `gemini-3.5-flash` on `location="global"` returns a real response
through the exact `runner.run_async()` / `event.is_final_response()` /
`event.content.parts[0].text` pattern now in `llm.py`. Vertex routing reads
`GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`
from the environment (confirmed against ADK's own docs, not assumed) —
`_ensure_vertex_env()` sets these with `setdefault` so a caller's own
environment isn't silently overridden.

**One real simplification this bought back**: ADK's response events give
plain text directly. The LangChain integration's `_response_text()` helper
— needed because Gemini 3.5's LangChain content came back as a list of
typed blocks with `thought_signature` noise (see the prior ADR) — is gone.
That helper existed to work around a LangChain-specific quirk, not a
Gemini one; it isn't needed on the other side of the swap.

**One session per mentor run, not one per step.** `AdkJudge` creates a
single ADK session and reuses it across all twelve steps, so the agent has
the whole recipe walk's prior exchanges in context rather than judging each
step in total isolation. `mentor.py`'s own step loop stays synchronous;
`AdkJudge.ask()` is the one place that bridges into `asyncio.run()`.

## Decision: a dedicated Firestore database, not the project's existing one

The `neofix-676da` project already had a `(default)` Firestore database
(created 2026-07-14, presumably for whatever else runs on that project).
Connecting to it via `google.cloud.firestore.Client(project=...)` failed
with a live `404: The database (default) does not exist`, despite
`gcloud firestore databases list` showing it present, IAM confirmed as
`roles/owner` (not a permissions issue), and the same failure with an
explicit `database="(default)"`. The one field that stood out:
`databaseEdition: ENTERPRISE`, versus the `STANDARD` edition every other
database on the account showed — the current stable `google-cloud-firestore`
client most likely doesn't reach an Enterprise-edition default database the
same way a Standard one is reached, though this wasn't fully root-caused
before moving on, since the actual fix was cheap and better practice
anyway.

**Fix: a new, dedicated Firestore Standard-edition database**, named
`recipe-mentor`, created via `gcloud firestore databases create
--database=recipe-mentor --location=us-central1 --type=firestore-native`.
Connectivity confirmed live with a real set/get/delete round trip.
Isolated from whatever the pre-existing default database is used for — the
same reasoning that led to piggybacking on `neofix-676da` itself rather
than touching another project's billing (see the Vertex Backend ADR):
prefer creating something new and isolated over risking something already
in use.

## Decision: Firestore is a real store, not a checkbox — but the local file stays the source of truth for portability

`firestore_store.py` maps a `Passport` onto one Firestore document
directly (`passport.to_compact_dict()` is already flat JSON — no
translation needed) under `passports/{user_key}` in the `recipe-mentor`
database. `mentor.py --store firestore` uses Firestore as the load-time
source of truth and **always also writes the local
`passports/lab_demo.json` mirror on save**, regardless of which store was
selected — so `dashboard/render_dashboard.py` and `record_results.py`
keep working unchanged either way, and SozoGraph's own central claim (a
portable, diffable JSON file you can move between runtimes) stays true
even when a session used Firestore. This isn't hedging the compliance
requirement; it's the same "one passport, multiple surfaces" model
SozoGraph already documents for any other persistence layer.

Verified live: an offline-backend `maize` session run with `--store
firestore` produced a real document in `recipe-mentor/passports/sozo_lab_demo`
with 12 real facts, confirmed by an independent read (a fresh `firestore.Client`
call, not the same process that wrote it). A subsequent `diesel_generator`
session with `--backend adk` (default `--store local`, reading the local
mirror the Firestore-backed session had also written) correctly recalled
maize's step 3 and step 6 lessons and judged the user's answers for real.

## Next steps

1. Full clean rehearsal: both projects, both backends, `--store firestore`
   end to end, from a passport wiped in both places first.
2. Architecture diagram for the submission, showing SozoGraph passport +
   ADK judge + Firestore store + the recall mechanism as one system.
3. Devpost submission text and the ~4-minute demo video (out of this
   session's scope to record, but the demo script is being written
   separately).
