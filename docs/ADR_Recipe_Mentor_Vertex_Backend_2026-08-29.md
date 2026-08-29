# ADR: Recipe Mentor — Passport Convention and Vertex AI Backend

Status: Implemented. Core mechanism and the Vertex AI backend are verified
live, not assumed. Pipelines (`pipelines/common_quant.py`,
`pipelines/mimii_anomaly.py`, `pipelines/maize_train.py`) not yet built.
Scope: `recipe_mentor/`. Non-library change: `src/sozograph/` is
untouched.
Covers: 2026-08-28 to 2026-08-29.

## Objective

Build "Recipe Mentor," a submission for the "All Things Agentic" hackathon's
Collaborative Partner track ("stateful, multi-turn dialogue with persistent
memory, so your agent adapts and personalizes based on past interactions
instead of starting over each time"). It walks a builder through two of the
lab's own Full Spectrum project cards (maize leaf disease diagnosis, vision;
diesel/mini-grid generator health monitoring, audio) against the paper's own
twelve-step production recipe, and must recall a lesson from one project
unprompted when a second, unrelated one starts — the demo's actual proof of
personalization.

Two constraints shaped every decision below. First, the passport schema
cannot change: every model in `schema.py` is `pydantic.ConfigDict(extra=
"forbid")`, so recipe-step tracking has to fit inside the existing
facts/observations/contradictions/entities/episodes sections. Second, the
backbone has to run through Vertex AI, not a plain Gemini API key — the
lab's live products are already hitting the AI Studio tier-1 spending cap
and haven't been upgraded, so a second surface hitting the same cap was not
acceptable.

## Decision: recipe state as a Fact/Observation convention, not a schema change

Per-step status, verified metrics, and licence checks are `Fact`s, keyed
`project_{key}_step_{NN}_status` (and `_metric_*`, `_licence_*`). Lessons —
what a builder got wrong, in the mentor's own words — are `Observation`s,
`source` tagged `project:{key}:step:{NN}`. Reversed decisions need no
special handling: `merge_passport_update` already appends a `Contradiction`
automatically whenever the same `Fact.key` is written twice with a different
value.

**The bug this convention exists to warn about.** The first pass used colons
throughout (`project:maize:step:03:status`), matching how `Observation`
sources are written. It silently lost every write. `resolver._upsert_kv`
runs every incoming `Fact.key` through `utils.normalize_key`, which collapses
any run of non-`[a-z0-9]` characters — colons included — into a single
underscore, so the key actually stored was
`project_maize_step_03_status`, not what the caller thought it wrote.
`recall.py`'s prefix filter, built against the colon form, matched nothing;
`project_progress()` returned all-pending for a passport that had, in fact,
recorded every step. `Observation.source` has no such normalization anywhere
in `resolver.py`, so the colon form is safe there and was kept for
readability. The two sections now use two different delimiters on purpose,
each documented in both `recall.py` and `mentor.py` so the asymmetry does
not get "fixed" back into a second silent break.

## Decision: `recall.py` is a deterministic tag filter, not a retrieval query

Plain BM25 plus entity-expansion (`retrieve.py`) ranks against the current
utterance's vocabulary. A maize/vision lesson about splitting by source and
a diesel-generator/audio session share no vocabulary and no named entity, so
BM25 would never lift the maize lesson into the diesel-generator session's
context. But the Full Spectrum recipe is a fixed, closed taxonomy — twelve
numbered steps, known at build time — so the fix does not need semantic
search. `recall.py` filters `passport.facts`/`passport.observations` by
`step:NN` tag directly, independent of relevance-to-current-query. This is
the one piece of this build that is not pure convention over the existing
passport; everything else is.

## Decision: `LangChainProvider` + `ChatVertexAI`, not SozoGraph's native `GeminiProvider`

`src/sozograph/providers/gemini.py` calls `genai.Client(api_key=...)` only —
no `project`/`location`/`credentials` passthrough, confirmed by reading the
installed `google-genai` SDK's own `Client.__init__` signature, not assumed
from the wrapper's field list. `LangChainProvider` (already in the library,
unmodified) driving `langchain_google_vertexai.ChatVertexAI` gets full
ADC-based Vertex billing with zero changes to `src/sozograph/`. `llm.py`
isolates this import so the offline judging path never requires
`langchain_google_vertexai` to be installed.

## Decision: piggyback on `neofix-676da`, not a fresh dedicated project

The plan called for a new `recipe-mentor-hackathon` project. Its target
billing account (`My Billing Account 2`, `014C04-6A8151-80BE08`) was already
at a 5-project linking quota (`neofix-676da`, `helenia-11f98`, `sozo-daac1`,
`fchat-38767`, `tunasonga`) — confirmed by the link call's own
`FAILED_PRECONDITION` / `QuotaFailure` response, not guessed at. Rather than
request a quota increase (too slow for a weekend) or unlink an existing
project (blast-radius risk against something already running), `neofix-676da`
was reused: it was already billing-enabled on that account, so no linking
step was needed at all. `aiplatform.googleapis.com` enabled and ADC quota
project set on it. `recipe-mentor-hackathon` still exists, created but never
billing-linked — inert, not cleaned up.

## Decision: `gemini-3.5-flash`, `location="global"`

The hackathon requires Gemini 3.5 generation or newer. This account has no
Pro-tier model at that floor — Pro tops out at `gemini-3.1-pro`, below 3.5,
off the table regardless of preference. `location="us-central1"` 404s on
this project ("Publisher model ... was not found or your project does not
have access to it"); `location="global"` works. Both facts came from a live
call against the real API, not documentation, since the two disagreed with
each other in a first pass (search aggregators returned inconsistent
generation numbering; Google's own docs page, fetched directly, was closer
but still needed the live call to settle the region question).

**Response shape changed with the generation.** `ChatVertexAI.invoke(...)
.content` for `gemini-3.5-flash` is a list of typed blocks
(`[{"type": "text", "text": "...", "thought_signature": "..."}]`), not the
plain string older Gemini generations returned through the same client
class. `str(response.content)` would have silently stringified the whole
block list — thought signature included — into every parsed judgment.
`mentor.py::_response_text()` handles both shapes; this is the second blind
spot a live call caught that reading documentation alone would not have.

## Decision: two judges, one write path

`mentor.py --backend offline` (default) uses a narrow keyword heuristic,
real only for the two steps this demo's scripted mistakes exercise (3, 6);
everything else accepts any non-empty answer. Zero dependencies, runs with
no API key, exists so the whole mechanism is testable without GCP access.
`--backend vertex` sends the free-text answer to `gemini-3.5-flash` for a
genuine judgment against the step's stated rule, and uses the model's own
explanation as the recorded lesson text. Both judges write through the exact
same `merge_passport_update` call — correctness of the *stored state* never
depends on which judge decided it, only the judgment itself and its wording
do. Verified live: the Vertex judge is noticeably stricter than the offline
heuristic (it wants a real stated mechanism, not a keyword), which means a
from-scratch run with terse placeholder answers corrects most of the twelve
steps, not just the two scripted ones — worth knowing before a recorded
demo, since a clean two-lesson recall citation needs genuinely thorough
answers on the other ten.

## Verified live (2026-08-29)

A full `maize` session judged by `gemini-3.5-flash`, then a `diesel_generator`
session started cold from that saved passport, recalling five lessons
(steps 1, 3, 6, 7, 11 — `diesel_generator`'s own risk steps) from the maize
session in its opening message, unprompted, with real model-authored
feedback text, correctly dated and sourced. `pytest` on `src/sozograph`
untouched (161 passed, 4 skipped, unchanged).

## Next steps

1. Pull a bounded MIMII subset (one machine type, one SNR slice) from
   Zenodo for `pipelines/mimii_anomaly.py`.
2. Write and test `pipelines/common_quant.py` (ONNX export, static QDQ
   INT8 calibration, FP32-vs-INT8 verification) against real data before
   claiming it works — this repo's own "Risk and data honesty" discipline
   applies to this build too.
3. `pipelines/maize_train.py` against a bounded PlantVillage subset,
   reusing `common_quant.py` unchanged.
4. Wire the real per-project metrics (`project_{key}_metric_*` facts) from
   both pipelines into the passport and confirm the dashboard renders them.
