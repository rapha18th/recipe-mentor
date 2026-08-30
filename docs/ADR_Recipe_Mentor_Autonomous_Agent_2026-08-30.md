# ADR: Recipe Mentor — The Autonomous Agent Path

Status: Implemented and verified live, both task types, both the tool-calling
loop and the full CLI.
Scope: `recipe_mentor/agent/`, `recipe_mentor/agent_runner.py`,
`recipe_mentor/pipelines/kaggle_fetch.py`,
`recipe_mentor/pipelines/generic_image_train.py`,
`recipe_mentor/pipelines/generic_audio_anomaly_train.py`,
`recipe_mentor/projects/dynamic.py`.
Covers: 2026-08-30, continuing the three 2026-08-29 ADRs.

## Objective

The Socratic path (`mentor.py`, `llm.py`'s `AdkJudge`) isn't actually an
agent. A human types free-text answers to fixed questions; an ADK `LlmAgent`
with no `tools=` grades them. It proves cross-session recall and
deterministic passport writes, but nothing calls a tool, decides a
hyperparameter, or acts on anything. This ADR covers turning that into a
real one: given a problem statement and a named Kaggle dataset, a genuine
ADK tool-calling agent fetches, trains, quantizes, verifies, and records a
full ML pipeline run on its own, then hands the result to the exact same
passport, recall, and dashboard machinery the Socratic path already built.

Read mid-build, not assumed going in: the track's own requirements text
asks for more than autonomy -- "ask clarifying questions, guide the user
step-by-step, and have a clear way to capture feedback, so it constantly
adapts to the user's unique way of thinking." Two decisions below
(the harness-asked check-ins, and step 12 as the feedback channel) exist
specifically to satisfy that, not the autonomy claim alone.

## Decision: five LLM-facing tools, not ten

The ADK agent gets exactly `fetch_kaggle_dataset`, `detect_task_type`,
`configure_run`, `train_and_verify`, `record_results` -- bound methods on
one `AgentToolkit` instance (`agent/toolkit.py`). State (file paths, class
lists, ONNX paths) lives in `toolkit.state`, never round-tripped through
the model; the LLM only ever passes small bounded scalars (`owner_slug`,
`epochs`, `batch_size`, `lr`). A ten-tool design (separate `export_onnx`/
`quantize`/`verify` calls, say) would require the model to correctly pass
file paths and arrays between calls across multiple turns -- needless
surface area for a demo, for zero functional gain, since none of those
finer-grained steps are places the model should have a real decision to
make anyway. Coarse tools, real work inside each one: `train_and_verify`
alone runs training, ONNX export, static QDQ INT8 quantization, and
FP32-vs-INT8 verification as one inseparable call.

**Verified live** (see below): a real ADK `LlmAgent` with these five tools,
against Gemini 3.5 via Vertex AI, called all five in the correct order
unprompted beyond the system instruction, read a tool's own clamp-notice
back and adjusted nothing further (didn't need to -- clamping is silent
and correct by design), and wrote a real closing paragraph citing its own
tool results (accuracy numbers, quantized size) in its own words.

## Decision: the tools enforce the recipe's discipline, not the model's judgment

`train_and_verify()` always runs quantize-then-verify; there is no tool
that would let the agent skip verification, the same way there's no tool
that does anything BUT static QDQ (`quantize_static_int8` has no dynamic
code path to call by mistake). `configure_run()` clamps `epochs`/
`batch_size`/`lr` into a hardcoded range per task type
(`agent/bounds.py`) and reports the clamp back in the tool's own result --
visible, not hidden, and the model can react to it like any other tool
output. Split discipline (recipe step 3) is enforced inside the training
pipelines themselves: `generic_audio_anomaly_train.py` splits by detected
source subfolder when the dataset has one, and falls back to a file-level
split with an honest recorded caveat when it doesn't -- the exact
`corrected` vs `done` distinction verified live below. Out-of-order tool
calls return a structured `{"error": ...}` dict, not an exception -- ADK
feeds that back to the model as the function's own response, which is real
agentic error recovery, not a crash. Verified directly (no ADK, no LLM):
calling `detect_task_type`/`configure_run`/`train_and_verify`/
`record_results` on a fresh `AgentToolkit` each returned the expected
"call X first" error.

## Decision: reuse the fixed 12-step `recipe.py`, no dynamic stage list

An agent-run project is not a new kind of thing in this passport -- it's
a `ProjectCard`, same as maize/diesel_generator, walking the same twelve
steps. Steps 8-10 (on-device port, runtime profiling, physical-device
benchmark) get `not_attempted` from `record_results()`, since this run
never touches a physical device -- itself an application of step 12's own
"report what's not yet done" rule, not a shortcut around it. This removed
the single largest architectural risk from a one-day build: `recall.py`,
`recipe.py`, `ProjectCard`, and the dashboard's step grid needed zero
structural changes.

## Decision: `passport.meta["projects"]`, not a schema change

A dynamically created project needs a full `ProjectCard` (for
`fact_prefix()`, `risk_steps`, the dashboard) but doesn't get one
hand-authored. `projects/dynamic.py::author_project_card()` builds one
deterministically -- templated prose from the problem statement, dataset
ref, and detected task type, not a second LLM call, since these fields
don't affect pipeline correctness and a second unstructured model call is
one more thing that can go wrong on demo day for no functional benefit.
The full card is stored in `passport.meta["projects"][key]` --
`meta: dict[str, Any]` is unconstrained in `sozograph`'s own schema, and
`resolver.py` never touches any key inside it but `"dedupe"`, confirmed by
reading the resolver before relying on it, not assumed. Round-trips fine
through `Passport.save()`/`load()` and Firestore's `.set()` since it's
plain JSON-safe data (tuples become lists on the JSON round-trip;
`dashboard/render_dashboard.py::_discover_projects()` converts them back).

The project's key is computed once, up front
(`slugify_project_key(problem_statement, dataset_ref)`), before the agent
has even called `detect_task_type` -- deliberately, so every tool call's
passport write (starting with `fetch_kaggle_dataset`'s own step-1 note)
lands under the same `fact_prefix()` from the very first write, with no
re-pointing needed once the task type (and the fuller, task-type-aware
`ProjectCard`) becomes known mid-run.

## Decision: cross-project recall, filtered by task type, without knowing it yet

`cross_project_recall()` gained an optional `task_type` filter
(`recall.py`), reading `passport.meta["projects"][key]["task_type"]` where
present -- absent for the two legacy hardcoded projects, so they're never
filtered out, unchanged behavior for the Socratic path. But at kickoff
time, `agent_runner.py` doesn't yet know which task type this run will
turn out to be (the dataset hasn't been fetched, let alone inspected).
Rather than a multi-turn message-injection scheme to recall lessons only
after `detect_task_type` runs, the kickoff prompt recalls for BOTH known
task types up front, labeled ("if this turns out to be a
`image_classification` project..."), and lets the model apply whichever
turns out relevant -- a small, real piece of judgment left to the agent,
not the harness, and it avoids the reliability cost of a second live
message mid-loop.

**Verified live**: after one completed `image_classification` agent run, a
fresh `_build_kickoff_prompt()` call for a second, unrelated
`image_classification` problem correctly recalled all six of the first
run's step-1/2/3/4/6/12 lessons, dated and sourced -- including the
captured user feedback from step 12 (see below). Separately, and
unplanned but real: the existing Socratic `maize` session's own opening
turn now also recalls the agent-run project's step 1-4 lessons (maize's
`risk_steps`), since both write into the same shared passport under the
same tag convention -- recall flows in both directions between the
Socratic and autonomous paths, not just within one of them.

## Decision: the harness asks the clarifying question and captures feedback, not the model mid-loop

The Collaborative Partner track's own text: "ask clarifying questions,
guide the user step-by-step, and have a clear way to capture feedback, so
it constantly adapts to the user's unique way of thinking." `agent_runner.py`
asks one real question before the run (`input()`, speed-vs-accuracy
priority, folded into the kickoff prompt as guidance the model can act
on -- verified live below, it did) and one after (`input()`, free-text
feedback). Neither is phrased live by the LLM mid-tool-loop. That's a
deliberate reliability choice, not an oversight: ADK's tool-calling loop is
built around executing tools to completion, not pausing for a human
mid-stream, and this build's whole philosophy (see `agent/toolkit.py`'s own
docstring) is keeping the model's freedom bounded and the harness's job
deterministic. The feedback is written via
`session.write_step_note(12, "noted", feedback_text)` -- the exact same
mechanism `record_results()` already uses for its own step-12 honesty
note, so it costs no new passport machinery, and it resurfaces
automatically for the next same-task-type project through
`cross_project_recall` once step 12 was added to both
`TASK_TYPE_RISK_STEPS` tuples.

## Two supported task types, out of deliberate scope discipline

`image_classification` (generalizes `maize_train.py`) and `audio_anomaly`
(generalizes `mimii_anomaly.py`), each with a concrete dataset-shape
contract `agent/detect.py` checks for before anything trains:
class-labelled image subfolders, or `normal`/`abnormal`-style folders of
`.wav` files. Tabular anomaly detection was left out on purpose -- no
existing pipeline template, and building one from nothing was the
highest-risk, lowest-necessity item available under this deadline.

`generic_audio_anomaly_train.py` tolerates a sample rate other than
MIMII's native 16kHz via plain linear-interpolation resampling in NumPy,
not a new dependency -- `features.py`'s own mel math is reused completely
unchanged; only the file-loading step needed to stop assuming 16kHz.

## Verified live (2026-08-30)

**`kaggle_fetch.py`**: real downloads via `KaggleApi`, both datasets --
`smaranjitghose/corn-or-maize-leaf-disease-dataset` (4188 files, 171.2MB,
already known-good from `maize_train.py`) and a newly-picked one for the
audio contract, `senaca/mimii-pump-sound-dataset` (519 files, 1.3GB,
16kHz/8-channel/10s clips, a `normal/`+`abnormal/` layout).

**`agent/detect.py`**: correctly identified both -- `image_classification`
with the exact same four classes and counts `maize_train.py`'s own
docstring documents (Blight 1146, Common_Rust 1306, Gray_Leaf_Spot 574,
Healthy 1162); `audio_anomaly` with 381 normal / 138 abnormal clips, no
source subfolders (flat layout, correctly triggering the file-level-split
fallback rather than assuming a source split that isn't there).

**`generic_image_train.py`**: 3-epoch smoke run, 600/128/128 split,
23,844-param CNN, FP32 35.94% -> INT8 36.72% accuracy, correctly flagged
`no_loss (n too small...)`, 97.1KB -> 32.3KB (3.0x). In line with
`maize_train.py`'s own committed numbers at comparable scale.

**`generic_audio_anomaly_train.py`**: two real runs. A 25-epoch smoke run
(267 train / 57 val / 57 test-normal / 138 test-abnormal) landed FP32 AUC
0.29 -- genuinely weak, undertrained, reported as-is, not discarded. A full
150-epoch run on the same split reached FP32 AUC 0.83 -> INT8 AUC 0.86,
`no_loss` (195 eval examples, under the 200 noise floor, correctly
downgraded rather than reported as an INT8 win). This pump dataset trains
to a real, usable AUC where MIMII valve (the existing `mimii_anomaly.py`
pipeline, 0.42 at the same scale) does not -- both numbers are genuine
properties of their respective datasets at this training scale, not a
tuning difference between the two pipelines.

**Live ADK tool-calling** (`llm.py::AgentRunner`, Gemini 3.5 via Vertex
AI): a real `LlmAgent` with the five bound-method tools called
`fetch_kaggle_dataset` -> `detect_task_type` -> `configure_run` ->
`train_and_verify` -> `record_results` in the correct order, unprompted
beyond the system instruction, and closed with a real narration paragraph
grounded in its own tool results (accuracy numbers, quantized size,
correctly noting steps 8-10 remain).

**Full `agent_runner.py` CLI**, end to end, piped stdin standing in for a
human at both check-ins: asked the clarifying priority question, folded
"favor speed" into the kickoff prompt, watched the agent itself propose
`epochs=2` (respecting that priority) and get silently clamped to 3 (the
`image_classification` floor) -- a real instance of the clamp mechanism
responding to genuine model behavior, not a hypothetical. Wrote 22 Facts
and 10 Observations into `passports/lab_demo.json` under the exact
`project_{key}_step_{NN}_status` / `project:{key}:step:{NN}` conventions
`recall.py`'s own docstring specifies, registered the project into
`passport.meta["projects"]`, captured the post-run feedback at step 12,
and rendered a three-project dashboard (`maize`, `diesel_generator`, and
the new agent-run project) with no server running.

## Addendum, 2026-08-30 (later the same day): mid-run collaboration, not just bookend questions

Fair critique, acted on the same day: a run that fetches, trains, and
records with no human input in the middle isn't "collaborative" just
because it asks one question before and one after. Real collaboration
means presenting actual decisions -- preprocessing, configuration -- as
choices, grounded in what this specific dataset's exploration finds and
what past runs' recorded gotchas suggest, not deciding silently and only
reporting the outcome afterward.

**This corrects a claim in the section above.** "ADK's tool-calling loop
is built around executing tools to completion, not pausing for a human
mid-stream" turned out to be wrong, checked rather than left as an
assumption: ADK's `FunctionTool._invoke_callable` (confirmed by reading
the installed `google-adk` package's own source, not guessed) detects
`inspect.iscoroutinefunction(target)` and `await`s it directly. An async
tool method that itself `await`s a real `asyncio.Future` genuinely
suspends the whole tool-calling loop until something outside resolves
that future -- no polling, no fake wait, the ADK `Runner`'s event stream
simply produces no further events until it does.

**Two new tools.** `explore_dataset()` (`agent/explore.py`) samples the
fetched dataset for real, empirical findings -- image dimension spread
and corrupt-file count for image_classification; sample-rate consistency,
duration range, and clipping for audio_anomaly -- distinct from
`detect_task_type`'s pure shape-matching. `ask_user(question, options)`
(`agent/toolkit.py`) is the one genuinely blocking tool: it creates an
`asyncio.Future`, awaits it, and returns whatever `answer_pending()` sets
it to from outside (`agent_runner.py`'s `input()`, or `web/app.py`'s new
`/api/run/{id}/choice` endpoint). The system instruction directs the
agent to call `ask_user` after `explore_dataset`, grounded explicitly in
its findings plus the cross-project recall already given at kickoff, with
2-4 concrete options and a stated recommendation -- not a vague question,
and not asked merely to seem collaborative when there's nothing to decide.

**A real gap this caught, live, in the first test run.** The agent asked a
genuinely well-grounded question -- image resize resolution, citing the
exact dimension spread `explore_dataset` found (256x256 to 4608x3456) --
that `configure_run` had no way to act on: `generic_image_train.py`
hardcoded `IMG_SIZE = 64`. Caught by testing the live loop, not by reading
the code, exactly the discipline this repo's own earlier ADRs already
practice. Fixed properly, not patched around: `image_size` is now a real
parameter threaded through `generic_image_train.run()` (safe because
`SmallCNN`'s `AdaptiveAvgPool2d` makes the architecture resolution-agnostic),
bounded in `agent/bounds.py` (32-224px) the same way epochs/batch_size/lr
already were, and `configure_run` accepts it so a user's answer to that
exact question changes what actually trains, not just what gets narrated.

**Verified live**: the full loop -- fetch (cached) -> detect -> explore ->
a real, specific, memory-and-finding-grounded `ask_user` call, citing the
actual class-imbalance ratio and dimension spread found -- confirmed twice
against the maize dataset. The harness's own deliberate design (state
never touches the model, only small scalars) held even under this new,
more open-ended tool: the model proposed real numbers, `configure_run`
clamped what needed clamping, and the pause-then-resume mechanism worked
exactly as designed.

## Addendum, 2026-08-30: a lighter reflective chat, no tools involved

The easier complement to the above, and genuinely simpler to build: a
plain conversation grounded in the passport's own recorded history, for
"what have we done so far, what's worth trying next." `agent/chat.py`
reuses `llm.py`'s `AdkJudge` session plumbing verbatim -- no new agent
machinery, just a different system instruction, built once from
`build_passport_summary()` (every project's real status, verified
metrics, and most recent lessons, reusing `recall.py`'s own
`all_project_keys()`). Exposed as `POST /api/chat` on the web interface,
with a chat panel always visible below the run form. Answers are honest
about weak results by instruction, and suggestions are directed to be
concrete and grounded in the actual recorded gaps -- not generic ML
advice a stranger to this project's history could have given.

## Next steps

1. A real local corpus for either supported task type, if there's time --
   the same "public baseline proves the pipeline, not the product" caveat
   the pipelines ADR already states for MIMII applies here too.
2. A third supported task type (tabular), if the deadline allows it --
   `detect_task_type`'s error-dict contract already makes this additive,
   not a rework.
3. `ask_user` could reasonably fire a second time after a surprising
   `train_and_verify` result (the system instruction already allows this,
   not yet observed live) -- worth confirming in rehearsal.
4. Full rehearsal: an agent run recorded start to finish for the demo,
   from a cold environment, alongside the existing Socratic sessions.
