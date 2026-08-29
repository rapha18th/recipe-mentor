# Recipe Mentor — a collaborative-partner hackathon build on SozoGraph

> **Editorial note:** this is the original implementation plan, written
> when Recipe Mentor was going to live inside the sozograph repo itself
> (`sozograph/examples/recipe_mentor/`, with `src/sozograph/...` links
> resolving against that monorepo). It was later extracted into this
> standalone repo, since sozograph is its own project with its own scope
> — this document is kept as-is for the reasoning trail; the file paths
> below reflect where things briefly lived, not this repo's current layout.
> See the root `README.md` and the `docs/ADR_*.md` files for what's
> actually here now.

## Context

The goal is a submission for the "All Things Agentic" hackathon's **Collaborative Partner** track: "stateful, multi-turn dialogue with real-time retrieval and persistent memory, so your agent adapts and personalizes based on past interactions instead of starting over each time." The track's own examples are a legal-document tutor that learns your weak spots and a UI/UX helper that learns your brand preferences from corrections — both are long-running relationships with one user, not a single-session spectacle.

The user has a real, internal working paper — **Full Spectrum** (Sozo Analytics Lab / Quantilytix AI, 19 Aug 2026) — that documents a twelve-step production recipe for shipping phone-class sensor ML products, already proven end to end by a prior shipped project, **SiloSense** (grain-pest detection: 106GB Kaggle dataset → 246MB verified subset → small CNN → 0.111MB static-INT8-quantized → ported to Kotlin → 3.47x on-device speedup, no accuracy loss). Full Spectrum then lists twenty project cards across vision, audio, vibration and fusion, each with the same seven fields (concept, sensors, baseline dataset, why it's a good baseline, the local gap, path to production, first 90 days). Several project cards explicitly reuse one recipe verbatim: the small-mill monitor, the diesel/mini-grid generator monitor, and the cold-room compressor monitor are all built on the same MIMII anomaly-detection pipeline, "unchanged, retargeted."

That reuse is the whole opportunity. A generic chatbot tutor cannot demonstrate real cross-session personalization convincingly in a two-minute judged demo — it has to be told to remember something and then asked to prove it. Full Spectrum hands us two project cards where the *same pipeline* is deliberately reapplied, so an agent that recalls a lesson from project 1 and applies it unprompted at the start of project 2 is a demo that proves itself, not one that claims to.

SozoGraph is the right memory substrate for this: it already models facts, dated observations, entities, episodes, and — critically — auto-generated contradictions when a belief changes, all in one portable JSON passport, no vector DB, no embeddings. Its one measured weakness (25.4% LoCoMo accuracy) is attributed by its own README to a weak local 8B extraction backbone, not the architecture — so swapping the backbone to Gemini via Vertex AI is expected to close most of that gap for free, no reindex, no migration.

The user also confirmed: use **Vertex AI**, not a plain AI Studio key — their live products are already hitting the AI Studio tier-1 spending cap and haven't been upgraded, so routing through a GCP project sidesteps a real operational blocker, not just a demo nicety.

The user also weighted the two demo projects after seeing the initial pairing: keep both vision and audio so the submission looks well-rounded, but don't spend the weekend re-running SiloSense's own domain (grain/mill acoustics) as a warm-up act — the small-mill card is dropped entirely. The **diesel/mini-grid generator health monitor gets priority and the most build depth**, because it's genuinely new work the user hasn't built before and wants to see actually run, not a rehearsed variation on something already shipped. The maize vision project stays real and load-bearing (not a fallback), it just gets less time, since fine-tuning a small vision CNN is comparatively well-trodden ground.

## The two demo projects

1. **Maize leaf disease diagnosis** (vision — session 1, lighter build). Baseline: PlantVillage maize classes (public, Kaggle/Hugging Face mirrors), swapped in for Full Spectrum's cassava card since Zimbabwe grows very little cassava and maize dominates local smallholder production instead. Honestly disclosed local gap, straight from the paper: PlantVillage's maize images are lab-photographed, not field-photographed. Real pipeline, real numbers, but scoped for speed — it exists to round out the submission and to give session 2 a lesson worth recalling.

2. **Diesel and mini-grid generator health monitor** (audio — session 2, the flagship, priority build). Baseline proxy: MIMII (Hitachi's industrial machine-sound corpus, Zenodo), the anomaly-only training approach Full Spectrum uses across its mill/generator/cold-room cards. Deeply Zimbabwe-relevant given chronic load-shedding and diesel-backup dependence, and genuinely new territory for the user. This is where the most build time, the most recipe depth, and the most stage time goes — and it's the session where the mentor must recall and apply session 1's lessons unprompted, proving personalization across a real modality switch (vision → audio), not just within one dataset family.

## Architecture

**State model — zero SozoGraph schema changes.** Every schema class in `src/sozograph/schema.py` is `extra="forbid"`, so recipe tracking is a *convention* over the existing sections, not a fork:

| Need | Section | Convention |
|---|---|---|
| Per-step status per project | `Fact` | `key="project:maize:step:03:status"`, `value="done"/"corrected"/"blocked"` |
| Verified metrics from a real run | `Fact` | `key="project:diesel_generator:metric:val_auc_int8"`, numeric value |
| Licence-check record | `Fact` | `key="project:diesel_generator:licence:mimii"`, dict value (`Fact.value` is `JSONValue`, a small dict is legal) |
| Dated lessons / mistakes | `Observation` | free text, `when` resolved to session date, append-only |
| Reversed decisions | `Contradiction` | **auto-generated** by `merge_passport_update` in [resolver.py](src/sozograph/resolver.py) whenever the same `Fact.key` is written twice with different values — no manual construction needed |
| Pipeline-reuse relationship | `Entity` | `Entity(name="diesel_generator_monitor", type="project", aliases=["mill_monitor_pipeline_reuse"])` |
| Session narrative | `Episode` | one per work session |

One shared passport (`passports/lab_demo.json`), one `user_key`, both projects namespaced by key prefix — the personalization demo depends on session 2 reading session 1's records from the same file.

**The one real gap, and the one real addition.** Plain BM25 + entity-expansion retrieval ([retrieve.py](src/sozograph/retrieve.py)) will not lexically surface a maize/vision lesson when the query is about a diesel-generator/audio session — no shared vocabulary, no shared named entity, different modality entirely. Since the Full Spectrum recipe is a fixed, closed taxonomy (steps 1–12, known at build time), the fix isn't semantic search — it's a small, deterministic **`recall.py`** helper in the new demo folder: a plain filter over `passport.facts`/`passport.observations` by `step:NN` tag, independent of BM25 relevance to the current utterance. This is the "memory skill" layer the user pre-authorized, and it's the only piece that isn't pure convention — and it's what makes the cross-modality recall demo possible at all.

**Backbone.** `SozoGraph("gemini:gemini-2.5-flash")` for the per-turn Socratic dialogue; consider `gemini-2.5-pro` for the cross-session recall citation specifically if Flash proves too terse in rehearsal.

**GCP routing — Vertex AI, not an AI Studio key.** Confirmed by reading [gemini.py](src/sozograph/providers/gemini.py): SozoGraph's `GeminiProvider` only supports `genai.Client(api_key=...)`, no Vertex/ADC path. The way in without forking the library is [providers/langchain.py](src/sozograph/providers/langchain.py)'s `LangChainProvider`, wrapping `langchain_google_vertexai.ChatVertexAI`:

```bash
gcloud auth login
gcloud projects create recipe-mentor-hackathon --set-as-default
gcloud config set project recipe-mentor-hackathon
gcloud billing projects link recipe-mentor-hackathon --billing-account=<BILLING_ACCOUNT_ID>
gcloud services enable aiplatform.googleapis.com --project=recipe-mentor-hackathon
gcloud iam service-accounts create recipe-mentor-sa --project=recipe-mentor-hackathon
gcloud projects add-iam-policy-binding recipe-mentor-hackathon \
  --member="serviceAccount:recipe-mentor-sa@recipe-mentor-hackathon.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
gcloud auth application-default login   # simplest for a weekend; or issue a service-account key if preferred
```
```python
pip install langchain-google-vertexai
```
```python
from langchain_google_vertexai import ChatVertexAI
from sozograph.providers.langchain import LangChainProvider
from sozograph import SozoGraph

sg = SozoGraph(LangChainProvider(chat_model=ChatVertexAI(
    model="gemini-2.5-flash", project="recipe-mentor-hackathon", location="us-central1")))
```
Set a GCP **budget alert**, not a hard spend cap — `gemini.py` retries transient 429/503s for up to ~30 minutes but fails fast on billing/spend-cap errors, which would kill a live demo mid-session.

**Repo layout** — new, disposable, inside the existing repo, `src/sozograph/` and `bench/` untouched:

```
sozograph/recipe_mentor/
├── README.md                # setup + demo script, judge-facing
├── recipe.py                 # the 12 Full Spectrum steps as data
├── projects/
│   ├── maize.py                # project card fields, verbatim from Full Spectrum + the swap rationale — session 1, lighter build
│   └── diesel_generator.py     # project card, priority build — session 2, the flagship
├── mentor.py                  # session loop: Socratic prompting, deterministic writes via merge_passport_update, sg.ingest() on dialogue
├── recall.py                  # the one enhancement layer: tag-based cross-project lookup by step:NN
├── pipelines/
│   ├── common_quant.py        # shared ONNX export + static QDQ INT8 calibration + FP32-vs-INT8 check — same module drives both projects' quantization step, step 6/7 code reuse across modalities
│   ├── maize_train.py          # PlantVillage subset download + small CNN + common_quant — kept intentionally lean
│   └── mimii_anomaly.py       # MIMII subset (Zenodo) + numpy mel-spectrogram (no librosa, per step 2) + autoencoder + common_quant — the priority pipeline, most build depth here
├── dashboard/
│   ├── render_dashboard.py    # reads Passport JSON → dashboard.html, no server/framework
│   └── template.html          # 12-step grid, metrics table, lessons timeline, contradictions log
├── passports/lab_demo.json    # the single shared passport
└── data/                      # gitignored downloads
```

## Weekend build order

- **Friday evening — environment.** GCP/Vertex setup above; `pip install -e ".[all]"` plus `torch`, `torchvision`, `onnxruntime`; smoke-test `SozoGraph` round-trip through `LangChainProvider`; scaffold the folder and write `recipe.py` + both project cards verbatim.
- **Saturday morning — Socratic loop + deterministic state, built against maize.** CLI session loop in `mentor.py`: ask what the user would do next per step, correct against the recipe's known risk points, advance `project:maize:step:NN:status` via `merge_passport_update`, and separately `sg.ingest()` the raw dialogue so natural-language lessons land as observations. Keep this scoped tight — it exists to prove the orchestration works and to seed real lessons for session 2, not to be the centerpiece.
- **Saturday afternoon — real maize pipeline, kept lean.** Bounded PlantVillage subset, small CNN, ONNX export, static QDQ INT8 via `common_quant.py`, FP32-vs-INT8 check on a held-out split, licence check recorded as a fact. Real numbers into the passport, not placeholders — but don't over-invest here; this is the lighter of the two builds by design.
- **Saturday evening — dashboard v1.** Render the maize card's 12-step grid, metrics, lessons timeline from `lab_demo.json`.
- **Sunday — diesel-generator session, `recall.py`, and the bulk of the pipeline work (protect this block, it's the priority).** This is where the weekend's real build time goes. `mimii_anomaly.py` (MIMII subset from Zenodo, numpy mel-spectrogram, no librosa per step 2, autoencoder anomaly model) gets more care than the maize pipeline did — more of the recipe's later steps genuinely exercised (calibrated static QDQ, FP32-vs-INT8 verification, confidence bands from the actual validation distribution per step 11, not an assumed cutoff). Wire `recall.py` so the diesel-generator session's opening turn proactively states session 1's step-3 and step-6 lessons, unprompted, before the user asks anything — the cross-modality recall proof.
- **Sunday evening — dashboard v2 + fallback capture.** Both cards side by side, lessons timeline spanning both, contradictions log with old/new values and timestamps, the recall annotation linking the two sessions. Record a short screen capture as a fallback in case live Vertex access is unstable on stage.
- **Monday (or whatever's left) — rehearsal + submission.** Run the full flow twice from a cold environment; write the Devpost description, giving the diesel-generator work the lead billing.

**On "real end to end":** both pipelines genuinely execute — bounded subsets, small models, few epochs, real static-INT8 verification, matching SiloSense's own scale — but they are not built to equal depth on purpose: maize is scoped for speed, the generator monitor gets the weekend's real attention. The only honest risk is live wall-clock time on stage for training, not whether the numbers are real: run both pipelines for real during the build, and during the judged demo itself, disclose plainly if a multi-minute training step is played back from a recording while everything SozoGraph-related — the dialogue, the deterministic writes, the recall citation, the quantization check — runs live. Never present a recording as live; that's the same data-honesty discipline Full Spectrum itself demands, applied to demo mechanics.

## Demo script

1. **Orient (30s).** Full Spectrum is real, from the user's own lab; the recipe is already proven on SiloSense (106GB → 246MB, 0.111MB quantized, 3.47x speedup, no loss).
2. **Session 1 — maize (live, kept brisk).** User deliberately answers step 3 wrong ("random 80/20 split"). Mentor corrects on screen, citing the recipe by number. Real pipeline runs, produces a genuine FP32-vs-INT8 number. Dashboard updates live: step 3 flips to `corrected`, a dated observation appears. This session moves quickly — it's here to be real and to seed the lesson, not to be the centerpiece.
3. **Show the raw passport (20s).** Open `lab_demo.json` directly — the dated observation visible in plain JSON.
4. **Session 2 — diesel generator (the proof moment, and the flagship — most of the demo's time lives here).** Before the user types anything about splits or quantization, the mentor's opening message states, unprompted: *"Two things tripped you up on the maize project — you split randomly instead of by source (step 3, corrected [date]), and initially quantized dynamically instead of statically (step 6). Applying the source-based split and static QDQ INT8 by default this time."* Point at the dashboard's lessons timeline pulling the same record forward, from a session in a completely different modality (vision → audio) — the strongest form of the personalization claim, since nothing about the domain overlaps except the recipe discipline itself. Then walk the generator pipeline in real depth: the MIMII anomaly-only setup, the numpy mel-spectrogram feature pipeline (explicitly no librosa, step 2 in practice), calibrated static QDQ quantization, FP32-vs-INT8 verification, and confidence bands read from the actual validation distribution (step 11) — this is where the weekend's real engineering shows, and where the most stage time should go, because it's the genuinely new work.
5. **Close on the dashboard (30s).** Both cards side by side, contradictions log with the reversed quantization decision (old/new values, both timestamps), the recall annotation linking the two sessions across modalities.

## Verification

- `pytest` still passes untouched in `src/sozograph` and `bench/` — nothing there changes.
- `python -m recipe_mentor.mentor --project maize --show-recipe` prints the 12-step checklist and project card (M0 sanity check).
- After Saturday's build: `lab_demo.json` contains real `project:maize:metric:*` facts with genuine numbers, not placeholders; open it directly to confirm.
- After Sunday's build: `lab_demo.json` contains real `project:diesel_generator:metric:*` facts, including a confidence band derived from the actual validation distribution (step 11), not an assumed cutoff.
- After Sunday's `recall.py` wiring: start the diesel-generator session cold from the saved passport and confirm the opening turn cites the dated step-3/step-6 records before any user input about splitting or quantization — this is the literal claim the whole submission rests on, so verify it by running the session fresh, not just reading the code.
- `render_dashboard.py` on the final passport renders both cards, the lessons timeline, and the contradictions log correctly in a browser with no server running.
