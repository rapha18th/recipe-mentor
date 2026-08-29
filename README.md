# Recipe Mentor

A Collaborative Partner agent, built on
[SozoGraph](https://github.com/Sozo-Analytics-Lab/sozograph)
([PyPI](https://pypi.org/project/sozograph/)). It adapts and personalizes
based on past interactions instead of starting over each time. It walks a
builder through a real twelve-step ML production recipe for two project
cards. It recalls what tripped the user up in one project, unprompted, the
moment they start a second, unrelated one.

Built for the "All Things Agentic" hackathon's Collaborative Partner track.
The pattern generalizes. Any structured, repeatable workflow, a recipe, a
runbook, a checklist someone follows more than once, can sit on top of the
same three pieces: a portable memory passport, a deterministic tag-based
recall layer, and a real LLM judge. Fork this as a template for that shape
of agent.

## What's actually being demonstrated

1. **Persistent memory, zero schema changes.** Recipe-step progress,
   metrics, and licence checks live as `Fact`s in an existing SozoGraph
   `Passport`. Lessons live as dated `Observation`s. No fork of the library.
2. **Deterministic recall.** `recall.py` filters facts and observations by
   a fixed tag (`step:NN`). No embedding similarity, no vector store. The
   thing being recalled belongs to a known, closed structure, a recipe's
   own step numbers.
3. **A real agent framework judging real answers.** `llm.py` runs a Google
   ADK agent (`LlmAgent` + `Runner` + session) against Gemini 3.5. It
   genuinely evaluates whether a free-text answer satisfies a stated rule.
4. **Real pipelines behind both project cards.** A maize leaf-disease CNN
   and a MIMII-based generator-fault autoencoder. Both trained, quantized,
   and verified end to end against real data. See
   `docs/ADR_Recipe_Mentor_Pipelines_2026-08-29.md`.

## Architecture

![Recipe Mentor architecture](docs/architecture.png)

```
 mentor.py (session loop)
   |
   |-- recipe.py            12-step recipe, as data
   |-- projects/*.py        the two project cards
   |-- recall.py            deterministic cross-project tag lookup
   |-- llm.py                ADK agent judge  (--backend adk)
   |-- firestore_store.py    Firestore passport store (--store firestore)
   |
   v
 sozograph.Passport  <-->  passports/lab_demo.json  (always mirrored locally)
                      <-->  Firestore (recipe-mentor DB, when --store firestore)
   |
   v
 dashboard/render_dashboard.py --> dashboard.html   (12-step grid, metrics, lessons, contradictions)

 pipelines/
   common_quant.py      shared ONNX export + static QDQ INT8 + verification
   features.py           pure NumPy + soundfile mel-spectrogram (no librosa)
   fetch_mimii.py         bounded MIMII subset via Zenodo range requests
   maize_train.py         real CNN, Kaggle corn/maize dataset
   mimii_anomaly.py       real conv autoencoder, MIMII valve subset
   record_results.py      writes each pipeline's report.json into the passport
```

## Quickstart

No GCP account, no API key, no ML libraries. This is the whole mechanism
running deterministically.

```bash
git clone https://github.com/rapha18th/recipe-mentor.git
cd recipe-mentor
pip install sozograph

python -m recipe_mentor.mentor --project maize --show-recipe

# a full interactive session, keyword-heuristic judge
python -m recipe_mentor.mentor --project maize
# start a second, unrelated project. watch it recall the first session's lessons
python -m recipe_mentor.mentor --project diesel_generator

python -m recipe_mentor.dashboard.render_dashboard
# open recipe_mentor/dashboard.html in a browser
```

`passports/lab_demo.json` is the shared passport both sessions read and
write. Delete it to start clean. It's plain JSON. Open it directly to see
what got recorded.

## Full setup

Real ADK judge, real Firestore store, real pipelines.

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. A GCP project with Gemini 3.5+ access via Vertex AI

Use your own project. The one this was built against carries no usable
access for anyone else. Three environment variables point everything at
it. Each has a fallback, and the fallback is this build's own project.

```bash
export RECIPE_MENTOR_GCP_PROJECT=your-project-id
export RECIPE_MENTOR_GCP_LOCATION=global   # try global first; some Gemini 3.5 access is global-endpoint-only
export RECIPE_MENTOR_MODEL=gemini-3.5-flash
```

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project "$RECIPE_MENTOR_GCP_PROJECT"
gcloud services enable aiplatform.googleapis.com --project="$RECIPE_MENTOR_GCP_PROJECT"
```

If `gemini-3.5-flash` 404s on your chosen location, that's a real,
documented failure mode. See "Model name and location" below.

Kaggle credentials, for `maize_train.py`'s dataset: put a `kaggle.json`
API token at `~/.kaggle/kaggle.json` (from
[kaggle.com/settings](https://www.kaggle.com/settings) → API →
"Create New Token").

```bash
python -m recipe_mentor.mentor --project maize --backend adk
python -m recipe_mentor.mentor --project diesel_generator --backend adk
```

### 3. Firestore as the passport store (optional)

```bash
export RECIPE_MENTOR_FIRESTORE_DATABASE=recipe-mentor
gcloud services enable firestore.googleapis.com --project="$RECIPE_MENTOR_GCP_PROJECT"
gcloud firestore databases create \
  --database="$RECIPE_MENTOR_FIRESTORE_DATABASE" \
  --location=us-central1 --type=firestore-native \
  --project="$RECIPE_MENTOR_GCP_PROJECT"
```

Create a dedicated database. If the project already has other services on
it, don't reuse an existing default database blind. This build hit a real
incompatibility connecting to a pre-existing Enterprise-edition default
database: a 404, despite `gcloud` confirming it existed and IAM being
correct. A fresh Standard-edition database worked immediately. See
`docs/ADR_Recipe_Mentor_ADK_Firestore_2026-08-29.md`.

```bash
python -m recipe_mentor.mentor --project maize --backend adk --store firestore
```

The local `passports/lab_demo.json` mirror is written on every save,
regardless of `--store`. The dashboard and `record_results.py` work
unchanged either way.

### 4. The ML pipelines

```bash
python -m recipe_mentor.pipelines.fetch_mimii        # ~615MB, bounded subset via HTTP range requests
python -m recipe_mentor.pipelines.mimii_anomaly       # real training, ~5-10 min on CPU
python -m recipe_mentor.pipelines.maize_train         # downloads its own Kaggle dataset first
python -m recipe_mentor.pipelines.record_results      # writes both reports into the passport
python -m recipe_mentor.dashboard.render_dashboard
```

Both pipelines report real, unmodified numbers. The MIMII AUC is
genuinely weak (~0.42), left as is. See
`docs/ADR_Recipe_Mentor_Pipelines_2026-08-29.md` for why, and what a
stronger result would actually require: more real data, not more epochs.

## Model name and location

Use Gemini 3.5 or newer. Skip `gemini-2.5-*` entirely. On the project this
was tested against, Pro-tier access tops out at `gemini-3.1-pro`, below
the 3.5 floor. Flash is the only compliant tier available there. Check
what Model Garden shows on your own project.

`location="global"` worked for `gemini-3.5-flash` here, confirmed live.
`us-central1` returned a 404, "Publisher model ... not found." Try
`global` first on a new project too.

## Design decisions and why

`docs/PLAN.md` is the original implementation plan this was built from.
Three ADRs, written as the build progressed, cover the reasoning in detail:

- `docs/ADR_Recipe_Mentor_Vertex_Backend_2026-08-29.md`: the passport
  convention. Fact keys are underscore-delimited; Observation sources are
  colon-delimited. `sozograph`'s own key normalization forces this. Why
  `recall.py` is a tag filter, not a retrieval query. The original Vertex
  routing decision.
- `docs/ADR_Recipe_Mentor_Pipelines_2026-08-29.md`: the bounded MIMII
  fetch via HTTP range requests, the pure-NumPy feature pipeline, two live
  bugs (a Windows-specific torch ONNX-export crash, a Windows
  case-insensitive-glob bug that could have leaked images across
  train/val), and the honest weak MIMII AUC finding.
- `docs/ADR_Recipe_Mentor_ADK_Firestore_2026-08-29.md`: why ADK replaced
  LangChain outright, a compliance fix and an architectural upgrade, and
  the Firestore database issue above.

## What's real, and what's a known gap

Everything above has been run, not just written. See the ADRs for the
actual verified numbers and transcripts. Two known gaps, documented rather
than fixed:

- `maize_train.py` uses PIL for image decode and resize. The recipe's own
  "pure NumPy" rule (step 2) flags this, and did, live, in an ADK-judged
  session. Left as is. Fixing it is a real follow-up.
- The MIMII anomaly detector's real AUC (~0.42) is weak. A genuine finding
  tied to training-set size, roughly 100 examples against the thousands
  MIMII's own paper uses. Not a bug. See the pipelines ADR for the
  diagnostic that ruled out the alternatives first.
