# Recipe Mentor

A worked example of a **Collaborative Partner** agent, built on
[SozoGraph](https://github.com/Sozo-Analytics-Lab/sozograph)
([PyPI](https://pypi.org/project/sozograph/)): an agent that "adapts and personalizes based
on past interactions instead of starting over each time." It walks a
builder through a real twelve-step ML production recipe for two project
cards, and — the actual point — recalls what tripped the user up in one
project, unprompted, when they start a second, unrelated one.

Built for the "All Things Agentic" hackathon's Collaborative Partner track,
but the pattern generalizes: **any structured, repeatable workflow** (a
recipe, a runbook, a checklist someone follows more than once) can sit on
top of the same three pieces — a portable memory passport, a deterministic
tag-based recall layer, and a real LLM judge — and get the same
cross-session personalization for free. Fork this if you want a template
for that shape of agent, not just this specific demo.

## What's actually being demonstrated

1. **Persistent memory with zero schema changes.** Recipe-step progress,
   metrics, and licence checks live as `Fact`s in an existing SozoGraph
   `Passport`; lessons live as dated `Observation`s. No fork of the library.
2. **Deterministic recall, not vector search.** `recall.py` filters facts
   and observations by a fixed tag (`step:NN`) rather than embedding
   similarity — appropriate when the thing being recalled belongs to a
   known, closed structure (a recipe's own step numbers), not free text.
3. **A real agent framework judging real answers.** `llm.py` runs a
   Google ADK agent (`LlmAgent` + `Runner` + session) against Gemini 3.5,
   not a scripted response — it genuinely evaluates whether a free-text
   answer satisfies a stated rule.
4. **Real pipelines behind the project cards**, not placeholders: a maize
   leaf-disease CNN and a MIMII-based generator-fault autoencoder, both
   trained, quantized, and verified end to end against real data. See
   `docs/ADR_Recipe_Mentor_Pipelines_2026-08-29.md`.

## Architecture

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

## Quickstart (offline, zero cloud dependencies)

Nothing below needs a GCP account, an API key, or any of the ML libraries.
This is the whole mechanism running deterministically.

```bash
git clone https://github.com/rapha18th/recipe-mentor.git
cd recipe-mentor
pip install sozograph

python -m recipe_mentor.mentor --project maize --show-recipe

# a full interactive session, keyword-heuristic judge
python -m recipe_mentor.mentor --project maize
# start a second, unrelated project -- watch it recall the first session's lessons
python -m recipe_mentor.mentor --project diesel_generator

python -m recipe_mentor.dashboard.render_dashboard
# open recipe_mentor/dashboard.html in a browser
```

`passports/lab_demo.json` is the shared passport both sessions read and
write — delete it to start clean. It's plain JSON; open it directly to see
what got recorded.

## Full setup (real ADK judge, real Firestore store, real pipelines)

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. A GCP project with Gemini 3.5+ access via Vertex AI

You need your own project — the one this was built against carries no
usable access for anyone else. Point everything at it with three
environment variables (all optional; each has a fallback, but the
fallback is this build's own project and won't work for you):

```bash
export RECIPE_MENTOR_GCP_PROJECT=your-project-id
export RECIPE_MENTOR_GCP_LOCATION=global   # try "global" first -- some Gemini 3.5 access is global-endpoint-only
export RECIPE_MENTOR_MODEL=gemini-3.5-flash
```

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project "$RECIPE_MENTOR_GCP_PROJECT"
gcloud services enable aiplatform.googleapis.com --project="$RECIPE_MENTOR_GCP_PROJECT"
```

If `gemini-3.5-flash` 404s on your chosen location, that's a real,
documented failure mode, not a setup mistake — see "Model name and
location" below.

Kaggle credentials, for `maize_train.py`'s dataset: put a `kaggle.json`
API token at `~/.kaggle/kaggle.json` (from
[kaggle.com/settings](https://www.kaggle.com/settings) → API →
"Create New Token").

```bash
python -m recipe_mentor.mentor --project maize --backend adk
python -m recipe_mentor.mentor --project diesel_generator --backend adk
```

### 3. (Optional) Firestore as the passport store

```bash
export RECIPE_MENTOR_FIRESTORE_DATABASE=recipe-mentor
gcloud services enable firestore.googleapis.com --project="$RECIPE_MENTOR_GCP_PROJECT"
gcloud firestore databases create \
  --database="$RECIPE_MENTOR_FIRESTORE_DATABASE" \
  --location=us-central1 --type=firestore-native \
  --project="$RECIPE_MENTOR_GCP_PROJECT"
```

Create a dedicated database rather than reusing an existing one if the
project already has other services on it — this build hit a real,
unresolved incompatibility connecting to a pre-existing *Enterprise*-edition
default database (404, despite `gcloud` confirming it existed and IAM being
correct); a fresh Standard-edition database worked immediately. See
`docs/ADR_Recipe_Mentor_ADK_Firestore_2026-08-29.md`.

```bash
python -m recipe_mentor.mentor --project maize --backend adk --store firestore
```

The local `passports/lab_demo.json` mirror is written on every save
regardless of `--store`, so the dashboard and `record_results.py` work
unchanged either way.

### 4. The ML pipelines

```bash
python -m recipe_mentor.pipelines.fetch_mimii        # ~615MB, bounded subset via HTTP range requests
python -m recipe_mentor.pipelines.mimii_anomaly       # real training, ~5-10 min on CPU
python -m recipe_mentor.pipelines.maize_train         # downloads its own Kaggle dataset first
python -m recipe_mentor.pipelines.record_results      # writes both reports into the passport
python -m recipe_mentor.dashboard.render_dashboard
```

Both pipelines report real, unmodified numbers — including a genuinely
weak MIMII AUC (~0.42), left as-is rather than tuned to look better. See
`docs/ADR_Recipe_Mentor_Pipelines_2026-08-29.md` for why, and what a
stronger result would actually require (more real data, not more epochs).

## Model name and location

**Use Gemini 3.5 or newer. Do not fall back to `gemini-2.5-*`.** As of this
build, on the project it was tested against, Pro-tier access tops out at
`gemini-3.1-pro` (below the 3.5 floor) — Flash is the only compliant,
available tier. Your project's access may differ; check what Model Garden
shows you before assuming.

**`location="global"`, not a region, worked for `gemini-3.5-flash` here** —
confirmed live (`us-central1` returned a 404 "Publisher model ... not
found"). Try `global` first on a new project too.

## Design decisions and why

`docs/PLAN.md` is the original implementation plan this was built from.
Four ADRs, written as the build progressed, cover the reasoning in detail:

- `docs/ADR_Recipe_Mentor_Vertex_Backend_2026-08-29.md` — the passport
  convention (why Fact keys are underscore-delimited but Observation
  sources are colon-delimited — not arbitrary, `sozograph`'s own key
  normalization forces it), why `recall.py` is a tag filter and not a
  retrieval query, and the original Vertex routing decision.
- `docs/ADR_Recipe_Mentor_Pipelines_2026-08-29.md` — the bounded MIMII
  fetch via HTTP range requests, the pure-NumPy feature pipeline, two live
  bugs (a Windows-specific torch ONNX-export crash, a Windows
  case-insensitive-glob bug that could have leaked images across
  train/val), and the honest sub-chance-to-weak MIMII AUC finding.
- `docs/ADR_Recipe_Mentor_ADK_Firestore_2026-08-29.md` — why ADK replaced
  LangChain outright (a compliance fix, not just a preference), and the
  Firestore database issue above.

## What's real vs. what's a known gap

Everything above has been run, not just written — see the ADRs for the
actual verified numbers and transcripts. Two known, deliberately
undisclosed-nowhere gaps, left as documented rather than fixed:

- `maize_train.py` uses PIL for image decode/resize. A strict reading of
  the recipe's own "pure NumPy" rule (step 2) flags this — and did, live,
  when an ADK-judged session ran it. Left as-is; fixing it is a real
  follow-up, not done here.
- The MIMII anomaly detector's real AUC (~0.42) is weak. This is
  documented as a genuine finding tied to training-set size (~100 examples
  vs. the thousands MIMII's own paper uses), not a bug — see the pipelines
  ADR for the diagnostic that ruled out the alternative explanations first.
