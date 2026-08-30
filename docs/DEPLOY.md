# Deploying the hosted interface

`recipe_mentor/web/app.py` on Cloud Run. One command, from the repo root.

## One-time setup

A service account with Vertex AI and Firestore access, and the APIs it needs enabled:

```bash
export PROJECT=your-project-id

gcloud services enable run.googleapis.com aiplatform.googleapis.com \
  firestore.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  --project="$PROJECT"

gcloud iam service-accounts create recipe-mentor-web-sa \
  --display-name="Recipe Mentor web interface" --project="$PROJECT"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:recipe-mentor-web-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:recipe-mentor-web-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/datastore.user"
```

A dedicated Firestore database, same as the Socratic path's `--store firestore` (see the
README's Firestore setup section) -- the web interface always uses Firestore, since Cloud
Run's own filesystem doesn't survive between container instances:

```bash
gcloud firestore databases create \
  --database=recipe-mentor --location=us-central1 --type=firestore-native \
  --project="$PROJECT"
```

## Deploy

```bash
gcloud run deploy recipe-mentor-web \
  --source=. \
  --project="$PROJECT" \
  --region=us-central1 \
  --service-account="recipe-mentor-web-sa@${PROJECT}.iam.gserviceaccount.com" \
  --set-env-vars="RECIPE_MENTOR_GCP_PROJECT=${PROJECT},RECIPE_MENTOR_GCP_LOCATION=global,RECIPE_MENTOR_MODEL=gemini-3.5-flash,RECIPE_MENTOR_FIRESTORE_DATABASE=recipe-mentor,RECIPE_MENTOR_STORE=firestore" \
  --memory=4Gi --cpu=4 \
  --timeout=1800 \
  --concurrency=1 --max-instances=1 \
  --allow-unauthenticated
```

Builds from the `Dockerfile` via Cloud Build -- no local Docker required. Prints the
service URL when it finishes.

**Why `--concurrency=1 --max-instances=1`.** A visitor's Kaggle credentials live in this
process's environment variables for the duration of their run (see `web/app.py`'s own
docstring). Two concurrent runs on the same container would race on those variables; two
separate container instances would let a second visitor's request land on an instance that
still has the first visitor's credentials set. Both flags together guarantee there is ever
only one request in flight, on one instance, so that race can't happen. The in-process
`asyncio.Lock` in `web/app.py` is a second, redundant guard for local testing (`uvicorn
--reload`, multiple workers), where these two flags don't apply.

**Why `--allow-unauthenticated`.** This is meant to be a public demo page. Anyone who opens
it can start a run -- mitigated by the concurrency cap above, the epoch/hyperparameter
bounds in `agent/bounds.py`, and the fact that a run requires the visitor's own Kaggle
credentials, not a shared one.

**Cost.** Vertex/Gemini calls are billed to the deploying project (the Cloud Run service
account's own credential, never the visitor's). Kaggle downloads and CPU time are the other
real costs; the single-instance cap bounds how much of either can happen at once.

## A real bug this deploy caught

The first deploy failed at container startup: `ModuleNotFoundError: No module named 'onnx'`,
inside `onnxruntime.quantization`'s own import chain (`common_quant.py` imports
`onnxruntime.quantization`, which imports `onnx` directly). `onnx` was present in the local
dev environment as someone else's transitive dependency, so this never surfaced locally --
only a clean container build, with only `requirements.txt`'s own contents installed, caught
it. Fixed by adding `onnx` to `requirements.txt` explicitly. A reminder that "works on my
machine" and "works in a container built only from the lockfile" are different claims, and
only one of them is the one that matters for a real deploy.

## Local testing, no Cloud Run

```bash
uvicorn recipe_mentor.web.app:app --reload
```

Uses `RECIPE_MENTOR_STORE=local` by default (unset the env var, or leave it), so no GCP
project is required to try the interface itself -- only for Vertex/Gemini calls, which still
need the same `RECIPE_MENTOR_GCP_*` environment variables and Application Default
Credentials the CLI path already documents in the README.
