"""
Firestore-backed passport persistence -- the Google Cloud infrastructure
service this build integrates (one of Cloud Run / Cloud SQL / Firestore /
GKE / Pub/Sub, per the hackathon's own mandatory technology requirement).

A dedicated Firestore Standard-edition database (`recipe-mentor`) on the
`neofix-676da` project, created deliberately separate from that project's
pre-existing `(default)` database (which is Enterprise edition and, as
found by testing, not reachable via the current `google-cloud-firestore`
client the same way -- see docs/ADR_Recipe_Mentor_ADK_Firestore_2026-08-29.md).

A Passport is already a plain, flat JSON document
(`sozograph.schema.Passport.to_compact_dict()`), so it maps onto one
Firestore document with no schema translation. This is the real
persistence layer behind the Collaborative Partner track's "persistent
memory" claim -- the local `passports/lab_demo.json` file remains the
portable, diffable, offline-testable copy SozoGraph's own design centers
on, and stays mirrored locally every time Firestore is used, so the
dashboard and `record_results.py` keep working unchanged either way.
"""
from __future__ import annotations

import os

from sozograph.schema import Passport

#: Environment-configurable, same reasoning as llm.py -- the literal
#: fallbacks are the project/database this was built against, not a value
#: that works for anyone else. On a fresh project, create your own database
#: first: `gcloud firestore databases create --database=recipe-mentor
#: --location=<your-region> --type=firestore-native`.
PROJECT = os.environ.get("RECIPE_MENTOR_GCP_PROJECT", "neofix-676da")
DATABASE = os.environ.get("RECIPE_MENTOR_FIRESTORE_DATABASE", "recipe-mentor")
COLLECTION = "passports"


def _client():
    # Imported lazily so the local-only path never requires
    # google-cloud-firestore to be installed.
    from google.cloud import firestore

    return firestore.Client(project=PROJECT, database=DATABASE)


def load(user_key: str) -> Passport | None:
    """None if no document exists yet for this user_key -- a fresh
    passport, not an error."""
    doc = _client().collection(COLLECTION).document(user_key).get()
    if not doc.exists:
        return None
    return Passport.from_dict(doc.to_dict())


def save(passport: Passport) -> None:
    if not passport.user_key:
        raise ValueError("passport.user_key must be set to save to Firestore")
    _client().collection(COLLECTION).document(passport.user_key).set(passport.to_compact_dict())
