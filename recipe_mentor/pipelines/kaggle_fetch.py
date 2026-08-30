"""
Kaggle dataset fetch tool for the autonomous agent.

Downloads a whole Kaggle dataset by owner/slug, using the `kaggle` package's
own KaggleApi -- the same credential convention (~/.kaggle/kaggle.json, or
the KAGGLE_USERNAME/KAGGLE_KEY environment variables) the README already
documents for the Socratic path. No credential handling code here:
api.authenticate() raises the library's own clear error if neither is
present. `kaggle` has been a listed dependency since the start of this repo
but nothing actually called it until now.

Whole-dataset download, not a bounded partial fetch like fetch_mimii.py's
Zenodo range-request trick: Kaggle's API doesn't support partial fetch, and
an autonomous agent that only knows a dataset's owner/slug (not its internal
zip layout, inspected ahead of time the way fetch_mimii.py's was) can't
pre-plan a bounded subset the way that module does for one specific,
already-known dataset.

Usage:
    python -m recipe_mentor.pipelines.kaggle_fetch owner/slug
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data" / "agent_runs"


def _dest_dir(owner_slug: str) -> Path:
    safe = owner_slug.replace("/", "__")
    return DATA_ROOT / safe


def fetch_dataset(owner_slug: str, *, force: bool = False) -> dict:
    """
    Download and unzip a Kaggle dataset by 'owner/slug' into
    data/agent_runs/{owner}__{slug}/. Returns a small summary:
    {"path", "n_files", "total_bytes", "already_cached"}.

    Reuses what's already on disk unless `force=True` -- a re-run against
    the same dataset shouldn't re-download it, and the demo shouldn't pay
    for a second download of a dataset already fetched during development.
    """
    if "/" not in owner_slug:
        raise ValueError(f"expected 'owner/slug', got {owner_slug!r}")

    dest = _dest_dir(owner_slug)
    already_cached = dest.exists() and any(dest.iterdir())
    if not already_cached or force:
        dest.mkdir(parents=True, exist_ok=True)
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(owner_slug, path=str(dest), unzip=True, quiet=False)

    files = [p for p in dest.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    return {
        "path": str(dest),
        "n_files": len(files),
        "total_bytes": total_bytes,
        "already_cached": already_cached and not force,
    }


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python -m recipe_mentor.pipelines.kaggle_fetch owner/slug")
    result = fetch_dataset(argv[0])
    print(
        f"path={result['path']} n_files={result['n_files']} "
        f"total_mb={result['total_bytes'] / 1e6:.1f} cached={result['already_cached']}"
    )


if __name__ == "__main__":
    main()
