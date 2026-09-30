"""Fetch the Essentia models the genre/tag analysis uses (readme §9, phase 5)
into <data_dir>/essentia_models. The analysis fetches them on first use anyway;
this is for doing it up front. Run in the container:
    docker compose exec app python scripts/fetch_essentia_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import ml_models  # noqa: E402

if __name__ == "__main__":
    ok = ml_models.ensure_models()
    s = ml_models.status()
    print(f"{'ok' if ok else 'FAILED'}: {s['dir']}" + (f" — {s['error']}" if s["error"] else ""))
    sys.exit(0 if ok else 1)
