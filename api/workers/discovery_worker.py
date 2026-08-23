"""Background worker for Discovery suggestions.

A suggestion run is one request per seed plus a bounded artist/set tail, and
``soundcloud_browse`` deliberately spaces requests out to protect the shared
client_id. Twenty-five seeds is therefore ~20 seconds of mostly waiting — an
obvious job, not a request.

This is also where the library filter is applied: ``ingest/`` does not import
``database`` (only ``matcher/`` does), so the engine takes the "what do I already
own" question as an injected callable and this module answers it.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from api import jobs
from database.models import songs_by_identity
from ingest.soundcloud_api import SoundCloudAPIError
from ingest.soundcloud_recommend import recommend

log = logging.getLogger(__name__)


def _owned_keys(rows: List[Dict]) -> set:
    """Which candidates the library already has, as identity keys.

    One query for the whole candidate set. Matches source_url first and track_id
    second — the same order ``api/routes/discovery.py::_annotate`` uses, so a row
    the browser badges "in library" is exactly a row suggestions will not offer.
    Both keys of a hit are returned because the engine checks either."""
    if not rows:
        return set()
    found = songs_by_identity(
        source_urls=[r.get("source_url") or "" for r in rows],
        track_ids=[str(r.get("track_id") or "") for r in rows])

    keys = set()
    for row in rows:
        url = row.get("source_url") or ""
        tid = str(row.get("track_id") or "")
        if found["by_url"].get(url) or (tid and found["by_track_id"].get(tid)):
            keys.update(k for k in (url, tid) if k)
    return keys


def suggest(job_id: str, seeds: Sequence[Dict], kinds: Sequence[str]) -> None:
    jobs.update(job_id, status="running", message="Asking SoundCloud…")
    try:
        result = recommend(seeds, kinds=kinds, owned=_owned_keys,
                           on_progress=jobs.progress_updater(job_id))
    except (ValueError, SoundCloudAPIError) as exc:
        # Both are explainable in one line: nothing seedable, or SoundCloud
        # declining. Neither deserves a traceback in the UI.
        jobs.fail(job_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("suggestion run raised")
        jobs.fail(job_id, f"Suggestion error: {type(exc).__name__}: {exc}")
        return

    jobs.done(job_id, {
        **result,
        "summary": (f"{len(result['tracks'])} tracks, "
                    f"{len(result['artists'])} artists and "
                    f"{len(result['playlists'])} sets "
                    f"from {result['seeds_used']} seed"
                    f"{'' if result['seeds_used'] == 1 else 's'}"),
    })
