"""Progressive metadata hydration for playlist previews.

The playlist preview endpoint enumerates a playlist *flat* (fast, one request,
shows every row including geo-blocked ones) — but flat rows are metadata-sparse.
This module back-fills them: `start()` snapshots the flat rows into an
in-memory session and enriches each one on a small thread pool; the frontend
polls `GET /api/playlists/preview/{id}` (routes to `get()`) every ~1.5s and
merges the progressively hydrated rows into the table. Row order and count
never change during a session, so the UI's index-keyed selection survives.

A shared URL→metadata cache (`cache_get`/`cache_put`) means a track hydrated
during preview is not re-fetched at ingest time, and single-track previews
seed the cache too.
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Optional

from config import ENRICH_WORKERS

log = logging.getLogger(__name__)

# Sessions live in memory (same lifetime story as api.jobs). Expired sessions
# are swept lazily on access; the frontend treats a 404 as "stop polling".
_SESSION_TTL = timedelta(minutes=30)
_MAX_SESSIONS = 20

_SESSIONS: dict[str, dict] = {}
_LOCK = Lock()

# URL → enriched metadata dict. Bounded so a long-running server doesn't grow
# without limit; eviction is oldest-inserted-first (dict preserves order).
_CACHE_MAX = 2000
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = Lock()

_POOL = ThreadPoolExecutor(max_workers=ENRICH_WORKERS,
                           thread_name_prefix="hydrate")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def cache_put(source_url: str, track: dict) -> None:
    if not source_url:
        return
    with _CACHE_LOCK:
        _CACHE.pop(source_url, None)
        _CACHE[source_url] = dict(track)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))


def cache_get(source_url: str) -> Optional[dict]:
    with _CACHE_LOCK:
        hit = _CACHE.get(source_url)
        return dict(hit) if hit else None


def _sweep_locked() -> None:
    """Drop expired/overflow sessions (caller holds _LOCK)."""
    cutoff = _now() - _SESSION_TTL
    dead = [sid for sid, s in _SESSIONS.items() if s["_created"] < cutoff]
    for sid in dead:
        _SESSIONS.pop(sid, None)
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.pop(next(iter(_SESSIONS)))


def start(tracks: list[dict]) -> str:
    """Begin hydrating a list of flat playlist rows. Returns the session id."""
    session_id = uuid.uuid4().hex
    rows = [dict(t) for t in tracks]
    with _LOCK:
        _sweep_locked()
        _SESSIONS[session_id] = {
            "_created": _now(),
            "tracks": rows,
            "count": len(rows),
            "hydrated_count": sum(1 for t in rows if t.get("hydrated")),
            "done": False,
        }
    for idx, row in enumerate(rows):
        if not row.get("hydrated"):
            _POOL.submit(_hydrate_one, session_id, idx)
    _maybe_finish(session_id)
    return session_id


def get(session_id: Optional[str]) -> Optional[dict]:
    """Session snapshot for the poll endpoint: {tracks, count, hydrated_count,
    done}. None for unknown/expired ids."""
    if not session_id:
        return None
    with _LOCK:
        _sweep_locked()
        s = _SESSIONS.get(session_id)
        if s is None:
            return None
        return {
            "tracks": [dict(t) for t in s["tracks"]],
            "count": s["count"],
            "hydrated_count": s["hydrated_count"],
            "done": s["done"],
        }


def _maybe_finish(session_id: str) -> None:
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s and s["hydrated_count"] >= s["count"]:
            s["done"] = True


def _hydrate_one(session_id: str, idx: int) -> None:
    """Enrich one flat row in place. Failures leave the flat row (marked
    hydrated so the session can complete) — ingest falls back to a live fetch."""
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is None:
            return
        row = dict(s["tracks"][idx])

    source_url = (row.get("source_url") or "").strip()
    rich: Optional[dict] = None
    if source_url:
        rich = cache_get(source_url)
        if rich is None:
            try:
                from ingest.soundcloud import enrich_track  # lazy: needs yt-dlp
                rich = enrich_track(source_url)
            except Exception:  # noqa: BLE001 — one bad track must not stop the sweep
                log.warning("hydration fetch failed for %s", source_url,
                            exc_info=True)
                rich = None
            if rich:
                cache_put(source_url, rich)

    merged = {**row, **(rich or {}), "hydrated": True}
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is None:
            return
        s["tracks"][idx] = merged
        s["hydrated_count"] += 1
        if s["hydrated_count"] >= s["count"]:
            s["done"] = True
