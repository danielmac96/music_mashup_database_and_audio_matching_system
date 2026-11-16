"""Implicit pair events: POST /api/events (readme §5.14).

The frontend buffers what is done with a pair — shown, looped, skipped,
expanded, opened in Studio, rendered, exported, hidden, compared, rated — and
posts a batch every few seconds and on unload. Nothing waits for the answer.

Three rules, all load-bearing:
  * the request does no database work: the batch is cleaned in memory and
    written after the response has gone (a background task);
  * nothing here raises to the client — a malformed event is dropped, a failed
    write is logged, and the answer is still 200;
  * compaction (old rows folded into pair_event_summary) rides on the same
    background task, at most once every few hours, never in a request.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body

router = APIRouter()
log = logging.getLogger(__name__)

COMPACT_EVERY_SECS = 6 * 3600
_compact_lock = threading.Lock()
_last_compact = 0.0


def _maybe_compact() -> None:
    global _last_compact
    now = time.monotonic()
    with _compact_lock:
        if _last_compact and now - _last_compact < COMPACT_EVERY_SECS:
            return
        _last_compact = now
    from database import models
    models.compact_pair_events()


def write_events(rows: list) -> None:
    """The background half: store the batch, then fold old rows if it is time.
    Swallows everything — an event is never worth an error."""
    try:
        from database import models
        models.insert_pair_events(rows)
        _maybe_compact()
    except Exception:  # noqa: BLE001
        log.warning("could not store %d pair events", len(rows), exc_info=True)


@router.post("")
def post_events(background: BackgroundTasks, body: Any = Body(default=None)) -> dict:
    """Queue a batch: {"events": [{vocal_song_id, inst_song_id, vocal_section,
    inst_section, event, value, context, scorer_version, rank_shown}, ...]}.
    Answers how many were accepted; the write happens after the response."""
    try:
        from database import models
        events = body.get("events") if isinstance(body, dict) else None
        rows = models.clean_pair_events(events)
        offered = len(events) if isinstance(events, list) else 0
    except Exception:  # noqa: BLE001
        log.warning("could not read a pair events batch", exc_info=True)
        return {"accepted": 0, "dropped": 0}
    if rows:
        background.add_task(write_events, rows)
    return {"accepted": len(rows), "dropped": max(0, offered - len(rows))}
