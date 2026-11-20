"""Partner-aware stem priority (readme §2, §5.7): separate first the imported
tracks whose first USEFUL pair is nearest.

A track's quick tier ends with provisional pairs — its mix standing in for
both stems against the whole library. Among them, the rows whose partner is
already separated (status 'stemmed' or 'analysed') and whose two songs already
satisfy the Drop-in rule's song half (compat.dropin_songs_sql: tempi within
3 %, keys that mix at the pair's transpose) are pairs that can become a
Drop-in the moment THIS track's stems and first cut exist — nothing else has
to be separated first. The more of those, the likelier one survives the slot
and the timing gate, so the more urgent the separation.

Applied once, when an import's job is routed to the stems queue
(queue_runner._dispatch): only a job still at PRIORITY_INGEST is promoted
(a button, a selection or a backfill keeps its own priority), and only ever
upward, into the band between PRIORITY_PREFETCH and PRIORITY_INGEST — a
selected track and its partners still go first. Within one band the order is
still arrival. Never raises: a priority is never worth a failed track.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def stem_promise(song_id: int, db_path=None) -> dict:
    """{"dropin_partners", "partners", "best", "role"}: how many distinct
    already separated partners this track's provisional rows pair with on the
    Drop-in rule's song half, how many separated partners it has provisional
    rows with at all, the best of those likely Drop-ins' provisional totals,
    and the role the upload declares in its title or tags
    (ingest/fit_hints.parse).

    An upload that calls itself an INSTRUMENTAL promises no Drop-in whatever
    its rows say: it has no vocal to drop in, and as a bed none of its
    sections is a vacated vocal slot (the slot term, §5.7) — so its
    dropin_partners is 0. Before the stems nothing else can tell: the quick
    tier's sections carry no vocal presence. Reads stored rows only — one
    indexed query per track."""
    from database.models import get_conn
    from ingest.fit_hints import parse
    from matcher.compat import dropin_songs_sql
    from matcher.match import PROVISIONAL_SCORER
    from matcher.recipe import harmony_unsure_below

    unsure = harmony_unsure_below(db_path)
    likely = dropin_songs_sql("mc")
    conn = get_conn(db_path) if db_path is not None else get_conn()
    try:
        song = conn.execute("SELECT title, tags FROM songs WHERE id=?",
                            (song_id,)).fetchone()
        role = parse({"title": song[0], "tags": song[1]})["role"] if song else None
        row = conn.execute(
            f"""SELECT COUNT(DISTINCT CASE WHEN {likely} THEN p.id END),
                       COUNT(DISTINCT p.id),
                       MAX(CASE WHEN {likely} THEN mc.score_total END)
                FROM mashup_candidates mc
                JOIN songs p ON p.id = CASE WHEN mc.vocal_song_id = ?
                                            THEN mc.inst_song_id
                                            ELSE mc.vocal_song_id END
                WHERE mc.combo_type = 'vocal_over_instrumental'
                  AND mc.scorer = ?
                  AND (mc.vocal_song_id = ? OR mc.inst_song_id = ?)
                  AND p.status IN ('stemmed', 'analysed')""",
            (unsure, unsure, song_id, PROVISIONAL_SCORER, song_id, song_id)
        ).fetchone()
    finally:
        conn.close()
    instrumental = role == "instrumental"
    return {"dropin_partners": 0 if instrumental else int(row[0] or 0),
            "partners": int(row[1] or 0),
            "best": None if instrumental or row[2] is None else round(float(row[2]), 4),
            "role": role}


def priority_for(promise: dict) -> int:
    """The stems-queue priority a promise earns (config.PRIORITY_* ladder):
    at least STEM_PROMISE_MANY likely Drop-in partners → PRIORITY_PROMISE_HIGH,
    at least STEM_PROMISE_SOME → PRIORITY_PROMISE_MID, at least one →
    PRIORITY_PROMISE_LOW, none → PRIORITY_INGEST (arrival order, as before)."""
    from config import (PRIORITY_INGEST, PRIORITY_PROMISE_HIGH,
                        PRIORITY_PROMISE_LOW, PRIORITY_PROMISE_MID,
                        STEM_PROMISE_MANY, STEM_PROMISE_SOME)
    n = int((promise or {}).get("dropin_partners") or 0)
    if n >= STEM_PROMISE_MANY:
        return PRIORITY_PROMISE_HIGH
    if n >= STEM_PROMISE_SOME:
        return PRIORITY_PROMISE_MID
    if n >= 1:
        return PRIORITY_PROMISE_LOW
    return PRIORITY_INGEST


def promote_for_stems(job_id: str, song_id: int) -> Optional[int]:
    """Raise an import's pipeline job before it enters the stems queue, by
    its promise. Returns the priority it now has, or None when untouched.
    The promise is recorded on the job (`stem_promise`) for the Queue."""
    try:
        from config import PRIORITY_INGEST
        from api import jobs
        job = jobs.get(job_id) or {}
        if int(job.get("priority", PRIORITY_INGEST)) != PRIORITY_INGEST:
            return None
        promise = stem_promise(song_id)
        prio = priority_for(promise)
        jobs.update(job_id, stem_promise=promise)
        if prio < PRIORITY_INGEST:
            jobs.update(job_id, priority=int(prio))
            return prio
        return None
    except Exception:  # noqa: BLE001
        log.debug("stem promise not computed for song %s", song_id, exc_info=True)
        return None


def declares_instrumental(song_id: int, db_path=None) -> bool:
    """Whether the upload calls itself an instrumental in its title or tags
    (ingest/fit_hints.parse — what Discover reads too). Never raises."""
    try:
        from database.models import get_conn
        from ingest.fit_hints import parse
        conn = get_conn(db_path) if db_path is not None else get_conn()
        try:
            row = conn.execute("SELECT title, tags FROM songs WHERE id=?",
                               (song_id,)).fetchone()
        finally:
            conn.close()
        return bool(row) and parse({"title": row[0], "tags": row[1]})["role"] == "instrumental"
    except Exception:  # noqa: BLE001
        return False


def import_priority(song_id: int, priority: int) -> int:
    """The priority an import is queued at: PRIORITY_INGEST, or
    PRIORITY_INGEST_INSTRUMENTAL for an upload that calls itself an
    instrumental — it cannot give a Drop-in (stem_promise), so its batch's
    other tracks reach the cores first. Any other priority is returned
    unchanged."""
    from config import PRIORITY_INGEST, PRIORITY_INGEST_INSTRUMENTAL
    if int(priority) == PRIORITY_INGEST and declares_instrumental(song_id):
        return PRIORITY_INGEST_INSTRUMENTAL
    return int(priority)
