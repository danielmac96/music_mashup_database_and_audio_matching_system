"""Background worker: export a mashup (or a shortlist of them) as FL session
folders. Mirrors api/workers/mixdown_worker.py."""
from __future__ import annotations

import logging

from render.session import build_session, build_session_batch

from api import jobs

log = logging.getLogger(__name__)


def _progress_sink(job_id: str, last_msg: dict):
    def _on_progress(pct, msg: str) -> None:
        last_msg["text"] = msg
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)
    return _on_progress


def run(job_id: str, vocal_song_id: int, inst_song_id: int) -> None:
    jobs.update(job_id, status="running", message="Exporting FL session…")
    last_msg = {"text": ""}
    try:
        out = build_session(job_id, vocal_song_id, inst_song_id,
                            on_progress=_progress_sink(job_id, last_msg))
    except Exception as exc:  # noqa: BLE001
        log.exception("build_session raised")
        jobs.fail(job_id, f"Session export error: {type(exc).__name__}: {exc}")
        return

    if out is None:
        jobs.fail(job_id, last_msg["text"] or "Session export failed")
        return

    _record_implicit_positive(vocal_song_id, inst_song_id)
    jobs.done(job_id, {
        "folder": str(out),
        "archive_url": f"/api/studio/session/{job_id}/archive",
    })


def _record_implicit_positive(vocal_song_id: int, inst_song_id: int,
                              db_path=None) -> None:
    """Exporting a pair to FL is the strongest signal the app ever gets.

    A ✓ in Discover means "worth a listen"; carrying a pair all the way to a
    session folder means the user intends to BUILD it. That was being thrown
    away. Recorded as 'ok' rather than 'love' — deciding to build something is
    not the same as loving the result — and never over an explicit verdict the
    user already gave, including a rejection.

    `db_path` is resolved at CALL time rather than taken from get_conn's default
    argument, which binds at import: without that this function could only ever
    talk to the process-wide database, which is also why it had no test.
    """
    try:
        from database.models import DB_PATH, get_conn, upsert_pair_feedback
        db = db_path or DB_PATH
        conn = get_conn(db)
        existing = conn.execute(
            "SELECT verdict FROM pair_feedback WHERE vocal_song_id=? "
            "AND inst_song_id=?", (vocal_song_id, inst_song_id)).fetchone()
        conn.close()
        if existing:
            return
        upsert_pair_feedback(vocal_song_id, inst_song_id, "ok", db_path=db)
    except Exception:  # noqa: BLE001 — never fail an export over a label
        log.warning("could not record implicit positive", exc_info=True)


def run_batch(job_id: str, pairs: list[dict]) -> None:
    jobs.update(job_id, status="running",
                message=f"Exporting {len(pairs)} FL sessions…")
    last_msg = {"text": ""}
    try:
        # Same implicit label as the single export — and this is the path that
        # actually gets used, since "Export top N" is one click against a
        # filtered list. Recording only the single-pair path meant the batch
        # button, the one a triage session ends with, trained nothing.
        out = build_session_batch(job_id, pairs,
                                  on_progress=_progress_sink(job_id, last_msg),
                                  on_exported=_record_implicit_positive)
    except Exception as exc:  # noqa: BLE001
        log.exception("build_session_batch raised")
        jobs.fail(job_id, f"Session export error: {type(exc).__name__}: {exc}")
        return

    if out is None:
        jobs.fail(job_id, last_msg["text"] or "Session export failed")
        return

    jobs.done(job_id, {
        "folder": str(out),
        "pair_count": len(pairs),
        "archive_url": f"/api/studio/session/{job_id}/archive",
    })
