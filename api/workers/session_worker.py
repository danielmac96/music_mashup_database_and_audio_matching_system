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

    jobs.done(job_id, {
        "folder": str(out),
        "archive_url": f"/api/studio/session/{job_id}/archive",
    })


def run_batch(job_id: str, pairs: list[dict]) -> None:
    jobs.update(job_id, status="running",
                message=f"Exporting {len(pairs)} FL sessions…")
    last_msg = {"text": ""}
    try:
        out = build_session_batch(job_id, pairs,
                                  on_progress=_progress_sink(job_id, last_msg))
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
