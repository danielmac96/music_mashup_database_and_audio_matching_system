"""Auto-chaining pipeline worker: run download → stems → analyse → structure
for a single track under ONE job, resuming from wherever the track currently is.

This is what makes the web app's "Save to library & auto-process" honest — the
user imports a playlist and every track walks the whole pipeline on its own,
instead of clicking four buttons per track in the Library tab.

Reuses the shared stage functions in api/workers/stages.py (the same code the
single-stage Library buttons call), so behaviour never diverges. Per-track
failure containment mirrors pipeline.py: a stage failure stops THIS track at its
error_* status and fails the job, but the queue keeps draining other tracks.
"""
from __future__ import annotations

import logging

from database.models import get_conn, get_song

from api import jobs
from api.workers import stages

log = logging.getLogger(__name__)

# How far along the lifecycle each status is. A stage runs only when the track
# has not yet reached the status that stage would produce.
_RANK = {
    "queued": 0, "error_download": 0,
    "downloaded": 1, "error_stems": 1,
    "stemmed": 2, "error_analysis": 2,
    "analysed": 3,
}


def _status(song_id: int) -> str:
    conn = get_conn()
    row = conn.execute("SELECT status FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    return row["status"] if row else "queued"


def _section_count(song_id: int) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM sections WHERE song_id=?", (song_id,)
    ).fetchone()
    conn.close()
    return row["n"] if row else 0


def run(job_id: str, song_id: int) -> None:
    song = get_song(song_id)
    if not song:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    jobs.update(job_id, status="running", song_id=song_id, message="Starting…")

    # DOWNLOAD
    if _RANK.get(_status(song_id), 0) < 1:
        jobs.update(job_id, stage="download", progress=0, message="download: starting…")
        try:
            stages.do_download(song_id, _cb(job_id, "download"))
        except stages.StageError as exc:
            jobs.fail(job_id, str(exc), exc.traceback_text)
            return

    # STEMS
    if _RANK.get(_status(song_id), 0) < 2:
        jobs.update(job_id, stage="stems", progress=0, message="stems: starting…")
        try:
            stages.do_stems(song_id, _cb(job_id, "stems"))
        except stages.StageError as exc:
            jobs.fail(job_id, str(exc), exc.traceback_text)
            return

    # ANALYSE
    if _RANK.get(_status(song_id), 0) < 3:
        jobs.update(job_id, stage="analyze", progress=0, message="analyze: starting…")
        try:
            stages.do_analyze(song_id, _cb(job_id, "analyze"))
        except stages.StageError as exc:
            jobs.fail(job_id, str(exc), exc.traceback_text)
            return

    # STRUCTURE — non-fatal: matching still works without sections, so a
    # structure failure leaves the track fully 'analysed' and the job succeeds.
    if _section_count(song_id) == 0:
        jobs.update(job_id, stage="structure", progress=0, message="structure: starting…")
        try:
            stages.do_structure(song_id, _cb(job_id, "structure"))
        except stages.StageError as exc:
            log.warning("structure detection non-fatal failure for %s: %s", song_id, exc)

    jobs.done(job_id, {"song_id": song_id, "status": _status(song_id)})
