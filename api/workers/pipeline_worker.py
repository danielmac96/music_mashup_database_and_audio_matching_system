"""Auto-chaining pipeline worker: run download → stems → analyse → structure
for a single track under ONE job, resuming from wherever the track currently is.

This is what makes the web app's "Save to library & auto-process" honest — the
user imports a playlist and every track walks the whole pipeline on its own,
instead of clicking four buttons per track in the Library tab.

Reuses the shared stage functions in api/workers/stages.py (the same code the
single-stage Library buttons call), so behaviour never diverges. Per-track
failure containment mirrors pipeline.py: a stage failure stops THIS track at its
error_* status and fails the job, but the queue keeps draining other tracks.

Two entry points:
  * ``run_stage(job_id, song_id, stage)`` — one stage, used by the per-stage
    queues in api/queue_runner.py so a track hops from the download pool to the
    stems pool to the analysis pool (downloads never wait behind Demucs).
  * ``run(job_id, song_id)`` — all stages in sequence on the calling thread
    (CLI-style; also keeps existing tests/monkeypatching valid).
"""
from __future__ import annotations

import logging
from typing import Optional

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

# Queue stages in lifecycle order. "analysis" covers do_analyze + the non-fatal
# do_structure pass (structure is not status-bearing, so it never gets its own
# resumable stage).
STAGES = ("download", "stems", "analysis")

_STAGE_MIN_RANK = {"download": 1, "stems": 2, "analysis": 3}


def _status(song_id: int) -> str:
    conn = get_conn()
    row = conn.execute("SELECT status FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    return row["status"] if row else "queued"


def next_stage(song_id: int) -> Optional[str]:
    """Which pipeline stage this track needs next, or None when fully analysed.
    Status-derived, so it is restart-safe and retry-safe (an error_* status
    re-enters at the stage that failed)."""
    rank = _RANK.get(_status(song_id), 0)
    for stage in STAGES:
        if rank < _STAGE_MIN_RANK[stage]:
            return stage
    return None


def _structure_pass(job_id: str, song_id: int) -> None:
    """Non-fatal structure detection: matching still works without sections, so
    a failure never fails the job or changes the track's 'analysed' status.

    Runs when the track has no sections AND when the sections it has predate a
    generation of feature. Asking only "are there rows?" is what made every
    bulk re-analysis a no-op for the whole Phase E / P2.1 backfill: the rows
    existed, so this gate said done, while every per-section chroma and
    tempo/grid column stayed NULL and the Settings badge went on reporting the
    library as stale. sections_are_current shares its definition with that
    badge, so the two can no longer disagree.
    """
    # Imported here rather than at module scope: bulk_worker resolves get_conn
    # per call so its db_path is not frozen at import, and this module's own
    # module-scope get_conn is exactly the binding the test suite has to reload
    # around. Keep the new query on the lazily-bound side.
    from api.workers.bulk_worker import sections_are_current

    if sections_are_current(song_id):
        return
    jobs.update(job_id, stage="structure", progress=0, message="structure: starting…")
    try:
        stages.do_structure(song_id, jobs.progress_updater(job_id, "structure"))
    except Exception as exc:  # noqa: BLE001
        # Deliberately wider than StageError: do_structure wraps only
        # detect_sections, while replace_sections, _persist_hooks and
        # warm_hooks(force=True) run after it unwrapped. Structure is
        # documented as non-fatal, and on a whole-library backfill this
        # exposure is taken once per track rather than once.
        log.warning("structure detection non-fatal failure for %s: %s", song_id, exc)
        return

    if not sections_are_current(song_id):
        # Not an error: analysis/structure.py falls back to the track BPM when a
        # section has no measurable grid. Logged because it is the difference
        # between "the backfill did not run" and "the audio would not measure",
        # and the staleness badge alone cannot tell you which.
        log.warning("structure ran for song %s but its sections are still not "
                    "current — the detector may have found nothing to measure",
                    song_id)


def _finalize(job_id: str, song_id: int) -> None:
    """Run the trailing structure pass if needed, then mark the job done."""
    _structure_pass(job_id, song_id)
    jobs.done(job_id, {"song_id": song_id, "status": _status(song_id)})


def run_stage(job_id: str, song_id: int, stage: str) -> str:
    """Run ONE stage for a track. Returns:
      * ``"next"``   — stage succeeded, the track needs another stage
      * ``"done"``   — the track is fully processed (job marked done here)
      * ``"failed"`` — stage failed (job marked failed here)
    """
    song = get_song(song_id)
    if not song:
        jobs.fail(job_id, f"Song {song_id} not found")
        return "failed"

    jobs.update(job_id, status="running", song_id=song_id, stage=stage,
                progress=0, message=f"{stage}: starting…")

    try:
        if stage == "download":
            stages.do_download(song_id, jobs.progress_updater(job_id, "download"))
        elif stage == "stems":
            stages.do_stems(song_id, jobs.progress_updater(job_id, "stems"))
        elif stage == "analysis":
            stages.do_analyze(song_id, jobs.progress_updater(job_id, "analyze"))
        else:
            jobs.fail(job_id, f"Unknown pipeline stage {stage!r}")
            return "failed"
    except stages.StageError as exc:
        jobs.fail(job_id, str(exc), exc.traceback_text)
        return "failed"

    if next_stage(song_id) is None:
        _finalize(job_id, song_id)
        return "done"
    return "next"


def run(job_id: str, song_id: int) -> None:
    """Run every remaining stage in sequence on the calling thread."""
    song = get_song(song_id)
    if not song:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    jobs.update(job_id, status="running", song_id=song_id, message="Starting…")

    while True:
        stage = next_stage(song_id)
        if stage is None:
            # Already fully analysed on entry (e.g. re-Process on a done track):
            # still give it the trailing structure pass.
            _finalize(job_id, song_id)
            return
        if run_stage(job_id, song_id, stage) != "next":
            return
