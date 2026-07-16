"""Background worker: extract per-stem audio features (Library "Analyze" button).

Thin job wrapper around stages.do_analyze. Structure detection (intro/verse/
chorus/…) is a separate step — see structure_worker.py — since it only needs the
full mix and producers may want to re-run it on its own."""
from __future__ import annotations

import logging

from api import jobs
from api.workers import stages

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", song_id=song_id,
                message="Analysing audio features…")

    try:
        result = stages.do_analyze(song_id, jobs.progress_updater(job_id))
    except stages.StageError as exc:
        jobs.fail(job_id, str(exc), exc.traceback_text)
        return

    jobs.done(job_id, result)
