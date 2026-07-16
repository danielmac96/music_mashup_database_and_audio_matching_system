"""Background worker: separate vocals / instrumental (Library "Separate" button).

Thin job wrapper around stages.do_stems."""
from __future__ import annotations

import logging

from api import jobs
from api.workers import stages

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", song_id=song_id,
                message="Separating stems (this can take a while)…")

    try:
        result = stages.do_stems(song_id, jobs.progress_updater(job_id))
    except stages.StageError as exc:
        jobs.fail(job_id, str(exc), exc.traceback_text)
        return

    jobs.done(job_id, result)
