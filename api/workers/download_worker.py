"""Background worker: download a single track (Library "Download" button).

Thin job wrapper around stages.do_download so the single-stage button and the
auto-chaining pipeline_worker share identical download logic."""
from __future__ import annotations

import logging

from api import jobs
from api.workers import stages

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", song_id=song_id, message="Downloading…")

    try:
        result = stages.do_download(song_id, jobs.progress_updater(job_id))
    except stages.StageError as exc:
        jobs.fail(job_id, str(exc), exc.traceback_text)
        return

    jobs.done(job_id, result)
