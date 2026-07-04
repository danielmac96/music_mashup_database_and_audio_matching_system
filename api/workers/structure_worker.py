"""Background worker: detect song structure (Library "Structure" button).

Thin job wrapper around stages.do_structure. Split out from analysis_worker so
structure detection is its own trackable step — it only needs the full mix
(+ optional vocal stem), not a full re-analysis of every stem."""
from __future__ import annotations

import logging

from api import jobs
from api.workers import stages

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", song_id=song_id,
                message="Detecting song structure (chorus/verse/drop)…")

    def _on_progress(pct, msg: str) -> None:
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)

    try:
        result = stages.do_structure(song_id, _on_progress)
    except stages.StageError as exc:
        jobs.fail(job_id, str(exc), exc.traceback_text)
        return

    jobs.done(job_id, result)
