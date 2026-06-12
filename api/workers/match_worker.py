"""Background worker: score every qualifying pair into mashup_candidates."""
from __future__ import annotations

import logging

from matcher.match import score_all_pairs

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str) -> None:
    jobs.update(job_id, status="running",
                message="Scoring all vocal/instrumental pairs…")
    try:
        results = score_all_pairs()
    except Exception as exc:  # noqa: BLE001
        log.exception("score_all_pairs raised")
        jobs.fail(job_id, f"Matching error: {type(exc).__name__}: {exc}")
        return

    vi = len(results.get("vocal_over_instrumental", []))
    ii = len(results.get("instrumental_over_instrumental", []))
    jobs.done(job_id, {
        "vocal_over_instrumental": vi,
        "instrumental_over_instrumental": ii,
    })
