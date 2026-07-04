"""Background worker: score every qualifying pair into mashup_candidates."""
from __future__ import annotations

import logging
from typing import Optional

from matcher.match import score_all_pairs

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, bpm_max_diff: Optional[float] = None,
        key_min_score: Optional[float] = None) -> None:
    jobs.update(job_id, status="running",
                message="Scoring all vocal/instrumental pairs…")
    try:
        results = score_all_pairs(bpm_max_diff=bpm_max_diff,
                                  key_min_score=key_min_score)
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
