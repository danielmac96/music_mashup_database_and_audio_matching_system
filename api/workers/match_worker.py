"""Background worker: score every qualifying pair into mashup_candidates."""
from __future__ import annotations

import logging
from typing import Optional

from matcher.match import score_all_pairs

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, bpm_max_diff: Optional[float] = None,
        key_min_score: Optional[float] = None, scorer: str = "auto") -> None:
    jobs.update(job_id, status="running", progress=0,
                message="Scoring all vocal/instrumental pairs…")
    # A 900-song library is minutes of work, so report block-by-block rather
    # than leaving the badge at 0% until it finishes.
    on_progress = jobs.progress_updater(job_id)
    try:
        results = score_all_pairs(bpm_max_diff=bpm_max_diff,
                                  key_min_score=key_min_score, scorer=scorer,
                                  progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        log.exception("score_all_pairs raised")
        jobs.fail(job_id, f"Matching error: {type(exc).__name__}: {exc}")
        return

    vi = len(results.get("vocal_over_instrumental", []))
    ii = len(results.get("instrumental_over_instrumental", []))
    jobs.done(job_id, {
        "vocal_over_instrumental": vi,
        "instrumental_over_instrumental": ii,
        "scorer": results.get("_scorer", "heuristic"),
        "model_version": results.get("_model_version"),
    })
