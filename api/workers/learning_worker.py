"""Background jobs for the learning loop (matcher/learning.py, readme §5.8):
a retrain with its shadow evaluation, and a change of the active model (which
re-blends every stored vocal-over-bed row). Both are jobs because both scale
with the library — re-blending 480,000 rows takes minutes at 10,000 tracks."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from api import jobs

log = logging.getLogger(__name__)


def retrain(job_id: str, reason: str = "manual") -> None:
    jobs.update(job_id, status="running", progress=0, message="Training on your verdicts…")
    try:
        from matcher import learning
        out = learning.train(reason=reason,
                             progress=lambda pct, msg: jobs.update(
                                 job_id, progress=pct, message=msg))
    except Exception as exc:  # noqa: BLE001
        log.exception("retrain failed")
        jobs.fail(job_id, f"Training error: {type(exc).__name__}: {exc}")
        return
    summary = out.get("decision") or out.get("reason") or "done"
    jobs.done(job_id, {**{k: v for k, v in out.items() if k != "metrics"},
                       "metrics": out.get("metrics"), "summary": summary})


def switch(job_id: str, model_id: Optional[int], rollback: bool = False) -> None:
    jobs.update(job_id, status="running", progress=10,
                message="Re-ranking the stored pairs…")
    try:
        from matcher import learning
        out = learning.rollback() if rollback else learning.activate_model(model_id)
    except ValueError as exc:
        jobs.fail(job_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("model switch failed")
        jobs.fail(job_id, f"Error: {type(exc).__name__}: {exc}")
        return
    jobs.done(job_id, {**(out or {}), "summary": (
        f"{out.get('version') or 'the rules alone'} — {out.get('rows', 0)} rows re-ranked")})


def start_retrain(reason: str) -> str:
    job_id = jobs.new_job(kind="learn", message=f"Queued retrain ({reason})")
    threading.Thread(target=retrain, args=(job_id, reason), daemon=True,
                     name="learning-retrain").start()
    return job_id


def start_ticker(interval_secs: float = 60.0) -> threading.Thread:
    """Check once a minute whether a retrain is due (learning.tick: debounced,
    and only while the pipeline queues are idle). Daemon: dies with the app."""
    stop = threading.Event()

    def loop():
        from matcher import learning
        while not stop.wait(interval_secs):
            learning.tick(run=start_retrain)
    t = threading.Thread(target=loop, daemon=True, name="learning-tick")
    t.stop = stop  # type: ignore[attr-defined]
    t.start()
    return t
