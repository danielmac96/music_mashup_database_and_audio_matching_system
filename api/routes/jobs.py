"""Job status endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import jobs, queue_eta, queue_runner

router = APIRouter()

# Which worker pool a running stage record occupies. Structure runs on the
# analysis pool (it is the trailing pass of that stage), so it counts there.
_POOL_OF = {"download": "download", "quick": "quick", "stems": "stems",
            "analysis": "analysis", "structure": "analysis"}


@router.get("")
def list_jobs(active_only: bool = False, kind: str = "") -> dict:
    """All tracked jobs (newest first). ``active_only=true`` drops finished
    jobs; ``kind=pipeline`` filters to auto-chain jobs. Used by the Library tab
    to drive live per-track pipeline progress + the batch banner."""
    items = jobs.list_jobs(active_only=active_only, kind=kind or None)
    # eta_secs + summary per job (api/queue_eta.py); never fails the poll.
    queue_eta.annotate(items, queue_runner.snapshot())
    return {"count": len(items), "jobs": items}


# Declared before /{job_id}, or "queue" would be looked up as a job id.
@router.get("/queue")
def queue_snapshot() -> dict:
    """The pipeline pools right now, for the Queue screen: per stage, how many
    workers, how many are busy and how many tracks wait; and each waiting job's
    place in line. Busy is counted from stage records, not job status — a job
    between stages is 'running' while it waits in the next queue."""
    snap = queue_runner.snapshot()
    busy = {stage: 0 for stage in snap}
    active = jobs.list_jobs(active_only=True, kind="pipeline")
    # eta_secs + summary per job and the time each pool takes to clear
    # (api/queue_eta.py). Estimates: null without history, never an error.
    pools = queue_eta.annotate(active, snap)
    by_id = {j["id"]: j for j in active}
    running: list[dict] = []
    for job in active:
        for name, rec in (job.get("stages") or {}).items():
            pool = _POOL_OF.get(name)
            if rec.get("state") == "running" and pool in busy:
                busy[pool] += 1
                running.append({"job_id": job["id"], "song_id": job.get("song_id"),
                                "stage": name, "eta_secs": rec.get("eta_secs"),
                                "summary": job.get("summary") or ""})

    positions: dict[str, dict] = {}
    stages_out: dict[str, dict] = {}
    for stage, info in snap.items():
        for i, job_id in enumerate(info["waiting"]):
            job = by_id.get(job_id) or {}
            positions[job_id] = {"stage": stage, "position": i + 1,
                                 "eta_secs": job.get("eta_secs"),
                                 "summary": job.get("summary") or ""}
        stages_out[stage] = {"workers": info["workers"], "running": busy[stage],
                             "waiting": len(info["waiting"]),
                             "clears_in_secs": (pools.get(stage) or {}).get("clears_in_secs")}
    # The soonest a track in the queue gets its first (provisional) pairs.
    return {"stages": stages_out, "positions": positions, "running": running,
            "next_playable_secs": queue_eta.next_playable_secs(active)}


# Declared before /{job_id}, like /queue.
@router.get("/timings")
def timing_summary(since: str = "") -> dict:
    """Where pipeline time goes: per stage and per analysis step (and stem),
    run count, failures, median / p90 / total milliseconds and the median
    real-time factor, from the persisted analysis_runs table. ``since`` is an
    SQLite datetime ('2026-09-28 12:00:00') to look at one batch only."""
    from database.models import analysis_timing_summary
    rows = analysis_timing_summary(since=since or None)
    return {"count": len(rows), "timings": rows}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    queue_eta.annotate([job], queue_runner.snapshot())
    return job
