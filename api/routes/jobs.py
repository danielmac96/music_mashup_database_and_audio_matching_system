"""Job status endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import jobs, queue_runner

router = APIRouter()

# Which worker pool a running stage record occupies. Structure runs on the
# analysis pool (it is the trailing pass of that stage), so it counts there.
_POOL_OF = {"download": "download", "stems": "stems",
            "analysis": "analysis", "structure": "analysis"}


@router.get("")
def list_jobs(active_only: bool = False, kind: str = "") -> dict:
    """All tracked jobs (newest first). ``active_only=true`` drops finished
    jobs; ``kind=pipeline`` filters to auto-chain jobs. Used by the Library tab
    to drive live per-track pipeline progress + the batch banner."""
    items = jobs.list_jobs(active_only=active_only, kind=kind or None)
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
    for job in jobs.list_jobs(active_only=True, kind="pipeline"):
        for name, rec in (job.get("stages") or {}).items():
            pool = _POOL_OF.get(name)
            if rec.get("state") == "running" and pool in busy:
                busy[pool] += 1

    positions: dict[str, dict] = {}
    stages_out: dict[str, dict] = {}
    for stage, info in snap.items():
        for i, job_id in enumerate(info["waiting"]):
            positions[job_id] = {"stage": stage, "position": i + 1}
        stages_out[stage] = {"workers": info["workers"], "running": busy[stage],
                             "waiting": len(info["waiting"])}
    return {"stages": stages_out, "positions": positions}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
