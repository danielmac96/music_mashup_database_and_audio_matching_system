"""Job status endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import jobs

router = APIRouter()


@router.get("")
def list_jobs(active_only: bool = False, kind: str = "") -> dict:
    """All tracked jobs (newest first). ``active_only=true`` drops finished
    jobs; ``kind=pipeline`` filters to auto-chain jobs. Used by the Library tab
    to drive live per-track pipeline progress + the batch banner."""
    items = jobs.list_jobs(active_only=active_only, kind=kind or None)
    return {"count": len(items), "jobs": items}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
