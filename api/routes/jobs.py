"""Job status endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import jobs

router = APIRouter()


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
