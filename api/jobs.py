"""In-memory job tracker for background tasks (download, separate)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

JOBS: dict[str, dict] = {}
_LOCK = Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job(kind: str, message: str = "Queued",
            song_id: Optional[int] = None, stage: Optional[str] = None) -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "song_id": song_id,   # track this job belongs to (None = library-wide)
            "stage": stage,       # current pipeline stage for a chained job
            "status": "queued",
            "progress": 0,
            "message": message,
            "result": None,
            "error": None,
            "traceback": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
    return job_id


def update(job_id: str, **fields: Any) -> None:
    with _LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = _now()


def done(job_id: str, result: Optional[dict] = None) -> None:
    update(job_id, status="completed", progress=100, result=result, message="Completed")


def fail(job_id: str, error: str, traceback_text: Optional[str] = None) -> None:
    fields: dict[str, Any] = {"status": "failed", "error": error, "message": error}
    if traceback_text is not None:
        fields["traceback"] = traceback_text
    update(job_id, **fields)


def get(job_id: str) -> Optional[dict]:
    with _LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs(active_only: bool = False, kind: Optional[str] = None) -> list[dict]:
    """Snapshot of jobs, newest first. ``active_only`` drops completed/failed
    terminal jobs; ``kind`` filters by job kind (e.g. 'pipeline')."""
    with _LOCK:
        jobs = [dict(j) for j in JOBS.values()]
    if kind:
        jobs = [j for j in jobs if j.get("kind") == kind]
    if active_only:
        jobs = [j for j in jobs if j.get("status") in ("queued", "running")]
    jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return jobs
