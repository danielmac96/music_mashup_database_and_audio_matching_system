"""In-memory job tracker for background tasks (download, separate)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Optional

JOBS: dict[str, dict] = {}
_LOCK = Lock()

# Cap on finished (completed/failed) jobs kept for the UI's history views.
# Oldest terminal jobs are dropped first; active jobs are never pruned.
MAX_TERMINAL_JOBS = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prune_terminal_locked() -> None:
    """Drop the oldest finished jobs beyond MAX_TERMINAL_JOBS (caller holds _LOCK)."""
    terminal = [j for j in JOBS.values() if j.get("status") in ("completed", "failed")]
    excess = len(terminal) - MAX_TERMINAL_JOBS
    if excess <= 0:
        return
    terminal.sort(key=lambda j: j.get("created_at") or "")
    for job in terminal[:excess]:
        JOBS.pop(job["id"], None)


def new_job(kind: str, message: str = "Queued",
            song_id: Optional[int] = None, stage: Optional[str] = None) -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        _prune_terminal_locked()
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


def progress_updater(job_id: str, stage: Optional[str] = None) -> Callable:
    """Standard (pct|None, message) callback the pipeline stages expect.

    Every worker used to define this same closure inline; keep it in one place.
    With `stage`, messages are prefixed 'stage: …' and the job's stage field is
    kept current (used by the auto-chain pipeline worker)."""
    def _on_progress(pct: Optional[int], msg: str) -> None:
        fields: dict[str, Any] = {"message": f"{stage}: {msg}" if stage else msg}
        if stage:
            fields["stage"] = stage
        if pct is not None:
            fields["progress"] = pct
        update(job_id, **fields)
    return _on_progress


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
