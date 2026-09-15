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
            # Per-stage timeline for a chained job: {stage: record}. `stage`
            # above is only where the job is NOW; this is where it has been, so
            # the Queue screen can show every stage's outcome and duration.
            "stages": {},
            "created_at": _now(),
            "updated_at": _now(),
        }
    return job_id


def _copy(job: dict) -> dict:
    """A snapshot the caller can hold outside the lock — deep enough that a
    worker updating a stage record does not mutate a copy mid-serialisation."""
    out = dict(job)
    out["stages"] = {k: dict(v) for k, v in (job.get("stages") or {}).items()}
    return out


def update(job_id: str, **fields: Any) -> None:
    with _LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = _now()


def progress_updater(job_id: str, stage: Optional[str] = None,
                     stage_key: Optional[str] = None) -> Callable:
    """Standard (pct|None, message) callback the pipeline stages expect.

    Every worker used to define this same closure inline; keep it in one place.
    With `stage`, messages are prefixed 'stage: …' and the job's stage field is
    kept current (used by the auto-chain pipeline worker). With `stage_key`, the
    progress and message also land on that stage's timeline record — passed
    explicitly because the label ('analyze') and the queue ('analysis') differ."""
    def _on_progress(pct: Optional[int], msg: str) -> None:
        fields: dict[str, Any] = {"message": f"{stage}: {msg}" if stage else msg}
        if stage:
            fields["stage"] = stage
        if pct is not None:
            fields["progress"] = pct
        update(job_id, **fields)
        if stage_key:
            rec: dict[str, Any] = {"message": msg}
            if pct is not None:
                rec["progress"] = pct
            _stage_merge(job_id, stage_key, rec)
    return _on_progress


# ── Per-stage timeline ────────────────────────────────────────────────────────
# A record is {state, enqueued_at, started_at, finished_at, progress, message,
# error}; state is waiting | running | done | failed | skipped.

def _stage_merge(job_id: str, stage: str, fields: dict) -> None:
    with _LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.setdefault("stages", {}).setdefault(stage, {}).update(fields)
        job["updated_at"] = _now()


def stage_wait(job_id: str, stage: str) -> None:
    """The job has been put in `stage`'s queue."""
    _stage_merge(job_id, stage, {"state": "waiting", "enqueued_at": _now(),
                                 "started_at": None, "finished_at": None,
                                 "progress": 0, "message": None, "error": None})


def stage_start(job_id: str, stage: str) -> None:
    _stage_merge(job_id, stage, {"state": "running", "started_at": _now(),
                                 "finished_at": None, "progress": 0,
                                 "message": "starting…", "error": None})


def stage_finish(job_id: str, stage: str, state: str,
                 error: Optional[str] = None, message: Optional[str] = None) -> None:
    """End a stage as done / failed / skipped."""
    fields: dict[str, Any] = {"state": state, "finished_at": _now()}
    if state == "done":
        fields["progress"] = 100
    if error is not None:
        fields["error"] = error
    if message is not None:
        fields["message"] = message
    _stage_merge(job_id, stage, fields)


def stage_drop(job_id: str, stage: str) -> None:
    """Forget a stage record — the job was waiting for a stage it no longer needs."""
    with _LOCK:
        job = JOBS.get(job_id)
        if job:
            (job.get("stages") or {}).pop(stage, None)


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
        return _copy(job) if job else None


def list_jobs(active_only: bool = False, kind: Optional[str] = None) -> list[dict]:
    """Snapshot of jobs, newest first. ``active_only`` drops completed/failed
    terminal jobs; ``kind`` filters by job kind (e.g. 'pipeline')."""
    with _LOCK:
        jobs = [_copy(j) for j in JOBS.values()]
    if kind:
        jobs = [j for j in jobs if j.get("kind") == kind]
    if active_only:
        jobs = [j for j in jobs if j.get("status") in ("queued", "running")]
    jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return jobs
