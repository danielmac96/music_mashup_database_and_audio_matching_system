"""Bounded background work queues for the auto-chaining pipeline.

Replaces the old fire-and-forget FastAPI BackgroundTasks fan-out (which could
launch a Demucs separation for every track in a playlist at once and thrash the
machine) — and the old single FIFO, where every queued track waited behind a
minutes-long Demucs run even for its 10-second download.

One queue + thread pool PER STAGE (download / quick / stems / analysis), sized
by the config knobs DOWNLOAD_WORKERS / QUICK_WORKERS / STEM_WORKERS /
ANALYSIS_WORKERS. A track hops queues as its status advances, so several
downloads, the quick tier on the next downloads, one Demucs separation, and a
couple of full analyses all run concurrently.

Each queue is a priority queue (config.PRIORITY_*): a button pressed on one
track goes ahead of an import, and an import ahead of a bulk backfill of the
library; within one priority the order is still first come, first served. A
job keeps its priority as it hops from stage to stage. Scheduling is status-derived
(api.workers.pipeline_worker.next_stage), which keeps restart resumability and
the single kind="pipeline" job per track (with a live ``stage`` field) intact.

Jobs live in the in-memory api.jobs registry. On restart the queues are empty,
so ``resume_pending()`` re-enqueues any track that was mid-pipeline (see its
docstring).
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
from typing import Optional

from config import (ANALYSIS_WORKERS, DOWNLOAD_WORKERS, PRIORITY_INGEST,
                    QUICK_WORKERS, STEM_WORKERS)

from api import jobs

log = logging.getLogger(__name__)

# Items are (priority, seq, job_id, song_id): lower priority first, then the
# order they arrived in (seq), so equal priorities stay FIFO.
_QUEUES: dict[str, "queue.PriorityQueue[tuple[int, int, str, int]]"] = {
    "download": queue.PriorityQueue(),
    "quick": queue.PriorityQueue(),
    "stems": queue.PriorityQueue(),
    "analysis": queue.PriorityQueue(),
}
_STAGE_WORKERS = {
    "download": DOWNLOAD_WORKERS,
    "quick": QUICK_WORKERS,
    "stems": STEM_WORKERS,
    "analysis": ANALYSIS_WORKERS,
}
_SEQ = itertools.count()
_STARTED = False
_LOCK = threading.Lock()
# Worker threads actually started per stage (start() may be forced to a size).
_POOL_SIZES: dict[str, int] = {}


def enqueue_song(song_id: int, priority: int = PRIORITY_INGEST) -> str:
    """Create a pipeline job for a track and queue it at the stage it needs
    next, at ``priority`` (config.PRIORITY_*). Returns the job id."""
    job_id = jobs.new_job(kind="pipeline", message="Queued for processing",
                          song_id=song_id, stage="queued")
    jobs.update(job_id, priority=int(priority))
    _dispatch(job_id, song_id)
    return job_id


def _put(stage: str, job_id: str, song_id: int) -> None:
    job = jobs.get(job_id) or {}
    _QUEUES[stage].put((int(job.get("priority", PRIORITY_INGEST)), next(_SEQ),
                        job_id, song_id))


def take(stage: str, block: bool = True) -> tuple[str, int]:
    """(job_id, song_id) of the next item in a stage's queue, highest
    priority first."""
    _prio, _seq, job_id, song_id = _QUEUES[stage].get(block=block)
    return job_id, song_id


def _dispatch(job_id: str, song_id: int) -> None:
    """Route a track into the queue for its next stage (status-derived). A
    track that is already fully processed completes its job immediately."""
    from api.workers import pipeline_worker

    stage = pipeline_worker.next_stage(song_id)
    if stage is None:
        # Nothing left to do — still run the trailing non-fatal structure pass
        # (cheap no-op when sections exist) so re-Process behaves like before.
        _put("analysis", job_id, song_id)
        return
    # Recorded before the put: a free worker can take the item at once, and its
    # 'running' must not be overwritten by a late 'waiting'.
    jobs.stage_wait(job_id, stage)
    _put(stage, job_id, song_id)


def _worker_loop(stage: str, worker_index: int) -> None:
    # Imported lazily so a queue import never drags in the audio stack.
    from api.workers import pipeline_worker

    q = _QUEUES[stage]
    while True:
        job_id, song_id = take(stage)
        try:
            # A done track landing here (see _dispatch) just finalizes.
            wanted = pipeline_worker.next_stage(song_id)
            if wanted is None:
                pipeline_worker._finalize(job_id, song_id)
            elif wanted != stage:
                # Status moved while queued (e.g. manual button) — re-route.
                jobs.stage_drop(job_id, stage)
                jobs.stage_wait(job_id, wanted)
                _put(wanted, job_id, song_id)
            else:
                outcome = pipeline_worker.run_stage(job_id, song_id, stage)
                if outcome == "next":
                    _dispatch(job_id, song_id)
        except Exception:  # noqa: BLE001 — a crash must not kill the worker thread
            log.exception("%s worker %d crashed on song %s", stage, worker_index, song_id)
            try:
                jobs.fail(job_id, "Pipeline worker crashed — see server logs")
            except Exception:  # noqa: BLE001
                pass
        finally:
            q.task_done()


def start(num_workers: Optional[int] = None) -> None:
    """Start the per-stage worker pools once (idempotent). Safe to call at app
    startup. ``num_workers`` (tests only) forces every stage to that size."""
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        for stage, q_workers in _STAGE_WORKERS.items():
            n = max(1, num_workers if num_workers is not None else q_workers)
            _POOL_SIZES[stage] = n
            for i in range(n):
                threading.Thread(
                    target=_worker_loop, args=(stage, i),
                    name=f"pipeline-{stage}-{i}", daemon=True,
                ).start()
            log.info("Pipeline %s queue started with %d worker(s)", stage, n)


def resume_pending() -> int:
    """Re-enqueue tracks that were mid-pipeline when the server last stopped.

    'Mid-pipeline' = a status strictly before 'analysed' and not a terminal
    error_* (those wait for an explicit user retry).
    queued/downloaded/stemmed tracks pick up from where they left
    off — a downloaded track whose quick tier never ran gets it first.
    Returns the number of tracks re-enqueued."""
    from database.models import get_songs_by_status

    pending = get_songs_by_status("queued", "downloaded", "stemmed")
    for song in pending:
        enqueue_song(song["id"])
    if pending:
        log.info("Resumed %d unfinished track(s) into the pipeline queues", len(pending))
    return len(pending)


def queued_count() -> int:
    return sum(q.qsize() for q in _QUEUES.values())


def snapshot() -> dict[str, dict]:
    """What each stage queue holds, in line order: {stage: {workers, waiting:
    [job_id, ...]}}. Reads the queue's heap under its own mutex and sorts it by
    (priority, arrival), which is the order workers will take them."""
    out: dict[str, dict] = {}
    for stage, q in _QUEUES.items():
        with q.mutex:
            # The heap is not in line order; sorted, it is.
            waiting = [job_id for _p, _s, job_id, _song in sorted(q.queue)]
        out[stage] = {
            "workers": _POOL_SIZES.get(stage, max(1, _STAGE_WORKERS[stage])),
            "waiting": waiting,
        }
    return out
