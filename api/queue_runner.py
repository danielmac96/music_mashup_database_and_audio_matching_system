"""Bounded background work queue for the auto-chaining pipeline.

Replaces the old fire-and-forget FastAPI BackgroundTasks fan-out (which could
launch a Demucs separation for every track in a playlist at once and thrash the
machine). Here a fixed pool of ``config.PIPELINE_WORKERS`` daemon threads drains
a single FIFO queue, so at most N tracks are ever processed concurrently.

Jobs live in the in-memory api.jobs registry. On restart the queue is empty, so
``resume_pending()`` re-enqueues any track that was mid-pipeline (see its
docstring) — the same status-derived resumability the CLI has.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from config import PIPELINE_WORKERS

from api import jobs

log = logging.getLogger(__name__)

_Q: "queue.Queue[tuple[str, int]]" = queue.Queue()
_STARTED = False
_LOCK = threading.Lock()


def enqueue_song(song_id: int) -> str:
    """Create a pipeline job for a track and queue it. Returns the job id."""
    job_id = jobs.new_job(kind="pipeline", message="Queued for processing",
                          song_id=song_id, stage="queued")
    _Q.put((job_id, song_id))
    return job_id


def _worker_loop(worker_index: int) -> None:
    # Imported lazily so a queue import never drags in the audio stack.
    from api.workers import pipeline_worker

    while True:
        job_id, song_id = _Q.get()
        try:
            pipeline_worker.run(job_id, song_id)
        except Exception:  # noqa: BLE001 — a crash must not kill the worker thread
            log.exception("pipeline worker %d crashed on song %s", worker_index, song_id)
            try:
                jobs.fail(job_id, "Pipeline worker crashed — see server logs")
            except Exception:  # noqa: BLE001
                pass
        finally:
            _Q.task_done()


def start(num_workers: Optional[int] = None) -> None:
    """Start the worker pool once (idempotent). Safe to call at app startup."""
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        n = num_workers if num_workers is not None else PIPELINE_WORKERS
        for i in range(max(1, n)):
            threading.Thread(
                target=_worker_loop, args=(i,), name=f"pipeline-worker-{i}",
                daemon=True,
            ).start()
        log.info("Pipeline queue started with %d worker(s)", max(1, n))


def resume_pending() -> int:
    """Re-enqueue tracks that were mid-pipeline when the server last stopped.

    'Mid-pipeline' = a status strictly before 'analysed' and not a terminal
    error_* (those wait for an explicit user retry). Matches the CLI's resume
    behaviour: queued/downloaded/stemmed tracks pick up from where they left
    off. Returns the number of tracks re-enqueued."""
    from database.models import get_songs_by_status

    pending = get_songs_by_status("queued", "downloaded", "stemmed")
    for song in pending:
        enqueue_song(song["id"])
    if pending:
        log.info("Resumed %d unfinished track(s) into the pipeline queue", len(pending))
    return len(pending)


def queued_count() -> int:
    return _Q.qsize()
