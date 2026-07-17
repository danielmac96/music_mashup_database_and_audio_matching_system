"""Bounded background work queues for the auto-chaining pipeline.

Replaces the old fire-and-forget FastAPI BackgroundTasks fan-out (which could
launch a Demucs separation for every track in a playlist at once and thrash the
machine) — and the old single FIFO, where every queued track waited behind a
minutes-long Demucs run even for its 10-second download.

One queue + thread pool PER STAGE (download / stems / analysis), sized by the
config knobs DOWNLOAD_WORKERS / STEM_WORKERS / ANALYSIS_WORKERS. A track hops
queues as its status advances, so several downloads, one Demucs separation, and
a couple of librosa analyses all run concurrently. Scheduling is status-derived
(api.workers.pipeline_worker.next_stage), which keeps restart resumability and
the single kind="pipeline" job per track (with a live ``stage`` field) intact.

Jobs live in the in-memory api.jobs registry. On restart the queues are empty,
so ``resume_pending()`` re-enqueues any track that was mid-pipeline (see its
docstring) — the same status-derived resumability the CLI has.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from config import ANALYSIS_WORKERS, DOWNLOAD_WORKERS, STEM_WORKERS

from api import jobs

log = logging.getLogger(__name__)

_QUEUES: dict[str, "queue.Queue[tuple[str, int]]"] = {
    "download": queue.Queue(),
    "stems": queue.Queue(),
    "analysis": queue.Queue(),
}
_STAGE_WORKERS = {
    "download": DOWNLOAD_WORKERS,
    "stems": STEM_WORKERS,
    "analysis": ANALYSIS_WORKERS,
}
_STARTED = False
_LOCK = threading.Lock()


def enqueue_song(song_id: int) -> str:
    """Create a pipeline job for a track and queue it at the stage it needs
    next. Returns the job id."""
    job_id = jobs.new_job(kind="pipeline", message="Queued for processing",
                          song_id=song_id, stage="queued")
    _dispatch(job_id, song_id)
    return job_id


def _dispatch(job_id: str, song_id: int) -> None:
    """Route a track into the queue for its next stage (status-derived). A
    track that is already fully processed completes its job immediately."""
    from api.workers import pipeline_worker

    stage = pipeline_worker.next_stage(song_id)
    if stage is None:
        # Nothing left to do — still run the trailing non-fatal structure pass
        # (cheap no-op when sections exist) so re-Process behaves like before.
        _QUEUES["analysis"].put((job_id, song_id))
        return
    _QUEUES[stage].put((job_id, song_id))


def _worker_loop(stage: str, worker_index: int) -> None:
    # Imported lazily so a queue import never drags in the audio stack.
    from api.workers import pipeline_worker

    q = _QUEUES[stage]
    while True:
        job_id, song_id = q.get()
        try:
            # A done track landing here (see _dispatch) just finalizes.
            wanted = pipeline_worker.next_stage(song_id)
            if wanted is None:
                pipeline_worker._finalize(job_id, song_id)
            elif wanted != stage:
                # Status moved while queued (e.g. manual button) — re-route.
                _QUEUES[wanted].put((job_id, song_id))
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
            for i in range(n):
                threading.Thread(
                    target=_worker_loop, args=(stage, i),
                    name=f"pipeline-{stage}-{i}", daemon=True,
                ).start()
            log.info("Pipeline %s queue started with %d worker(s)", stage, n)


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
        log.info("Resumed %d unfinished track(s) into the pipeline queues", len(pending))
    return len(pending)


def queued_count() -> int:
    return sum(q.qsize() for q in _QUEUES.values())
