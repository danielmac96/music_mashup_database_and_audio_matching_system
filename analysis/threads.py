"""Per-worker thread cap for the analysis pools (readme §9, phase 3 finding).

Analysis runs on threads (api/queue_runner.py), and numpy/scipy's BLAS and
OpenMP pools each default to every core — so six analysis workers on a
20-core host ran 6 × 20 threads and each track slowed from ~56 s to 150–200 s
(1.4 → ~1.7 tracks/min for three times the workers). The pools are process-wide,
so the cap is cores ÷ the analysis and quick workers that share them; Demucs
has its own cap (stems/separate.thread_env), being a subprocess.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def worker_thread_cap() -> int:
    from config import ANALYSIS_WORKERS, QUICK_WORKERS
    workers = max(1, int(ANALYSIS_WORKERS) + int(QUICK_WORKERS))
    return max(1, (os.cpu_count() or 2) // workers)


def cap_analysis_threads() -> int:
    """Apply the cap to every BLAS/OpenMP pool loaded so far. Called at the
    start of each analysis, so a library first imported mid-run (librosa pulls
    scipy's in lazily) is capped too; a scan of loaded libraries costs ~1 ms.
    Never raises — without threadpoolctl the pools keep their defaults."""
    n = worker_thread_cap()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:  # noqa: BLE001
        log.debug("thread cap not applied", exc_info=True)
    return n
