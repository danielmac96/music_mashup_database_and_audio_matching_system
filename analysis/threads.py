"""One core budget for every CPU stage (readme §2, §9; criterion 5.2).

Analysis runs on threads (api/queue_runner.py), and numpy/scipy's BLAS and
OpenMP pools each default to every core — so six analysis workers on a
20-core host ran 6 × 20 threads and each track slowed from ~56 s to 150–200 s
(1.4 → ~1.7 tracks/min for three times the workers). The pools are process-wide,
so each analysis/quick worker is capped at cores ÷ the workers that share them
(``worker_thread_cap``); Demucs has its own cap (stems/separate.thread_env),
being a subprocess; allin1 sets torch's; EffNet's TensorFlow takes the
worker's cap (``tf_thread_env``).

Those caps alone still add up past the machine: Demucs at cores − 1 beside
four analysis workers at one thread each is seven threads on four cores. So
every CPU stage also takes its threads from ONE budget of exactly the
machine's cores (``BUDGET``) for as long as it computes — Demucs its
DEMUCS_THREADS, allin1 its ALLIN1_THREADS, an analysis / quick / structure
run its worker cap, a scoring run one. A stage waits for its share rather
than oversubscribing; the queue is first come, first served, so a separation
waiting for three cores is not starved by a stream of one-core analyses, and
an idle machine still runs four analyses at once. A thread already holding a
share takes nothing more (nested holds are free), and a request is clamped to
the budget (with a startup warning) so nothing can wait for ever.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections import deque
from typing import Iterator, Optional

log = logging.getLogger(__name__)


def cores() -> int:
    return max(1, os.cpu_count() or 2)


def worker_thread_cap() -> int:
    from config import ANALYSIS_WORKERS, QUICK_WORKERS
    workers = max(1, int(ANALYSIS_WORKERS) + int(QUICK_WORKERS))
    return max(1, cores() // workers)


def tf_thread_env() -> dict:
    """TensorFlow (Essentia's EffNet heads) reads its pool sizes from these
    at session creation: intra-op at one analysis worker's cap, inter-op 1.
    The predictors run one at a time (ml_models._LOCK) inside an analysis
    worker's share, so they add nothing to the budget. Set with setdefault —
    an explicit environment wins."""
    return {"TF_NUM_INTRAOP_THREADS": str(worker_thread_cap()),
            "TF_NUM_INTEROP_THREADS": "1"}


def apply_tf_thread_env() -> None:
    for k, v in tf_thread_env().items():
        os.environ.setdefault(k, v)


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


class CoreBudget:
    """A fair counting budget of cores. ``hold(n, label)`` blocks until ``n``
    are free AND every earlier request has been served, holds them for the
    block, and gives them back. Instrumented: what each label holds now, and
    the most ever held at once (``peak``)."""

    def __init__(self, total: int):
        self.total = max(1, int(total))
        self._free = self.total
        self._cv = threading.Condition()
        self._line: deque = deque()
        self._local = threading.local()
        self.peak = 0
        self.held: dict[str, int] = {}
        self.waits = 0
        self.wait_secs = 0.0

    def clamp(self, n) -> int:
        return max(1, min(int(n or 1), self.total))

    @contextlib.contextmanager
    def hold(self, n, label: str = "cpu") -> Iterator[int]:
        if getattr(self._local, "depth", 0):
            # Already inside a share on this thread (a stage calling a scorer):
            # the outer share covers it.
            self._local.depth += 1
            try:
                yield 0
            finally:
                self._local.depth -= 1
            return
        n = self.clamp(n)
        ticket = object()
        t0 = time.monotonic()
        with self._cv:
            self._line.append(ticket)
            while self._line[0] is not ticket or self._free < n:
                self._cv.wait()
            self._line.popleft()
            self._free -= n
            in_use = self.total - self._free
            self.peak = max(self.peak, in_use)
            self.held[label] = self.held.get(label, 0) + n
            waited = time.monotonic() - t0
            if waited > 0.05:
                self.waits += 1
                self.wait_secs += waited
            self._cv.notify_all()
        self._local.depth = 1
        try:
            yield n
        finally:
            self._local.depth = 0
            with self._cv:
                self._free += n
                left = self.held.get(label, 0) - n
                if left > 0:
                    self.held[label] = left
                else:
                    self.held.pop(label, None)
                self._cv.notify_all()

    def snapshot(self) -> dict:
        with self._cv:
            return {"total": self.total, "in_use": self.total - self._free,
                    "waiting": len(self._line), "peak": self.peak,
                    "held": dict(self.held), "waits": self.waits,
                    "wait_secs": round(self.wait_secs, 1)}


BUDGET = CoreBudget(cores())


def hold(n, label: str):
    """``with threads.hold(n, "demucs"):`` — a share of the one budget."""
    return BUDGET.hold(n, label)


def requests() -> dict:
    """What each CPU stage asks the budget for, as configured."""
    from config import ALLIN1_THREADS, DEMUCS_THREADS
    cap = worker_thread_cap()
    return {"demucs": int(DEMUCS_THREADS), "allin1": int(ALLIN1_THREADS),
            "analysis": cap, "quick": cap, "structure": cap, "effnet": cap,
            "scoring": 1}


def budget_report() -> dict:
    """The startup check (criterion 5.2) and the Queue readout's line: each
    stage's configured threads, the budget they share, what the stages that
    can run at once would sum to without it, and a warning per problem.

    The budget guarantees that at most ``cores`` threads compute at once
    (asserted by ``check``): ``summed`` is what the stages would add up to
    if they all ran together without it, ``max_at_once`` what they can.
    A warning means a setting cannot be honoured on this machine: a stage
    asked for more threads than it has (and is clamped), or allin1_parallel
    asks for an overlap the cores cannot hold."""
    from config import (ALLIN1_PARALLEL, ALLIN1_WORKERS, ANALYSIS_WORKERS,
                        QUICK_WORKERS, STEM_WORKERS)
    req = requests()
    c = BUDGET.total
    heavy = int(STEM_WORKERS) * req["demucs"]
    if ALLIN1_PARALLEL:
        heavy += int(ALLIN1_WORKERS) * req["allin1"]
    else:
        heavy = max(heavy, req["allin1"])
    light = (int(ANALYSIS_WORKERS) + int(QUICK_WORKERS)) * req["analysis"]
    unbudgeted = heavy + light + req["scoring"]
    warnings = []
    for stage in ("demucs", "allin1"):
        if req[stage] > c:
            warnings.append(f"{stage} is set to {req[stage]} threads on {c} cores; "
                            f"it runs with {c}")
    if ALLIN1_PARALLEL and req["demucs"] + req["allin1"] > c:
        warnings.append(
            f"allin1_parallel is on but Demucs ({req['demucs']}) and allin1 "
            f"({req['allin1']}) need more than {c} cores together — they take "
            f"turns; split the cores between them")
    return {"cores": c, "requests": req, "heavy": heavy, "light": light,
            "summed": unbudgeted, "budget": c, "max_at_once": c,
            "oversubscribed": False, "warnings": warnings,
            "live": BUDGET.snapshot()}


def check() -> dict:
    """Run at server start: assert the budget is the machine's cores and no
    single request can exceed it, log the report, and return it."""
    report = budget_report()
    assert BUDGET.total == cores(), "the core budget must be the machine's cores"
    assert all(BUDGET.clamp(v) <= BUDGET.total for v in report["requests"].values())
    for w in report["warnings"]:
        log.warning("threads: %s", w)
    log.info("threads: one budget of %d cores — %s", report["cores"],
             ", ".join(f"{k} {v}" for k, v in report["requests"].items()))
    return report


def last_warning() -> Optional[str]:
    w = budget_report()["warnings"]
    return w[0] if w else None
