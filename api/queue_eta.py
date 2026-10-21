"""Queue readability: an ETA and one plain sentence per job.

The Queue screen used to work both out in the browser from /api/jobs/timings.
They are decided here so every screen reads the same estimate: the median wall
time per audio-second of each stage on THIS library (analysis_runs, the same
rows /api/jobs/timings summarises) × the track's length, less what a running
stage has already spent, plus the line in front of a waiting one.

Everything here is an estimate and is allowed to be absent: no history is
`eta_secs: null`, never 0, and nothing in this module raises.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from statistics import median
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

# analysis_runs groups that make up each pool's work for one track (the stage
# rows api/workers/stages._timed writes). ".cached" / ".reused" runs are left
# out: they are what an unchanged file costs, not what the next new track will.
POOL_GROUPS = {"download": ("download",), "quick": ("quick",), "stems": ("stems",),
               "analysis": ("analysis", "structure")}
# The groups one stage RECORD covers while it runs. Structure is its own record
# on the analysis pool.
RECORD_GROUPS = {"download": ("download",), "quick": ("quick",), "stems": ("stems",),
                 "analysis": ("analysis",), "structure": ("structure",)}
POOL_OF = {"download": "download", "quick": "quick", "stems": "stems",
           "analysis": "analysis", "structure": "analysis"}

_DOING = {"download": "Downloading", "quick": "Quick analysis",
          "stems": "Separating vocals", "analysis": "Analysing",
          "structure": "Finding sections"}
_WAITING_FOR = {"download": "download", "quick": "quick analysis",
                "stems": "stems", "analysis": "analysis", "structure": "sections"}
_FAILED = {"download": "downloading", "quick": "in quick analysis",
           "stems": "separating stems", "analysis": "analysing",
           "structure": "finding sections"}
_STAGE_ORDER = ("download", "quick", "stems", "analysis", "structure")


def _parse(ts: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except (TypeError, ValueError):
        return None


def about(secs: float) -> str:
    """'about 40 s' / 'about 2 min' / 'about 1.5 h'."""
    secs = max(0.0, float(secs))
    if secs >= 3600:
        return f"about {secs / 3600:.1f} h"
    if secs >= 90:
        return f"about {round(secs / 60)} min"
    return f"about {max(1, round(secs))} s"


def took(secs: float) -> str:
    """'4 m 12 s' / '38 s' / '1 h 3 m'."""
    secs = int(round(max(0.0, float(secs))))
    if secs >= 3600:
        return f"{secs // 3600} h {secs % 3600 // 60} m"
    if secs >= 60:
        return f"{secs // 60} m {secs % 60} s"
    return f"{secs} s"


# ── history ──────────────────────────────────────────────────────────────────

def stage_rates(db_path=None) -> Dict[str, Dict[str, Optional[float]]]:
    """{grp: {"per_audio_sec", "typical_secs"}} from successful stage-level
    rows of analysis_runs: the median wall seconds per second of audio, and the
    median wall seconds. {} when nothing has run (or the read fails)."""
    groups = sorted({g for gs in RECORD_GROUPS.values() for g in gs})
    try:
        from database import models
        conn = models.get_conn(db_path if db_path is not None else models.DB_PATH)
        try:
            rows = conn.execute(
                f"""SELECT grp, ms, audio_secs FROM analysis_runs
                    WHERE ok = 1 AND stem_type IS NULL
                      AND grp IN ({','.join('?' * len(groups))})""",
                groups).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — an estimate is never worth a 500
        log.warning("could not read stage timings for ETAs", exc_info=True)
        return {}
    ms: Dict[str, list] = {}
    rtf: Dict[str, list] = {}
    for r in rows:
        if r["ms"] is None:
            continue
        ms.setdefault(r["grp"], []).append(r["ms"] / 1000.0)
        if r["audio_secs"]:
            rtf.setdefault(r["grp"], []).append(r["ms"] / 1000.0 / r["audio_secs"])
    return {g: {"per_audio_sec": median(rtf[g]) if rtf.get(g) else None,
                "typical_secs": median(v)} for g, v in ms.items()}


def _durations(song_ids: Iterable[int], db_path=None) -> Dict[int, float]:
    ids = sorted({int(s) for s in song_ids if s is not None})
    if not ids:
        return {}
    try:
        from database import models
        conn = models.get_conn(db_path if db_path is not None else models.DB_PATH)
        try:
            rows = conn.execute(
                f"SELECT id, duration_secs FROM songs WHERE id IN ({','.join('?' * len(ids))})",
                ids).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        log.warning("could not read track lengths for ETAs", exc_info=True)
        return {}
    return {r["id"]: float(r["duration_secs"]) for r in rows if r["duration_secs"]}


def estimate(groups: Iterable[str], rates: Dict, duration: Optional[float]) -> Optional[float]:
    """Seconds these groups take for a track `duration` seconds long: the
    per-audio-second rate × its length where both are known, else the typical
    time. None when any group has no history at all... unless another has —
    a partial estimate is still the honest floor, so known groups are summed."""
    total, seen = 0.0, False
    for g in groups:
        r = rates.get(g)
        if not r:
            continue
        if r.get("per_audio_sec") is not None and duration:
            total += r["per_audio_sec"] * duration
        elif r.get("typical_secs") is not None:
            total += r["typical_secs"]
        else:
            continue
        seen = True
    return total if seen else None


def typical(groups: Iterable[str], rates: Dict) -> Optional[float]:
    return estimate(groups, rates, None)


# ── annotation ───────────────────────────────────────────────────────────────

def _active_stage(job: dict):
    """(name, record) of the stage the job is in now: the running one, else the
    waiting one."""
    stages = job.get("stages") or {}
    for state in ("running", "waiting"):
        for name in _STAGE_ORDER:
            rec = stages.get(name)
            if rec and rec.get("state") == state:
                return name, rec
    return None, None


def _first_line(text: Optional[str]) -> str:
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def _done_secs(job: dict) -> Optional[float]:
    recs = [r for r in (job.get("stages") or {}).values() if r]
    starts = [_parse(r.get("started_at")) for r in recs]
    ends = [_parse(r.get("finished_at")) for r in recs]
    starts, ends = [s for s in starts if s], [e for e in ends if e]
    a = min(starts) if starts else _parse(job.get("created_at"))
    b = max(ends) if ends else _parse(job.get("updated_at"))
    return (b - a).total_seconds() if a and b and b >= a else None


def _failed_stage(job: dict) -> Optional[str]:
    for name in _STAGE_ORDER:
        rec = (job.get("stages") or {}).get(name)
        if rec and rec.get("state") == "failed":
            return name
    stage = job.get("stage")
    return {"analyze": "analysis"}.get(stage, stage) if stage in (
        *_STAGE_ORDER, "analyze") else None


def annotate(jobs_list: List[dict], snapshot: Optional[Dict[str, dict]] = None,
             db_path=None, now: Optional[datetime] = None) -> Dict[str, dict]:
    """Add `eta_secs` and `summary` to every job in place (and `eta_secs` to
    each waiting or running stage record), and return the per-pool totals:
    {pool: {"clears_in_secs"}}.

    `snapshot` is queue_runner.snapshot() — who waits where, in line order."""
    snapshot = snapshot or {}
    now = now or datetime.now(timezone.utc)
    active = [j for j in jobs_list if j.get("status") in ("queued", "running")]
    try:
        rates = stage_rates(db_path) if active else {}
        durations = _durations((j.get("song_id") for j in active), db_path) if rates else {}
    except Exception:  # noqa: BLE001
        rates, durations = {}, {}

    position: Dict[str, tuple] = {}
    for pool, info in snapshot.items():
        for i, job_id in enumerate(info.get("waiting") or []):
            position[job_id] = (pool, i + 1, max(1, int(info.get("workers") or 1)))

    load: Dict[str, float] = {p: 0.0 for p in POOL_GROUPS}
    unknown: Dict[str, bool] = {p: False for p in POOL_GROUPS}
    for job in jobs_list:
        try:
            _annotate_one(job, rates, durations, position, now, load, unknown)
        except Exception:  # noqa: BLE001 — never fail a poll over a sentence
            log.warning("could not summarise job %s", job.get("id"), exc_info=True)
            job.setdefault("eta_secs", None)
            job.setdefault("summary", _first_line(job.get("message")) or "")

    pools: Dict[str, dict] = {}
    for pool in POOL_GROUPS:
        workers = max(1, int((snapshot.get(pool) or {}).get("workers") or 1))
        pools[pool] = {"clears_in_secs": None if unknown[pool]
                       else round(load[pool] / workers, 1)}
    return pools


def _playable(job, name, eta, rates, duration) -> None:
    """When this track will have pairs: its quick tier done (the provisional
    pairs are scored straight after it, matcher.score_song_pairs). Set on jobs
    still downloading or in the quick tier; None once past it or unknown."""
    if eta is None or name not in ("download", "quick"):
        return
    if name == "quick":
        job["playable_eta_secs"] = eta
        return
    quick = estimate(RECORD_GROUPS["quick"], rates, duration)
    if quick is not None:
        job["playable_eta_secs"] = round(eta + quick, 1)


def next_playable_secs(jobs_list: List[dict]) -> Optional[float]:
    """The soonest any track in the queue gets its first pairs (annotate first)."""
    etas = [j["playable_eta_secs"] for j in jobs_list if j.get("playable_eta_secs") is not None]
    return min(etas) if etas else None


def _annotate_one(job, rates, durations, position, now, load, unknown) -> None:
    status = job.get("status")
    job["eta_secs"] = None
    job["playable_eta_secs"] = None
    if status == "completed":
        secs = _done_secs(job)
        job["summary"] = f"Done in {took(secs)}" if secs is not None else "Done"
        return
    if status == "failed":
        stage = _failed_stage(job)
        err = _first_line(job.get("error")) or "unknown error"
        where = f" {_FAILED[stage]}" if stage in _FAILED else ""
        retry = " — Retry" if job.get("song_id") is not None else ""
        job["summary"] = f"Failed{where}: {err}{retry}"
        return

    duration = durations.get(job.get("song_id"))
    name, rec = _active_stage(job)
    if rec is None:
        # A library-wide job (scoring, a bulk pass…) or a pipeline job between
        # records: its own progress message is the sentence.
        job["summary"] = _first_line(job.get("message")) or (
            "Queued" if status == "queued" else "Working…")
        return

    pool = POOL_OF[name]
    if rec.get("state") == "running":
        est = estimate(RECORD_GROUPS[name], rates, duration)
        eta = None
        if est is not None:
            started = _parse(rec.get("started_at"))
            elapsed = (now - started).total_seconds() if started else 0.0
            eta = round(max(0.0, est - max(0.0, elapsed)), 1)
            load[pool] += eta
        else:
            unknown[pool] = True
        rec["eta_secs"] = job["eta_secs"] = eta
        _playable(job, name, eta, rates, duration)
        job["summary"] = (f"{_DOING[name]} — {about(eta)} left" if eta is not None
                          else f"{_DOING[name]}…")
        return

    # Waiting: its own work, plus the line in front of it shared across the
    # pool's workers.
    own = estimate(POOL_GROUPS[pool], rates, duration)
    _pool, pos, workers = position.get(job.get("id"), (pool, None, 1))
    eta = None
    if own is not None:
        load[pool] += own
        typ = typical(POOL_GROUPS[pool], rates)
        eta = round(own + (pos or 0) * (typ or 0.0) / workers, 1)
    else:
        unknown[pool] = True
    rec["eta_secs"] = job["eta_secs"] = eta
    _playable(job, name, eta, rates, duration)
    line = f" (#{pos} in line)" if pos else ""
    job["summary"] = f"Waiting for {_WAITING_FOR[name]}{line}"
