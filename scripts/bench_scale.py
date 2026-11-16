#!/usr/bin/env python
"""scripts/bench_scale.py — what the app costs at 2k / 5k / 10k tracks
(readme §8).

Builds a synthetic library per size (scripts/synth_library.py — no audio, no
Demucs, no Essentia) in its own folder and measures, on it:

  * **per-track scoring** — `score_song_pairs` for a few sampled tracks: what
    one import costs;
  * **Score library** — the full run's wall time and peak resident memory.
    Run for real when the per-track sample says it fits `--full-score-budget`
    seconds; otherwise reported as an ESTIMATE (per-track seconds × tracks ÷ 2,
    since one track's run covers it against the library in both roles), with
    the sample's standard error — one track's cost varies several-fold with
    what it is — and marked `measured: false`. An estimate is never printed as
    a measurement;
  * **the dock listing** — `get_candidates_enriched` library-wide and scoped
    to a track, p50 / p95. When the full run did not happen the candidates
    table is padded to the size a full run would leave (clones of the sampled
    tracks' real rows on other song ids), and the report says so;
  * **Segments** — `GET /api/segments` (cold, warm, bytes) and
    `/api/segments/{id}/matches` p50 / p95;
  * **the Library payload** — `GET /api/tracks`: time and size, raw and
    gzipped. (Time to first row in a browser is that plus parse and paint;
    the browser half is measured with the served bundle, not here.)
  * **event ingestion** — `POST /api/events` round trip, the request-path
    half on its own (the route function, write queued but not run), the p95
    it adds to another request's read and write while batches arrive at fifty
    times a browser's rate, and the database bytes 100,000 events take.

    python scripts/bench_scale.py                       # 2000, 5000, 10000
    python scripts/bench_scale.py --tracks 500 --keep   # one size, keep the library

Each size runs in a child process, so its peak memory is its own and the paths
it binds (`MASHUP_DB_PATH`, `MASHUP_AUDIO_ROOT`, `MASHUP_DATA_DIR`,
`MASHUP_SETTINGS_DIR`) are never the live library's. Results go to
`<out>/bench_scale.json` and `bench_scale.md` (default `<data_dir>/bench/scale`).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SIZES = (2000, 5000, 10000)
EVENTS_FOR_SIZE = 100_000
INTERFERENCE_BATCHES_PER_SEC = 10       # a browser sends one batch per 5 s
INTERFERENCE_CALLS = 300


def peak_rss_mb() -> Optional[float]:
    """This process's peak resident memory so far, in MB."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0), 1)
    except ImportError:                      # Windows
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 1048576.0, 1)
        except Exception:  # noqa: BLE001
            return None


def percentiles(ms: List[float]) -> Dict[str, Optional[float]]:
    if not ms:
        return {"n": 0, "p50_ms": None, "p95_ms": None}
    ordered = sorted(ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"n": len(ms), "p50_ms": round(st.median(ordered), 2), "p95_ms": round(p95, 2)}


def timed(fn: Callable, repeat: int) -> Dict:
    """p50 / p95 of `repeat` warm calls, with the first (cold) call apart."""
    t = time.perf_counter()
    fn()
    cold = (time.perf_counter() - t) * 1000.0
    ms = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1000.0)
    return {"cold_ms": round(cold, 2), **percentiles(ms)}


# ── One size (the child process) ──────────────────────────────────────────────

def _count(conn, sql: str, args=()) -> int:
    return int(conn.execute(sql, args).fetchone()[0])


def pad_candidates(db_path: Path, n_tracks: int, target_rows: int) -> int:
    """Grow mashup_candidates to about `target_rows` by cloning its rows onto
    other song ids (both sides shifted by the same amount, so a pair stays two
    different songs). Listing cost depends on the table's size and its score
    distribution, both of which this keeps; which songs a clone names is
    meaningless. Returns the row count afterwards."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")
                if r[1] != "id"]
        have = _count(conn, "SELECT COUNT(*) FROM mashup_candidates")
        if not have:
            return 0
        select = ", ".join(
            f"((({c} - 1 + :k) % :n) + 1)" if c in ("vocal_song_id", "inst_song_id") else c
            for c in cols)
        top = _count(conn, "SELECT MAX(id) FROM mashup_candidates")
        shift = 0
        while have < target_rows and shift < n_tracks - 1:
            shift += 1
            conn.execute(
                f"INSERT OR IGNORE INTO mashup_candidates ({', '.join(cols)}) "
                f"SELECT {select} FROM mashup_candidates WHERE id <= :top "
                f"LIMIT :room",
                {"k": shift, "n": n_tracks, "top": top, "room": target_rows - have})
            conn.commit()
            have = _count(conn, "SELECT COUNT(*) FROM mashup_candidates")
        return have
    finally:
        conn.close()


def events_bench(client, repeat: int) -> Dict:
    """POST /api/events: the round trip (the test client runs the background
    write before it returns, so this includes the write), the request-path
    half alone, and what 100,000 stored events weigh."""
    import sqlite3
    import config
    from database import models
    batch = [{"vocal_song_id": 1 + (i % 50), "inst_song_id": 60 + (i % 50),
              "vocal_section": 1002, "inst_section": 1004,
              "event": models.PAIR_EVENTS[i % len(models.PAIR_EVENTS)],
              "value": 12.5, "context": "dock", "scorer_version": "heuristic",
              "rank_shown": 1 + i} for i in range(20)]
    round_trip = timed(lambda: client.post("/api/events", json={"events": batch}), repeat)
    # The request-path half: the route function itself, with the write only
    # queued (a BackgroundTasks nobody runs) — what a caller waits for.
    from fastapi import BackgroundTasks
    from api.routes.events import post_events
    clean = timed(lambda: post_events(BackgroundTasks(), {"events": batch}), repeat * 10)
    write = timed(lambda: models.insert_pair_events(models.clean_pair_events(batch)),
                  repeat)

    # What ingestion costs OTHER requests: a small read and a small write,
    # timed alone and then while a thread stores event batches at
    # INTERFERENCE_BATCHES_PER_SEC — fifty times what one browser sends.
    import threading

    def _other(fn) -> List[float]:
        ms = []
        for _ in range(INTERFERENCE_CALLS):
            t = time.perf_counter()
            fn()
            ms.append((time.perf_counter() - t) * 1000.0)
            time.sleep(0.002)
        return ms

    def _read():
        models.get_pref("library_view")

    def _write():
        models.set_pref("bench_scale_probe", {"n": 1})

    alone = {"read": percentiles(_other(_read)), "write": percentiles(_other(_write))}
    stop = threading.Event()

    def _ingest():
        gap = 1.0 / INTERFERENCE_BATCHES_PER_SEC
        while not stop.is_set():
            client.post("/api/events", json={"events": batch})
            stop.wait(gap)
    worker = threading.Thread(target=_ingest, daemon=True)
    worker.start()
    try:
        busy = {"read": percentiles(_other(_read)), "write": percentiles(_other(_write))}
    finally:
        stop.set()
        worker.join(timeout=10)
    models.clear_pref("bench_scale_probe")
    interference = {
        "batches_per_sec": INTERFERENCE_BATCHES_PER_SEC, "alone": alone, "busy": busy,
        "added_p95_ms": {k: round(busy[k]["p95_ms"] - alone[k]["p95_ms"], 2)
                         for k in ("read", "write")}}

    def _bytes() -> int:
        conn = sqlite3.connect(config.DB_PATH)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return _count(conn, "PRAGMA page_count") * _count(conn, "PRAGMA page_size") \
                - _count(conn, "PRAGMA freelist_count") * _count(conn, "PRAGMA page_size")
        finally:
            conn.close()

    before = _bytes()
    rows = models.clean_pair_events(batch)
    conn = models.get_conn()
    try:
        many = [(1 + (i % 9000), 1 + ((i * 7) % 9000), 1000 + i % 8, 1000 + i % 6, *rows[i % 20][4:])
                for i in range(EVENTS_FOR_SIZE)]
        conn.executemany(
            "INSERT INTO pair_events (vocal_song_id, inst_song_id, vocal_section, "
            "inst_section, event, value, context, scorer_version, rank_shown) "
            "VALUES (?,?,?,?,?,?,?,?,?)", many)
        conn.commit()
    finally:
        conn.close()
    per_100k = (_bytes() - before) / 1e6
    t = time.perf_counter()
    folded = models.compact_pair_events(max_raw=1000)
    compact_ms = (time.perf_counter() - t) * 1000.0
    return {"batch_events": len(batch), "round_trip": round_trip,
            "request_path_clean": clean, "write": write,
            "interference": interference,
            "mb_per_100k_events": round(per_100k, 2),
            "compact_100k": {"ms": round(compact_ms, 1), **folded}}


def run_one(n_tracks: int, out: Path, seed: int, score_tracks: int,
            full_budget: float, repeat: int, say=print) -> Dict:
    """Generate and measure one library. Must run in a process whose MASHUP_*
    paths point at `out` (see `spawn`): config binds them at import."""
    import numpy as np
    import config
    assert Path(config.DB_PATH) == out / "mashup.db", \
        f"refusing to benchmark {config.DB_PATH}: not the synthetic library"
    from scripts_synth import generate           # noqa: E402 (loaded by _child)

    result: Dict = {"tracks": n_tracks, "seed": seed}
    say(f"[{n_tracks}] generating…")
    manifest = generate(out, n_tracks, seed, force=True)
    result["generate"] = {"seconds": manifest["seconds"],
                          "db_mb": round(manifest["db_bytes"] / 1e6, 1),
                          "kinds": manifest["kinds"],
                          "planted": len(manifest["planted"])}

    from database import models
    from matcher.match import score_all_pairs, score_song_pairs
    conn = models.get_conn()
    try:
        result["sections"] = {
            "essentia": _count(conn, "SELECT COUNT(*) FROM sections"),
            "allin1": _count(conn, "SELECT COUNT(*) FROM sections_allin1")}
    finally:
        conn.close()

    # ── per-track scoring ────────────────────────────────────────────────────
    rng = np.random.default_rng([seed, 30_000_001])
    sample = sorted(int(i) + 1 for i in rng.choice(n_tracks, size=min(score_tracks, n_tracks),
                                                   replace=False))
    secs = []
    for sid in sample:
        t = time.perf_counter()
        score_song_pairs(sid)
        secs.append(time.perf_counter() - t)
        say(f"[{n_tracks}] scored track {sid} in {secs[-1]:.1f} s")
    conn = models.get_conn()
    try:
        sampled_rows = _count(conn, "SELECT COUNT(*) FROM mashup_candidates")
    finally:
        conn.close()
    per_track = st.mean(secs) if secs else 0.0
    result["score_track"] = {
        "tracks": sample, "seconds": [round(s, 2) for s in secs],
        "mean_s": round(per_track, 2), "max_s": round(max(secs), 2) if secs else None,
        "rows_per_track": round(sampled_rows / max(1, len(sample)), 1),
        "peak_rss_mb": peak_rss_mb()}

    # ── Score library ────────────────────────────────────────────────────────
    estimate = per_track * n_tracks / 2.0
    # One track's cost depends heavily on what it is (an instrumental plays
    # one role, a tempo few records share passes the gate rarely), so the
    # estimate carries the sample's standard error.
    spread = (st.stdev(secs) / len(secs) ** 0.5 * n_tracks / 2.0) if len(secs) > 1 else None
    full: Dict = {"estimate_s": round(estimate, 1), "budget_s": full_budget,
                  "estimate_se_s": None if spread is None else round(spread, 1)}
    if estimate <= full_budget:
        say(f"[{n_tracks}] full Score library (estimated {estimate:.0f} s)…")
        before = peak_rss_mb()
        t = time.perf_counter()
        score_all_pairs()
        full.update({"measured": True, "seconds": round(time.perf_counter() - t, 1),
                     "peak_rss_mb": peak_rss_mb(), "rss_before_mb": before})
    else:
        full.update({"measured": False, "seconds": None, "peak_rss_mb": None,
                     "reason": f"estimated {_span(estimate)}, over the "
                               f"{full_budget:.0f} s budget"})
        target = int(min(config.MAX_CANDIDATE_ROWS,
                         result["score_track"]["rows_per_track"] * n_tracks / 2.0))
        full["padded_rows"] = pad_candidates(Path(config.DB_PATH), n_tracks, target)
    conn = models.get_conn()
    try:
        full["candidate_rows"] = _count(conn, "SELECT COUNT(*) FROM mashup_candidates")
    finally:
        conn.close()
    result["score_library"] = full

    # ── the dock listing ─────────────────────────────────────────────────────
    scoped_id = sample[0]
    dock = dict(combo_type="vocal_over_instrumental", limit=40, max_per_song=3)
    result["dock"] = {
        "library_wide": timed(lambda: models.get_candidates_enriched(**dock), repeat),
        "scoped": timed(lambda: models.get_candidates_enriched(
            vocal_song_id=scoped_id, **dock), repeat),
        "candidate_rows": full["candidate_rows"],
        "padded": not full["measured"]}

    # ── the API: Library payload, Segments, events ───────────────────────────
    from fastapi.testclient import TestClient
    from api.server import app
    client = TestClient(app)            # no `with`: startup hooks stay off

    body = {}

    def _tracks():
        body["tracks"] = client.get("/api/tracks").content
    lib = timed(_tracks, max(3, repeat // 4))
    lib.update({"mb": round(len(body["tracks"]) / 1e6, 2),
                "gzip_mb": round(len(gzip.compress(body["tracks"], 5)) / 1e6, 2)})
    result["library_payload"] = lib

    seg: Dict = {}
    seg_ids: Dict[str, List[int]] = {}
    for role in ("vocal", "bed"):
        def _list(role=role):
            body[role] = client.get(f"/api/segments?role={role}",
                                    headers={"accept-encoding": "identity"}).content
        t = timed(_list, max(3, repeat // 4))
        packed = json.loads(body[role])
        at = packed["columns"].index("id") if "id" in packed["columns"] else 0
        ids = [row[at] for row in packed["rows"]]
        n_matches = min(max(6, repeat // 3), len(ids))
        seg_ids[role] = [ids[int(i)] for i in rng.choice(len(ids), size=n_matches,
                                                         replace=False)] if ids else []
        t.update({"rows": len(ids), "mb": round(len(body[role]) / 1e6, 2),
                  "gzip_mb": round(len(gzip.compress(body[role], 5)) / 1e6, 2)})
        seg[f"list_{role}"] = t
    for role in ("vocal", "bed"):
        ms = []
        cold = None
        for k, seg_id in enumerate(seg_ids[role]):
            t = time.perf_counter()
            r = client.get(f"/api/segments/{seg_id}/matches?role={role}")
            took = (time.perf_counter() - t) * 1000.0
            assert r.status_code == 200, r.text[:200]
            if k == 0 and cold is None:
                cold = took                 # builds the library snapshot
            else:
                ms.append(took)
        seg[f"matches_{role}"] = {"cold_ms": None if cold is None else round(cold, 1),
                                  **percentiles(ms)}
    result["segments"] = seg

    result["events"] = events_bench(client, max(20, repeat))
    result["peak_rss_mb"] = peak_rss_mb()
    return result


def _child(args) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scripts_synth", ROOT / "scripts" / "synth_library.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts_synth"] = module
    spec.loader.exec_module(module)
    out = Path(args.out)
    result = run_one(args.one, out, args.seed, args.score_tracks,
                     args.full_score_budget, args.requests,
                     say=lambda s: print(s, file=sys.stderr, flush=True))
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


def spawn(n_tracks: int, out: Path, args) -> Dict:
    """Run one size in a child whose library paths are all under `out`."""
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"MASHUP_DB_PATH": str(out / "mashup.db"),
                "MASHUP_AUDIO_ROOT": str(out / "audio"),
                "MASHUP_DATA_DIR": str(out / "data"),
                "MASHUP_SETTINGS_DIR": str(out / "settings")})
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", str(n_tracks),
           "--out", str(out), "--seed", str(args.seed),
           "--score-tracks", str(args.score_tracks),
           "--full-score-budget", str(args.full_score_budget),
           "--requests", str(args.requests)]
    subprocess.run(cmd, env=env, check=True, cwd=str(ROOT))
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


# ── The report ────────────────────────────────────────────────────────────────

def _ms(d: Optional[dict]) -> str:
    if not d or d.get("p50_ms") is None:
        return "—"
    return f"{d['p50_ms']:.0f} / {d['p95_ms']:.0f}"


def _span(seconds: float) -> str:
    if seconds >= 5400:
        return f"{seconds / 3600.0:.1f} h"
    return f"{seconds / 60.0:.0f} min" if seconds >= 120 else f"{seconds:.0f} s"


def _mb(value: Optional[float]) -> str:
    return "not measured" if value is None else f"{value:.0f} MB"


def report(results: List[Dict]) -> str:
    head = "| Measure | " + " | ".join(f"{r['tracks']:,} tracks" for r in results) + " |"
    rule = "|---|" + "---|" * len(results)

    def row(label: str, cell: Callable[[Dict], str]) -> str:
        return f"| {label} | " + " | ".join(cell(r) for r in results) + " |"

    def full(r: Dict) -> str:
        f = r["score_library"]
        if f["measured"]:
            return f"{f['seconds']:.0f} s (measured)"
        se = f.get("estimate_se_s")
        pm = f" ± {_span(se)}" if se else ""
        return f"~{_span(f['estimate_s'])}{pm} (estimated, not run)"

    def full_rss(r: Dict) -> str:
        f = r["score_library"]
        return _mb(f["peak_rss_mb"]) if f["measured"] else "not measured"

    lines = [
        head, rule,
        row("Library on disk", lambda r: f"{r['generate']['db_mb']:.0f} MB"),
        row("Score one track (min / mean / max)",
            lambda r: f"{min(r['score_track']['seconds']):.1f} / "
                      f"{r['score_track']['mean_s']:.1f} / {r['score_track']['max_s']:.1f} s"),
        row("Score library, wall", full),
        row("Score library, peak RSS", full_rss),
        row("Candidate rows listed over",
            lambda r: f"{r['dock']['candidate_rows']:,}"
            + (" (padded)" if r["dock"]["padded"] else "")),
        row("Dock, library-wide p50 / p95 ms", lambda r: _ms(r["dock"]["library_wide"])),
        row("Dock, scoped p50 / p95 ms", lambda r: _ms(r["dock"]["scoped"])),
        row("GET /api/segments vocal, cold ms · MB (gzip)",
            lambda r: f"{r['segments']['list_vocal']['cold_ms']:.0f} · "
                      f"{r['segments']['list_vocal']['mb']} ({r['segments']['list_vocal']['gzip_mb']})"),
        row("GET /api/segments bed, cold ms · MB (gzip)",
            lambda r: f"{r['segments']['list_bed']['cold_ms']:.0f} · "
                      f"{r['segments']['list_bed']['mb']} ({r['segments']['list_bed']['gzip_mb']})"),
        row("GET /api/segments warm p50 / p95 ms (vocal)",
            lambda r: _ms(r["segments"]["list_vocal"])),
        row("Segment matches, vocal lead p50 / p95 ms",
            lambda r: _ms(r["segments"]["matches_vocal"])),
        row("Segment matches, bed lead p50 / p95 ms",
            lambda r: _ms(r["segments"]["matches_bed"])),
        row("Segment matches, first call ms (snapshot)",
            lambda r: f"{r['segments']['matches_vocal']['cold_ms']:.0f}"
            if r["segments"]["matches_vocal"]["cold_ms"] is not None else "—"),
        row("GET /api/tracks p50 / p95 ms", lambda r: _ms(r["library_payload"])),
        row("GET /api/tracks MB (gzip)",
            lambda r: f"{r['library_payload']['mb']} ({r['library_payload']['gzip_mb']})"),
        row("POST /api/events round trip p50 / p95 ms (20 events, write included)",
            lambda r: f"{r['events']['round_trip']['p50_ms']:.2f} / "
                      f"{r['events']['round_trip']['p95_ms']:.2f}"),
        row("POST /api/events request path p50 / p95 ms",
            lambda r: f"{r['events']['request_path_clean']['p50_ms']:.2f} / "
                      f"{r['events']['request_path_clean']['p95_ms']:.2f}"),
        row("Event write, off the request path, p50 / p95 ms",
            lambda r: f"{r['events']['write']['p50_ms']:.2f} / {r['events']['write']['p95_ms']:.2f}"),
        row("p95 added to another request while ingesting 10 batches/s (read · write) ms",
            lambda r: (lambda a: f"{a['read']:+.2f} · {a['write']:+.2f}")(
                r["events"]["interference"]["added_p95_ms"])
            if r["events"].get("interference") else "—"),
        row("DB per 100k events", lambda r: f"{r['events']['mb_per_100k_events']} MB"),
        row("Peak RSS, whole bench", lambda r: _mb(r["peak_rss_mb"])),
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", default=",".join(str(n) for n in DEFAULT_SIZES),
                    help="comma-separated library sizes")
    ap.add_argument("--out", default=None,
                    help="work folder (default <data_dir>/bench/scale)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score-tracks", type=int, default=8,
                    help="tracks sampled for per-track scoring")
    ap.add_argument("--full-score-budget", type=float, default=900.0,
                    help="run the full Score library only if estimated under this many seconds")
    ap.add_argument("--requests", type=int, default=30,
                    help="warm calls per latency measurement")
    ap.add_argument("--keep", action="store_true",
                    help="keep the synthetic libraries afterwards")
    ap.add_argument("--one", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.one is not None:
        return _child(args)

    if args.out:
        out = Path(args.out)
    else:
        import config
        out = Path(config.DATA_DIR) / "bench" / "scale"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in [int(x) for x in str(args.tracks).split(",") if x.strip()]:
        folder = out / f"synth_{n}"
        results.append(spawn(n, folder, args))
        if not args.keep:
            shutil.rmtree(folder, ignore_errors=True)
    table = report(results)
    meta = {"cpu_count": os.cpu_count(), "python": sys.version.split()[0],
            "platform": sys.platform, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "bench_scale.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    (out / "bench_scale.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nWritten to {out / 'bench_scale.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
