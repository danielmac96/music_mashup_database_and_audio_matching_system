#!/usr/bin/env python
"""scripts/sim_import.py — how soon an import yields its first USEFUL pair
(readme §2, §8; criterion 5.1).

On a COPY of a scored synthetic library (scripts/synth_library.py, then Score
library — the Drop-in slot floor is the full run's), `--imports` tracks are
treated as one batch arriving at once. Nothing is separated or analysed for
real; the pipeline's queues are replayed as a discrete-event simulation on
durations read from the library's own `analysis_runs` medians (ms per audio
second, per stage; stages it has no rows for fall back to the readme's
measured references), with the scoring time of every track MEASURED by
running the real scorer at each of the three states a track passes through:

  1. after its quick tier — its mix standing in for both stems, the stems and
     allin1 rows set aside, its sections provisional: `score_song_pairs`
     writes the provisional rows, and the production
     `api.workers.stem_priority.stem_promise` / `priority_for` read them, as
     the stems queue would;
  2. after its first cut — stems back, allin1 still set aside: the real
     scorer's rows say which pairs are Drop-ins at that point
     (`compat.dropin_sql`, the card's rule in SQL);
  3. after allin1 — everything back: the Drop-ins once refined.

Then each arrival order is replayed twice on the same durations — Phase 0's
ordering (every import at PRIORITY_INGEST: arrival order) and the
partner-aware one (the priority stem_priority gave after the quick tier) —
and the time to the first Drop-in pair is the earliest moment both its songs
are scored in a state where the pair is one (a library track is ready at 0).

    python scripts/sim_import.py --library /path/to/scored5k --imports 50 \\
        --orders 20 --out /tmp/sim_import

Writes `<out>/sim_import.json` and `<out>/sim_import.md`. The copy is bound
through MASHUP_* paths; the source library is never touched.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ms per audio second where the library has no analysis_runs rows for a stage:
# the readme's measurements (§5.4: quick ~8.7 s on a 220 s song; §5.3 Demucs
# 2–3 min a track; §5.5 allin1 fold0 ≈ ⅛ of the ensemble's ~525 s inference).
REFERENCE_MS_PER_SEC = {"download": 25.0, "quick": 40.0, "stems": 620.0,
                        "analysis": 54.0, "structure": 14.0, "allin1": 300.0}
STAGES = tuple(REFERENCE_MS_PER_SEC)


def _bind(lib: Path) -> None:
    """Point config at the copy BEFORE anything from the repo is imported."""
    os.environ["MASHUP_DB_PATH"] = str(lib / "mashup.db")
    os.environ["MASHUP_AUDIO_ROOT"] = str(lib / "audio")
    os.environ["MASHUP_DATA_DIR"] = str(lib / "data")
    os.environ["MASHUP_SETTINGS_DIR"] = str(lib / "settings")


# ── States ────────────────────────────────────────────────────────────────────

_SET_ASIDE = {
    "allin1": ("sections_allin1", "allin1_analysis"),
    "stems": ("stems",),
}


def _set_aside(conn, ids, what: str) -> None:
    """Move these tracks' rows of a kind into _sim_* tables (idempotent)."""
    q = ",".join("?" * len(ids))
    tables = list(_SET_ASIDE.get(what, ()))
    for table in tables:
        conn.execute(f"CREATE TABLE IF NOT EXISTS _sim_{table} AS "
                     f"SELECT * FROM {table} WHERE 0")
        conn.execute(f"INSERT INTO _sim_{table} SELECT * FROM {table} "
                     f"WHERE song_id IN ({q})", ids)
        conn.execute(f"DELETE FROM {table} WHERE song_id IN ({q})", ids)
    if what == "stems":
        conn.execute("CREATE TABLE IF NOT EXISTS _sim_features AS "
                     "SELECT * FROM features WHERE 0")
        conn.execute("INSERT INTO _sim_features SELECT * FROM features WHERE "
                     f"stem_type IN ('vocals','instrumental') AND song_id IN ({q})", ids)
        conn.execute("DELETE FROM features WHERE stem_type IN ('vocals','instrumental') "
                     f"AND song_id IN ({q})", ids)
        conn.execute("CREATE TABLE IF NOT EXISTS _sim_sections AS "
                     "SELECT id, provisional, section_class FROM sections WHERE 0")
        conn.execute("INSERT INTO _sim_sections SELECT id, provisional, section_class "
                     f"FROM sections WHERE song_id IN ({q})", ids)
        conn.execute("UPDATE sections SET provisional=1, section_class='unknown' "
                     f"WHERE song_id IN ({q})", ids)
        conn.execute(f"UPDATE songs SET status='downloaded', quick_state='done' "
                     f"WHERE id IN ({q})", ids)
    conn.commit()


def _restore(conn, what: str) -> None:
    for table in _SET_ASIDE.get(what, ()):
        conn.execute(f"INSERT INTO {table} SELECT * FROM _sim_{table}")
        conn.execute(f"DROP TABLE _sim_{table}")
    if what == "stems":
        conn.execute("INSERT INTO features SELECT * FROM _sim_features")
        conn.execute("DROP TABLE _sim_features")
        conn.execute("UPDATE sections SET provisional=(SELECT b.provisional FROM "
                     "_sim_sections b WHERE b.id=sections.id), section_class=(SELECT "
                     "b.section_class FROM _sim_sections b WHERE b.id=sections.id) "
                     "WHERE id IN (SELECT id FROM _sim_sections)")
        conn.execute("DROP TABLE _sim_sections")
        conn.execute("UPDATE songs SET status='analysed' WHERE id IN "
                     "(SELECT DISTINCT song_id FROM stems)")
    conn.commit()


def _dropins_of(conn, song_id: int, floor, unsure) -> set:
    from matcher.compat import dropin_sql
    expr = dropin_sql(floor, "mc")
    rows = conn.execute(
        f"SELECT vocal_song_id, inst_song_id FROM mashup_candidates mc "
        f"WHERE combo_type='vocal_over_instrumental' AND {expr} = 1 "
        f"AND (vocal_song_id=? OR inst_song_id=?)", (unsure, song_id, song_id)).fetchall()
    return {(int(a), int(b)) for a, b in rows}


def measure_states(imports: list, log=print) -> dict:
    """Run the real scorer for every import at its three states. Returns
    per-track promise, priorities and score seconds, and the Drop-in pairs
    at first cut and after allin1."""
    from api.workers.stem_priority import priority_for, stem_promise
    from database.models import get_conn
    from matcher.compat import slot_floor
    from matcher.match import score_song_pairs
    from matcher.recipe import harmony_unsure_below

    floor, unsure = slot_floor(), harmony_unsure_below()
    if floor is None:
        raise SystemExit("no Drop-in slot floor: run Score library on the library first")
    conn = get_conn()
    out = {"floor": floor, "promise": {}, "priority": {}, "score_secs": {},
           "dropins_first": set(), "dropins_refined": set()}

    def _score_all(state: str, collect=None) -> None:
        secs = []
        for n, sid in enumerate(imports, 1):
            t = time.perf_counter()
            score_song_pairs(sid)
            dt = time.perf_counter() - t
            secs.append(dt)
            out["score_secs"].setdefault(state, {})[sid] = round(dt, 3)
            if collect is not None:
                collect |= _dropins_of(conn, sid, floor, unsure)
            if n % 10 == 0:
                log(f"  {state}: {n}/{len(imports)} scored "
                    f"(median {statistics.median(secs):.1f} s)")

    # 1. After the quick tier.
    _set_aside(conn, imports, "allin1")
    _set_aside(conn, imports, "stems")
    _score_all("provisional")
    for sid in imports:
        promise = stem_promise(sid)
        out["promise"][sid] = promise
        out["priority"][sid] = priority_for(promise)
    # 2. After the first cut (stems back, allin1 still aside).
    _restore(conn, "stems")
    _score_all("first_cut", out["dropins_first"])
    # 3. After allin1.
    _restore(conn, "allin1")
    _score_all("refined", out["dropins_refined"])
    conn.close()
    return out


# ── The queues, replayed ──────────────────────────────────────────────────────

def stage_durations(imports: list) -> dict:
    """{stage: {song_id: seconds}} from the library's analysis_runs medians
    (ms per audio second), the reference where a stage has no rows."""
    from database.models import get_conn
    conn = get_conn()
    try:
        rate = {}
        for stage in STAGES:
            vals = [r[0] for r in conn.execute(
                "SELECT ms / audio_secs FROM analysis_runs WHERE grp=? AND ok=1 "
                "AND audio_secs > 0", (stage,))]
            rate[stage] = (statistics.median(vals), "analysis_runs") if vals \
                else (REFERENCE_MS_PER_SEC[stage], "readme reference")
        q = ",".join("?" * len(imports))
        secs = {int(r[0]): float(r[1] or 220.0) for r in conn.execute(
            f"SELECT id, duration_secs FROM songs WHERE id IN ({q})", imports)}
    finally:
        conn.close()
    durs = {stage: {sid: rate[stage][0] * secs[sid] / 1000.0 for sid in imports}
            for stage in STAGES}
    return {"durations": durs,
            "rates": {s: {"ms_per_sec": round(r, 2), "source": src}
                      for s, (r, src) in rate.items()}}


def simulate(order: list, durs: dict, score_secs: dict, priority: dict,
             workers: dict) -> tuple:
    """Replay one batch through the per-stage queues (api/queue_runner.py):
    a priority heap per stage, (priority, arrival) order; Demucs and allin1
    share the stems gate, analysis and structure share the analysis gate; a
    backfill allin1 waits while separations are queued; a track is scored on
    the quick worker after its quick tier, on the structure worker after its
    first cut, and on the allin1 worker after allin1. Returns
    ({song_id: first-cut ready time}, {song_id: refined ready time})."""
    from config import PRIORITY_BACKFILL, PRIORITY_INGEST
    gates = {"stems": 1, "A": workers["analysis"], "quick": workers["quick"],
             "download": workers["download"]}
    gate_of = {"download": "download", "quick": "quick", "stems": "stems",
               "analysis": "A", "structure": "A", "allin1": "stems"}
    busy = {s: 0 for s in workers}
    gate_free = dict(gates)
    queues = {s: [] for s in workers}
    seq = iter(range(10 ** 9))
    prio = {sid: PRIORITY_INGEST for sid in order}
    first, refined = {}, {}
    events = []          # (time, seq, kind, payload)
    t = 0.0

    def put(stage, sid, p):
        heapq.heappush(queues[stage], (p, next(seq), sid))

    for sid in order:
        put("download", sid, PRIORITY_INGEST)

    def phases(stage, sid):
        """[(seconds, holds the gate?)] of one stage run on its worker."""
        if stage == "quick":
            return [(durs["quick"][sid], True), (score_secs["provisional"][sid], False)]
        if stage == "structure":
            return [(durs["structure"][sid], True), (score_secs["first_cut"][sid], False)]
        if stage == "allin1":
            return [(durs["allin1"][sid], True), (score_secs["refined"][sid], False)]
        return [(durs[stage][sid], True)]

    nxt = {"download": "quick", "quick": "stems", "stems": "analysis",
           "analysis": "structure"}

    def try_start():
        started = True
        while started:
            started = False
            for stage in ("download", "quick", "stems", "analysis", "structure", "allin1"):
                q = queues[stage]
                if not q or busy[stage] >= workers[stage]:
                    continue
                g = gate_of[stage]
                if gate_free[g] <= 0:
                    continue
                if stage == "allin1" and queues["stems"]:
                    continue      # yield_to_stems: pairs first
                p, _s, sid = heapq.heappop(q)
                busy[stage] += 1
                gate_free[g] -= 1
                ph = phases(stage, sid)
                gated = ph[0][0]
                rest = sum(d for d, _ in ph[1:])
                heapq.heappush(events, (t + gated, next(seq), "ungate", (stage, sid)))
                heapq.heappush(events, (t + gated + rest, next(seq), "done", (stage, sid)))
                started = True

    try_start()
    while events:
        t, _s, kind, (stage, sid) = heapq.heappop(events)
        if kind == "ungate":
            gate_free[gate_of[stage]] += 1
        else:
            busy[stage] -= 1
            if stage == "quick":
                prio[sid] = priority[sid]          # stem_priority on the way in
            if stage == "structure":
                first[sid] = t
                put("allin1", sid, PRIORITY_BACKFILL)
            elif stage == "allin1":
                refined[sid] = t
            else:
                put(nxt[stage], sid, prio[sid])
        try_start()
    return first, refined


def first_dropins(pairs_first: set, pairs_refined: set, imports: set,
                  first: dict, refined: dict, k: int = 5) -> list:
    """The times of the first k Drop-in pairs to become visible."""
    def ready(pair, when):
        return max((when[s] for s in pair if s in imports), default=0.0)
    times = {}
    for pair in pairs_first:
        times[pair] = ready(pair, first)
    for pair in pairs_refined:
        t = ready(pair, refined)
        times[pair] = min(times.get(pair, t), t)
    return sorted(times.values())[:k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True, help="a scored synthetic library folder")
    ap.add_argument("--imports", type=int, default=50)
    ap.add_argument("--orders", type=int, default=20, help="arrival orders replayed")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse <out>/states.json instead of re-scoring")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = out / "library"
    states_file = out / "states.json"
    if not (args.reuse and states_file.exists() and lib.exists()):
        if lib.exists():
            shutil.rmtree(lib)
        shutil.copytree(args.library, lib)
    _bind(lib)

    from config import (ANALYSIS_WORKERS, ALLIN1_WORKERS, DOWNLOAD_WORKERS,
                        PRIORITY_INGEST, QUICK_WORKERS, STEM_WORKERS)
    from database.models import get_conn

    conn = get_conn()
    all_ids = [r[0] for r in conn.execute("SELECT id FROM songs ORDER BY id")]
    conn.close()
    rng = random.Random(args.seed)
    imports = sorted(rng.sample(all_ids, args.imports))

    if args.reuse and states_file.exists():
        raw = json.loads(states_file.read_text(encoding="utf-8"))
        imports = raw["imports"]
        states = {"floor": raw["floor"],
                  "promise": {int(k): v for k, v in raw["promise"].items()},
                  "priority": {int(k): v for k, v in raw["priority"].items()},
                  "score_secs": {s: {int(k): v for k, v in d.items()}
                                 for s, d in raw["score_secs"].items()},
                  "dropins_first": {tuple(p) for p in raw["dropins_first"]},
                  "dropins_refined": {tuple(p) for p in raw["dropins_refined"]}}
    else:
        t0 = time.time()
        states = measure_states(imports)
        states_file.write_text(json.dumps({
            "imports": imports, "floor": states["floor"],
            "promise": states["promise"], "priority": states["priority"],
            "score_secs": states["score_secs"],
            "dropins_first": sorted(states["dropins_first"]),
            "dropins_refined": sorted(states["dropins_refined"]),
            "measure_secs": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")

    d = stage_durations(imports)
    workers = {"download": DOWNLOAD_WORKERS, "quick": QUICK_WORKERS,
               "stems": STEM_WORKERS, "analysis": ANALYSIS_WORKERS,
               "structure": ANALYSIS_WORKERS, "allin1": ALLIN1_WORKERS}
    fifo_prio = {sid: PRIORITY_INGEST for sid in imports}
    imp = set(imports)
    runs = []
    for i in range(args.orders):
        order = list(imports)
        random.Random(1000 + i).shuffle(order)
        res = {}
        for name, prio in (("phase0", fifo_prio), ("partner_aware", states["priority"])):
            first, refined = simulate(order, d["durations"], states["score_secs"],
                                      prio, workers)
            res[name] = first_dropins(states["dropins_first"], states["dropins_refined"],
                                      imp, first, refined)
            res[name + "_batch_done"] = max(refined.values())
        a, b = res["phase0"], res["partner_aware"]
        res["improvement_first"] = (1 - b[0] / a[0]) if a and b and a[0] > 0 else None
        res["improvement_fifth"] = (1 - b[-1] / a[-1]) if len(a) >= 5 and len(b) >= 5 else None
        runs.append(res)

    def _med(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    promise_hist = {}
    for sid in imports:
        promise_hist[states["priority"][sid]] = promise_hist.get(states["priority"][sid], 0) + 1
    with_dropin = {s for p in states["dropins_first"] | states["dropins_refined"]
                   for s in p if s in imp}
    summary = {
        "library_tracks": len(all_ids), "imports": len(imports),
        "orders": args.orders, "floor": states["floor"],
        "rates": d["rates"], "workers": workers,
        "score_secs_median": {s: round(statistics.median(v.values()), 2)
                              for s, v in states["score_secs"].items()},
        "dropin_pairs": {"first_cut": len(states["dropins_first"]),
                         "refined": len(states["dropins_refined"]),
                         "imports_with_one": len(with_dropin)},
        "priority_histogram": promise_hist,
        "promised_and_delivered": sum(1 for s in imports if states["priority"][s] < PRIORITY_INGEST
                                      and s in with_dropin),
        "promised": sum(1 for s in imports if states["priority"][s] < PRIORITY_INGEST),
        "first_dropin_secs_median": {"phase0": round(statistics.median(r["phase0"][0] for r in runs), 1),
                                     "partner_aware": round(statistics.median(
                                         r["partner_aware"][0] for r in runs), 1)},
        "fifth_dropin_secs_median": {"phase0": round(statistics.median(r["phase0"][-1] for r in runs), 1),
                                     "partner_aware": round(statistics.median(
                                         r["partner_aware"][-1] for r in runs), 1)},
        "improvement_first_median": _med("improvement_first"),
        "improvement_first_worst": round(min(r["improvement_first"] for r in runs), 4),
        "improvement_fifth_median": _med("improvement_fifth"),
        "batch_done_secs_median": {"phase0": round(statistics.median(r["phase0_batch_done"] for r in runs), 1),
                                   "partner_aware": round(statistics.median(
                                       r["partner_aware_batch_done"] for r in runs), 1)},
        "runs": runs,
    }
    (out / "sim_import.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    md = [f"# Simulated import: {len(imports)} tracks into {len(all_ids):,}",
          "", f"Arrival orders replayed: {args.orders}; Drop-in slot floor {states['floor']}.",
          "", "| | Phase 0 (arrival) | Partner-aware |", "|---|---|---|",
          f"| First Drop-in pair, median | {summary['first_dropin_secs_median']['phase0']} s "
          f"| {summary['first_dropin_secs_median']['partner_aware']} s |",
          f"| Fifth Drop-in pair, median | {summary['fifth_dropin_secs_median']['phase0']} s "
          f"| {summary['fifth_dropin_secs_median']['partner_aware']} s |",
          f"| Whole batch refined, median | {summary['batch_done_secs_median']['phase0']} s "
          f"| {summary['batch_done_secs_median']['partner_aware']} s |",
          "", f"Improvement to the first Drop-in: median {summary['improvement_first_median']}, "
          f"worst {summary['improvement_first_worst']}; to the fifth: median "
          f"{summary['improvement_fifth_median']}.",
          f"Promoted: {summary['promised']} of {len(imports)} "
          f"({summary['promised_and_delivered']} of them have a Drop-in); "
          f"imports with any Drop-in: {len(with_dropin)}."]
    (out / "sim_import.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
