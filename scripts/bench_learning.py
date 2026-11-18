#!/usr/bin/env python
"""scripts/bench_learning.py — what a retrain costs (readme §5.8, §8;
criterion 2.3).

On a COPY of a scored synthetic library (scripts/synth_library.py, then
Score library), plants `--strong` verdicts and `--weak` pair events on
stored pairings (a seeded synthetic taste: breakdown beds liked, plus
a coin), then times:

  * `learning.train` — labels, features, grouped cross-validation (the
    ranker and the logistic baseline), calibration, the shadow gate; and
  * `learning.blend_table` — re-blending every stored row with the result
    (it runs only when the model is activated, so it is timed on its own).

    python scripts/bench_learning.py --library /path/to/scored10k \\
        --strong 500 --weak 20000 --out /tmp/bench_learning

The copy is bound through MASHUP_* paths; the live library is never touched.
Writes `<out>/bench_learning.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# One event per weak pairing, each of a kind that yields a weak label (a
# `shown` in the top five, a loop of 20 s or more, a skip, a build, a hide):
# `--weak` is the number of weak LABELS, as criterion 2.3 counts them.
EVENTS = (("shown", 0.50), ("loop_play", 0.18), ("skip", 0.14),
          ("open_studio", 0.06), ("render", 0.03), ("export", 0.05), ("hide", 0.04))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True, help="a scored synthetic library folder")
    ap.add_argument("--strong", type=int, default=500)
    ap.add_argument("--weak", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-blend", action="store_true", help="time the retrain only")
    args = ap.parse_args(argv)

    out = Path(args.out)
    lib = out / "library"
    if lib.exists():
        shutil.rmtree(lib)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    shutil.copytree(args.library, lib)
    copy_secs = time.time() - t
    os.environ.update(MASHUP_DB_PATH=str(lib / "mashup.db"),
                      MASHUP_AUDIO_ROOT=str(lib / "audio"),
                      MASHUP_DATA_DIR=str(lib / "data"),
                      MASHUP_SETTINGS_DIR=str(lib / "settings"),
                      MASHUP_AUTO_RETRAIN="0")

    from database.models import get_conn
    from matcher import learning

    rng = random.Random(args.seed)
    conn = get_conn()
    for table in ("pair_feedback", "pair_events", "pair_event_summary", "models"):
        conn.execute(f"DELETE FROM {table}")
    conn.execute("DELETE FROM app_prefs WHERE key='learning_state'")
    rows = conn.execute(
        "SELECT vocal_song_id, inst_song_id, vocal_section_idx, inst_section_idx, "
        "inst_section_label, dropin_mask FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchall()
    n_rows = len(rows)
    picks = rng.sample(range(n_rows), min(n_rows, args.strong + args.weak))
    strong, weak = picks[:args.strong], picks[args.strong:]
    fb = []
    for j in strong:
        r = rows[j]
        good = (r[4] == "breakdown") + (rng.random() < 0.35)
        verdict = "love" if good >= 2 else "ok" if good == 1 else "no"
        fb.append((r[0], r[1], r[2], r[3], verdict, {"love": 5, "ok": 3, "no": 1}[verdict]))
    conn.executemany(
        "INSERT INTO pair_feedback (vocal_song_id, inst_song_id, vocal_section, "
        "inst_section, verdict, rating) VALUES (?,?,?,?,?,?)", fb)
    names = [e for e, _w in EVENTS]
    weights = [w for _e, w in EVENTS]
    ev = []
    for j in weak:
        r = rows[j]
        e = rng.choices(names, weights)[0]
        value = (rng.uniform(20, 90) if e == "loop_play" else rng.uniform(0.2, 2.9)
                 if e == "skip" else None)
        ev.append((r[0], r[1], r[2], r[3], e, value, "dock", "heuristic",
                   rng.randint(1, 5) if e == "shown" else None))
    conn.executemany(
        "INSERT INTO pair_events (vocal_song_id, inst_song_id, vocal_section, "
        "inst_section, event, value, context, scorer_version, rank_shown) "
        "VALUES (?,?,?,?,?,?,?,?,?)", ev)
    conn.commit()
    tracks = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    conn.close()

    try:
        import psutil
        proc = psutil.Process()
    except Exception:  # noqa: BLE001
        proc = None
    t = time.time()
    res = learning.train(reason="benchmark", activate=False)
    train_secs = time.time() - t
    report = {
        "tracks": tracks, "stored_rows": n_rows, "strong": args.strong,
        "weak_events": args.weak, "cpus": os.cpu_count(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "copy_secs": round(copy_secs, 1),
        "train_secs": round(train_secs, 2),
        "features_secs": (res.get("metrics") or {}).get("secs_features"),
        "train_counts": (res.get("metrics") or {}).get("counts"),
        "rows_fitted": (res.get("metrics") or {}).get("n_rows"),
        "auc_strong": (res.get("metrics") or {}).get("auc_strong"),
        "rule_auc": (res.get("metrics") or {}).get("rule_auc"),
        "logreg_auc": (res.get("metrics") or {}).get("logreg_auc"),
        "decision": res.get("decision") or res.get("reason"),
    }
    if not args.no_blend and res.get("model_id"):
        t = time.time()
        learning.activate_model(res["model_id"], reblend=False)
        blended = learning.blend_table()
        report["blend_secs"] = round(time.time() - t, 1)
        report["blend_rows"] = blended.get("rows")
    if proc is not None:
        report["rss_mb"] = round(proc.memory_info().rss / 1e6)
    (out / "bench_learning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
