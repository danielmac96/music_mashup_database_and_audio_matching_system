#!/usr/bin/env python
"""scripts/sim_learning.py — does the learning loop beat the rules for a
listener the rules do not know? (readme §5.8, criterion 2.1)

A synthetic user judges pairs on a scored synthetic library
(scripts/synth_library.py) through the real dock route — exploration slots
included — and the real learning loop retrains as it goes:

  * the user's taste: Drop-ins (matcher/compat.dropin, as the card tags them)
    plus one hidden taste the rules do not have — they love a **breakdown**
    bed — plus a little of the rule's own score, and noise on every verdict;
  * each judgement: a random vocal from the training half, its scoped dock
    of vocal pairs (`GET /api/mashups?vocal_song_id=…`, the default order),
    and one pair from its top ten the user has not judged, picked with a mild
    preference for the top (weight 1/√rank) — so an exploration slot is
    judged about as often as a listener would. Every fetch logs `shown` for
    the top ten; the judged pair is looped (25 s when liked, a 1.5 s skip when
    not) and opened in Studio when loved — so the weak labels are exercised;
  * at every checkpoint the loop trains (learning.train: grouped CV, the
    shadow gate, the blend over the stored table when it activates), then
    NDCG@10 is measured on the **held-out** vocals — never judged — for the
    stored blended order (score_total) against the rule's (score_rule), each
    vocal's partners one row per bed song, gains from the user's noiseless
    utility.

    python scripts/sim_learning.py --tracks 500 --judgements 200 --every 20

Writes a JSON report to `<out>/sim_learning.json`. The library is built in
`<out>` (or copied from `--library`), never the live one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

W_DROPIN, W_BREAKDOWN, W_RULE = 1.0, 1.0, 0.5
NOISE = 0.15
# A Drop-in alone, or a breakdown bed alone, is "ok"; both is "love"; neither
# is "no" — so the dock's top (mostly Drop-ins) is not all loved.
LOVE_AT, OK_AT = 1.6, 0.9


def _bind(lib: Path) -> None:
    os.environ.update(MASHUP_DB_PATH=str(lib / "mashup.db"),
                      MASHUP_AUDIO_ROOT=str(lib / "audio"),
                      MASHUP_DATA_DIR=str(lib / "data"),
                      MASHUP_SETTINGS_DIR=str(lib / "settings"),
                      MASHUP_AUTO_RETRAIN="0")


def utility(row: dict, dropin: bool) -> float:
    """The user's noiseless taste for one pairing."""
    breakdown = (row.get("inst_section_label") or "") == "breakdown"
    rule = row.get("score_rule")
    rule = row.get("score_total") if rule is None else rule
    return W_DROPIN * dropin + W_BREAKDOWN * breakdown + W_RULE * float(rule or 0.0)


def ndcg(gains_in_order, ideal, k=10) -> float:
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains_in_order[:k]))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--judgements", type=int, default=200)
    ap.add_argument("--every", type=int, default=20, help="judgements per checkpoint")
    ap.add_argument("--noise", type=float, default=NOISE,
                    help="sd of the noise on each verdict (a less consistent listener)")
    ap.add_argument("--out", default=None, help="folder for the library and report")
    ap.add_argument("--library", default=None,
                    help="an already scored synthetic library to copy instead of building one")
    args = ap.parse_args(argv)

    out = Path(args.out or (ROOT / "bench" / "sim_learning"))
    lib = out / "library"
    if lib.exists():
        shutil.rmtree(lib)
    out.mkdir(parents=True, exist_ok=True)
    if args.library:
        shutil.copytree(args.library, lib)
    _bind(lib)
    if args.library:
        # A clean slate on the copy: no verdicts, events or models, the rule's totals.
        import sqlite3
        c = sqlite3.connect(lib / "mashup.db")
        for t in ("pair_feedback", "pair_events", "pair_event_summary", "models"):
            c.execute(f"DELETE FROM {t}")
        c.execute("DELETE FROM app_prefs WHERE key='learning_state'")
        c.execute("UPDATE mashup_candidates SET score_total = score_rule, model_prob = NULL, "
                  "model_version = NULL WHERE score_rule IS NOT NULL")
        c.commit()
        c.close()
    t0 = time.time()
    if not args.library:
        from scripts.synth_library import generate
        generate(lib, args.tracks, args.seed, force=True)
        from matcher.match import score_all_pairs
        score_all_pairs()
    print(f"library ready in {time.time() - t0:.0f} s", flush=True)

    from api.routes.mashups import list_candidates
    from database.models import get_conn, insert_pair_events, upsert_pair_feedback
    from matcher import learning
    from matcher.compat import dropin, slot_floor
    from matcher.recipe import bed_shift, harmony_unsure_below

    floor, unsure = slot_floor(), harmony_unsure_below()

    def is_dropin(row):
        return bool(dropin({**row, "semitone_shift": bed_shift(row, unsure)}, floor)[0])

    conn = get_conn()
    vocals = [r[0] for r in conn.execute(
        "SELECT vocal_song_id FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental' GROUP BY vocal_song_id "
        "HAVING COUNT(DISTINCT inst_song_id) >= 15 ORDER BY vocal_song_id")]
    conn.close()
    rng = random.Random(args.seed)
    rng.shuffle(vocals)
    held = set(vocals[: len(vocals) * 3 // 10])
    train_vocals = [v for v in vocals if v not in held]

    def evaluate():
        cols = learning._row_columns()
        conn = get_conn()
        try:
            by_vocal = {}
            marks = ",".join("?" * len(held))
            for r in conn.execute(
                    f"SELECT {', '.join(cols)} FROM mashup_candidates "
                    f"WHERE combo_type='vocal_over_instrumental' "
                    f"AND vocal_song_id IN ({marks}) ORDER BY id", sorted(held)):
                d = dict(zip(cols, r))
                by_vocal.setdefault(d["vocal_song_id"], []).append(d)
        finally:
            conn.close()
        rule_n, blend_n = [], []
        for rows in by_vocal.values():
            for d in rows:
                d["_g"] = max(0.0, utility(d, is_dropin(d)))

            def best_per_bed(key):
                best = {}
                for d in sorted(rows, key=lambda d: -(d[key] if d[key] is not None
                                                       else d["score_total"])):
                    best.setdefault(d["inst_song_id"], d)
                return list(best.values())
            # The ideal: each bed song at its best pairing for this listener.
            top_g = {}
            for d in rows:
                top_g[d["inst_song_id"]] = max(top_g.get(d["inst_song_id"], 0.0), d["_g"])
            ideal = list(top_g.values())
            rule_n.append(ndcg([d["_g"] for d in best_per_bed("score_rule")], ideal))
            blend_n.append(ndcg([d["_g"] for d in best_per_bed("score_total")], ideal))
        return (sum(rule_n) / len(rule_n), sum(blend_n) / len(blend_n))

    judged = set()
    checkpoints = []
    stars_given = {1: 0, 3: 0, 5: 0}
    explored = 0
    for n in range(1, args.judgements + 1):
        for _attempt in range(20):
            v = rng.choice(train_vocals)
            page = list_candidates(vocal_song_id=v, limit=10,
                                   combo_type="vocal_over_instrumental")["candidates"][:10]
            events = [("shown", r, k + 1) for k, r in enumerate(page)]
            open_ = [(k, r) for k, r in enumerate(page)
                     if (r["vocal_song_id"], r["inst_song_id"], r.get("vocal_section_idx"),
                         r.get("inst_section_idx")) not in judged]
            if open_:
                pick = rng.choices([r for _k, r in open_],
                                   weights=[1.0 / math.sqrt(k + 1) for k, _r in open_])[0]
                break
        explored += bool(pick.get("explore"))
        u = utility(pick, bool(pick.get("dropin"))) + rng.gauss(0.0, args.noise)
        stars = 5 if u >= LOVE_AT else 3 if u >= OK_AT else 1
        stars_given[stars] += 1
        key = (pick["vocal_song_id"], pick["inst_song_id"],
               pick.get("vocal_section_idx"), pick.get("inst_section_idx"))
        judged.add(key)
        upsert_pair_feedback(key[0], key[1], None, vocal_section=key[2],
                             inst_section=key[3], rating=stars)
        events.append(("loop_play", pick, None) if stars >= 3 else ("skip", pick, None))
        if stars == 5:
            events.append(("open_studio", pick, None))
        insert_pair_events([(r["vocal_song_id"], r["inst_song_id"],
                             r.get("vocal_section_idx"), r.get("inst_section_idx"),
                             ev, (25.0 if ev == "loop_play" else 1.5 if ev == "skip" else None),
                             "dock", None, rank)
                            for ev, r, rank in events])
        if n % args.every == 0:
            t = time.time()
            res = learning.train(reason="simulation")
            rule_ndcg, blend_ndcg = evaluate()
            cp = {"judgements": n, "rule_ndcg10": round(rule_ndcg, 4),
                  "blended_ndcg10": round(blend_ndcg, 4),
                  "gain": round(blend_ndcg - rule_ndcg, 4),
                  "alpha": learning.status()["alpha"],
                  "decision": res.get("decision") or res.get("reason"),
                  "auc_strong": (res.get("metrics") or {}).get("auc_strong"),
                  "rule_auc": (res.get("metrics") or {}).get("rule_auc"),
                  "train_secs": res.get("secs"), "checkpoint_secs": round(time.time() - t, 1)}
            checkpoints.append(cp)
            print(json.dumps(cp), flush=True)

    after60 = [c for c in checkpoints if c["judgements"] >= 60]
    report = {
        "tracks": args.tracks, "seed": args.seed, "judgements": args.judgements,
        "held_out_vocals": len(held), "train_vocals": len(train_vocals),
        "stars": stars_given, "explored_judged": explored,
        "taste": {"dropin": W_DROPIN, "breakdown_bed": W_BREAKDOWN, "rule": W_RULE,
                  "noise": args.noise},
        "checkpoints": checkpoints,
        "criterion": {
            "gain_at_60_ge_0.05": bool(after60) and all(c["gain"] >= 0.05 for c in after60),
            "never_below_rule": all(c["gain"] >= 0 for c in checkpoints),
        },
        "seconds": round(time.time() - t0, 1),
    }
    (out / "sim_learning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["criterion"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
