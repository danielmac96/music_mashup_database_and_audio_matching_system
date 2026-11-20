#!/usr/bin/env python
"""scripts/timing_gate_share.py — what share of a library's section pairings
would pass the timing gate at 25 / 40 / 60 ms (readme §5.7, §9 item 3), and,
with hand marks, how close the stored vocal phrase starts are.

Read-only. For a random sample of vocal × bed song pairs that pass the tempo
gate (the same fold and BPM_MAX_DIFF the scorer uses), every usable vocal
section is placed on every usable bed section by the scorer's own
`phrase_align.place` — on `match._full_by_song`'s tempo, the stretch
`compute_stretch_factor` gives, the vocal's stored phrases — with the gate
opened wide, and the worst bar-line error of each placement is kept. A
pairing passes at T ms when that error is at most T (and its phrase ratio is
allowed — a ratio the gate refuses is counted as failing at every T). A
vocal section passes when any of its bed sections does, which is how the
scorer tries the next bed section.

    python scripts/timing_gate_share.py                       # the live library
    python scripts/timing_gate_share.py --pairs 3000 --tols 25,40,60
    python scripts/timing_gate_share.py --marks marks.csv     # phrase starts

`--marks` is a CSV of hand-marked vocal phrase starts:
`song_id,onset_sec[,pickup_ms]`, one row per phrase you marked (≥ 10 tracks
for §9 item 3). Each mark is matched to the nearest stored phrase start; the
report gives the share within one beat (target 90 %) and, where a pickup was
marked, the share of pickups within 50 ms.

Do not change `timing_tol_ms` on numbers from a library scored before the
fitted-BPM rule shipped (§9 item 3): re-analyse it first.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDE_OPEN_MS = 1.0e9


def _fold_diff(a: float, b: float) -> float:
    return min(abs(a * m - b) for m in (0.5, 1.0, 2.0))


def gate_shares(pairs: int = 2000, tols=(25.0, 40.0, 60.0), seed: int = 3,
                max_sections: int = 6, db_path=None) -> dict:
    from config import BPM_MAX_DIFF, TIMING_PHRASE_RATIOS
    from database.alignment import get_vocal_phrases
    from database.models import get_all_features, sections_for_songs
    from matcher.match import _full_by_song, compute_stretch_factor
    from matcher.phrase_align import place
    from matcher.sections import usable_sections

    kw = {} if db_path is None else {"db_path": db_path}
    full = _full_by_song(get_all_features(stem_type="full", **kw), db_path)
    vocal_ids = {f["song_id"] for f in get_all_features(stem_type="vocals",
                                                         columns=("song_id",), **kw)}
    bed_ids = {f["song_id"] for f in get_all_features(stem_type="instrumental",
                                                      columns=("song_id",), **kw)}
    songs = [s for s in full if full[s].get("bpm")]
    rng = random.Random(seed)
    sample, tries = [], 0
    while len(sample) < pairs and tries < pairs * 50 and len(songs) > 1:
        tries += 1
        v, b = rng.sample(songs, 2)
        if v not in vocal_ids or b not in bed_ids:
            continue
        if _fold_diff(full[v]["bpm"], full[b]["bpm"]) > BPM_MAX_DIFF:
            continue
        sample.append((v, b))
    ids = sorted({s for p in sample for s in p})
    sections = sections_for_songs(ids, **kw)
    phrases: dict = {}

    errs, vocal_best = [], {}
    unjudged = 0
    for v, b in sample:
        vs = usable_sections(sections.get(v) or [], True)[:max_sections]
        bs = usable_sections(sections.get(b) or [], False)[:max_sections]
        if not vs or not bs:
            continue
        if v not in phrases:
            phrases[v] = get_vocal_phrases(v, db_path=db_path)
        stretch = compute_stretch_factor(full[v]["bpm"], full[b]["bpm"]) or 1.0
        for i, sv in enumerate(vs):
            for sb in bs:
                p = place(sv, sb, stretch, phrases=phrases[v], tol_ms=WIDE_OPEN_MS,
                          ratios=TIMING_PHRASE_RATIOS)
                e = p.get("timing_err_max_ms")
                if p.get("timing_ok") is None and e is None:
                    unjudged += 1
                    continue
                e = float("inf") if (e is None or p.get("timing_ok") == 0) else float(e)
                errs.append(e)
                key = (v, b, i)
                vocal_best[key] = min(vocal_best.get(key, float("inf")), e)
    finite = [e for e in errs if e != float("inf")]
    out = {"song_pairs": len(sample), "pairings": len(errs), "unjudged": unjudged,
           "vocal_sections": len(vocal_best),
           "median_err_ms": round(statistics.median(finite), 1) if finite else None,
           "p90_err_ms": round(statistics.quantiles(finite, n=10)[-1], 1)
           if len(finite) >= 10 else None,
           "shares": {}}
    for t in tols:
        out["shares"][f"{t:g}"] = {
            "pairings": round(sum(e <= t for e in errs) / len(errs), 4) if errs else None,
            "vocal_sections": round(sum(e <= t for e in vocal_best.values())
                                    / len(vocal_best), 4) if vocal_best else None}
    return out


def phrase_marks(path: Path, db_path=None) -> dict:
    """Stored phrase starts against hand marks (song_id,onset_sec[,pickup_ms])."""
    from database.alignment import get_vocal_phrases
    from database.models import get_features_for_song

    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    rows = [r for r in rows if r and r[0].strip().isdigit()]
    within_beat, pickups_ok, pickups, n, songs = 0, 0, 0, 0, set()
    cache: dict = {}
    for r in rows:
        sid, onset = int(r[0]), float(r[1])
        mark_pickup = float(r[2]) if len(r) > 2 and r[2].strip() else None
        if sid not in cache:
            feat = get_features_for_song(sid, "full") or {}
            beat = 60.0 / float(feat.get("bpm") or 120.0)
            cache[sid] = (get_vocal_phrases(sid, db_path=db_path), beat)
        stored, beat = cache[sid]
        if not stored:
            n += 1
            songs.add(sid)
            continue
        near = min(stored, key=lambda p: abs(float(p["onset_sec"]) - onset))
        n += 1
        songs.add(sid)
        within_beat += abs(float(near["onset_sec"]) - onset) <= beat
        if mark_pickup is not None and near.get("pickup_sec") is not None:
            pickups += 1
            pickups_ok += abs(float(near["pickup_sec"]) * 1000.0 - mark_pickup) <= 50.0
    return {"marks": n, "tracks": len(songs),
            "within_a_beat": round(within_beat / n, 4) if n else None,
            "pickups_marked": pickups,
            "pickups_within_50ms": round(pickups_ok / pickups, 4) if pickups else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=int, default=2000, help="song pairs sampled")
    ap.add_argument("--tols", default="25,40,60", help="tolerances in ms")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--marks", type=Path, default=None, help="hand-marked phrase starts CSV")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args(argv)
    tols = tuple(float(t) for t in args.tols.split(",") if t.strip())
    out = {"gate": gate_shares(args.pairs, tols, args.seed)}
    if args.marks:
        out["phrases"] = phrase_marks(args.marks)
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    g = out["gate"]
    print(f"{g['song_pairs']} song pairs in the tempo gate, {g['pairings']} section "
          f"pairings placed ({g['unjudged']} without a grid to judge), "
          f"{g['vocal_sections']} vocal sections")
    print(f"worst bar-line error: median {g['median_err_ms']} ms, p90 {g['p90_err_ms']} ms")
    print(f"{'tolerance':>10} {'pairings pass':>14} {'vocal sections pass':>20}")
    for t, s in g["shares"].items():
        fmt = lambda x: "—" if x is None else f"{100 * x:.1f} %"
        print(f"{t + ' ms':>10} {fmt(s['pairings']):>14} {fmt(s['vocal_sections']):>20}")
    if "phrases" in out:
        p = out["phrases"]
        print(f"phrase starts: {p['marks']} marks on {p['tracks']} tracks — "
              f"{p['within_a_beat']} within a beat; pickups within 50 ms: "
              f"{p['pickups_within_50ms']} of {p['pickups_marked']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
