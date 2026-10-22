"""Hand-check vocal phrase detection (readme §5.5, §8): list each track's
detected phrase starts with their bar line and pickup, and — given a CSV of
the true entries you marked by ear — score them against the brief's bar:
at least 90% within one beat of the true vocal entry, pickups within 50 ms.

    docker compose exec app python scripts/check_phrases.py --songs 12,40
    python scripts/check_phrases.py --truth truth.csv

truth.csv: ``song_id,onset_sec`` — one row per true phrase start (the moment
the voice comes in, as you hear it; mark it in Studio by scrubbing). Phrases
are read for the song's live pipeline, from the allin1 grid when it exists,
else the Essentia grid (``--grid`` to pick one).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def score(detected: list[dict], truth: list[float], beat: float) -> dict:
    """Each true entry matched to the nearest detected onset: hit when within
    a beat; pickup error = |detected onset − true onset| for the hits."""
    hits, errs = 0, []
    for t in truth:
        if not detected:
            break
        p = min(detected, key=lambda d: abs(d["onset_sec"] - t))
        if abs(p["onset_sec"] - t) <= beat:
            hits += 1
            errs.append(abs(p["onset_sec"] - t) * 1000.0)
    return {"truth": len(truth), "hits": hits,
            "hit_rate": hits / len(truth) if truth else None,
            "within_50ms": sum(1 for e in errs if e <= 50.0),
            "median_err_ms": sorted(errs)[len(errs) // 2] if errs else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--songs", default="")
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--grid", choices=("allin1", "essentia"), default=None)
    args = ap.parse_args(argv)
    from database import alignment, models

    truth: dict[int, list[float]] = {}
    if args.truth:
        with open(args.truth, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                truth.setdefault(int(r["song_id"]), []).append(float(r["onset_sec"]))
    ids = [int(x) for x in args.songs.split(",") if x.strip()] or sorted(truth)
    if not ids:
        ap.error("give --songs or --truth")

    total_truth = total_hits = total_50 = 0
    for sid in ids:
        song = models.get_song(sid) or {}
        ph = alignment.get_vocal_phrases(sid, grid_source=args.grid)
        feat = models.get_features_for_song(sid, "full") or {}
        beat = 60.0 / float(feat.get("bpm") or 120.0)
        print(f"\n#{sid} {song.get('title')} — {len(ph)} phrases "
              f"({ph[0]['grid_source'] if ph else '-'} grid, {60 / beat:.1f} BPM)")
        for p in ph:
            print(f"   bar {p['downbeat_sec']:8.2f}s  sung {p['onset_sec']:8.2f}s  "
                  f"pickup {p['pickup_sec'] * 1000:+6.0f} ms  {p['bars'] or '?':>2} bars  "
                  f"conf {p['confidence'] or 0:.2f}")
        if sid in truth:
            s = score(ph, truth[sid], beat)
            total_truth += s["truth"]
            total_hits += s["hits"]
            total_50 += s["within_50ms"]
            print(f"   vs truth: {s['hits']}/{s['truth']} within a beat, "
                  f"{s['within_50ms']} within 50 ms (median {s['median_err_ms']} ms)")
    if total_truth:
        rate = total_hits / total_truth
        print(f"\nAll: {total_hits}/{total_truth} = {rate:.0%} within a beat "
              f"(target 90%), {total_50}/{total_hits or 1} hits within 50 ms")
        return 0 if rate >= 0.9 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
