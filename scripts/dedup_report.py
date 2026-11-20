"""How the EffNet duplicate check splits a library (readme §5.6, §9).

Read-only. For every pair of songs whose titles normalise to the same work
but whose artists differ — the only pairs the audio check decides — and for a
random sample of unrelated pairs, it prints the cosine of their stored EffNet
mean embeddings, and how many of each clear the threshold
(`matcher.dedup.AUDIO_CONFIRM_MIN`) and a few around it. A good threshold
passes the true re-uploads and almost no random pair. Same-title pairs are
listed with their titles, so the ones near the line can be checked by ear.

    python scripts/dedup_report.py                 # the live library
    python scripts/dedup_report.py --db /data/mashup.db --random 5000
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(db: Path):
    from database.models import EMBEDDING_KIND, unpack_vector
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        songs = {int(r["id"]): dict(r) for r in conn.execute(
            "SELECT id, title, artist FROM songs").fetchall()}
        for r in conn.execute("SELECT song_id, vector FROM embeddings WHERE kind = ?",
                              (EMBEDDING_KIND,)).fetchall():
            if int(r["song_id"]) in songs:
                songs[int(r["song_id"])]["embedding"] = unpack_vector(r["vector"])
    finally:
        conn.close()
    return songs


def main(argv=None) -> int:
    from matcher.dedup import (AUDIO_CONFIRM_MIN, embedding_cosine, normalise_artist,
                               normalise_title)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=None, help="mashup.db (default: the live one)")
    ap.add_argument("--random", type=int, default=3000, help="random pairs sampled")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=40, help="same-title pairs listed")
    args = ap.parse_args(argv)
    if args.db is None:
        import config
        args.db = Path(config.DB_PATH)
    songs = _load(args.db)
    with_emb = [s for s in songs.values() if s.get("embedding")]
    print(f"{len(songs)} songs, {len(with_emb)} with an EffNet embedding")
    if len(with_emb) < 2:
        print("Nothing to compare: run analysis (or POST /api/analysis/embeddings/backfill).")
        return 0

    by_title: dict = {}
    for s in with_emb:
        work = normalise_title(s["title"])
        if work:
            by_title.setdefault(work, []).append(s)
    same = []
    for group in by_title.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if normalise_artist(a["artist"]) == normalise_artist(b["artist"]) \
                        and normalise_artist(a["artist"]):
                    continue                 # same artist: no audio check is asked
                c = embedding_cosine(a, b)
                if c is not None:
                    same.append((c, a, b))
    rng = random.Random(args.seed)
    rand = []
    for _ in range(args.random):
        a, b = rng.sample(with_emb, 2)
        c = embedding_cosine(a, b)
        if c is not None:
            rand.append(c)

    def share(xs, t):
        return (sum(1 for x in xs if x >= t) / len(xs)) if xs else 0.0

    cuts = sorted({0.90, 0.93, AUDIO_CONFIRM_MIN, 0.97, 0.99})
    print(f"\nthreshold   same-title, different artist ({len(same)})   random pairs ({len(rand)})")
    for t in cuts:
        mark = "  ← AUDIO_CONFIRM_MIN" if abs(t - AUDIO_CONFIRM_MIN) < 1e-9 else ""
        print(f"  {t:.2f}      {share([c for c, *_ in same], t):6.1%}"
              f"                          {share(rand, t):6.1%}{mark}")
    if rand:
        rs = sorted(rand)
        print(f"\nrandom pairs: median {rs[len(rs) // 2]:.3f}, "
              f"p99 {rs[min(len(rs) - 1, int(len(rs) * 0.99))]:.3f}, max {rs[-1]:.3f}")
    if same:
        print(f"\nsame-title pairs nearest the threshold (cosine, then the two uploads):")
        for c, a, b in sorted(same, key=lambda x: abs(x[0] - AUDIO_CONFIRM_MIN))[:args.show]:
            verdict = "variant" if c >= AUDIO_CONFIRM_MIN else "kept apart"
            print(f"  {c:.4f}  {verdict:10s}  {a['artist']} — {a['title']}  |  "
                  f"{b['artist']} — {b['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
