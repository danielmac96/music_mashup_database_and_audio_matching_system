"""Time Track detail's ghost-slot overlay on a library (master plan Phase 4,
criteria 4.1 and 4.5; readme §8).

    python scripts/bench_overlay.py --library /tmp/synth10k [--pairs 40]

Points the app at the library's three MASHUP_* paths (a synthetic one from
scripts/synth_library.py, or a copy of a real one), then asks
GET /api/tracks/{bed}/overlay for N section pairs — the planted drop-ins from
synth_manifest.json first, then stored pairs from the candidates table (or, on
an unscored library, random vocal × bed sections) — and reports:

  * the payload uncached (the route's cache emptied before every call) and
    cached, p50 / p95, with its size raw and gzipped;
  * how many placements each prices;
  * criterion 4.5: on the planted drop-ins, the share of the incoming vocal's
    phrase starts drawn within one 46 ms analysis frame of the bed's original
    phrase starts, at the library's landing.

Writes <data_dir>/bench/overlay/bench_overlay.{json,md}. Needs no audio.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _pairs(m, manifest, n, seed):
    out = [(p["vocal_song_id"], p["inst_song_id"], p["vocal_section"], p["inst_section"], True)
           for p in (manifest or {}).get("planted", [])]
    conn = m.get_conn()
    try:
        rows = conn.execute(
            "SELECT vocal_song_id, inst_song_id, vocal_section_idx, inst_section_idx "
            "FROM mashup_candidates WHERE combo_type='vocal_over_instrumental' "
            "AND vocal_section_idx IS NOT NULL AND inst_section_idx IS NOT NULL "
            "ORDER BY id LIMIT 5000").fetchall()
        if not rows:
            secs = conn.execute(
                "SELECT song_id, section_index, label FROM sections_allin1").fetchall()
            vox = [s for s in secs if s["label"] in ("chorus", "verse")]
            beds = [s for s in secs if s["label"] in ("chorus", "drop", "verse", "breakdown")]
            rng = random.Random(seed)
            rows = []
            for _ in range(n * 4):
                v, b = rng.choice(vox), rng.choice(beds)
                if v["song_id"] != b["song_id"]:
                    rows.append((v["song_id"], b["song_id"], v["section_index"],
                                 b["section_index"]))
    finally:
        conn.close()
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    planted = len(out)
    out += [(*tuple(r), False) for r in rows]
    return out[:max(n, planted)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True, help="folder holding mashup.db (synth_library --out)")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--cached-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv)
    lib = Path(args.library)
    os.environ["MASHUP_DB_PATH"] = str(lib / "mashup.db")
    os.environ.setdefault("MASHUP_AUDIO_ROOT", str(lib / "audio"))
    os.environ.setdefault("MASHUP_DATA_DIR", str(lib / "data"))
    os.environ.setdefault("MASHUP_SETTINGS_DIR", str(lib / "settings"))
    from fastapi.testclient import TestClient
    import api.routes.tracks as tracks
    import config
    import database.models as m
    from api.server import app

    manifest = None
    if (lib / "synth_manifest.json").exists():
        manifest = json.loads((lib / "synth_manifest.json").read_text(encoding="utf-8"))
    client = TestClient(app)
    pairs = _pairs(m, manifest, args.pairs, args.seed)
    # Warm the process (imports, library stats) on one pair, untimed.
    v, b, vs, bs, _ = pairs[0]
    client.get(f"/api/tracks/{b}/overlay", params={"vocal": v, "vs": vs, "bs": bs})

    uncached, cached, sizes, gz, n_place = [], [], [], [], []
    within = total = 0
    failures = 0
    for v, b, vs, bs, planted in pairs:
        url = f"/api/tracks/{b}/overlay"
        params = {"vocal": v, "vs": vs, "bs": bs}
        with tracks._OVERLAY_LOCK:
            tracks._OVERLAY_CACHE.clear()
        t = time.perf_counter()
        r = client.get(url, params=params)
        uncached.append((time.perf_counter() - t) * 1000)
        if r.status_code != 200:
            failures += 1
            continue
        sizes.append(len(r.content))
        gz.append(len(gzip.compress(r.content, compresslevel=5)))
        d = r.json()
        n_place.append(len(d["placements"]))
        for _ in range(args.cached_repeats):
            t = time.perf_counter()
            client.get(url, params=params)
            cached.append((time.perf_counter() - t) * 1000)
        if planted:
            zero = next((x for x in d["placements"] if x["offset_bars"] == 0), None)
            orig = [q["at"] for q in d["bed_phrases"]]
            lo, hi = d["bed_section"]["start"] - 1e-3, d["bed_section"]["end"] + 1e-3
            for q in (zero or {}).get("incoming_phrases", []):
                if lo <= q["at"] <= hi and orig:
                    total += 1
                    within += min(abs(u - q["at"]) for u in orig) <= d["frame_secs"]

    conn = m.get_conn()
    n_tracks = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    n_rows = conn.execute("SELECT COUNT(*) FROM mashup_candidates").fetchone()[0]
    conn.close()
    res = {
        "tracks": n_tracks, "candidate_rows": n_rows, "pairs": len(pairs), "failures": failures,
        "uncached_ms": {"p50": _pct(uncached, 0.5), "p95": _pct(uncached, 0.95),
                        "max": max(uncached) if uncached else None},
        "cached_ms": {"p50": _pct(cached, 0.5), "p95": _pct(cached, 0.95)},
        "payload_kb": {"median": statistics.median(sizes) / 1024 if sizes else None,
                       "gzip_median": statistics.median(gz) / 1024 if gz else None},
        "placements_median": statistics.median(n_place) if n_place else None,
        "planted_phrase_starts_within_frame": {"within": within, "total": total,
                                               "share": within / total if total else None},
    }
    out = Path(config.DATA_DIR) / "bench" / "overlay"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_overlay.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = lambda x: "—" if x is None else f"{x:.1f}"  # noqa: E731
    md = (f"# Overlay benchmark — {n_tracks} tracks, {n_rows} candidate rows\n\n"
          f"| | p50 | p95 |\n|---|---|---|\n"
          f"| uncached | {f(res['uncached_ms']['p50'])} ms | {f(res['uncached_ms']['p95'])} ms |\n"
          f"| cached | {f(res['cached_ms']['p50'])} ms | {f(res['cached_ms']['p95'])} ms |\n\n"
          f"Payload {f(res['payload_kb']['median'])} KB ({f(res['payload_kb']['gzip_median'])} KB gzip), "
          f"{res['placements_median']} placements; planted phrase starts within a frame: "
          f"{within} of {total}.\n")
    (out / "bench_overlay.md").write_text(md, encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
