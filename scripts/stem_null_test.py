"""The quality check required before switching to the unified pipeline
(readme §8, §5.3): is the instrumental summed from a 4-source htdemucs pass the
same as the 2-stem pass's no_vocals?

For each track: run Demucs both ways on the downloaded mix (in a scratch
folder — library files are never touched), then null-test instrumental against
instrumental. Reports the residual (dBFS, RMS of the difference), the SNR
(dB, instrumental energy over residual energy), the peak difference, and writes
an A/B pair per track for listening:

    <data_dir>/bench/stem_null/<timestamp>/<song>_A_2stem.wav
    <data_dir>/bench/stem_null/<timestamp>/<song>_B_4stem_sum.wav
    <data_dir>/bench/stem_null/<timestamp>/<song>_residual.wav (the difference)

Exits 1 (STOP) when any track's residual is above --max-residual-db (default
−40 dBFS): the brief says to stop and flag if the difference is audible.

    docker compose exec app python scripts/stem_null_test.py --auto 10
    python scripts/stem_null_test.py --songs 12,40,77
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _demucs(mix: Path, out: Path, two: bool) -> dict:
    from config import DEMUCS_MODEL
    from stems.separate import thread_env
    import os
    cmd = [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL, "--out", str(out), str(mix)]
    if two:
        cmd[3:3] = ["--two-stems", "vocals"]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env={**os.environ, **thread_env()},
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    secs = time.perf_counter() - t0
    return {"dir": out / DEMUCS_MODEL / mix.stem, "secs": secs}


def _read(p: Path):
    import soundfile as sf
    y, sr = sf.read(str(p), dtype="float64", always_2d=True)
    return y, sr


def null_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Residual of b against a, sample-aligned (Demucs keeps the mix's clock)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = b - a
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)) + 1e-30))  # noqa: E731
    return {"residual_dbfs": round(20 * np.log10(rms(d)), 2),
            "snr_db": round(20 * np.log10(rms(a) / rms(d)), 2),
            "peak_diff": round(float(np.max(np.abs(d))), 6),
            "samples": int(n)}


def _songs(args):
    from database import models
    conn = models.get_conn()
    try:
        if args.songs:
            ids = [int(x) for x in args.songs.split(",") if x.strip()]
            rows = [conn.execute("SELECT id, title, raw_path FROM songs WHERE id=?",
                                 (i,)).fetchone() for i in ids]
        else:
            rows = conn.execute(
                "SELECT id, title, raw_path FROM songs WHERE status='analysed' "
                "AND raw_path IS NOT NULL ORDER BY RANDOM() LIMIT ?", (args.auto,)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows if r and r["raw_path"] and Path(r["raw_path"]).exists()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--songs", default="", help="comma-separated song ids")
    ap.add_argument("--auto", type=int, default=10, help="random analysed tracks (default 10)")
    ap.add_argument("--max-residual-db", type=float, default=-40.0)
    args = ap.parse_args(argv)
    import soundfile as sf
    from config import DATA_DIR
    out_dir = DATA_DIR / "bench" / "stem_null" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    songs = _songs(args)
    if len(songs) < 10:
        print(f"note: {len(songs)} track(s) — the brief asks for at least 10", file=sys.stderr)
    rows, worst = [], -999.0
    for s in songs:
        mix = Path(s["raw_path"])
        work = Path(tempfile.mkdtemp(prefix="nulltest_"))
        try:
            two = _demucs(mix, work / "two", two=True)
            four = _demucs(mix, work / "four", two=False)
            a, sr = _read(two["dir"] / "no_vocals.wav")
            b = None
            for src in ("drums", "bass", "other"):
                y, _ = _read(four["dir"] / f"{src}.wav")
                b = y if b is None else b + y
            res = null_test(a, b)
            name = f"{s['id']:05d}"
            sf.write(str(out_dir / f"{name}_A_2stem.wav"), a, sr)
            peak = float(np.max(np.abs(b)))
            sf.write(str(out_dir / f"{name}_B_4stem_sum.wav"), b / peak if peak > 1 else b, sr)
            n = min(len(a), len(b))
            sf.write(str(out_dir / f"{name}_residual.wav"), b[:n] - a[:n], sr)
            row = {"song_id": s["id"], "title": s["title"], **res,
                   "demucs_2stem_secs": round(two["secs"], 1),
                   "demucs_4stem_secs": round(four["secs"], 1)}
            rows.append(row)
            worst = max(worst, res["residual_dbfs"])
            print(f"{s['id']:>6}  residual {res['residual_dbfs']:>7.2f} dBFS  "
                  f"SNR {res['snr_db']:>6.2f} dB  peak {res['peak_diff']:.4f}  {s['title']}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if rows:
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nA/B files and results.csv in {out_dir}")
    if rows and worst > args.max_residual_db:
        print(f"STOP: worst residual {worst:.2f} dBFS is above {args.max_residual_db} dBFS — "
              "listen to the A/B pairs before switching to unified_4stem.")
        return 1
    print(f"OK: worst residual {worst:.2f} dBFS (threshold {args.max_residual_db}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
