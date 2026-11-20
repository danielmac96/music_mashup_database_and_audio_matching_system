#!/usr/bin/env python
"""scripts/storage_report.py — what a library keeps on disk per kind and per
track, and the same at another size (readme §4, §8; Phase 5).

    python scripts/storage_report.py                      # the live library, measured
    python scripts/storage_report.py --project 10000      # ...and scaled to 10,000 tracks
    python scripts/storage_report.py --calibrate --project 10000

Measured: `database.storage.storage_report` — every file the database points
at, stat-ed (the same numbers Settings → Storage shows). Projected: each
kind's bytes per audio second times the mean track length times the track
count. A library with no audio on disk (a synthetic one) has no rates of its
own: `--calibrate` writes one synthetic minute of each stem in the formats
the pipeline writes (16-bit FLAC for the four sources, 24-bit FLAC for the
instrumental; a decaying-harmonic signal with noise, which FLAC compresses
about as badly as dense dance music) and uses the readme's measured sidecar
rate (§5.4: ~1.5 KB per audio second per stem, four stems). Real music
differs: run it on your own library for its numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIDECAR_BYTES_PER_SEC_PER_STEM = 1500.0     # readme §5.4, measured
SIDECAR_STEMS = 4                           # full, vocals, instrumental, bass
MIX_BYTES_PER_SEC = 128_000 / 8             # a 128 kbit/s download


def calibrate(secs: float = 60.0) -> dict:
    """Bytes per audio second of each stem, written as the pipeline writes it."""
    import numpy as np
    import soundfile as sf
    sr = 44100
    rng = np.random.default_rng(1)
    t = np.arange(int(sr * secs)) / sr

    def tone(f0, decay, noise):
        beat = (t * 2.0) % 1.0                         # 120 BPM envelope
        env = np.exp(-decay * beat)
        x = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 9))
        x = 0.25 * env * x + noise * rng.standard_normal(t.size) * env
        return np.stack([x, np.roll(x, 37)], axis=1).astype("float32") * 0.5

    sigs = {"vocals": tone(330, 1.0, 0.03), "drums": tone(60, 9.0, 0.25),
            "bass": tone(55, 2.0, 0.01), "other": tone(220, 1.5, 0.08)}
    sigs["instrumental"] = np.clip(sigs["drums"] + sigs["bass"] + sigs["other"], -1, 1)
    rates = {}
    with tempfile.TemporaryDirectory() as d:
        for kind, x in sigs.items():
            p = Path(d) / f"{kind}.flac"
            sf.write(str(p), x, sr, subtype="PCM_24" if kind == "instrumental" else "PCM_16")
            rates[kind] = p.stat().st_size / secs
    rates["sidecars"] = SIDECAR_BYTES_PER_SEC_PER_STEM * SIDECAR_STEMS
    rates["mix"] = MIX_BYTES_PER_SEC
    return rates


def _fmt(n) -> str:
    if n is None:
        return "—"
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n:.0f} B"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", type=int, default=None, help="tracks to project to")
    ap.add_argument("--calibrate", action="store_true",
                    help="use synthetic per-second rates (a library without audio)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from database.storage import projection, storage_report
    rep = storage_report()
    mean_secs = (rep["audio_secs"] / rep["tracks"]) if rep["tracks"] else 220.0
    out = {"measured": rep}
    if args.project:
        rates = calibrate() if args.calibrate else {
            k: v["per_audio_sec"] for k, v in rep["kinds"].items() if v.get("per_audio_sec")}
        proj = projection(rates, args.project, mean_secs)
        total = sum(proj.values())
        analysis_only = sum(proj.get(k, 0) for k in ("drums", "bass", "other"))
        out["projection"] = {"tracks": args.project, "mean_secs": round(mean_secs, 1),
                             "rates_bytes_per_sec": {k: round(v, 1) for k, v in rates.items()},
                             "source": "calibrated (synthetic)" if args.calibrate else "measured",
                             "bytes": proj, "total": total,
                             "per_track": round(total / args.project),
                             "keep_analysis_stems_off_frees": analysis_only}
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f"Measured: {rep['tracks']} tracks, {_fmt(rep['total_bytes'])} "
          f"({_fmt(rep['per_track_bytes'])} a track); prunable now "
          f"{_fmt(rep['prunable']['bytes'])} on {rep['prunable']['tracks']} tracks")
    for k, v in rep["kinds"].items():
        print(f"  {k:>13}: {_fmt(v.get('bytes')):>9}  a track {_fmt(v.get('per_track')):>9}"
              f"  missing {v.get('missing', 0)}")
    if "projection" in out:
        p = out["projection"]
        print(f"Projected to {p['tracks']:,} tracks of {p['mean_secs']} s ({p['source']}): "
              f"{_fmt(p['total'])}, {_fmt(p['per_track'])} a track; "
              f"keep_analysis_stems off frees {_fmt(p['keep_analysis_stems_off_frees'])}")
        for k, v in p["bytes"].items():
            print(f"  {k:>13}: {_fmt(v):>9}  a track {_fmt(v / p['tracks']):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
