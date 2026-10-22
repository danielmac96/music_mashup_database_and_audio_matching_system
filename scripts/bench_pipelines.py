"""Profile the two separation pipelines end to end (readme §8, §5.5): per
stage, the wall time, the peak resident memory and the storage per track, for

  legacy   legacy_2stem — 2-stem Demucs, Essentia, then allin1 separating its
           own four stems (what running allin1 costs without the unified pass)
  unified  unified_4stem — one 4-source Demucs pass feeding allin1 and the
           playback stems, Essentia on the summed stems

and print the brief's success criteria as PASS / FAIL:

  * unified total time per track <= legacy total (2-stem Demucs + allin1's own)
  * peak memory within allin1_mem_budget_mb
  * analysis data per track (allin1 core + embedding + sections + phrases)
    under 100 KB (kept stems excluded)

Each pipeline runs in a child process against a scratch library (its own
database, audio root and settings) holding COPIES of the chosen tracks'
downloads, so the live library is never written. Needs the full stack (Docker):
Demucs weights, Essentia, allin1 + its model weights.

    docker compose exec app python scripts/bench_pipelines.py --auto 5
    python scripts/bench_pipelines.py --songs 12,40 --keep
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGES = ("stems", "analysis", "structure", "allin1")
PIPELINES = {"legacy": "legacy_2stem", "unified": "unified_4stem"}


# ── Child: one pipeline over the copied tracks ────────────────────────────────

def _child(pipeline: str, files: list[str], out_json: str) -> None:
    from analysis.allin1_runner import PeakRss
    from api.workers import stages
    from database import alignment, models
    import api.workers.hook_worker as hook_worker
    hook_worker.warm_hooks = lambda *a, **k: {}          # not part of the comparison
    models.init_db()
    results = []
    for fp in files:
        src = Path(fp)
        sid = models.upsert_song(title=src.stem[:40], artist="bench",
                                 source_url=f"bench://{src.name}")
        models.update_song_status(sid, "downloaded", raw_path=str(src))
        row = {"file": src.name, "stages": {}}
        for stage, fn in (("stems", stages.do_stems), ("analysis", stages.do_analyze),
                          ("structure", stages.do_structure), ("allin1", stages.do_allin1)):
            t0 = time.perf_counter()
            with PeakRss() as rss:
                try:
                    fn(sid)
                    ok, err = True, None
                except Exception as exc:  # noqa: BLE001
                    ok, err = False, f"{type(exc).__name__}: {exc}"
            row["stages"][stage] = {"secs": round(time.perf_counter() - t0, 2),
                                    "peak_mb": rss.peak_mb, "ok": ok, "error": err}
        # Inner steps (decode, demix write, spectrogram, inference, summarise…)
        conn = models.get_conn()
        row["steps"] = {r["grp"]: round(r["ms"] / 1000.0, 3) for r in conn.execute(
            "SELECT grp, SUM(ms) AS ms FROM analysis_runs WHERE song_id=? GROUP BY grp",
            (sid,))}
        stems = {r["stem_type"]: r["file_path"] for r in conn.execute(
            "SELECT stem_type, file_path FROM stems WHERE song_id=? AND stem_type!='full'",
            (sid,))}
        conn.close()
        row["storage"] = {
            "analysis_bytes": sum(alignment.storage_bytes(sid).values()),
            "analysis_detail": alignment.storage_bytes(sid),
            "stem_bytes": {k: Path(v).stat().st_size for k, v in stems.items()
                           if Path(v).exists()},
        }
        row["audio_secs"] = (models.get_song(sid) or {}).get("duration_secs")
        results.append(row)
    Path(out_json).write_text(json.dumps({"pipeline": pipeline, "tracks": results},
                                         indent=2), encoding="utf-8")


# ── Parent: choose tracks, run both children, report ──────────────────────────

def _pick(args) -> list[Path]:
    from database import models
    conn = models.get_conn()
    try:
        if args.songs:
            ids = [int(x) for x in args.songs.split(",") if x.strip()]
            rows = [conn.execute("SELECT raw_path FROM songs WHERE id=?", (i,)).fetchone()
                    for i in ids]
        else:
            rows = conn.execute("SELECT raw_path FROM songs WHERE status='analysed' AND "
                                "raw_path IS NOT NULL ORDER BY RANDOM() LIMIT ?",
                                (args.auto,)).fetchall()
    finally:
        conn.close()
    return [Path(r[0]) for r in rows if r and r[0] and Path(r[0]).exists()]


def _run_child(name: str, pipeline: str, files: list[Path], work: Path) -> dict:
    root = work / name
    (root / "audio").mkdir(parents=True, exist_ok=True)
    copies = []
    for f in files:
        dst = root / "audio" / "full_song" / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
        copies.append(str(dst))
    out = root / "result.json"
    env = {**os.environ,
           "MASHUP_DB_PATH": str(root / "bench.db"),
           "MASHUP_AUDIO_ROOT": str(root / "audio"),
           "MASHUP_DATA_DIR": str(root / "data"),
           "MASHUP_SETTINGS_DIR": str(root / "settings"),
           "MASHUP_SEPARATION_PIPELINE": pipeline,
           "MASHUP_ANALYSIS_CACHE": "0",           # measure the work, not the cache
           "MASHUP_ALLIN1_MAX_SECS": "7200"}
    subprocess.run([sys.executable, __file__, "--child", pipeline, "--out", str(out),
                    *copies], env=env, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _summary(res: dict) -> dict:
    tracks = res["tracks"]
    per = {st: _median([t["stages"][st]["secs"] for t in tracks]) for st in STAGES}
    peak = max((t["stages"][st]["peak_mb"] or 0) for t in tracks for st in STAGES) if tracks else None
    total = _median([sum(t["stages"][st]["secs"] for st in STAGES) for t in tracks])
    kb = _median([t["storage"]["analysis_bytes"] / 1024 for t in tracks])
    stem_mb = _median([sum(t["storage"]["stem_bytes"].values()) / 1e6 for t in tracks])
    fails = [(t["file"], st, t["stages"][st]["error"]) for t in tracks for st in STAGES
             if not t["stages"][st]["ok"]]
    return {"per_stage_secs": per, "total_secs": total, "peak_mb": peak,
            "analysis_kb": kb, "stem_mb": stem_mb, "failures": fails}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--songs", default="")
    ap.add_argument("--auto", type=int, default=5)
    ap.add_argument("--keep", action="store_true", help="keep the scratch libraries")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        _child(args.child, args.files, args.out)
        return 0

    from config import DATA_DIR, current_int
    files = _pick(args)
    if not files:
        print("no downloaded tracks to benchmark", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="bench_pipelines_"))
    out_dir = DATA_DIR / "bench" / "pipelines" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = {name: _run_child(name, p, files, work) for name, p in PIPELINES.items()}
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    sums = {k: _summary(v) for k, v in results.items()}
    budget = current_int("allin1_mem_budget_mb")

    lines = [f"# Pipeline benchmark — {len(files)} tracks", "",
             "| stage (median s/track) | legacy | unified |", "|---|---|---|"]
    for st in STAGES:
        lines.append(f"| {st} | {sums['legacy']['per_stage_secs'][st]} | "
                     f"{sums['unified']['per_stage_secs'][st]} |")
    lines += [f"| **total** | {sums['legacy']['total_secs']} | {sums['unified']['total_secs']} |",
              f"| peak MB | {sums['legacy']['peak_mb']} | {sums['unified']['peak_mb']} |",
              f"| analysis KB/track | {sums['legacy']['analysis_kb']} | {sums['unified']['analysis_kb']} |",
              f"| stems MB/track | {sums['legacy']['stem_mb']} | {sums['unified']['stem_mb']} |", ""]
    checks = [
        ("unified time <= legacy (2-stem Demucs + allin1's own demix)",
         (sums["unified"]["total_secs"] or 0) <= (sums["legacy"]["total_secs"] or 0)),
        (f"peak memory within {budget} MB",
         all((s["peak_mb"] or 0) <= budget for s in sums.values())),
        ("analysis data < 100 KB/track",
         all((s["analysis_kb"] or 0) < 100 for s in sums.values())),
    ]
    for label, ok in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {label}")
    for name, s in sums.items():
        for f, st, err in s["failures"]:
            lines.append(f"- {name}: {f} failed at {st}: {err}")
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n{out_dir}")
    return 0 if all(ok for _l, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
