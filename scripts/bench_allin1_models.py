"""scripts/bench_allin1_models.py — which allin1 checkpoint is worth its time.

`harmonix-all` is an 8-fold ensemble, and it is where the allin1 stage's time
goes. Measured on the 27 real tracks of this library (readme §1, §9 item 12;
MEDIANS of `analysis_runs`): the whole stage 593.8 s a track, of which
inference 467.7 s, allin1's own Demucs 125.1 s and the spectrograms 4.4 s.
(The means of the same 27 runs: stage 662 s, inference 525 s, own Demucs 132 s,
over a 330–1080 s range. Peak resident 3,280 MB against the 6,000 MB budget.)

A single fold is roughly an eighth of the **inference** — about 410 s off that
594 s median stage, since the separation and the spectrograms do not get
cheaper. Which is why this script times the stage and the inference separately.
What a fold costs in accuracy is the question, and it is not one to answer by
intuition: the 25 ms timing gate, every section boundary and the FL export's
bar 1 all sit on these downbeats.

For each candidate checkpoint this runs the real inference on the same tracks
(the first named is the reference) and reports:

  * seconds, inference ms and peak resident MB per track
  * downbeat agreement with the reference: the median |Δt| at matched
    downbeats, the share within 25 / 40 / 70 ms, and the share a WHOLE BEAT or
    more out — an off-phase bar line, the failure that matters, because it
    moves every section and every placement
  * segment agreement: boundary F at ±0.5 s and ±3 s (edge boundaries trimmed:
    allin1 brackets every track with `start` / `end` segments) and
    overlap-weighted label agreement
  * BPM: |Δ| and the metrical relation

**Two things this does on purpose.**

1. It separates the four Demucs sources ITSELF, once per track, and hands the
   same four to every candidate. The library this exists to measure is
   `legacy_2stem` — no track has four sources on disk — so a picker that
   required them would select nothing; and letting allin1 demix per candidate
   would pay its own 125 s separation N times over, which is the exact waste
   the unified pipeline exists to remove. One separation also means every
   checkpoint is judged on identical input, which is what makes this a
   comparison. Its seconds are reported on their own, never as a model's.
2. It touches nothing live: the song rows are read through a READ-ONLY
   connection, the separation goes to a temp directory (deleted unless
   `--keep`), and the only thing written is the output directory. It needs no
   scratch library, unlike scripts/bench_pipelines.py — do not add one.

Not part of the test suite: it needs allin1, its weights and Demucs, i.e.
Docker. Every part of it is unit-tested with allin1 faked
(`tests/test_bench_allin1_models.py`); the measurement itself is owed to the
container.

    docker compose exec app python scripts/bench_allin1_models.py --auto 10
    docker compose exec app python scripts/bench_allin1_models.py --songs 12,40 \
        --models harmonix-all,harmonix-fold0,harmonix-fold1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import statistics as stats
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Candidates, reference first. The fold names come from allin1's documentation:
# no checkpoint but the ensemble has ever been loaded on any machine here, so
# `loadable_models` checks every one of them before a single track runs.
DEFAULT_MODELS = ("harmonix-all", "harmonix-fold0")

# The 25 ms timing gate (config.timing_tol_ms), the 40 ms readme §9 item 12.3
# asks to report, and the 70 ms the Engines report already uses.
BEAT_TOLERANCES_MS = (25.0, 40.0, 70.0)

# A bar line this far from the reference's is not jitter, it is off phase.
OFF_PHASE_BEATS = 0.9

# When a fold may replace the ensemble (the plan's decision rule).
PASS_RULE = {"off_phase": 0.0, "within_25ms": 0.95,
             "boundary_f_0_5s": 0.9, "label_agreement": 0.9}


# ── The library, read only ────────────────────────────────────────────────────

def _ro_conn(db_path: Path) -> sqlite3.Connection:
    """A read-only connection: this script must never write the library."""
    uri = "file:" + Path(db_path).as_posix() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _db(db_path: Optional[Path] = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    from config import DB_PATH
    return DB_PATH


def parse_song_ids(text: str) -> List[int]:
    return [int(x) for x in str(text or "").split(",") if x.strip()]


def pick_songs(limit: int, db_path: Optional[Path] = None) -> List[int]:
    """Analysed tracks with a downloaded file and a measured tempo, spread over
    the BPM range so a fold is not judged on one tempo.

    Deliberately NOT "tracks with four sources on disk": the four sources are
    something this script produces (see the module docstring), and every track
    of the library it exists to measure is on `legacy_2stem`.
    """
    conn = _ro_conn(_db(db_path))
    try:
        rows = conn.execute(
            """SELECT s.id AS id, f.bpm AS bpm FROM songs s
                 JOIN features f ON f.song_id = s.id AND f.stem_type = 'full'
                WHERE s.status = 'analysed' AND s.raw_path IS NOT NULL
                  AND s.raw_path != '' AND f.bpm IS NOT NULL
                ORDER BY f.bpm, s.id""").fetchall()
    finally:
        conn.close()
    if not rows or limit <= 0:
        return []
    if limit >= len(rows):
        return [int(r["id"]) for r in rows]
    # Even fractions of the sorted range rather than a fixed step, so the pick
    # holds both ends: a step sampler leaves the fastest tracks out entirely.
    idx = sorted({round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)}
                 if limit > 1 else {0})
    return [int(rows[i]["id"]) for i in idx]


def song_meta(song_id: int, db_path: Optional[Path] = None) -> Optional[dict]:
    conn = _ro_conn(_db(db_path))
    try:
        row = conn.execute("SELECT id, title, artist, raw_path, duration_secs "
                           "FROM songs WHERE id=?", (song_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


# ── Separating once per track ────────────────────────────────────────────────

def separate_four(audio_path: Path, out_dir: Path, threads: Optional[int] = None,
                  run=None) -> Dict[str, Path]:
    """htdemucs's four sources into ``out_dir``, as a {source: path} dict.

    A plain `python -m demucs` subprocess into a directory of its own: never
    `stems.separate.separate(unified=True)`, which writes under the live audio
    root. Returns {} when Demucs failed or a source is missing.
    """
    from analysis.allin1_runner import SOURCES
    from config import DEMUCS_MODEL
    from stems.separate import thread_env

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL,
           "--out", str(out_dir), str(audio_path)]
    env = {**os.environ, **thread_env()}
    if threads:
        env.update({k: str(max(1, int(threads))) for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    runner = run or subprocess.run
    try:
        res = runner(cmd, env=env, capture_output=True, text=True, timeout=7200)
    except Exception as exc:  # noqa: BLE001 — a missing demucs, a timeout
        print(f"    demucs failed: {type(exc).__name__}: {exc}")
        return {}
    if getattr(res, "returncode", 1) != 0:
        print(f"    demucs exited {res.returncode}: "
              f"{(getattr(res, 'stdout', '') or '')[-400:]}")
        return {}
    made = out_dir / DEMUCS_MODEL / Path(audio_path).stem
    found = {src: made / f"{src}.wav" for src in SOURCES}
    missing = [s for s, p in found.items() if not p.exists()]
    if missing:
        print(f"    demucs left no {', '.join(missing)} in {made}")
        return {}
    return found


# ── Agreement ────────────────────────────────────────────────────────────────

def _nearest_deltas(ref: Sequence[float], got: Sequence[float]) -> List[float]:
    got = sorted(float(x) for x in got)
    if not got:
        return []
    import bisect
    out = []
    for t in (float(x) for x in ref):
        i = bisect.bisect_left(got, t)
        near = [got[j] for j in (i - 1, i) if 0 <= j < len(got)]
        out.append(min(abs(t - x) for x in near))
    return out


def downbeat_report(ref: Sequence[float], got: Sequence[float],
                    beat_secs: Optional[float]) -> dict:
    """How far the candidate's bar lines sit from the reference's.

    The shares come from `analysis.compare.downbeat_agreement` (one definition
    of "agree" for the whole app); the median |Δt| and the off-phase share are
    this script's own, because a bar line a whole beat out is the failure the
    tolerances cannot see — at 120 BPM it is 500 ms, which reads simply as
    "outside 70 ms" and would look like any other miss.
    """
    from analysis.compare import downbeat_agreement

    out: dict = {"n_ref": len(list(ref) or [])}
    deltas = _nearest_deltas(ref, got)
    if not deltas:
        return out
    out["median_ms"] = round(stats.median(deltas) * 1000.0, 2)
    for tol in BEAT_TOLERANCES_MS:
        out[f"within_{int(tol)}ms"] = downbeat_agreement(ref, got, tol / 1000.0)
    out["off_phase"] = (round(sum(1 for d in deltas
                                  if d >= beat_secs * OFF_PHASE_BEATS)
                              / len(deltas), 4) if beat_secs else None)
    return out


def _inner_boundaries(segments: Sequence[Sequence]) -> List[float]:
    starts = sorted(float(s[0]) for s in segments)
    return starts[1:]


def label_agreement(ref_segments: Sequence[Sequence],
                    got_segments: Sequence[Sequence]) -> Optional[float]:
    """Share of the reference's duration whose label the candidate agrees with,
    weighted by how long each overlap lasts."""
    total = agree = 0.0
    for start, end, label in ((s[0], s[1], s[2]) for s in ref_segments):
        for g0, g1, glabel in ((s[0], s[1], s[2]) for s in got_segments):
            overlap = max(0.0, min(float(end), float(g1)) - max(float(start), float(g0)))
            if overlap <= 0:
                continue
            total += overlap
            if str(glabel) == str(label):
                agree += overlap
    return round(agree / total, 4) if total else None


def segment_report(ref_segments: Sequence[Sequence], got_segments: Sequence[Sequence],
                   duration: Optional[float]) -> dict:
    """Boundary F at ±0.5 s and ±3 s, and label agreement.

    `boundary_prf`'s ``trim`` is passed deliberately: allin1 brackets every
    track with `start` / `end` silence segments, and counting those boundaries
    would mark two identical cuts as disagreeing.
    """
    from analysis.compare import boundary_prf

    ref_b, got_b = _inner_boundaries(ref_segments), _inner_boundaries(got_segments)
    span = float(duration) if duration else max(
        [float(s[1]) for s in list(ref_segments) + list(got_segments)] or [0.0])
    trim = (0.0, span) if span else None
    out = {}
    for tol, name in ((0.5, "boundary_f_0_5s"), (3.0, "boundary_f_3s")):
        prf = boundary_prf(got_b, ref_b, window=tol, trim=trim)
        out[name] = prf["f"]
    out["label_agreement"] = label_agreement(ref_segments, got_segments)
    return out


# ── The checkpoints ──────────────────────────────────────────────────────────

def loadable_models(names: Sequence[str], device: str = "cpu"
                    ) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Which candidates this build can actually load, and why the rest cannot.

    Run before any track: loading is also what the run will do (the model cache
    keeps it, so this costs the load once), and a two-hour run that discovers a
    misspelt checkpoint at the end is the failure this guards.
    """
    from analysis import allin1_runner as runner

    kept: List[str] = []
    failures: List[Tuple[str, str]] = []
    for name in names:
        try:
            runner._model(device, name)
            kept.append(name)
            print(f"  ok {name}")
        except Exception as exc:  # noqa: BLE001 — any load failure drops it
            reason = f"{type(exc).__name__}: {exc}"
            failures.append((name, reason))
            print(f"  ! {name}: {reason}")
    return kept, failures


def run_one(full_path: Path, sources: Dict[str, Path], model: str,
            device: str = "cpu", threads: Optional[int] = None) -> dict:
    """One real allin1 run on the four sources this script separated. Returns
    the summary plus the wall seconds, the inner timings and the peak MB."""
    from analysis import allin1_runner as runner

    four = {s: sources[s] for s in runner.SOURCES if s in sources}
    timings: dict = {}
    t0 = time.perf_counter()
    with runner.PeakRss() as rss:
        result = runner.run(Path(full_path), four, device=device, timings=timings,
                            threads=threads, model=model)
        secs = time.perf_counter() - t0
        summary = runner.summarise(
            result, content_hash=None,
            demixed_by="pipeline" if len(four) == len(runner.SOURCES) else "allin1",
            ms=secs * 1000.0, model=model)
        del result
    summary["_secs"] = round(secs, 2)
    summary["_peak_mb"] = rss.peak_mb
    summary["_timings"] = timings
    return summary


def compare_to_reference(ref: dict, got: dict, duration: Optional[float]) -> dict:
    """Every number that says how far a candidate is from the reference."""
    from analysis.compare import bpm_relation

    beat_secs = None
    for bpm in (ref.get("bpm"), got.get("bpm")):
        if bpm:
            beat_secs = 60.0 / float(bpm)
            break
    rec = downbeat_report(ref["downbeats"], got["downbeats"], beat_secs)
    rec.update(segment_report(ref["segments"], got["segments"], duration))
    if ref.get("bpm") and got.get("bpm"):
        rec["bpm_delta"] = round(abs(float(got["bpm"]) - float(ref["bpm"])), 3)
        rec["bpm_relation"] = bpm_relation(got["bpm"], ref["bpm"])
    return rec


# ── The report ───────────────────────────────────────────────────────────────

def _say(text: str, stream=None) -> None:
    """Print, surviving a console that cannot encode the report (Δ, ±, ≥ — a
    Windows cp1252 console raises). The files are written before this runs, so
    a failure here would throw away nothing but the echo — and still end a
    two-hour run in a traceback."""
    out = stream if stream is not None else sys.stdout
    try:
        out.write(text + "\n")
        out.flush()
    except UnicodeEncodeError:
        out.write(text.encode("ascii", "replace").decode("ascii") + "\n")
        out.flush()


def _median(rows: List[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None and r.get(key) != ""]
    return round(stats.median(vals), 4) if vals else None


def _verdict(med) -> Tuple[bool, List[str]]:
    """Does this candidate clear the decision rule, and what fails if not."""
    broken = []
    for key, floor in PASS_RULE.items():
        val = med(key)
        if val is None:
            broken.append(f"{key} unmeasured")
        elif key == "off_phase" and val > floor:
            broken.append(f"{key} {val} > {floor}")
        elif key != "off_phase" and val < floor:
            broken.append(f"{key} {val} < {floor}")
    return (not broken), broken


def write_report(rows: List[dict], reference: str, out_dir: Path,
                 separations: Optional[List[dict]] = None) -> Path:
    """`summary.md`, `results.csv` and `results.json` in ``out_dir``.

    The reference leads: its row per track comes first in the CSV and its block
    first in the summary, because every other number is measured against it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    separations = separations or []
    ordered = sorted(rows, key=lambda r: (r.get("song_id") or 0,
                                          0 if r.get("model") == reference else 1,
                                          str(r.get("model"))))
    keys = ["song_id", "model"] + sorted({k for r in ordered for k in r}
                                         - {"song_id", "model"})
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(ordered)
    (out_dir / "results.json").write_text(
        json.dumps({"reference": reference, "separations": separations,
                    "rows": ordered}, indent=2, default=str), encoding="utf-8")

    models = [reference] + [m for m in dict.fromkeys(
        str(r.get("model")) for r in ordered) if m != reference]
    sep_med = _median([{"secs": s.get("secs")} for s in separations], "secs")
    lines = [f"# allin1 checkpoints — reference `{reference}`", "",
             f"Tracks: {len({r.get('song_id') for r in ordered})}. "
             f"Separation (htdemucs 4-source, ONCE per track, the same four "
             f"handed to every checkpoint): median {sep_med} s — not a model's "
             f"cost, and not paid again per candidate.", ""]
    for model in models:
        mine = [r for r in ordered if str(r.get("model")) == model]
        if not mine:
            continue

        def med(key, _mine=mine):
            return _median(_mine, key)

        lines += [f"## {model}", "",
                  f"- tracks: {len(mine)}",
                  f"- seconds: median {med('secs')}",
                  f"- inference ms: median {med('inference_ms')}",
                  f"- peak MB: median {med('peak_mb')}"]
        if model != reference:
            ok, broken = _verdict(med)
            lines += [f"- downbeat |Δt| median: {med('median_ms')} ms",
                      *[f"- within {int(t)} ms: {med(f'within_{int(t)}ms')}"
                        for t in BEAT_TOLERANCES_MS],
                      f"- **off phase (a beat or more out): {med('off_phase')}**",
                      f"- boundary F ±0.5 s: {med('boundary_f_0_5s')}",
                      f"- boundary F ±3 s: {med('boundary_f_3s')}",
                      f"- label agreement: {med('label_agreement')}",
                      f"- |ΔBPM| median: {med('bpm_delta')}",
                      f"- {'PASS' if ok else 'FAIL'} — may replace the reference"
                      + ("" if ok else f" (fails: {'; '.join(broken)})")]
        lines.append("")
    lines += ["The rule: a fold replaces the ensemble only with off phase 0, "
              "within 25 ms ≥ 0.95, boundary F ±0.5 s ≥ 0.9 and label agreement "
              "≥ 0.9. Anything less and the ensemble stays — the timing gate is "
              "25 ms and the point of allin1 here is bar lines you can trust.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--auto", type=int, default=10,
                    help="pick N analysed tracks spread over the BPM range")
    ap.add_argument("--songs", default="", help="comma-separated song ids")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated checkpoints; the FIRST is the reference")
    ap.add_argument("--device", default=None, help="cpu (default) or cuda")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch / Demucs threads (default: DEMUCS_THREADS)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the separated sources instead of deleting them")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <data_dir>/bench/allin1_models/<ts>)")
    ap.add_argument("--db", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    from analysis import allin1_runner as runner

    why = runner.missing()
    if why:
        print(f"allin1 cannot run here: {why}")
        return 2

    from config import DATA_DIR, DEMUCS_THREADS, current_allin1_device
    device = args.device or current_allin1_device()
    threads = args.threads or DEMUCS_THREADS
    db_path = Path(args.db) if args.db else None

    names = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not names:
        print("No checkpoints named.")
        return 1
    reference = names[0]
    print(f"Checking {len(names)} checkpoint(s) on {device}…")
    kept, _failures = loadable_models(names, device)
    if reference not in kept:
        print(f"The reference checkpoint {reference} will not load — nothing to "
              f"measure against. Fix the name (config.ALLIN1_MODELS) or the "
              f"install before running tracks.")
        return 2
    if len(kept) < 2:
        print("Need at least two loadable checkpoints to compare.")
        return 1
    candidates = [reference] + [m for m in kept if m != reference]

    songs = parse_song_ids(args.songs) or pick_songs(args.auto, db_path)
    if not songs:
        print("No analysed tracks with a downloaded file and a measured tempo.")
        return 1
    print(f"Reference: {reference}. Candidates: {candidates}. Tracks: {songs}")

    rows: List[dict] = []
    separations: List[dict] = []
    for song_id in songs:
        meta = song_meta(song_id, db_path)
        audio = Path(meta["raw_path"]) if meta and meta.get("raw_path") else None
        if not audio or not audio.exists():
            print(f"  song {song_id}: no audio on disk — skipped")
            continue
        work = Path(tempfile.mkdtemp(prefix="bench_allin1_"))
        try:
            print(f"  song {song_id} · separating once for every checkpoint…",
                  flush=True)
            t0 = time.perf_counter()
            sources = separate_four(audio, work / "demucs", threads=threads)
            sep_secs = time.perf_counter() - t0
            if len(sources) != len(runner.SOURCES):
                print(f"  song {song_id}: could not separate it — skipped rather "
                      f"than letting allin1 demix per checkpoint")
                continue
            separations.append({"song_id": song_id, "secs": round(sep_secs, 2)})
            print(f"    separated in {sep_secs:.1f} s")
            ref_summary = None
            for model in candidates:
                print(f"    {model} …", flush=True)
                try:
                    summary = run_one(audio, sources, model, device=device,
                                      threads=threads)
                except Exception as exc:  # noqa: BLE001 — one model failing is data
                    print(f"      failed: {type(exc).__name__}: {exc}")
                    continue
                timings = summary.get("_timings") or {}
                rec = {"song_id": song_id, "model": model,
                       "secs": summary["_secs"], "peak_mb": summary["_peak_mb"],
                       "inference_ms": timings.get("inference"),
                       "spectrogram_ms": timings.get("spectrogram"),
                       "model_load_ms": timings.get("model_load"),
                       "separate_secs": round(sep_secs, 2),
                       "demixed_by": summary.get("demixed_by"),
                       "bpm": summary.get("bpm"),
                       "downbeats": len(summary.get("downbeats") or []),
                       "segments": len(summary.get("segments") or []),
                       "conf_mean": summary.get("conf_mean")}
                if model == reference:
                    ref_summary = summary
                elif ref_summary is not None:
                    rec.update(compare_to_reference(
                        ref_summary, summary,
                        (meta or {}).get("duration_secs")))
                rows.append(rec)
        finally:
            if args.keep:
                print(f"    sources kept in {work}")
            else:
                shutil.rmtree(work, ignore_errors=True)

    if not rows:
        print("Nothing measured.")
        return 1
    out_dir = Path(args.out) if args.out else (
        Path(DATA_DIR) / "bench" / "allin1_models"
        / datetime.now().strftime("%Y%m%d-%H%M%S"))
    write_report(rows, reference, out_dir, separations)
    _say((out_dir / "summary.md").read_text(encoding="utf-8"))
    _say(f"\n{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
