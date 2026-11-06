"""scripts/bench_one_pass.py — the whole pipeline over local files, measured.

Builds a library from a folder of audio files (no download, no SoundCloud), runs
every track through the real pipeline, and prints what each stage cost from
`analysis_runs` — the same table the Queue screen's typical times come from.

It also checks the claim this exists to test: that each stage ran ONCE per
track. A second `download`, `quick`, `stems`, `analysis`, `structure` or
`allin1` row for one track is a finding, not a detail, and the report names the
tracks it happened to.

    docker compose exec app python scripts/bench_one_pass.py /data/incoming
    docker compose exec app python scripts/bench_one_pass.py /data/incoming --limit 5
    docker compose exec app python scripts/bench_one_pass.py --skip-ingest --limit 30

The test library is a separate MASHUP_DB_PATH / MASHUP_AUDIO_ROOT /
MASHUP_DATA_DIR (readme §4), not a flag — so the database, the data directory
and the track count are printed before anything runs.

The run itself needs Demucs, Essentia and allin1, i.e. Docker, so it is not
part of the test suite; every part of it is exercised in
tests/test_bench_one_pass.py with the pipeline faked at `pipeline_worker.run`.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}

# Stages that must appear exactly once per track for "one run" to be true.
#
# `structure` is in here because it is the stage that actually ran twice: the
# allin1 cut writes a plain `structure` row of its own (stages.py line ~1105,
# analyzer="allin1") beside the Essentia novelty cut's, so an allin1 library
# cut every track twice. Measured on the live 206-track library before this
# plan: 232 `structure` runs for 204 tracks, 27 songs cut more than once, one
# cut three times. There is no `allin1.structure` grp — only the per-step
# `allin1.structure.*` rows — so a check keyed on that name can never fire.
#
# The match is EXACT, never a prefix: a reused separation is filed as
# `stems.reused` and a cached analysis as `analysis.cached`, which is the point
# of those names. So an exact count of `stems` reads zero on a track whose
# stems were reused — printed as coverage below, not treated as a run.
ONCE_PER_TRACK = ("download", "quick", "stems", "analysis", "structure", "allin1")

# `<stage>` → the grps that mean "it did not have to do the work this time".
SKIPPED_SUFFIXES = (".cached", ".reused")


def _say(text: str, stream=None) -> None:
    """Print, surviving a console that cannot encode the report (an OEM
    codepage has no em dash, and cp437 no ×). Both files are written before
    this runs, so a failure here would throw away nothing but the echo — and
    still end a multi-hour run in a traceback (the same fix as
    bench_allin1_models.py's `_say`)."""
    out = stream if stream is not None else sys.stdout
    try:
        out.write(text + "\n")
    except UnicodeEncodeError:
        out.write(text.encode("ascii", "replace").decode("ascii") + "\n")
    out.flush()


def _ingest(folder: Path, limit: int = 0,
            db_path: Optional[Path] = None) -> List[int]:
    """One song row per file, already 'downloaded' — raw_path points at the file
    in place, so nothing is copied and nothing is fetched. One connection for
    the batch (models.connection)."""
    from database import models

    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in AUDIO_SUFFIXES)
    if limit:
        files = files[:limit]
    ids = []
    with models.connection(db_path if db_path is not None else models.DB_PATH) as conn:
        for path in files:
            title, artist = path.stem, ""
            if " - " in title:
                artist, title = (s.strip() for s in title.split(" - ", 1))
            ids.append(models.upsert_song(
                title=title, artist=artist, source_url=path.resolve().as_uri(),
                raw_path=str(path), status="downloaded", source="local",
                conn=conn))
    return ids


def _has_column(conn, table: str, column: str) -> bool:
    """PRAGMA, not a try/except around the SELECT: SQLite resolves column names
    when it PREPARES the statement, so a missing column raises before any row
    is read and no COALESCE can save it (readme §7, the same class of bug as
    the `sections_allin1` guard)."""
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def _read_runs(conn, song_ids: List[int]) -> Tuple[Dict[str, List[dict]], bool]:
    """`analysis_runs` for these songs, grouped by grp, and whether `peak_mb`
    was there to read. Without the column every row reports it as None —
    unmeasured, never 0 (§5 conventions)."""
    peak_measured = _has_column(conn, "analysis_runs", "peak_mb")
    peak = ", peak_mb" if peak_measured else ""
    marks = ",".join("?" * len(song_ids))
    rows = conn.execute(
        f"SELECT song_id, grp, stem_type, ms, audio_secs, ok, analyzer{peak} "
        f"FROM analysis_runs WHERE song_id IN ({marks}) ORDER BY id",
        list(song_ids)).fetchall()
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        row = dict(r)
        row.setdefault("peak_mb", None)
        by_group[row["grp"]].append(row)
    return dict(by_group), peak_measured


def _timings(song_ids: List[int],
             db_path: Optional[Path] = None) -> Tuple[Dict[str, List[dict]], bool]:
    from database import models

    conn = models.get_conn(db_path) if db_path is not None else models.get_conn()
    try:
        return _read_runs(conn, song_ids)
    finally:
        conn.close()


def _did_no_work(grp: str) -> bool:
    return grp.endswith(SKIPPED_SUFFIXES)


def _stage_records(by_group: Dict[str, List[dict]],
                   song_ids: List[int]) -> List[dict]:
    """One record per grp, heaviest first.

    A cached or reused run has NO real-time factor. It is a row that says the
    work was not done — `stems.reused` is a 20 ms file check against a 200 s
    track, `analysis.cached` is 0 ms — so folding it into a median would price
    a stage by how often it was skipped. The runs are still counted and still
    timed; only the RTF is left out, which is the honest answer."""
    recs = []
    tracks = max(1, len(song_ids))
    for grp, rows in sorted(by_group.items(),
                            key=lambda kv: -sum(r["ms"] or 0 for r in kv[1])):
        ms = [r["ms"] or 0 for r in rows]
        rtf = [] if _did_no_work(grp) else [
            (r["ms"] / 1000.0) / r["audio_secs"] for r in rows
            if r["audio_secs"] and r["ms"]]
        peaks = [r["peak_mb"] for r in rows if r.get("peak_mb")]
        recs.append({
            "group": grp,
            "runs": len(rows),
            "runs_per_track": round(len(rows) / tracks, 2),
            "median_ms": round(st.median(ms)) if ms else 0,
            "total_s": round(sum(ms) / 1000.0, 1),
            "median_rtf": round(st.median(rtf), 3) if rtf else None,
            "peak_mb": round(max(peaks), 1) if peaks else None,
            "failed": sum(1 for r in rows if not r["ok"]),
            "did_no_work": _did_no_work(grp),
        })
    return recs


def _per_song(rows: List[dict]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for r in rows:
        counts[r["song_id"]] += 1
    return dict(counts)


def _name_songs(counts: Dict[int, int], cap: int = 12) -> str:
    """"song 1×3, song 7×2" — the tracks, not just the total."""
    worst = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    named = ", ".join(f"song {s}×{n}" for s, n in worst[:cap])
    if len(worst) > cap:
        named += f", and {len(worst) - cap} more"
    return named


def _findings(by_group: Dict[str, List[dict]], song_ids: List[int]) -> List[str]:
    """Did anything run twice? This is the claim the script exists to make, so
    it has to be able to fail — see ONCE_PER_TRACK."""
    findings: List[str] = []
    for grp in ONCE_PER_TRACK:
        twice = {s: n for s, n in _per_song(by_group.get(grp, [])).items() if n > 1}
        if twice:
            findings.append(
                f"`{grp}` ran more than once for {len(twice)} of "
                f"{len(song_ids)} track(s): {_name_songs(twice)}")
    cuts = by_group.get("structure", [])
    if len(cuts) > len(song_ids):
        by_analyzer = defaultdict(int)
        for r in cuts:
            by_analyzer[r.get("analyzer") or "unknown"] += 1
        split = ", ".join(f"{n} {a}" for a, n in sorted(by_analyzer.items()))
        findings.append(
            f"{len(cuts)} cuts for {len(song_ids)} tracks ({split}) — check "
            f"pipeline_worker._wants_essentia_cut")
    demix = by_group.get("allin1.demix", [])
    if demix:
        findings.append(
            f"`allin1.demix` ran {len(demix)} time(s) — allin1 separated its own "
            f"stems, so Demucs ran twice — the four sources were not handed "
            f"over (analysis/allin1_runner.hand_over_demix).")
    return findings


def _out_dir(data_dir: Path) -> Path:
    """`<data_dir>/bench/one_pass/<timestamp>`."""
    return Path(data_dir) / "bench" / "one_pass" / time.strftime("%Y%m%d-%H%M%S")


def _write_report(out_dir: Path, recs: List[dict], lines: List[str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "one_pass.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]) if recs else ["group"])
        w.writeheader()
        w.writerows(recs)
    text = "\n".join(lines) + "\n"
    (out_dir / "one_pass.md").write_text(text, encoding="utf-8")


def _library_songs(limit: int, db_path: Optional[Path]) -> List[int]:
    """The tracks already in THIS database, bounded by --limit.

    Unbounded this is a loaded gun: pointed at the 206-track library it would
    re-run the whole pipeline over all of it. The test library is a separate
    MASHUP_DB_PATH (readme §4), and the caller is told which one it got before
    anything runs."""
    from database import models

    conn = models.get_conn(db_path) if db_path is not None else models.get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        sql = "SELECT id FROM songs ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        ids = [r["id"] for r in conn.execute(sql)]
    finally:
        conn.close()
    print(f"{total} track(s) in the database; running {len(ids)}"
          + ("" if limit else " — all of them (no --limit given)"))
    return ids


def _song_state(song_id: int, db_path: Path) -> dict:
    from database import models

    return models.get_song(song_id, db_path=db_path) or {}


def _report_lines(*, song_ids, recs, findings, wall, frames_n, frames_bytes,
                  peak_measured, by_group, pipeline, config) -> List[str]:
    tracks = max(1, len(song_ids))
    lines = [f"# One pass over {len(song_ids)} tracks", "",
             f"- wall clock: {wall / 60:.1f} min ({wall / tracks:.0f} s a track)",
             f"- pipeline: {pipeline} · allin1 "
             f"{config.current_allin1_model()}",
             f"- frame sidecars: {frames_n} files, {frames_bytes / 1e6:.1f} MB "
             f"({frames_bytes / tracks / 1e6:.2f} MB a track)",
             "- peak resident MB: " + ("recorded where it was measured (allin1)"
                                       if peak_measured else
                                       "unmeasured — no peak_mb column"),
             ""]
    lines += ["## Findings", ""]
    lines += ([f"- **{f}**" for f in findings] if findings
              else ["- none: every stage ran once per track."])
    lines += ["", "## Once per track", "",
              "| stage | tracks that ran it | runs | reused or cached |",
              "|---|---|---|---|"]
    for grp in ONCE_PER_TRACK:
        rows = by_group.get(grp, [])
        skipped = sum(len(by_group.get(grp + sfx, []))
                      for sfx in SKIPPED_SUFFIXES)
        lines.append(f"| {grp} | {len({r['song_id'] for r in rows})} of "
                     f"{len(song_ids)} | {len(rows)} | {skipped} |")
    lines += ["", "A reused separation or a cached analysis is its own grp, so "
              "an exact count of `stems` reads 0 on a track whose stems were "
              "reused — that is the column beside it, not a missing run.", "",
              "## Per stage", "",
              "| group | runs | runs/track | median ms | total s | median RTF "
              "| peak MB | failed |", "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        rtf = "—" if r["median_rtf"] is None else r["median_rtf"]
        peak = "—" if r["peak_mb"] is None else r["peak_mb"]
        lines.append(f"| {r['group']} | {r['runs']} | {r['runs_per_track']} | "
                     f"{r['median_ms']} | {r['total_s']} | {rtf} | {peak} | "
                     f"{r['failed']} |")
    return lines


def main(argv=None, *, db_path: Optional[Path] = None,
         data_dir: Optional[Path] = None) -> int:
    """`db_path` / `data_dir` are for the tests, which pass them explicitly:
    `config.DB_PATH` binds at import, so a test that redirected it by env var
    would run against whichever library is configured (readme §4, §7)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", nargs="?", default=None,
                    help="a folder of audio files (registered in place)")
    ap.add_argument("--limit", type=int, default=0,
                    help="at most this many tracks")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="re-measure the tracks already in this library")
    args = ap.parse_args(argv)
    if not args.skip_ingest and not args.folder:
        ap.error("give a folder of audio files, or --skip-ingest")

    import config
    from api import jobs
    from api.workers import pipeline_worker
    from database import alignment

    db = Path(db_path) if db_path is not None else Path(config.DB_PATH)
    data = Path(data_dir) if data_dir is not None else Path(config.DATA_DIR)
    pipeline = config.current_separation_pipeline()

    print(f"Database: {db}")
    print(f"Data dir: {data}")
    print(f"Pipeline: {pipeline} · allin1 model: "
          f"{config.current_allin1_model()}")

    if args.skip_ingest:
        song_ids = _library_songs(args.limit, db)
    else:
        song_ids = _ingest(Path(args.folder), args.limit, db_path=db)
    if not song_ids:
        print("No audio files found." if not args.skip_ingest
              else "No tracks in this database.")
        return 1
    print(f"{len(song_ids)} track(s): {song_ids}")

    wall0 = time.perf_counter()
    for sid in song_ids:
        job_id = jobs.new_job("pipeline", song_id=sid)
        t0 = time.perf_counter()
        pipeline_worker.run(job_id, sid)
        row = _song_state(sid, db)
        print(f"  song {sid}: {row.get('status')} / allin1 "
              f"{row.get('allin1_state')} in {time.perf_counter() - t0:.0f} s"
              + (f" — {row['last_error']}" if row.get("last_error") else ""))
    wall = time.perf_counter() - wall0

    by_group, peak_measured = _timings(song_ids, db_path=db)
    recs = _stage_records(by_group, song_ids)
    findings = _findings(by_group, song_ids)

    frames_n = frames_bytes = 0
    for sid in song_ids:
        for row in alignment.get_frames(sid, db_path=db):
            frames_n += 1
            frames_bytes += row.get("bytes") or 0

    lines = _report_lines(song_ids=song_ids, recs=recs, findings=findings,
                          wall=wall, frames_n=frames_n,
                          frames_bytes=frames_bytes,
                          peak_measured=peak_measured, by_group=by_group,
                          pipeline=pipeline, config=config)
    out = _out_dir(data)
    _write_report(out, recs, lines)
    _say("\n".join(lines))
    _say(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
