#!/usr/bin/env python3
"""
test_flow.py — Mashup engine entry point.

Usage:
    # Full pipeline from a SoundCloud playlist (one-command MVP):
    python test_flow.py --url https://soundcloud.com/user/sets/playlist-name

    # Run only specific stages:
    python test_flow.py --url URL --stages ingest download

    # Resume from a specific stage (songs already downloaded):
    python test_flow.py --stages stems analysis

    # Match against a different seed song:
    python test_flow.py --stages match --seed 2

    # Inspect the database:
    python test_flow.py --db-report

    # Reset database and start fresh:
    python test_flow.py --url URL --reset

    # Point the audio library / database at a different location:
    python test_flow.py --url URL --audio-root D:/music_lib --db-path D:/library.db
"""
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout/stderr so the box-drawing chars below render on Windows
# without requiring users to set PYTHONIOENCODING by hand.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Mashup engine")
    p.add_argument("--url", default="",
                   help="SoundCloud playlist URL")
    p.add_argument("--seed", type=int, default=1,
                   help="Song ID to use as mashup seed (default: 1)")
    p.add_argument("--stages", nargs="*",
                   choices=["ingest", "download", "stems", "analysis", "match"],
                   default=None,
                   help="Run only these stages (default: all of ingest/download/stems/analysis)")
    p.add_argument("--seed-stem", default="vocals",
                   choices=["full", "vocals", "instrumental"],
                   help="Which stem to use for the seed song (default: vocals)")
    p.add_argument("--cand-stem", default="instrumental",
                   choices=["full", "vocals", "instrumental"],
                   help="Which stem to compare candidates against (default: instrumental)")
    p.add_argument("--reset", action="store_true",
                   help="Delete the database before running")
    p.add_argument("--reverify", action="store_true",
                   help="Re-check downloaded tracks for stale ~30s SoundCloud Go+ "
                        "previews; re-download full versions and re-stem/analyse them.")
    p.add_argument("--db-report", action="store_true",
                   help="Print a summary of the current database and exit")
    p.add_argument("--audio-root", metavar="DIR", default=None,
                   help="Override the audio library root (default: <repo>/audio). "
                        "Sets MASHUP_AUDIO_ROOT env var for downstream modules.")
    p.add_argument("--db-path", metavar="PATH", default=None,
                   help="Override the SQLite database file location "
                        "(default: <repo>/mashup.db). Sets MASHUP_DB_PATH env var.")
    p.add_argument("--export-mashups", metavar="FILE", nargs="?", const="mashup_report",
                   default=None,
                   help="Export ranked mashup report as FILE.csv + FILE.txt "
                        "(default stem: 'mashup_report')")
    p.add_argument("--prep-session", metavar="DIR", nargs="?", const="fl_session",
                   default=None,
                   help="Create FL Studio session folders in DIR "
                        "(default: 'fl_session/')")
    p.add_argument("--top-n", type=int, default=20,
                   help="Number of top pairs to include in export/prep (default: 20)")
    return p.parse_args()


def _apply_path_overrides(args) -> None:
    """Translate --audio-root / --db-path into env vars BEFORE any module
    imports config (config.py reads these at import time)."""
    if args.audio_root:
        os.environ["MASHUP_AUDIO_ROOT"] = str(Path(args.audio_root).expanduser().resolve())
    if args.db_path:
        os.environ["MASHUP_DB_PATH"] = str(Path(args.db_path).expanduser().resolve())


def print_db_report():
    from database.models import get_conn, init_db, count_songs_by_status
    init_db()

    status_counts = count_songs_by_status()
    total = sum(status_counts.values())

    conn = get_conn()
    songs = conn.execute("SELECT * FROM songs ORDER BY id").fetchall()
    print(f"\n{'='*60}")
    print(f"  Database Report — {total} songs")
    if status_counts:
        breakdown = ", ".join(f"{n} {s}" for s, n in sorted(status_counts.items()))
        print(f"  Status: {breakdown}")
    print(f"{'='*60}")
    for s in songs:
        feat_rows = conn.execute(
            """SELECT stem_type, bpm, bpm_confidence, key, mode, camelot,
                      loudness_rms, energy, spectral_centroid,
                      spectral_rolloff, zero_crossing_rate, mfcc_json
               FROM features WHERE song_id=? ORDER BY stem_type""",
            (s["id"],)
        ).fetchall()
        print(f"\n  [{s['id']:>2}] {s['title']} — {s['artist']}")
        print(f"       Status: {s['status']}  |  Genre: {s['genre'] or '—'}")
        print(f"       URL:    {s['source_url']}")
        dur_s = s["duration_secs"] if s["duration_secs"] is not None else None
        d_str = s["duration_str"] or (f"{dur_s:.0f}s" if dur_s else "—")
        print(
            f"       Meta:   track_id={s['track_id'] or '—'}  "
            f"length={d_str}  "
            f"plays={s['plays']!s}  likes={s['likes']!s}  "
            f"upload={s['upload_date'] or '—'}"
        )
        for f in feat_rows:
            import json
            mfcc = json.loads(f['mfcc_json']) if f['mfcc_json'] else []
            print(f"\n       [{f['stem_type']:>12}]")
            print(f"         Tempo:    BPM={f['bpm']}  confidence={f['bpm_confidence']:.3f}")
            print(f"         Harmony:  Key={f['key']} {f['mode']}  Camelot={f['camelot']}")
            print(f"         Dynamics: RMS={f['loudness_rms']}  energy={f['energy']}")
            print(f"         Spectral: centroid={f['spectral_centroid']}  "
                  f"rolloff={f['spectral_rolloff']}  ZCR={f['zero_crossing_rate']}")
            if mfcc:
                print(f"         MFCC:     {[round(v,1) for v in mfcc]}")
    conn.close()

    # Mashup candidates summary (only print when populated; out of MVP scope)
    conn = get_conn()
    for combo in ("vocal_over_instrumental", "instrumental_over_instrumental"):
        label = "Vocals → Instrumental" if combo == "vocal_over_instrumental" \
                else "Instrumental → Instrumental"
        candidates = conn.execute(
            "SELECT * FROM mashup_candidates WHERE combo_type=? ORDER BY score_total DESC LIMIT 20",
            (combo,)
        ).fetchall()
        if candidates:
            print(f"\n{'='*60}")
            print(f"  {label}  ({len(candidates)} qualifying pairs)")
            print(f"{'='*60}")
            for c in candidates:
                print(f"\n  Score: {c['score_total']:.3f}  "
                      f"BPM={c['score_bpm']:.2f}  Key={c['score_key']:.2f}  "
                      f"Energy={c['score_energy']:.2f}  Timbre={c['score_timbre']:.2f}")
                print(f"    TOP: {c['vocal_title']} — {c['vocal_artist']}"
                      f"  [{c['vocal_bpm']} BPM  {c['vocal_camelot']}]")
                print(f"    BED: {c['inst_title']} — {c['inst_artist']}"
                      f"  [{c['inst_bpm']} BPM  {c['inst_camelot']}]")
    conn.close()
    print()


def _exit_code_from_db() -> int:
    """0 if at least one song reached 'analysed', else 1."""
    from database.models import count_songs_by_status
    counts = count_songs_by_status()
    return 0 if counts.get("analysed", 0) > 0 else 1


def main():
    args = parse_args()
    _apply_path_overrides(args)

    stages = set(args.stages) if args.stages else None

    print("\n" + "═" * 60)
    print("  Mashup Engine")
    if args.url:
        print(f"  Playlist: {args.url}")
    if args.audio_root:
        print(f"  Audio root: {os.environ['MASHUP_AUDIO_ROOT']}")
    if args.db_path:
        print(f"  DB path:    {os.environ['MASHUP_DB_PATH']}")
    print("═" * 60 + "\n")

    if args.reset:
        # Late import so any --db-path env var is already in place.
        from config import DB_PATH
        if DB_PATH.exists():
            DB_PATH.unlink()
            log.info("Database reset.")

    if args.db_report:
        print_db_report()
        return

    # ── Export-only mode (no pipeline stages, no URL) ─────────────────────────
    if (args.export_mashups is not None or args.prep_session is not None) \
            and not args.url and not args.stages:
        from config import DB_PATH
        from matcher.match import export_mashup_report, prep_fl_session
        if args.export_mashups is not None:
            export_mashup_report(DB_PATH, args.export_mashups, top_n=args.top_n)
        if args.prep_session is not None:
            prep_fl_session(DB_PATH, args.prep_session, top_n=args.top_n)
        return

    from database.models import init_db
    init_db()
    log.info("Database initialised")

    from pipeline import (run_ingest, run_download, run_stems, run_analysis,
                          run_match, run_reverify)

    # ── Re-verify mode: fix stale previews, then reprocess any swapped tracks ──
    if args.reverify:
        run_reverify()
        run_stems()
        run_analysis()
        print_db_report()
        sys.exit(_exit_code_from_db())

    if stages is None or "ingest" in stages:
        if not args.url:
            log.error("--url is required for the ingest stage.")
            sys.exit(1)
        song_ids = run_ingest(playlist_url=args.url)
        if not song_ids and stages is None:
            log.error("No songs ingested. Check the playlist URL.")
            sys.exit(1)
    else:
        log.info("Skipping ingest stage")

    if stages is None or "download" in stages:
        run_download()
    else:
        log.info("Skipping download stage")

    if stages is None or "stems" in stages:
        run_stems()
    else:
        log.info("Skipping stems stage")

    if stages is None or "analysis" in stages:
        run_analysis()
    else:
        log.info("Skipping analysis stage")

    if stages is not None and "match" in stages:
        # match is opt-in: only runs when explicitly requested via --stages.
        run_match(
            seed_song_id=args.seed,
            seed_stem=args.seed_stem,
            candidate_stem=args.cand_stem,
        )

    print_db_report()

    # ── Optional exports (run after pipeline or standalone) ───────────────────
    if args.export_mashups is not None:
        from config import DB_PATH
        from matcher.match import export_mashup_report
        log.info(f"Exporting mashup report → {args.export_mashups}.csv / .txt")
        export_mashup_report(DB_PATH, args.export_mashups, top_n=args.top_n)

    if args.prep_session is not None:
        from config import DB_PATH
        from matcher.match import prep_fl_session
        log.info(f"Preparing FL Studio session → {args.prep_session}/")
        prep_fl_session(DB_PATH, args.prep_session, top_n=args.top_n)

    code = _exit_code_from_db()
    if code == 0:
        print("✓ Done.\n")
    else:
        print("✗ No tracks reached 'analysed'. See logs above.\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
