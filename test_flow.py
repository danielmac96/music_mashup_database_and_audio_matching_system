#!/usr/bin/env python3
"""
test_flow.py — Minimum test flow for the mashup engine.

Runs the full pipeline in mock mode (no network, no GPU required).
Each stage can be skipped independently for targeted testing.

Usage:
    # Full mock run (no dependencies needed):
    python test_flow.py

    # Real SoundCloud playlist:
    python test_flow.py --url https://soundcloud.com/user/playlist-name

    # Run only specific stages:
    python test_flow.py --stages ingest download

    # Match against a different seed:
    python test_flow.py --seed 3

    # Use real audio analysis (requires librosa):
    python test_flow.py --no-mock-analysis

    # Reset database and start fresh:
    python test_flow.py --reset
"""
import argparse
import logging
import sys
import os
from pathlib import Path

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Mashup engine test flow")
    p.add_argument("--url",    default="",
                   help="SoundCloud or YouTube playlist URL (default: mock data)")
    p.add_argument("--seed",   type=int, default=1,
                   help="Song ID to use as mashup seed (default: 1)")
    p.add_argument("--stages", nargs="*",
                   choices=["ingest","download","stems","analysis","match"],
                   default=None,
                   help="Run only these stages (default: all)")
    p.add_argument("--seed-stem", default="vocals",
                   choices=["full","vocals","instrumental"],
                   help="Which stem to use for the seed song")
    p.add_argument("--cand-stem", default="instrumental",
                   choices=["full","vocals","instrumental"],
                   help="Which stem to compare candidates against")
    p.add_argument("--no-mock", action="store_true",
                   help="Disable mock mode (requires yt-dlp, demucs, librosa)")
    p.add_argument("--reset", action="store_true",
                   help="Delete the database before running")
    p.add_argument("--db-report", action="store_true",
                   help="Print a summary of the current database state and exit")
    return p.parse_args()


# ── DB report ─────────────────────────────────────────────────────────────────
def print_db_report():
    from database.models import get_conn, init_db
    init_db()
    conn = get_conn()

    songs = conn.execute("SELECT * FROM songs ORDER BY id").fetchall()
    print(f"\n{'='*60}")
    print(f"  Database Report")
    print(f"{'='*60}")
    print(f"  Songs: {len(songs)}")
    print()

    for s in songs:
        feat_rows = conn.execute(
            "SELECT stem_type, bpm, key, mode, camelot FROM features WHERE song_id=?",
            (s["id"],)
        ).fetchall()
        print(f"  [{s['id']:>2}] {s['title']} — {s['artist']}")
        print(f"       Status: {s['status']}  |  Genre: {s['genre'] or '—'}")
        for f in feat_rows:
            print(f"       [{f['stem_type']:>12}] BPM={f['bpm']}  "
                  f"Key={f['key']} {f['mode']}  Camelot={f['camelot']}")
        print()

    conn.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    use_mock = not args.no_mock
    stages   = set(args.stages) if args.stages else None  # None = all

    print("\n" + "═" * 60)
    print("  Mashup Engine — Test Flow")
    print("  Mode: " + ("MOCK (no downloads/GPU needed)" if use_mock
                         else "REAL (requires yt-dlp, demucs, librosa)"))
    if args.url:
        print(f"  Playlist: {args.url}")
    print("═" * 60 + "\n")

    # ── Optional: reset ───────────────────────────────────────────────────────
    if args.reset:
        from config import DB_PATH
        if DB_PATH.exists():
            DB_PATH.unlink()
            log.info("Database reset.")

    # ── DB report mode ────────────────────────────────────────────────────────
    if args.db_report:
        print_db_report()
        return

    # ── Init DB ───────────────────────────────────────────────────────────────
    from database.models import init_db
    init_db()
    log.info("Database initialised")

    # ── Import pipeline stages ────────────────────────────────────────────────
    from pipeline import (run_ingest, run_download, run_stems,
                           run_analysis, run_match)

    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    if stages is None or "ingest" in stages:
        song_ids = run_ingest(playlist_url=args.url, use_mock=use_mock)
        if not song_ids:
            log.error("No songs ingested. Exiting.")
            sys.exit(1)
    else:
        log.info("Skipping ingest stage")

    # ── Stage 2: Download ─────────────────────────────────────────────────────
    if stages is None or "download" in stages:
        downloaded = run_download(use_mock=use_mock)
        if not downloaded and (stages is None):
            log.warning("No tracks downloaded. Check URLs or use --no-mock with a real playlist.")
    else:
        log.info("Skipping download stage")

    # ── Stage 3: Stems ────────────────────────────────────────────────────────
    if stages is None or "stems" in stages:
        run_stems(use_mock=use_mock)
    else:
        log.info("Skipping stems stage")

    # ── Stage 4: Analysis ─────────────────────────────────────────────────────
    if stages is None or "analysis" in stages:
        run_analysis(use_mock=use_mock)
    else:
        log.info("Skipping analysis stage")

    # ── Stage 5: Match ────────────────────────────────────────────────────────
    if stages is None or "match" in stages:
        results = run_match(
            seed_song_id=args.seed,
            seed_stem=args.seed_stem,
            candidate_stem=args.cand_stem,
        )
        if not results:
            log.warning(
                "No match results. Make sure the seed song has been analysed "
                f"(song_id={args.seed}, stem={args.seed_stem})."
            )
    else:
        log.info("Skipping match stage")

    # ── Final DB report ───────────────────────────────────────────────────────
    print_db_report()

    print("\n✓ Test flow complete.\n")
    print("Next steps:")
    print("  python test_flow.py --db-report          # inspect the database")
    print("  python test_flow.py --seed 3             # try a different seed song")
    print("  python test_flow.py --no-mock --url URL  # run with a real playlist")
    print()


if __name__ == "__main__":
    main()