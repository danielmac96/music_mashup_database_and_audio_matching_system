#!/usr/bin/env python3
"""
test_flow.py — Mashup engine entry point.

Usage:
    # Full pipeline from a SoundCloud playlist:
    python test_flow.py --url https://soundcloud.com/user/sets/playlist-name

    # Run only specific stages:
    python test_flow.py --url URL --stages ingest download

    # Resume from a specific stage (songs already downloaded):
    python test_flow.py --stages stems analysis match

    # Match against a different seed song:
    python test_flow.py --stages match --seed 2

    # Inspect the database:
    python test_flow.py --db-report

    # Reset database and start fresh:
    python test_flow.py --url URL --reset
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

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
                   help="Run only these stages (default: all)")
    p.add_argument("--seed-stem", default="vocals",
                   choices=["full", "vocals", "instrumental"],
                   help="Which stem to use for the seed song (default: vocals)")
    p.add_argument("--cand-stem", default="instrumental",
                   choices=["full", "vocals", "instrumental"],
                   help="Which stem to compare candidates against (default: instrumental)")
    p.add_argument("--reset", action="store_true",
                   help="Delete the database before running")
    p.add_argument("--db-report", action="store_true",
                   help="Print a summary of the current database and exit")
    return p.parse_args()


def print_db_report():
    from database.models import get_conn, init_db
    init_db()
    conn = get_conn()
    songs = conn.execute("SELECT * FROM songs ORDER BY id").fetchall()
    print(f"\n{'='*60}")
    print(f"  Database Report — {len(songs)} songs")
    print(f"{'='*60}")
    for s in songs:
        feat_rows = conn.execute(
            "SELECT stem_type, bpm, key, mode, camelot FROM features WHERE song_id=?",
            (s["id"],)
        ).fetchall()
        print(f"\n  [{s['id']:>2}] {s['title']} — {s['artist']}")
        print(f"       Status: {s['status']}  |  Genre: {s['genre'] or '—'}")
        for f in feat_rows:
            print(f"       [{f['stem_type']:>12}] BPM={f['bpm']}  "
                  f"Key={f['key']} {f['mode']}  Camelot={f['camelot']}")
    conn.close()
    print()


def main():
    args   = parse_args()
    stages = set(args.stages) if args.stages else None

    print("\n" + "═" * 60)
    print("  Mashup Engine")
    if args.url:
        print(f"  Playlist: {args.url}")
    print("═" * 60 + "\n")

    if args.reset:
        from config import DB_PATH
        if DB_PATH.exists():
            DB_PATH.unlink()
            log.info("Database reset.")

    if args.db_report:
        print_db_report()
        return

    from database.models import init_db
    init_db()
    log.info("Database initialised")

    from pipeline import run_ingest, run_download, run_stems, run_analysis, run_match

    if stages is None or "ingest" in stages:
        if not args.url:
            log.error("--url is required for the ingest stage.")
            sys.exit(1)
        song_ids = run_ingest(playlist_url=args.url)
        if not song_ids:
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

    if stages is None or "match" in stages:
        run_match(
            seed_song_id=args.seed,
            seed_stem=args.seed_stem,
            candidate_stem=args.cand_stem,
        )
    else:
        log.info("Skipping match stage")

    print_db_report()
    print("✓ Done.\n")


if __name__ == "__main__":
    main()