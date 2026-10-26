r"""scripts/ingest_urls.py — build a library from a list of SoundCloud URLs.

One URL a line (blank lines and `#` comments ignored), in the order you want
them. Everything goes through `ingest_rows` — the one ingest implementation the
paste bar, Discover, crates and mix import all use — so this script cannot
drift from the app: dedup, the partial-metadata flag, crate relinking and the
import accounting are all that function's, not a second copy.

With --process it then walks each track through the real pipeline in this
process (download → quick → stems → analyse → allin1 → structure) and prints
what each one cost. Without it the tracks are queued the way an import from the
UI queues them, and the server's own workers pick them up.

    docker compose exec app python scripts/ingest_urls.py /data/urls.txt --process
    docker compose exec app python scripts/ingest_urls.py - --process < urls.txt

To build a library of its own, point the path variables somewhere new:

    docker compose exec \
      -e MASHUP_DB_PATH=/data/test30/mashup.db \
      -e MASHUP_AUDIO_ROOT=/data/test30/audio \
      -e MASHUP_DATA_DIR=/data/test30 \
      app python scripts/ingest_urls.py /data/urls.txt --process
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List


def read_urls(path_or_dash: str) -> List[str]:
    """One URL a line, in order. '-' reads stdin."""
    if path_or_dash == "-":
        text = sys.stdin.read()
    else:
        text = Path(path_or_dash).read_text(encoding="utf-8")
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def main(argv=None) -> int:
    # The docstring (printed by --help) and the progress lines use · → — …; a
    # Windows console on cp1252 would crash on them, so write UTF-8 regardless
    # of the console (same as scripts/db_report.py).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Raw formatter: the example commands in the docstring stay on their own lines.
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", help="a file of URLs, one a line, or - for stdin")
    ap.add_argument("--group", default="", help="also save them as a library group")
    ap.add_argument("--process", action="store_true",
                    help="run the pipeline here instead of leaving it queued")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import config
    from api.routes.playlists import ingest_rows

    urls = read_urls(args.urls)
    if args.limit:
        urls = urls[:args.limit]
    if not urls:
        print("No URLs.")
        return 1

    print(f"database: {config.DB_PATH}")
    print(f"audio:    {config.AUDIO_DIR}")
    print(f"pipeline: {config.current_separation_pipeline()} · engine "
          f"{config.current_segmentation_engine()} · analyser "
          f"{config.current_analyzer()}")
    print(f"{len(urls)} URL(s)\n")

    t0 = time.perf_counter()
    out = ingest_rows([{"source_url": u} for u in urls],
                      group_name=args.group or None)
    print(f"ingest: saved {out['count']}, already here {out['skipped_count']}, "
          f"unusable {out['unresolvable_count']}, partial metadata "
          f"{out['partial_count']}, network fetches {out.get('enriched_count')} "
          f"in {time.perf_counter() - t0:.1f} s")
    for row in out["unresolvable"]:
        print(f"  unusable: {row['title']}")
    for row in out["skipped"]:
        print(f"  already here: {row['title']} (song {row['id']})")

    ids = out["inserted_ids"]
    if not ids or not args.process:
        if ids:
            print("\nQueued. The server's workers will pick them up; watch the "
                  "Queue screen or GET /api/jobs/queue.")
        return 0

    from api import jobs
    from api.workers import pipeline_worker
    from database.models import get_conn

    print(f"\nProcessing {len(ids)} track(s) in this process…")
    wall = time.perf_counter()
    for sid in ids:
        job_id = jobs.new_job("pipeline")
        t = time.perf_counter()
        try:
            pipeline_worker.run(job_id, sid)
        except Exception as exc:  # noqa: BLE001 — one track must not stop the rest
            print(f"  song {sid}: crashed — {type(exc).__name__}: {exc}")
            continue
        conn = get_conn()
        try:
            row = conn.execute("SELECT title, status, last_error, allin1_state "
                               "FROM songs WHERE id=?", (sid,)).fetchone()
        finally:
            conn.close()
        print(f"  song {sid} {row['title'][:38]:38s} {row['status']:10s} "
              f"allin1 {str(row['allin1_state']):7s} {time.perf_counter() - t:6.0f} s"
              + (f"  — {row['last_error']}" if row["last_error"] else ""))
    print(f"\n{len(ids)} track(s) in {(time.perf_counter() - wall) / 60:.1f} min")
    print("Now score the library: POST /api/mashups/score, or\n"
          "  python -c \"from matcher.match import score_all_pairs; "
          "print(score_all_pairs())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
