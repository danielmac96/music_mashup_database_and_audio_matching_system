"""scripts/db_report.py — what the database is made of, and how it is read.

The before/after evidence for the store work: byte weight per table and per
column, the schema version, and for each query the app actually runs hot, its
plan and its wall time. Read-only; safe against the live database.

    python scripts/db_report.py
    docker compose exec app python scripts/db_report.py --db /data/mashup.db
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Run as `python scripts/db_report.py`, sys.path[0] is scripts/, so the
# `database` and `config` packages at the repo root are not importable without this.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The queries whose cost the app pays over and over. Keep this list honest: a
# query that is not here is a query nobody is watching.
HOT: Tuple[Tuple[str, str, tuple], ...] = (
    ("dock, scoped to a track",
     "SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? "
     "ORDER BY score_total DESC LIMIT 40", ("vocal_over_instrumental", 1)),
    ("dock, library-wide",
     "SELECT * FROM mashup_candidates WHERE combo_type=? "
     "ORDER BY score_total DESC LIMIT 40", ("vocal_over_instrumental",)),
    ("pairings of two songs",
     "SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? "
     "AND inst_song_id=?", ("vocal_over_instrumental", 1, 2)),
    ("features, every full mix",
     "SELECT * FROM features f WHERE f.stem_type='full'", ()),
    ("sections of one song",
     "SELECT * FROM sections WHERE song_id=?", (1,)),
    ("library listing",
     "SELECT * FROM songs ORDER BY id DESC LIMIT 200", ()),
)


def _columns(conn, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info([{table}])")]


def _open_read_only(db: str) -> sqlite3.Connection:
    # `mode=ro` so this script cannot write, checkpoint or create anything, even
    # by accident: the live database is WAL-mode, and a plain connect would
    # checkpoint the WAL on close. A read-only URI needs a file: URI.
    uri = Path(db).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="")
    ap.add_argument("--top", type=int, default=6,
                    help="heaviest columns to print per table")
    args = ap.parse_args(argv)

    db = args.db
    if not db:
        import config
        db = str(config.DB_PATH)
    if not os.path.isfile(db):
        print(f"no database at {db}", file=sys.stderr)
        return 1
    print(f"database: {db}")
    print(f"file: {os.path.getsize(db) / 1e6:.1f} MB")

    conn = _open_read_only(db)
    conn.row_factory = sqlite3.Row
    try:
        from database import schema
        print(f"schema version: {schema.current_version(conn)} "
              f"(current {schema.SCHEMA_VERSION})")
    except Exception as exc:  # noqa: BLE001 — a pre-ledger database
        print(f"schema version: unknown ({exc})")

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

    weights = []
    for t in tables:
        cols = _columns(conn, t)
        if not cols:
            continue
        expr = "+".join(f"coalesce(length([{c}]),0)" for c in cols)
        total = conn.execute(f"SELECT coalesce(sum({expr}),0) FROM [{t}]").fetchone()[0]
        rows = conn.execute(f"SELECT count(*) FROM [{t}]").fetchone()[0]
        weights.append((t, rows, total))

    print("\n── bytes by table ──")
    for t, rows, total in sorted(weights, key=lambda w: -w[2]):
        if not total:
            continue
        print(f"  {t:22s} rows={rows:8d} {total / 1e6:8.2f} MB")

    print("\n── heaviest columns ──")
    for t, _rows, total in sorted(weights, key=lambda w: -w[2])[:5]:
        if total < 1e5:
            continue
        sizes = []
        for c in _columns(conn, t):
            b = conn.execute(
                f"SELECT coalesce(sum(length([{c}])),0) FROM [{t}]").fetchone()[0]
            sizes.append((c, b))
        print(f"  {t}:")
        for c, b in sorted(sizes, key=lambda s: -s[1])[:args.top]:
            print(f"     {c:28s} {b / 1e6:7.2f} MB")

    print("\n── hot queries ──")
    for label, sql, params in HOT:
        try:
            plan = [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params)]
            t0 = time.perf_counter()
            n = len(conn.execute(sql, params).fetchall())
            ms = (time.perf_counter() - t0) * 1000.0
        except sqlite3.Error as exc:
            print(f"  {label}: {exc}")
            continue
        scan = any(p.startswith("SCAN") for p in plan)
        sort = any("TEMP B-TREE" in p for p in plan)
        flags = " ".join(x for x in ("SCAN" if scan else "",
                                     "TEMPSORT" if sort else "") if x)
        print(f"  {ms:8.1f} ms  n={n:6d}  {label}{('  ⚠ ' + flags) if flags else ''}")
        for p in plan:
            print(f"             {p}")
    conn.close()
    return 0


if __name__ == "__main__":
    # The report prints box-drawing and warning glyphs. A redirected stdout on
    # Windows is cp1252 and would raise UnicodeEncodeError halfway through, so
    # the report would be cut off at its most useful line.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
