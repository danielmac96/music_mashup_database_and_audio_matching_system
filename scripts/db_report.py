"""scripts/db_report.py — what the database is made of, and how it is read.

The before/after evidence for the store work: byte weight per table and per
column, the schema version, and for each query the app actually runs hot, its
plan and its wall time. It never writes to the database it reports on: the
connection is opened `mode=ro`. Without --db it imports `config` to find the
live path, and that import creates the audio and data folders if they are
missing; the database file is not written either way.

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
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

# Run as `python scripts/db_report.py`, sys.path[0] is scripts/, so the
# `database` and `config` packages at the repo root are not importable without this.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COMBO = "vocal_over_instrumental"


class Hot(NamedTuple):
    label: str
    sql: str
    # Gets the probe ids chosen by _probe_ids and returns the bound parameters.
    params: Callable[[Dict[str, int]], tuple]
    # Why a full SCAN of this query is correct, or "" (the default), in which
    # case a SCAN is printed as a finding. Give a reason only when it is real.
    expect_scan: str = ""


# The queries whose cost the app pays over and over. Keep this list honest: a
# query that is not here is a query nobody is watching.
HOT: Tuple[Hot, ...] = (
    Hot("dock, scoped to a track",
        "SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? "
        "ORDER BY score_total DESC LIMIT 40",
        lambda p: (COMBO, p["vocal"])),
    Hot("dock, library-wide",
        "SELECT * FROM mashup_candidates WHERE combo_type=? "
        "ORDER BY score_total DESC LIMIT 40",
        lambda p: (COMBO,)),
    Hot("pairings of two songs",
        "SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? "
        "AND inst_song_id=?",
        lambda p: (COMBO, p["vocal"], p["inst"])),
    Hot("features, every full mix",
        "SELECT * FROM features f WHERE f.stem_type='full'",
        lambda p: ()),
    Hot("sections of one song",
        "SELECT * FROM sections WHERE song_id=?",
        lambda p: (p["song"],)),
    # library listing: "SCAN songs" is correct here, so its SCAN is silenced on
    # purpose and only here. The plan is a reverse rowid walk: SQLite reads
    # songs from the top and stops after 200 rows, with no sort and no index
    # that could help. Without this exception the warning fires on every
    # database, and readers learn to ignore the glyph the other rows need them
    # to notice. The wall time is still reported. Keep the exception on this
    # entry alone, and do not add a global "ignore SCAN" to silence the flag.
    Hot("library listing",
        "SELECT * FROM songs ORDER BY id DESC LIMIT 200",
        lambda p: (),
        expect_scan="reverse rowid walk that stops after 200 rows; no sort, no useful index"),
)


def _busiest(conn, sql: str, params: tuple) -> Optional[tuple]:
    try:
        return conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None  # a missing table is a fallback, not a crash


def _probe_ids(conn) -> Tuple[Dict[str, int], List[str]]:
    """Pick probe ids that have rows in this database.

    The hot queries are timed on a real id. On a library where song 1 has no
    candidates, `vocal_song_id=1` returns zero rows in a fraction of a
    millisecond, which reads as a fast query when it is an empty one. So the
    vocal with the most candidate rows, its most common partner, and the song
    with the most sections are chosen from the data. An empty or missing table
    falls back to the old literals, and the output says so.
    """
    probes = {"vocal": 1, "inst": 2, "song": 1}
    lines: List[str] = []

    top = _busiest(conn,
                   "SELECT vocal_song_id, count(*) FROM mashup_candidates "
                   "WHERE combo_type=? GROUP BY vocal_song_id ORDER BY 2 DESC, 1 LIMIT 1",
                   (COMBO,))
    if top:
        probes["vocal"] = top[0]
        lines.append(f"vocal_song_id={top[0]}  most candidate rows ({top[1]})")
        partner = _busiest(conn,
                           "SELECT inst_song_id, count(*) FROM mashup_candidates "
                           "WHERE combo_type=? AND vocal_song_id=? GROUP BY inst_song_id "
                           "ORDER BY 2 DESC, 1 LIMIT 1",
                           (COMBO, top[0]))
        if partner:
            probes["inst"] = partner[0]
            lines.append(f"inst_song_id={partner[0]}  most common partner of that vocal "
                         f"({partner[1]} rows)")
        else:
            lines.append("inst_song_id=2  literal fallback: no partner rows")
    else:
        lines.append("vocal_song_id=1, inst_song_id=2  literal fallback: "
                     "no vocal_over_instrumental rows")

    song = _busiest(conn,
                    "SELECT song_id, count(*) FROM sections GROUP BY song_id "
                    "ORDER BY 2 DESC, 1 LIMIT 1", ())
    if song:
        probes["song"] = song[0]
        lines.append(f"song_id={song[0]}  most section rows ({song[1]})")
    else:
        lines.append("song_id=1  literal fallback: sections empty or missing")
    return probes, lines


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
    probes, probe_lines = _probe_ids(conn)
    print("  probe ids, chosen from this database:")
    for line in probe_lines:
        print(f"    {line}")
    for h in HOT:
        try:
            params = h.params(probes)
            plan = [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + h.sql, params)]
            t0 = time.perf_counter()
            n = len(conn.execute(h.sql, params).fetchall())
            ms = (time.perf_counter() - t0) * 1000.0
        except sqlite3.Error as exc:
            print(f"  {h.label}: {exc}")
            continue
        scan = any(p.startswith("SCAN") for p in plan)
        sort = any("TEMP B-TREE" in p for p in plan)
        # A SCAN is a finding unless the entry says why it is correct (see HOT).
        flags = " ".join(x for x in ("SCAN" if scan and not h.expect_scan else "",
                                     "TEMPSORT" if sort else "") if x)
        print(f"  {ms:8.1f} ms  n={n:6d}  {h.label}{('  ⚠ ' + flags) if flags else ''}")
        if n == 0:
            print("             ⚠ 0 rows: this time measured an empty result")
        for p in plan:
            print(f"             {p}")
        if scan and h.expect_scan:
            print(f"             SCAN expected: {h.expect_scan}")
    conn.close()
    return 0


if __name__ == "__main__":
    # The report prints box-drawing and warning glyphs. A redirected stdout on
    # Windows is cp1252 and would raise UnicodeEncodeError halfway through, so
    # the report would be cut off at its most useful line.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
