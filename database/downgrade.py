"""database/downgrade.py — down-migrations (readme §2, "Rollback").

Every migration here undoes one additive migration exactly: the tables it
created are dropped and the columns it added are removed. Nothing that existed
before the migration is touched, so a downgraded database is the database the
older code expects.

    python -m database.downgrade alignment            # undo database/alignment.py
    python -m database.downgrade alignment --db PATH

A downgrade refuses while a song's live rows belong to a pipeline other than
legacy_2stem: those rows would be the only copy left after the parked legacy
rows were dropped. Switch back first (Settings → separation pipeline →
legacy_2stem, which swaps every song), then downgrade. ``get_conn`` re-runs
the up-migration on the next open, so run this with the newer code stopped —
or use it to prove the round trip in a test.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Optional


class DowngradeRefused(RuntimeError):
    pass


def _cols(conn: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def down_alignment(conn: sqlite3.Connection) -> Dict[str, int]:
    """Undo database/alignment.migrate on an open connection (the caller
    commits). Returns {"tables": n dropped, "columns": n dropped}."""
    from database.alignment import ADDED_COLUMNS, TABLES
    if "separation_pipeline" in _cols(conn, "songs"):
        n = conn.execute("SELECT COUNT(*) FROM songs WHERE COALESCE("
                         "separation_pipeline, 'legacy_2stem') != 'legacy_2stem'"
                         ).fetchone()[0]
        if n:
            raise DowngradeRefused(
                f"{n} song(s) are live on a non-legacy pipeline. Switch "
                "separation_pipeline back to legacy_2stem first.")
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND "
                    "name='pair_feedback'").fetchone():
        if "vocal_section" in _cols(conn, "pair_feedback"):
            from database.models import ALLIN1_SECTION_BASE
            n = conn.execute(
                "SELECT COUNT(*) FROM pair_feedback WHERE vocal_section >= ? "
                "OR inst_section >= ?", (ALLIN1_SECTION_BASE, ALLIN1_SECTION_BASE)
            ).fetchone()[0]
            if n:
                raise DowngradeRefused(
                    f"{n} judgement(s) name allin1 sections, which the older "
                    "code cannot resolve. Export or clear them first.")
    dropped_t = 0
    for t in TABLES:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (t,)).fetchone():
            conn.execute(f"DROP TABLE {t}")
            dropped_t += 1
    dropped_c = 0
    for table, col, _decl in ADDED_COLUMNS:
        if col in _cols(conn, table):
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
            dropped_c += 1
    return {"tables": dropped_t, "columns": dropped_c}


DOWNGRADES = {"alignment": down_alignment}


def downgrade(name: str, db_path: Optional[Path] = None) -> Dict[str, int]:
    """Run one named down-migration in a transaction. Opens the database with
    plain sqlite3 — never models.get_conn, which would migrate it up again."""
    if name not in DOWNGRADES:
        raise KeyError(f"unknown migration {name!r} (have: {', '.join(DOWNGRADES)})")
    if db_path is None:
        from config import DB_PATH
        db_path = DB_PATH
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("BEGIN")
        out = DOWNGRADES[name](conn)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    # The next get_conn in this process must re-check the schema.
    from database import models
    models._INITIALIZED_PATHS.discard(str(db_path))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", choices=sorted(DOWNGRADES))
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        out = downgrade(args.name, args.db)
    except DowngradeRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"{args.name}: dropped {out['tables']} table(s), {out['columns']} column(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
