"""
database/schema.py — the numbered migration ledger.

``PRAGMA user_version`` says which schema a database is at. Migration 7 is the
baseline: ``database/schema.sql``, the whole schema, created in one step on an
empty database. (Versions 1–6 were the history of an earlier build; that
history was squashed on 2026-11-06, and a database still at one of them is
refused with a "start a fresh library" message rather than half-migrated.)

Adding a migration: append a ``Migration`` with the next version, a name that
says what it does, and an ``up`` that is safe to run on a database holding real
rows. Four rules, all load-bearing:
  * a shape change belongs only in ``MIGRATIONS`` — never in schema.sql, which
    only an empty database ever reads;
  * a shipped migration is never edited;
  * each is idempotent;
  * none calls ``executescript``, which commits early and so breaks the
    per-migration transaction ``apply`` gives it.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

log = logging.getLogger(__name__)

SCHEMA_SQL = Path(__file__).with_name("schema.sql")

# Versions an earlier build wrote. A database at one of them predates the
# one-pass pipeline and cannot be read by this one.
RETIRED_VERSIONS = range(1, 7)


class RetiredSchema(RuntimeError):
    """The database was made by a build before the one-pass pipeline."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    up: Callable[[sqlite3.Connection], None]


def statements(sql: str) -> List[str]:
    """Split SQL into complete statements, so they can run one at a time
    inside a transaction (never ``executescript``)."""
    out, buf = [], ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                out.append(buf)
            buf = ""
    return out


def _baseline(conn: sqlite3.Connection) -> None:
    """Every table and index, from schema.sql. Runs only on an empty database."""
    for statement in statements(SCHEMA_SQL.read_text(encoding="utf-8")):
        conn.execute(statement)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(7, "baseline", _baseline),
)

SCHEMA_VERSION = MIGRATIONS[-1].version


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply(conn: sqlite3.Connection) -> List[int]:
    """Run every migration above the stored version. Returns the versions
    applied, in order.

    Each runs in its own transaction with ``user_version`` set inside it, so a
    failure rolls that migration back and leaves the stored version on the last
    one that succeeded — the next start retries exactly the one that broke. The
    error names the failed version and chains the original exception. Only an
    ``Exception`` is wrapped that way; a ``KeyboardInterrupt`` or ``SystemExit``
    rolls back and propagates as itself.

    Why not SAVEPOINTs: Python's sqlite3 opens implicit transactions of its own
    and ``executescript`` COMMITs whatever is pending, which silently ends a
    savepoint. So ``isolation_level`` is switched off for the duration and the
    transaction is explicit.

    ``PRAGMA user_version`` cannot be parameterised, which is why it is
    formatted; the value is an int from this module's own table.
    """
    at = current_version(conn)
    if at in RETIRED_VERSIONS:
        raise RetiredSchema(
            f"This library's database (schema version {at}) was made by an "
            "older build of Mashup Engine, before the one-pass pipeline. Start "
            "a fresh library: point MASHUP_DB_PATH / MASHUP_AUDIO_ROOT / "
            "MASHUP_DATA_DIR at empty folders, or move ./data aside.")
    done: List[int] = []
    previous = conn.isolation_level
    conn.isolation_level = None
    try:
        for migration in MIGRATIONS:
            if migration.version <= at:
                continue
            log.info("schema: applying %d (%s)", migration.version, migration.name)
            conn.execute("BEGIN")
            try:
                migration.up(conn)
                conn.execute(f"PRAGMA user_version = {int(migration.version)}")
                conn.execute("COMMIT")
            except Exception as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise RuntimeError(
                    f"schema migration {migration.version} ({migration.name}) "
                    f"failed: {exc}") from exc
            except BaseException:
                # Ctrl-C or SystemExit part-way through a long migration: undo
                # it, but do not rename it "migration N failed" — it did not.
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            done.append(migration.version)
    finally:
        conn.isolation_level = previous
    return done


def shape(conn: sqlite3.Connection) -> Dict[str, dict]:
    """The database's shape, normalised for comparison: every table's columns
    with their declared type, NOT NULL and primary-key flags, and every named
    index's SQL with whitespace collapsed. SQLite's own auto-indexes and
    internal tables are left out — they follow from the declarations."""
    tables: Dict[str, list] = {}
    for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
        tables[name] = [
            (r[1], (r[2] or "").upper(), int(r[3]), int(r[5]))
            for r in conn.execute(f"PRAGMA table_info([{name}])")]
    indexes: Dict[str, str] = {}
    for name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_autoindex_%' ORDER BY name").fetchall():
        indexes[name] = " ".join((sql or "").split())
    return {"tables": tables, "indexes": indexes}
