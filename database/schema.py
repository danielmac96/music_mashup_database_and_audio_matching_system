"""
database/schema.py — the numbered migration ledger.

Before this, the schema was whatever ``models.SCHEMA`` plus twelve
``_migrate_*`` functions and ``alignment.migrate`` happened to produce, each
re-deriving what it needed from ``PRAGMA table_info`` on the first open of a
path. Every one of them is idempotent and they all work; what they could not do
is answer "which schema is this database at", which is the question a library
built from scratch has to be able to answer against one that grew.

So the ledger does not rewrite them. Migration 1 **is** that code: on a
database with ``user_version = 0`` — a blank file or any database that existed
before this file — it runs the whole pre-ledger path, which lands on the same
shape either way (pinned in tests/test_schema_ledger.py). Everything after it
is numbered, forward-only, and runs exactly once.

Adding a migration: append a ``Migration`` with the next version, a name that
says what it does, and an ``up`` that is safe to run on a database holding real
rows. Never edit a migration that has shipped — a database that already ran it
will not run it again, so an edit only changes what *new* databases get, which
is precisely the divergence this file exists to prevent. The same goes for
``models.SCHEMA`` and the ``_migrate_*`` functions: they are migration 1 now,
and a change to them reaches only databases that have not run it yet.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    up: Callable[[sqlite3.Connection], None]


def _baseline(conn: sqlite3.Connection) -> None:
    """Every table and column the app had before the ledger existed.

    Written against a bare connection (positional ``PRAGMA table_info`` rows,
    no ``sqlite3.Row``), which is how the legacy functions already read."""
    from database import alignment, models

    conn.executescript(models.SCHEMA)
    models._migrate_songs_columns(conn)
    models._migrate_features_columns(conn)
    models._migrate_candidates_columns(conn)
    models._migrate_sections_columns(conn)
    models._migrate_candidates_unique_key(conn)
    models._migrate_stems_columns(conn)
    models._migrate_mixtracks_columns(conn)
    models._migrate_mashuppairs_columns(conn)
    models._migrate_crates_columns(conn)
    models._migrate_sets_columns(conn)
    models._migrate_set_items_kinds(conn)
    models._migrate_pair_feedback_key(conn)
    alignment.migrate(conn)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline", _baseline),
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
    error names the failed version and chains the original exception.

    Why not SAVEPOINTs: Python's sqlite3 opens implicit transactions of its own
    and ``executescript`` (migration 1 uses it) COMMITs whatever is pending,
    which silently ends a savepoint and makes ``ROLLBACK TO`` fail with "no such
    savepoint". So ``isolation_level`` is switched off for the duration and the
    transaction is explicit. A migration that itself uses ``executescript``
    commits early and is not atomic — which is safe only because it is
    idempotent, as migration 1 is; a later migration should avoid it.

    ``PRAGMA user_version`` cannot be parameterised, which is why it is
    formatted; the value is an int from this module's own table.
    """
    at = current_version(conn)
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
                if conn.in_transaction:      # an executescript may have committed
                    conn.execute("COMMIT")
            except BaseException as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise RuntimeError(
                    f"schema migration {migration.version} ({migration.name}) "
                    f"failed: {exc}") from exc
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
