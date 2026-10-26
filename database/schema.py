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


def _feature_arrays(conn: sqlite3.Connection) -> None:
    """Move `features`' two big JSON arrays into `feature_arrays`.

    beat_times_json and waveform_rms_json were 4.5 MB of the table's 5.6 MB and
    are read by nothing on the scoring or listing path, while `SELECT f.*`
    carried them into every scan. Copied value for value as float32, then the
    columns are dropped — neither appears in an index or a view, so SQLite can
    drop them in place.

    Idempotent, because a rollback (database/downgrade.py) resets
    ``user_version`` and replays every migration: migration 1 puts the two
    (empty) columns back, this one finds nothing to move and drops them again.
    Only statements, never ``executescript`` — see `apply`.
    """
    import json

    from database import arrays

    # The caller's factory is whatever it set: `get_conn` uses sqlite3.Row, a
    # bare `sqlite3.connect` does not. This reads columns by name.
    previous = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        arrays.ensure_schema(conn)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(features)")}
        moves = [("beat_times_json", "beat_times"),
                 ("waveform_rms_json", "waveform_rms")]
        present = [(col, name) for col, name in moves if col in cols]
        if present:
            select = ", ".join(col for col, _ in present)
            for row in conn.execute(
                    f"SELECT song_id, stem_type, {select} FROM features").fetchall():
                for col, name in present:
                    raw = row[col]
                    if not raw:
                        continue
                    try:
                        values = json.loads(raw)
                    except (TypeError, ValueError):
                        log.warning("features.%s for song %s/%s is not JSON; dropped",
                                    col, row["song_id"], row["stem_type"])
                        continue
                    arrays.put(conn, int(row["song_id"]), row["stem_type"], name,
                               values)
        for col, _name in moves:
            if col in cols:
                conn.execute(f"ALTER TABLE features DROP COLUMN {col}")
        # The scorer and dedup read every full-mix row; give that the index it
        # never had (`features` was only indexed on bpm and (key, mode)).
        conn.execute("CREATE INDEX IF NOT EXISTS idx_features_stem "
                     "ON features(stem_type, song_id)")
    finally:
        conn.row_factory = previous


def _slim_candidates(conn: sqlite3.Connection) -> None:
    """Drop what nothing reads from `mashup_candidates`, and index what does.

    Measured on a 206-track library: the table was 31.6 MB over 63,918 rows.
    `reason` was 4.42 MB of a sentence rebuilt from the row's own numbers
    (matcher.recipe.alignment_reason), and `scored_at` 1.21 MB of one timestamp
    repeated — the run's, which `app_prefs.last_score` already holds. The
    other five (`models.CANDIDATE_DROPPED_COLUMNS`) have no reader outside the
    table's own writer. None of the seven is in an index, view or trigger, so
    SQLite drops them in place.

    Then the indexes. `idx_candidates_type` is on a column with two distinct
    values, so the dock's query scanned half the table and sorted it in a temp
    B-tree: 119 ms scoped to one track, at 63,918 rows, against a cap of
    200,000. Each index below serves one access path the app actually takes.

    Also drops `idx_feature_arrays_song` (migration 2): it is a strict prefix
    of that table's PRIMARY KEY autoindex (song_id, stem_type, name), so SQLite
    never chooses it and it costs a b-tree write on every `arrays.put`. Dropped
    here rather than edited out of migration 2, which has shipped.

    Idempotent: a rollback resets ``user_version`` and replays every migration,
    and migration 1 puts the columns and the old indexes back. Statements only,
    never ``executescript`` — see `apply`.
    """
    from database.models import CANDIDATE_DROPPED_COLUMNS

    cols = {r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    for col in CANDIDATE_DROPPED_COLUMNS:
        if col in cols:
            conn.execute(f"ALTER TABLE mashup_candidates DROP COLUMN {col}")
    for statement in (
            # the dock scoped to a track, each side
            "CREATE INDEX IF NOT EXISTS idx_candidates_vocal_score "
            "ON mashup_candidates(combo_type, vocal_song_id, score_total DESC)",
            "CREATE INDEX IF NOT EXISTS idx_candidates_inst_score "
            "ON mashup_candidates(combo_type, inst_song_id, score_total DESC)",
            # the dock library-wide, and every "best in library" order
            "CREATE INDEX IF NOT EXISTS idx_candidates_combo_score "
            "ON mashup_candidates(combo_type, score_total DESC)",
            # the stepper's pairings of two songs, and the per-row
            # pairing_count / pairing_rank subqueries the listing runs per row
            "CREATE INDEX IF NOT EXISTS idx_candidates_pair "
            "ON mashup_candidates(combo_type, vocal_song_id, inst_song_id, "
            "score_total DESC)",
            # superseded: combo_type has two distinct values, and score_total
            # alone cannot serve a filtered order
            "DROP INDEX IF EXISTS idx_candidates_type",
            "DROP INDEX IF EXISTS idx_candidates_score",
            # a strict prefix of feature_arrays' primary key (migration 2)
            "DROP INDEX IF EXISTS idx_feature_arrays_song",
    ):
        conn.execute(statement)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline", _baseline),
    Migration(2, "feature_arrays", _feature_arrays),
    Migration(3, "slim_candidates", _slim_candidates),
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
