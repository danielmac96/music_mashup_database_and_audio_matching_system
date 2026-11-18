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


def _pair_events(conn: sqlite3.Connection) -> None:
    """Implicit signals about a pair (readme §5.13): one row per thing done
    with it — shown, looped, skipped, expanded, opened in Studio, rendered,
    exported, hidden, compared, rated. Keyed by the pair's four ids like every
    other per-pair table (never candidate.id, readme §7).

    `pair_event_summary` is what raw rows are folded into once they are old
    (models.compact_pair_events): a count per pair, event and context, so the
    raw table stays bounded and nothing learned from is lost.
    """
    conn.execute("""CREATE TABLE IF NOT EXISTS pair_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        vocal_song_id   INTEGER NOT NULL,
        inst_song_id    INTEGER NOT NULL,
        vocal_section   INTEGER,
        inst_section    INTEGER,
        event           TEXT NOT NULL,
        value           REAL,
        context         TEXT,
        scorer_version  TEXT,
        rank_shown      INTEGER,
        created_at      TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pair_events_created "
                 "ON pair_events(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pair_events_pair "
                 "ON pair_events(vocal_song_id, inst_song_id)")
    conn.execute("""CREATE TABLE IF NOT EXISTS pair_event_summary (
        vocal_song_id   INTEGER NOT NULL,
        inst_song_id    INTEGER NOT NULL,
        vocal_section   INTEGER,
        inst_section    INTEGER,
        event           TEXT NOT NULL,
        context         TEXT,
        n               INTEGER NOT NULL DEFAULT 0,
        n_top5          INTEGER NOT NULL DEFAULT 0,
        value_sum       REAL NOT NULL DEFAULT 0,
        first_at        TEXT,
        last_at         TEXT
    )""")
    conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_pair_event_summary
        ON pair_event_summary(vocal_song_id, inst_song_id,
                              COALESCE(vocal_section, -1), COALESCE(inst_section, -1),
                              event, COALESCE(context, ''))""")


_LEARN_CANDIDATE_COLUMNS = (("score_rule", "REAL"), ("model_prob", "REAL"))
_LEARN_MODEL_COLUMNS = (
    ("kind", "TEXT"), ("alpha", "REAL"), ("n_strong", "INTEGER"),
    ("n_weak", "INTEGER"), ("auc_strong", "REAL"), ("rule_auc", "REAL"),
    ("decision", "TEXT"), ("activated_at", "TEXT"),
)


def _learning_loop(conn: sqlite3.Connection) -> None:
    """The learning loop (readme §5.8). On mashup_candidates: `score_rule`,
    the rule's own total, kept beside `score_total` (which carries the
    shrinkage blend once a model is active, and equals score_rule until then),
    and `model_prob`, the active model's calibrated probability for the row.
    On models: what each trained model was trained on and how the shadow
    evaluation judged it, and when it was last activated (rollback reads it).

    Nullable, added only; pair_feedback is not touched. Existing rows get
    score_rule = score_total, except rows the retired "model replaces the
    total" path wrote (scorer='model'), whose total is not a rule score —
    they stay NULL until the next Score library rewrites them."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    for name, kind in _LEARN_CANDIDATE_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE mashup_candidates ADD COLUMN {name} {kind}")
    conn.execute("UPDATE mashup_candidates SET score_rule = score_total "
                 "WHERE score_rule IS NULL AND COALESCE(scorer, '') <> 'model'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_rule "
                 "ON mashup_candidates(combo_type, score_rule)")
    have = {r[1] for r in conn.execute("PRAGMA table_info(models)")}
    for name, kind in _LEARN_MODEL_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE models ADD COLUMN {name} {kind}")


_SLOT_COLUMNS = (
    ("score_slot", "REAL"), ("slot_occupancy", "REAL"),
    ("slot_phrase_match", "REAL"), ("slot_register", "REAL"),
    ("slot_phrase_err_ms", "REAL"),
    ("window_bars", "INTEGER"), ("vocal_window_bar", "INTEGER"),
    ("bed_window_bar", "INTEGER"), ("score_window", "REAL"),
    ("dropin_mask", "INTEGER"), ("camelot_shift", "INTEGER"),
)


def _slot_and_dropin(conn: sqlite3.Connection) -> None:
    """The vacated-slot term, the best phrase window and what the Drop-in rule
    reads (readme §5.7), on mashup_candidates. All nullable: a row scored
    before this is unmeasured, never zero, until the next Score library.
    Only that table is touched — it is rewritten by every scoring run and
    holds no user input."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    for name, kind in _SLOT_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE mashup_candidates ADD COLUMN {name} {kind}")


SETS_BACKUP_PREF = "sets_backup"


def _db_file(conn: sqlite3.Connection) -> Path | None:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main" and row[2]:
            return Path(row[2])
    return None


def sets_export_dir(conn: sqlite3.Connection) -> Path:
    """`<data_dir>/exports` for the database this connection is on: the
    configured data folder when it is the configured database, else the
    folder holding the file (a test's or a benchmark's library)."""
    db = _db_file(conn)
    try:
        import config
        if db is not None and Path(config.DB_PATH).resolve() == db.resolve():
            return Path(config.DATA_DIR) / "exports"
    except Exception:  # noqa: BLE001 — config is optional here
        pass
    if db is None:
        raise RuntimeError("the database has no file, so there is nowhere to "
                           "write the sets backup")
    return db.parent / "exports"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def _rows(conn: sqlite3.Connection, sql: str) -> List[dict]:
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def write_sets_backup(conn: sqlite3.Connection, out_dir: Path) -> dict:
    """Every `sets` and `set_items` row, each item's frozen pair row parsed,
    written to `out_dir/sets_backup_<UTC timestamp>.json`, then read back
    and counted against the tables. Raises on any failure (the caller must
    then drop nothing). Returns {path, sets, items}."""
    import json
    import os
    from datetime import datetime, timezone
    sets = _rows(conn, "SELECT * FROM sets ORDER BY id")
    items = _rows(conn, "SELECT * FROM set_items ORDER BY set_id, position, id")
    for it in items:
        raw = it.get("payload_json")
        try:
            it["payload"] = json.loads(raw) if raw else None
        except ValueError:
            it["payload"] = None          # the raw text is kept beside it
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sets_backup_{stamp}.json"
    n = 2
    while path.exists():
        path = out_dir / f"sets_backup_{stamp}_{n}.json"
        n += 1
    doc = {"kind": "mashup-engine sets backup", "written_at": stamp,
           "counts": {"sets": len(sets), "set_items": len(items)},
           "sets": [{**s, "items": [i for i in items if i["set_id"] == s["id"]]}
                    for s in sets],
           # Items whose set row is missing would vanish from the nesting above.
           "orphan_items": [i for i in items
                            if i["set_id"] not in {s["id"] for s in sets}]}
    tmp = path.with_suffix(".json.part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    back = json.loads(path.read_text(encoding="utf-8"))
    got_sets = len(back["sets"])
    got_items = sum(len(s["items"]) for s in back["sets"]) + len(back["orphan_items"])
    want_sets = conn.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
    want_items = conn.execute("SELECT COUNT(*) FROM set_items").fetchone()[0]
    if (got_sets, got_items) != (want_sets, want_items):
        raise RuntimeError(
            f"sets backup {path} holds {got_sets} sets / {got_items} items, the "
            f"tables {want_sets} / {want_items}; nothing was dropped")
    return {"path": str(path), "sets": got_sets, "items": got_items}


def _remove_sets(conn: sqlite3.Connection) -> None:
    """Sets are removed (owner-approved, Phase 3). Their rows are first
    written to `<data_dir>/exports/sets_backup_<timestamp>.json` and read back
    and counted; any failure raises, so the migration rolls back with both
    tables in place and the next start tries again. Only then are `sets` and
    `set_items` dropped. A library with no sets writes no file. The backup's
    path is kept in app_prefs (`sets_backup`) so the UI can say it once.
    Idempotent: with the tables gone there is nothing to do."""
    have_sets, have_items = _table_exists(conn, "sets"), _table_exists(conn, "set_items")
    if not (have_sets or have_items):
        return
    n_sets = conn.execute("SELECT COUNT(*) FROM sets").fetchone()[0] if have_sets else 0
    n_items = conn.execute("SELECT COUNT(*) FROM set_items").fetchone()[0] if have_items else 0
    if n_sets or n_items:
        if not (have_sets and have_items):
            raise RuntimeError("only one of sets / set_items exists; refusing to "
                               "guess, nothing was dropped")
        info = write_sets_backup(conn, sets_export_dir(conn))
        log.warning("schema: sets removed — %d sets and %d items backed up to %s",
                    info["sets"], info["items"], info["path"])
        import json
        conn.execute(
            "INSERT INTO app_prefs (key, value, updated_at) VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (SETS_BACKUP_PREF, json.dumps({**info, "seen": False})))
    conn.execute("DROP INDEX IF EXISTS idx_set_items_set")
    conn.execute("DROP INDEX IF EXISTS ux_set_items_pair")
    conn.execute("DROP TABLE IF EXISTS set_items")
    conn.execute("DROP TABLE IF EXISTS sets")


MIGRATIONS: tuple[Migration, ...] = (
    Migration(7, "baseline", _baseline),
    Migration(8, "pair_events", _pair_events),
    Migration(9, "slot_and_dropin", _slot_and_dropin),
    Migration(10, "learning_loop", _learning_loop),
    Migration(11, "remove_sets", _remove_sets),
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
