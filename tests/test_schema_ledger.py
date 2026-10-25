"""database/schema.py — the migration ledger."""
from __future__ import annotations

import sqlite3

import pytest


def test_migrations_are_numbered_from_one_without_gaps():
    from database import schema

    versions = [m.version for m in schema.MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1))
    assert schema.SCHEMA_VERSION == versions[-1]
    assert len({m.name for m in schema.MIGRATIONS}) == len(versions)


def test_apply_brings_a_blank_file_to_the_current_version(tmp_path):
    from database import schema

    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        assert schema.current_version(conn) == 0
        applied = schema.apply(conn)
        assert applied == [m.version for m in schema.MIGRATIONS]
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
    finally:
        conn.close()


def test_apply_is_idempotent(tmp_path):
    from database import schema

    conn = sqlite3.connect(tmp_path / "twice.db")
    try:
        schema.apply(conn)
        before = schema.shape(conn)
        assert schema.apply(conn) == []          # nothing left to do
        assert schema.shape(conn) == before
    finally:
        conn.close()


def test_a_fresh_database_matches_a_migrated_one(tmp_path):
    """The whole point. A library built from scratch and one upgraded from the
    pre-ledger code must be the same database, column for column."""
    from database import models, schema

    # Upgraded: the pre-ledger path, exactly as it ran before this task.
    legacy = sqlite3.connect(tmp_path / "legacy.db")
    try:
        legacy.executescript(models.SCHEMA)
        models._migrate_songs_columns(legacy)
        models._migrate_features_columns(legacy)
        models._migrate_candidates_columns(legacy)
        models._migrate_sections_columns(legacy)
        models._migrate_candidates_unique_key(legacy)
        models._migrate_stems_columns(legacy)
        models._migrate_mixtracks_columns(legacy)
        models._migrate_mashuppairs_columns(legacy)
        models._migrate_crates_columns(legacy)
        models._migrate_sets_columns(legacy)
        models._migrate_set_items_kinds(legacy)
        models._migrate_pair_feedback_key(legacy)
        from database import alignment
        alignment.migrate(legacy)
        legacy.commit()
        schema.apply(legacy)                     # the ledger finishes the job
        upgraded = schema.shape(legacy)
    finally:
        legacy.close()

    fresh = sqlite3.connect(tmp_path / "fresh.db")
    try:
        schema.apply(fresh)
        built = schema.shape(fresh)
    finally:
        fresh.close()

    assert set(built["tables"]) == set(upgraded["tables"])
    for table in sorted(built["tables"]):
        assert built["tables"][table] == upgraded["tables"][table], table
    assert built["indexes"] == upgraded["indexes"]


def test_get_conn_records_the_version(tmp_path, monkeypatch):
    from database import models, schema

    db = tmp_path / "viaconn.db"
    conn = models.get_conn(db)
    try:
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
    finally:
        conn.close()


def test_a_failing_migration_names_its_version_and_is_not_recorded(tmp_path, monkeypatch):
    """The stored version must stay on the last migration that succeeded, the
    failed one's partial work must be undone, and the error must say which one
    broke — so the next start retries exactly that one."""
    from database import schema

    def good(conn):
        conn.execute("CREATE TABLE ledger_probe (id INTEGER PRIMARY KEY)")

    def bad(conn):
        conn.execute("CREATE TABLE half_done (id INTEGER PRIMARY KEY)")
        raise ValueError("boom")

    monkeypatch.setattr(schema, "MIGRATIONS", (
        schema.Migration(1, "good", good),
        schema.Migration(2, "bad", bad),
    ))
    conn = sqlite3.connect(tmp_path / "broken.db")
    try:
        with pytest.raises(RuntimeError, match=r"migration 2 \(bad\)"):
            schema.apply(conn)
        assert schema.current_version(conn) == 1
        tables = set(schema.shape(conn)["tables"])
        assert "ledger_probe" in tables
        assert "half_done" not in tables
    finally:
        conn.close()
