"""database/schema.py — the migration ledger."""
from __future__ import annotations

import sqlite3

import pytest


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
        schema.Migration(7, "good", good),
        schema.Migration(8, "bad", bad),
    ))
    conn = sqlite3.connect(tmp_path / "broken.db")
    try:
        with pytest.raises(RuntimeError, match=r"migration 8 \(bad\)"):
            schema.apply(conn)
        assert schema.current_version(conn) == 7
        tables = set(schema.shape(conn)["tables"])
        assert "ledger_probe" in tables
        assert "half_done" not in tables
    finally:
        conn.close()




def test_migrations_are_numbered_without_gaps_from_the_baseline():
    from database import schema

    versions = [m.version for m in schema.MIGRATIONS]
    assert versions[0] == 7 and schema.MIGRATIONS[0].name == "baseline"
    assert versions == list(range(7, 7 + len(versions)))
    assert schema.SCHEMA_VERSION == versions[-1]
    assert len({m.name for m in schema.MIGRATIONS}) == len(versions)


def test_no_migration_uses_executescript():
    """executescript COMMITs inside apply's explicit BEGIN and leaves
    user_version unset for that migration."""
    import inspect

    from database import schema

    for migration in schema.MIGRATIONS:
        src = inspect.getsource(migration.up)
        assert ".executescript(" not in src, migration.name


@pytest.mark.parametrize("old", [1, 4, 6])
def test_a_database_from_an_older_build_is_refused(tmp_path, old):
    """Versions 1-6 predate the one-pass pipeline. They are refused with a
    'start a fresh library' message, never half-migrated."""
    from database import schema

    conn = sqlite3.connect(tmp_path / "old.db")
    try:
        conn.execute(f"PRAGMA user_version = {old}")
        with pytest.raises(schema.RetiredSchema, match="fresh library"):
            schema.apply(conn)
        assert schema.current_version(conn) == old
    finally:
        conn.close()


def test_the_baseline_has_the_tables_the_pipeline_writes(tmp_path):
    from database import schema

    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        schema.apply(conn)
        tables = set(schema.shape(conn)["tables"])
        for t in ("songs", "stems", "features", "feature_arrays", "sections",
                  "sections_allin1", "allin1_analysis", "allin1_embeddings",
                  "vocal_phrases", "analysis_frames", "feature_cache",
                  "mashup_candidates", "pair_feedback", "analysis_runs"):
            assert t in tables, t
        assert "pipeline_parked" not in tables
        cols = {r[1] for r in conn.execute("PRAGMA table_info(allin1_analysis)")}
        assert {"beat_conf", "downbeat_conf_beats", "segment_label_probs",
                "segment_labels_order"} <= cols
        assert "scored_at" in {r[1] for r in conn.execute("PRAGMA table_info(songs)")}
    finally:
        conn.close()
