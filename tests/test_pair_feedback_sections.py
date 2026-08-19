"""pair_feedback must be keyed on the section pair, not just the song pair.

The old UNIQUE(vocal_song_id, inst_song_id) sat directly above nullable
vocal_section/inst_section columns. Since E.3 made the section pair the
candidate row, judging "chorus over drop" and then "verse over breakdown" on the
same two records DESTROYED the first verdict — silently, and in the one table
holding data that cannot be recomputed.

The migration therefore gets more scrutiny than the mashup_candidates one it is
modelled on: that table is truncated on every re-score, this one is the only
training signal the learned scorer has.
"""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    return models, tmp_path / "test.db"


LEGACY_DDL = """
CREATE TABLE pair_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vocal_song_id   INTEGER NOT NULL,
    inst_song_id    INTEGER NOT NULL,
    vocal_section   INTEGER,
    inst_section    INTEGER,
    verdict         TEXT NOT NULL CHECK(verdict IN ('love','ok','no')),
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(vocal_song_id, inst_song_id)
);
"""


def _legacy_db(db_path, rows=()):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(LEGACY_DDL)
    conn.executemany(
        "INSERT INTO pair_feedback (vocal_song_id, inst_song_id, vocal_section, "
        "inst_section, verdict) VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


# ── the bug itself ───────────────────────────────────────────────────────────

def test_two_section_pairs_of_the_same_songs_are_two_verdicts(env):
    """The regression this whole file exists for."""
    models, _ = env
    models.init_db()

    models.upsert_pair_feedback(1, 2, "love", vocal_section=0, inst_section=3)
    models.upsert_pair_feedback(1, 2, "no", vocal_section=1, inst_section=4)

    rows = models.get_pair_feedback()
    assert len(rows) == 2
    by_section = {(r["vocal_section"], r["inst_section"]): r["verdict"] for r in rows}
    assert by_section[(0, 3)] == "love"
    assert by_section[(1, 4)] == "no"


def test_rejudging_the_same_section_pair_corrects_it(env):
    """Still an upsert — one verdict per section pair, not an append-only log."""
    models, _ = env
    models.init_db()

    models.upsert_pair_feedback(1, 2, "love", vocal_section=0, inst_section=3)
    models.upsert_pair_feedback(1, 2, "no", vocal_section=0, inst_section=3)

    rows = models.get_pair_feedback()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "no"


def test_null_sections_are_one_key_not_many(env):
    """A pair judged with no section context collapses to a single row, which is
    what COALESCE(-1) in the index buys — SQLite treats NULLs as distinct."""
    models, _ = env
    models.init_db()

    models.upsert_pair_feedback(1, 2, "love")
    models.upsert_pair_feedback(1, 2, "ok")

    rows = models.get_pair_feedback()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "ok"


def test_song_level_and_section_level_verdicts_coexist(env):
    models, _ = env
    models.init_db()
    models.upsert_pair_feedback(1, 2, "ok")
    models.upsert_pair_feedback(1, 2, "love", vocal_section=0, inst_section=1)
    assert len(models.get_pair_feedback()) == 2


# ── the feature snapshot ─────────────────────────────────────────────────────

def test_features_are_stored_with_the_verdict(env):
    """score_all_pairs truncates mashup_candidates on every run, so without this
    a saved verdict is only as reproducible as the last re-score."""
    models, _ = env
    models.init_db()
    feats = {"bpm_score": 0.95, "key_score": 1.0, "stretch": 1.02}
    models.upsert_pair_feedback(1, 2, "love", vocal_section=0, inst_section=1,
                                features=feats)

    row = models.get_pair_feedback()[0]
    assert json.loads(row["features_json"]) == feats


def test_correcting_a_verdict_keeps_the_original_snapshot(env):
    """A correction from the UI carries no features; blanking what the first
    judgement recorded would lose the only copy."""
    models, _ = env
    models.init_db()
    models.upsert_pair_feedback(1, 2, "love", vocal_section=0, inst_section=1,
                                features={"bpm_score": 0.9})
    models.upsert_pair_feedback(1, 2, "no", vocal_section=0, inst_section=1)

    row = models.get_pair_feedback()[0]
    assert row["verdict"] == "no"
    assert json.loads(row["features_json"]) == {"bpm_score": 0.9}


# ── the migration ────────────────────────────────────────────────────────────

def test_legacy_verdicts_survive_the_migration(env):
    models, db_path = env
    _legacy_db(db_path, [(1, 2, 0, 3, "love"), (3, 4, None, None, "no"),
                         (5, 6, 1, 1, "ok")])

    models.init_db()

    rows = models.get_pair_feedback()
    assert len(rows) == 3
    assert {(r["vocal_song_id"], r["inst_song_id"], r["verdict"]) for r in rows} == {
        (1, 2, "love"), (3, 4, "no"), (5, 6, "ok")}


def test_migration_installs_the_section_key(env):
    models, db_path = env
    _legacy_db(db_path, [(1, 2, 0, 3, "love")])
    models.init_db()

    conn = sqlite3.connect(db_path)
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='pair_feedback'").fetchone()[0]
    indexes = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='pair_feedback'").fetchall()}
    conn.close()

    assert "UNIQUE(vocal_song_id, inst_song_id)" not in ddl
    assert "ux_pair_feedback_section" in indexes

    # And the new key actually works on the migrated table.
    models.upsert_pair_feedback(1, 2, "no", vocal_section=9, inst_section=9)
    assert len(models.get_pair_feedback()) == 2


def test_migration_is_idempotent(env):
    """get_conn runs migrations per process per path; a second pass must not
    duplicate or drop anything."""
    models, db_path = env
    _legacy_db(db_path, [(1, 2, 0, 3, "love"), (3, 4, 1, 1, "ok")])
    models.init_db()

    import database.models as m
    m._INITIALIZED_PATHS.discard(str(db_path))   # force the migrations to re-run
    models.init_db()

    assert len(models.get_pair_feedback()) == 2


def test_migration_adds_features_json_to_a_legacy_table(env):
    models, db_path = env
    _legacy_db(db_path, [(1, 2, 0, 3, "love")])
    models.init_db()

    cols = {r["name"] if isinstance(r, dict) else r[1] for r in
            models.get_conn().execute("PRAGMA table_info(pair_feedback)").fetchall()}
    assert "features_json" in cols


def test_a_short_copy_leaves_the_original_intact(env, monkeypatch):
    """The safety net. If the copy loses rows for any reason, keep the old table
    rather than dropping it — two tables is recoverable, no verdicts is not."""
    models, db_path = env
    _legacy_db(db_path, [(1, 2, 0, 3, "love"), (3, 4, 1, 1, "ok")])

    import database.models as m

    def short_copy(conn, col_list):
        # Simulate a partial transfer: one of the two rows makes it across.
        conn.execute(f"INSERT INTO pair_feedback_new ({col_list}) "
                     f"SELECT {col_list} FROM pair_feedback LIMIT 1")

    monkeypatch.setattr(m, "_copy_pair_feedback", short_copy)
    with pytest.raises(RuntimeError, match="left untouched"):
        models.init_db()
    monkeypatch.undo()

    # Both verdicts still there, under the original table, ready to retry.
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM pair_feedback").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE "
                        "name='pair_feedback_new'").fetchone()[0] == 0
    conn.close()

    # And a clean re-run afterwards migrates it properly.
    m._INITIALIZED_PATHS.discard(str(db_path))
    models.init_db()
    assert len(models.get_pair_feedback()) == 2


# ── the API keeps its side of the bargain ────────────────────────────────────

def test_endpoint_records_sections(env, monkeypatch):
    from fastapi.testclient import TestClient
    models, _ = env
    models.init_db()

    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.server as server
    importlib.reload(server)
    client = TestClient(server.app)

    for payload in ({"vocal_song_id": 1, "inst_song_id": 2, "verdict": "love",
                     "vocal_section": 0, "inst_section": 3},
                    {"vocal_song_id": 1, "inst_song_id": 2, "verdict": "no",
                     "vocal_section": 1, "inst_section": 4}):
        assert client.post("/api/mashups/feedback", json=payload).status_code == 200

    body = client.get("/api/mashups/feedback").json()
    assert body["count"] == 2
    assert {(f["vocal_section"], f["inst_section"], f["verdict"])
            for f in body["feedback"]} == {(0, 3, "love"), (1, 4, "no")}
