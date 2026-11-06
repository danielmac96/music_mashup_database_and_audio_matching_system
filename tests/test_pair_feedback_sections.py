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


