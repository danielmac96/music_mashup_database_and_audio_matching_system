"""The songs.track_id index and identity lookups by track id."""
import importlib
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


def _indexes(db_path):
    conn = sqlite3.connect(db_path)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    conn.close()
    return names


def test_index_exists_after_init(env):
    models, db_path = env
    models.init_db()
    assert "idx_songs_track_id" in _indexes(db_path)


def test_index_is_not_unique(env):
    """track_id is '' for mixes-ingested rows and anything predating the column,
    so a UNIQUE index would collide the moment there were two of them."""
    models, db_path = env
    models.init_db()
    a = models.upsert_song(title="A", artist="X", source_url="https://soundcloud.com/a/1")
    b = models.upsert_song(title="B", artist="Y", source_url="https://soundcloud.com/b/2")
    assert a != b
    rows = models.get_all_songs()
    assert {r["track_id"] for r in rows} <= {"", None}


def test_get_song_by_track_id(env):
    models, _ = env
    models.init_db()
    sid = models.upsert_song(title="A", artist="X",
                             source_url="https://soundcloud.com/a/1", track_id="98765")
    assert models.get_song_by_track_id("98765")["id"] == sid
    assert models.get_song_by_track_id("00000") is None
    # An empty id must never match the rows whose id is unknown.
    assert models.get_song_by_track_id("") is None


def test_songs_by_identity_matches_both_keys(env):
    models, _ = env
    models.init_db()
    by_url = models.upsert_song(title="ByUrl", artist="X",
                                source_url="https://soundcloud.com/a/url-only")
    by_tid = models.upsert_song(title="ByTid", artist="Y",
                                source_url="https://soundcloud.com/a/renamed-since",
                                track_id="4242")

    found = models.songs_by_identity(
        source_urls=["https://soundcloud.com/a/url-only", "https://soundcloud.com/a/absent"],
        track_ids=["4242", "9999"])

    assert found["by_url"]["https://soundcloud.com/a/url-only"]["id"] == by_url
    assert found["by_track_id"]["4242"]["id"] == by_tid
    assert "https://soundcloud.com/a/absent" not in found["by_url"]
    assert "9999" not in found["by_track_id"]


def test_songs_by_identity_ignores_empty_track_ids(env):
    """'' is the default for rows that never learned their id — matching on it
    would claim every such row is the track being looked at."""
    models, _ = env
    models.init_db()
    models.upsert_song(title="NoTid", artist="X", source_url="https://soundcloud.com/a/1")
    found = models.songs_by_identity(source_urls=[], track_ids=["", "  "])
    assert found["by_url"] == {} and found["by_track_id"] == {}


def test_songs_by_identity_empty_input_is_cheap(env):
    models, _ = env
    models.init_db()
    assert models.songs_by_identity() == {"by_url": {}, "by_track_id": {}}
