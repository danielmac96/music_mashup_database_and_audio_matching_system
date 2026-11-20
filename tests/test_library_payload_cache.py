"""GET /api/tracks keeps its encoded payload while the database is unchanged
(master plan 3.2) — and never serves a stale one: any commit, from any
connection, including writers that stamp no timestamp, rebuilds it."""
import gzip
import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.routes.tracks as tracks
    importlib.reload(tracks)
    import api.server as server
    importlib.reload(server)
    return TestClient(server.app), models, tracks


def test_unchanged_database_serves_the_kept_payload(app, monkeypatch):
    client, models, tracks = app
    models.upsert_song("One", "A", "https://sc/1", 200, "House", status="analysed")
    first = client.get("/api/tracks").json()
    calls = []
    real = tracks.build_tracks_payload
    monkeypatch.setattr(tracks, "build_tracks_payload", lambda: calls.append(1) or real())
    assert client.get("/api/tracks").json() == first
    assert calls == [], "nothing changed: no rebuild"


def test_any_write_rebuilds_even_without_a_timestamp(app):
    client, models, tracks = app
    sid = models.upsert_song("One", "A", "https://sc/1", 200, "House", status="analysed")
    assert client.get("/api/tracks").json()["tracks"][0]["audio_provenance"] is None
    # set_audio_provenance stamps no updated_at — a timestamp fingerprint misses it.
    models.set_audio_provenance(sid, {"via": "youtube", "url": "https://yt/x"})
    got = client.get("/api/tracks").json()["tracks"][0]
    assert got["audio_provenance"]["via"] == "youtube"
    models.upsert_song("Two", "B", "https://sc/2", 200, "House", status="queued")
    assert client.get("/api/tracks").json()["count"] == 2


def test_gzip_when_asked(app):
    client, models, tracks = app
    models.upsert_song("One", "A", "https://sc/1", 200, "House", status="analysed")
    r = client.get("/api/tracks", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"
    assert r.json()["count"] == 1               # the client decodes it
    body, gz = tracks.library_payload()
    assert json.loads(gzip.decompress(gz)) == json.loads(body)
