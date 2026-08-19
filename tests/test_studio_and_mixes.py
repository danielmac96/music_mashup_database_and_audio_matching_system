"""Tests for the Studio mixdown endpoint, URL classification, and the pasted
tracklist parser. Pure-Python paths only — the DSP render itself needs
librosa and is exercised by hand, but request validation, parsing, and DB
wiring must hold without the audio stack.
"""
from pathlib import Path
import importlib
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingest.sources import classify_url  # noqa: E402
from api.routes.mixes import _parse_tracklist  # noqa: E402


# ── classify_url ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://soundcloud.com/artist/track", ("soundcloud", "track")),
    ("soundcloud.com/artist/sets/some-playlist", ("soundcloud", "playlist")),
    ("https://www.youtube.com/watch?v=abc", ("youtube", "track")),
    ("https://youtube.com/watch?v=abc&list=PL123", ("youtube", "playlist")),
    ("https://music.youtube.com/playlist?list=PL9", ("youtube", "playlist")),
    ("https://youtu.be/abc123", ("youtube", "track")),
    ("https://example.com/whatever", ("unknown", "track")),
    ("", ("unknown", "track")),
])
def test_classify_url(url, expected):
    assert classify_url(url) == expected


# ── tracklist paste parser ────────────────────────────────────────────────────

def test_parse_tracklist_variants():
    text = (
        "1. [0:00] Kanye West - Stronger\n"
        "w/ [0:45] Whitney Houston - I Wanna Dance With Somebody\n"
        "2. Avicii - Levels\n"
        "follow us on twitter\n"
        "3. 12:30 Zedd & Grey - The Middle\n"
    )
    rows = _parse_tracklist(text)
    assert [r["title"] for r in rows] == [
        "Stronger", "I Wanna Dance With Somebody", "Levels", "The Middle"]
    expected = {"entry_index": 1, "cue_secs": 0.0, "is_overlay": False,
                "artist": "Kanye West", "title": "Stronger"}
    # Parser output grew extra fields (raw_label, is_id, …) — the original
    # contract keys must still hold exactly.
    assert {k: rows[0][k] for k in expected} == expected
    assert rows[1]["is_overlay"] is True and rows[1]["cue_secs"] == 45
    assert rows[3]["cue_secs"] == 12 * 60 + 30


def test_parse_tracklist_html_and_dedupe():
    html = "<li>Artist - Song</li><li>Artist - Song</li><li>Other - Tune</li>"
    rows = _parse_tracklist(html)
    assert [(r["artist"], r["title"]) for r in rows] == [
        ("Artist", "Song"), ("Other", "Tune")]


# ── mixdown endpoint validation ───────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))

    import config
    import database.models as models
    for mod in (config, models):
        importlib.reload(mod)
    models.init_db()
    models.upsert_song(title="Known", source_url="u://one")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import studio

    # studio.py does `from database.models import get_conn` at import time, so it
    # holds a function whose default db_path was frozen to whatever DB_PATH was
    # when it first loaded. Without this reload the route reads a previous test's
    # database and the fixture's song is invisible — an ordering bug that only
    # shows up once some earlier test file reloads database.models.
    importlib.reload(studio)

    app = FastAPI()
    app.include_router(studio.router, prefix="/api/studio")
    return TestClient(app)


def test_mixdown_rejects_empty_and_unknown(client):
    assert client.post("/api/studio/mixdown", json={"clips": []}).status_code == 400
    res = client.post("/api/studio/mixdown",
                      json={"clips": [{"song_id": 999, "stem": "full"}]})
    assert res.status_code == 404
    assert "999" in res.json()["detail"]


def test_mixdown_queues_job_for_known_song(client):
    res = client.post("/api/studio/mixdown",
                      json={"clips": [{"song_id": 1, "stem": "full",
                                       "offset_sec": 0.5, "rate": 1.1,
                                       "semitones": -2, "gain": 0.8}]})
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"]
    assert body["audio_url"].endswith(f"/api/studio/mixdown/{body['job_id']}/audio")


def test_mixdown_token_is_sanitised(client):
    # A malformed token must 400 (it lands in a filename) — traversal-ish
    # tokens must never reach the filesystem.
    res = client.get("/api/studio/mixdown/not-a-hex-token!/audio")
    assert res.status_code == 400
    res = client.get("/api/studio/mixdown/deadbeefdeadbeef/audio")
    assert res.status_code == 404  # valid shape, nothing rendered
