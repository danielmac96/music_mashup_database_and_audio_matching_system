"""Phase 2: library management — URL dedup, remove song, edit URL.

Covers ingest.sources.normalize_url, the database.models helpers
(get_song_by_url / delete_song / update_song_url), and the tracks route
handlers (DELETE /{id}, PATCH /{id}/url) including on-disk file cleanup.
"""
import importlib

import pytest
from fastapi import HTTPException


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    return models


# ── normalize_url ────────────────────────────────────────────────────────────

def test_normalize_url_strips_tracking_and_canonicalizes():
    from ingest.sources import normalize_url
    a = normalize_url("https://www.youtube.com/watch?v=abc&si=xyz&t=30")
    b = normalize_url("http://youtube.com/watch?v=abc")
    assert a == b == "https://youtube.com/watch?v=abc"


def test_normalize_url_soundcloud_drops_query_keeps_secret():
    from ingest.sources import normalize_url
    assert normalize_url("https://m.soundcloud.com/a/track?in=x/sets/y/") \
        == "https://soundcloud.com/a/track"
    assert normalize_url("https://soundcloud.com/a/track?secret_token=s-9") \
        == "https://soundcloud.com/a/track?secret_token=s-9"


def test_normalize_url_empty():
    from ingest.sources import normalize_url
    assert normalize_url("") == ""
    assert normalize_url("   ") == ""


# ── get_song_by_url ────────────────────────────────────────────────────────────

def test_get_song_by_url(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="A", source_url="https://soundcloud.com/a/x")
    assert m.get_song_by_url("https://soundcloud.com/a/x")["id"] == sid
    assert m.get_song_by_url("https://soundcloud.com/a/y") is None
    assert m.get_song_by_url("") is None


# ── delete_song ────────────────────────────────────────────────────────────────

def test_delete_song_removes_rows_and_reports_files(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="A", source_url="https://sc/x", raw_path="/tmp/a.mp3")
    m.upsert_stem(sid, "vocals", "/tmp/a_vocals.wav")
    m.upsert_features(sid, "full", {"bpm": 120})
    res = m.delete_song(sid)
    assert res["existed"] is True
    assert set(res["files"]) == {"/tmp/a.mp3", "/tmp/a_vocals.wav"}
    assert m.get_song(sid) is None
    assert m.delete_song(sid) == {"existed": False, "files": []}


# ── update_song_url ──────────────────────────────────────────────────────────

def test_update_song_url_resets_pipeline(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="A", source_url="https://sc/x",
                        raw_path="/tmp/a.mp3", status="analysed")
    m.upsert_stem(sid, "vocals", "/tmp/a_vocals.wav")
    m.upsert_features(sid, "full", {"bpm": 120})
    res = m.update_song_url(sid, "https://sc/new")
    assert "/tmp/a.mp3" in res["files"]
    row = m.get_song(sid)
    assert row["source_url"] == "https://sc/new"
    assert row["status"] == "queued"
    assert row["raw_path"] == ""
    assert m.get_features_for_song(sid, "full") in (None, {})


def test_update_song_url_rejects_collision_and_empty(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    a = m.upsert_song(title="A", source_url="https://sc/a")
    b = m.upsert_song(title="B", source_url="https://sc/b")
    with pytest.raises(ValueError, match="already uses"):
        m.update_song_url(b, "https://sc/a")
    with pytest.raises(ValueError, match="empty"):
        m.update_song_url(a, "   ")


# ── route handlers ─────────────────────────────────────────────────────────────

def _tracks_module(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from api import queue_runner
    monkeypatch.setattr(queue_runner, "enqueue_song", lambda sid: f"job-{sid}")
    from api.routes import tracks
    importlib.reload(tracks)
    return tracks


def test_delete_route_unlinks_files(tmp_path, monkeypatch):
    tracks = _tracks_module(tmp_path, monkeypatch)
    from database import models as m
    mp3 = tmp_path / "a.mp3"; mp3.write_text("x")
    voc = tmp_path / "a_vocals.wav"; voc.write_text("x")
    sid = m.upsert_song(title="A", source_url="https://sc/x", raw_path=str(mp3))
    m.upsert_stem(sid, "vocals", str(voc))
    out = tracks.delete_track(sid)
    assert out["deleted"] and out["files_removed"] == 2
    assert not mp3.exists() and not voc.exists()
    with pytest.raises(HTTPException) as ei:
        tracks.delete_track(sid)
    assert ei.value.status_code == 404


def test_change_url_route_reprocesses_and_409_on_clash(tmp_path, monkeypatch):
    tracks = _tracks_module(tmp_path, monkeypatch)
    from database import models as m
    a = m.upsert_song(title="A", source_url="https://soundcloud.com/a/x")
    b = m.upsert_song(title="B", source_url="https://soundcloud.com/b/y")
    out = tracks.change_url(a, tracks.UrlUpdate(source_url="https://soundcloud.com/a/z"))
    assert out["updated"] and out["job_id"] == f"job-{a}"
    assert m.get_song(a)["source_url"] == "https://soundcloud.com/a/z"
    # collision with b's URL → 409
    with pytest.raises(HTTPException) as ei:
        tracks.change_url(a, tracks.UrlUpdate(source_url="https://soundcloud.com/b/y"))
    assert ei.value.status_code == 409
    # unknown host → 400
    with pytest.raises(HTTPException) as ei2:
        tracks.change_url(a, tracks.UrlUpdate(source_url="https://example.com/nope"))
    assert ei2.value.status_code == 400
