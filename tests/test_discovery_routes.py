"""The /api/discovery routes.

The browse layer is monkeypatched in the route module's namespace — the pattern
the mixes tests already use for `sc_search_candidates`. Nothing here touches the
network, and nothing here re-tests the browse layer itself.
"""
import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


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

    import api.routes.discovery as disc
    importlib.reload(disc)
    import api.routes.playlists as pl
    importlib.reload(pl)
    import api.server as server
    importlib.reload(server)

    # Never let a test enqueue real pipeline work.
    monkeypatch.setattr(pl.queue_runner, "enqueue_song", lambda sid: f"job-{sid}")
    return TestClient(server.app), disc, models


def track(tid, title="T", url=None, **kw):
    """A canonical browse row, as soundcloud_browse.track_row emits it."""
    return {"title": title, "artist": "A", "artist_id": "900", "track_id": str(tid),
            "duration_secs": 200.0, "duration_str": "3:20",
            "source_url": url or f"https://soundcloud.com/a/{title.lower()}",
            "upload_date": "20240115", "likes": 1, "reposts": 0, "comments": 0,
            "plays": 10, "thumbnail": "", "genre": "House", "tags": "",
            "release_year": 2024, "is_snip": False, **kw}


# ── search ───────────────────────────────────────────────────────────────────

def test_search_returns_items_and_cursor(app, monkeypatch):
    client, disc, _ = app
    monkeypatch.setattr(disc.browse, "search",
                        lambda kind, q, **kw: {"items": [track(1, "One")],
                                               "next_cursor": "https://api-v2.soundcloud.com/next"})
    r = client.get("/api/discovery/search", params={"q": "night", "kind": "tracks"})
    assert r.status_code == 200
    body = r.json()
    assert body["items"][0]["title"] == "One"
    assert body["next_cursor"].endswith("/next")


def test_search_passes_kind_and_cursor_through(app, monkeypatch):
    client, disc, _ = app
    seen = {}

    def fake(kind, q, **kw):
        seen.update(kind=kind, q=q, **kw)
        return {"items": [], "next_cursor": None}

    monkeypatch.setattr(disc.browse, "search", fake)
    client.get("/api/discovery/search",
               params={"q": "x", "kind": "playlists", "cursor": "C", "limit": 5})
    assert seen["kind"] == "playlists"
    assert seen["cursor"] == "C"
    assert seen["limit"] == 5


def test_search_rejects_unknown_kind(app):
    client, _, _ = app
    assert client.get("/api/discovery/search",
                      params={"q": "x", "kind": "albums"}).status_code == 422


# ── the in_library annotation ────────────────────────────────────────────────

def test_annotates_tracks_already_in_the_library(app, monkeypatch):
    """Without this the browser has no way to stop you importing a track twice."""
    client, disc, models = app
    have = models.upsert_song(title="Have", artist="A",
                              source_url="https://soundcloud.com/a/have")
    monkeypatch.setattr(disc.browse, "search", lambda *a, **kw: {
        "items": [track(1, "Have", "https://soundcloud.com/a/have"),
                  track(2, "Missing", "https://soundcloud.com/a/missing")],
        "next_cursor": None})

    items = client.get("/api/discovery/search", params={"q": "x"}).json()["items"]
    assert items[0]["in_library"]["song_id"] == have
    assert items[0]["in_library"]["status"] == "queued"
    assert items[1]["in_library"] is None


def test_annotation_matches_on_track_id_when_the_permalink_moved(app, monkeypatch):
    """A renamed permalink is the case source_url alone cannot catch."""
    client, disc, models = app
    sid = models.upsert_song(title="Renamed", artist="A",
                             source_url="https://soundcloud.com/a/old-name",
                             track_id="4242")
    monkeypatch.setattr(disc.browse, "search", lambda *a, **kw: {
        "items": [track(4242, "Renamed", "https://soundcloud.com/a/new-name")],
        "next_cursor": None})

    item = client.get("/api/discovery/search", params={"q": "x"}).json()["items"][0]
    assert item["in_library"]["song_id"] == sid


def test_annotation_ignores_empty_track_ids(app, monkeypatch):
    """'' is the default for rows that never learned their id; matching on it
    would mark every unrelated result as already-owned."""
    client, disc, models = app
    models.upsert_song(title="NoTid", artist="A", source_url="https://soundcloud.com/a/x")
    monkeypatch.setattr(disc.browse, "search", lambda *a, **kw: {
        "items": [track("", "Other", "https://soundcloud.com/a/other")],
        "next_cursor": None})

    assert client.get("/api/discovery/search",
                      params={"q": "x"}).json()["items"][0]["in_library"] is None


def test_playlist_and_user_rows_are_not_annotated(app, monkeypatch):
    client, disc, _ = app
    monkeypatch.setattr(disc.browse, "search", lambda *a, **kw: {
        "items": [{"kind": "playlist", "playlist_id": "7", "title": "Set",
                   "source_url": "https://soundcloud.com/a/sets/set"}],
        "next_cursor": None})
    item = client.get("/api/discovery/search",
                      params={"q": "x", "kind": "playlists"}).json()["items"][0]
    assert "in_library" not in item


# ── resolve ──────────────────────────────────────────────────────────────────

def test_resolve_playlist_returns_its_tracks(app, monkeypatch):
    """Pasting a set link should land on its tracks, not on a stub to click."""
    client, disc, _ = app
    monkeypatch.setattr(disc.browse, "resolve", lambda url, **kw: {
        "kind": "playlist", "item": {"kind": "playlist"}, "raw_id": "7001"})
    monkeypatch.setattr(disc.browse, "playlist", lambda pid, **kw: {
        "playlist": {"kind": "playlist", "title": "Summer Crate", "track_count": 2},
        "items": [track(1, "One"), track(2, "Two")], "next_cursor": None})

    body = client.post("/api/discovery/resolve",
                       json={"url": "https://soundcloud.com/a/sets/summer"}).json()
    assert body["kind"] == "playlist"
    assert body["item"]["title"] == "Summer Crate"
    assert [i["title"] for i in body["items"]] == ["One", "Two"]


def test_resolve_user_returns_their_tracks(app, monkeypatch):
    client, disc, _ = app
    monkeypatch.setattr(disc.browse, "resolve", lambda url, **kw: {
        "kind": "user", "item": {"kind": "user", "username": "artistone"}, "raw_id": "900"})
    monkeypatch.setattr(disc.browse, "user_tracks", lambda uid, **kw: {
        "items": [track(1, "One")], "next_cursor": None})

    body = client.post("/api/discovery/resolve",
                       json={"url": "https://soundcloud.com/artistone"}).json()
    assert body["kind"] == "user"
    assert body["item"]["username"] == "artistone"
    assert len(body["items"]) == 1


def test_resolve_rejects_a_non_soundcloud_url(app):
    client, _, _ = app
    r = client.post("/api/discovery/resolve",
                    json={"url": "https://youtube.com/watch?v=abc"})
    assert r.status_code == 400
    assert "SoundCloud" in r.json()["detail"]


def test_resolve_requires_a_url(app):
    client, _, _ = app
    assert client.post("/api/discovery/resolve", json={"url": "  "}).status_code == 400


# ── errors ───────────────────────────────────────────────────────────────────

def test_upstream_failure_is_502_not_500(app, monkeypatch):
    client, disc, _ = app

    def boom(*a, **kw):
        raise disc.SoundCloudAPIError("SoundCloud HTTP 404")

    monkeypatch.setattr(disc.browse, "search", boom)
    r = client.get("/api/discovery/search", params={"q": "x"})
    assert r.status_code == 502
    assert "404" in r.json()["detail"]


def test_cooling_down_is_503(app, monkeypatch):
    """A distinct code so the UI can say "try again shortly" rather than
    presenting a deliberate backoff as a failure."""
    client, disc, _ = app

    def boom(*a, **kw):
        raise disc.SoundCloudUnavailable("cooling down for 42s")

    monkeypatch.setattr(disc.browse, "search", boom)
    r = client.get("/api/discovery/search", params={"q": "x"})
    assert r.status_code == 503
    assert "cooling down" in r.json()["detail"]


# ── import ───────────────────────────────────────────────────────────────────

def test_import_saves_and_queues_through_the_playlists_path(app):
    client, _, models = app
    r = client.post("/api/discovery/import",
                    json={"rows": [track(1, "One"), track(2, "Two")]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["job_ids"]) == 2

    saved = {s["title"] for s in models.get_all_songs()}
    assert saved == {"One", "Two"}


def test_import_preserves_the_soundcloud_track_id(app):
    """The id the in_library annotation depends on, so it has to survive ingest."""
    client, _, models = app
    client.post("/api/discovery/import", json={"rows": [track(4242, "One")]})
    assert models.get_song_by_track_id("4242") is not None


def test_import_reports_duplicates_instead_of_redownloading(app):
    client, _, models = app
    models.upsert_song(title="One", artist="A", source_url="https://soundcloud.com/a/one")
    body = client.post("/api/discovery/import",
                       json={"rows": [track(1, "One"), track(2, "Two")]}).json()
    assert body["count"] == 1
    assert body["skipped_count"] == 1
    assert body["skipped"][0]["title"] == "One"


def test_import_does_not_refetch_metadata(app, monkeypatch):
    """Browse rows are already hydrated; a refetch here would make importing 40
    tracks 40 needless network calls."""
    client, _, _ = app
    import api.routes.playlists as pl
    monkeypatch.setattr(pl, "enrich_track",
                        lambda url: pytest.fail("refetched an already-hydrated row"))
    assert client.post("/api/discovery/import",
                       json={"rows": [track(1, "One")]}).json()["count"] == 1


def test_import_rejects_an_empty_list(app):
    client, _, _ = app
    assert client.post("/api/discovery/import", json={"rows": []}).status_code == 400


# ── status ───────────────────────────────────────────────────────────────────

def test_status_reports_read_on_write_off(app):
    """Read works with the scraped client_id; write needs a registered app."""
    client, _, _ = app
    body = client.get("/api/discovery/status").json()
    assert body["read_enabled"] is True
    assert body["write_enabled"] is False
    assert body["account"]["configured"] is False
