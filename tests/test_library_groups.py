"""Crates as LIBRARY GROUPS — the same table, asked the other question.

Discover asks "which crates hold this permalink" (crate_membership); the Library
asks "which of my songs are in this group". The second question is what makes a
crate more than a shopping list: once its items are ingested it is a shelf you
can filter the library down to, and a SoundCloud set imported with a name stays
a set afterwards.

What is worth pinning here is mostly the failure modes: a group must never claim
a song the library no longer has, a saved playlist must contain the tracks you
ALREADY owned as well as the new ones, and a library song with no permalink must
not collide with every other one under UNIQUE(crate_id, source_url).
"""
import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    return models


@pytest.fixture()
def app(db, monkeypatch):
    import api.routes.playlists as pl
    importlib.reload(pl)
    import api.routes.crates as crates
    importlib.reload(crates)
    import api.routes.discovery as disc
    importlib.reload(disc)
    import api.server as server
    importlib.reload(server)
    monkeypatch.setattr(pl.queue_runner, "enqueue_song", lambda sid: f"job-{sid}")
    return TestClient(server.app), db, pl


def _song(db, n, url=None):
    return db.upsert_song(title=f"Track {n}", artist="A",
                          source_url=url if url is not None
                          else f"https://soundcloud.com/a/t{n}",
                          duration_secs=200.0, track_id=str(n))


def _row(n, url=None):
    return {"title": f"Track {n}", "artist": "A", "track_id": str(n),
            "duration_secs": 200.0, "duration_str": "3:20", "thumbnail": "",
            "genre": "House", "plays": 10, "hydrated": True,
            "source_url": url or f"https://soundcloud.com/a/t{n}"}


# ── the model ────────────────────────────────────────────────────────────────

def test_a_group_lists_its_library_songs_in_crate_order(db):
    cid = db.create_crate("set")["id"]
    a, b, c = _song(db, 1), _song(db, 2), _song(db, 3)
    db.add_songs_to_crate(cid, [c, a, b])

    group = db.library_groups()[0]
    assert group["name"] == "set"
    assert group["song_ids"] == [c, a, b], "crate position, not song id order"
    assert group["item_count"] == 3 and group["ingested_count"] == 3


def test_items_are_written_already_linked(db):
    """The track is in the library by definition, so there is nothing left to
    ingest and no relink pass to wait for — the group filters immediately."""
    cid = db.create_crate("set")["id"]
    sid = _song(db, 1)
    db.add_songs_to_crate(cid, [sid])
    item = db.get_crate(cid)["items"][0]
    assert item["song_id"] == sid
    assert db.crate_payloads(cid, only_unlinked=True) == []


def test_adding_the_same_song_twice_is_a_skip(db):
    cid = db.create_crate("set")["id"]
    sid = _song(db, 1)
    assert db.add_songs_to_crate(cid, [sid, sid])["added"] == 1
    again = db.add_songs_to_crate(cid, [sid])
    assert again == {"added": 0, "skipped": 1, "item_ids": []}
    assert db.library_groups()[0]["song_ids"] == [sid]


def test_a_song_with_no_url_gets_its_own_sentinel_key(db):
    """'' would collide across every such song under UNIQUE(crate_id,
    source_url), silently making them one item."""
    cid = db.create_crate("set")["id"]
    a = _song(db, 1, url="")
    b = _song(db, 2, url=None)
    b_local = db.upsert_song(title="Local", artist="A", source_url=None,
                             duration_secs=10.0)
    db.add_songs_to_crate(cid, [a, b, b_local])
    urls = [i["source_url"] for i in db.get_crate(cid)["items"]]
    assert len(set(urls)) == 3
    assert sorted(db.library_groups()[0]["song_ids"]) == sorted([a, b, b_local])


def test_a_deleted_song_leaves_the_group(db):
    """The shortlist entry survives a deleted download — that is what a crate is
    for — but it must stop claiming to be a member, or the rail's count outruns
    the rows the filter can show."""
    cid = db.create_crate("set")["id"]
    a, b = _song(db, 1), _song(db, 2)
    db.add_songs_to_crate(cid, [a, b])
    db.delete_song(a)

    group = db.library_groups()[0]
    assert group["song_ids"] == [b]
    assert group["item_count"] == 2, "the crate item itself is kept"
    assert len(db.get_crate(cid)["items"]) == 2


def test_removing_a_song_redenses_positions(db):
    cid = db.create_crate("set")["id"]
    ids = [_song(db, n) for n in range(4)]
    db.add_songs_to_crate(cid, ids)
    assert db.remove_songs_from_crate(cid, [ids[1]]) == 1
    positions = [i["position"] for i in db.get_crate(cid)["items"]]
    assert positions == [0, 1, 2]
    assert db.library_groups()[0]["song_ids"] == [ids[0], ids[2], ids[3]]


def test_get_or_create_crate_is_case_insensitive(db):
    """Re-importing the rest of a playlist a week later must land on the same
    shelf rather than 409ing on UNIQUE(name)."""
    first = db.get_or_create_crate("Bootie 21")
    again = db.get_or_create_crate("bootie 21")
    assert again["id"] == first["id"]
    assert len(db.list_crates()) == 1


def test_relink_covers_every_crate(db):
    """A track imported from the Library paste bar can be the record a crate has
    been holding since you shortlisted it on Discover."""
    cid = db.create_crate("wishlist")["id"]
    url = "https://soundcloud.com/a/t9"
    db.add_crate_items(cid, [{"source_url": url, "title": "Track 9"}])
    assert db.library_groups()[0]["song_ids"] == []

    sid = db.upsert_song(title="Track 9", artist="A", source_url=url,
                         duration_secs=200.0)
    assert db.relink_crate_songs() == 1
    assert db.library_groups()[0]["song_ids"] == [sid]


# ── the routes ───────────────────────────────────────────────────────────────

def test_groups_is_declared_before_the_int_path(app):
    """/{crate_id} is typed int, so "groups" 422s rather than resolving if the
    declaration order is wrong. The hazard /membership already documents."""
    client, db, _ = app
    cid = db.create_crate("set")["id"]
    db.add_songs_to_crate(cid, [_song(db, 1)])

    res = client.get("/api/crates/groups")
    assert res.status_code == 200, res.text
    groups = res.json()["groups"]
    assert [g["name"] for g in groups] == ["set"]
    assert len(groups[0]["song_ids"]) == 1


def test_add_and_remove_songs_through_the_api(app):
    client, db, _ = app
    cid = db.create_crate("set")["id"]
    a, b = _song(db, 1), _song(db, 2)

    res = client.post(f"/api/crates/{cid}/songs", json={"song_ids": [a, b]})
    assert res.status_code == 200, res.text
    assert res.json()["added"] == 2

    res = client.post(f"/api/crates/{cid}/songs/remove", json={"song_ids": [a]})
    assert res.json()["removed"] == 1
    assert client.get("/api/crates/groups").json()["groups"][0]["song_ids"] == [b]


def test_adding_songs_to_a_missing_crate_is_a_404(app):
    client, _, _ = app
    res = client.post("/api/crates/999/songs", json={"song_ids": [1]})
    assert res.status_code == 404


def test_an_import_can_save_itself_as_a_group(app):
    client, db, _ = app
    res = client.post("/api/playlists/ingest",
                      json={"tracks": [_row(1), _row(2)], "group_name": "Bootie 21"})
    body = res.json()
    assert body["count"] == 2
    assert body["group"]["name"] == "Bootie 21"
    assert body["group"]["song_ids"] == body["inserted_ids"]


def test_a_saved_playlist_keeps_the_tracks_you_already_owned(app):
    """A group built only from the new rows would be missing exactly the tracks
    you already had — most of them, the second time you import from an artist
    you follow."""
    client, db, _ = app
    owned = db.upsert_song(title="Track 1", artist="A", duration_secs=200.0,
                           source_url="https://soundcloud.com/a/t1")

    body = client.post("/api/playlists/ingest",
                       json={"tracks": [_row(1), _row(2)],
                             "group_name": "set"}).json()
    assert body["count"] == 1 and body["skipped_count"] == 1
    assert body["group"]["song_ids"] == [owned, *body["inserted_ids"]], \
        "playlist order, with the already-owned track in its place"


def test_importing_without_a_name_groups_nothing(app):
    client, db, _ = app
    body = client.post("/api/playlists/ingest", json={"tracks": [_row(1)]}).json()
    assert body["group"] is None
    assert db.list_crates() == []


def test_a_second_import_lands_in_the_same_group(app):
    client, db, _ = app
    client.post("/api/playlists/ingest",
                json={"tracks": [_row(1)], "group_name": "set"})
    body = client.post("/api/playlists/ingest",
                       json={"tracks": [_row(2)], "group_name": "SET"}).json()
    assert len(db.list_crates()) == 1
    assert len(body["group"]["song_ids"]) == 2


def test_a_failed_grouping_never_fails_the_import(app, monkeypatch):
    """The tracks are saved and queued by the time this runs. Reporting the whole
    import as failed because a shelf label collided is a lie about the audio."""
    client, db, pl = app
    monkeypatch.setattr(pl, "get_or_create_crate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    body = client.post("/api/playlists/ingest",
                       json={"tracks": [_row(1)], "group_name": "set"}).json()
    assert body["count"] == 1
    assert body["group"] is None


def test_discovery_import_can_name_a_group(app):
    """Importing a SoundCloud set from Discover keeps it a set in the library."""
    client, db, _ = app
    body = client.post("/api/discovery/import",
                       json={"rows": [_row(1), _row(2)], "group_name": "A set"}).json()
    assert body["group"]["name"] == "A set"
    assert len(body["group"]["song_ids"]) == 2


def test_the_url_export_omits_local_sentinels(app):
    """A local:song/<id> key is deliberately not a URL — writing it into a URL
    file hands the importer something it can never resolve."""
    client, db, _ = app
    cid = db.create_crate("set")["id"]
    real = _song(db, 1)
    local = db.upsert_song(title="Local", artist="A", source_url=None,
                           duration_secs=10.0)
    db.add_songs_to_crate(cid, [real, local])
    body = client.get(f"/api/crates/{cid}/export?format=urls").text
    assert body.strip().splitlines() == ["https://soundcloud.com/a/t1"]
