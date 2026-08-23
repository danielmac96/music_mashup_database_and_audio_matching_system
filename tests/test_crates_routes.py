"""The /api/crates routes: shortlist, reorder, ingest, export."""
import importlib
import json
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

    import api.routes.playlists as pl
    importlib.reload(pl)
    import api.routes.crates as crates
    importlib.reload(crates)
    import api.server as server
    importlib.reload(server)

    monkeypatch.setattr(pl.queue_runner, "enqueue_song", lambda sid: f"job-{sid}")
    return TestClient(server.app), crates, models


def row(n, title=None, url=None):
    title = title or f"Track {n}"
    return {"title": title, "artist": "A", "track_id": str(n),
            "duration_secs": 200.0, "duration_str": "3:20",
            "source_url": url or f"https://soundcloud.com/a/t{n}",
            "thumbnail": "", "genre": "House", "plays": 10}


def make_crate(client, name="crate"):
    return client.post("/api/crates", json={"name": name}).json()["id"]


# ── CRUD ─────────────────────────────────────────────────────────────────────

def test_create_and_list(app):
    client, _, _ = app
    r = client.post("/api/crates", json={"name": "summer", "note": "warm"})
    assert r.status_code == 200
    assert r.json()["name"] == "summer"
    assert client.get("/api/crates").json()["crates"][0]["item_count"] == 0


def test_duplicate_name_is_409(app):
    client, _, _ = app
    client.post("/api/crates", json={"name": "dupe"})
    assert client.post("/api/crates", json={"name": "dupe"}).status_code == 409


def test_rename(app):
    client, _, _ = app
    cid = make_crate(client)
    assert client.patch(f"/api/crates/{cid}",
                        json={"name": "renamed"}).json()["name"] == "renamed"


def test_unknown_crate_is_404(app):
    client, _, _ = app
    assert client.get("/api/crates/999").status_code == 404
    assert client.post("/api/crates/999/items", json={"rows": [row(1)]}).status_code == 404
    assert client.post("/api/crates/999/ingest").status_code == 404


def test_delete(app):
    client, _, _ = app
    cid = make_crate(client)
    assert client.delete(f"/api/crates/{cid}").status_code == 200
    assert client.get(f"/api/crates/{cid}").status_code == 404


# ── items ────────────────────────────────────────────────────────────────────

def test_add_items_reports_added_and_skipped(app):
    client, _, _ = app
    cid = make_crate(client)
    first = client.post(f"/api/crates/{cid}/items",
                        json={"rows": [row(1), row(2)]}).json()
    assert (first["added"], first["skipped"]) == (2, 0)

    again = client.post(f"/api/crates/{cid}/items",
                        json={"rows": [row(1), row(3)]}).json()
    assert (again["added"], again["skipped"]) == (1, 1)
    assert len(again["crate"]["items"]) == 3


def test_add_normalises_urls_before_deduping(app):
    """Tracking params and a www. host are the same track, not three."""
    client, _, _ = app
    cid = make_crate(client)
    body = client.post(f"/api/crates/{cid}/items", json={"rows": [
        row(1, url="https://soundcloud.com/a/t1"),
        row(1, url="https://soundcloud.com/a/t1?utm_source=clipboard"),
        row(1, url="https://www.soundcloud.com/a/t1"),
    ]}).json()
    assert body["added"] == 1
    assert body["skipped"] == 2


def test_add_rejects_rows_without_a_url(app):
    client, _, _ = app
    cid = make_crate(client)
    r = client.post(f"/api/crates/{cid}/items", json={"rows": [{"title": "No URL"}]})
    assert r.status_code == 400


def test_remove_items(app):
    client, _, _ = app
    cid = make_crate(client)
    items = client.post(f"/api/crates/{cid}/items",
                        json={"rows": [row(1), row(2)]}).json()["crate"]["items"]
    body = client.post(f"/api/crates/{cid}/items/remove",
                       json={"item_ids": [items[0]["id"]]}).json()
    assert body["removed"] == 1
    assert [i["position"] for i in body["crate"]["items"]] == [0]


def test_reorder(app):
    client, _, _ = app
    cid = make_crate(client)
    items = client.post(f"/api/crates/{cid}/items",
                        json={"rows": [row(1), row(2), row(3)]}).json()["crate"]["items"]
    ids = [i["id"] for i in items]
    body = client.post(f"/api/crates/{cid}/reorder",
                       json={"item_ids": list(reversed(ids))}).json()
    assert [i["id"] for i in body["items"]] == list(reversed(ids))


def test_reorder_rejects_a_partial_list(app):
    """A short list would silently drop the tracks it omits."""
    client, _, _ = app
    cid = make_crate(client)
    items = client.post(f"/api/crates/{cid}/items",
                        json={"rows": [row(1), row(2)]}).json()["crate"]["items"]
    r = client.post(f"/api/crates/{cid}/reorder", json={"item_ids": [items[0]["id"]]})
    assert r.status_code == 400


# ── ingest ───────────────────────────────────────────────────────────────────

def test_ingest_saves_queues_and_relinks(app):
    client, _, models = app
    cid = make_crate(client)
    client.post(f"/api/crates/{cid}/items", json={"rows": [row(1), row(2)]})

    body = client.post(f"/api/crates/{cid}/ingest").json()
    assert body["count"] == 2
    assert len(body["job_ids"]) == 2
    assert body["linked"] == 2
    assert all(i["song_id"] for i in body["crate"]["items"])
    assert {s["title"] for s in models.get_all_songs()} == {"Track 1", "Track 2"}


def test_ingest_is_idempotent(app):
    """Re-ingesting must be a no-op, not a re-download."""
    client, _, _ = app
    cid = make_crate(client)
    client.post(f"/api/crates/{cid}/items", json={"rows": [row(1)]})
    assert client.post(f"/api/crates/{cid}/ingest").json()["count"] == 1

    second = client.post(f"/api/crates/{cid}/ingest").json()
    assert second["count"] == 0
    assert second["skipped_count"] == 0   # skipped BEFORE the call, so the count is honest


def test_ingest_needs_no_network(app, monkeypatch):
    """payload_json is what makes this true — the whole reason it exists."""
    client, _, _ = app
    import api.routes.playlists as pl
    monkeypatch.setattr(pl, "enrich_track",
                        lambda url: pytest.fail("crate ingest hit the network"))
    cid = make_crate(client)
    client.post(f"/api/crates/{cid}/items", json={"rows": [row(1)]})
    assert client.post(f"/api/crates/{cid}/ingest").json()["count"] == 1


def test_ingest_skips_items_already_in_the_library(app):
    client, _, models = app
    models.upsert_song(title="Track 1", artist="A",
                       source_url="https://soundcloud.com/a/t1")
    cid = make_crate(client)
    client.post(f"/api/crates/{cid}/items", json={"rows": [row(1), row(2)]})
    assert client.post(f"/api/crates/{cid}/ingest").json()["count"] == 1


# ── export ───────────────────────────────────────────────────────────────────

def test_export_urls_one_per_line_in_crate_order(app):
    client, _, _ = app
    cid = make_crate(client)
    items = client.post(f"/api/crates/{cid}/items",
                        json={"rows": [row(1), row(2)]}).json()["crate"]["items"]
    client.post(f"/api/crates/{cid}/reorder",
                json={"item_ids": [items[1]["id"], items[0]["id"]]})

    r = client.get(f"/api/crates/{cid}/export", params={"format": "urls"})
    assert r.status_code == 200
    assert r.text.strip().splitlines() == ["https://soundcloud.com/a/t2",
                                           "https://soundcloud.com/a/t1"]
    assert "attachment" in r.headers["content-disposition"]


def test_export_filename_is_sanitised(app):
    """The crate name lands in a Content-Disposition filename."""
    client, _, _ = app
    cid = make_crate(client, name="my crate / 2024!")
    disp = client.get(f"/api/crates/{cid}/export").headers["content-disposition"]
    assert "/" not in disp.split("filename=")[1]
    assert " " not in disp.split("filename=")[1]


def test_export_json_round_trips_through_add_items(app):
    """Export is an interchange format, not a dead end."""
    client, _, _ = app
    src = make_crate(client, "source")
    client.post(f"/api/crates/{src}/items", json={"rows": [row(1), row(2)]})
    exported = json.loads(client.get(f"/api/crates/{src}/export",
                                     params={"format": "json"}).text)

    dst = make_crate(client, "destination")
    body = client.post(f"/api/crates/{dst}/items", json={"rows": exported}).json()
    assert body["added"] == 2
    assert [i["title"] for i in body["crate"]["items"]] == ["Track 1", "Track 2"]


def test_export_m3u_lists_only_tracks_with_audio_on_disk(app, tmp_path):
    """An M3U of URLs is a lie to a media player."""
    client, _, models = app
    cid = make_crate(client)
    client.post(f"/api/crates/{cid}/items", json={"rows": [row(1), row(2)]})
    client.post(f"/api/crates/{cid}/ingest")

    song = models.get_song_by_url("https://soundcloud.com/a/t1")
    conn = models.get_conn()
    conn.execute("UPDATE songs SET raw_path=? WHERE id=?",
                 (str(tmp_path / "t1.mp3"), song["id"]))
    conn.commit()
    conn.close()

    body = client.get(f"/api/crates/{cid}/export", params={"format": "m3u"}).text
    assert body.startswith("#EXTM3U")
    assert "t1.mp3" in body
    assert "Track 2" not in body


def test_export_rejects_an_unknown_format(app):
    client, _, _ = app
    cid = make_crate(client)
    assert client.get(f"/api/crates/{cid}/export",
                      params={"format": "xspf"}).status_code == 422


# ── import from URLs ─────────────────────────────────────────────────────────

def test_import_urls_builds_a_crate(app, monkeypatch):
    client, crates, _ = app
    from ingest import soundcloud_browse as browse
    monkeypatch.setattr(browse, "resolve", lambda url, **kw: {
        "kind": "track", "item": row(1), "raw_id": "1"})

    body = client.post("/api/crates/import", json={
        "name": "imported", "urls": ["https://soundcloud.com/a/t1"]}).json()
    assert body["added"] == 1
    assert body["crate"]["name"] == "imported"


def test_import_urls_expands_a_playlist(app, monkeypatch):
    client, _, _ = app
    from ingest import soundcloud_browse as browse
    monkeypatch.setattr(browse, "resolve", lambda url, **kw: {
        "kind": "playlist", "item": {}, "raw_id": "7001"})
    monkeypatch.setattr(browse, "playlist", lambda pid, **kw: {
        "playlist": {}, "items": [row(1), row(2)], "next_cursor": None})

    body = client.post("/api/crates/import", json={
        "name": "set", "urls": ["https://soundcloud.com/a/sets/x"]}).json()
    assert body["added"] == 2


def test_import_urls_reports_what_it_could_not_resolve(app, monkeypatch):
    """One dead link must not lose the other nineteen."""
    client, _, _ = app
    from ingest import soundcloud_browse as browse
    from ingest.soundcloud_api import SoundCloudAPIError

    def resolve(url, **kw):
        if "dead" in url:
            raise SoundCloudAPIError("HTTP 404")
        return {"kind": "track", "item": row(1), "raw_id": "1"}

    monkeypatch.setattr(browse, "resolve", resolve)
    body = client.post("/api/crates/import", json={
        "name": "mixed",
        "urls": ["https://soundcloud.com/a/t1", "https://soundcloud.com/a/dead"]}).json()
    assert body["added"] == 1
    assert len(body["failed"]) == 1


def test_import_urls_rejects_an_empty_list(app):
    client, _, _ = app
    assert client.post("/api/crates/import",
                       json={"name": "x", "urls": ["  "]}).status_code == 400


# ── membership: crate badges on Discovery rows ───────────────────────────────

def test_membership_resolves_a_messy_url_and_keys_by_the_string_as_sent(app):
    """The route normalises, the model does not — the same split add_items uses.
    The response is keyed by what the caller sent so the frontend can look up
    row.source_url directly instead of re-implementing normalise in JS."""
    client, _, _ = app
    cid = make_crate(client, "Vocals")
    messy = "http://www.soundcloud.com/a/t1/?si=abc123&utm_source=clipboard"
    client.post(f"/api/crates/{cid}/items", json={"rows": [row(1, url=messy)]})

    body = client.post("/api/crates/membership", json={"urls": [messy]}).json()
    assert list(body["membership"]) == [messy]
    assert [c["name"] for c in body["membership"][messy]] == ["Vocals"]


def test_membership_resolves_an_unnormalised_variant_of_a_stored_url(app):
    client, _, _ = app
    cid = make_crate(client, "Vocals")
    client.post(f"/api/crates/{cid}/items",
                json={"rows": [row(1, url="https://soundcloud.com/a/t1")]})

    variant = "http://m.soundcloud.com/a/t1?si=deadbeef"
    body = client.post("/api/crates/membership", json={"urls": [variant]}).json()
    assert [c["name"] for c in body["membership"][variant]] == ["Vocals"]


def test_membership_of_an_empty_list_is_200_and_empty(app):
    client, _, _ = app
    r = client.post("/api/crates/membership", json={"urls": []})
    assert r.status_code == 200
    assert r.json() == {"membership": {}}


def test_membership_reports_two_crates_for_one_row(app):
    client, _, _ = app
    url = "https://soundcloud.com/a/t1"
    for name in ("Vocals", "Instrumentals"):
        client.post(f"/api/crates/{make_crate(client, name)}/items",
                    json={"rows": [row(1, url=url)]})
    body = client.post("/api/crates/membership", json={"urls": [url]}).json()
    assert [c["name"] for c in body["membership"][url]] == ["Instrumentals", "Vocals"]


def test_membership_matches_on_track_id_too(app):
    client, _, _ = app
    cid = make_crate(client, "Vocals")
    client.post(f"/api/crates/{cid}/items", json={"rows": [row(7)]})
    body = client.post("/api/crates/membership",
                       json={"urls": [], "track_ids": ["7"]}).json()
    assert [c["name"] for c in body["membership"]["7"]] == ["Vocals"]


def test_membership_caps_the_request(app):
    client, _, _ = app
    urls = [f"https://soundcloud.com/a/t{n}" for n in range(201)]
    assert client.post("/api/crates/membership", json={"urls": urls}).status_code == 400


def test_membership_does_not_collide_with_the_crate_detail_route(app):
    """/{crate_id} is typed int, so /membership must be declared first or it
    422s instead of resolving."""
    client, _, _ = app
    assert client.post("/api/crates/membership", json={"urls": []}).status_code == 200
    cid = make_crate(client, "real")
    assert client.get(f"/api/crates/{cid}").json()["name"] == "real"
    assert client.get("/api/crates/membership").status_code in (405, 422)
