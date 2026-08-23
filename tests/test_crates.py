"""Crates — the local shortlist that stands in for SoundCloud playlist editing.

Model-level coverage: dedup on add, dense positions, the reorder contract, and
the ingest handoff (payload_json is what makes a crate ingestable with no further
network calls).
"""
import importlib
import sys
from pathlib import Path

import pytest

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


def _row(url, title="T", artist="A", track_id="", **extra):
    return {"source_url": url, "title": title, "artist": artist,
            "track_id": track_id, "duration_secs": 200.0, **extra}


def test_create_and_list(db):
    cid = db.create_crate("summer", note="warm ones")["id"]
    crates = db.list_crates()
    assert len(crates) == 1
    assert crates[0]["name"] == "summer"
    assert crates[0]["item_count"] == 0
    assert crates[0]["ingested_count"] == 0
    assert db.get_crate(cid)["items"] == []


def test_duplicate_name_rejected(db):
    db.create_crate("dupe")
    with pytest.raises(Exception):
        db.create_crate("dupe")


def test_add_dedups_within_request_and_against_db(db):
    cid = db.create_crate("c")["id"]
    url = "https://soundcloud.com/artist/track"
    # Same URL twice in one call, then again in a second call.
    first = db.add_crate_items(cid, [_row(url), _row(url), _row(url + "-2")])
    assert (first["added"], first["skipped"]) == (2, 1)
    second = db.add_crate_items(cid, [_row(url)])
    assert (second["added"], second["skipped"]) == (0, 1)
    assert len(db.get_crate(cid)["items"]) == 2


def test_positions_stay_dense_across_add_and_remove(db):
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row(f"https://soundcloud.com/a/t{i}") for i in range(4)])
    items = db.get_crate(cid)["items"]
    assert [i["position"] for i in items] == [0, 1, 2, 3]

    db.remove_crate_items(cid, [items[1]["id"]])
    after = db.get_crate(cid)["items"]
    assert [i["position"] for i in after] == [0, 1, 2]
    assert [i["source_url"] for i in after] == [
        "https://soundcloud.com/a/t0", "https://soundcloud.com/a/t2",
        "https://soundcloud.com/a/t3"]

    # A later add continues from the compacted end rather than colliding.
    db.add_crate_items(cid, [_row("https://soundcloud.com/a/t9")])
    assert [i["position"] for i in db.get_crate(cid)["items"]] == [0, 1, 2, 3]


def test_reorder_rewrites_positions(db):
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row(f"https://soundcloud.com/a/t{i}") for i in range(3)])
    ids = [i["id"] for i in db.get_crate(cid)["items"]]
    db.reorder_crate(cid, list(reversed(ids)))
    assert [i["id"] for i in db.get_crate(cid)["items"]] == list(reversed(ids))


def test_reorder_rejects_partial_id_set(db):
    """A short list would silently drop the tracks it omits."""
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row(f"https://soundcloud.com/a/t{i}") for i in range(3)])
    ids = [i["id"] for i in db.get_crate(cid)["items"]]
    with pytest.raises(ValueError):
        db.reorder_crate(cid, ids[:2])
    with pytest.raises(ValueError):
        db.reorder_crate(cid, ids + [99999])


def test_add_links_song_already_in_library(db):
    url = "https://soundcloud.com/a/known"
    sid = db.upsert_song(title="Known", artist="A", source_url=url)
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row(url)])
    item = db.get_crate(cid)["items"][0]
    assert item["song_id"] == sid
    assert item["song_status"] == "queued"


def test_relink_after_ingest(db):
    url = "https://soundcloud.com/a/later"
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row(url)])
    assert db.get_crate(cid)["items"][0]["song_id"] is None

    db.upsert_song(title="Later", artist="A", source_url=url)
    assert db.relink_crate_songs(cid) == 1
    assert db.get_crate(cid)["items"][0]["song_id"] is not None
    # Idempotent: a second pass has nothing left to link.
    assert db.relink_crate_songs(cid) == 0


def test_crate_payloads_round_trip_the_full_row(db):
    """payload_json is what lets a crate be ingested with no network calls."""
    cid = db.create_crate("c")["id"]
    row = _row("https://soundcloud.com/a/t", title="Real Title", artist="Real Artist",
               track_id="12345", genre="House", plays=999, hydrated=True)
    db.add_crate_items(cid, [row])
    payloads = db.crate_payloads(cid)
    assert len(payloads) == 1
    assert payloads[0]["title"] == "Real Title"
    assert payloads[0]["genre"] == "House"
    assert payloads[0]["plays"] == 999
    assert payloads[0]["hydrated"] is True


def test_crate_payloads_skips_already_ingested(db):
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row("https://soundcloud.com/a/t1"),
                             _row("https://soundcloud.com/a/t2")])
    db.upsert_song(title="T1", artist="A", source_url="https://soundcloud.com/a/t1")
    db.relink_crate_songs(cid)

    assert len(db.crate_payloads(cid, only_unlinked=True)) == 1
    assert len(db.crate_payloads(cid, only_unlinked=False)) == 2


def test_delete_crate_removes_items(db):
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row("https://soundcloud.com/a/t")])
    assert db.delete_crate(cid) is True
    assert db.get_crate(cid) is None
    conn = db.get_conn()
    assert conn.execute("SELECT COUNT(*) AS n FROM crate_items").fetchone()["n"] == 0
    conn.close()


def test_counts_reflect_ingest_state(db):
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row("https://soundcloud.com/a/t1"),
                             _row("https://soundcloud.com/a/t2")])
    db.upsert_song(title="T1", artist="A", source_url="https://soundcloud.com/a/t1")
    db.relink_crate_songs(cid)
    crate = db.list_crates()[0]
    assert crate["item_count"] == 2
    assert crate["ingested_count"] == 1


# ── app_prefs ────────────────────────────────────────────────────────────────
# The kv store behind the connected SoundCloud profile. It exists rather than a
# settings.json key because config.save_settings ignores empty values, so nothing
# written there can ever be unset — and this one has a Disconnect button.

def test_pref_round_trips(db):
    assert db.get_pref("sc") is None
    db.set_pref("sc", {"user_id": "55", "username": "Me"})
    assert db.get_pref("sc")["username"] == "Me"


def test_pref_overwrites_rather_than_duplicating(db):
    db.set_pref("sc", {"user_id": "55"})
    db.set_pref("sc", {"user_id": "66"})
    assert db.get_pref("sc") == {"user_id": "66"}


def test_pref_clears(db):
    db.set_pref("sc", {"user_id": "55"})
    assert db.clear_pref("sc") is True
    assert db.get_pref("sc") is None
    # Clearing something that was never set is not an error, just False.
    assert db.clear_pref("sc") is False


def test_corrupt_pref_reads_as_absent(db):
    """A scrap of UI state that no longer parses must not be able to take down
    the route that reads it."""
    conn = db.get_conn()
    conn.execute("INSERT INTO app_prefs (key, value) VALUES ('sc', 'not json')")
    conn.commit()
    conn.close()
    assert db.get_pref("sc") is None


def test_pref_rejects_a_non_dict(db):
    with pytest.raises(ValueError):
        db.set_pref("sc", ["not", "a", "dict"])


# ── crate membership: which crates already hold this Discovery row ───────────
# Badging a page of results is one query, not one per row — the same contract
# songs_by_identity holds for the "in library" flag.

def test_membership_lists_every_crate_a_url_is_in_ordered_by_name(db):
    url = "https://soundcloud.com/a/t1"
    for name in ("Vocals", "Instrumentals"):
        db.add_crate_items(db.create_crate(name)["id"], [_row(url)])

    got = db.crate_membership(source_urls=[url])["by_url"][url]
    assert [c["name"] for c in got] == ["Instrumentals", "Vocals"]
    # item_id is the handle remove_crate_items takes; carried so a future
    # per-row remove needs no reshaped response.
    assert all(isinstance(c["item_id"], int) for c in got)
    assert all(isinstance(c["crate_id"], int) for c in got)


def test_membership_omits_a_url_in_no_crate(db):
    db.add_crate_items(db.create_crate("c")["id"], [_row("https://soundcloud.com/a/t1")])
    got = db.crate_membership(source_urls=["https://soundcloud.com/a/t1",
                                           "https://soundcloud.com/a/unfiled"])
    assert "https://soundcloud.com/a/unfiled" not in got["by_url"]
    assert list(got["by_url"]) == ["https://soundcloud.com/a/t1"]


def test_membership_never_matches_on_an_empty_track_id(db):
    """'' is the default for a row that never learned its id. Matching on it
    would claim every such row is in every crate holding any other such row —
    the exact bug songs_by_identity guards against."""
    cid = db.create_crate("c")["id"]
    db.add_crate_items(cid, [_row("https://soundcloud.com/a/t1", track_id="")])
    db.add_crate_items(cid, [_row("https://soundcloud.com/a/t2", track_id="99")])

    got = db.crate_membership(track_ids=["", "99"])
    assert "" not in got["by_track_id"]
    assert [c["name"] for c in got["by_track_id"]["99"]] == ["c"]


def test_membership_matches_on_track_id_when_the_url_differs(db):
    db.add_crate_items(db.create_crate("c")["id"],
                       [_row("https://soundcloud.com/a/t1", track_id="42")])
    got = db.crate_membership(source_urls=["https://soundcloud.com/somewhere/else"],
                              track_ids=["42"])
    assert got["by_url"] == {}
    assert [c["name"] for c in got["by_track_id"]["42"]] == ["c"]


def test_membership_with_no_input_does_not_touch_the_database(db, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("crate_membership opened a connection for no input")
    monkeypatch.setattr(db, "get_conn", boom)
    assert db.crate_membership() == {"by_url": {}, "by_track_id": {}}
    assert db.crate_membership(source_urls=[""], track_ids=[""]) == {
        "by_url": {}, "by_track_id": {}}


def test_membership_is_one_query_for_a_whole_page(db, monkeypatch):
    cid = db.create_crate("c")["id"]
    urls = [f"https://soundcloud.com/a/t{n}" for n in range(50)]
    db.add_crate_items(cid, [_row(u, track_id=str(n)) for n, u in enumerate(urls)])

    real_conn = db.get_conn
    calls = []

    def counting_conn(*a, **kw):
        # sqlite3.Connection.execute is read-only, so count through the trace
        # callback instead of wrapping the method.
        conn = real_conn(*a, **kw)
        conn.set_trace_callback(
            lambda sql: calls.append(sql) if "crate_items" in sql else None)
        return conn

    monkeypatch.setattr(db, "get_conn", counting_conn)
    got = db.crate_membership(source_urls=urls, track_ids=[str(n) for n in range(50)])
    assert len(calls) == 1, calls
    assert len(got["by_url"]) == 50
    assert len(got["by_track_id"]) == 50
