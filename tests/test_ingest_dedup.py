"""ingest_rows: ask the library before asking the network, and dedup on identity."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A temp library, the network stubbed, and a count of enrichment calls.

    `config.DB_PATH` binds at import, so the DB is pointed at the temp folder
    and the modules reloaded (the order the other ingest tests use). Without
    that, `ingest_rows` would write to whatever database `config` resolved to
    at collection time: the live one.
    """
    db = tmp_path / "i.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import playlists
    importlib.reload(playlists)

    calls: list[str] = []

    def fake_enrich(url):
        calls.append(url)
        return {"source_url": url, "title": f"T {url[-1]}", "artist": "A",
                "duration_secs": 200.0, "track_id": url[-1], "hydrated": True}

    monkeypatch.setattr(playlists, "enrich_track", fake_enrich)
    monkeypatch.setattr(playlists.queue_runner, "enqueue_song",
                        lambda sid, **kw: f"job-{sid}")
    monkeypatch.setattr(playlists.preview_hydrator, "cache_get", lambda url: None)
    monkeypatch.setattr(playlists.preview_hydrator, "cache_put", lambda url, row: None)
    return SimpleNamespace(db=db, models=models, playlists=playlists, enriched=calls)


def test_a_row_already_in_the_library_is_never_enriched(lib):
    lib.models.upsert_song(title="Owned", source_url="https://soundcloud.com/a/1",
                           track_id="1", db_path=lib.db)

    out = lib.playlists.ingest_rows([{"source_url": "https://soundcloud.com/a/1"},
                                     {"source_url": "https://soundcloud.com/a/2"}])

    assert lib.enriched == ["https://soundcloud.com/a/2"], lib.enriched
    assert out["skipped_count"] == 1
    assert out["count"] == 1
    assert out["enriched_count"] == 1


def test_the_same_record_under_two_links_imports_once(lib):
    """A share link and the canonical link are one record, one pipeline run."""
    sid = lib.models.upsert_song(title="Owned", source_url="https://soundcloud.com/a/7",
                                 track_id="7", db_path=lib.db)

    out = lib.playlists.ingest_rows(
        [{"source_url": "https://soundcloud.com/a/7?secret_token=xyz",
          "track_id": "7", "title": "Owned", "hydrated": True}])

    assert out["count"] == 0
    assert out["skipped_count"] == 1
    assert out["ordered_song_ids"] == [sid]


def test_ordered_song_ids_stays_positionally_aligned(lib):
    owned = lib.models.upsert_song(title="Owned", source_url="https://soundcloud.com/a/1",
                                   track_id="1", db_path=lib.db)

    out = lib.playlists.ingest_rows([
        {"source_url": "https://soundcloud.com/a/1"},   # owned
        {"source_url": ""},                              # unresolvable
        {"source_url": "https://soundcloud.com/a/3"},   # new
    ])

    ids = out["ordered_song_ids"]
    assert len(ids) == 3
    assert ids[0] == owned
    assert ids[1] is None
    assert ids[2] in out["inserted_ids"]


def test_the_batch_runs_on_one_connection(lib, monkeypatch):
    opens = {"n": 0}
    real_get_conn = lib.models.get_conn

    def counting(*a, **k):
        opens["n"] += 1
        return real_get_conn(*a, **k)

    monkeypatch.setattr(lib.models, "get_conn", counting)
    lib.playlists.ingest_rows([{"source_url": f"https://soundcloud.com/a/{i}"}
                               for i in range(20)])
    # One for the batch, plus whatever the group/crate steps need: nowhere near
    # one per row.
    assert opens["n"] <= 6, opens["n"]


def test_a_record_repeated_inside_one_import_is_saved_once(lib):
    """The library check runs once, up front, so it cannot see a row the same
    import is about to write. Without a second look after enrichment the repeat
    upserts onto the first row, resets it to `queued` and is queued twice."""
    out = lib.playlists.ingest_rows([
        {"source_url": "https://soundcloud.com/a/5"},
        {"source_url": "https://soundcloud.com/a/5"},
    ])

    assert out["count"] == 1
    assert out["skipped_count"] == 1
    first, second = out["ordered_song_ids"]
    assert first == second == out["inserted_ids"][0]


def test_one_unsaveable_row_does_not_lose_the_batch(lib, monkeypatch):
    import sqlite3
    real = lib.playlists.upsert_song

    def flaky(**kw):
        if kw["source_url"].endswith("/2"):
            raise sqlite3.IntegrityError("boom")
        return real(**kw)

    monkeypatch.setattr(lib.playlists, "upsert_song", flaky)
    out = lib.playlists.ingest_rows([{"source_url": f"https://soundcloud.com/a/{i}"}
                                     for i in (1, 2, 3)])

    assert out["count"] == 2
    assert out["unresolvable_count"] == 1
    ids = out["ordered_song_ids"]
    assert ids[0] and ids[1] is None and ids[2]
    assert lib.models.get_song_by_url("https://soundcloud.com/a/3",
                                      db_path=lib.db) is not None
