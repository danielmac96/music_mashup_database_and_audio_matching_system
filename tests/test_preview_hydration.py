"""Tests for progressive playlist-preview hydration (api/preview_hydrator.py)
and its wiring into the preview/ingest routes. Network (enrich_track) is mocked.
"""
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def clean_cache():
    from api import preview_hydrator
    # The two caches have separate locks: _CACHE (url -> enriched metadata) is
    # guarded by _CACHE_LOCK, _SESSIONS by _LOCK. Clearing both under _LOCK alone
    # would be taking the wrong lock for half of it.
    with preview_hydrator._CACHE_LOCK:
        preview_hydrator._CACHE.clear()
    with preview_hydrator._LOCK:
        preview_hydrator._SESSIONS.clear()
    yield


def _wait_done(preview_id, timeout=5.0):
    from api import preview_hydrator
    deadline = time.time() + timeout
    while time.time() < deadline:
        session = preview_hydrator.get(preview_id)
        if session and session["done"]:
            return session
        time.sleep(0.02)
    raise AssertionError("hydration did not finish in time")


def test_hydrator_merges_rich_metadata(monkeypatch):
    import ingest.soundcloud as sc
    from api import preview_hydrator

    def fake_enrich(url):
        if url.endswith("/blocked"):
            return None  # e.g. geo/Go+ track — stays flat
        return {"title": f"Rich {url[-1]}", "artist": "Artist", "likes": 42,
                "source_url": url}

    monkeypatch.setattr(sc, "enrich_track", fake_enrich)

    flat = [
        {"title": "Unknown", "artist": "", "source_url": "http://h/1"},
        {"title": "Unknown", "artist": "", "source_url": "http://h/blocked"},
    ]
    pid = preview_hydrator.start(flat)
    session = _wait_done(pid)

    assert session["count"] == 2
    assert session["hydrated_count"] == 2   # both fetches finished…
    assert session["enriched_count"] == 1   # …but only one succeeded
    row_ok, row_blocked = session["tracks"]
    assert row_ok["hydrated"] is True
    assert row_ok["title"] == "Rich 1"
    assert row_ok["artist"] == "Artist"
    # `hydrated` means "we finished trying", not "it worked" — deliberately, and
    # documented on _hydrate_one. Two things depend on it: the session can only
    # reach done when every row is marked, and playlists._resolve_metadata reads
    # it to decide whether to refetch at ingest time (a blocked track SHOULD get
    # one more live attempt there). enriched_count above is the "did it actually
    # work" signal, which is what this row is really being asked about.
    assert row_blocked["hydrated"] is True
    assert row_blocked["title"] == "Unknown"  # flat row preserved

    # Successful fetches land in the ingest-reuse cache.
    assert preview_hydrator.cache_get("http://h/1")["likes"] == 42
    assert preview_hydrator.cache_get("http://h/blocked") is None


def test_hydrator_merge_never_blanks_flat_fields(monkeypatch):
    import ingest.soundcloud as sc
    from api import preview_hydrator

    monkeypatch.setattr(sc, "enrich_track", lambda url: {
        "title": "Rich", "thumbnail": "", "source_url": url})

    pid = preview_hydrator.start(
        [{"title": "Flat", "thumbnail": "http://img/1.jpg", "source_url": "http://h/2"}])
    session = _wait_done(pid)
    row = session["tracks"][0]
    assert row["title"] == "Rich"
    assert row["thumbnail"] == "http://img/1.jpg"  # empty rich value kept flat's


def test_ingest_uses_hydrated_rows_without_refetch(monkeypatch):
    import api.routes.playlists as playlists_route

    calls = []
    monkeypatch.setattr(playlists_route, "enrich_track",
                        lambda url: calls.append(url) or None)

    saved = []

    def fake_upsert(title, **kw):
        saved.append({"title": title, **kw})
        return len(saved)

    monkeypatch.setattr(playlists_route, "upsert_song", fake_upsert)
    monkeypatch.setattr(playlists_route.queue_runner, "enqueue_song",
                        lambda sid: f"job-{sid}")

    body = playlists_route.ingest(playlists_route.IngestRequest(tracks=[
        {"title": "Hydrated", "artist": "A", "source_url": "http://h/10",
         "hydrated": True, "likes": 7},
        {"title": "Flat", "artist": "", "source_url": "http://h/11"},
    ]))

    assert body["count"] == 2
    # The hydrated row was trusted as-is; only the flat row hit enrich_track.
    assert calls == ["http://h/11"]
    assert body["partial_count"] == 1
    assert saved[0]["title"] == "Hydrated"
    assert saved[0]["metadata_partial"] == 0
    assert saved[0]["likes"] == 7
    assert saved[1]["metadata_partial"] == 1


def test_ingest_uses_preview_cache_before_refetching(monkeypatch):
    import api.routes.playlists as playlists_route
    from api import preview_hydrator

    preview_hydrator.cache_put("http://h/20", {
        "title": "Cached", "artist": "C", "source_url": "http://h/20", "plays": 9})

    monkeypatch.setattr(playlists_route, "enrich_track",
                        lambda url: pytest.fail("cache hit must not refetch"))
    saved = []
    monkeypatch.setattr(playlists_route, "upsert_song",
                        lambda title, **kw: saved.append({"title": title, **kw}) or 1)
    monkeypatch.setattr(playlists_route.queue_runner, "enqueue_song",
                        lambda sid: f"job-{sid}")

    body = playlists_route.ingest(playlists_route.IngestRequest(tracks=[
        {"title": "Unknown", "artist": "", "source_url": "http://h/20"},
    ]))
    assert body["partial_count"] == 0
    assert saved[0]["title"] == "Cached"
    assert saved[0]["plays"] == 9


def test_preview_status_endpoint_404s_unknown_session():
    from fastapi import HTTPException
    import api.routes.playlists as playlists_route

    with pytest.raises(HTTPException) as exc:
        playlists_route.preview_status("nope")
    assert exc.value.status_code == 404
