import json

import pytest

from ingest import soundcloud_api as sc


@pytest.fixture(autouse=True)
def _reset_cache():
    sc._client_id = None
    yield
    sc._client_id = None


def _fake_get(client_id="CID123456789012345678901", collection=None):
    """Build an injectable _get that serves a homepage, a JS bundle with the
    client_id, and the search JSON."""
    homepage = '<script src="https://a-v2.sndcdn.com/assets/1-abc.js"></script>'
    bundle = f'window.x={{client_id:"{client_id}"}};'
    payload = json.dumps({"collection": collection or []})

    def _get(url):
        if url == "https://soundcloud.com/":
            return homepage
        if url.endswith(".js"):
            return bundle
        if url.startswith(sc._SEARCH_URL):
            assert f"client_id={client_id}" in url
            return payload
        raise AssertionError(f"unexpected url {url}")
    return _get


def test_scrape_client_id():
    cid = sc.get_client_id(_get=_fake_get())
    assert cid == "CID123456789012345678901"


def test_find_track_ranks_official_over_reupload():
    collection = [
        {"title": "Welcome To The Jungle - Guns N' Roses (slowed)", "permalink_url": "https://sc/reup",
         "duration": 336000, "playback_count": 243510, "user": {"username": "user-474"}},
        {"title": "Welcome To The Jungle", "permalink_url": "https://sc/official",
         "duration": 274000, "playback_count": 7983932, "user": {"username": "Guns N' Roses"}},
    ]
    out = sc.find_track("Guns N' Roses", "Welcome To The Jungle",
                        _get=_fake_get(collection=collection))
    assert out["url"] == "https://sc/official"
    assert out["duration_secs"] == 274.0
    assert out["playback_count"] == 7983932
    assert out["score"] > 0.7


def test_find_track_playcount_breaks_score_tie():
    # Identical titles/artist -> identical score; higher play count must win.
    collection = [
        {"title": "Desire", "permalink_url": "https://sc/low", "duration": 200000,
         "playback_count": 10, "user": {"username": "Years & Years"}},
        {"title": "Desire", "permalink_url": "https://sc/high", "duration": 200000,
         "playback_count": 900000, "user": {"username": "Years & Years"}},
    ]
    out = sc.find_track("Years & Years", "Desire", _get=_fake_get(collection=collection))
    assert out["url"] == "https://sc/high"


def test_find_track_no_hits_returns_none():
    out = sc.find_track("Nobody", "Nothing", _get=_fake_get(collection=[]))
    assert out is None


def test_find_track_uses_explicit_query():
    seen = {}
    payload = json.dumps({"collection": []})

    def _get(url):
        if url == "https://soundcloud.com/":
            return '<script src="https://x/a.js"></script>'
        if url.endswith(".js"):
            return 'client_id:"CID123456789012345678901"'
        seen["q"] = url
        return payload

    sc.find_track("A", "B", query="custom search string", _get=_get)
    assert "custom+search+string" in seen["q"]
