"""The SoundCloud browse layer behind the Discovery tab.

Offline: every test either injects a `_get` callable or replays a recorded
fixture, the two patterns the repo already uses for the SoundCloud layer. No
network, and no real sleeps — `_get` bypasses throttle, backoff and cache by
design, so the only tests that touch timing monkeypatch `time.sleep`.
"""
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ingest import soundcloud_browse as br          # noqa: E402
from ingest import soundcloud_api as sc_api         # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "sc_browse"


def fixture(name: str) -> str:
    return (FIX / f"{name}.json").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_state():
    """Both modules keep process-level caches; a leaked one makes a later test lie."""
    sc_api._client_id = "test-client-id"
    br.reset_state()
    yield
    sc_api._client_id = None
    br.reset_state()


def serve(mapping, calls=None):
    """A `_get` that answers by substring match on the requested URL."""
    def _get(url):
        if calls is not None:
            calls.append(url)
        for needle, body in mapping.items():
            if needle in url:
                return body
        raise AssertionError(f"unexpected request: {url}")
    return _get


# ── the ingest contract ──────────────────────────────────────────────────────

def test_track_row_key_set_matches_the_ingest_contract():
    """The whole reason browse results need no adaptation to be ingestable.

    If ingest.soundcloud._normalise ever grows or loses a field, this fails and
    tells you to update track_row rather than letting rows silently drop data
    somewhere inside POST /api/playlists/ingest."""
    from ingest.soundcloud import _normalise

    canonical = set(_normalise({
        "title": "T", "uploader": "A", "uploader_id": "1", "id": "2",
        "duration": 100, "webpage_url": "https://soundcloud.com/a/t",
    }).keys())

    row = br.track_row(json.loads(fixture("resolve_track")))
    assert canonical <= set(row), f"missing from track_row: {canonical - set(row)}"


def test_track_row_field_mapping():
    row = br.track_row(json.loads(fixture("resolve_track")))
    assert row["title"] == "Night Drive"
    assert row["artist"] == "artistone"
    assert row["track_id"] == "1001"
    assert row["artist_id"] == "900"
    assert row["duration_secs"] == pytest.approx(210.0)   # 210000 ms
    assert row["duration_str"] == "3:30"
    assert row["source_url"] == "https://soundcloud.com/artistone/night-drive"
    assert row["upload_date"] == "20240115"
    assert row["release_year"] == 2024
    assert row["plays"] == 5000 and row["likes"] == 120
    assert row["genre"] == "House"


def test_tag_list_parses_quoted_multiword_tags():
    """tag_list is space-separated with quoted multi-word tags — a naive split
    would turn "Deep House" into two bogus tags."""
    row = br.track_row(json.loads(fixture("resolve_track")))
    assert json.loads(row["tags"]) == ["Dance", "Deep House", "Remix"]


def test_artwork_is_upgraded_to_500px():
    row = br.track_row(json.loads(fixture("resolve_track")))
    assert row["thumbnail"].endswith("-t500x500.jpg")


def test_artwork_falls_back_to_uploader_avatar():
    hit = json.loads(fixture("resolve_track"))
    hit["artwork_url"] = None
    assert "avatars" in br.track_row(hit)["thumbnail"]


def test_go_plus_snippet_is_flagged_and_uses_full_duration():
    """A SNIP's `duration` is the ~30s preview; full_duration is the real track.

    Taking `duration` would make Discovery show 0:30 and let the user ingest a
    track the pipeline later has to reverify and re-download."""
    page = br.search("tracks", "x", _get=serve({"/search/tracks": fixture("search_tracks")}))
    snip = next(r for r in page["items"] if r["track_id"] == "2002")
    assert snip["is_snip"] is True
    assert snip["duration_secs"] == pytest.approx(245.0)

    normal = next(r for r in page["items"] if r["track_id"] == "1001")
    assert normal["is_snip"] is False


def test_rows_without_a_permalink_are_dropped():
    """Stubs and removed uploads have nothing to show or ingest."""
    page = br.search("tracks", "x", _get=serve({"/search/tracks": fixture("search_tracks")}))
    assert "3003" not in {r["track_id"] for r in page["items"]}
    assert len(page["items"]) == 2


# ── pagination ───────────────────────────────────────────────────────────────

def test_pagination_follows_next_href():
    calls = []
    get = serve({"offset=2": fixture("user_tracks_p2"),
                 "/users/900/tracks": fixture("user_tracks_p1")}, calls)

    p1 = br.user_tracks("900", _get=get)
    assert len(p1["items"]) == 2
    assert p1["next_cursor"] and "offset=2" in p1["next_cursor"]
    assert "linked_partitioning=1" in calls[0]

    p2 = br.user_tracks("900", cursor=p1["next_cursor"], _get=get)
    assert len(p2["items"]) == 1
    assert p2["next_cursor"] is None


def test_cursor_gets_a_fresh_client_id():
    """SoundCloud sometimes omits client_id from next_href; a cursor that lost it
    would 401 on the second page only, which is a miserable bug to chase."""
    calls = []
    get = serve({"offset=2": fixture("user_tracks_p2"),
                 "/users/900/tracks": fixture("user_tracks_p1")}, calls)
    cursor = "https://api-v2.soundcloud.com/users/900/tracks?offset=2&limit=2"
    br.user_tracks("900", cursor=cursor, _get=get)
    assert "client_id=test-client-id" in calls[-1]


@pytest.mark.parametrize("bad", [
    "https://evil.example/users/900/tracks",
    "http://api-v2.soundcloud.com/users/900/tracks",     # not https
    "https://api-v2.soundcloud.com.evil.example/x",
    "file:///etc/passwd",
])
def test_cursor_must_point_at_soundcloud(bad):
    """The cursor round-trips through our API to the browser and back, so it is
    attacker-influenced input aimed at a URL we fetch server-side."""
    with pytest.raises(br.SoundCloudAPIError):
        br.user_tracks("900", cursor=bad, _get=serve({"": "{}"}))


# ── resolve / playlists / users ──────────────────────────────────────────────

def test_resolve_identifies_each_kind():
    for name, kind in (("resolve_track", "track"),
                       ("resolve_playlist", "playlist"),
                       ("resolve_user", "user")):
        out = br.resolve("https://soundcloud.com/x", _get=serve({"/resolve": fixture(name)}))
        assert out["kind"] == kind, name


def test_resolve_user_row():
    out = br.resolve("https://soundcloud.com/artistone",
                     _get=serve({"/resolve": fixture("resolve_user")}))
    assert out["item"]["username"] == "artistone"
    assert out["item"]["followers"] == 4200
    assert out["item"]["verified"] is True


def test_playlist_hydrates_track_stubs_in_order():
    """/playlists/{id} returns full objects for the first few tracks and bare
    {id} stubs after. Without hydration a 60-track set shows five tracks."""
    get = serve({"/tracks?ids": fixture("playlist_stub_hydration"),
                 "/playlists/7001": fixture("resolve_playlist")})
    out = br.playlist("7001", _get=get)

    assert out["playlist"]["title"] == "Summer Crate"
    assert out["playlist"]["track_count"] == 4
    assert [i["track_id"] for i in out["items"]] == ["1001", "1002", "1003", "1004"]
    assert all(i["source_url"] for i in out["items"])


def test_playlist_without_hydration_returns_only_full_rows():
    out = br.playlist("7001", hydrate=False,
                      _get=serve({"/playlists/7001": fixture("resolve_playlist")}))
    assert [i["track_id"] for i in out["items"]] == ["1001", "1002"]


def test_user_likes_unwraps_the_like_envelope():
    """Like entries wrap the track; non-track likes are dropped rather than
    normalised into a track row with empty everything."""
    out = br.user_likes("900", _get=serve({"/users/900/likes": fixture("user_likes")}))
    assert len(out["items"]) == 1
    assert out["items"][0]["title"] == "Night Drive"


def test_search_playlists_and_users():
    pl = br.search("playlists", "crate",
                   _get=serve({"/search/playlists": fixture("search_playlists")}))
    assert pl["items"][0]["kind"] == "playlist"
    assert pl["items"][0]["track_count"] == 4

    us = br.search("users", "artist",
                   _get=serve({"/search/users": fixture("search_users")}))
    assert us["items"][0]["kind"] == "user"
    assert us["items"][0]["username"] == "artistone"


def test_related_tracks():
    out = br.related("1001", _get=serve({"/tracks/1001/related": fixture("related")}))
    assert out["items"][0]["title"] == "Similar Vibe"


def test_search_rejects_unknown_kind():
    with pytest.raises(ValueError):
        br.search("albums", "x", _get=serve({"": "{}"}))


def test_empty_query_makes_no_request():
    def _boom(url):
        raise AssertionError("should not have fetched")
    assert br.search("tracks", "   ", _get=_boom) == {"items": [], "next_cursor": None}


# ── transport behaviour ──────────────────────────────────────────────────────

def test_injected_get_bypasses_cache(monkeypatch):
    """One rule keeps the tests honest: with `_get`, nothing is cached or slept."""
    monkeypatch.setattr(br.time, "sleep", lambda s: pytest.fail("slept during a test"))
    calls = []
    get = serve({"/search/tracks": fixture("search_tracks")}, calls)
    br.search("tracks", "x", _get=get)
    br.search("tracks", "x", _get=get)
    assert len(calls) == 2


def test_cache_serves_the_second_identical_request(monkeypatch):
    calls = []

    def fake_fetch(url, *, _get=None):
        calls.append(url)
        return fixture("search_tracks")

    monkeypatch.setattr(br, "_fetch", fake_fetch)
    br.search("tracks", "night")
    br.search("tracks", "night")
    assert len(calls) == 1


def test_cache_key_ignores_client_id(monkeypatch):
    """A client_id refresh must not invalidate everything already fetched."""
    a = br._cache_key("https://api-v2.soundcloud.com/x?q=1&client_id=OLD")
    b = br._cache_key("https://api-v2.soundcloud.com/x?client_id=NEW&q=1")
    assert a == b


def test_stale_client_id_triggers_one_rescrape(monkeypatch):
    """A 401 means the scraped id expired — re-scrape once, then give up."""
    seen = {"n": 0}

    def _get(url):
        if "soundcloud.com/" in url and "api-v2" not in url:
            return '<script src="https://a.sndcdn.com/b.js"></script>'
        if url.endswith(".js"):
            return 'client_id:"freshclientid0123456789"'
        seen["n"] += 1
        if seen["n"] == 1:
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        return fixture("search_tracks")

    sc_api._client_id = "stale"
    page = br.search("tracks", "x", _get=_get)
    assert len(page["items"]) == 2
    assert seen["n"] == 2


def test_429_backs_off_and_honours_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr(br.time, "sleep", slept.append)
    monkeypatch.setattr(br, "_throttle", lambda: None)

    attempts = {"n": 0}

    def _http_get(url):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many", {"Retry-After": "2"}, None)
        return fixture("search_tracks")

    monkeypatch.setattr(br, "_http_get", _http_get)
    page = br.search("tracks", "x")
    assert len(page["items"]) == 2
    assert slept == [2.0, 2.0]


def test_repeated_429_eventually_raises(monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda s: None)
    monkeypatch.setattr(br, "_throttle", lambda: None)
    monkeypatch.setattr(br, "_http_get", lambda url: (_ for _ in ()).throw(
        urllib.error.HTTPError(url, 429, "Too Many", {}, None)))
    with pytest.raises(br.SoundCloudAPIError):
        br.search("tracks", "x")


def test_circuit_breaker_opens_after_repeated_failures(monkeypatch):
    """A dead client_id would otherwise mean one failed request per interaction,
    forever — and both layers share that id, so it would take the frozen mixes
    resolver down with it."""
    monkeypatch.setattr(br.time, "sleep", lambda s: None)
    monkeypatch.setattr(br, "_throttle", lambda: None)
    monkeypatch.setattr(br, "_http_get", lambda url: (_ for _ in ()).throw(OSError("down")))

    for _ in range(2):
        with pytest.raises(br.SoundCloudAPIError):
            br.search("tracks", "x")

    # Breaker is open now: the next call must fail without touching the transport.
    monkeypatch.setattr(br, "_http_get", lambda url: pytest.fail("breaker did not hold"))
    with pytest.raises(br.SoundCloudUnavailable):
        br.search("tracks", "x")


def test_reset_state_closes_the_breaker(monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda s: None)
    monkeypatch.setattr(br, "_throttle", lambda: None)
    monkeypatch.setattr(br, "_http_get", lambda url: (_ for _ in ()).throw(OSError("down")))
    for _ in range(2):
        with pytest.raises(br.SoundCloudAPIError):
            br.search("tracks", "x")

    br.reset_state()
    monkeypatch.setattr(br, "_http_get", lambda url: fixture("search_tracks"))
    assert len(br.search("tracks", "x")["items"]) == 2


def test_unparseable_json_is_a_soundcloud_error():
    with pytest.raises(br.SoundCloudAPIError):
        br.search("tracks", "x", _get=serve({"/search/tracks": "<html>nope</html>"}))


def test_unavailable_is_a_soundcloud_error_subclass():
    """Routes answer 503 for one and 502 for the other, but nothing that catches
    SoundCloudAPIError should be able to miss the cooling-down case."""
    assert issubclass(br.SoundCloudUnavailable, br.SoundCloudAPIError)


# ── the frozen path ──────────────────────────────────────────────────────────

def test_frozen_module_does_not_depend_on_browse():
    """soundcloud_api feeds the mixes auto-resolver, which EXECUTION_PLAN §0.1
    freezes. The dependency runs browse -> api and must never reverse, or the
    throttle and breaker here would start governing that path's timing."""
    src = (ROOT / "ingest" / "soundcloud_api.py").read_text(encoding="utf-8")
    assert "soundcloud_browse" not in src
