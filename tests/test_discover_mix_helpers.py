"""Discover + Mixes helpers for the redesign: GET /api/discovery/for-track/{id}
and the `summary` on a mix (detail and list)."""
import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SUMMARY_KEYS = {"tracks", "beds", "overlays", "ids", "linked", "library",
                "needs_review", "unresolved", "ingested", "pairs_documented",
                "pairs_in_library"}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_DATA_DIR", str(tmp_path / "data"))
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
    import api.routes.mixes as mixes
    importlib.reload(mixes)
    import api.server as server
    importlib.reload(server)
    monkeypatch.setattr(pl.queue_runner, "enqueue_song", lambda sid, **_kw: f"job-{sid}")
    return TestClient(server.app), disc, models


def track(tid, title, **kw):
    """A canonical browse row, as soundcloud_browse.track_row emits it."""
    return {"title": title, "artist": "A", "artist_id": "900", "track_id": str(tid),
            "duration_secs": 200.0, "duration_str": "3:20",
            "source_url": f"https://soundcloud.com/a/{tid}",
            "upload_date": "20240115", "likes": 1, "reposts": 0, "comments": 0,
            "plays": 10, "thumbnail": "", "genre": "House", "tags": "",
            "release_year": 2024, "is_snip": False, **kw}


def _seed(models, title="Midnight Signal (Extended Mix)", artist="Lumen Park ft. Cass",
          genre="House", bpm=126.2):
    sid = models.upsert_song(title, artist, "https://soundcloud.com/x/seed", 200, genre,
                             status="analysed")
    if bpm:
        conn = models.get_conn()
        conn.execute("INSERT INTO features (song_id, stem_type, bpm, camelot) "
                     "VALUES (?,?,?,?)", (sid, "full", bpm, "8A"))
        conn.commit()
        conn.close()
    return sid


# ── for-track ────────────────────────────────────────────────────────────────

def test_for_track_runs_the_three_searches_and_merges(app, monkeypatch):
    client, disc, models = app
    sid = _seed(models)
    owned = models.upsert_song("Owned", "A", "https://soundcloud.com/a/7", 200,
                               status="analysed")
    pages = {
        "Lumen Park Midnight Signal acapella": [track(1, "MS Acapella 126 BPM 8A"),
                                                track(2, "Dupe")],
        "Lumen Park Midnight Signal instrumental": [track(3, "MS Instrumental"),
                                                    track(2, "Dupe"),
                                                    track(7, "Owned")],
        "House 126 bpm": [track(4, "Other", is_snip=True),
                          track(0, "Seed itself",
                                source_url="https://soundcloud.com/x/seed")],
    }
    calls = []

    def fake(kind, q, **kw):
        calls.append((kind, q, kw.get("limit")))
        return {"items": [dict(r) for r in pages[q]], "next_cursor": None}
    monkeypatch.setattr(disc.browse, "search", fake)

    r = client.get(f"/api/discovery/for-track/{sid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"rows", "queries"}
    # The record as it is searched everywhere: no featured artist, no format tag.
    assert body["queries"] == list(pages)
    assert [c[:2] for c in calls] == [("tracks", q) for q in pages]
    # Round-robin across the searches, deduplicated by URL, owned tracks gone.
    assert [row["track_id"] for row in body["rows"]] == ["1", "3", "4", "2"]
    assert owned and all(row["in_library"] is None for row in body["rows"])
    for row in body["rows"]:
        # /search rows, annotated the same way.
        assert set(track(0, "x")) <= set(row)
        assert {"fit_hint", "in_library", "is_snip"} <= set(row)
        assert "fits" in row["fit_hint"]
    by_id = {row["track_id"]: row for row in body["rows"]}
    assert by_id["4"]["is_snip"] is True
    assert by_id["1"]["fit_hint"]["fits"] == 1      # prints 126 BPM 8A: sits with the seed


def test_search_rows_and_for_track_rows_share_a_key_set(app, monkeypatch):
    client, disc, models = app
    sid = _seed(models)
    monkeypatch.setattr(disc.browse, "search",
                        lambda kind, q, **kw: {"items": [track(1, "One")],
                                               "next_cursor": None})
    searched = client.get("/api/discovery/search", params={"q": "x"}).json()["items"][0]
    found = client.get(f"/api/discovery/for-track/{sid}").json()["rows"][0]
    assert set(found) == set(searched)


@pytest.mark.parametrize("role,want", [
    ("acapella", ["Lumen Park Midnight Signal acapella", "House acapella 126 bpm"]),
    ("instrumental", ["Lumen Park Midnight Signal instrumental",
                      "House instrumental 126 bpm"]),
])
def test_a_role_asks_for_that_cut(app, monkeypatch, role, want):
    client, disc, models = app
    sid = _seed(models)
    monkeypatch.setattr(disc.browse, "search",
                        lambda kind, q, **kw: {"items": [], "next_cursor": None})
    body = client.get(f"/api/discovery/for-track/{sid}", params={"role": role}).json()
    assert body == {"rows": [], "queries": want}


def test_limit_and_validation(app, monkeypatch):
    client, disc, models = app
    sid = _seed(models, genre="", bpm=None)
    monkeypatch.setattr(
        disc.browse, "search",
        lambda kind, q, **kw: {"items": [track(f"{q[-3:]}{n}", "T") for n in range(10)],
                               "next_cursor": None})
    body = client.get(f"/api/discovery/for-track/{sid}", params={"limit": 5}).json()
    assert len(body["rows"]) == 5
    assert len(body["queries"]) == 2            # no genre, no tempo: no third search
    assert client.get("/api/discovery/for-track/99999").status_code == 404
    assert client.get(f"/api/discovery/for-track/{sid}",
                      params={"role": "drums"}).status_code == 422


def test_one_failed_search_does_not_cost_the_others(app, monkeypatch):
    client, disc, models = app
    sid = _seed(models)

    def flaky(kind, q, **kw):
        if "acapella" in q:
            raise disc.SoundCloudAPIError("boom")
        return {"items": [track(len(q), q)], "next_cursor": None}
    monkeypatch.setattr(disc.browse, "search", flaky)
    r = client.get(f"/api/discovery/for-track/{sid}")
    assert r.status_code == 200 and len(r.json()["rows"]) == 2

    def dead(kind, q, **kw):
        raise disc.SoundCloudAPIError("boom")
    monkeypatch.setattr(disc.browse, "search", dead)
    assert client.get(f"/api/discovery/for-track/{sid}").status_code == 502

    seen = []

    def backing_off(kind, q, **kw):
        seen.append(q)
        raise disc.SoundCloudUnavailable("backing off")
    monkeypatch.setattr(disc.browse, "search", backing_off)
    assert client.get(f"/api/discovery/for-track/{sid}").status_code == 503
    assert len(seen) == 1                       # an open breaker is not hammered


def test_for_track_uses_no_client_of_its_own():
    """Through the browse layer only — its throttle, cache and breaker."""
    src = (ROOT / "api/routes/discovery.py").read_text(encoding="utf-8")
    body = src[src.index("def for_track_queries"):src.index('@router.get("/gaps")')]
    assert "browse.search" in body
    for banned in ("api-v2", "client_id", "requests.", "urllib", "soundcloud_api"):
        assert banned not in body


# ── mix summary ──────────────────────────────────────────────────────────────

SET = """1. Lumen Park - Midnight Signal
w/ Cass Rowan - Fireproof (Acapella)
2. Two Rooms - Bassline Theory
w/ Nobody - Not Owned
3. Far Away - Unknown Record
4. ID - ID
"""


def _mix(client, models):
    for title, artist, slug in (("Midnight Signal", "Lumen Park", "ms"),
                                ("Fireproof", "Cass Rowan", "fp"),
                                ("Bassline Theory", "Two Rooms", "bt")):
        models.upsert_song(title, artist, f"https://soundcloud.com/x/{slug}", 200,
                           status="analysed")
    r = client.post("/api/mixes/import-paste", json={
        "content": SET, "url": "https://www.1001tracklists.com/tracklist/t/set.html"})
    assert r.status_code == 200, r.text
    return r.json()


def test_mix_detail_and_list_carry_the_same_summary(app):
    client, _disc, models = app
    mix = _mix(client, models)
    detail = client.get(f"/api/mixes/{mix['id']}").json()
    summary = detail["summary"]
    assert set(summary) == SUMMARY_KEYS
    assert all(isinstance(v, int) for v in summary.values())
    # A row matched to a library song takes that song's link, so it is linked.
    assert summary == {"tracks": 6, "beds": 4, "overlays": 2, "ids": 1, "linked": 3,
                       "library": 3, "needs_review": 0, "unresolved": 2,
                       "ingested": 3, "pairs_documented": 2, "pairs_in_library": 1}
    listed = client.get("/api/mixes").json()["mixes"]
    assert listed[0]["summary"] == summary
    # The list's older counts still read the same numbers.
    assert listed[0]["track_count"] == 6 and listed[0]["resolved_count"] == 3


def test_needs_review_is_the_trust_gate(app):
    client, _disc, models = app
    mix = _mix(client, models)
    by_title = {t["title"]: t for t in client.get(f"/api/mixes/{mix['id']}").json()["tracks"]}
    weak, strong = by_title["Unknown Record"]["id"], by_title["Not Owned"]["id"]
    conn = models.get_conn()
    # An auto link under the score floor, and one that clears every check.
    conn.execute("UPDATE mix_tracks SET link_url='https://soundcloud.com/a/weak', "
                 "resolve_status='auto', resolve_score=0.5, "
                 "resolve_duration_secs=200, resolve_artist_score=0.9 WHERE id=?", (weak,))
    conn.execute("UPDATE mix_tracks SET link_url='https://soundcloud.com/a/ok', "
                 "resolve_status='auto', resolve_score=0.95, "
                 "resolve_duration_secs=200, resolve_artist_score=0.9 WHERE id=?", (strong,))
    conn.commit()
    conn.close()
    detail = client.get(f"/api/mixes/{mix['id']}").json()
    s = detail["summary"]
    assert s["linked"] == 5 and s["needs_review"] == 1 and s["unresolved"] == 0
    flagged = [t for t in detail["tracks"] if t["resolved_url"] and not t["trusted"]]
    assert len(flagged) == s["needs_review"]        # the same gate the rows show
    assert client.get("/api/mixes").json()["mixes"][0]["summary"] == s


def test_the_list_is_three_queries_however_many_mixes(app, monkeypatch):
    client, _disc, models = app
    _mix(client, models)
    client.post("/api/mixes/import-paste", json={
        "content": "1. A - B\n2. C - D\n",
        "url": "https://www.1001tracklists.com/tracklist/t/other.html"})
    import api.routes.mixes as mixes
    seen = []
    real = mixes.get_conn

    def traced(*a, **kw):
        conn = real(*a, **kw)
        conn.set_trace_callback(seen.append)
        return conn
    monkeypatch.setattr(mixes, "get_conn", traced)
    body = client.get("/api/mixes").json()
    assert body["count"] == 2
    assert len([q for q in seen if q.lstrip().upper().startswith("SELECT")]) == 3
    assert [m["summary"]["tracks"] for m in body["mixes"]] == [2, 6]


def test_an_empty_mix_list(app):
    client, _disc, _models = app
    assert client.get("/api/mixes").json() == {"count": 0, "mixes": []}
