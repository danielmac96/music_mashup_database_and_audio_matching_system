"""The `summary` on a mix (detail and list)."""
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


# ── for-track ────────────────────────────────────────────────────────────────

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
