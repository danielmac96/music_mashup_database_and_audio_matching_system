"""GET /api/mixes/tracks/{id}/candidates and the auto-resolve `relink` flag.

Both exist because auto-link can pick the wrong link: the candidate list shows
what it saw so the right one can be clicked, and relink re-searches rows it
already got wrong (which the normal "unlinked only" filter would skip forever).
"""
import importlib
import json

import pytest
from fastapi import BackgroundTasks, HTTPException


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mixes
    importlib.reload(mixes)
    return models, mixes


def _mk(mixes, artist="jeonghyeon", title="On The World", label="12. jeonghyeon - On The World"):
    rows = [{"entry_index": 1, "cue_secs": None, "is_overlay": False, "artist": artist,
             "title": title, "raw_label": label, "is_id": 0, "remixer": None,
             "mashup_parts": [], "parse_confidence": 1.0}]
    return mixes._persist_mix("M", "https://src/c", rows, method="paste")


_HITS = [
    {"url": "https://soundcloud.com/jeonghyeonmusic/on-the-world", "title": "On The World",
     "uploader": "jeonghyeon", "duration_secs": 213.0, "score": 0.97,
     "artist_score": 1.0, "playback_count": 8617},
    {"url": "https://soundcloud.com/youmeatsixofficial/take-on-the-world",
     "title": "Take On The World", "uploader": "youmeatsixofficial",
     "duration_secs": 271.0, "score": 0.55, "artist_score": 0.0,
     "playback_count": 279429},
]


def test_candidates_use_the_cleaned_label_as_the_query(tmp_path, monkeypatch):
    """The search must see "jeonghyeon - On The World", not the "12. " prefix —
    the same cleanup auto-link applies, so the list explains what it saw."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    seen = {}

    def fake(artist, title, query=None, *, limit=8, _get=None):
        seen.update(artist=artist, title=title, query=query, limit=limit)
        return list(_HITS)

    monkeypatch.setattr(mixes, "sc_search_candidates", fake)
    out = mixes.track_candidates(d["tracks"][0]["id"])

    assert seen["query"] == "jeonghyeon - On The World"
    assert out["query"] == "jeonghyeon - On The World"
    assert out["platform"] == "soundcloud"
    assert [c["url"] for c in out["candidates"]] == [h["url"] for h in _HITS]


def test_candidates_respect_the_limit(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    monkeypatch.setattr(mixes, "sc_search_candidates",
                        lambda *a, **k: list(_HITS))
    out = mixes.track_candidates(d["tracks"][0]["id"], limit=1)
    assert len(out["candidates"]) == 1


def test_candidates_reject_unknown_platform(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    with pytest.raises(HTTPException) as exc:
        mixes.track_candidates(d["tracks"][0]["id"], platform="spotify")
    assert exc.value.status_code == 400


def test_candidates_404_for_unknown_track(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    _mk(mixes)
    with pytest.raises(HTTPException) as exc:
        mixes.track_candidates(999999)
    assert exc.value.status_code == 404


def test_candidates_skip_id_entries(tmp_path, monkeypatch):
    """Searching "ID - ID" is noise — return nothing rather than query for it."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, artist="ID", title="ID", label="5. ID - ID")
    monkeypatch.setattr(mixes, "sc_search_candidates",
                        lambda *a, **k: pytest.fail("must not search an ID entry"))
    out = mixes.track_candidates(d["tracks"][0]["id"])
    assert out["candidates"] == []


def test_candidates_surface_soundcloud_failure_as_502(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)

    def boom(*a, **k):
        raise mixes.SoundCloudAPIError("no client_id")

    monkeypatch.setattr(mixes, "sc_search_candidates", boom)
    with pytest.raises(HTTPException) as exc:
        mixes.track_candidates(d["tracks"][0]["id"])
    assert exc.value.status_code == 502


# ── Candidate cache ──────────────────────────────────────────────────────────
#
# Auto-link fetched and scored these hits anyway, so reusing them makes the
# picker instant and free on every row instead of a search per click.

def _cache(models, track_id, hits, platform="soundcloud"):
    conn = models.get_conn()
    conn.execute("UPDATE mix_tracks SET link_url=?, link_platform=?, "
                 "resolve_status='auto', resolve_candidates=? WHERE id=?",
                 (hits[0]["url"], platform, json.dumps(hits), track_id))
    conn.commit()
    conn.close()


def test_cached_candidates_are_served_without_searching(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    tid = d["tracks"][0]["id"]
    _cache(models, tid, _HITS)
    monkeypatch.setattr(mixes, "sc_search_candidates",
                        lambda *a, **k: pytest.fail("must not search when cached"))

    out = mixes.track_candidates(tid)

    assert out["cached"] is True
    assert [c["url"] for c in out["candidates"]] == [h["url"] for h in _HITS]


def test_refresh_bypasses_the_cache(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    tid = d["tracks"][0]["id"]
    _cache(models, tid, _HITS)
    fresh = [{**_HITS[0], "url": "https://soundcloud.com/fresh/hit"}]
    monkeypatch.setattr(mixes, "sc_search_candidates", lambda *a, **k: fresh)

    out = mixes.track_candidates(tid, refresh=True)

    assert out["cached"] is False
    assert out["candidates"][0]["url"] == "https://soundcloud.com/fresh/hit"


def test_cache_from_another_platform_is_not_reused(tmp_path, monkeypatch):
    """A YouTube-linked track's cache holds YouTube hits — asking for SoundCloud
    must search rather than hand back the wrong platform's list."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    tid = d["tracks"][0]["id"]
    _cache(models, tid, _HITS, platform="youtube")
    monkeypatch.setattr(mixes, "sc_search_candidates", lambda *a, **k: list(_HITS))

    out = mixes.track_candidates(tid, platform="soundcloud")
    assert out["cached"] is False


def test_unreadable_cache_falls_back_to_a_live_search(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    tid = d["tracks"][0]["id"]
    conn = models.get_conn()
    conn.execute("UPDATE mix_tracks SET link_url='u://x', link_platform='soundcloud', "
                 "resolve_candidates='not json' WHERE id=?", (tid,))
    conn.commit()
    conn.close()
    monkeypatch.setattr(mixes, "sc_search_candidates", lambda *a, **k: list(_HITS))

    out = mixes.track_candidates(tid)      # must not raise
    assert out["cached"] is False
    assert len(out["candidates"]) == 2


def test_mix_payload_exposes_has_candidates_not_the_blob(tmp_path, monkeypatch):
    """The blob would add ~250 KB to a 200-track mix the list view never renders."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    tid = d["tracks"][0]["id"]
    _cache(models, tid, _HITS)

    conn = models.get_conn()
    detail = mixes._mix_detail(conn, d["id"])
    one = mixes._track_row(conn, tid)
    conn.close()

    for row in (detail["tracks"][0], one):
        assert row["has_candidates"] is True
        assert "resolve_candidates" not in row


def test_has_candidates_is_false_without_a_cache(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    conn = models.get_conn()
    detail = mixes._mix_detail(conn, d["id"])
    conn.close()
    assert detail["tracks"][0]["has_candidates"] is False


# ── relink flag on /auto-resolve ─────────────────────────────────────────────

def _link(models, track_id, status):
    conn = models.get_conn()
    conn.execute("UPDATE mix_tracks SET link_url='u://old', link_platform='soundcloud', "
                 "resolve_status=? WHERE id=?", (status, track_id))
    conn.commit()
    conn.close()


def test_relink_queues_an_already_auto_linked_track(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    _link(models, d["tracks"][0]["id"], "auto")

    out = mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(relink=True),
                                 BackgroundTasks())
    assert out["queued"] == 1
    assert out["relink"] is True


def test_without_relink_a_linked_track_is_rejected(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    _link(models, d["tracks"][0]["id"], "auto")

    with pytest.raises(HTTPException) as exc:
        mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(), BackgroundTasks())
    assert exc.value.status_code == 400


def test_relink_ignores_manual_links(tmp_path, monkeypatch):
    """A link the user pasted or confirmed must never be re-searched away."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    _link(models, d["tracks"][0]["id"], "manual")

    with pytest.raises(HTTPException) as exc:
        mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(relink=True),
                               BackgroundTasks())
    assert exc.value.status_code == 400
