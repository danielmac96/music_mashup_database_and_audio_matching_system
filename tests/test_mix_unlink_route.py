"""POST /api/mixes/{id}/unlink — clear links in bulk.

Exists because a bad auto-link batch was otherwise unfixable except by pasting
over each row one at a time.
"""
import importlib

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


def _mk(mixes, n=3):
    rows = [{"entry_index": i + 1, "cue_secs": None, "is_overlay": False, "artist": f"A{i}",
             "title": f"T{i}", "raw_label": f"{i+1}. A{i} - T{i}", "is_id": 0, "remixer": None,
             "mashup_parts": [], "parse_confidence": 1.0} for i in range(n)]
    return mixes._persist_mix("M", "https://src/u", rows, method="paste")


def _link(models, track_id, status="auto", song_id=None):
    conn = models.get_conn()
    conn.execute(
        "UPDATE mix_tracks SET link_url='u://old', link_platform='soundcloud', "
        "resolve_status=?, resolve_score=0.9, resolve_artist_score=1.0, "
        "resolve_duration_secs=200, resolve_candidates='[{\"url\":\"u://old\"}]', "
        "song_id=? WHERE id=?", (status, song_id, track_id))
    conn.commit()
    conn.close()


def _row(models, track_id):
    conn = models.get_conn()
    r = dict(conn.execute("SELECT * FROM mix_tracks WHERE id=?", (track_id,)).fetchone())
    conn.close()
    return r


def test_unlink_clears_every_resolve_field(tmp_path, monkeypatch):
    """An unlinked row must be indistinguishable from one never resolved —
    a leftover score or cached candidate list would describe a link that's gone."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=1)
    tid = d["tracks"][0]["id"]
    _link(models, tid)

    out = mixes.unlink_tracks(d["id"], mixes.UnlinkRequest(track_ids=[tid]))

    assert out["unlinked"] == 1
    assert out["skipped_ingested"] == 0
    r = _row(models, tid)
    assert r["link_url"] is None
    assert r["link_platform"] is None
    assert r["resolve_status"] == "unresolved"
    assert r["resolve_score"] is None
    assert r["resolve_artist_score"] is None
    assert r["resolve_duration_secs"] is None
    assert r["resolve_candidates"] is None


def test_unlink_honours_track_ids(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=3)
    ids = [t["id"] for t in d["tracks"]]
    for tid in ids:
        _link(models, tid)

    out = mixes.unlink_tracks(d["id"], mixes.UnlinkRequest(track_ids=[ids[1]]))

    assert out["unlinked"] == 1
    assert _row(models, ids[0])["link_url"] == "u://old"
    assert _row(models, ids[1])["link_url"] is None
    assert _row(models, ids[2])["link_url"] == "u://old"


def test_unlink_without_ids_clears_the_whole_mix(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=3)
    for t in d["tracks"]:
        _link(models, t["id"])

    out = mixes.unlink_tracks(d["id"], mixes.UnlinkRequest())

    assert out["unlinked"] == 3
    assert out["resolved_count"] == 0


def test_unlink_skips_ingested_tracks(tmp_path, monkeypatch):
    """Their link is the provenance of a real downloaded file — clearing it
    would orphan the library song."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=2)
    ids = [t["id"] for t in d["tracks"]]
    conn = models.get_conn()
    conn.execute("INSERT INTO songs (title, source_url) VALUES ('S','u://s')")
    song_id = conn.execute("SELECT id FROM songs").fetchone()["id"]
    conn.commit()
    conn.close()
    _link(models, ids[0])
    _link(models, ids[1], status="resolved", song_id=song_id)

    out = mixes.unlink_tracks(d["id"], mixes.UnlinkRequest())

    assert out["unlinked"] == 1
    assert out["skipped_ingested"] == 1
    assert _row(models, ids[1])["link_url"] == "u://old"


def test_unlink_all_ingested_is_rejected(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=1)
    conn = models.get_conn()
    conn.execute("INSERT INTO songs (title, source_url) VALUES ('S','u://s')")
    song_id = conn.execute("SELECT id FROM songs").fetchone()["id"]
    conn.commit()
    conn.close()
    _link(models, d["tracks"][0]["id"], status="resolved", song_id=song_id)

    with pytest.raises(HTTPException) as exc:
        mixes.unlink_tracks(d["id"], mixes.UnlinkRequest())
    assert exc.value.status_code == 400
    assert "library" in exc.value.detail


def test_unlink_nothing_linked_is_rejected(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=1)
    with pytest.raises(HTTPException) as exc:
        mixes.unlink_tracks(d["id"], mixes.UnlinkRequest())
    assert exc.value.status_code == 400


def test_unlink_404s_unknown_mix(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    _mk(mixes, n=1)
    with pytest.raises(HTTPException) as exc:
        mixes.unlink_tracks(999999, mixes.UnlinkRequest())
    assert exc.value.status_code == 404


def test_unlinked_track_is_auto_linkable_again(tmp_path, monkeypatch):
    """The point of unlinking: the row goes back into the auto-link pool."""
    models, mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=1)
    _link(models, d["tracks"][0]["id"])
    mixes.unlink_tracks(d["id"], mixes.UnlinkRequest())

    out = mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(), BackgroundTasks())
    assert out["queued"] == 1
