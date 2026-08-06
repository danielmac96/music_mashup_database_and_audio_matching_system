"""Tests for manual mix-track entry: add (with/without link, bad URL, no title),
remove (with match-pair cleanup), and the ingested-row delete guard. Route
functions are called directly against an isolated tmp DB, mirroring
test_mix_reorder.py.
"""
import importlib

import pytest
from fastapi import HTTPException


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
    return mixes


def _mk(mixes):
    rows = [{"entry_index": i + 1, "cue_secs": None, "is_overlay": False,
             "artist": f"A{i}", "title": f"T{i}", "raw_label": f"{i+1}. A{i} - T{i}",
             "is_id": 0, "remixer": None, "mashup_parts": [], "parse_confidence": 1.0}
            for i in range(3)]
    return mixes._persist_mix("M", "https://src/m", rows, method="paste")


def test_add_track_with_link(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    t = mixes.add_track(d["id"], mixes.AddTrackRequest(
        artist="Kanye West", title="Stronger",
        link="https://soundcloud.com/kanyewest/stronger"))
    assert t["title"] == "Stronger"
    assert t["resolved_url"] == "https://soundcloud.com/kanyewest/stronger"
    assert t["link_platform"] == "soundcloud"
    assert t["resolve_status"] == "manual"
    assert t["trusted"] is True
    # appended after the 3 seeded tracks
    assert t["idx"] == 3
    assert mixes.get_mix(d["id"])["track_count"] == 4


def test_add_track_without_link_is_unresolved(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    t = mixes.add_track(d["id"], mixes.AddTrackRequest(artist="Avicii", title="Levels"))
    assert t["resolve_status"] == "unresolved"
    assert not t["resolved_url"]
    assert t["trusted"] is False
    assert t["raw_label"] == "Avicii - Levels"


def test_add_track_rejects_non_sc_yt_link(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    with pytest.raises(HTTPException) as ei:
        mixes.add_track(d["id"], mixes.AddTrackRequest(
            title="X", link="https://example.com/not-a-track"))
    assert ei.value.status_code == 400


def test_add_track_requires_title(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    with pytest.raises(HTTPException) as ei:
        mixes.add_track(d["id"], mixes.AddTrackRequest(artist="A", title="   "))
    assert ei.value.status_code == 400


def test_add_track_missing_mix_404(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as ei:
        mixes.add_track(999, mixes.AddTrackRequest(title="X"))
    assert ei.value.status_code == 404


def test_delete_track_removes_match_pairs(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    ids = [t["id"] for t in d["tracks"]]
    inst_id, vocal_id = ids[0], ids[1]
    # Record a manual instrumental↔vocal pair, then delete the vocal track.
    mixes.save_assignments(d["id"], mixes.AssignmentsRequest(
        matches=[mixes.MatchAssignment(vocal_track_id=vocal_id, inst_track_id=inst_id)]))
    assert mixes.get_mix(d["id"])["match_count"] == 1

    out = mixes.delete_track(d["id"], vocal_id)
    assert out["match_count"] == 0
    assert vocal_id not in [t["id"] for t in out["tracks"]]
    assert out["track_count"] == 2


def test_delete_ingested_track_blocked(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    tid = d["tracks"][0]["id"]
    from database.models import get_conn
    conn = get_conn()
    conn.execute("UPDATE mix_tracks SET song_id=42 WHERE id=?", (tid,))
    conn.commit()
    conn.close()
    with pytest.raises(HTTPException) as ei:
        mixes.delete_track(d["id"], tid)
    assert ei.value.status_code == 409


def test_delete_track_missing_404(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    with pytest.raises(HTTPException) as ei:
        mixes.delete_track(d["id"], 123456)
    assert ei.value.status_code == 404


def _mk_with_overlay(mixes):
    # A bed followed by a 'w/' overlay → one documented (parsed) pair on import.
    rows = [
        {"entry_index": 1, "cue_secs": None, "is_overlay": False, "artist": "Bed",
         "title": "Beat", "raw_label": "1. Bed - Beat", "is_id": 0, "remixer": None,
         "mashup_parts": [], "parse_confidence": 1.0},
        {"entry_index": None, "cue_secs": None, "is_overlay": True, "artist": "Voc",
         "title": "Acapella", "raw_label": "w/ Voc - Acapella", "is_id": 0,
         "remixer": None, "mashup_parts": [], "parse_confidence": 1.0},
    ]
    return mixes._persist_mix("MO", "https://src/mo", rows, method="paste")


def test_reset_restores_original_grouping(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk_with_overlay(mixes)
    # Import seeds one parsed pair (bed ↔ overlay) and roles.
    assert d["match_count"] == 1
    bed_id = next(t["id"] for t in d["tracks"] if not t["is_overlay"])
    voc_id = next(t["id"] for t in d["tracks"] if t["is_overlay"])
    assert next(t["role"] for t in d["tracks"] if t["id"] == bed_id) == "instrumental"

    # User re-homes the vocal off the bed (manual edit) and drops the parsed pair.
    mixes.save_assignments(d["id"], mixes.AssignmentsRequest(
        matches=[mixes.MatchAssignment(vocal_track_id=voc_id, inst_track_id=None)]))
    assert mixes.get_mix(d["id"])["match_count"] == 0

    # Reset rebuilds the original parsed pair + roles.
    out = mixes.reset_matches(d["id"])
    assert out["match_count"] == 1
    p = out["pairs"][0]
    assert p["inst_mix_track_id"] == bed_id and p["vocal_mix_track_id"] == voc_id
    assert p["origin"] == "parsed"


def test_reset_missing_mix_404(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as ei:
        mixes.reset_matches(999)
    assert ei.value.status_code == 404
