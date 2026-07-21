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
    return mixes._persist_mix("M", "https://src/r", rows, method="paste")


def test_reorder_rewrites_positions(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    ids = [t["id"] for t in d["tracks"]]
    out = mixes.reorder_tracks(d["id"], mixes.ReorderRequest(track_ids=list(reversed(ids))))
    assert [t["id"] for t in out["tracks"]] == list(reversed(ids))
    assert [t["idx"] for t in out["tracks"]] == [0, 1, 2]


def test_reorder_incomplete_set_400(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    with pytest.raises(HTTPException) as ei:
        mixes.reorder_tracks(d["id"], mixes.ReorderRequest(track_ids=[d["tracks"][0]["id"]]))
    assert ei.value.status_code == 400
