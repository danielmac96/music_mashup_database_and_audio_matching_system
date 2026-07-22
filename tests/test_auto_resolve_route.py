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
    return mixes


def _mk(mixes, n=1):
    rows = [{"entry_index": i + 1, "cue_secs": None, "is_overlay": False, "artist": f"A{i}",
             "title": f"T{i}", "raw_label": f"{i+1}. A{i} - T{i}", "is_id": 0, "remixer": None,
             "mashup_parts": [], "parse_confidence": 1.0} for i in range(n)]
    return mixes._persist_mix("M", "https://src/ar", rows, method="paste")


def test_both_platform_is_accepted(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    out = mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(platform="both"),
                                 BackgroundTasks())
    assert out["platform"] == "both"
    assert out["queued"] == 1


def test_default_platform_is_both(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    out = mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(), BackgroundTasks())
    assert out["platform"] == "both"


def test_track_ids_limits_queued_count(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes, n=5)
    picked = [d["tracks"][0]["id"], d["tracks"][2]["id"]]
    out = mixes.auto_resolve_mix(
        d["id"], mixes.AutoResolveRequest(platform="both", track_ids=picked),
        BackgroundTasks())
    assert out["queued"] == 2   # only the selected subset, not all 5


def test_invalid_platform_rejected(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    with pytest.raises(HTTPException) as ei:
        mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(platform="spotify"),
                               BackgroundTasks())
    assert ei.value.status_code == 400
