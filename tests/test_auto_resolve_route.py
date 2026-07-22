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


def _mk(mixes):
    rows = [{"entry_index": 1, "cue_secs": None, "is_overlay": False, "artist": "A",
             "title": "T", "raw_label": "1. A - T", "is_id": 0, "remixer": None,
             "mashup_parts": [], "parse_confidence": 1.0}]
    return mixes._persist_mix("M", "https://src/ar", rows, method="paste")


def test_both_platform_is_accepted(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    out = mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(platform="both"),
                                 BackgroundTasks())
    assert out["platform"] == "both"
    assert out["queued"] == 1


def test_invalid_platform_rejected(tmp_path, monkeypatch):
    mixes = _setup(tmp_path, monkeypatch)
    d = _mk(mixes)
    with pytest.raises(HTTPException) as ei:
        mixes.auto_resolve_mix(d["id"], mixes.AutoResolveRequest(platform="spotify"),
                               BackgroundTasks())
    assert ei.value.status_code == 400
