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
    return models, mixes


def test_scrape_link_sets_youtube(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    rows = [{"entry_index": 1, "cue_secs": None, "is_overlay": False, "artist": "A",
             "title": "T", "raw_label": "1. A - T", "is_id": 0, "remixer": None,
             "mashup_parts": [], "parse_confidence": 1.0,
             "tl_track_url": "https://www.1001tracklists.com/track/1/index.html"}]
    detail = mixes._persist_mix("M", "https://src/9", rows, method="scrape")
    tid = detail["tracks"][0]["id"]
    monkeypatch.setattr(mixes, "scrape_track_links",
                        lambda u, **kw: {"soundcloud_url": "", "youtube_url": "https://youtu.be/Q"})
    out = mixes.scrape_track_link(tid)
    assert out["link_url"] == "https://youtu.be/Q"
    assert out["link_platform"] == "youtube"
    assert out["resolve_status"] == "scraped"
    assert models.is_trusted_link("scraped", None, None) is True


def test_scrape_link_no_tl_url_400(tmp_path, monkeypatch):
    models, mixes = _setup(tmp_path, monkeypatch)
    rows = [{"entry_index": 1, "cue_secs": None, "is_overlay": False, "artist": "A",
             "title": "T", "raw_label": "1. A - T", "is_id": 0, "remixer": None,
             "mashup_parts": [], "parse_confidence": 1.0, "tl_track_url": ""}]
    detail = mixes._persist_mix("M", "https://src/8", rows, method="scrape")
    with pytest.raises(HTTPException) as ei:
        mixes.scrape_track_link(detail["tracks"][0]["id"])
    assert ei.value.status_code == 400
