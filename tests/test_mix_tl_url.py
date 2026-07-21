import importlib


def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    return models


def test_tl_track_url_persisted(tmp_path, monkeypatch):
    models = _fresh_db(tmp_path, monkeypatch)
    from api.routes import mixes
    importlib.reload(mixes)
    rows = [{"entry_index": 1, "cue_secs": None, "is_overlay": False,
             "artist": "A", "title": "T", "raw_label": "1. A - T", "is_id": 0,
             "remixer": None, "mashup_parts": [], "parse_confidence": 1.0,
             "tl_track_url": "https://www.1001tracklists.com/track/x/index.html"}]
    detail = mixes._persist_mix("Mix", "https://src/1", rows, method="scrape")
    conn = models.get_conn()
    got = conn.execute("SELECT tl_track_url FROM mix_tracks WHERE mix_id=?",
                       (detail["id"],)).fetchone()[0]
    conn.close()
    assert got.endswith("/index.html")
