import importlib


def test_import_uses_firecrawl_for_1001(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-k")
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mixes
    importlib.reload(mixes)

    def fake_scrape(url, api_key=..., **kw):
        return [
            {"position": "01", "artist": "A", "title": "Bed", "is_overlay": False,
             "tl_track_url": "https://www.1001tracklists.com/track/1/index.html"},
            {"position": "w/", "artist": "B", "title": "Voc", "is_overlay": True,
             "tl_track_url": ""},
        ]
    monkeypatch.setattr(mixes, "scrape_tracklist", fake_scrape)

    detail = mixes.import_mix(mixes.ImportRequest(
        url="https://www.1001tracklists.com/tracklist/abc/two-friends.html"))
    assert detail["track_count"] == 2
    assert detail["match_count"] == 1        # 'w/' overlay paired to the bed
    assert detail["import_method"] == "scrape"
