import importlib

import pytest
from fastapi import HTTPException

_TL_URL = "https://www.1001tracklists.com/tracklist/abc/two-friends.html"


def _fresh_mixes(tmp_path, monkeypatch):
    """Load the routes with NO Firecrawl key anywhere."""
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from ingest import firecrawl_scrape
    importlib.reload(firecrawl_scrape)
    from api.routes import mixes
    importlib.reload(mixes)
    return mixes, config


def test_1001_without_a_key_is_501_and_never_fetched(tmp_path, monkeypatch):
    mixes, _ = _fresh_mixes(tmp_path, monkeypatch)

    def no_fetch(url):
        raise AssertionError("a walled page must not be fetched without a key")
    monkeypatch.setattr(mixes, "_fetch_tracklist_html", no_fetch)

    with pytest.raises(HTTPException) as exc:
        mixes.import_mix(mixes.ImportRequest(url=_TL_URL))
    assert exc.value.status_code == 501
    assert "Firecrawl" in exc.value.detail


def test_key_saved_after_startup_is_used_without_a_restart(tmp_path, monkeypatch):
    # The bug: mixes.py imported FIRECRAWL_API_KEY at load, so a key saved from
    # the app was ignored until the server restarted.
    mixes, config = _fresh_mixes(tmp_path, monkeypatch)
    config.save_settings({"firecrawl_api_key": "fc-live"})

    seen = {}

    def fake_scrape(url, api_key=None, **kw):
        seen["url"] = url
        return [{"position": "01", "artist": "A", "title": "Bed",
                 "is_overlay": False, "tl_track_url": ""}]
    monkeypatch.setattr(mixes, "scrape_tracklist", fake_scrape)

    detail = mixes.import_mix(mixes.ImportRequest(url=_TL_URL))
    assert seen["url"] == _TL_URL
    assert detail["track_count"] == 1


def test_other_walled_site_is_502_not_a_key_prompt(tmp_path, monkeypatch):
    # Firecrawl's parser only understands 1001tracklists, so a key would not help
    # here — and 501 is what makes the Mixes tab ask for one.
    mixes, _ = _fresh_mixes(tmp_path, monkeypatch)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>Just a moment... challenges.cloudflare.com</html>"
    monkeypatch.setattr(mixes.urllib.request, "urlopen", lambda *a, **k: _Resp())

    with pytest.raises(HTTPException) as exc:
        mixes.import_mix(mixes.ImportRequest(url="https://example.com/set.html"))
    assert exc.value.status_code == 502
    assert "FIRECRAWL" not in exc.value.detail.upper()


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


def test_rejected_key_is_501_so_the_prompt_comes_back(tmp_path, monkeypatch):
    # A mistyped key saved from the Mixes tab must not strand the user: a 502
    # would hide the prompt with the bad key still saved and no way to replace it.
    mixes, config = _fresh_mixes(tmp_path, monkeypatch)
    config.save_settings({"firecrawl_api_key": "fc-WRONGKEY"})

    def rejected(url, api_key=None, **kw):
        raise mixes.FirecrawlAuthError("Firecrawl HTTP 401: Unauthorized")
    monkeypatch.setattr(mixes, "scrape_tracklist", rejected)

    with pytest.raises(HTTPException) as exc:
        mixes.import_mix(mixes.ImportRequest(url=_TL_URL))
    assert exc.value.status_code == 501
    assert "fc-WRONGKEY" not in exc.value.detail
