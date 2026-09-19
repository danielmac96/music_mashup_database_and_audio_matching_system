"""POST /api/mixes/import — the Firecrawl branch.

A 1001tracklists URL now answers with a job_id and the scrape runs in
api/workers/mix_ingest_worker. These tests call the route and then run that job
inline. The no-key 501 stays on the route itself: it is free and instant, and
501 is the one status the Mixes tab turns into a key prompt.
"""
import importlib

import pytest
from fastapi import BackgroundTasks, HTTPException

_TL_URL = "https://www.1001tracklists.com/tracklist/abc/two-friends.html"


def _reload_routes():
    from ingest import firecrawl_scrape
    importlib.reload(firecrawl_scrape)
    from api.workers import mix_ingest_worker
    importlib.reload(mix_ingest_worker)
    from api.routes import mixes
    importlib.reload(mixes)
    return mixes, firecrawl_scrape, mix_ingest_worker


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
    mixes, firecrawl_scrape, worker = _reload_routes()
    return mixes, config, firecrawl_scrape, worker


def _run_import(mixes, worker, url, refresh=False):
    """Call the route, run its job inline, and return (route_out, job)."""
    from api import jobs
    out = mixes.import_mix(mixes.ImportRequest(url=url, refresh=refresh),
                           BackgroundTasks())
    if "job_id" not in out:
        return out, None                      # the plain-HTML branch answers inline
    worker.run_import(out["job_id"], url, refresh)
    return out, jobs.get(out["job_id"])


def test_1001_without_a_key_is_501_and_never_fetched(tmp_path, monkeypatch):
    mixes, _c, _f, _w = _fresh_mixes(tmp_path, monkeypatch)

    def no_fetch(url):
        raise AssertionError("a walled page must not be fetched without a key")
    monkeypatch.setattr(mixes, "_fetch_tracklist_html", no_fetch)

    with pytest.raises(HTTPException) as exc:
        mixes.import_mix(mixes.ImportRequest(url=_TL_URL), BackgroundTasks())
    assert exc.value.status_code == 501
    assert "Firecrawl" in exc.value.detail


def test_key_saved_after_startup_is_used_without_a_restart(tmp_path, monkeypatch):
    # The bug: mixes.py imported FIRECRAWL_API_KEY at load, so a key saved from
    # the app was ignored until the server restarted.
    mixes, config, firecrawl_scrape, worker = _fresh_mixes(tmp_path, monkeypatch)
    config.save_settings({"firecrawl_api_key": "fc-live"})

    seen = {}

    def fake_scrape(url, api_key=None, **kw):
        seen["url"] = url
        return [{"position": "01", "artist": "A", "title": "Bed",
                 "is_overlay": False, "tl_track_url": ""}]
    monkeypatch.setattr(firecrawl_scrape, "scrape_tracklist", fake_scrape)

    _out, job = _run_import(mixes, worker, _TL_URL)
    assert seen["url"] == _TL_URL
    assert job["status"] == "completed", job.get("error")
    assert job["result"]["track_count"] == 1


def test_other_walled_site_is_502_not_a_key_prompt(tmp_path, monkeypatch):
    # Firecrawl's parser only understands 1001tracklists, so a key would not help
    # here — and 501 is what makes the Mixes tab ask for one.
    mixes, _c, _f, _w = _fresh_mixes(tmp_path, monkeypatch)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>Just a moment... challenges.cloudflare.com</html>"
    monkeypatch.setattr(mixes.urllib.request, "urlopen", lambda *a, **k: _Resp())

    with pytest.raises(HTTPException) as exc:
        mixes.import_mix(mixes.ImportRequest(url="https://example.com/set.html"),
                         BackgroundTasks())
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
    mixes, firecrawl_scrape, worker = _reload_routes()

    def fake_scrape(url, api_key=..., **kw):
        return [
            {"position": "01", "artist": "A", "title": "Bed", "is_overlay": False,
             "tl_track_url": "https://www.1001tracklists.com/track/1/index.html"},
            {"position": "w/", "artist": "B", "title": "Voc", "is_overlay": True,
             "tl_track_url": ""},
        ]
    monkeypatch.setattr(firecrawl_scrape, "scrape_tracklist", fake_scrape)

    _out, job = _run_import(mixes, worker, _TL_URL)
    assert job["status"] == "completed", job.get("error")

    detail = mixes.get_mix(job["result"]["mix_id"])
    assert detail["track_count"] == 2
    assert detail["match_count"] == 1        # 'w/' overlay paired to the bed
    assert detail["import_method"] == "scrape"


def test_a_heavy_scrape_does_not_ride_on_the_request(tmp_path, monkeypatch):
    # The reason this is a job: a stealth render of a ~200-track page can take
    # minutes, and the route used to hold the request open for all of it.
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-k")
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    mixes, firecrawl_scrape, _w = _reload_routes()

    def never(url, api_key=None, **kw):
        raise AssertionError("the route must not scrape inline")
    monkeypatch.setattr(firecrawl_scrape, "scrape_tracklist", never)

    out = mixes.import_mix(mixes.ImportRequest(url=_TL_URL), BackgroundTasks())
    assert out["job_id"]


def test_rejected_key_asks_for_a_new_one_rather_than_stranding_the_user(
        tmp_path, monkeypatch):
    # A mistyped key saved from the Mixes tab must not strand the user: the tab
    # raises its key prompt off needs_key, so a bad key discovered inside the
    # job has to set it. The key itself is never echoed back.
    mixes, config, firecrawl_scrape, worker = _fresh_mixes(tmp_path, monkeypatch)
    config.save_settings({"firecrawl_api_key": "fc-WRONGKEY"})

    def rejected(url, api_key=None, **kw):
        raise firecrawl_scrape.FirecrawlAuthError("Firecrawl HTTP 401: Unauthorized")
    monkeypatch.setattr(firecrawl_scrape, "scrape_tracklist", rejected)

    _out, job = _run_import(mixes, worker, _TL_URL)
    assert job["status"] == "failed"
    assert job["result"]["needs_key"] is True
    assert "fc-WRONGKEY" not in job["error"]


def test_a_scrape_failure_is_reported_on_the_job_not_as_a_crash(tmp_path, monkeypatch):
    mixes, config, firecrawl_scrape, worker = _fresh_mixes(tmp_path, monkeypatch)
    config.save_settings({"firecrawl_api_key": "fc-k"})

    def boom(url, api_key=None, **kw):
        raise firecrawl_scrape.FirecrawlError("Firecrawl HTTP 529: overloaded")
    monkeypatch.setattr(firecrawl_scrape, "scrape_tracklist", boom)

    _out, job = _run_import(mixes, worker, _TL_URL)
    assert job["status"] == "failed"
    assert "529" in job["error"]
    assert not (job["result"] or {}).get("needs_key")


def test_refresh_is_passed_through_to_the_scraper(tmp_path, monkeypatch):
    mixes, config, firecrawl_scrape, worker = _fresh_mixes(tmp_path, monkeypatch)
    config.save_settings({"firecrawl_api_key": "fc-k"})
    seen = {}

    def fake_scrape(url, api_key=None, *, refresh=False, **kw):
        seen["refresh"] = refresh
        return [{"position": "01", "artist": "A", "title": "Bed",
                 "is_overlay": False, "tl_track_url": ""}]
    monkeypatch.setattr(firecrawl_scrape, "scrape_tracklist", fake_scrape)

    _run_import(mixes, worker, _TL_URL, refresh=True)
    assert seen["refresh"] is True
