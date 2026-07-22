import pytest

from ingest import firecrawl_scrape as fc


def _fake_post(payload):
    def _post(url, body, headers):
        assert "Authorization" in headers
        assert body["proxy"] == "stealth"
        # v2 shape: JSON options are a typed object inside formats, not a
        # top-level "jsonOptions" key (that returns HTTP 400).
        assert "jsonOptions" not in body
        assert body["formats"][0]["type"] == "json"
        assert "schema" in body["formats"][0]
        return payload
    return _post


def test_scrape_tracklist_parses_tracks():
    payload = {"success": True, "data": {"json": {"tracks": [
        {"position": "01", "artist": "Dr. Dre ft. Snoop Dogg", "title": "Still D.R.E.",
         "is_overlay": False, "youtube_url": "https://www.1001tracklists.com/track/4tcxlv5/x/index.html"},
        {"position": "w/", "artist": "Eminem", "title": "Without Me",
         "is_overlay": True, "youtube_url": ""},
    ]}, "metadata": {"statusCode": 200}}}
    rows = fc.scrape_tracklist("https://www.1001tracklists.com/tracklist/x.html",
                               api_key="fc-k", _post=_fake_post(payload))
    assert rows[0]["position"] == "01"
    assert rows[0]["is_overlay"] is False
    assert rows[0]["tl_track_url"].endswith("/index.html")
    assert rows[1]["is_overlay"] is True
    assert rows[1]["tl_track_url"] == ""


def test_scrape_tracklist_empty_raises():
    payload = {"success": True, "data": {"json": {"tracks": []}, "metadata": {"statusCode": 200}}}
    with pytest.raises(fc.FirecrawlError):
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=_fake_post(payload))


def test_scrape_tracklist_no_key_raises():
    with pytest.raises(fc.FirecrawlError):
        fc.scrape_tracklist("https://x", api_key="")


def test_scrape_track_links():
    payload = {"success": True, "data": {"json": {
        "soundcloud_url": "https://soundcloud.com/x", "youtube_url": "https://www.youtube.com/watch?v=Q"}}}
    out = fc.scrape_track_links("https://www.1001tracklists.com/track/2/index.html",
                                api_key="fc-k", _post=_fake_post(payload))
    assert out["youtube_url"] == "https://www.youtube.com/watch?v=Q"
