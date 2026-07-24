import pytest

from ingest import firecrawl_scrape as fc


def _fake_post(payload, *, expect_format):
    def _post(url, body, headers):
        assert "Authorization" in headers
        assert body["proxy"] == "stealth"
        # v2 shape: never the old top-level jsonOptions key (that returns HTTP 400).
        assert "jsonOptions" not in body
        if expect_format == "markdown":
            assert body["formats"] == ["markdown"]
        else:
            assert body["formats"][0]["type"] == "json"
            assert "schema" in body["formats"][0]
        return payload
    return _post


# A trimmed sample of real rendered 1001tracklists markdown: a bed, then two
# "w/" overlays, then another bed. A bare "w/" line marks the next track overlay.
_MD = (
    "Guns N' Roses \\- Welcome To The Jungle[open track page]"
    "(https://www.1001tracklists.com/track/2r55l8f/x/index.html \"open track page\")LABEL\n"
    "w/\n"
    "Siik & Andrew A ft. Barmuda \\- Saviour[open track page]"
    "(https://www.1001tracklists.com/track/2rfu19r5/y/index.html \"open track page\")FUTURE\n"
    "w/\n"
    "Blink-182 \\- I Miss You[open track page]"
    "(https://www.1001tracklists.com/track/1r4bflsf/z/index.html \"open track page\") "
    "(Two Friends Remix)[open track page](https://www.1001tracklists.com/track/qqq/r/index.html)\n"
    "M83 \\- Midnight City[open track page]"
    "(https://www.1001tracklists.com/track/2kw1hrp/m/index.html \"open track page\")\n"
)


def test_scrape_tracklist_parses_all_tracks_from_markdown():
    payload = {"success": True, "data": {"markdown": _MD}}
    rows = fc.scrape_tracklist("https://www.1001tracklists.com/tracklist/x.html",
                               api_key="fc-k", _post=_fake_post(payload, expect_format="markdown"))
    # All four rows survive — no LLM truncation.
    assert len(rows) == 4
    assert rows[0]["is_overlay"] is False
    assert rows[0]["artist"] == "Guns N' Roses" and rows[0]["title"] == "Welcome To The Jungle"
    assert rows[0]["tl_track_url"].endswith("/index.html")
    assert rows[1]["is_overlay"] is True
    assert rows[2]["is_overlay"] is True
    # Remix credit trailing the first link is folded back into the title.
    assert "Two Friends Remix" in rows[2]["title"]
    assert rows[3]["is_overlay"] is False


def test_parse_markdown_overlay_markers():
    rows = fc.parse_markdown_tracklist(_MD)
    assert [r["is_overlay"] for r in rows] == [False, True, True, False]


def test_rework_link_tooltip_not_folded_into_title():
    # 1001tracklists renders a "rework of track X" annotation as a markdown link
    # with a title attribute: [text](url "rework of track …"). The remix-word in
    # that tooltip must NOT drag the URL (and tooltip) into the title — an
    # unsearchable title breaks SoundCloud/YouTube linking.
    md = (
        "Ian Asher & Olly Alexander \\- Desire[open track page]"
        "(https://www.1001tracklists.com/track/aaa/x/index.html \"open track page\")"
        "[Desire](https://www.1001tracklists.com/track/zw050jf/years-years-desire/index.html "
        "\"rework of track Years & Years - Desire\")\n"
    )
    rows = fc.parse_markdown_tracklist(md)
    assert len(rows) == 1
    assert rows[0]["artist"] == "Ian Asher & Olly Alexander"
    assert rows[0]["title"] == "Desire"
    assert "http" not in rows[0]["title"]


def test_scrape_tracklist_empty_raises():
    payload = {"success": True, "data": {"markdown": "no tracks here\njust prose\n"}}
    with pytest.raises(fc.FirecrawlError):
        fc.scrape_tracklist("https://x", api_key="fc-k",
                            _post=_fake_post(payload, expect_format="markdown"))


def test_scrape_tracklist_no_key_raises():
    with pytest.raises(fc.FirecrawlError):
        fc.scrape_tracklist("https://x", api_key="")


def test_scrape_track_links():
    payload = {"success": True, "data": {"json": {
        "soundcloud_url": "https://soundcloud.com/x", "youtube_url": "https://www.youtube.com/watch?v=Q"}}}
    out = fc.scrape_track_links("https://www.1001tracklists.com/track/2/index.html",
                                api_key="fc-k", _post=_fake_post(payload, expect_format="json"))
    assert out["youtube_url"] == "https://www.youtube.com/watch?v=Q"
