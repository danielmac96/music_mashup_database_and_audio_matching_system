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


# What the stealth proxy actually returns when Turnstile does not clear inside
# the render budget: HTTP 206 and the interstitial, wrapped in success:true.
_CHALLENGE_MD = (
    "# Please wait, you will be forwarded to the requested page\n\n"
    "Checking your Browser…\n\nVerifying...\n\n"
    "Stuck? [Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/b/"
    "turnstile/f/av0/rch/76m0l/0x4AAAAAACGccIXqjGsL5W5F/auto/fbE/new/normal?lang=auto#refresh)\n"
)


def _challenge_payload():
    return {"success": True,
            "data": {"markdown": _CHALLENGE_MD, "metadata": {"statusCode": 206}}}


def _recording_post(payloads):
    """Serve `payloads` in order, recording each request body."""
    sent = []

    def _post(url, body, headers):
        sent.append(body)
        return payloads[min(len(sent) - 1, len(payloads) - 1)]
    return _post, sent


def test_challenge_page_is_retried_with_a_longer_render_budget():
    # The bug: a Turnstile interstitial arrives as success:true with zero track
    # links, so the caller reported "no tracks" on a page that has ~200 of them.
    post, sent = _recording_post([
        _challenge_payload(),
        {"success": True, "data": {"markdown": _MD, "metadata": {"statusCode": 200}}},
    ])
    rows = fc.scrape_tracklist("https://www.1001tracklists.com/tracklist/x.html",
                               api_key="fc-k", _post=post)
    assert len(rows) == 4
    assert len(sent) == 2
    # Second attempt waits longer AND bypasses the cache — Firecrawl caches the
    # 206 interstitial, so a plain retry just replays the same wall.
    assert sent[1]["waitFor"] > sent[0]["waitFor"]
    assert sent[1]["maxAge"] == 0


def test_challenge_on_every_attempt_raises_a_challenge_error():
    post, sent = _recording_post([_challenge_payload()])
    with pytest.raises(fc.FirecrawlChallenge):
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=post)
    assert len(sent) == len(fc._WAIT_SCHEDULE) > 1


def test_track_links_scrape_also_retries_the_challenge():
    # The per-track JSON scrape hits the same wall; it has no markdown to sniff,
    # so the 206 status is what identifies the interstitial.
    post, sent = _recording_post([
        {"success": True, "data": {"json": {}, "metadata": {"statusCode": 206}}},
        {"success": True, "data": {"json": {"soundcloud_url": "https://soundcloud.com/x"},
                                   "metadata": {"statusCode": 200}}},
    ])
    out = fc.scrape_track_links("https://www.1001tracklists.com/track/2/index.html",
                                api_key="fc-k", _post=post)
    assert out["soundcloud_url"] == "https://soundcloud.com/x"
    assert len(sent) == 2


def test_genuinely_empty_page_is_not_retried():
    # A rendered page we simply can't parse is a parser/markup problem, not a
    # wall — burning three stealth scrapes on it wastes credits.
    post, sent = _recording_post([
        {"success": True, "data": {"markdown": "no tracks here\njust prose\n",
                                   "metadata": {"statusCode": 200}}},
    ])
    with pytest.raises(fc.FirecrawlError):
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=post)
    assert len(sent) == 1


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
