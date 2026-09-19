import pytest

from ingest import firecrawl_scrape as fc


@pytest.fixture(autouse=True)
def _isolated_markdown_cache(tmp_path, monkeypatch):
    """Point the rendered-markdown cache at a temp dir. It is a module global
    bound from DATA_DIR at import, so without this the suite would write into
    the repo and one test's cached page would answer another's scrape."""
    monkeypatch.setattr(fc, "MARKDOWN_CACHE_DIR", tmp_path / "tracklist_cache")


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


def test_real_page_with_a_turnstile_footer_widget_is_not_the_wall():
    # The bug: 1001tracklists embeds a Turnstile widget in the footer of the real,
    # fully rendered page (HTTP 200, all track links). The marker sniff matched it,
    # burned every attempt, and reported "challenge did not clear" (Big Bootie 27).
    footer = (
        "\nChecking your Browser…\n\nVerifying...\n\n"
        "Stuck? [Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/"
        "h/g/turnstile/f/av0/rch/pva50/0x4AAAAAACGccIXqjGsL5W5F/auto/fbE/new/normal"
        "?lang=auto#refresh)\n\nSuccess!\n\nVerification failed\n"
    )
    post, sent = _recording_post([
        {"success": True, "data": {"markdown": _MD + footer, "metadata": {"statusCode": 200}}},
    ])
    rows = fc.scrape_tracklist("https://www.1001tracklists.com/tracklist/x.html",
                               api_key="fc-k", _post=post)
    assert len(rows) == 4
    assert len(sent) == 1


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


def test_scrape_tracklist_no_key_raises(monkeypatch):
    monkeypatch.setattr(fc.config, "current_firecrawl_api_key", lambda: "")
    with pytest.raises(fc.FirecrawlError):
        fc.scrape_tracklist("https://x", api_key="")


def test_key_defaults_to_the_live_setting_not_an_import_time_constant(monkeypatch):
    # A default argument of FIRECRAWL_API_KEY froze the key when the module
    # loaded, so a key saved from the Mixes tab never reached the request.
    monkeypatch.setattr(fc.config, "current_firecrawl_api_key", lambda: "fc-saved-later")
    post, _ = _recording_post([{"success": True, "data": {"markdown": _MD}}])
    headers_seen = []

    def _post(url, body, headers):
        headers_seen.append(headers)
        return post(url, body, headers)
    fc.scrape_tracklist("https://x", _post=_post)
    assert headers_seen[0]["Authorization"] == "Bearer fc-saved-later"


def test_scrape_track_links():
    payload = {"success": True, "data": {"json": {
        "soundcloud_url": "https://soundcloud.com/x", "youtube_url": "https://www.youtube.com/watch?v=Q"}}}
    out = fc.scrape_track_links("https://www.1001tracklists.com/track/2/index.html",
                                api_key="fc-k", _post=_fake_post(payload, expect_format="json"))
    assert out["youtube_url"] == "https://www.youtube.com/watch?v=Q"


@pytest.mark.parametrize("code, auth", [(401, True), (403, True), (402, False), (500, False)])
def test_only_a_refused_key_raises_the_auth_error(monkeypatch, code, auth):
    import io
    import urllib.error

    def refuse(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, code, "x", {},
                                     io.BytesIO(b'{"error": "nope"}'))
    monkeypatch.setattr(fc.urllib.request, "urlopen", refuse)
    with pytest.raises(fc.FirecrawlError) as exc:
        fc._real_post("https://api.firecrawl.dev/v2/scrape", {}, {})
    assert isinstance(exc.value, fc.FirecrawlAuthError) is auth


# ── transient upstream failures (the 429/5xx/529 class) ──────────────────────
#
# These used to escape the whole retry schedule on the first attempt: every
# HTTPError became a plain FirecrawlError and the user got a 502 for what was a
# passing hiccup on Firecrawl's side.

def _http_error(code, retry_after=None):
    import io
    import urllib.error
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("https://api.firecrawl.dev/v2/scrape", code, "x",
                                  headers, io.BytesIO(b'{"error": "upstream"}'))


# _real_post is what classifies an HTTPError, so drive the classification
# through it rather than re-implementing the mapping in a double.
def _classify(code, retry_after=None):
    def refuse(req, timeout=None):
        raise _http_error(code, retry_after)
    return refuse


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 529])
def test_transient_codes_are_retried_not_raised(monkeypatch, code):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def _post(url, body, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            raise fc._Transient(f"Firecrawl HTTP {code}")
        return {"success": True, "data": {"markdown": _MD, "metadata": {"statusCode": 200}}}

    rows = fc.scrape_tracklist("https://x", api_key="fc-k", _post=_post)
    assert len(rows) == 4
    assert calls["n"] == 2          # the hiccup was retried, not surfaced


def test_transient_retries_are_bounded_and_then_reported(monkeypatch):
    slept = []
    monkeypatch.setattr(fc.time, "sleep", slept.append)
    calls = {"n": 0}

    def _post(url, body, headers):
        calls["n"] += 1
        raise fc._Transient("Firecrawl HTTP 529")

    with pytest.raises(fc.FirecrawlError) as exc:
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=_post)
    assert "529" in str(exc.value)
    # Every attempt costs credits, so one click has a hard ceiling.
    assert calls["n"] == fc._MAX_REQUESTS
    assert slept and all(s > 0 for s in slept)


def test_retry_after_beats_our_own_backoff(monkeypatch):
    slept = []
    monkeypatch.setattr(fc.time, "sleep", slept.append)
    calls = {"n": 0}

    def _post(url, body, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            raise fc._Transient("Firecrawl HTTP 429", retry_after=7)
        return {"success": True, "data": {"markdown": _MD}}

    fc.scrape_tracklist("https://x", api_key="fc-k", _post=_post)
    assert slept == [7]


@pytest.mark.parametrize("code, kind", [
    (429, fc._Transient), (503, fc._Transient), (529, fc._Transient),
    (402, fc.FirecrawlQuotaError), (401, fc.FirecrawlAuthError),
    (400, fc.FirecrawlError),
])
def test_http_codes_are_classified(monkeypatch, code, kind):
    monkeypatch.setattr(fc.urllib.request, "urlopen", _classify(code))
    with pytest.raises(kind):
        fc._real_post("https://api.firecrawl.dev/v2/scrape", {}, {})


def test_retry_after_header_is_read_and_capped(monkeypatch):
    monkeypatch.setattr(fc.urllib.request, "urlopen", _classify(429, retry_after="3"))
    with pytest.raises(fc._Transient) as exc:
        fc._real_post("https://api.firecrawl.dev/v2/scrape", {}, {})
    assert exc.value.retry_after == 3
    monkeypatch.setattr(fc.urllib.request, "urlopen",
                        _classify(429, retry_after="99999"))
    with pytest.raises(fc._Transient) as exc:
        fc._real_post("https://api.firecrawl.dev/v2/scrape", {}, {})
    assert exc.value.retry_after == fc._MAX_BACKOFF


def test_a_socket_timeout_is_transient_not_fatal(monkeypatch):
    def time_out(req, timeout=None):
        raise TimeoutError("timed out")
    monkeypatch.setattr(fc.urllib.request, "urlopen", time_out)
    with pytest.raises(fc._Transient):
        fc._real_post("https://api.firecrawl.dev/v2/scrape", {}, {})


# ── the socket budget ────────────────────────────────────────────────────────

def test_the_socket_budget_covers_the_render_budget(monkeypatch):
    # A fixed 90s was shorter than a 25s render of a ~200-track page plus proxy
    # overhead: Firecrawl rendered and billed the page, we hung up on it, and
    # the user saw a failed scrape.
    post, sent = _recording_post([_challenge_payload()])
    with pytest.raises(fc.FirecrawlChallenge):
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=post)
    for body in sent:
        assert body["timeout"] > body["waitFor"]
        assert fc.socket_timeout(body) > body["waitFor"] / 1000


# ── envelope shape ───────────────────────────────────────────────────────────

def test_a_nested_data_envelope_is_still_read():
    # Reading only resp["data"]["markdown"] turned any other shape into a silent
    # "no tracks" on a scrape that succeeded and was billed.
    payload = {"success": True, "data": {"data": {"markdown": _MD}}}
    rows = fc.scrape_tracklist("https://x", api_key="fc-k",
                               _post=_fake_post(payload, expect_format="markdown"))
    assert len(rows) == 4


def test_a_non_dict_data_is_a_clear_error_not_an_attributeerror():
    payload = {"success": True, "data": ["nope"]}
    with pytest.raises(fc.FirecrawlError) as exc:
        fc.scrape_tracklist("https://x", api_key="fc-k",
                            _post=_fake_post(payload, expect_format="markdown"))
    assert "unexpected payload shape" in str(exc.value)


def test_a_missing_success_flag_is_reported_not_swallowed(monkeypatch):
    # It used to `continue` in silence, burning every attempt on an envelope
    # nobody ever saw, and reporting the generic "returned no data".
    post, sent = _recording_post([{"data": {"markdown": _MD}}])
    with pytest.raises(fc.FirecrawlError) as exc:
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=post)
    assert "success=true" in str(exc.value)
    assert len(sent) == len(fc._WAIT_SCHEDULE)


def test_a_relabelled_track_link_is_not_mistaken_for_the_wall():
    # The link TEXT was the only "this render is good" signal, and the real page
    # always carries the footer challenge markers — so a site-side label change
    # would flag a perfect render as the wall and burn every attempt on it.
    md = _MD.replace("[open track page]", "[view track]")
    post, sent = _recording_post([
        {"success": True, "data": {"markdown": md, "metadata": {"statusCode": 200}}}])
    with pytest.raises(fc.FirecrawlError) as exc:
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=post)
    # The href still identifies it as a rendered page, so we stop at one request
    # and report a parse problem rather than pretending it is Cloudflare.
    assert not isinstance(exc.value, fc.FirecrawlChallenge)
    assert len(sent) == 1


def test_the_challenge_error_carries_the_payload_it_saw():
    post, _sent = _recording_post([_challenge_payload()])
    with pytest.raises(fc.FirecrawlChallenge) as exc:
        fc.scrape_tracklist("https://x", api_key="fc-k", _post=post)
    assert "statusCode=206" in str(exc.value)


# ── the rendered-markdown disk cache ─────────────────────────────────────────
#
# A stealth render costs real credits. Nothing used to be kept, so every retry
# — and every re-import — paid again, and a failed scrape left nothing behind
# to look at.

def test_a_scraped_page_is_cached_and_the_next_import_is_free():
    post, sent = _recording_post([{"success": True, "data": {"markdown": _MD}}])
    url = "https://www.1001tracklists.com/tracklist/x.html"
    assert len(fc.scrape_tracklist(url, api_key="fc-k", _post=post)) == 4
    assert fc.markdown_cache_path(url).exists()

    again = fc.scrape_tracklist(url, api_key="fc-k", _post=post)
    assert len(again) == 4
    assert len(sent) == 1          # no second request, no second charge


def test_refresh_pays_for_a_fresh_render_and_bypasses_both_caches():
    post, sent = _recording_post([{"success": True, "data": {"markdown": _MD}}])
    url = "https://www.1001tracklists.com/tracklist/x.html"
    fc.scrape_tracklist(url, api_key="fc-k", _post=post)

    fc.scrape_tracklist(url, api_key="fc-k", refresh=True, _post=post)
    assert len(sent) == 2
    assert sent[1]["maxAge"] == 0   # Firecrawl's own cache bypassed too


def test_an_unparseable_render_is_kept_on_disk_for_diagnosis():
    post, _sent = _recording_post([
        {"success": True, "data": {"markdown": "rendered, but nothing we know\n",
                                   "metadata": {"statusCode": 200}}}])
    url = "https://www.1001tracklists.com/tracklist/y.html"
    with pytest.raises(fc.FirecrawlError):
        fc.scrape_tracklist(url, api_key="fc-k", _post=post)
    assert fc.markdown_cache_path(url).exists()


def test_a_cached_page_that_parses_to_nothing_is_re_scraped():
    url = "https://www.1001tracklists.com/tracklist/z.html"
    fc.MARKDOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fc.markdown_cache_path(url).write_text("stale junk\n", encoding="utf-8")
    post, sent = _recording_post([{"success": True, "data": {"markdown": _MD}}])
    assert len(fc.scrape_tracklist(url, api_key="fc-k", _post=post)) == 4
    assert len(sent) == 1
