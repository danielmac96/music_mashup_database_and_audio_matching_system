"""Firecrawl-backed scrape of 1001tracklists.

A plain urllib GET of 1001tracklists returns a Cloudflare Turnstile interstitial
(see api/routes/mixes.py _FIRECRAWL_KEY_MSG). Firecrawl's hosted stealth proxy renders
the page and returns its content. We call the HTTP /v2/scrape endpoint directly
with stdlib urllib (no firecrawl-py dependency).

Two page shapes, two strategies:
  * the tracklist listing -> scraped as MARKDOWN and parsed deterministically.
    LLM json extraction truncates a ~200-track set (it capped at ~33), so we parse
    the rendered markdown ourselves: every track line carries an "[open track page]"
    link, and a bare "w/" line marks the next track as a mashup overlay.
  * a per-track sub-page   -> scraped with LLM json extraction (a small page, no
    truncation risk) for the real SoundCloud/YouTube streaming URLs.
The listing does NOT carry the external URLs; those live on the per-track pages,
fetched on-demand (~9 credits each) — never all ~200 at once.

Every request can cost credits, so failure handling is built around not wasting
them: a rendered listing is cached to disk and re-parsed for free, the socket
budget is derived from the render budget we asked for (a fixed 90s used to
abandon pages Firecrawl had already rendered and billed), a transient upstream
failure — timeout, 429, any 5xx including 529 — is retried with backoff instead
of escaping on the first attempt, and every failure branch logs the payload so a
scrape the dashboard calls a success is not a mystery on this side.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import config
from config import DATA_DIR, FIRECRAWL_SCRAPE_URL

log = logging.getLogger(__name__)


class FirecrawlError(RuntimeError):
    """Firecrawl was unreachable, unauthenticated, or returned no usable data."""


class FirecrawlChallenge(FirecrawlError):
    """The stealth proxy returned Cloudflare's interstitial instead of the page."""


class FirecrawlAuthError(FirecrawlError):
    """Firecrawl refused the API key (HTTP 401/403). Separate from other failures
    because a different key fixes it, so the Mixes tab asks for one again."""


class FirecrawlQuotaError(FirecrawlError):
    """Firecrawl accepted the key but the account is out of credits (HTTP 402).
    Retrying cannot fix it and neither can a different key, so it says so."""


# When Turnstile does not clear inside the render budget, Firecrawl still reports
# success:true — the payload is just the interstitial, with HTTP 206 and none of
# the page's track links. Undetected, that reads downstream as "this tracklist has
# no tracks", which is how two of the Big Bootie mixes appeared permanently
# unimportable. Sniff the interstitial and retry with a longer budget instead.
# The markers alone are not evidence: the real page embeds a Turnstile widget in
# its footer ("Checking your Browser… Stuck? [Troubleshoot](challenges.cloudflare…)"),
# which flagged every good render as the wall. Track links win over markers.
_CHALLENGE_MARKERS = (
    "checking your browser",
    "you will be forwarded to the requested page",
    "challenges.cloudflare.com",
)

# Every rendered track row carries this link; the interstitial never does.
_TRACK_LINK = "[open track page]"
# …and the href behind it. Checked as a SECOND positive signal, because the link
# TEXT is the only thing separating "good page" from "wall" (the real page always
# carries the footer markers above), and a site-side label change would otherwise
# flag a perfect render as the wall and burn every attempt on it.
_TRACK_HREF = "1001tracklists.com/track/"

# Escalating render budgets, in ms. 6s clears the wall on most tracklist pages;
# the heaviest ones (~240 tracks) need appreciably longer.
_WAIT_SCHEDULE = (6000, 15000, 25000)

# Transient upstream failures worth another go: request timeout, rate limit, and
# anything 5xx — Firecrawl answers 502/503/504 when the target site is slow and
# 529 when it is overloaded. None of these mean the page is unscrapable. (The
# 5xx half is a range, checked alongside this tuple rather than listed here.)
_TRANSIENT_CODES = (408, 425, 429)
# Backoff between transient retries, in seconds. A Retry-After header wins.
_BACKOFF = (2, 4, 8)
_MAX_BACKOFF = 30

# Slack on top of the render budget before the socket gives up, and the floor
# under it so a body with no timeout field still waits a sane while.
_SOCKET_HEADROOM_SECS = 60

# Hard ceiling on requests per scrape call, shared by the render-budget
# escalation and the transient retry. Every attempt can cost credits, so this is
# what stops one click quietly spending a handful of them.
_MAX_REQUESTS = 5


def _rendered_ok(md: str) -> bool:
    """True when this markdown carries real tracklist rows."""
    low = md.lower()
    return _TRACK_LINK.lower() in low or _TRACK_HREF in low


def _is_challenge(data: dict) -> bool:
    """True when `data` is Cloudflare's interstitial rather than the real page."""
    md = data.get("markdown") or ""
    if _rendered_ok(md):
        return False
    if (data.get("metadata") or {}).get("statusCode") == 206:
        return True
    low = md.lower()
    return any(marker in low for marker in _CHALLENGE_MARKERS)


def _describe(data: dict) -> str:
    """A one-line fingerprint of a payload, for logs and error messages. Without
    it a failed scrape leaves nothing behind but "returned no data", and there is
    no way to tell a wall from an envelope we failed to read."""
    md = data.get("markdown") if isinstance(data, dict) else None
    meta = (data.get("metadata") or {}) if isinstance(data, dict) else {}
    excerpt = (md or "")[:200].replace("\n", " ")
    return (f"keys={sorted(data)[:8] if isinstance(data, dict) else type(data).__name__} "
            f"statusCode={meta.get('statusCode')} markdown_len={len(md or '')} "
            f"excerpt={excerpt!r}")


# A rendered track row: "Artist \- Title[open track page](https://.../track/ID/index.html ...".
_TRACK_LINE_RE = re.compile(
    r"^(?P<body>.+?)" + re.escape(_TRACK_LINK)
    + r"\((?P<url>https://www\.1001tracklists\.com/track/[^ )]+)")
# A remix/edit annotation that trails after the first link — fold it back into the title
# so the shared parse_line can derive the remixer.
_REMIX_PAREN_RE = re.compile(
    r"\(([^)]*\b(?:remix|mix|edit|flip|bootleg|mashup|vip|rework|refix)\b[^)]*)\)", re.I)
# A URL fragment leaked into a title — possibly wrapped in "(" and possibly
# carrying a markdown-link title attribute (…index.html "rework of track …").
# Both the URL and the trailing quoted tooltip must go, or an unsearchable
# remnant survives and breaks the downstream SoundCloud/YouTube title search.
_URL_FRAGMENT_RE = re.compile(r'\(?\s*https?://[^\s)]*(?:\s+"[^"]*")?\s*\)?', re.I)

_LINKS_SCHEMA = {
    "type": "object",
    "properties": {
        "soundcloud_url": {"type": "string"},
        "youtube_url": {"type": "string"},
    },
}
_LINKS_PROMPT = (
    "Extract the external streaming links for this track: return the SoundCloud URL "
    "and the YouTube watch URL if present.")


class _Transient(FirecrawlError):
    """An upstream hiccup worth retrying. Carries the server's Retry-After, if
    it sent one, and never escapes this module — the last attempt re-raises as a
    plain FirecrawlError so callers keep one exception vocabulary."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_secs(exc: urllib.error.HTTPError) -> float | None:
    raw = (exc.headers or {}).get("Retry-After") if exc.headers else None
    try:
        return max(0.0, min(float(raw), _MAX_BACKOFF)) if raw else None
    except (TypeError, ValueError):
        return None   # an HTTP-date Retry-After: fall back to our own backoff


def socket_timeout(body: dict) -> float:
    """The socket budget for one request, derived from the render budget in its
    own body. A fixed 90s was shorter than a 25s render of a ~200-track page
    plus proxy overhead, so a page Firecrawl rendered AND BILLED was abandoned
    client-side and reported to the user as a failed scrape."""
    return (body.get("timeout") or 0) / 1000 + _SOCKET_HEADROOM_SECS


def _real_post(url: str, body: dict, headers: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=socket_timeout(body)) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        # Surface Firecrawl's error body — it names the offending field on a 400.
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace"))
            msg = detail.get("error") or detail.get("code") or ""
        except Exception:
            msg = ""
        label = f"Firecrawl HTTP {exc.code}{f': {msg}' if msg else ''}"
        if exc.code in (401, 403):
            raise FirecrawlAuthError(label) from exc
        if exc.code == 402:
            raise FirecrawlQuotaError(
                f"{label} — the Firecrawl account is out of credits.") from exc
        if exc.code in _TRANSIENT_CODES or exc.code >= 500:
            raise _Transient(label, _retry_after_secs(exc)) from exc
        raise FirecrawlError(label) from exc
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        # A socket timeout usually means the render outran our budget rather
        # than that the page is unscrapable, so it is worth one more go.
        raise _Transient(f"Firecrawl request failed: {exc}") from exc


class _Budget:
    """Requests left for one scrape call, shared by the render-budget escalation
    and the transient retry so a single click cannot quietly spend a handful of
    credits chasing a page that is not coming back."""

    def __init__(self, limit: int):
        self.left = limit

    def take(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        return True


def _send(post, body: dict, headers: dict, budget: _Budget) -> dict:
    """One scrape request, retrying a transient upstream failure with backoff.

    Transient = timeout, 408/425/429, or any 5xx (Firecrawl answers 502/503/504
    when the target site is slow and 529 when it is overloaded). None of those
    mean the page is unscrapable, and none of them used to be retried at all:
    the exception escaped the whole schedule on the first attempt."""
    last: _Transient | None = None
    tries = 0
    while budget.take():
        if last is not None:
            delay = (last.retry_after if last.retry_after is not None
                     else _BACKOFF[min(tries - 1, len(_BACKOFF) - 1)])
            log.warning("Firecrawl transient failure (%s) — retrying in %.0fs", last, delay)
            time.sleep(delay)
        tries += 1
        try:
            return post(FIRECRAWL_SCRAPE_URL, body, headers)
        except _Transient as exc:
            last = exc
    raise FirecrawlError(str(last) if last else
                         "Firecrawl request budget exhausted before any answer.")


def _unwrap(resp: dict) -> dict:
    """The `data` object out of a /v2/scrape envelope, tolerating one extra
    level of nesting. Reading only `resp["data"]["markdown"]` meant an envelope
    of any other shape became a silent "no tracks" on a scrape that succeeded
    and was billed."""
    data = resp.get("data")
    if isinstance(data, dict) and not data.get("markdown") and not data.get("json"):
        inner = data.get("data")
        if isinstance(inner, dict):
            return inner
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise FirecrawlError(
            f"Firecrawl returned an unexpected payload shape ({type(data).__name__}); "
            "the scrape succeeded but could not be read.")
    return data


def _post_scrape(url: str, formats: list, api_key: str, _post,
                 *, bypass_cache: bool = False) -> dict:
    """POST a /v2/scrape request and return the `data` object, or raise.

    Two things get retried, for different reasons:
      * a Cloudflare interstitial — with a longer render budget, and with
        Firecrawl's cache bypassed (it caches the 206 wall like any other
        response, which is what made these failures look permanent).
      * a transient upstream failure — with exponential backoff (see _send).
    A page that rendered but parsed to nothing is neither: that is a markup
    problem and costs credits to re-scrape for no gain.
    """
    # Resolved per call, not as a default argument: a key saved from the Mixes
    # tab must reach the very next request without a server restart.
    api_key = api_key or config.current_firecrawl_api_key()
    if not api_key:
        raise FirecrawlError("No Firecrawl API key — add one in the Mixes tab or "
                             "set FIRECRAWL_API_KEY.")
    post = _post or _real_post
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "Accept": "application/json"}
    budget = _Budget(_MAX_REQUESTS)
    failure = FirecrawlError("Firecrawl returned no data (challenge or empty page).")
    for attempt, wait_ms in enumerate(_WAIT_SCHEDULE):
        if budget.left <= 0:
            break
        body = {"url": url, "formats": formats, "proxy": "stealth",
                "waitFor": wait_ms,
                # Bound Firecrawl's own render too, so its answer arrives inside
                # our socket budget rather than after it.
                "timeout": wait_ms + _SOCKET_HEADROOM_SECS * 1000}
        if attempt or bypass_cache:
            body["maxAge"] = 0
        try:
            resp = _send(post, body, headers, budget)
        except FirecrawlError as exc:
            # Transient and out of retries. Keep it as the reported cause — it
            # is far more useful than the generic "no data" below.
            failure = exc
            break
        if not resp or not resp.get("success"):
            # This used to `continue` in silence, burning every attempt and
            # three billed scrapes on an envelope nobody ever saw.
            log.warning("Firecrawl answered without success=true for %s "
                        "(attempt %d, keys=%s)", url, attempt + 1,
                        sorted(resp) if isinstance(resp, dict) else type(resp).__name__)
            failure = FirecrawlError(
                "Firecrawl answered without success=true — the scrape may have "
                "been billed; see the server log for the payload.")
            continue
        data = _unwrap(resp)
        if not _is_challenge(data):
            return data
        log.warning("Firecrawl returned the Cloudflare wall for %s "
                    "(attempt %d, waitFor=%dms): %s",
                    url, attempt + 1, wait_ms, _describe(data))
        failure = FirecrawlChallenge(
            f"Cloudflare interstitial returned on every attempt "
            f"(up to {_WAIT_SCHEDULE[-1] // 1000}s render budget) — retry shortly. "
            f"Last payload: {_describe(data)}")
    raise failure


# Raw rendered markdown, cached to disk keyed by URL — the same write-once disk
# cache the plain-HTML tracklist fetch already uses (api/routes/mixes.py
# _HTML_CACHE_DIR). A stealth render of a heavy tracklist costs real credits, so
# a page scraped once is re-parsed from here for free, and a scrape that fails
# downstream leaves the payload behind to look at instead of nothing at all.
MARKDOWN_CACHE_DIR = DATA_DIR / "tracklist_cache"


def markdown_cache_path(url: str) -> Path:
    return MARKDOWN_CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".md")


def _cached_markdown(url: str) -> str:
    path = markdown_cache_path(url)
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:  # the cache is an optimisation, never a failure mode
        log.warning("could not read cached markdown for %s", url, exc_info=True)
        return ""


def _cache_markdown(url: str, md: str) -> None:
    try:
        MARKDOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        markdown_cache_path(url).write_text(md, encoding="utf-8")
    except OSError:
        log.warning("could not cache markdown for %s", url, exc_info=True)


def _scrape_json(url: str, schema: dict, prompt: str, api_key: str, _post) -> dict:
    # Firecrawl v2: JSON extraction options live INSIDE the formats array as a
    # typed object — the old top-level "jsonOptions" key is rejected with HTTP 400.
    data = _post_scrape(url, [{"type": "json", "prompt": prompt, "schema": schema}], api_key, _post)
    return data.get("json") or {}


def parse_markdown_tracklist(md: str) -> list[dict]:
    """Deterministically parse a rendered 1001tracklists page into track rows.

    Each track is a line ending in an "[open track page](…/track/ID…)" link; a bare
    "w/" line immediately before a track marks it as a mashup overlay on the previous
    (non-overlay) bed. Returns rows shaped like the old LLM output:
    {position, artist, title, is_overlay, tl_track_url}.
    """
    rows: list[dict] = []
    pending_overlay = False
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower() == "w/":
            pending_overlay = True
            continue
        m = _TRACK_LINE_RE.match(line)
        if not m:
            continue
        body = m.group("body").replace("\\-", "-").replace("\\", "").strip()
        # Defensive: never let a leaked "(https://…/track/…" URL fragment into the
        # artist/title — it wrecks a downstream title search.
        body = _URL_FRAGMENT_RE.sub("", body).strip()
        if " - " in body:
            artist, title = body.split(" - ", 1)
        else:
            artist, title = "", body
        # A genuine remix credit prints as a "(X Remix)" parenthetical after the
        # first link; fold it back into the title so parse_line can pull out the
        # remixer downstream. But 1001tracklists also renders a "rework of track …"
        # annotation as a markdown link whose *title attribute* holds a remix word
        # — [text](url "rework of track …"). That paren carries a URL, not a
        # remixer name; folding it in produces an unsearchable title, so skip any
        # match that contains a URL and take the first credit that doesn't.
        for rem in _REMIX_PAREN_RE.finditer(line[m.end():]):
            credit = rem.group(1).strip()
            if "http" in credit.lower():
                continue
            if "remix" not in title.lower():
                title = f"{title.strip()} ({credit})"
            break
        # Defensive last resort: never let a leaked URL fragment survive into a title.
        title = _URL_FRAGMENT_RE.sub("", title).strip()
        rows.append({
            "position": "w/" if pending_overlay else "",
            "artist": artist.strip(" -"),
            "title": title.strip(" -"),
            "is_overlay": pending_overlay,
            "tl_track_url": m.group("url"),
        })
        pending_overlay = False
    return rows


def scrape_tracklist(url: str, api_key: str | None = None, *,
                     refresh: bool = False, _post=None) -> list[dict]:
    """Rendered tracklist rows for `url`, from the disk cache when we have it.

    `refresh` pays for a fresh render and bypasses both caches, ours and
    Firecrawl's — the escape hatch for a page whose tracklist has since changed.
    """
    if not refresh:
        cached = _cached_markdown(url)
        if cached:
            rows = _rows_from(cached)
            if rows:
                log.info("re-parsed %d tracks for %s from the markdown cache "
                         "(no Firecrawl request)", len(rows), url)
                return rows
            log.warning("cached markdown for %s parses to no tracks — re-scraping",
                        url)

    data = _post_scrape(url, ["markdown"], api_key, _post, bypass_cache=refresh)
    md = data.get("markdown") or ""
    rows = _rows_from(md)
    if not rows:
        # The scrape succeeded and was billed; keep the payload so the failure
        # can be read rather than guessed at.
        _cache_markdown(url, md)
        log.warning("Firecrawl rendered %s but no track rows parsed out: %s",
                    url, _describe(data))
        raise FirecrawlError(
            "Firecrawl scraped the page but found no tracks "
            f"({len(md)} chars of markdown saved to {markdown_cache_path(url).name}).")
    _cache_markdown(url, md)
    return rows


def _rows_from(md: str) -> list[dict]:
    return [r for r in parse_markdown_tracklist(md) if r["artist"] or r["title"]]


def scrape_track_links(track_page_url: str, api_key: str | None = None, *, _post=None) -> dict:
    data = _scrape_json(track_page_url, _LINKS_SCHEMA, _LINKS_PROMPT, api_key, _post)
    return {
        "soundcloud_url": (data.get("soundcloud_url") or "").strip(),
        "youtube_url": (data.get("youtube_url") or "").strip(),
    }
