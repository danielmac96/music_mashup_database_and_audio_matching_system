"""Firecrawl-backed scrape of 1001tracklists.

A plain urllib GET of 1001tracklists returns a Cloudflare Turnstile interstitial
(see api/routes/mixes.py _TURNSTILE_MSG). Firecrawl's hosted stealth proxy renders
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
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from config import FIRECRAWL_API_KEY, FIRECRAWL_SCRAPE_URL


class FirecrawlError(RuntimeError):
    """Firecrawl was unreachable, unauthenticated, or returned no usable data."""


class FirecrawlChallenge(FirecrawlError):
    """The stealth proxy returned Cloudflare's interstitial instead of the page."""


# When Turnstile does not clear inside the render budget, Firecrawl still reports
# success:true — the payload is just the interstitial, with HTTP 206 and none of
# the page's track links. Undetected, that reads downstream as "this tracklist has
# no tracks", which is how two of the Big Bootie mixes appeared permanently
# unimportable. Sniff the interstitial and retry with a longer budget instead.
_CHALLENGE_MARKERS = (
    "checking your browser",
    "you will be forwarded to the requested page",
    "challenges.cloudflare.com",
)

# Escalating render budgets, in ms. 6s clears the wall on most tracklist pages;
# the heaviest ones (~240 tracks) need appreciably longer.
_WAIT_SCHEDULE = (6000, 15000, 25000)


def _is_challenge(data: dict) -> bool:
    """True when `data` is Cloudflare's interstitial rather than the real page."""
    if (data.get("metadata") or {}).get("statusCode") == 206:
        return True
    md = (data.get("markdown") or "").lower()
    return any(marker in md for marker in _CHALLENGE_MARKERS)


# A rendered track row: "Artist \- Title[open track page](https://.../track/ID/index.html ...".
_TRACK_LINE_RE = re.compile(
    r"^(?P<body>.+?)\[open track page\]\((?P<url>https://www\.1001tracklists\.com/track/[^ )]+)")
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


def _real_post(url: str, body: dict, headers: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        # Surface Firecrawl's error body — it names the offending field on a 400.
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace"))
            msg = detail.get("error") or detail.get("code") or ""
        except Exception:
            msg = ""
        raise FirecrawlError(f"Firecrawl HTTP {exc.code}{f': {msg}' if msg else ''}") from exc
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        raise FirecrawlError(f"Firecrawl request failed: {exc}") from exc


def _post_scrape(url: str, formats: list, api_key: str, _post) -> dict:
    """POST a /v2/scrape request and return the `data` object, or raise.

    Retries a Cloudflare interstitial with a longer render budget. Only the wall
    is retried — a page that rendered but parsed to nothing is a markup problem
    and costs credits to re-scrape for no gain.
    """
    if not api_key:
        raise FirecrawlError("FIRECRAWL_API_KEY is not set — configure it to scrape URLs.")
    post = _post or _real_post
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    failure = FirecrawlError("Firecrawl returned no data (challenge or empty page).")
    for attempt, wait_ms in enumerate(_WAIT_SCHEDULE):
        body = {"url": url, "formats": formats, "proxy": "stealth", "waitFor": wait_ms}
        if attempt:
            # Firecrawl caches the interstitial like any other response (default
            # ~2 days), so a walled URL keeps replaying its own wall until the
            # cache is bypassed. That is what made these failures look permanent
            # and URL-specific rather than transient.
            body["maxAge"] = 0
        resp = post(FIRECRAWL_SCRAPE_URL, body, headers)
        if not resp or not resp.get("success"):
            continue
        data = resp.get("data") or {}
        if not _is_challenge(data):
            return data
        failure = FirecrawlChallenge(
            f"Cloudflare challenge did not clear in {len(_WAIT_SCHEDULE)} attempts "
            f"(up to {_WAIT_SCHEDULE[-1] // 1000}s) — the site is rate-limiting; retry shortly.")
    raise failure


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


def scrape_tracklist(url: str, api_key: str = FIRECRAWL_API_KEY, *, _post=None) -> list[dict]:
    data = _post_scrape(url, ["markdown"], api_key, _post)
    md = data.get("markdown") or ""
    rows = [r for r in parse_markdown_tracklist(md) if r["artist"] or r["title"]]
    if not rows:
        raise FirecrawlError("Firecrawl scraped the page but found no tracks.")
    return rows


def scrape_track_links(track_page_url: str, api_key: str = FIRECRAWL_API_KEY, *, _post=None) -> dict:
    data = _scrape_json(track_page_url, _LINKS_SCHEMA, _LINKS_PROMPT, api_key, _post)
    return {
        "soundcloud_url": (data.get("soundcloud_url") or "").strip(),
        "youtube_url": (data.get("youtube_url") or "").strip(),
    }
