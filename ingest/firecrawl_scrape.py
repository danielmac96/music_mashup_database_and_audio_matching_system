"""Firecrawl-backed structured scrape of 1001tracklists.

A plain urllib GET of 1001tracklists returns a Cloudflare Turnstile interstitial
(see api/routes/mixes.py _TURNSTILE_MSG). Firecrawl's hosted stealth proxy renders
the page and returns structured JSON. We call its HTTP /v2/scrape endpoint directly
with stdlib urllib (no firecrawl-py dependency).

Two page shapes:
  * the tracklist listing  -> beds/overlays + each track's internal /track/{id} URL
  * a per-track sub-page    -> the real SoundCloud/YouTube streaming URLs
The listing does NOT carry the external URLs; those live on the per-track pages,
fetched on-demand (~9 credits each) — never all 216 at once.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import FIRECRAWL_API_KEY, FIRECRAWL_SCRAPE_URL


class FirecrawlError(RuntimeError):
    """Firecrawl was unreachable, unauthenticated, or returned no usable data."""


_TRACKLIST_SCHEMA = {
    "type": "object",
    "properties": {"tracks": {"type": "array", "items": {"type": "object", "properties": {
        "position": {"type": "string"},          # "01".."NN" for beds, "w/" for overlays
        "artist": {"type": "string"},
        "title": {"type": "string"},
        "is_overlay": {"type": "boolean"},
        "youtube_url": {"type": "string"},        # here = the internal /track/{id} link
    }}}},
}
_TRACKLIST_PROMPT = (
    "Extract every track in this DJ set tracklist in order. For each track return: "
    "the printed entry number as position ('w/' for a mashup overlay line), the artist, "
    "the title, is_overlay true for 'w/' overlay lines, and youtube_url = the track's "
    "detail-page link on this site. Include 'w/' sub-entries as separate tracks.")

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


def _scrape(url: str, schema: dict, prompt: str, api_key: str, _post) -> dict:
    if not api_key:
        raise FirecrawlError("FIRECRAWL_API_KEY is not set — configure it to scrape URLs.")
    post = _post or _real_post
    # Firecrawl v2: JSON extraction options live INSIDE the formats array as a
    # typed object — the old top-level "jsonOptions" key is rejected with HTTP 400.
    body = {
        "url": url,
        "formats": [{"type": "json", "prompt": prompt, "schema": schema}],
        "proxy": "stealth",
        "waitFor": 6000,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = post(FIRECRAWL_SCRAPE_URL, body, headers)
    if not resp or not resp.get("success"):
        raise FirecrawlError("Firecrawl returned no data (challenge or empty page).")
    return (resp.get("data") or {}).get("json") or {}


def scrape_tracklist(url: str, api_key: str = FIRECRAWL_API_KEY, *, _post=None) -> list[dict]:
    data = _scrape(url, _TRACKLIST_SCHEMA, _TRACKLIST_PROMPT, api_key, _post)
    tracks = data.get("tracks") or []
    rows = []
    for t in tracks:
        rows.append({
            "position": str(t.get("position") or "").strip(),
            "artist": (t.get("artist") or "").strip(),
            "title": (t.get("title") or "").strip(),
            "is_overlay": bool(t.get("is_overlay")),
            "tl_track_url": (t.get("youtube_url") or "").strip(),
        })
    rows = [r for r in rows if r["artist"] or r["title"]]
    if not rows:
        raise FirecrawlError("Firecrawl scraped the page but found no tracks.")
    return rows


def scrape_track_links(track_page_url: str, api_key: str = FIRECRAWL_API_KEY, *, _post=None) -> dict:
    data = _scrape(track_page_url, _LINKS_SCHEMA, _LINKS_PROMPT, api_key, _post)
    return {
        "soundcloud_url": (data.get("soundcloud_url") or "").strip(),
        "youtube_url": (data.get("youtube_url") or "").strip(),
    }
