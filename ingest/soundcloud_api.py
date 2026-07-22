"""Resolve mix tracks to real SoundCloud URLs via SoundCloud's own v2 API.

This decouples link resolution from 1001tracklists' anti-bot entirely: we search
SoundCloud's internal search endpoint (https://api-v2.soundcloud.com/search/tracks)
by "Artist - Title" and pick the best match with a confidence score. No account or
registered app is needed — SoundCloud embeds a working ``client_id`` in its public
JS bundles, which we scrape once (stdlib urllib, no new dependency) and cache. If a
cached id expires (HTTP 401), we re-scrape once and retry.

Confidence is the same fuzzy artist+title score the yt-dlp path uses
(ingest.soundcloud._search_score), plus a play-count tiebreak so a low-play
re-upload never beats the official upload on an otherwise equal title match.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from ingest.soundcloud import _search_score

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_SEARCH_URL = "https://api-v2.soundcloud.com/search/tracks"
_CLIENT_ID_RE = re.compile(r'client_id\s*[:=]\s*"([A-Za-z0-9]{20,})"')
_SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="(https://[^"]+\.js)"')

_client_id: str | None = None   # module-level cache


class SoundCloudAPIError(RuntimeError):
    """SoundCloud was unreachable or exposed no usable client_id."""


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace")


def get_client_id(*, _get=None, force: bool = False) -> str:
    """Return a working SoundCloud client_id, scraping + caching it on first use.

    Scrapes the public JS bundles linked from the SoundCloud homepage; the id sits
    in a ``client_id:"…"`` literal. ``force`` re-scrapes past the cache (used after a
    401). ``_get`` is an injectable ``(url) -> text`` for tests."""
    global _client_id
    if _client_id and not force:
        return _client_id
    get = _get or _http_get
    try:
        html = get("https://soundcloud.com/")
        # Later bundles carry the api client_id more often — try them first.
        for src in reversed(_SCRIPT_SRC_RE.findall(html)):
            try:
                js = get(src)
            except Exception:  # noqa: BLE001 — skip a bundle that won't load
                continue
            m = _CLIENT_ID_RE.search(js)
            if m:
                _client_id = m.group(1)
                return _client_id
    except Exception as exc:  # noqa: BLE001
        raise SoundCloudAPIError(f"Could not reach SoundCloud: {exc}") from exc
    raise SoundCloudAPIError("No client_id found in SoundCloud's public bundles.")


def _search(query: str, limit: int, *, _get=None) -> list[dict]:
    """Raw v2 search — returns the `collection` list. Re-scrapes the client_id once
    on a 401 (expired id) and retries."""
    get = _get or _http_get
    for attempt in (0, 1):
        cid = get_client_id(_get=_get, force=(attempt == 1))
        qs = urllib.parse.urlencode({"q": query, "limit": limit, "client_id": cid})
        try:
            body = get(f"{_SEARCH_URL}?{qs}")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403) and attempt == 0:
                continue  # stale client_id — force a re-scrape and retry once
            raise SoundCloudAPIError(f"SoundCloud search HTTP {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001
            raise SoundCloudAPIError(f"SoundCloud search failed: {exc}") from exc
        try:
            return json.loads(body).get("collection") or []
        except (ValueError, AttributeError) as exc:
            raise SoundCloudAPIError(f"SoundCloud returned unparseable JSON: {exc}") from exc
    return []


def _entry_for_scoring(hit: dict) -> dict:
    """Adapt a v2 search hit to the {title, uploader, duration(secs)} shape that
    ingest.soundcloud._search_score expects (v2 duration is milliseconds)."""
    return {
        "title": hit.get("title") or "",
        "uploader": (hit.get("user") or {}).get("username") or "",
        "duration": (hit.get("duration") or 0) / 1000.0,
    }


def find_track(artist: str, title: str, query: str | None = None, *,
               limit: int = 8, _get=None) -> dict | None:
    """Search SoundCloud and return the best match for a mix entry, or None.

    ``query`` is the search string (default "Artist - Title"); ``artist``/``title``
    drive the confidence score. Returns
    {url, title, uploader, duration_secs, score, playback_count} for the best hit —
    ties on score broken by play count so the official upload wins over re-uploads.
    """
    q = (query or " - ".join(p for p in (artist.strip(), title.strip()) if p)).strip()
    if not q:
        return None
    hits = _search(q, limit, _get=_get)
    best = None
    best_key = (-2.0, -1)
    for h in hits:
        if not isinstance(h, dict) or not h.get("permalink_url"):
            continue
        score = _search_score(artist, title, _entry_for_scoring(h))
        key = (round(score, 3), h.get("playback_count") or 0)
        if key > best_key:
            best_key, best = key, h
    if best is None:
        return None
    return {
        "url": best.get("permalink_url") or "",
        "title": best.get("title") or "",
        "uploader": (best.get("user") or {}).get("username") or "",
        "duration_secs": float((best.get("duration") or 0) / 1000.0),
        "score": round(best_key[0], 3),
        "playback_count": best.get("playback_count") or 0,
    }
