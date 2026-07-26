"""ingest/sources.py — classify a pasted URL by source site and link kind.

Single shared helper so the importer preview endpoint and the ingest endpoint
agree on what a link is. Mirrors frontend/src/sources.js (classifyUrl) — keep
the two in sync when adding a source.
"""
from __future__ import annotations

from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

# Query params that carry no identity — just analytics/sharing context. Stripped
# before comparing/storing a URL so trivial variants of the same link dedup.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "si", "ref", "ref_", "fbclid", "gclid", "feature", "pp", "ab_channel",
    "spm", "igshid", "t", "start", "time_continue", "index",
}


def classify_url(url: str) -> tuple[str, str]:
    """Return (source, kind) for a pasted link.

    source: 'soundcloud' | 'youtube' | 'unknown'
    kind:   'track' | 'playlist'  (kind is meaningless when source is unknown)
    """
    raw = (url or "").strip()
    if not raw:
        return "unknown", "track"
    if "://" not in raw:
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
    except ValueError:
        return "unknown", "track"
    host = (parsed.netloc or "").lower().removeprefix("www.").removeprefix("m.")
    path = parsed.path or ""
    query = parse_qs(parsed.query or "")

    if host in ("soundcloud.com", "on.soundcloud.com", "api.soundcloud.com"):
        # Playlist/album links contain /sets/; a "discover/sets/…" page is a
        # station, which yt-dlp also enumerates like a playlist.
        kind = "playlist" if "/sets/" in path else "track"
        return "soundcloud", kind

    if host in ("youtube.com", "music.youtube.com", "youtu.be"):
        # Any list= parameter (or an explicit /playlist path) is a playlist —
        # even watch?v=…&list=… enumerates the whole list on ingest.
        kind = "playlist" if ("list" in query or path.startswith("/playlist")) else "track"
        return "youtube", kind

    return "unknown", "track"


def normalize_url(url: str) -> str:
    """Canonicalize a link for dedup + storage: force https, lowercase host,
    drop www./m. and the fragment, strip tracking params, and sort the rest.

    SoundCloud track identity lives entirely in the path, so its query is
    dropped (keeping only ?secret_token= for private tracks). Returns the
    original (trimmed) string if it can't be parsed."""
    raw = (url or "").strip()
    if not raw:
        return ""
    had_scheme = "://" in raw
    if not had_scheme:
        raw = "https://" + raw
    try:
        parsed = urlparse(raw)
    except ValueError:
        return (url or "").strip()

    scheme = "https"
    host = (parsed.netloc or "").lower().removeprefix("www.").removeprefix("m.")
    if not host:
        return (url or "").strip()
    path = (parsed.path or "").rstrip("/")

    params = [(k, v) for k, v in parse_qsl(parsed.query or "", keep_blank_values=False)
              if k.lower() not in _TRACKING_PARAMS]
    source, _ = classify_url(raw)
    if source == "soundcloud":
        params = [(k, v) for k, v in params if k.lower() == "secret_token"]
    params.sort()
    query = urlencode(params)
    return urlunparse((scheme, host, path, "", query, ""))
