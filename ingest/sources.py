"""ingest/sources.py — classify a pasted URL by source site and link kind.

Single shared helper so the importer preview endpoint and the ingest endpoint
agree on what a link is. Mirrors frontend/src/sources.js (classifyUrl) — keep
the two in sync when adding a source.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse


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
