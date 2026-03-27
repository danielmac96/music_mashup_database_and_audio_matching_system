"""
ingest/soundcloud.py — Pull track metadata from a SoundCloud playlist via yt-dlp.
"""
import json
import subprocess
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


def fetch_playlist(url: str) -> list:
    """
    Fetch track metadata from a SoundCloud playlist URL.
    Each item includes title, artist, source_url, duration_secs, genre,
    artist_id, track_id, duration_str, upload_date, likes, reposts,
    comments, plays, thumbnail (when yt-dlp provides them).
    """
    log.info(f"Fetching playlist metadata: {url}")
    tracks = _fetch_via_ytdlp(url)
    if not tracks:
        log.error("No tracks found. Check the playlist URL.")
    return tracks


def fetch_single(url: str) -> Optional[dict]:
    """Fetch metadata for a single track URL."""
    log.info(f"Fetching single track: {url}")
    tracks = _fetch_via_ytdlp(url)
    return tracks[0] if tracks else None


def _fetch_via_ytdlp(url: str) -> list:
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-warnings", url],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.warning(f"yt-dlp exited {result.returncode}: {result.stderr[:200]}")
            return []

        tracks = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                info = json.loads(line)
                if info.get("_type") == "playlist" and info.get("entries"):
                    for entry in info["entries"]:
                        if entry is None or not isinstance(entry, dict):
                            continue
                        tracks.append(_normalise(entry))
                else:
                    tracks.append(_normalise(info))
            except json.JSONDecodeError:
                continue

        log.info(f"Fetched {len(tracks)} tracks via yt-dlp")
        return tracks

    except FileNotFoundError:
        log.error("yt-dlp not found. Install with: pip install yt-dlp")
        return []
    except subprocess.TimeoutExpired:
        log.error("yt-dlp timed out")
        return []


def _str_or_empty(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


def _int_or_zero(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _format_duration_str(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return ""
    s = int(round(seconds))
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _thumbnail_url(info: dict) -> str:
    t = info.get("thumbnail")
    if isinstance(t, str) and t.strip():
        return t.strip()
    thumbs = info.get("thumbnails") or []
    for entry in reversed(thumbs):
        if isinstance(entry, dict):
            u = entry.get("url")
            if u:
                return str(u)
        elif isinstance(entry, str) and entry:
            return entry
    return ""


def _normalise(info: dict) -> dict:
    artist = (
        info.get("uploader")
        or info.get("channel")
        or info.get("artist")
        or "Unknown"
    )
    raw_duration = info.get("duration")
    duration_f = float(raw_duration) if raw_duration is not None else 0.0
    webpage = info.get("webpage_url") or info.get("url") or ""

    return {
        "title": info.get("title") or "Unknown",
        "artist": artist,
        "artist_id": _str_or_empty(info.get("uploader_id")),
        "track_id": _str_or_empty(info.get("id")),
        "duration_secs": duration_f,
        "duration_str": _format_duration_str(duration_f),
        "source_url": webpage,
        "upload_date": _str_or_empty(info.get("upload_date")),
        "likes": _int_or_zero(info.get("like_count")),
        "reposts": _int_or_zero(info.get("repost_count")),
        "comments": _int_or_zero(info.get("comment_count")),
        "plays": _int_or_zero(info.get("view_count")),
        "thumbnail": _thumbnail_url(info),
        "genre": _normalise_genre(info.get("genre")),
    }


def _normalise_genre(g: Any) -> str:
    if g is None:
        return ""
    if isinstance(g, str):
        return g.strip()
    return str(g).strip()
