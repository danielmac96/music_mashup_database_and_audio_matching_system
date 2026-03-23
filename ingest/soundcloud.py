"""
ingest/soundcloud.py — Pull track metadata from a SoundCloud playlist via yt-dlp.
"""
import json
import subprocess
import logging
from typing import Optional

log = logging.getLogger(__name__)


def fetch_playlist(url: str) -> list:
    """
    Fetch track metadata from a SoundCloud playlist URL.
    Returns a list of dicts: { title, artist, source_url, duration_secs, genre }
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
            capture_output=True, text=True, timeout=60
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


def _normalise(info: dict) -> dict:
    artist = (
        info.get("uploader")
        or info.get("channel")
        or info.get("artist")
        or "Unknown"
    )
    return {
        "title":         info.get("title", "Unknown"),
        "artist":        artist,
        "source_url":    info.get("url") or info.get("webpage_url", ""),
        "duration_secs": float(info.get("duration") or 0),
        "genre":         info.get("genre", ""),
    }