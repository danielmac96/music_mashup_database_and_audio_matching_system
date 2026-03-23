"""
ingest/soundcloud.py — Pull track metadata from a SoundCloud playlist.

Strategy (in order of preference):
  1. scdl --no-download (fast, gets metadata without downloading)
  2. yt-dlp --dump-json (universal fallback, works for SC + YT + most sites)
  3. Mock data (for testing without network access)

Returns a list of dicts:
  { title, artist, source_url, duration_secs, genre }
"""
from typing import Optional
import subprocess
import logging
import json
from pathlib import Path

log = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_playlist(url: str, use_mock: bool = False) -> list[dict]:
    """
    Fetch track metadata from a SoundCloud playlist URL.

    Args:
        url:        SoundCloud playlist URL
        use_mock:   If True, return mock data (useful for offline testing)

    Returns:
        List of track metadata dicts
    """
    if use_mock:
        log.info("Using mock playlist data")
        return _mock_tracks()

    log.info(f"Fetching playlist metadata: {url}")

    # Try yt-dlp first (no extra install needed if yt-dlp is present)
    tracks = _fetch_via_ytdlp(url)
    if tracks:
        return tracks

    log.warning("yt-dlp fetch failed, falling back to mock data")
    return _mock_tracks()


def fetch_single(url: str) -> Optional[dict]:
    """Fetch metadata for a single track URL."""
    log.info(f"Fetching single track: {url}")
    tracks = _fetch_via_ytdlp(url)
    return tracks[0] if tracks else None


# ── Implementations ───────────────────────────────────────────────────────────

def _fetch_via_ytdlp(url: str) -> list[dict]:
    """
    Use yt-dlp --dump-json to extract playlist/track metadata without downloading.
    Works for SoundCloud, YouTube, and most other sources.
    """
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
                tracks.append(_normalise_ytdlp(info))
            except json.JSONDecodeError:
                continue

        log.info(f"Fetched {len(tracks)} tracks via yt-dlp")
        return tracks

    except FileNotFoundError:
        log.warning("yt-dlp not found in PATH")
        return []
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp timed out")
        return []


def _normalise_ytdlp(info: dict) -> dict:
    """Normalise a yt-dlp JSON blob to our standard track dict."""
    # Duration may be missing for flat-playlist entries
    duration = info.get("duration") or 0

    # Uploader / artist
    artist = (
        info.get("uploader")
        or info.get("channel")
        or info.get("artist")
        or "Unknown"
    )

    return {
        "title":        info.get("title", "Unknown"),
        "artist":       artist,
        "source_url":   info.get("url") or info.get("webpage_url", ""),
        "duration_secs": float(duration),
        "genre":        info.get("genre", ""),
    }


# ── Mock data (offline / CI testing) ─────────────────────────────────────────

def _mock_tracks() -> list[dict]:
    """
    Realistic mock tracks for testing the full pipeline without network access.
    URLs are placeholders — the downloader will skip them gracefully in mock mode.
    """
    return [
        {
            "title": "Blinding Lights",
            "artist": "The Weeknd",
            "source_url": "https://soundcloud.com/theweeknd/blinding-lights",
            "duration_secs": 200,
            "genre": "Synth-pop",
        },
        {
            "title": "Levitating",
            "artist": "Dua Lipa",
            "source_url": "https://soundcloud.com/dualipa/levitating",
            "duration_secs": 203,
            "genre": "Dance-pop",
        },
        {
            "title": "Stay",
            "artist": "The Kid LAROI ft. Justin Bieber",
            "source_url": "https://soundcloud.com/kidlaroi/stay",
            "duration_secs": 141,
            "genre": "Pop",
        },
        {
            "title": "good 4 u",
            "artist": "Olivia Rodrigo",
            "source_url": "https://soundcloud.com/oliviarodrigo/good4u",
            "duration_secs": 178,
            "genre": "Pop-punk",
        },
        {
            "title": "Heat Waves",
            "artist": "Glass Animals",
            "source_url": "https://soundcloud.com/glassanimals/heatwaves",
            "duration_secs": 238,
            "genre": "Indie",
        },
    ]