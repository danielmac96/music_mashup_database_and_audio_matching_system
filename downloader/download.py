"""
downloader/download.py — Download audio in best quality via yt-dlp.

SoundCloud Go+ handling:
  If the downloaded file is under PREVIEW_MAX_SECS (30s preview),
  fall back to a YouTube search for the full song using the title + artist.

Output: MP3 in config.RAW_DIR / "{title}_{artist}.mp3"
"""
from typing import Optional
import subprocess
import logging
import re
from pathlib import Path

from config import RAW_DIR, YTDLP_FORMAT, YTDLP_POSTARGS

log = logging.getLogger(__name__)

# Files shorter than this are considered previews and trigger the YT fallback
PREVIEW_MAX_SECS = 35


def download_track(song_id: int, title: str, source_url: str,
                   artist: str = "") -> Optional[Path]:
    out_path = RAW_DIR / f"{_safe(title)}_{_safe(artist)}.mp3"

    if out_path.exists():
        duration = _get_duration(out_path)
        if duration and duration > PREVIEW_MAX_SECS:
            log.info(f"Already downloaded (full): {out_path.name}")
            return out_path
        else:
            log.warning(f"Existing file is a preview ({duration:.0f}s) — re-downloading")
            out_path.unlink()

    # Try SoundCloud URL first
    path = _download_ytdlp(source_url, out_path)

    if path and path.exists():
        duration = _get_duration(path)
        if duration and duration <= PREVIEW_MAX_SECS:
            log.warning(
                f"Downloaded file is only {duration:.0f}s — SoundCloud Go+ preview detected. "
                f"Searching YouTube for full track..."
            )
            path.unlink()
            path = _fallback_youtube(title, artist, out_path)

    return path


# ── yt-dlp download ───────────────────────────────────────────────────────────

def _download_ytdlp(url: str, out_path: Path,
                    use_cookies: bool = False) -> Optional[Path]:
    tmp_template = str(out_path.with_suffix("")) + ".%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", YTDLP_FORMAT,
        "--output", tmp_template,
        "--no-playlist",
        "--no-warnings",
        *YTDLP_POSTARGS,
    ]
    if use_cookies:
        cmd += ["--cookies-from-browser", "chrome"]
    else:
        # iOS client bypasses YouTube bot detection without cookies
        cmd += ["--extractor-args", "youtube:player_client=ios"]
    cmd.append(url)
    log.info(f"Downloading: {url}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.error(f"yt-dlp failed [{result.returncode}]: {result.stderr[:300]}")
            return None

        if out_path.exists():
            log.info(f"Downloaded: {out_path.name}")
            return out_path

        for candidate in out_path.parent.glob(f"{out_path.stem}.*"):
            candidate.rename(out_path)
            log.info(f"Downloaded (renamed): {out_path.name}")
            return out_path

        log.error("Download succeeded but output file not found")
        return None

    except FileNotFoundError:
        log.error("yt-dlp not found. Install with: pip install yt-dlp")
        return None
    except subprocess.TimeoutExpired:
        log.error(f"Download timed out: {url}")
        return None


# ── YouTube fallback ──────────────────────────────────────────────────────────

def _fallback_youtube(title: str, artist: str, out_path: Path) -> Optional[Path]:
    """
    Search YouTube for the full track using multiple query strategies.
    Strips parenthetical suffixes from title for cleaner search results.
    Falls back to --cookies-from-browser if initial attempts get 403.
    """
    # Strip parentheticals like (From "Black Panther: The Album") for cleaner search
    clean_title = re.sub(r'\s*[\(\[].*?[\)\]]', '', title).strip()

    queries = [
        f"ytsearch1:{artist} {clean_title} official audio",
        f"ytsearch1:{artist} {clean_title} lyrics",
        f"ytsearch1:{artist} {clean_title}",
    ]

    for query in queries:
        log.info(f"YouTube search: {query}")
        path = _download_ytdlp(query, out_path, use_cookies=False)
        if path and path.exists():
            duration = _get_duration(path)
            if duration and duration > PREVIEW_MAX_SECS:
                log.info(f"YouTube fallback succeeded ({duration:.0f}s): {out_path.name}")
                return path
            else:
                log.warning(f"YouTube result too short ({duration:.0f}s), trying next")
                if path.exists():
                    path.unlink()

    log.error(f"Could not find full version of '{title}' by '{artist}' on YouTube")
    return None


# ── Duration check ────────────────────────────────────────────────────────────

def _get_duration(path: Path) -> Optional[float]:
    """
    Use ffprobe to get the duration of an audio file in seconds.
    Returns None if ffprobe is unavailable or the file is unreadable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # ffprobe unavailable — fall back to file size heuristic
    # 128kbps MP3 ≈ 16KB/s, so <560KB is likely a 35s preview
    size_kb = path.stat().st_size / 1024
    if size_kb < 560:
        return 30.0   # assume preview
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(name: str, max_len: int = 40) -> str:
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[\s-]+', '_', name)
    return name[:max_len]