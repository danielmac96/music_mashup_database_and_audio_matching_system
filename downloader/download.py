"""
downloader/download.py — Download audio in best quality via yt-dlp.

SoundCloud Go+ handling:
  If the downloaded file is under PREVIEW_MAX_SECS (30s preview),
  fall back to a YouTube search for the full song using the title + artist.

Output: MP3 in config.RAW_DIR / "{title}_{artist}.mp3"
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple
import subprocess
import logging
import re
from pathlib import Path

from config import RAW_DIR, YTDLP_FORMAT, YTDLP_FORMAT_FALLBACK, YTDLP_POSTARGS

log = logging.getLogger(__name__)

# Files shorter than this are considered previews and trigger the YT fallback
PREVIEW_MAX_SECS = 35

# How many YouTube search results to try per query (1 = top hit, …, N = Nth hit)
YOUTUBE_SEARCH_MAX_RESULTS = 5


class DownloadResult(NamedTuple):
    """download_track return value. duration_secs is set when audio came from YouTube fallback."""
    path: Path
    duration_secs: Optional[float] = None  # if set, persist to songs.duration_secs


class _YtAttempt(NamedTuple):
    label: str
    format_str: str
    player_client: str
    use_cookies: bool


def _youtube_attempts() -> tuple[_YtAttempt, ...]:
    return (
        _YtAttempt("ios+bestaudio", YTDLP_FORMAT, "youtube:player_client=ios", False),
        _YtAttempt(
            "android_web+bestaudio",
            YTDLP_FORMAT,
            "youtube:player_client=android,web",
            False,
        ),
        _YtAttempt(
            "android_web+ba/b",
            YTDLP_FORMAT_FALLBACK,
            "youtube:player_client=android,web",
            False,
        ),
        _YtAttempt(
            "cookies+android_web+ba/b",
            YTDLP_FORMAT_FALLBACK,
            "youtube:player_client=android,web",
            True,
        ),
    )


def download_track(song_id: int, title: str, source_url: str,
                   artist: str = "") -> Optional[DownloadResult]:
    out_path = RAW_DIR / f"{_safe(title)}_{_safe(artist)}.mp3"

    if out_path.exists():
        duration = _get_duration(out_path)
        if duration and duration > PREVIEW_MAX_SECS:
            log.info(f"Already downloaded (full): {out_path.name}")
            return DownloadResult(out_path)
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
            fb = _fallback_youtube(title, artist, out_path)
            if fb:
                yt_path, yt_secs = fb
                return DownloadResult(yt_path, yt_secs)
            return None

    if path and path.exists():
        return DownloadResult(path)
    return None


# ── yt-dlp download ───────────────────────────────────────────────────────────


def _is_youtube_like(url: str) -> bool:
    u = url.lower().strip()
    if u.startswith("ytsearch"):
        return True
    return (
        "youtube.com/" in u
        or "youtu.be/" in u
        or "music.youtube.com" in u
    )


def _cleanup_stem_outputs(out_path: Path) -> None:
    pattern = f"{out_path.stem}.*"
    for p in out_path.parent.glob(pattern):
        try:
            p.unlink()
        except OSError:
            pass


def _run_ytdlp(
    url: str,
    out_path: Path,
    format_str: str,
    *,
    extractor_args: Optional[str] = None,
    use_cookies: bool,
    playlist_item: Optional[int] = None,
) -> bool:
    """
    Run yt-dlp once. Returns True if process exited 0 and an output file exists.
    """
    tmp_template = str(out_path.with_suffix("")) + ".%(ext)s"
    cmd: list[str] = [
        "yt-dlp",
        "-f", format_str,
        "--output", tmp_template,
        "--no-warnings",
        *YTDLP_POSTARGS,
    ]
    if playlist_item is None:
        cmd.append("--no-playlist")
    else:
        cmd.extend(["--playlist-items", str(playlist_item)])
    if use_cookies:
        cmd.extend(["--cookies-from-browser", "chrome"])
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    cmd.append(url)

    log.info(f"Downloading: {url}" + (f" (item {playlist_item})" if playlist_item else ""))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        log.error("yt-dlp not found. Install with: pip install yt-dlp")
        return False
    except subprocess.TimeoutExpired:
        log.error(f"Download timed out: {url}")
        return False

    if result.returncode != 0:
        log.warning(f"yt-dlp attempt failed [{result.returncode}]: {result.stderr[:400]}")
        return False

    if out_path.exists():
        return True

    for candidate in out_path.parent.glob(f"{out_path.stem}.*"):
        candidate.rename(out_path)
        return True

    log.warning("yt-dlp exited 0 but output file not found")
    return False


def _download_ytdlp(
    url: str,
    out_path: Path,
    *,
    playlist_item: Optional[int] = None,
) -> Optional[Path]:
    """
    Download with yt-dlp. YouTube / ytsearch URLs use a retry ladder;
    other sites (e.g. SoundCloud) use a single plain invocation.
    """
    _cleanup_stem_outputs(out_path)

    if not _is_youtube_like(url):
        ok = _run_ytdlp(
            url,
            out_path,
            YTDLP_FORMAT,
            extractor_args=None,
            use_cookies=False,
            playlist_item=playlist_item,
        )
        if ok and out_path.exists():
            log.info(f"Downloaded: {out_path.name}")
            return out_path
        log.error("yt-dlp failed for non-YouTube URL after 1 attempt")
        return None

    for att in _youtube_attempts():
        _cleanup_stem_outputs(out_path)
        ok = _run_ytdlp(
            url,
            out_path,
            att.format_str,
            extractor_args=att.player_client,
            use_cookies=att.use_cookies,
            playlist_item=playlist_item,
        )
        if ok and out_path.exists():
            log.info(f"Downloaded ({att.label}): {out_path.name}")
            return out_path

    log.error(f"yt-dlp failed for URL after all YouTube retries: {url[:120]}")
    return None


# ── YouTube fallback ──────────────────────────────────────────────────────────


def _fallback_youtube(title: str, artist: str, out_path: Path) -> Optional[Tuple[Path, float]]:
    """
    Search YouTube for the full track using multiple query strategies.
    Strips parenthetical suffixes from title for cleaner search results.
    Uses ytsearchN and walks top results so one bad hit does not sink the track.
    """
    clean_title = re.sub(r'\s*[\(\[].*?[\)\]]', '', title).strip()
    n = YOUTUBE_SEARCH_MAX_RESULTS
    queries = [
        f"ytsearch{n}:{artist} {clean_title} official audio",
        f"ytsearch{n}:{artist} {clean_title} lyrics",
        f"ytsearch{n}:{artist} {clean_title}",
    ]

    for query in queries:
        for rank in range(1, n + 1):
            log.info(f"YouTube search: {query}  [trying result #{rank}]")
            path = _download_ytdlp(query, out_path, playlist_item=rank)
            if path and path.exists():
                duration = _get_duration(path)
                if duration and duration > PREVIEW_MAX_SECS:
                    log.info(f"YouTube fallback succeeded ({duration:.0f}s): {out_path.name}")
                    return path, duration
                log.warning(
                    f"YouTube result #{rank} too short ({duration or 0:.0f}s), trying next"
                )
                if path.exists():
                    path.unlink()
            _cleanup_stem_outputs(out_path)

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