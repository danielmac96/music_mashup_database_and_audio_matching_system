"""
downloader/download.py — Download audio in best quality via yt-dlp.
Output: MP3 in config.RAW_DIR / "{title}_{artist}.mp3"
"""
from typing import Optional
import subprocess
import logging
import re
from pathlib import Path

from config import RAW_DIR, YTDLP_FORMAT, YTDLP_POSTARGS

log = logging.getLogger(__name__)


def download_track(song_id: int, title: str, source_url: str,
                   artist: str = "") -> Optional[Path]:
    out_path = RAW_DIR / f"{_safe(title)}_{_safe(artist)}.mp3"

    if out_path.exists():
        log.info(f"Already downloaded: {out_path.name}")
        return out_path

    return _download_ytdlp(source_url, out_path)


def _download_ytdlp(url: str, out_path: Path) -> Optional[Path]:
    tmp_template = str(out_path.with_suffix("")) + ".%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", YTDLP_FORMAT,
        "--output", tmp_template,
        "--no-playlist",
        "--no-warnings",
        *YTDLP_POSTARGS,
        url,
    ]
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


def _safe(name: str, max_len: int = 40) -> str:
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[\s-]+', '_', name)
    return name[:max_len]