"""
downloader/download.py — Download audio in best quality.

Uses yt-dlp which supports SoundCloud, YouTube, Bandcamp, and 1000+ other sites.
Falls back to mock (copy a test sine-wave file) when network is unavailable.

Output: 320k MP3 in config.RAW_DIR / "{song_id}_{safe_title}.mp3"
"""
from typing import Optional
import subprocess
import logging
import re
import shutil
from pathlib import Path

from config import RAW_DIR, YTDLP_FORMAT, YTDLP_POSTARGS

log = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def download_track(song_id: int, title: str, source_url: str,
                   artist: str = "", use_mock: bool = False) -> Optional[Path]:
    """
    Download a single track to RAW_DIR.

    Args:
        song_id:    DB song id
        title:      Track title (used in filename)
        artist:     Artist name (used in filename)
        source_url: URL to download from
        use_mock:   If True, generate a synthetic audio file instead

    Returns:
        Path to the downloaded file, or None on failure
    """
    out_path = RAW_DIR / f"{_safe(title)}_{_safe(artist)}.mp3"

    if out_path.exists():
        log.info(f"Already downloaded: {out_path.name}")
        return out_path

    if use_mock:
        return _generate_mock_audio(out_path, f"{title}_{artist}")

    return _download_ytdlp(source_url, out_path)


# ── yt-dlp download ───────────────────────────────────────────────────────────

def _download_ytdlp(url: str, out_path: Path) -> Optional[Path]:
    """
    Run yt-dlp to download + convert to MP3.
    Uses a temp filename then renames to avoid partial files.
    """
    # yt-dlp accepts a template; we fix the output name ourselves
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

        # yt-dlp names the file with .mp3 due to --audio-format mp3
        if out_path.exists():
            log.info(f"Downloaded: {out_path.name}")
            return out_path

        # Handle any extension mismatch
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


# ── Mock audio generator ──────────────────────────────────────────────────────

def _generate_mock_audio(out_path: Path, title: str) -> Path:
    """
    Generate a short synthetic MP3 using ffmpeg (sine tones at different
    frequencies per track for distinguishable mock analysis results).
    Falls back to writing a minimal valid MP3 header if ffmpeg is absent.
    """
    # Use title hash to vary the tone so each mock track analyses differently
    freq = 220 + (hash(title) % 20) * 20   # 220–600 Hz range

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration=10",
            "-acodec", "libmp3lame", "-b:a", "128k",
            str(out_path)
        ], capture_output=True, timeout=30)

        if result.returncode == 0:
            log.info(f"Generated mock audio ({freq}Hz): {out_path.name}")
            return out_path
    except FileNotFoundError:
        pass

    # Last resort: write a silent minimal valid WAV (librosa can read it)
    _write_silent_wav(out_path.with_suffix(".wav"))
    log.warning(f"ffmpeg unavailable — wrote silent WAV: {out_path.stem}.wav")
    return out_path.with_suffix(".wav")


def _write_silent_wav(path: Path, duration_secs: int = 5,
                      sample_rate: int = 22050):
    """Write a minimal PCM WAV file (silence) without any dependencies."""
    import struct
    n_samples = duration_secs * sample_rate
    data_size = n_samples * 2  # 16-bit mono

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH",
            16,          # chunk size
            1,           # PCM
            1,           # mono
            sample_rate,
            sample_rate * 2,  # byte rate
            2,           # block align
            16           # bits per sample
        ))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(name: str, max_len: int = 40) -> str:
    """Sanitise a string for use in a filename."""
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[\s-]+', '_', name)
    return name[:max_len]