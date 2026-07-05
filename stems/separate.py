"""
stems/separate.py — Split a track into vocals + instrumental using Demucs.

Output files:
  audio/vocals/         {title}_{artist}_vocals.wav
  audio/instrumentals/  {title}_{artist}_instrumental.wav
"""
from typing import Callable, Optional, Dict
import subprocess
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from config import VOCALS_DIR, INSTRUMENTALS_DIR, DEMUCS_MODEL, sanitize_filename_chars

log = logging.getLogger(__name__)

# Optional progress callback. percent is None for status-only updates
# (so the bar holds its last value while the message ticker keeps moving).
ProgressCb = Optional[Callable[[Optional[int], str], None]]


def separate(song_id: int, title: str, audio_path: Path,
             artist: str = "",
             on_progress: ProgressCb = None) -> Optional[Dict[str, Path]]:
    safe_title  = sanitize_filename_chars(title)[:40]
    safe_artist = sanitize_filename_chars(artist)[:30]
    safe_name   = f"{safe_title}_{safe_artist}"

    vocals_path       = VOCALS_DIR        / f"{safe_name}_vocals.wav"
    instrumental_path = INSTRUMENTALS_DIR / f"{safe_name}_instrumental.wav"

    if vocals_path.exists() and instrumental_path.exists():
        log.info(f"Stems already exist for: {title}")
        if on_progress:
            on_progress(100, "Stems already on disk")
        return {"vocals": vocals_path, "instrumental": instrumental_path}

    VOCALS_DIR.mkdir(parents=True, exist_ok=True)
    INSTRUMENTALS_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.gettempdir()) / f"mashup_tmp_{song_id:04d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    return _run_demucs(audio_path, tmp_dir, vocals_path, instrumental_path, on_progress)


def _run_demucs(audio_path: Path, tmp_dir: Path,
                vocals_path: Path, instrumental_path: Path,
                on_progress: ProgressCb = None) -> Optional[Dict]:
    from api.workers._progress import parse_demucs, stream_subprocess
    log.info(f"Running Demucs ({DEMUCS_MODEL}) on: {audio_path.name}")

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", DEMUCS_MODEL,
        "--out", str(tmp_dir),
        str(audio_path),
    ]

    if on_progress:
        on_progress(0, "Loading Demucs model (first run downloads ~400MB)…")

    def _on_line(line: str) -> None:
        if not on_progress:
            return
        pct = parse_demucs(line)
        if pct is not None:
            on_progress(pct, f"Separating: {pct}%")
        elif line.strip():
            on_progress(None, line.strip()[:120])

    try:
        result = stream_subprocess(cmd, _on_line, timeout=1800)
        if result.returncode != 0:
            log.error(f"Demucs failed: {result.stdout[-500:]}")
            return None
    except FileNotFoundError:
        log.error("Demucs not found. Install with: pip install demucs")
        return None
    except subprocess.TimeoutExpired:
        log.error("Demucs timed out (>30 min)")
        return None

    stem_name  = audio_path.stem
    demucs_out = tmp_dir / DEMUCS_MODEL / stem_name
    raw_vocals = demucs_out / "vocals.wav"
    raw_no_vox = demucs_out / "no_vocals.wav"

    if not raw_vocals.exists() or not raw_no_vox.exists():
        log.error(f"Expected demucs output not found in {demucs_out}")
        return None

    shutil.move(str(raw_vocals), str(vocals_path))
    shutil.move(str(raw_no_vox), str(instrumental_path))
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log.info(f"Stems ready: {vocals_path.name}, {instrumental_path.name}")
    return {"vocals": vocals_path, "instrumental": instrumental_path}