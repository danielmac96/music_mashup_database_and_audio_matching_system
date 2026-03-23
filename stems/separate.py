"""
stems/separate.py — Split a track into vocals + instrumental using Demucs.

Demucs (by Meta) is the current state-of-the-art for music source separation.
Model choice (config.DEMUCS_MODEL):
  - "htdemucs"   → best quality, ~4GB VRAM or slow on CPU
  - "mdx_extra"  → faster, still excellent quality
  - "htdemucs_ft" → fine-tuned version, best for vocals

Output directory structure (Demucs default):
  audio/stems/{model}/{song_stem}/{vocals,drums,bass,other}.wav

We rename/copy to:
  audio/stems/{song_id}_{title}/vocals.wav
  audio/stems/{song_id}_{title}/instrumental.wav   (no_vocals mix)
"""
from typing import Optional, Dict
import subprocess
import logging
import shutil
from pathlib import Path

from config import VOCALS_DIR, INSTRUMENTALS_DIR, DEMUCS_MODEL

log = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def separate(song_id: int, title: str, audio_path: Path,
             artist: str = "", use_mock: bool = False) -> Optional[Dict[str, Path]]:
    """
    Separate a track into vocals and instrumental stems.

    Args:
        song_id:    DB song id
        title:      Track title (for filename)
        artist:     Artist name (for filename)
        audio_path: Path to the source MP3/WAV
        use_mock:   Copy the source file as both stems (for testing)

    Returns:
        { "vocals": Path, "instrumental": Path }  or None on failure
    """
    from re import sub
    safe_title  = sub(r'[^\w]', '_', title)[:40]
    safe_artist = sub(r'[^\w]', '_', artist)[:30]
    safe_name   = f"{safe_title}_{safe_artist}"

    vocals_path       = VOCALS_DIR       / f"{safe_name}_vocals.wav"
    instrumental_path = INSTRUMENTALS_DIR / f"{safe_name}_instrumental.wav"
    tmp_dir           = VOCALS_DIR / f"_tmp_{song_id:04d}"

    if vocals_path.exists() and instrumental_path.exists():
        log.info(f"Stems already exist for: {title}")
        return {"vocals": vocals_path, "instrumental": instrumental_path}

    VOCALS_DIR.mkdir(parents=True, exist_ok=True)
    INSTRUMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if use_mock:
        return _mock_separate(audio_path, vocals_path, instrumental_path)

    return _run_demucs(audio_path, tmp_dir, vocals_path, instrumental_path)


# ── Demucs separation ─────────────────────────────────────────────────────────

def _run_demucs(audio_path: Path, tmp_dir: Path,
                vocals_path: Path, instrumental_path: Path) -> Optional[Dict]:
    """
    Run Demucs CLI. Outputs into tmp_dir, then moves final stems to STEMS_DIR
    with the clean title_artist_stem.wav naming.
    """
    import sys
    log.info(f"Running Demucs ({DEMUCS_MODEL}) on: {audio_path.name}")

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", DEMUCS_MODEL,
        "--out", str(tmp_dir),
        str(audio_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            log.error(f"Demucs failed: {result.stderr[-500:]}")
            return None
    except FileNotFoundError:
        log.error("Demucs not found. Install with: pip install demucs")
        return None
    except subprocess.TimeoutExpired:
        log.error("Demucs timed out (>30 min)")
        return None

    # Locate Demucs output: tmp_dir/DEMUCS_MODEL/audio_stem_name/vocals.wav
    stem_name   = audio_path.stem
    demucs_out  = tmp_dir / DEMUCS_MODEL / stem_name
    raw_vocals  = demucs_out / "vocals.wav"
    raw_no_vox  = demucs_out / "no_vocals.wav"

    if not raw_vocals.exists() or not raw_no_vox.exists():
        log.error(f"Expected demucs output not found in {demucs_out}")
        return None

    shutil.move(str(raw_vocals),  str(vocals_path))
    shutil.move(str(raw_no_vox),  str(instrumental_path))

    # Clean up the tmp working directory
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log.info(f"Stems ready: {vocals_path.name}, {instrumental_path.name}")
    return {"vocals": vocals_path, "instrumental": instrumental_path}


# ── Mock separation ───────────────────────────────────────────────────────────

def _mock_separate(source: Path, vocals_path: Path,
                   instrumental_path: Path) -> dict[str, Path]:
    """
    For testing: copy the source file as both stems.
    Analysis will still run; results won't differ between stems but
    the whole pipeline can be tested end-to-end.
    """
    log.info(f"Mock stem separation: {source.name}")
    shutil.copy2(str(source), str(vocals_path))
    shutil.copy2(str(source), str(instrumental_path))
    return {"vocals": vocals_path, "instrumental": instrumental_path}