"""
stems/separate.py — Split a track into vocals + instrumental.

Two engines, chosen by the `stem_separator` setting (config.current_stem_separator):
  * "demucs" — htdemucs subprocess. Best quality; slow on CPU.
  * "mdx"    — audio-separator (UVR MDX-Net, ONNX). ~2-4x faster on CPU,
               slightly lower quality. First run downloads the model (~50-100MB).

Output files:
  audio/vocals/         {title}_{artist}_vocals.wav
  audio/instrumentals/  {title}_{artist}_instrumental.wav

The returned dict includes a "separator" provenance tag ("demucs:htdemucs" /
"mdx:<model>") that callers store on the stems rows, or None when existing
stems were reused untouched (keep whatever tag the DB already has).
"""
from typing import Callable, Optional, Dict
import re
import subprocess
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from config import (
    DATA_DIR, DEMUCS_MODEL, INSTRUMENTALS_DIR, MDX_MODEL, VOCALS_DIR,
    current_stem_separator, sanitize_filename_chars,
)

log = logging.getLogger(__name__)

# Optional progress callback. percent is None for status-only updates
# (so the bar holds its last value while the message ticker keeps moving).
ProgressCb = Optional[Callable[[Optional[int], str], None]]

# Where audio-separator caches downloaded UVR models (persists across runs so
# the ~50-100MB download happens once).
MDX_MODEL_DIR = DATA_DIR / "uvr_models"


def separator_tag(separator: Optional[str] = None) -> str:
    """Provenance tag for the given (or currently configured) engine."""
    sep = separator or current_stem_separator()
    if sep == "mdx":
        return f"mdx:{Path(MDX_MODEL).stem}"
    return f"demucs:{DEMUCS_MODEL}"


def separate(song_id: int, title: str, audio_path: Path,
             artist: str = "",
             on_progress: ProgressCb = None,
             separator: Optional[str] = None,
             force: bool = False) -> Optional[Dict]:
    separator = separator or current_stem_separator()
    safe_title  = sanitize_filename_chars(title)[:40]
    safe_artist = sanitize_filename_chars(artist)[:30]
    safe_name   = f"{safe_title}_{safe_artist}"

    vocals_path       = VOCALS_DIR        / f"{safe_name}_vocals.wav"
    instrumental_path = INSTRUMENTALS_DIR / f"{safe_name}_instrumental.wav"

    if not force and vocals_path.exists() and instrumental_path.exists():
        log.info(f"Stems already exist for: {title}")
        if on_progress:
            on_progress(100, "Stems already on disk")
        # separator=None → reused as-is; caller keeps the DB's existing tag.
        return {"vocals": vocals_path, "instrumental": instrumental_path,
                "separator": None}

    VOCALS_DIR.mkdir(parents=True, exist_ok=True)
    INSTRUMENTALS_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.gettempdir()) / f"mashup_tmp_{song_id:04d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if separator == "mdx":
        result = _run_mdx(audio_path, tmp_dir, vocals_path, instrumental_path, on_progress)
    else:
        result = _run_demucs(audio_path, tmp_dir, vocals_path, instrumental_path, on_progress)
    if result is not None:
        result["separator"] = separator_tag(separator)
    return result


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


_MDX_PCT = re.compile(r"(\d{1,3})%\|")   # tqdm lines: " 42%|████     | …"


def _run_mdx(audio_path: Path, tmp_dir: Path,
             vocals_path: Path, instrumental_path: Path,
             on_progress: ProgressCb = None) -> Optional[Dict]:
    """Fast separation via audio-separator (UVR MDX-Net ONNX) as a subprocess —
    same isolation/progress pattern as Demucs. The package has no __main__, so
    its CLI entry function is invoked through `python -c`."""
    from api.workers._progress import stream_subprocess
    log.info(f"Running MDX-Net ({MDX_MODEL}) on: {audio_path.name}")

    MDX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-c", "from audio_separator.utils.cli import main; main()",
        str(audio_path),
        "-m", MDX_MODEL,
        "--output_dir", str(tmp_dir),
        "--model_file_dir", str(MDX_MODEL_DIR),
        "--output_format", "WAV",
    ]

    if on_progress:
        on_progress(0, "Loading MDX-Net model (first run downloads ~75MB)…")

    def _on_line(line: str) -> None:
        if not on_progress:
            return
        m = _MDX_PCT.search(line)
        if m:
            pct = min(100, int(m.group(1)))
            on_progress(pct, f"Separating: {pct}%")
        elif line.strip():
            on_progress(None, line.strip()[:120])

    try:
        result = stream_subprocess(cmd, _on_line, timeout=1800)
        if result.returncode != 0:
            log.error(f"audio-separator failed: {result.stdout[-500:]}")
            return None
    except FileNotFoundError:
        log.error("audio-separator not found. Install with: pip install audio-separator")
        return None
    except subprocess.TimeoutExpired:
        log.error("audio-separator timed out (>30 min)")
        return None

    # Output names look like "{input}_(Vocals)_{model}.wav" — match by marker.
    raw_vocals = next(iter(tmp_dir.glob("*(Vocals)*.wav")), None)
    raw_inst   = next(iter(tmp_dir.glob("*(Instrumental)*.wav")), None)
    if not raw_vocals or not raw_inst:
        log.error(f"Expected MDX output not found in {tmp_dir}: "
                  f"{[p.name for p in tmp_dir.glob('*')]}")
        return None

    shutil.move(str(raw_vocals), str(vocals_path))
    shutil.move(str(raw_inst), str(instrumental_path))
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log.info(f"Stems ready: {vocals_path.name}, {instrumental_path.name}")
    return {"vocals": vocals_path, "instrumental": instrumental_path}
