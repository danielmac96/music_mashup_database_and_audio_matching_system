"""
stems/separate.py — Split a track into its four htdemucs sources.

One pass of htdemucs (subprocess, CPU) writes, as FLAC under ``AUDIO_DIR/stems``:

  vocals/        {title}_{artist}_vocals.flac
  instrumentals/ {title}_{artist}_instrumental.flac   (drums + bass + other)
  drums/ bass/ other/

The vocal stem and the summed instrumental are what playback, analysis and
the matcher read; the four sources are what allin1 reads (it is trained on
htdemucs's sources, so it never separates again — analysis/allin1_runner).
Every ``stems`` row carries the provenance tag ``separator_tag()``.
"""
from typing import Callable, Optional, Dict
import subprocess
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from config import (
    DEMUCS_MODEL, DEMUCS_SOURCES, INSTRUMENTAL_SOURCES, STEM_FORMAT, STEMS_DIR,
    sanitize_filename_chars,
)

STEM_KINDS = ("vocals", "instrumental", "drums", "bass", "other")


def stem_dirs() -> Dict[str, Path]:
    return {"vocals": STEMS_DIR / "vocals", "instrumental": STEMS_DIR / "instrumentals",
            "drums": STEMS_DIR / "drums", "bass": STEMS_DIR / "bass",
            "other": STEMS_DIR / "other"}


def separator_tag() -> str:
    """Provenance stored on every stems row. A row with any other tag was made
    by an older build and is re-separated by stages.do_stems."""
    return f"demucs:{DEMUCS_MODEL}:4:unified"


log = logging.getLogger(__name__)

# Optional progress callback. percent is None for status-only updates
# (so the bar holds its last value while the message ticker keeps moving).
ProgressCb = Optional[Callable[[Optional[int], str], None]]


def thread_env() -> Dict[str, str]:
    """Cap the separator subprocess's thread pools at config.DEMUCS_THREADS.

    torch defaults to every core, which leaves nothing for the quick tier
    analysing the next downloads. The OpenMP/MKL variables are read at
    interpreter start, so they must be in the child's environment."""
    from config import DEMUCS_THREADS
    n = str(max(1, int(DEMUCS_THREADS)))
    return {"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n,
            "OPENBLAS_NUM_THREADS": n, "NUMEXPR_NUM_THREADS": n}


def separate(song_id: int, title: str, audio_path: Path,
             artist: str = "",
             on_progress: ProgressCb = None,
             force: bool = False) -> Optional[Dict]:
    """Separate ``audio_path`` into the four htdemucs sources plus the summed
    instrumental. Returns ``{vocals, instrumental, drums, bass, other: Path,
    separator: tag}``; ``separator`` is None when every file was already on
    disk and reused (the caller keeps the stored tag). None on failure."""
    safe_title  = sanitize_filename_chars(title)[:40]
    safe_artist = sanitize_filename_chars(artist)[:30]
    safe_name   = f"{safe_title}_{safe_artist}"
    dirs = stem_dirs()

    def _p(kind: str) -> Path:
        return dirs[kind] / f"{safe_name}_{kind}.{STEM_FORMAT}"

    if not force:
        have = {k: _p(k) for k in STEM_KINDS}
        if all(p.exists() for p in have.values()):
            log.info(f"Stems already exist for: {title}")
            if on_progress:
                on_progress(100, "Stems already on disk")
            return {**have, "separator": None}

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.gettempdir()) / f"mashup_tmp_{song_id:04d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result = _run_demucs_four(audio_path, tmp_dir, {k: _p(k) for k in STEM_KINDS},
                              on_progress, subtype="PCM_24")
    if result is not None:
        result["separator"] = separator_tag()
    return result


def _run_demucs_four(audio_path: Path, tmp_dir: Path,
                     out_paths: Dict[str, Path],
                     on_progress: ProgressCb = None,
                     subtype: Optional[str] = None) -> Optional[Dict]:
    """Full four-source Demucs: drums / bass / other / vocals.

    Also writes `instrumental` as the sum of drums+bass+other — the bed that
    playback, Studio and the exports read.
    """
    from api.workers._progress import parse_demucs, stream_subprocess
    log.info(f"Running Demucs ({DEMUCS_MODEL}, 4 stems) on: {audio_path.name}")

    cmd = [
        sys.executable, "-m", "demucs",
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
            on_progress(pct, f"Separating 4 stems: {pct}%")
        elif line.strip():
            on_progress(None, line.strip()[:120])

    try:
        result = stream_subprocess(cmd, _on_line, timeout=3600, env=thread_env())
        if result.returncode != 0:
            log.error(f"Demucs failed: {result.stdout[-500:]}")
            return None
    except FileNotFoundError:
        log.error("Demucs not found. Install with: pip install demucs")
        return None
    except subprocess.TimeoutExpired:
        log.error("Demucs timed out (>60 min)")
        return None

    demucs_out = tmp_dir / DEMUCS_MODEL / audio_path.stem
    raw = {src: demucs_out / f"{src}.wav" for src in DEMUCS_SOURCES}
    missing = [s for s, p in raw.items() if not p.exists()]
    if missing:
        log.error(f"Expected demucs sources missing in {demucs_out}: {missing}")
        return None

    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        log.error("Four-stem separation needs numpy + soundfile")
        return None

    if on_progress:
        on_progress(92, "Writing stems…")

    out: Dict[str, Path] = {}
    try:
        # Transcode each source to the configured format.
        for src in DEMUCS_SOURCES:
            data, sr = sf.read(str(raw[src]))
            sf.write(str(out_paths[src]), data, sr, subtype=subtype)
            out[src] = out_paths[src]

        # instrumental = drums + bass + other, summed in float at the source rate.
        if on_progress:
            on_progress(96, "Summing instrumental…")
        mix = None
        sr = None
        for src in INSTRUMENTAL_SOURCES:
            data, sr = sf.read(str(raw[src]))
            mix = data if mix is None else mix + data
        # Clip protection only: scaled down when (and only when) the sum would
        # clip, never normalised or limited otherwise.
        peak = float(np.max(np.abs(mix))) if mix is not None else 0.0
        if peak > 1.0:
            mix = mix / peak
        sf.write(str(out_paths["instrumental"]), mix, sr, subtype=subtype)
        out["instrumental"] = out_paths["instrumental"]
    except Exception:  # noqa: BLE001
        log.exception("failed writing four-stem output")
        return None
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log.info("Four stems ready: %s", ", ".join(sorted(out)))
    return out
