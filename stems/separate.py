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
    BASS_DIR, DATA_DIR, DEMUCS_MODEL, DEMUCS_SOURCES, DRUMS_DIR,
    INSTRUMENTALS_DIR, INSTRUMENTAL_SOURCES, MDX_MODEL, OTHER_DIR, STEM_FORMAT,
    VOCALS_DIR, current_stem_mode, current_stem_separator,
    sanitize_filename_chars,
)

# Where each Demucs source is written.
_SOURCE_DIRS = {
    "drums": DRUMS_DIR, "bass": BASS_DIR, "other": OTHER_DIR,
    "vocals": VOCALS_DIR,
}

log = logging.getLogger(__name__)

# Optional progress callback. percent is None for status-only updates
# (so the bar holds its last value while the message ticker keeps moving).
ProgressCb = Optional[Callable[[Optional[int], str], None]]

# Where audio-separator caches downloaded UVR models (persists across runs so
# the ~50-100MB download happens once).
MDX_MODEL_DIR = DATA_DIR / "uvr_models"


def separator_tag(separator: Optional[str] = None,
                  mode: Optional[str] = None) -> str:
    """Provenance tag for the given (or currently configured) engine and mode.

    The mode is part of the tag so stages.do_stems re-separates when the user
    switches to four stems — the existing two-stem files on disk are not wrong,
    they are just missing three of the four sources."""
    sep = separator or current_stem_separator()
    if sep == "mdx":
        return f"mdx:{Path(MDX_MODEL).stem}"
    mode = mode or current_stem_mode()
    return f"demucs:{DEMUCS_MODEL}:{4 if mode == 'four' else 2}"


def separate(song_id: int, title: str, audio_path: Path,
             artist: str = "",
             on_progress: ProgressCb = None,
             separator: Optional[str] = None,
             force: bool = False,
             mode: Optional[str] = None) -> Optional[Dict]:
    separator = separator or current_stem_separator()
    safe_title  = sanitize_filename_chars(title)[:40]
    safe_artist = sanitize_filename_chars(artist)[:30]
    safe_name   = f"{safe_title}_{safe_artist}"

    # MDX is a two-stem model; asking it for four would silently give two.
    mode = "two" if separator == "mdx" else (mode or current_stem_mode())

    def _p(directory: Path, kind: str) -> Path:
        return directory / f"{safe_name}_{kind}.{STEM_FORMAT}"

    # Legacy WAV stems from before STEM_FORMAT are still valid audio, so prefer
    # an existing .wav over re-separating a track that already has one.
    def _existing(directory: Path, kind: str) -> Optional[Path]:
        for ext in (STEM_FORMAT, "wav"):
            cand = directory / f"{safe_name}_{kind}.{ext}"
            if cand.exists():
                return cand
        return None

    wanted = (["vocals", "instrumental"] if mode == "two"
              else ["vocals", "instrumental", "drums", "bass", "other"])
    dirs = {"instrumental": INSTRUMENTALS_DIR, **_SOURCE_DIRS}

    if not force:
        have = {k: _existing(dirs[k], k) for k in wanted}
        if all(have.values()):
            log.info(f"Stems already exist for: {title}")
            if on_progress:
                on_progress(100, "Stems already on disk")
            # separator=None → reused as-is; caller keeps the DB's existing tag.
            return {**have, "separator": None}

    for d in set(dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    vocals_path       = _p(VOCALS_DIR, "vocals")
    instrumental_path = _p(INSTRUMENTALS_DIR, "instrumental")

    tmp_dir = Path(tempfile.gettempdir()) / f"mashup_tmp_{song_id:04d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if separator == "mdx":
        result = _run_mdx(audio_path, tmp_dir, vocals_path, instrumental_path, on_progress)
    elif mode == "four":
        result = _run_demucs_four(audio_path, tmp_dir,
                                  {k: _p(dirs[k], k) for k in wanted}, on_progress)
    else:
        result = _run_demucs(audio_path, tmp_dir, vocals_path, instrumental_path, on_progress)
    if result is not None:
        result["separator"] = separator_tag(separator, mode)
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


def _run_demucs_four(audio_path: Path, tmp_dir: Path,
                     out_paths: Dict[str, Path],
                     on_progress: ProgressCb = None) -> Optional[Dict]:
    """Full four-source Demucs: drums / bass / other / vocals.

    Also writes `instrumental` as the sum of drums+bass+other, so every existing
    consumer — the ranked list, the audition, Studio, the session export — keeps
    working with no change. The four sources are what make a bed's residual
    topline and its band occupancy measurable, and they are what let the user do
    the actual producer move of dropping the bed's bass or swapping its drums.
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
        result = stream_subprocess(cmd, _on_line, timeout=3600)
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
            sf.write(str(out_paths[src]), data, sr)
            out[src] = out_paths[src]

        # instrumental = drums + bass + other, summed at the source sample rate.
        # Demucs sources sum back to the original mix, so this is the same
        # signal the two-stem path's no_vocals would have produced.
        if on_progress:
            on_progress(96, "Summing instrumental…")
        mix = None
        sr = None
        for src in INSTRUMENTAL_SOURCES:
            data, sr = sf.read(str(raw[src]))
            mix = data if mix is None else mix + data
        peak = float(np.max(np.abs(mix))) if mix is not None else 0.0
        if peak > 1.0:
            mix = mix / peak
        sf.write(str(out_paths["instrumental"]), mix, sr)
        out["instrumental"] = out_paths["instrumental"]
    except Exception:  # noqa: BLE001
        log.exception("failed writing four-stem output")
        return None
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log.info("Four stems ready: %s", ", ".join(sorted(out)))
    return out
