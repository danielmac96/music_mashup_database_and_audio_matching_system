"""
render/mixdown.py — Render a Studio (DAW tab) arrangement to a single WAV.

The Studio plays N clips live in the browser (SoundTouch worklet); this is the
offline, faithful version of the same math for export:

  * each clip = one stem file + offset (display seconds) + rate + semitones + gain
  * rate is the playback-speed factor (1.0 = native; 1.1 = 10% faster), matching
    the engine's `playbackRate`, so display duration = raw duration / rate
  * tempo and pitch are decoupled (librosa phase vocoder), exactly like the
    two-stem Audition export in render/preview.py
  * clips are summed on one timeline and peak-limited

librosa/soundfile import lazily so the API keeps working without the audio
stack; the worker reports a clear error instead.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

from config import PREVIEWS_DIR
from database.models import get_conn

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[Optional[int], str], None]]

MIXDOWN_SR = 44100
MAX_MIXDOWN_SECS = 900.0   # 15 min safety cap so a runaway render can't fill the disk
MAX_CLIPS = 16

_STEM_TYPES = {"full", "vocals", "instrumental"}
_TOKEN_RE = re.compile(r"^[0-9a-f]{8,64}$")


def mixdown_path(token: str) -> Optional[Path]:
    """Path for a mixdown WAV by job token. None when the token is malformed
    (the token lands in a filename, so refuse anything but plain hex)."""
    if not _TOKEN_RE.match(token or ""):
        return None
    return PREVIEWS_DIR / f"studio_mix_{token}.wav"


def _resolve_stem_path(song_id: int, stem_type: str) -> Optional[Path]:
    conn = get_conn()
    row = conn.execute(
        "SELECT file_path FROM stems WHERE song_id=? AND stem_type=?",
        (song_id, stem_type),
    ).fetchone()
    if row and row["file_path"] and Path(row["file_path"]).exists():
        conn.close()
        return Path(row["file_path"])
    if stem_type == "full":
        song = conn.execute("SELECT raw_path FROM songs WHERE id=?", (song_id,)).fetchone()
        conn.close()
        if song and song["raw_path"] and Path(song["raw_path"]).exists():
            return Path(song["raw_path"])
        return None
    conn.close()
    return None


def build_mixdown(token: str, clips: list[dict],
                  on_progress: ProgressCb = None) -> Optional[Path]:
    """Render `clips` to one WAV. Each clip dict:
        song_id: int, stem: str, offset_sec: float,
        rate: float (>0), semitones: int, gain: float (linear)
    Returns the output path, or None on a caller-fixable problem (details are
    reported through on_progress so the job message explains itself)."""
    def _tick(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    out = mixdown_path(token)
    if out is None:
        _tick(None, "Invalid mixdown token")
        return None
    if not clips:
        _tick(None, "No clips to render")
        return None

    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except ImportError as exc:
        log.error("mixdown needs librosa + soundfile: %s", exc)
        _tick(None, "Audio stack (librosa/soundfile) is not installed on the server")
        return None

    # Everything is placed relative to the earliest clip start, so a clip
    # dragged left of zero still renders instead of being clipped away.
    base = min(0.0, *(float(c.get("offset_sec") or 0.0) for c in clips))

    rendered: list[tuple[int, "np.ndarray", float]] = []  # (start_sample, samples, gain)
    n = len(clips)
    for idx, c in enumerate(clips):
        song_id = int(c["song_id"])
        stem = str(c.get("stem") or "full")
        if stem not in _STEM_TYPES:
            _tick(None, f"Clip {idx + 1}: unknown stem '{stem}'")
            return None
        path = _resolve_stem_path(song_id, stem)
        if path is None:
            _tick(None, f"Clip {idx + 1}: no {stem} audio for song {song_id} — "
                        "separate/download it first")
            return None

        rate = float(c.get("rate") or 1.0)
        rate = min(4.0, max(0.25, rate))
        semitones = int(c.get("semitones") or 0)
        semitones = min(24, max(-24, semitones))
        gain = max(0.0, float(c.get("gain") if c.get("gain") is not None else 0.8))
        offset = float(c.get("offset_sec") or 0.0)

        pct_lo = int(5 + 85 * idx / n)
        _tick(pct_lo, f"Clip {idx + 1}/{n}: loading {path.name}…")
        y, _ = librosa.load(str(path), sr=MIXDOWN_SR, mono=True,
                            duration=MAX_MIXDOWN_SECS * rate)

        if abs(rate - 1.0) > 1e-3:
            _tick(pct_lo + 3, f"Clip {idx + 1}/{n}: time-stretching ×{rate:.3f}…")
            y = librosa.effects.time_stretch(y, rate=rate, n_fft=1024)
        if semitones:
            _tick(pct_lo + 6, f"Clip {idx + 1}/{n}: pitch-shifting {semitones:+d} st…")
            y = librosa.effects.pitch_shift(y, sr=MIXDOWN_SR, n_steps=semitones, n_fft=1024)

        start = int(round((offset - base) * MIXDOWN_SR))
        rendered.append((start, y, gain))

    _tick(92, "Summing timeline…")
    total = max(start + len(y) for start, y, _ in rendered)
    total = min(total, int(MAX_MIXDOWN_SECS * MIXDOWN_SR))
    if total <= 0:
        _tick(None, "Empty timeline")
        return None

    mix = np.zeros(total, dtype="float32")
    for start, y, gain in rendered:
        if start >= total:
            continue
        end = min(total, start + len(y))
        mix[start:end] += y[: end - start] * gain

    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 1.0:
        mix = mix / peak

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix.astype("float32"), MIXDOWN_SR)
    _tick(100, "Mixdown ready")
    log.info("studio mixdown rendered: %s (%d clips)", out.name, len(clips))
    return out
