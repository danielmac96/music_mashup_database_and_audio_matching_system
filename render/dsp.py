"""render/dsp.py — the offline audio primitives both render paths share.

render/mixdown.py sums an arrangement into one WAV; render/session.py writes the
same clips out as separate, conformed stems for a DAW. They must agree exactly
about what "play this stem at rate R, shifted N semitones" means — an export
that disagrees with the mixdown is an export that disagrees with what the user
heard in Studio — so the loading, clamping and stretch/shift live here once.

Tempo and pitch are decoupled (librosa phase vocoder), matching the browser's
SoundTouch worklet: `rate` is the playback-speed factor (1.0 = native, 1.1 = 10%
faster), so display duration = raw duration / rate, and `semitones` is applied
independently on top.

librosa/soundfile import lazily so the API keeps working without the audio
stack; callers report a clear error instead of 500ing.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[Optional[int], str], None]]

RENDER_SR = 44100
MAX_RENDER_SECS = 900.0   # 15 min safety cap so a runaway render can't fill the disk
MAX_CLIPS = 16

STEM_TYPES = {"full", "vocals", "instrumental"}

# Tokens land in filenames, so refuse anything but plain hex.
TOKEN_RE = re.compile(r"^[0-9a-f]{8,64}$")

MIN_RATE, MAX_RATE = 0.25, 4.0
MAX_SEMITONES = 24


class AudioStackMissing(RuntimeError):
    """librosa/soundfile are not installed on this server."""


def require_audio_stack():
    """Import the audio stack, or raise AudioStackMissing with a clear message."""
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AudioStackMissing(
            "Audio stack (librosa/soundfile) is not installed on the server"
        ) from exc
    return np, librosa, sf


def clamp_rate(value) -> float:
    try:
        r = float(value if value is not None else 1.0)
    except (TypeError, ValueError):
        return 1.0
    if r <= 0:
        return 1.0
    return min(MAX_RATE, max(MIN_RATE, r))


def clamp_semitones(value) -> int:
    try:
        s = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return min(MAX_SEMITONES, max(-MAX_SEMITONES, s))


def clamp_gain(value) -> float:
    try:
        g = float(value if value is not None else 0.8)
    except (TypeError, ValueError):
        return 0.8
    return max(0.0, g)


def is_valid_token(token: str) -> bool:
    return bool(TOKEN_RE.match(token or ""))


def resolve_stem_path(song_id: int, stem_type: str, db_path=None) -> Optional[Path]:
    """The audio file for one stem of one song.

    'full' falls back to the song's raw download when the stems table has no
    row for it — a track that was downloaded but never separated still has a
    playable full mix."""
    from database.models import get_conn

    conn = get_conn(db_path) if db_path else get_conn()
    try:
        row = conn.execute(
            "SELECT file_path FROM stems WHERE song_id=? AND stem_type=?",
            (song_id, stem_type),
        ).fetchone()
        if row and row["file_path"] and Path(row["file_path"]).exists():
            return Path(row["file_path"])
        if stem_type == "full":
            song = conn.execute("SELECT raw_path FROM songs WHERE id=?",
                                (song_id,)).fetchone()
            if song and song["raw_path"] and Path(song["raw_path"]).exists():
                return Path(song["raw_path"])
        return None
    finally:
        conn.close()


def load_segment(path: Path, sr: int = RENDER_SR, *,
                 start_sec: Optional[float] = None,
                 end_sec: Optional[float] = None,
                 max_secs: float = MAX_RENDER_SECS,
                 rate: float = 1.0):
    """Mono samples for [start_sec, end_sec) of `path`, at `sr`.

    `rate` only sizes the read: playing faster consumes proportionally more raw
    audio for the same output duration, so the cap is applied in output terms.
    """
    _np, librosa, _sf = require_audio_stack()
    offset = max(0.0, float(start_sec or 0.0))
    if end_sec is not None and end_sec > offset:
        duration = min(float(end_sec) - offset, max_secs * rate)
    else:
        duration = max_secs * rate
    y, _ = librosa.load(str(path), sr=sr, mono=True,
                        offset=offset, duration=duration)
    return y


def conform(y, sr: int, rate: float, semitones: int,
            on_progress: ProgressCb = None, label: str = ""):
    """Time-stretch then pitch-shift, decoupled. Returns the new samples.

    Both operations are skipped when they would be identity, so a clip that
    needs neither is returned untouched rather than round-tripped through a
    phase vocoder for nothing.
    """
    _np, librosa, _sf = require_audio_stack()
    if abs(rate - 1.0) > 1e-3:
        if on_progress:
            on_progress(None, f"{label}time-stretching ×{rate:.3f}…")
        y = librosa.effects.time_stretch(y, rate=rate, n_fft=1024)
    if semitones:
        if on_progress:
            on_progress(None, f"{label}pitch-shifting {semitones:+d} st…")
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones, n_fft=1024)
    return y


def peak_normalise(mix):
    """Scale down to unity only if the sum clipped. Never boosts a quiet mix."""
    _np, _librosa, _sf = require_audio_stack()
    peak = float(_np.max(_np.abs(mix))) if len(mix) else 0.0
    if peak > 1.0:
        return mix / peak
    return mix
