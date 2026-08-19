"""
render/mixdown.py — Render a Studio (DAW tab) arrangement to a single WAV.

The Studio plays N clips live in the browser (SoundTouch worklet); this is the
offline, faithful version of the same math for export:

  * each clip = one stem file + offset (display seconds) + rate + semitones + gain
  * rate is the playback-speed factor (1.0 = native; 1.1 = 10% faster), matching
    the engine's `playbackRate`, so display duration = raw duration / rate
  * tempo and pitch are decoupled (librosa phase vocoder)
  * clips are summed on one timeline and peak-limited

The loading, clamping and stretch/shift primitives live in render/dsp.py, shared
with render/session.py — an FL session export that disagreed with the mixdown
would disagree with what the user heard in Studio.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import PREVIEWS_DIR
from render.dsp import (
    MAX_CLIPS, MAX_RENDER_SECS, RENDER_SR, STEM_TYPES, AudioStackMissing,
    ProgressCb, clamp_gain, clamp_rate, clamp_semitones, conform, is_valid_token,
    load_segment, peak_normalise, require_audio_stack, resolve_stem_path,
)

log = logging.getLogger(__name__)

# Kept as module-level names because callers (api/routes/studio.py, tests) and
# the readme refer to them here.
MIXDOWN_SR = RENDER_SR
MAX_MIXDOWN_SECS = MAX_RENDER_SECS
__all__ = ["MAX_CLIPS", "MAX_MIXDOWN_SECS", "MIXDOWN_SR", "build_mixdown",
           "mixdown_path"]


def mixdown_path(token: str) -> Optional[Path]:
    """Path for a mixdown WAV by job token. None when the token is malformed
    (the token lands in a filename, so refuse anything but plain hex)."""
    if not is_valid_token(token):
        return None
    return PREVIEWS_DIR / f"studio_mix_{token}.wav"


def build_mixdown(token: str, clips: list[dict],
                  on_progress: ProgressCb = None,
                  db_path=None) -> Optional[Path]:
    """Render `clips` to one WAV. Each clip dict:
        song_id: int, stem: str, offset_sec: float,
        rate: float (>0), semitones: int, gain: float (linear),
        start_sec: float | None, end_sec: float | None   (trim, optional)

    `start_sec`/`end_sec` take a SECTION out of the stem rather than playing it
    whole — what a candidate preview needs, and what Claude_next_steps.md calls
    the single biggest gap in Studio. Omitted, the clip behaves exactly as
    before, so no existing caller changes.
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
        np, _librosa, sf = require_audio_stack()
    except AudioStackMissing as exc:
        log.error("mixdown needs librosa + soundfile: %s", exc)
        _tick(None, str(exc))
        return None

    # Everything is placed relative to the earliest clip start, so a clip
    # dragged left of zero still renders instead of being clipped away.
    base = min(0.0, *(float(c.get("offset_sec") or 0.0) for c in clips))

    rendered: list[tuple[int, "np.ndarray", float]] = []  # (start_sample, samples, gain)
    n = len(clips)
    for idx, c in enumerate(clips):
        song_id = int(c["song_id"])
        stem = str(c.get("stem") or "full")
        if stem not in STEM_TYPES:
            _tick(None, f"Clip {idx + 1}: unknown stem '{stem}'")
            return None
        path = resolve_stem_path(song_id, stem, db_path=db_path)
        if path is None:
            _tick(None, f"Clip {idx + 1}: no {stem} audio for song {song_id} — "
                        "separate/download it first")
            return None

        rate = clamp_rate(c.get("rate"))
        semitones = clamp_semitones(c.get("semitones"))
        gain = clamp_gain(c.get("gain"))
        offset = float(c.get("offset_sec") or 0.0)
        # None (not 0.0) means "no trim" — a clip genuinely starting at 0.0 is a
        # different instruction from one that was never given a start.
        start_sec = c.get("start_sec")
        end_sec = c.get("end_sec")
        start_sec = float(start_sec) if start_sec is not None else None
        end_sec = float(end_sec) if end_sec is not None else None
        if start_sec is not None and end_sec is not None and end_sec <= start_sec:
            _tick(None, f"Clip {idx + 1}: end_sec must be after start_sec")
            return None

        pct_lo = int(5 + 85 * idx / n)
        label = f"Clip {idx + 1}/{n}: "
        _tick(pct_lo, f"{label}loading {path.name}…")
        y = load_segment(path, MIXDOWN_SR, start_sec=start_sec, end_sec=end_sec,
                         max_secs=MAX_MIXDOWN_SECS, rate=rate)
        y = conform(y, MIXDOWN_SR, rate, semitones,
                    on_progress=on_progress, label=label)

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

    mix = peak_normalise(mix)

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix.astype("float32"), MIXDOWN_SR)
    _tick(100, "Mixdown ready")
    log.info("studio mixdown rendered: %s (%d clips)", out.name, len(clips))
    return out
