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
    ProgressCb, apply_eq, apply_fades, apply_filters, clamp_gain, clamp_rate,
    clamp_semitones, conform, duck_depth, duck_envelope, duck_gain, is_valid_token,
    load_segment, require_audio_stack, resolve_stem_path,
)

log = logging.getLogger(__name__)

# Clips rendered at once. Kept small: a pipeline import may be running beside
# an export, and each Rubber Band clip is its own ffmpeg process.
RENDER_THREADS = 3

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
                  db_path=None, loudness_target: Optional[float] = None,
                  report: Optional[dict] = None) -> Optional[Path]:
    """Render `clips` to one WAV. Each clip dict:
        song_id: int, stem: str, offset_sec: float,
        rate: float (>0), semitones: int, gain: float (linear),
        start_sec: float | None, end_sec: float | None   (trim, optional)
        fade_in / fade_out: float seconds, hp_hz / lp_hz: float (0 = off)
        eq_low / eq_mid / eq_high: float dB (0 = flat; render/dsp.EQ_BANDS)
        duck_db: float dB (0 = off) — turned down while the vocals sing; the
            key is every `vocals`-stem clip as heard (render/dsp duck_*)
        key_only: bool — feeds the duck key and is not mixed (a stem render
            of a ducked lane sends the vocals this way)

    `offset_sec` is where the clip's FIRST RENDERED SAMPLE lands — the trim
    start when there is a trim, not where the stem's 0:00 would sit.

    `start_sec`/`end_sec` take a SECTION out of the stem rather than playing it
    whole — what a candidate preview needs. Omitted, the clip behaves exactly
    as before, so no existing caller changes.
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

    # Check every clip first (cheap, and a bad one fails before any DSP), then
    # render them side by side: each is independent, and the heavy parts —
    # decoding, Rubber Band in its own ffmpeg process, scipy filters — release
    # the GIL. One at a time, a Rubber Band export of a pair took ~16 s.
    jobs = []
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
        # None (not 0.0) means "no trim" — a clip genuinely starting at 0.0 is a
        # different instruction from one that was never given a start.
        start_sec = c.get("start_sec")
        end_sec = c.get("end_sec")
        start_sec = float(start_sec) if start_sec is not None else None
        end_sec = float(end_sec) if end_sec is not None else None
        if start_sec is not None and end_sec is not None and end_sec <= start_sec:
            _tick(None, f"Clip {idx + 1}: end_sec must be after start_sec")
            return None
        jobs.append((idx, c, path, start_sec, end_sec))

    def _render(job):
        idx, c, path, start_sec, end_sec = job
        rate = clamp_rate(c.get("rate"))
        semitones = clamp_semitones(c.get("semitones"))
        label = f"Clip {idx + 1}/{n}: "
        _tick(int(5 + 85 * idx / n), f"{label}loading {path.name}…")
        y = load_segment(path, MIXDOWN_SR, start_sec=start_sec, end_sec=end_sec,
                         max_secs=MAX_MIXDOWN_SECS, rate=rate)
        y = conform(y, MIXDOWN_SR, rate, semitones,
                    on_progress=on_progress, label=label)
        y = apply_filters(y, MIXDOWN_SR, float(c.get("hp_hz") or 0.0),
                          float(c.get("lp_hz") or 0.0))
        y = apply_eq(y, MIXDOWN_SR, c.get("eq_low"), c.get("eq_mid"), c.get("eq_high"))
        y = apply_fades(y, MIXDOWN_SR, float(c.get("fade_in") or 0.0),
                        float(c.get("fade_out") or 0.0))
        start = int(round((float(c.get("offset_sec") or 0.0) - base) * MIXDOWN_SR))
        duck_db = float(c.get("duck_db") or 0.0) if duck_depth(c.get("duck_db")) > 0 else 0.0
        key_only = bool(c.get("key_only"))
        is_key = str(c.get("stem") or "full") == "vocals" or key_only
        return (start, y, clamp_gain(c.get("gain")), duck_db, is_key, key_only)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, min(RENDER_THREADS, n))) as pool:
        # (start_sample, samples, gain, duck_db, is_key, key_only)
        rendered = list(pool.map(_render, jobs))

    _tick(92, "Summing timeline…")
    total = max(start + len(y) for start, y, *_ in rendered)
    total = min(total, int(MAX_MIXDOWN_SECS * MIXDOWN_SR))
    if total <= 0:
        _tick(None, "Empty timeline")
        return None

    def _sum(parts):
        out = np.zeros(total, dtype="float32")
        for start, y, gain in parts:
            if start >= total:
                continue
            end = min(total, start + len(y))
            out[start:end] += y[: end - start] * gain
        return out

    if all(r[5] for r in rendered):
        _tick(None, "Only key clips — nothing to mix")
        return None
    # The duck key: every vocals clip as heard, before any lane is ducked.
    env = None
    if any(r[3] > 0 for r in rendered):
        env = duck_envelope(_sum([(s, y, g) for s, y, g, _d, key, _o in rendered if key]),
                            MIXDOWN_SR)
    mix = np.zeros(total, dtype="float32")
    for start, y, gain, duck_db, is_key, key_only in rendered:
        if key_only or start >= total:
            continue
        end = min(total, start + len(y))
        part = y[: end - start] * gain
        if env is not None and duck_db > 0 and not is_key:     # a vocal never ducks itself
            part = part * duck_gain(env[start:end], duck_db)
        mix[start:end] += part

    # Mastered, not just un-clipped: to the loudness target when one is asked
    # for, and never past −1 dBTP (render/master.py). A plain peak-normalise
    # left exports at whatever level the lanes summed to, true peak 0 dBFS.
    _tick(95, "Mastering (loudness, true-peak limit)…")
    from render.master import master
    mix, meas = master(mix, MIXDOWN_SR, target_lufs=loudness_target)
    if report is not None:
        report.update(meas)

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix.astype("float32"), MIXDOWN_SR)
    lufs = f"{meas['lufs']:.1f} LUFS" if meas.get("lufs") is not None else "silent"
    tp = f"{meas['true_peak']:.1f} dBTP" if meas.get("true_peak") is not None else ""
    _tick(100, f"Mixdown ready — {lufs}{' · ' + tp if tp else ''}")
    log.info("studio mixdown rendered: %s (%d clips)", out.name, len(clips))
    return out
