"""
analysis/analyze.py — the tempo, beat-grid and key helpers the Essentia
analyser (analysis/essentia_groups.py) and the section cut
(analysis/structure.py) share: Camelot codes, bar-1 phase, grid confidence,
BPM fitted through the beats, Krumhansl key from a chroma vector and the
waveform envelope.
"""
from typing import Callable, Optional
import logging
import numpy as np

# Optional progress callback. percent is None (status-only) for analysis since
# librosa stages aren't streamable; we just push stage messages so the UI can
# show liveness alongside the elapsed timer.
ProgressCb = Optional[Callable[[Optional[int], str], None]]

log = logging.getLogger(__name__)

CAMELOT = {
    (0,  "major"): "8B",  (1,  "major"): "3B",  (2,  "major"): "10B",
    (3,  "major"): "5B",  (4,  "major"): "12B", (5,  "major"): "7B",
    (6,  "major"): "2B",  (7,  "major"): "9B",  (8,  "major"): "4B",
    (9,  "major"): "11B", (10, "major"): "6B",  (11, "major"): "1B",
    (0,  "minor"): "5A",  (1,  "minor"): "12A", (2,  "minor"): "7A",
    (3,  "minor"): "2A",  (4,  "minor"): "9A",  (5,  "minor"): "4A",
    (6,  "minor"): "11A", (7,  "minor"): "6A",  (8,  "minor"): "1A",
    (9,  "minor"): "8A",  (10, "minor"): "3A",  (11, "minor"): "10A",
}

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ── Per-metric step functions ─────────────────────────────────────────────────
# Each takes the loaded signal (+ sr/hop) and returns a dict of feature keys.
# None of these mutate shared state, so any one of them can fail without
# corrupting the others.

def _pick_beat_phase(onset_env: np.ndarray, beat_frames: np.ndarray) -> int:
    """Which of the 4 positions in a bar the detected beat grid starts on.

    Consumers treat "every 4th beat from the first detected beat" as a downbeat,
    but librosa's tracker latches wherever the onset evidence is strongest and
    is just as happy to start on beat 3. Summing onset strength at each of the 4
    candidate phases and taking the argmax recovers the real bar line: the kick
    that starts the bar is louder than the beats between.

    Returns 0 when there is nothing to go on (too few beats, or a flat envelope
    where every phase scores the same), so an un-analysable track renders on the
    old assumption rather than a made-up offset.
    """
    beat_frames = np.asarray(beat_frames, dtype=int)
    if beat_frames.size < 4:
        return 0
    n = onset_env.shape[0]
    strengths = np.array([
        float(onset_env[[f for f in beat_frames[phase::4] if 0 <= f < n]].sum())
        if any(0 <= f < n for f in beat_frames[phase::4]) else 0.0
        for phase in range(4)
    ])
    # A flat envelope carries no downbeat information — argmax would return an
    # arbitrary 0 anyway, but be explicit so the intent survives refactoring.
    if float(strengths.max() - strengths.min()) <= 1e-9:
        return 0
    return int(np.argmax(strengths))


# ── Beat-grid confidence ─────────────────────────────────────────────────────
#
# This used to be `len(beats) / n_frames`, which is beats-per-frame — i.e.
# `bpm / 2580` at our sample rate and hop. It ranged 0.027 (70 BPM) to 0.067
# (174 BPM), never approached 1.0, and said nothing whatsoever about whether the
# grid was trustworthy. Everything downstream read it as a 0-1 confidence:
# `effort.grid_cost` was ~0.95 for every track in the library, which put a
# constant floor of ~0.24 on the effort penalty and made `effort_label`'s "Free"
# bucket (<= 0.20) unreachable by construction. The vocal-beat gate in
# api/routes/tracks.py (VOCAL_BEAT_CONFIDENCE_MIN = 0.35) was likewise dead.
#
# What actually predicts "can I trust this grid in a DAW" is two things, and
# both are physical rather than calibrated guesses:
#
#   steadiness — how constant the inter-beat interval is. A programmed record
#                holds it to a fraction of a percent; a live take, a track with
#                rubato, or a tracker that lost the beat wanders.
#   salience   — how much more onset energy lands ON the detected beats than
#                the track average. A grid that does not sit on the transients
#                is a grid you will be dragging by hand.
#
# They are multiplied because either one failing is disqualifying: a perfectly
# even grid that sits between the kicks is still wrong.

# Relative std of the inter-beat interval at which the grid carries no
# information. 10% jitter is roughly "the tracker is guessing".
BEAT_JITTER_MAX = 0.10

# Onset strength on the beats, as a multiple of the track's mean. 1.0 means the
# beats are no louder than anywhere else (no evidence); 2.0 is a clear grid.
BEAT_SALIENCE_FLOOR = 1.0
BEAT_SALIENCE_FULL = 2.0

# Below this many beats there is not enough of a grid to measure.
BEAT_MIN_COUNT = 8


def _ramp01(value: float, low: float, high: float) -> float:
    """0 at or below `low`, 1 at or above `high`, linear between."""
    if not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def beat_grid_confidence(beat_times, onset_env=None, beat_frames=None) -> float:
    """How much to trust this track's beat grid, 0 (useless) to 1 (locked).

    `onset_env` / `beat_frames` are optional: with them the salience term is
    measured, without them the result is the steadiness term alone (which is
    what a caller holding only stored `beat_times` can compute).
    """
    t = np.asarray(beat_times, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < BEAT_MIN_COUNT:
        return 0.0

    ibi = np.diff(t)
    ibi = ibi[ibi > 0]
    if ibi.size < 2:
        return 0.0
    median = float(np.median(ibi))
    if median <= 0:
        return 0.0
    jitter = float(np.std(ibi) / median)
    steadiness = 1.0 - _ramp01(jitter, 0.0, BEAT_JITTER_MAX)

    salience = 1.0
    if onset_env is not None and beat_frames is not None:
        env = np.asarray(onset_env, dtype=float)
        frames = np.asarray(beat_frames, dtype=int)
        frames = frames[(frames >= 0) & (frames < env.shape[0])]
        mean_env = float(env.mean()) if env.size else 0.0
        if frames.size and mean_env > 1e-9:
            ratio = float(env[frames].mean()) / mean_env
            salience = _ramp01(ratio, BEAT_SALIENCE_FLOOR, BEAT_SALIENCE_FULL)

    return float(np.clip(steadiness * salience, 0.0, 1.0))


# ── Tempo from the beats ─────────────────────────────────────────────────────
#
# librosa's tempo comes from a tempogram with discrete bins (…123.05, 126.05,
# 129.2…), and its beats sit on the 23 ms hop grid, so neither the tempo nor a
# median beat interval can say 128.0: a 128 BPM record was stored as 129.2, a
# 1% error that drifts a full beat over 32 bars once Studio syncs to it. A line
# fitted through every beat — time against beat number — averages the grid
# away. Measured on the library (2026-09-30), it matched Essentia's continuous
# BPM to ~0.1 on every track where the two agreed on the octave; librosa had
# been up to 3 BPM off. The octave is still librosa's: the beats already follow
# its tempo, and the fit only refines it.

# Beats further than this fraction of a period off the fitted line are the
# tracker slipping (a stray onset, half a beat of phase), not the tempo.
BEAT_FIT_TOLERANCE = 0.25
# The share of beats that must sit on the line for the fit to speak for the track.
BEAT_FIT_MIN_INLIERS = 0.3


def bpm_from_beats(beat_times) -> Optional[float]:
    """The tempo of a least-squares line through the beats, or None when there
    are too few beats or too few of them agree on one line."""
    t = np.asarray(beat_times, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < BEAT_MIN_COUNT:
        return None
    ibi = np.diff(t)
    positive = ibi[ibi > 0]
    if positive.size < 2:
        return None
    period = float(np.median(positive))
    # Beat numbers from the local gaps: a dropped beat skips a number, a stray
    # beat between two shares its neighbour's and falls out as an outlier.
    idx = np.concatenate([[0.0], np.cumsum(np.round(ibi / period))])
    keep = np.ones(t.size, dtype=bool)
    slope = period
    for _ in range(3):
        if keep.sum() < BEAT_MIN_COUNT or np.ptp(idx[keep]) <= 0:
            return None
        slope, icpt = np.polyfit(idx[keep], t[keep], 1)
        if slope <= 0:
            return None
        idx = np.round((t - icpt) / slope)
        keep = np.abs(t - (slope * idx + icpt)) <= BEAT_FIT_TOLERANCE * slope
    if keep.mean() < BEAT_FIT_MIN_INLIERS or keep.sum() < BEAT_MIN_COUNT:
        return None
    slope, _icpt = np.polyfit(idx[keep], t[keep], 1)
    return float(60.0 / slope) if slope > 0 else None


def fitted_bpm(beat_times, estimate: Optional[float]) -> Optional[float]:
    """``bpm_from_beats``, kept only when it refines ``estimate`` within its own
    octave (±6%); otherwise the estimate. Rounded to 2 decimals."""
    fit = bpm_from_beats(beat_times)
    if fit is not None and estimate and abs(fit - estimate) / estimate <= 0.06:
        return round(fit, 2)
    if fit is not None and not estimate:
        return round(fit, 2)
    return round(float(estimate), 2) if estimate else None


def key_from_chroma(chroma_mean: np.ndarray) -> dict:
    """Krumhansl key estimate + confidence from one 12-bin chroma vector.

    Extracted from _step_key so Phase E can run the identical estimator on a
    SECTION's chroma. A track has one key only in the sense that an average has
    one value: real records modulate, and the chorus is frequently not the key
    the whole-track mean reports.
    """
    chroma_mean = np.asarray(chroma_mean, dtype=float)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_corrs = [np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1]
                   for i in range(12)]
    minor_corrs = [np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1]
                   for i in range(12)]
    best_major_idx = int(np.argmax(major_corrs))
    best_minor_idx = int(np.argmax(minor_corrs))
    if major_corrs[best_major_idx] >= minor_corrs[best_minor_idx]:
        key_idx, mode = best_major_idx, "major"
    else:
        key_idx, mode = best_minor_idx, "minor"

    # Key is the heaviest score weight and the least reliable number we store,
    # so report how much to trust it. Detection fails two independent ways and
    # confidence must collapse if EITHER holds, hence the product:
    #
    #   margin — how far the winning profile beat the next best of the other 23.
    #            Near 0 means two keys are effectively tied (tonal but ambiguous).
    #   peak   — how peaked the chroma is. Near 0 means there is no tonal centre
    #            to find at all (percussion, noise, a drum-led instrumental).
    #
    # Margin alone is not enough: corrcoef normalises away scale, so the tiny
    # random wiggles in a flat chroma still correlate strongly with whichever
    # profile happens to match them. Measured on real stems, white noise scores
    # a *higher* bare margin (0.27) than any track in the library (0.005–0.20).
    ranked = sorted((c for c in (major_corrs + minor_corrs) if np.isfinite(c)),
                    reverse=True)
    margin = float(ranked[0] - ranked[1]) if len(ranked) >= 2 else 0.0
    peak = float((chroma_mean.max() - chroma_mean.mean())
                 / (chroma_mean.max() + 1e-9)) if chroma_mean.max() > 0 else 0.0
    confidence = min(max(margin, 0.0), 1.0) * min(max(peak, 0.0), 1.0)

    return {
        "key": KEY_NAMES[key_idx],
        "mode": mode,
        "camelot": CAMELOT.get((key_idx, mode), "?"),
        "key_confidence": float(confidence),
    }


def _step_waveform(y: np.ndarray, n_points: int = 360) -> dict:
    chunk = max(1, len(y) // n_points)
    wf = [float(np.sqrt(np.mean(y[i * chunk:(i + 1) * chunk] ** 2))) for i in range(n_points)]
    mx = max(wf) or 1.0
    return {"waveform_rms": [round(v / mx, 5) for v in wf]}


