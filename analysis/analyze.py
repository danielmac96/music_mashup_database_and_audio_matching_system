"""
analysis/analyze.py — Extract musical features from an audio file.

Features: BPM, key, Camelot, loudness, energy, MFCC, spectral shape.
Requires: librosa, numpy

Each metric group below is its own step function (tempo, key, dynamics,
timbre, waveform) so one failing measurement doesn't take the rest down with
it — a stem that defeats the key detector (e.g. a noisy/atonal stem) still
comes back with BPM, loudness and timbre filled in.
"""
from typing import Callable, Optional
import logging
import numpy as np
from pathlib import Path

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

# The ordered metric steps analyze_file runs. Exposed so callers (and tests)
# can see what "fully analysed" means without re-deriving it from the code.
STEPS = ("tempo", "key", "dynamics", "timbre", "waveform")


# ── Per-metric step functions ─────────────────────────────────────────────────
# Each takes the loaded signal (+ sr/hop) and returns a dict of feature keys.
# None of these mutate shared state, so any one of them can fail without
# corrupting the others.

def _step_tempo(y: np.ndarray, sr: int, hop_length: int) -> dict:
    import librosa
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    return {
        "bpm": float(round(float(np.atleast_1d(tempo)[0]), 2)),
        "bpm_confidence": float(min(len(beats) / (len(y) / hop_length), 1.0)),
        "beat_times": [round(float(t), 4) for t in beat_times],
    }


def _step_key(y: np.ndarray, sr: int, hop_length: int) -> dict:
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = chroma.mean(axis=1)
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


def _step_dynamics(y: np.ndarray, sr: int, hop_length: int) -> dict:
    import librosa
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    return {
        "loudness_rms": float(round(float(rms.mean()), 6)),
        "energy": float(round(float((S ** 2).mean()), 6)),
    }


def _step_timbre(y: np.ndarray, sr: int, hop_length: int, n_mfcc: int) -> dict:
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    return {
        "mfcc": [round(float(v), 4) for v in mfcc.mean(axis=1)],
        "spectral_centroid": float(round(float(centroid.mean()), 2)),
        "spectral_rolloff": float(round(float(rolloff.mean()), 2)),
        "zero_crossing_rate": float(round(float(zcr.mean()), 6)),
    }


def _step_waveform(y: np.ndarray, n_points: int = 360) -> dict:
    chunk = max(1, len(y) // n_points)
    wf = [float(np.sqrt(np.mean(y[i * chunk:(i + 1) * chunk] ** 2))) for i in range(n_points)]
    mx = max(wf) or 1.0
    return {"waveform_rms": [round(v / mx, 5) for v in wf]}


def analyze_file(audio_path: Path, trim_secs: Optional[int] = None,
                  on_progress: ProgressCb = None) -> dict:
    def _tick(msg: str) -> None:
        if on_progress:
            on_progress(None, msg)

    try:
        import librosa
    except ImportError:
        log.error("librosa not installed. Run: pip install librosa")
        return {}

    log.info(f"Analysing: {audio_path.name}"
             + (f" (first {trim_secs}s)" if trim_secs else ""))

    try:
        from config import SAMPLE_RATE, HOP_LENGTH, N_MFCC
    except ImportError:
        SAMPLE_RATE, HOP_LENGTH, N_MFCC = 22050, 512, 13

    _tick("Loading audio…")
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE,
                          duration=trim_secs, mono=True)

    features: dict = {}
    failed_steps: list[str] = []

    step_plan = (
        ("tempo",    "Detecting BPM…",               lambda: _step_tempo(y, sr, HOP_LENGTH)),
        ("key",      "Detecting key…",                lambda: _step_key(y, sr, HOP_LENGTH)),
        ("dynamics", "Computing loudness + energy…",  lambda: _step_dynamics(y, sr, HOP_LENGTH)),
        ("timbre",   "Computing MFCC + spectral shape…", lambda: _step_timbre(y, sr, HOP_LENGTH, N_MFCC)),
        ("waveform", "Computing waveform envelope…",  lambda: _step_waveform(y)),
    )

    for step_name, msg, run_step in step_plan:
        _tick(msg)
        try:
            features.update(run_step())
        except Exception:  # noqa: BLE001
            log.exception("  step '%s' failed for %s", step_name, audio_path.name)
            failed_steps.append(step_name)

    if failed_steps:
        log.warning(f"  → steps failed: {', '.join(failed_steps)}")
    if "bpm" in features:
        rms_text = f", RMS={features['loudness_rms']:.4f}" if features.get("loudness_rms") is not None else ""
        log.info(
            f"  → BPM={features.get('bpm')}, "
            f"Key={features.get('key', '?')} {features.get('mode', '')}, "
            f"Camelot={features.get('camelot', '?')}{rms_text}"
        )

    return features
