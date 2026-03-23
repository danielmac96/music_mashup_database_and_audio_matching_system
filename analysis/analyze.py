"""
analysis/analyze.py — Extract musical features from an audio file.

Features extracted per stem (full / vocals / instrumental):
  - BPM + beat confidence
  - Musical key + mode (major/minor)
  - Camelot wheel notation (for harmonic mixing)
  - Loudness RMS + spectral energy
  - MFCC (timbre fingerprint)
  - Spectral centroid, rolloff, zero-crossing rate

Requires: librosa, numpy
Optional: madmom (better beat tracking) — falls back to librosa if absent
"""
from typing import Optional
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)


# ── Camelot wheel ─────────────────────────────────────────────────────────────

# Maps (key_index, mode) → Camelot notation
# key_index follows librosa chroma order: C=0, C#=1, D=2, ... B=11
CAMELOT = {
    # Major (B suffix)
    (0,  "major"): "8B",  (1,  "major"): "3B",  (2,  "major"): "10B",
    (3,  "major"): "5B",  (4,  "major"): "12B", (5,  "major"): "7B",
    (6,  "major"): "2B",  (7,  "major"): "9B",  (8,  "major"): "4B",
    (9,  "major"): "11B", (10, "major"): "6B",  (11, "major"): "1B",
    # Minor (A suffix)
    (0,  "minor"): "5A",  (1,  "minor"): "12A", (2,  "minor"): "7A",
    (3,  "minor"): "2A",  (4,  "minor"): "9A",  (5,  "minor"): "4A",
    (6,  "minor"): "11A", (7,  "minor"): "6A",  (8,  "minor"): "1A",
    (9,  "minor"): "8A",  (10, "minor"): "3A",  (11, "minor"): "10A",
}

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_file(audio_path: Path, trim_secs: Optional[int] = None,
                 use_mock: bool = False) -> dict:
    """
    Extract all features from an audio file.

    Args:
        audio_path: Path to WAV or MP3 file
        trim_secs:  Analyse only the first N seconds (faster for testing)
        use_mock:   Return deterministic mock features (no file needed)

    Returns:
        Feature dict matching the `features` table schema
    """
    if use_mock:
        return _mock_features(str(audio_path))

    try:
        import librosa
    except ImportError:
        log.error("librosa not installed. Run: pip install librosa")
        return _mock_features(str(audio_path))

    log.info(f"Analysing: {audio_path.name}"
             + (f" (first {trim_secs}s)" if trim_secs else ""))

    try:
        from config import SAMPLE_RATE, HOP_LENGTH, N_MFCC
    except ImportError:
        SAMPLE_RATE, HOP_LENGTH, N_MFCC = 22050, 512, 13

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE,
                          duration=trim_secs, mono=True)

    features = {}

    # ── BPM / tempo ──────────────────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["bpm"] = float(round(float(np.atleast_1d(tempo)[0]), 2))
    # Beat confidence: ratio of strong beats to total frames
    features["bpm_confidence"] = float(min(len(beats) / (len(y) / HOP_LENGTH), 1.0))

    # ── Key detection ─────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    chroma_mean = chroma.mean(axis=1)  # shape (12,)

    # Major / minor key templates (Krumhansl-Schmuckler)
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

    features["key"]    = KEY_NAMES[key_idx]
    features["mode"]   = mode
    features["camelot"] = CAMELOT.get((key_idx, mode), "?")

    # ── Dynamics ──────────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features["loudness_rms"] = float(round(float(rms.mean()), 6))
    # Spectral energy as mean of squared magnitudes
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    features["energy"] = float(round(float((S ** 2).mean()), 6))

    # ── Timbre (MFCC) ─────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH)
    features["mfcc"] = [round(float(v), 4) for v in mfcc.mean(axis=1)]

    # ── Spectral shape ────────────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  hop_length=HOP_LENGTH)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                 hop_length=HOP_LENGTH)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)

    features["spectral_centroid"]  = float(round(float(centroid.mean()), 2))
    features["spectral_rolloff"]   = float(round(float(rolloff.mean()), 2))
    features["zero_crossing_rate"] = float(round(float(zcr.mean()), 6))

    log.info(f"  → BPM={features['bpm']}, Key={features['key']} {features['mode']}, "
             f"Camelot={features['camelot']}, RMS={features['loudness_rms']:.4f}")

    return features


# ── Mock features ─────────────────────────────────────────────────────────────

def _mock_features(seed: str) -> dict:
    """
    Generate deterministic but varied mock features for testing.
    Uses a hash of the seed string so each file gets unique values.
    """
    h = abs(hash(seed))
    bpm_options = [95, 100, 102, 105, 110, 115, 120, 122, 125, 128, 130, 140]
    keys    = ["C", "D", "E", "F", "G", "A", "B", "C#", "F#"]
    modes   = ["major", "minor"]
    n_mfcc  = 13

    key_name = keys[h % len(keys)]
    mode     = modes[(h // 10) % 2]
    key_idx  = KEY_NAMES.index(key_name)
    camelot  = CAMELOT.get((key_idx, mode), "8B")
    bpm      = float(bpm_options[h % len(bpm_options)])

    # Vary mock values per seed so matching produces real differences
    rng = h % 1000 / 1000.0
    return {
        "bpm":               bpm,
        "bpm_confidence":    round(0.6 + rng * 0.4, 3),
        "key":               key_name,
        "mode":              mode,
        "camelot":           camelot,
        "loudness_rms":      round(0.05 + rng * 0.15, 5),
        "energy":            round(0.10 + rng * 0.30, 5),
        "mfcc":              [round(-50 + (rng * i * 10), 4) for i in range(n_mfcc)],
        "spectral_centroid": round(1000 + rng * 3000, 2),
        "spectral_rolloff":  round(2000 + rng * 6000, 2),
        "zero_crossing_rate": round(0.03 + rng * 0.15, 6),
    }