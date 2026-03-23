"""
analysis/analyze.py — Extract musical features from an audio file.

Features: BPM, key, Camelot, loudness, energy, MFCC, spectral shape.
Requires: librosa, numpy
"""
from typing import Optional
import logging
import numpy as np
from pathlib import Path

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


def analyze_file(audio_path: Path, trim_secs: Optional[int] = None) -> dict:
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

    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE,
                          duration=trim_secs, mono=True)
    features = {}

    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["bpm"] = float(round(float(np.atleast_1d(tempo)[0]), 2))
    features["bpm_confidence"] = float(min(len(beats) / (len(y) / HOP_LENGTH), 1.0))

    # Key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
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
    features["key"]    = KEY_NAMES[key_idx]
    features["mode"]   = mode
    features["camelot"] = CAMELOT.get((key_idx, mode), "?")

    # Dynamics
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features["loudness_rms"] = float(round(float(rms.mean()), 6))
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    features["energy"] = float(round(float((S ** 2).mean()), 6))

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    features["mfcc"] = [round(float(v), 4) for v in mfcc.mean(axis=1)]

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features["spectral_centroid"]   = float(round(float(centroid.mean()), 2))
    features["spectral_rolloff"]    = float(round(float(rolloff.mean()), 2))
    features["zero_crossing_rate"]  = float(round(float(zcr.mean()), 6))

    log.info(f"  → BPM={features['bpm']}, Key={features['key']} {features['mode']}, "
             f"Camelot={features['camelot']}, RMS={features['loudness_rms']:.4f}")

    return features