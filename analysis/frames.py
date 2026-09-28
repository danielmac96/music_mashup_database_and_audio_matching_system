"""
analysis/frames.py — the expensive transforms, computed once per decoded signal.

analyze_file, detect_sections and the quality pass each asked librosa for the
same things on the same audio: a beat track and an onset envelope (tempo step,
then structure), chroma (key step, then structure — and the structure pass
again on each stem), MFCC (timbre step, then structure), frame RMS (dynamics,
then structure's vocal activity) and a 2048/512 power spectrogram (dynamics,
then band occupancy, then HF loss — per stem).

Each function below is exactly the librosa call its callers made, with the same
arguments, so the numbers are unchanged; ``decode.memo`` remembers the result
against the signal when that signal came from the decode cache, and computes it
straight through otherwise. Results are shared between callers: never mutate
one.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from analysis.decode import memo


def beat_track(y: np.ndarray, sr: int, hop: int):
    """(tempo, beat_frames) — librosa.beat.beat_track(y, sr, hop_length)."""
    def _run():
        import librosa
        return librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    return memo(y, ("beat_track", sr, hop), _run)


def onset_env(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    def _run():
        import librosa
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return memo(y, ("onset_strength", sr, hop), _run)


def chroma_cqt(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    def _run():
        import librosa
        return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    return memo(y, ("chroma_cqt", sr, hop), _run)


def mfcc(y: np.ndarray, sr: int, n_mfcc: int, hop: int) -> np.ndarray:
    def _run():
        import librosa
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
    return memo(y, ("mfcc", sr, n_mfcc, hop), _run)


def rms(y: np.ndarray, hop: int) -> np.ndarray:
    """librosa.feature.rms(y, hop_length) — shape (1, frames)."""
    def _run():
        import librosa
        return librosa.feature.rms(y=y, hop_length=hop)
    return memo(y, ("rms", hop), _run)


def power_stats(y: np.ndarray, sr: int, band_edges: Sequence[float],
                hf_hz: float, n_fft: int = 2048, hop: int = 512) -> dict:
    """Everything the pipeline reads off one |STFT|² — without keeping it.

    The spectrogram itself is ~40 MB for a 4-minute track, so what is
    remembered is only the handful of numbers taken from it, each computed
    with the exact expression its original caller used:

      mean        (S ** 2).mean()               — _step_dynamics "energy"
      bands       S2[mask].sum() per band        — quality._band_energy
      hf, total   S2[freqs >= hf_hz].sum(), S2.sum() — quality._hf_loss
      band_frames the same per-band power per frame, (bands, frames) float32
                  — a section's occupancy is a slice of it (structure.py)
    """
    def _run():
        import librosa
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        S2 = S ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        bands = []
        band_frames = np.zeros((len(band_edges) - 1, S2.shape[1]), dtype=np.float32)
        for i, (lo, hi) in enumerate(zip(band_edges[:-1], band_edges[1:])):
            mask = (freqs >= lo) & (freqs < hi)
            bands.append(float(S2[mask].sum()) if mask.any() else 0.0)
            if mask.any():
                band_frames[i] = S2[mask].sum(axis=0)
        band_frames.setflags(write=False)
        hf_mask = freqs >= hf_hz
        return {
            "mean": float(S2.mean()),
            "bands": bands,
            "hf": float(S2[hf_mask].sum()),
            "total": float(S2.sum()),
            "band_frames": band_frames,
        }
    return memo(y, ("power_stats", sr, n_fft, hop, tuple(band_edges), float(hf_hz)), _run)
