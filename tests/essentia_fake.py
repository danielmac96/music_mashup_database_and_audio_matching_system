"""A librosa-backed stand-in for the Essentia analyser, for the test suite.

Essentia is the app's only analyser and has no Windows wheels, so the suite
cannot rely on it being installed. This module answers in Essentia's payload
shape (the four core groups ``rhythm``, ``tonal``, ``loudness`` and
``spectral``; see analysis/project.core_from_essentia) by measuring the audio
with librosa, so tests that analyse synthetic audio still get real tempo, key
and timbre readings from it.

``tests/conftest.py`` installs it for every test in place of
``analysis.essentia_groups.analyze_file_essentia`` (and makes ``available()``
true). A test marked ``real_essentia`` gets the real thing instead.

Like the real one it reads and writes the feature cache per step (so cache
behaviour is testable, and the structure stage finds the "Essentia" beat grid
it would in production) and decodes only when some step is not cached.
Deliberately simpler:
  * ``frames_out`` is left empty — no per-frame sidecar is written, so the
    fallback cut runs on librosa transforms;
  * the extras (melody, EffNet tags) are never produced.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

CORE = ("rhythm", "tonal", "loudness", "spectral")


def _rhythm(y, sr, hop):
    import librosa
    from analysis import frames
    from analysis.analyze import _pick_beat_phase, beat_grid_confidence, fitted_bpm
    tempo, beats = frames.beat_track(y, sr, hop)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    onset_env = frames.onset_env(y, sr, hop)
    return {
        "bpm": fitted_bpm(beat_times, float(np.atleast_1d(tempo)[0])),
        "bpm_confidence": beat_grid_confidence(beat_times, onset_env, beats),
        "beat_times": [round(float(t), 4) for t in beat_times],
        "beat_phase": _pick_beat_phase(onset_env, beats),
    }


def _tonal(y, sr, hop):
    from analysis import frames
    from analysis.analyze import key_from_chroma
    return key_from_chroma(frames.chroma_cqt(y, sr, hop).mean(axis=1))


def _loudness(y, sr, hop):
    from analysis import frames
    return {"loudness_rms": float(round(float(frames.rms(y, hop).mean()), 6))}


def _spectral(y, sr, hop, path):
    import librosa
    from analysis import frames
    from analysis.analyze import _step_waveform
    from analysis.quality import BAND_EDGES, HF_BAND_HZ, band_energy
    from config import N_MFCC
    power = frames.power_stats(y, sr, BAND_EDGES, HF_BAND_HZ, n_fft=2048, hop=hop)
    mfcc = frames.mfcc(y, sr, N_MFCC, hop)
    bands = band_energy(path)
    return {
        "energy": float(round(power["mean"], 6)),
        "mfcc": [round(float(v), 4) for v in mfcc.mean(axis=1)],
        "spectral_centroid": float(round(float(
            librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop).mean()), 2)),
        "spectral_rolloff": float(round(float(
            librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop).mean()), 2)),
        "zero_crossing_rate": float(round(float(
            librosa.feature.zero_crossing_rate(y, hop_length=hop).mean()), 6)),
        "band_energy": bands if any(bands) else None,
        **_step_waveform(y),
    }


def analyze_file_essentia(path, cache=None, timings: Optional[dict] = None,
                          on_progress=None, extra_steps=(), frames_out=None,
                          force_steps=(), only_steps=()) -> dict:
    """Same signature as analysis.essentia_groups.analyze_file_essentia."""
    from analysis.decode import load_mono
    from config import HOP_LENGTH, SAMPLE_RATE

    wanted = [s for s in (only_steps or CORE)]
    out: dict = {}
    cached_steps = []
    if cache is not None:
        for name in wanted:
            if name in force_steps:
                continue
            hit = cache.get(name)
            if hit is not None:
                out[name] = hit
                cached_steps.append(name)
    if timings is not None:
        timings["cached_steps"] = list(cached_steps)
        timings["failed_steps"] = []
    todo = [n for n in wanted if n not in cached_steps]
    if not todo:
        return out

    t0 = time.perf_counter()
    y = load_mono(Path(path), sr=SAMPLE_RATE)
    if timings is not None:
        timings["load"] = (time.perf_counter() - t0) * 1000.0
        timings["audio_secs"] = len(y) / float(SAMPLE_RATE)
    sr, hop = SAMPLE_RATE, HOP_LENGTH
    steps = {
        "rhythm": lambda: _rhythm(y, sr, hop),
        "tonal": lambda: _tonal(y, sr, hop),
        "loudness": lambda: _loudness(y, sr, hop),
        "spectral": lambda: _spectral(y, sr, hop, path),
    }
    for name in todo:
        t1 = time.perf_counter()
        try:
            result = steps[name]()
        except Exception:  # noqa: BLE001 — a failed group is an absent one
            if timings is not None:
                timings["failed_steps"].append(name)
            continue
        ms = (time.perf_counter() - t1) * 1000.0
        out[name] = result
        if cache is not None:
            cache.put(name, result, ms)
        if timings is not None:
            timings[name] = ms
    return out


def install(monkeypatch) -> None:
    """Swap the fake in for the real analyser (tests/conftest.py)."""
    from analysis import essentia_groups
    monkeypatch.setattr(essentia_groups, "analyze_file_essentia", analyze_file_essentia)
    monkeypatch.setattr(essentia_groups, "available", lambda: True)
    monkeypatch.setattr(essentia_groups, "version", lambda: "test-stand-in")
