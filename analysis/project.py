"""
analysis/project.py — turn an analyser's cached group payloads into the
``features`` row the matcher, the routes and the UI already read.

Two kinds of column:

  * **core** — the columns both analysers fill (bpm, beat grid, key, loudness,
    MFCC, spectral shape, band occupancy, waveform). Exactly one analyser owns
    them for a given row; ``features.analyzer`` says which. The matcher ranks
    several of them *against the library* (timbre z-scores, confidence
    percentiles), so a library must not mix analysers in the core — the switch
    to Essentia happens for every track at once (readme §9).
  * **extras** — what only Essentia measures (LUFS, true peak, tuning, chords,
    danceability, …). They are written whenever Essentia ran, whichever
    analyser owns the core, so shadow mode already fills them.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

ESSENTIA_CORE_STEPS = ("rhythm", "tonal", "loudness", "spectral")

# features columns written by write_extras, with the payload they come from.
EXTRA_COLUMNS = (
    "analyzer", "key_strength", "key_candidates_json", "bpm_candidates_json",
    "tuning_hz", "lufs", "lra", "true_peak", "replay_gain", "dynamic_complexity",
    "danceability", "onset_rate", "chords_json", "dissonance", "bands3_json",
    "descriptors_json",
)


def essentia_core_complete(payloads: Dict[str, dict]) -> bool:
    return all(payloads.get(s) for s in ESSENTIA_CORE_STEPS)


def core_from_essentia(payloads: Dict[str, dict]) -> dict:
    """The dict upsert_features takes, from the four Essentia groups. Keys
    match analysis.analyze.analyze_file's output exactly."""
    r, t = payloads["rhythm"], payloads["tonal"]
    lo, sp = payloads["loudness"], payloads["spectral"]
    return {
        "bpm": r.get("bpm"), "bpm_confidence": r.get("bpm_confidence"),
        "beat_times": r.get("beat_times"), "beat_phase": r.get("beat_phase") or 0,
        "key": t.get("key"), "mode": t.get("mode"), "camelot": t.get("camelot"),
        "key_confidence": t.get("key_confidence"),
        "loudness_rms": lo.get("loudness_rms"), "energy": sp.get("energy"),
        "mfcc": sp.get("mfcc"),
        "spectral_centroid": sp.get("spectral_centroid"),
        "spectral_rolloff": sp.get("spectral_rolloff"),
        "zero_crossing_rate": sp.get("zero_crossing_rate"),
        "waveform_rms": sp.get("waveform_rms"),
        "band_energy": sp.get("band_energy"),
    }


def _dumps(v) -> Optional[str]:
    return json.dumps(v) if v is not None else None


def extras_from_essentia(payloads: Dict[str, dict], analyzer: str) -> dict:
    """The extra columns. Every value is None (unmeasured) for a group that did
    not run — the extras never outlive the analysis that produced them."""
    r = payloads.get("rhythm") or {}
    t = payloads.get("tonal") or {}
    lo = payloads.get("loudness") or {}
    sp = payloads.get("spectral") or {}
    spectral = dict(sp.get("spectral") or {})
    descriptors = {
        # Kept together rather than as twenty columns: nothing filters or sorts
        # on these yet, and a column is cheap to promote when something does.
        "spectral": spectral or None,
        "key_consensus": t.get("key_consensus"),
        "hpcp": t.get("hpcp"),
        "rhythm_confidence": r.get("rhythm_confidence"),
        "short_term_lufs_max": lo.get("short_term_lufs_max"),
        "sample_peak_db": lo.get("sample_peak_db"),
        "crest_db": lo.get("crest_db"),
        "stereo_width": lo.get("stereo_width"),
    }
    has_any = bool(payloads)
    return {
        "analyzer": analyzer,
        "key_strength": t.get("key_strength"),
        "key_candidates_json": _dumps(t.get("key_candidates")),
        "bpm_candidates_json": _dumps(r.get("bpm_candidates")),
        "tuning_hz": t.get("tuning_hz"),
        "lufs": lo.get("lufs"),
        "lra": lo.get("lra"),
        "true_peak": lo.get("true_peak"),
        "replay_gain": lo.get("replay_gain"),
        "dynamic_complexity": lo.get("dynamic_complexity"),
        "danceability": r.get("danceability"),
        "onset_rate": r.get("onset_rate"),
        "chords_json": _dumps(t.get("chords")),
        "dissonance": spectral.get("dissonance"),
        "bands3_json": _dumps(sp.get("bands3")),
        "descriptors_json": _dumps(descriptors) if has_any else None,
    }
