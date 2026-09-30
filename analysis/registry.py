"""
analysis/registry.py — every cached feature group, and what invalidates it.

A group is one unit of analysis whose output is stored in ``feature_cache``
under the hash of the audio it read (analysis/cache.py). A stored result is
reused while three things hold:

  * the audio is byte-for-byte the same (the content hash),
  * the group's ``version`` is the one declared here,
  * its ``params()`` — the config values the code reads — hash the same.

**Bump a group's version whenever its code changes what it returns.** That
recomputes that group, on every track, on the next analysis, and nothing else.
Changing a config value it reads needs no bump: params() picks it up. Forgetting
a bump is the one way this cache serves a stale answer, so a change to any
function a group calls (analysis/analyze.py step, analysis/quality.py,
analysis/structure.py, analysis/frames.py) should come with one.

Groups over one file are keyed by that file's hash; groups over several files
(residual vocal ratio: vocals + mix; structure: mix + stems) by a combination
of their inputs' hashes, so a re-separated stem recomputes both.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    version: int
    analyzer: str
    tier: int
    params: Callable[[], dict]
    description: str
    # analyze_file step this group caches, when it is one.
    step: Optional[str] = None

    def params_hash(self) -> str:
        blob = json.dumps(self.params(), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


def _analysis_params(**extra) -> Callable[[], dict]:
    def _p() -> dict:
        from config import BEAT_TRIM_SECS, HOP_LENGTH, SAMPLE_RATE
        return {"sr": SAMPLE_RATE, "hop": HOP_LENGTH, "trim_secs": BEAT_TRIM_SECS,
                **{k: (v() if callable(v) else v) for k, v in extra.items()}}
    return _p


def _n_mfcc() -> int:
    from config import N_MFCC
    return N_MFCC


def _quality_params() -> dict:
    from analysis.quality import BAND_EDGES, MAX_SECS, QUALITY_SR
    return {"sr": QUALITY_SR, "edges": list(BAND_EDGES), "max_secs": MAX_SECS}


def _residual_params() -> dict:
    from analysis.quality import MAX_SECS, QUALITY_SR
    return {"sr": QUALITY_SR, "max_secs": MAX_SECS}


def _structure_params() -> dict:
    from config import (HOP_LENGTH, SAMPLE_RATE, SECTION_MAX_COUNT,
                        SECTION_MIN_LEN_SECS, SECTION_SIM_THRESHOLD)
    return {"sr": SAMPLE_RATE, "hop": HOP_LENGTH,
            "min_len": SECTION_MIN_LEN_SECS, "max_count": SECTION_MAX_COUNT,
            "sim": SECTION_SIM_THRESHOLD}


def _essentia_params(**extra) -> Callable[[], dict]:
    """Every Essentia group's params carry the Essentia version: an upgrade
    can move any estimator, and must recompute rather than mix."""
    def _p() -> dict:
        try:
            from analysis.essentia_groups import version
            ver = version()
        except Exception:  # noqa: BLE001
            ver = None
        return {"essentia": ver,
                **{k: (v() if callable(v) else v) for k, v in extra.items()}}
    return _p


def _key_profile() -> str:
    from config import current_essentia_key_profile
    return current_essentia_key_profile()


def _rhythm_method() -> str:
    from config import current_essentia_rhythm_method
    return current_essentia_rhythm_method()


def _band_edges() -> list:
    from analysis.quality import BAND_EDGES
    return list(BAND_EDGES)


def _ml_models() -> list:
    from analysis.ml_models import catalogue_ids
    return catalogue_ids()


def _key_voters() -> list:
    from analysis.essentia_groups import KEY_PROFILES
    return list(KEY_PROFILES)


_GROUPS = (
    # v2: BPM fitted through the beats, not librosa's tempogram bin.
    FeatureGroup("librosa.tempo", 2, "librosa", 1, _analysis_params(),
                 "BPM, grid confidence, beat times, beat phase", step="tempo"),
    FeatureGroup("librosa.key", 1, "librosa", 1, _analysis_params(),
                 "Krumhansl key, mode, Camelot, key confidence", step="key"),
    FeatureGroup("librosa.dynamics", 1, "librosa", 1, _analysis_params(),
                 "mean RMS loudness, mean spectral energy", step="dynamics"),
    FeatureGroup("librosa.timbre", 1, "librosa", 1, _analysis_params(n_mfcc=_n_mfcc),
                 "mean MFCC, spectral centroid/rolloff, ZCR", step="timbre"),
    FeatureGroup("librosa.waveform", 1, "librosa", 1, _analysis_params(),
                 "360-point normalised RMS envelope", step="waveform"),
    FeatureGroup("librosa.bands", 1, "librosa", 1, _quality_params,
                 "8-band energy occupancy (analysis/quality.band_energy)"),
    FeatureGroup("librosa.residual", 1, "librosa", 2, _residual_params,
                 "residual vocal ratio of a bed (vocals + mix)"),
    # v2 (phase 4): sections also carry vocal_activity, per-stem band
    # occupancy and, given a melody, the sung range. v3: track and section
    # BPM fitted through the beats.
    FeatureGroup("librosa.structure", 3, "librosa", 1, _structure_params,
                 "sections: boundaries, labels, per-section measurements "
                 "(mix + vocal/instrumental/bass stems)"),
    # ── Essentia (analysis/essentia_groups.py) ────────────────────────────────
    FeatureGroup("essentia.rhythm", 1, "essentia", 1,
                 _essentia_params(method=_rhythm_method),
                 "BPM + beat grid (RhythmExtractor2013), grid confidence, kick-band "
                 "beat phase, Percival + histogram votes, onset rate, danceability",
                 step="rhythm"),
    FeatureGroup("essentia.tonal", 1, "essentia", 1,
                 _essentia_params(profile=_key_profile, voters=_key_voters),
                 "key per profile + cross-profile consensus, tuning, chords, "
                 "12-bin HPCP", step="tonal"),
    FeatureGroup("essentia.loudness", 1, "essentia", 1, _essentia_params(),
                 "EBU R128 LUFS + LRA, true peak, ReplayGain, dynamic complexity, "
                 "crest, stereo width, frame RMS", step="loudness"),
    FeatureGroup("essentia.spectral", 1, "essentia", 1,
                 _essentia_params(n_mfcc=_n_mfcc, edges=_band_edges),
                 "MFCC, centroid/rolloff/ZCR, flux, flatness, HFC, contrast, "
                 "complexity, moments, dissonance, 8- and 3-band energy, envelope",
                 step="spectral"),
    FeatureGroup("essentia.melody", 1, "essentia", 2,
                 _essentia_params(hop=128, frame=1024, step=0.05),
                 "vocal stem only: sung pitch (PitchMelodia), 50 ms f0 curve, "
                 "sung range and centre", step="melody"),
    FeatureGroup("essentia.effnet", 1, "essentia", 2,
                 _essentia_params(models=_ml_models),
                 "full mix only: Discogs-EffNet genre (400 styles) and tags — "
                 "voice, gender, danceable, tonal, bright, seven moods, "
                 "mood/theme and instrument top-5", step="effnet"),
    # v3: section BPM fitted through the beats.
    FeatureGroup("essentia.structure", 3, "essentia", 1, _structure_params,
                 "sections on the Essentia beat grid (the librosa segmenter fed "
                 "essentia.rhythm's beats and phase)"),
)

GROUPS: Dict[str, FeatureGroup] = {g.name: g for g in _GROUPS}

# analyze_file step name -> its librosa group; analyze_file_essentia step -> its
# Essentia group. The two share step names only by accident, so they are kept apart.
STEP_GROUPS: Dict[str, FeatureGroup] = {
    g.step: g for g in _GROUPS if g.step and g.analyzer == "librosa"}
ESSENTIA_STEP_GROUPS: Dict[str, FeatureGroup] = {
    g.step: g for g in _GROUPS if g.step and g.analyzer == "essentia"}


def group(name: str) -> FeatureGroup:
    return GROUPS[name]


def describe() -> list[dict]:
    """The registry as data (for an API or a log line)."""
    return [{"name": g.name, "version": g.version, "analyzer": g.analyzer,
             "tier": g.tier, "params": g.params(), "params_hash": g.params_hash(),
             "description": g.description} for g in _GROUPS]
