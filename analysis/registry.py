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


def _n_mfcc() -> int:
    from config import N_MFCC
    return N_MFCC


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


def _frames_params() -> dict:
    """The sidecar's own format: its grid, its matrices and its layout version.
    A change to any of them recomputes the frames and nothing else."""
    from analysis.timeseries import ROWS, STEP_SECS, VERSION
    return {"step": round(STEP_SECS, 9), "rows": dict(ROWS), "fmt": VERSION}


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


def _allin1_params() -> dict:
    """allin1's version, the model actually configured, and the pooling: an
    upgrade or a different checkpoint recomputes rather than mixing two
    models' grids. The model is read live (config.current_allin1_model), so
    switching it invalidates these groups by itself."""
    from config import current_allin1_model
    try:
        from analysis.allin1_runner import POOLING, version
        ver, pooling = version(), POOLING
    except Exception:  # noqa: BLE001
        ver, pooling = None, None
    return {"allin1": ver, "model": current_allin1_model(), "pooling": pooling}


def _phrase_params() -> dict:
    from config import (HOP_LENGTH, PHRASE_MIN_SUNG_SECS,
                        PHRASE_PICKUP_MAX_BEATS, SAMPLE_RATE, current_float)
    return {"sr": SAMPLE_RATE, "hop": HOP_LENGTH,
            "gap": current_float("phrase_gap_secs"), "min_sung": PHRASE_MIN_SUNG_SECS,
            "pickup_max_beats": PHRASE_PICKUP_MAX_BEATS}


_GROUPS = (
    FeatureGroup("librosa.residual", 1, "librosa", 2, _residual_params,
                 "residual vocal ratio of a bed (vocals + mix)"),
    # v2 (phase 4): sections also carry vocal_activity, per-stem band
    # occupancy and, given a melody, the sung range. v3: track and section
    # BPM fitted through the beats. v4: a voiceless section is a drop or a
    # breakdown, never a verse (label_segments).
    # v5: chroma and MFCC come from Essentia's own per-frame output
    # (analysis/timeseries.py) instead of a second librosa CQT/STFT pass, and
    # the bass chroma is the bass stem's rather than a band-passed mix.
    FeatureGroup("librosa.structure", 5, "librosa", 1, _structure_params,
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
    # Not a step: nothing in analyze_file_essentia serves it. Its payload is the
    # sidecar's metadata (shape, content hash); the arrays are a .npz beside the
    # audio. essentia.tonal / essentia.spectral are unchanged by it.
    FeatureGroup("essentia.frames", 1, "essentia", 1,
                 _essentia_params(frames=_frames_params),
                 "per-frame chroma / MFCC / 8-band power / centroid-rolloff-flux-"
                 "zcr-rms for one stem, on the shared 46.44 ms grid. The payload "
                 "is the sidecar's metadata; the arrays live in a .npz beside "
                 "the audio (analysis/timeseries.py)"),
    # v3: section BPM fitted through the beats.
    # v5: see librosa.structure v5 — the frames are Essentia's now.
    FeatureGroup("essentia.structure", 5, "essentia", 1, _structure_params,
                 "sections on the Essentia beat grid, cut from the stored "
                 "per-frame chroma and MFCC (analysis/timeseries.py)"),
    # ── allin1 (analysis/allin1_runner.py, readme §5.5) ───────────────────────
    # v2: the summary also carries the activations at BEAT resolution
    # (beat_conf, downbeat_conf_beats) and each segment's full label
    # distribution. None of the three is derivable from a v1 payload — the
    # 100-fps arrays are dropped as soon as the summary exists — so a v1 row
    # must recompute rather than store NULLs for the new columns.
    FeatureGroup("allin1.analysis", 2, "allin1", 2, _allin1_params,
                 "full mix + the 4 Demucs sources: beats, downbeats, beat-in-bar, "
                 "BPM, labelled segments, activation and embedding summaries"),
    # v2: see librosa.structure v5 — it is the same detect_sections, so the
    # frames are Essentia's here too.
    FeatureGroup("allin1.structure", 2, "allin1", 2,
                 lambda: {**_structure_params(), **_allin1_params()},
                 "sections on allin1's segments and downbeats, measured by the "
                 "same per-section code as the Essentia cut"),
    FeatureGroup("vocal.phrases", 1, "librosa", 2, _phrase_params,
                 "vocal stem: sung phrase starts snapped to a downbeat, with the "
                 "signed pickup, grouped into 4/8/16-bar phrases"),
)

GROUPS: Dict[str, FeatureGroup] = {g.name: g for g in _GROUPS}

# analyze_file_essentia step name -> its Essentia group.
ESSENTIA_STEP_GROUPS: Dict[str, FeatureGroup] = {
    g.step: g for g in _GROUPS if g.step and g.analyzer == "essentia"}


def group(name: str) -> FeatureGroup:
    return GROUPS[name]


def describe() -> list[dict]:
    """The registry as data (for an API or a log line)."""
    return [{"name": g.name, "version": g.version, "analyzer": g.analyzer,
             "tier": g.tier, "params": g.params(), "params_hash": g.params_hash(),
             "description": g.description} for g in _GROUPS]
