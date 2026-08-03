"""Tests for the learned scorer's pair-feature vector (matcher/features.py).

Pure numpy — no DB, no audio. The contract under test: pair_features produces
exactly the FEATURE_NAMES columns, deterministically, as finite floats, and
features_to_row preserves that order (train/serve alignment).
"""
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from matcher.features import (  # noqa: E402
    FEATURE_NAMES, features_to_row, pair_features,
)


def _feat(bpm=120.0, camelot="8A", loud=0.1, energy=0.5, mfcc=None,
          centroid=2000.0, rolloff=4000.0, zcr=0.05, song_id=1):
    return {
        "song_id": song_id, "bpm": bpm, "camelot": camelot,
        "loudness_rms": loud, "energy": energy, "mfcc": mfcc or [1.0] * 13,
        "spectral_centroid": centroid, "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr,
    }


def test_pair_features_covers_feature_names_exactly():
    feats = pair_features(_feat(), _feat(song_id=2), [], [])
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_pair_features_all_finite_floats():
    feats = pair_features(_feat(bpm=0, camelot="?"), _feat(song_id=2, mfcc=[]), [], [])
    for name in FEATURE_NAMES:
        v = feats[name]
        assert isinstance(v, float)
        assert math.isfinite(v)


def test_features_to_row_is_ordered_and_stable():
    feats = pair_features(_feat(), _feat(song_id=2), [], [])
    row = features_to_row(feats)
    assert len(row) == len(FEATURE_NAMES)
    assert row == [float(feats[n]) for n in FEATURE_NAMES]


def test_pair_features_is_deterministic():
    a, b = _feat(), _feat(song_id=2, bpm=124.0, camelot="9A")
    assert pair_features(a, b, [], []) == pair_features(a, b, [], [])


def test_identical_sides_give_neutral_deltas():
    same = _feat()
    feats = pair_features(same, dict(same, song_id=2), [], [])
    assert feats["bpm_min_diff"] == 0.0
    assert feats["camelot_distance"] == 0.0
    assert feats["energy_ratio"] == 1.0
    assert feats["loudness_diff"] == 0.0
    assert feats["spectral_centroid_diff"] == 0.0


def test_section_features_populate_from_structure():
    bed_sections = [
        {"energy": 0.3, "vocal_presence": 0.0},
        {"energy": 0.9, "vocal_presence": 0.1},
    ]
    top_sections = [{"energy": 0.5, "vocal_presence": 0.8}]
    feats = pair_features(_feat(), _feat(song_id=2), top_sections, bed_sections)
    assert feats["bed_section_count"] == 2.0
    assert feats["bed_energy_max"] == 0.9
    assert abs(feats["bed_energy_mean"] - 0.6) < 1e-9
    assert abs(feats["top_vocal_presence_mean"] - 0.8) < 1e-9


def test_camelot_distance_letters_and_ring():
    # Same code → 0; relative maj/minor (8A vs 8B) → 0.5 letter penalty; ring wrap.
    assert pair_features(_feat(camelot="8A"), _feat(song_id=2, camelot="8A"), [], [])["camelot_distance"] == 0.0
    assert pair_features(_feat(camelot="8A"), _feat(song_id=2, camelot="8B"), [], [])["camelot_distance"] == 0.5
    assert pair_features(_feat(camelot="1A"), _feat(song_id=2, camelot="12A"), [], [])["camelot_distance"] == 1.0
    assert pair_features(_feat(camelot="?"), _feat(song_id=2, camelot="8A"), [], [])["camelot_distance"] == 6.0
