"""T2.3 — the shared train/serve feature builder.

FEATURE_NAMES is the contract between training and inference: a model stores
the names it was trained on, and scoring with a different order silently feeds
every coefficient the wrong number. These tests pin the contract, the
degradation behaviour, and the rule that sub-scores are passed through rather
than re-derived (so the heuristic and the model can never drift apart).
"""
import importlib
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from matcher import match
    importlib.reload(match)
    from matcher import features
    importlib.reload(features)
    return models, match, features


def _feat(**kw):
    base = {
        "song_id": 1, "stem_type": "vocals", "bpm": 120.0, "bpm_confidence": 0.8,
        "camelot": "8B", "key": "C", "mode": "major", "key_confidence": 0.04,
        "loudness_rms": 0.1, "energy": 20.0, "mfcc": [-160.0] + [5.0] * 12,
        "spectral_centroid": 2000.0, "spectral_rolloff": 4000.0,
        "zero_crossing_rate": 0.05,
    }
    base.update(kw)
    return base


def _sections(n=3, vocal=0.8, energy=0.6):
    return [
        {"label": "intro", "start_sec": 0, "end_sec": 20, "energy": 0.2,
         "vocal_presence": 0.0, "confidence": 0.5, "repetition": 1},
        {"label": "chorus", "start_sec": 20, "end_sec": 52, "energy": energy,
         "vocal_presence": vocal, "confidence": 0.9, "repetition": 2},
        {"label": "verse", "start_sec": 52, "end_sec": 80, "energy": 0.4,
         "vocal_presence": vocal * 0.5, "confidence": 0.6, "repetition": 1},
    ][:n]


# ── the contract ─────────────────────────────────────────────────────────────

def test_feature_names_are_unique(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    assert len(features.FEATURE_NAMES) == len(set(features.FEATURE_NAMES))


def test_pair_features_returns_exactly_the_declared_names(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    f = features.pair_features(_feat(), _feat(stem_type="instrumental"),
                               _sections(), _sections())
    assert set(f) == set(features.FEATURE_NAMES)


def test_row_length_matches_the_declared_names(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    f = features.pair_features(_feat(), _feat(stem_type="instrumental"),
                               _sections(), _sections())
    assert len(features.features_to_row(f)) == len(features.FEATURE_NAMES)


def test_identical_inputs_give_identical_output(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    a, b, sa, sb = _feat(), _feat(stem_type="instrumental"), _sections(), _sections()
    assert features.pair_features(a, b, sa, sb) == features.pair_features(a, b, sa, sb)


# ── degradation: never a NaN, whatever is missing ────────────────────────────

@pytest.mark.parametrize("missing", [
    {},                                    # nothing missing
    {"mfcc": []},                          # unanalysed timbre
    {"bpm": None, "bpm_confidence": None},
    {"key_confidence": None, "camelot": ""},
    {"loudness_rms": None, "energy": None},
    {"spectral_centroid": None, "spectral_rolloff": None, "zero_crossing_rate": None},
])
def test_no_nan_or_inf_when_inputs_are_missing(tmp_path, monkeypatch, missing):
    _, _, features = _setup(tmp_path, monkeypatch)
    row = features.features_to_row(features.pair_features(
        _feat(**missing), _feat(stem_type="instrumental", **missing),
        _sections(), _sections()))
    assert all(math.isfinite(v) for v in row), dict(zip(features.FEATURE_NAMES, row))


def test_no_nan_when_a_pair_has_no_sections_at_all(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    row = features.features_to_row(features.pair_features(
        _feat(), _feat(stem_type="instrumental"), [], []))
    assert all(math.isfinite(v) for v in row)


def test_empty_features_still_produce_a_full_row(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    row = features.features_to_row(features.pair_features({}, {}, None, None))
    assert len(row) == len(features.FEATURE_NAMES)
    assert all(math.isfinite(v) for v in row)


# ── the terms T2.3 requires ──────────────────────────────────────────────────

def test_sub_scores_are_passed_through_not_re_derived(tmp_path, monkeypatch):
    """If these ever disagree, the heuristic ranking and the model's view of the
    same pair have drifted — the exact failure sub_scores exists to prevent."""
    _, match, features = _setup(tmp_path, monkeypatch)
    a, b = _feat(), _feat(stem_type="instrumental", bpm=124.0, camelot="9B")
    f = features.pair_features(a, b, _sections(), _sections())
    s = match.sub_scores(a, b)
    for k in ("bpm_score", "key_score", "energy_score", "timbre_score"):
        assert f[k] == pytest.approx(s[k])


def test_semitone_shift_uses_the_corrected_camelot_math(tmp_path, monkeypatch):
    """Relative major/minor needs no transposition (T1.2), so the model must not
    be told there is a 3-semitone gap there."""
    _, _, features = _setup(tmp_path, monkeypatch)
    rel = features.pair_features(_feat(camelot="8B"),
                                 _feat(stem_type="instrumental", camelot="8A"),
                                 _sections(), _sections())
    far = features.pair_features(_feat(camelot="8B"),
                                 _feat(stem_type="instrumental", camelot="11B"),
                                 _sections(), _sections())
    assert rel["abs_semitone_shift"] == 0
    assert far["abs_semitone_shift"] == 3


def test_unknown_key_does_not_masquerade_as_a_perfect_shift(tmp_path, monkeypatch):
    """compute_semitone_shift returns None for an unknown key. Coercing that to
    0.0 would tell the model 'no transposition needed' — the same value a
    perfectly matched pair gets."""
    _, _, features = _setup(tmp_path, monkeypatch)
    f = features.pair_features(_feat(camelot="?"),
                               _feat(stem_type="instrumental", camelot="8A"),
                               _sections(), _sections())
    assert f["abs_semitone_shift"] > 0
    assert f["semitone_shift_known"] == 0.0


def test_confidence_terms_are_carried_for_both_sides(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    f = features.pair_features(
        _feat(bpm_confidence=0.11, key_confidence=0.022),
        _feat(stem_type="instrumental", bpm_confidence=0.77, key_confidence=0.088),
        _sections(), _sections())
    assert f["top_bpm_confidence"] == pytest.approx(0.11)
    assert f["bed_bpm_confidence"] == pytest.approx(0.77)
    assert f["top_key_confidence"] == pytest.approx(0.022)
    assert f["bed_key_confidence"] == pytest.approx(0.088)


def test_section_terms_reflect_the_sections_that_would_be_used(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    quiet = features.pair_features(
        _feat(), _feat(stem_type="instrumental"),
        _sections(vocal=0.9), _sections(energy=0.2))
    loud = features.pair_features(
        _feat(), _feat(stem_type="instrumental"),
        _sections(vocal=0.9), _sections(energy=0.95))
    assert quiet["hook_energy_delta"] > loud["hook_energy_delta"]
    assert quiet["top_section_vocal_presence"] == pytest.approx(0.9)


def _bed(end_sec):
    return [{"label": "drop", "start_sec": 0, "end_sec": end_sec, "energy": 0.6,
             "vocal_presence": None, "confidence": 0.9, "repetition": 2}]


def test_duration_fit_is_one_when_the_bed_section_matches_the_vocal(tmp_path, monkeypatch):
    """Fit is judged after the bed is conformed to the vocal's tempo, using the
    same inst_duration / stretch_factor that build_pairings reports in the Plan."""
    _, _, features = _setup(tmp_path, monkeypatch)
    v = _sections()                                  # chorus 20-52 => 32s
    f = features.pair_features(_feat(bpm=120.0),
                               _feat(stem_type="instrumental", bpm=120.0),
                               v, _bed(32))
    assert f["duration_fit"] == pytest.approx(1.0, abs=0.02)


def test_duration_fit_falls_when_the_bed_section_is_the_wrong_length(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    v = _sections()                                  # 32s chorus
    good = features.pair_features(_feat(bpm=120.0),
                                  _feat(stem_type="instrumental", bpm=120.0),
                                  v, _bed(32))
    short = features.pair_features(_feat(bpm=120.0),
                                   _feat(stem_type="instrumental", bpm=120.0),
                                   v, _bed(8))
    assert short["duration_fit"] < good["duration_fit"]
    assert short["duration_fit"] == pytest.approx(0.25, abs=0.02)


def test_a_model_trained_on_a_different_feature_set_is_refused(tmp_path, monkeypatch):
    """model_score orders by the bundle's own names, so a stale name is simply
    absent from the dict and coerces to 0.0 — the model would keep scoring, on
    zeros, silently. Falling back to the heuristic is the honest failure."""
    _setup(tmp_path, monkeypatch)
    from matcher import model_scorer
    importlib.reload(model_scorer)

    assert model_scorer._feature_names_match(
        {"feature_names": list(_names(tmp_path, monkeypatch))}) is True
    assert model_scorer._feature_names_match(
        {"feature_names": ["energy_ratio", "mfcc_cosine"]}) is False
    assert model_scorer._feature_names_match({}) is False


def _names(tmp_path, monkeypatch):
    from matcher import features
    return features.FEATURE_NAMES


def test_all_features_are_finite_across_a_random_sweep(tmp_path, monkeypatch):
    _, _, features = _setup(tmp_path, monkeypatch)
    rng = np.random.default_rng(0)
    for _ in range(60):
        kw = lambda: dict(
            bpm=float(rng.choice([0.0, 60.0, 128.0, 200.0])),
            camelot=str(rng.choice(["8B", "?", "", "12A"])),
            loudness_rms=float(rng.choice([0.0, 0.05, 0.3])),
            mfcc=list(rng.normal(0, 50, 13)) if rng.random() > 0.3 else [],
            key_confidence=float(rng.choice([0.0, 0.05])),
        )
        secs = _sections() if rng.random() > 0.5 else []
        row = features.features_to_row(features.pair_features(
            _feat(**kw()), _feat(stem_type="instrumental", **kw()), secs, secs))
        assert all(math.isfinite(v) for v in row)
