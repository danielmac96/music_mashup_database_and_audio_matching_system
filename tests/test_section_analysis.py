"""Section-level tempo, grid, energy shape and class (P2.1).

The helpers are deliberately pure so they can be tested without librosa or a
decode: everything detect_sections adds per section is computed from arrays it
already holds for the boundary work, and the decisions worth pinning are the
fallbacks and the thresholds, not numpy's arithmetic.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.structure import (  # noqa: E402
    SECTION_BPM_MIN_BEATS, _energy_shape, _phrase_length, _section_bpm,
    _section_class,
)


def _beats(bpm, n, start=0.0):
    """Perfectly steady beat times at the given tempo."""
    step = 60.0 / bpm
    return np.array([start + i * step for i in range(n)])


# ── per-section BPM ──────────────────────────────────────────────────────────

def test_a_steady_section_reports_its_own_tempo():
    bpm, source = _section_bpm(_beats(128.0, 32), track_bpm=120.0, confidence=0.9)
    assert source == "section_estimate"
    assert bpm == pytest.approx(128.0, abs=0.1)


def test_a_short_section_inherits_the_track_tempo():
    """Two bars of beats is not a tempo measurement, and saying so matters more
    than the number: a caller weighing a match should know which it has."""
    bpm, source = _section_bpm(_beats(128.0, SECTION_BPM_MIN_BEATS - 1),
                               track_bpm=120.0, confidence=0.9)
    assert (bpm, source) == (120.0, "track_fallback")


def test_an_unsteady_section_inherits_the_track_tempo():
    bpm, source = _section_bpm(_beats(128.0, 32), track_bpm=120.0, confidence=0.05)
    assert (bpm, source) == (120.0, "track_fallback")


def test_half_time_is_folded_back_to_the_track_tempo():
    """A half-time bridge is the same grid read at half speed, not a tempo
    change. Reporting 64 against a 128 library would push every match for that
    section into a stretch it does not need."""
    bpm, source = _section_bpm(_beats(64.0, 32), track_bpm=128.0, confidence=0.9)
    assert bpm == pytest.approx(128.0)
    assert source == "section_estimate"


def test_double_time_is_folded_back_too():
    bpm, _ = _section_bpm(_beats(256.0, 32), track_bpm=128.0, confidence=0.9)
    assert bpm == pytest.approx(128.0)


def test_a_genuine_tempo_change_is_not_folded():
    """140 is not a fold of 128, so it is reported as measured."""
    bpm, source = _section_bpm(_beats(140.0, 32), track_bpm=128.0, confidence=0.9)
    assert bpm == pytest.approx(140.0, abs=0.1)
    assert source == "section_estimate"


def test_no_track_tempo_and_no_usable_beats_gives_none():
    assert _section_bpm(np.array([1.0]), track_bpm=None, confidence=0.9) == (None, None)


def test_bpm_source_is_only_ever_one_of_the_two_documented_values():
    for beats, conf, track in ((_beats(128.0, 32), 0.9, 120.0),
                               (_beats(128.0, 4), 0.9, 120.0),
                               (_beats(128.0, 32), 0.01, 120.0)):
        _, source = _section_bpm(beats, track_bpm=track, confidence=conf)
        assert source in ("section_estimate", "track_fallback")


# ── energy shape ─────────────────────────────────────────────────────────────

def test_a_build_reads_as_increasing():
    slope, trend = _energy_shape(np.linspace(0.2, 0.9, 32), duration=15.0)
    assert trend == "increasing"
    assert slope > 0


def test_a_decay_reads_as_decreasing():
    slope, trend = _energy_shape(np.linspace(0.9, 0.2, 32), duration=15.0)
    assert trend == "decreasing"
    assert slope < 0


def test_a_flat_section_reads_as_stable():
    _, trend = _energy_shape(np.full(32, 0.7), duration=15.0)
    assert trend == "stable"


def test_noise_around_a_flat_level_is_still_stable():
    """Calling every gentle wobble a build would make the label useless for
    picking one."""
    rng = np.random.default_rng(0)
    _, trend = _energy_shape(0.7 + rng.normal(0, 0.02, 64), duration=20.0)
    assert trend == "stable"


def test_slope_is_fitted_not_taken_from_the_endpoints():
    """A drop's last beat can be a tail and its first a pickup; a difference of
    the two would flip the sign while saying nothing about the shape between."""
    curve = np.concatenate([[0.9], np.linspace(0.2, 0.85, 30), [0.1]])
    slope, trend = _energy_shape(curve, duration=15.0)
    assert trend == "increasing", "endpoint-differencing would have said decreasing"


def test_degenerate_curves_do_not_raise():
    assert _energy_shape(None, 10.0) == (None, "stable")
    assert _energy_shape(np.array([0.5]), 10.0) == (None, "stable")
    assert _energy_shape(np.linspace(0, 1, 10), 0.0) == (None, "stable")


# ── section class ────────────────────────────────────────────────────────────

def test_class_thresholds():
    assert _section_class(0.9) == "vocal"
    assert _section_class(0.02) == "instrumental"
    assert _section_class(0.25) == "mixed"


def test_missing_vocal_stem_is_unknown_not_instrumental():
    """The spec says not to match unknown sections unless explicitly enabled.
    Conflating "no stem" with "no vocal" would silently mark a whole
    un-separated library as instrumental and let it through that gate."""
    assert _section_class(None) == "unknown"


# ── phrase length ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bars,expected", [
    (8.0, 8.0), (7.8, 8.0), (16.2, 16.0), (4.0, 4.0), (32.0, 32.0), (1.0, 1.0),
])
def test_phrase_length_snaps_to_the_nearest_power_of_two(bars, expected):
    assert _phrase_length(bars) == expected


def test_a_very_long_section_reports_its_real_length():
    """Better than claiming a 200-bar section is a 64-bar phrase."""
    assert _phrase_length(200.0) == 200.0


def test_phrase_length_of_nothing_is_none():
    assert _phrase_length(0) is None
    assert _phrase_length(None) is None


# ── persistence ──────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    return models


def test_the_new_fields_round_trip(db):
    sid = db.upsert_song(title="T", artist="A", source_url="u://t")
    db.replace_sections(sid, [{
        "start_sec": 0.0, "end_sec": 30.0, "label": "chorus",
        "energy": 0.8, "vocal_presence": 0.7, "repetition": 2, "confidence": 0.9,
        "bpm": 128.0, "bpm_source": "section_estimate", "bpm_confidence": 0.82,
        "energy_absolute": 0.0431, "energy_slope": 0.0121,
        "energy_trend": "increasing",
        "beat_times": [0.0, 0.47, 0.94], "downbeats": [0.0, 1.88],
        "beat_count": 64, "bar_count": 16.0, "beats_per_bar": 4,
        "phrase_length_bars": 16.0, "section_class": "vocal",
    }])

    got = db.get_sections(sid)[0]
    assert got["bpm"] == 128.0
    assert got["bpm_source"] == "section_estimate"
    assert got["bpm_confidence"] == pytest.approx(0.82)
    assert got["energy_trend"] == "increasing"
    assert got["bar_count"] == 16.0
    assert got["beats_per_bar"] == 4
    assert got["section_class"] == "vocal"
    # Decoded back to lists, the way the chroma columns already are.
    assert got["beat_times"] == [0.0, 0.47, 0.94]
    assert got["downbeats"] == [0.0, 1.88]
    assert "beat_times_json" not in got


def test_sections_analysed_before_this_still_load(db):
    """A library analysed before P2.1 has NULLs here, and every reader has to
    treat that as "not measured" rather than as zero."""
    sid = db.upsert_song(title="T", artist="A", source_url="u://t")
    db.replace_sections(sid, [{
        "start_sec": 0.0, "end_sec": 30.0, "label": "verse",
        "energy": 0.5, "vocal_presence": 0.4, "repetition": 1, "confidence": 0.5,
    }])
    got = db.get_sections(sid)[0]
    assert got["bpm"] is None
    assert got["section_class"] is None
    assert "beat_times" not in got
