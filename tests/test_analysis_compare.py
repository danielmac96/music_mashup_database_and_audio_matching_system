"""analysis/compare.py — pitch-class spelling, and the allin1 benchmark's
definition of when two analyses of a track agree."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.compare import (  # noqa: E402
    boundary_prf, bpm_relation, downbeat_agreement, pitch_class,
)


@pytest.mark.parametrize("a,b,rel", [
    (128.0, 128.0, "same"),
    (126.0, 128.0, "same"),          # within 2%
    (122.0, 128.0, "other"),
    (174.0, 87.0, "double"),
    (64.0, 128.0, "half"),
    (135.0, 90.0, "three_halves"),
    (80.0, 120.0, "two_thirds"),
    (None, 128.0, None),
    (0.0, 128.0, None),
])
def test_bpm_relation_names_metrical_folds(a, b, rel):
    assert bpm_relation(a, b) == rel


def test_pitch_class_spellings():
    assert pitch_class("Bb") == pitch_class("A#") == 10
    assert pitch_class("x") is None


def test_boundary_prf_matches_each_reference_once():
    # Two estimates near one reference: only one may hit it.
    prf = boundary_prf([10.0, 10.4, 30.0], [10.2, 50.0], window=0.5)
    assert prf["hits"] == 1
    assert prf["precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert prf["recall"] == 0.5


def test_boundary_prf_window_and_trim():
    est, ref = [0.0, 20.0, 41.0, 100.0], [0.0, 22.0, 40.8, 100.0]
    assert boundary_prf(est, ref, window=0.5, trim=(0.0, 100.0))["f"] == 0.5
    assert boundary_prf(est, ref, window=3.0, trim=(0.0, 100.0))["f"] == 1.0
    # untrimmed, the track edges count as free hits
    assert boundary_prf(est, ref, window=0.5)["hits"] == 3


def test_boundary_prf_empty_is_zero_not_an_error():
    assert boundary_prf([], [1.0])["f"] == 0.0
    assert boundary_prf([1.0], [])["f"] == 0.0


def test_downbeat_agreement():
    a = [0.0, 1.875, 3.75, 5.625]
    assert downbeat_agreement(a, [0.03, 1.9, 3.75, 9.0]) == 0.75
    assert downbeat_agreement(a, []) is None


