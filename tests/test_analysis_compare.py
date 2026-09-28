"""analysis/compare.py — the benchmark's (and later the shadow analyser's)
definition of when two analyses of a track agree."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.compare import (  # noqa: E402
    boundaries_from_sections, boundary_prf, bpm_relation, downbeat_agreement,
    key_relation, mirex_key_score, parse_key_text, pitch_class, tally,
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


@pytest.mark.parametrize("a,b,rel", [
    (("C", "major"), ("C", "major"), "same"),
    (("Db", "minor"), ("C#", "minor"), "same"),     # enharmonic spellings
    (("C", "major"), ("A", "minor"), "relative"),
    (("A", "minor"), ("C", "major"), "relative"),
    (("F#", "minor"), ("A", "major"), "relative"),
    (("C", "major"), ("G", "major"), "fifth"),
    (("G", "major"), ("C", "major"), "fifth"),
    (("C", "major"), ("C", "minor"), "parallel"),
    (("C", "major"), ("D", "major"), "other"),
    (("C", "major"), ("E", "minor"), "other"),
    (("C", "major"), (None, None), None),
])
def test_key_relation(a, b, rel):
    assert key_relation(a[0], a[1], b[0], b[1]) == rel


def test_mirex_weights():
    assert mirex_key_score("same") == 1.0
    assert mirex_key_score("fifth") == 0.5
    assert mirex_key_score("relative") == 0.3
    assert mirex_key_score("parallel") == 0.2
    assert mirex_key_score("other") == 0.0
    assert mirex_key_score(None) is None


@pytest.mark.parametrize("text,expected", [
    ("Am", ("A", "minor")),
    ("A minor", ("A", "minor")),
    ("F#m", ("F#", "minor")),
    ("Dbmaj", ("C#", "major")),
    ("C", ("C", "major")),
    ("8B", ("C", "major")),       # Camelot: 8B = C major
    ("8A", ("A", "minor")),       # 8A = A minor
    ("5A", ("C", "minor")),
    ("1A", ("G#", "minor")),      # Ab minor
    ("11B", ("A", "major")),
    ("7A", ("D", "minor")),
    ("", (None, None)),
    ("banana", (None, None)),
    ("1d", (None, None)),         # Open Key is not guessed at
])
def test_parse_key_text(text, expected):
    assert parse_key_text(text) == expected


def test_parse_key_text_agrees_with_the_analysers_camelot_table():
    from analysis.analyze import CAMELOT, KEY_NAMES
    for (idx, mode), code in CAMELOT.items():
        assert parse_key_text(code) == (KEY_NAMES[idx], mode), code


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


def test_tally_and_section_boundaries():
    assert tally(["same", "same", None, "half"]) == {"same": 2, "half": 1}
    secs = [{"start_sec": 30.0}, {"start_sec": 0.0}, {"start_sec": 12.5}]
    assert boundaries_from_sections(secs) == [12.5, 30.0]
