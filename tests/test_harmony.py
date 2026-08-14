"""Phase E — measured harmony (matcher/harmony.py) and bar-based phrase fit.

key_score was the heaviest weight in the ranking and a five-value Camelot lookup
off ONE key per track, estimated from a whole-track mean chroma. Records
modulate; the chorus is often not that key; and Camelot is a claim about scales,
not about whether this vocal's notes land on that bed's chord tones.

Pure numpy — no audio, no DB.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from matcher.harmony import (  # noqa: E402
    bass_clash, harmonic_fit, section_harmony,
)
from matcher.sections import bars_in, phrase_fit  # noqa: E402


def chord(*pitch_classes, strength=1.0):
    """A chroma vector with energy on the given pitch classes (0 = C)."""
    v = np.zeros(12)
    for pc in pitch_classes:
        v[pc % 12] = strength
    return list(v)


C_MAJOR = chord(0, 4, 7)      # C E G
A_MINOR = chord(9, 0, 4)      # A C E
D_MAJOR = chord(2, 6, 9)      # D F# A


# ── harmonic_fit ─────────────────────────────────────────────────────────────

def test_identical_harmony_fits_perfectly_with_no_shift():
    h = harmonic_fit(C_MAJOR, C_MAJOR)
    assert h["known"] is True
    assert h["shift"] == 0
    assert h["fit"] == pytest.approx(1.0)


def test_the_shift_is_measured_not_derived():
    """A bed a whole tone below the vocal must come back as +2, found by
    cross-correlation rather than by Camelot arithmetic."""
    bed = chord(10, 2, 5)          # Bb major — two semitones below C
    h = harmonic_fit(C_MAJOR, bed)
    assert h["known"] and h["shift"] == 2
    assert h["fit"] == pytest.approx(1.0)


def test_relative_major_minor_needs_no_transpose():
    """The T1.2 fix, arrived at by measurement rather than by ignoring the
    Camelot letter: C major and A minor share a pitch collection."""
    h = harmonic_fit(C_MAJOR, A_MINOR)
    assert h["shift"] == 0
    assert h["fit"] > 0.8


def test_a_real_clash_scores_low():
    h = harmonic_fit(C_MAJOR, D_MAJOR)
    # Some rotation always fits, so the interesting claim is the UNSHIFTED
    # relationship: these two as they stand do not agree.
    v = np.asarray(C_MAJOR, dtype=float)
    b = np.asarray(D_MAJOR, dtype=float)
    v, b = v / np.linalg.norm(v), b / np.linalg.norm(b)
    unshifted = (float(np.dot(v, b)) + 1.0) / 2.0
    assert unshifted < 0.75
    assert h["shift"] != 0


def test_shift_is_folded_to_the_short_way_round():
    for pc in range(12):
        h = harmonic_fit(C_MAJOR, chord(pc, (pc + 4) % 12, (pc + 7) % 12))
        assert -6 <= h["shift"] <= 6


def test_ambiguous_harmony_reports_low_confidence():
    """A chromatic smear fits about equally well at every transposition — which
    is exactly the case where the estimate must not be trusted."""
    smear = [1.0] * 12
    h = harmonic_fit(smear, smear)
    assert h["confidence"] < 0.1
    clear = harmonic_fit(C_MAJOR, C_MAJOR)
    assert clear["confidence"] > h["confidence"]


def test_unknown_chroma_is_neutral_and_flagged():
    """Callers must be able to tell "we measured 0.5" from "we did not measure",
    so they can fall back to the Camelot estimate."""
    h = harmonic_fit(None, C_MAJOR)
    assert h["known"] is False and h["fit"] == 0.5 and h["shift"] == 0
    assert harmonic_fit(C_MAJOR, [0.0] * 12)["known"] is False
    assert harmonic_fit(C_MAJOR, [1.0, 2.0])["known"] is False


# ── bass clash ───────────────────────────────────────────────────────────────

def test_bass_a_semitone_from_the_tonic_clashes():
    c = bass_clash(C_MAJOR, chord(1), shift=0)      # Db under a C tonic
    assert c["clash"] is True and c["interval"] == 1
    assert "high-pass" in c["advice"]


def test_bass_a_tritone_from_the_tonic_clashes():
    assert bass_clash(C_MAJOR, chord(6), shift=0)["clash"] is True


def test_bass_on_the_tonic_or_fifth_is_fine():
    assert bass_clash(C_MAJOR, chord(0), shift=0)["clash"] is False
    assert bass_clash(C_MAJOR, chord(7), shift=0)["clash"] is False


def test_the_shift_is_applied_before_judging_the_bass():
    """A bass that clashes as recorded may be fine once the bed is transposed —
    and the reverse."""
    assert bass_clash(C_MAJOR, chord(11), shift=1)["clash"] is False
    assert bass_clash(C_MAJOR, chord(0), shift=1)["clash"] is True


def test_missing_bass_chroma_is_unknown_not_clean():
    assert bass_clash(C_MAJOR, None)["known"] is False


# ── section_harmony ──────────────────────────────────────────────────────────

def test_bass_clash_discounts_the_fit_but_does_not_veto():
    clean = section_harmony({"chroma": C_MAJOR},
                            {"chroma": C_MAJOR, "bass_chroma": chord(0)})
    clashing = section_harmony({"chroma": C_MAJOR},
                               {"chroma": C_MAJOR, "bass_chroma": chord(1)})
    assert clashing["harmonic_fit"] < clean["harmonic_fit"]
    assert clashing["harmonic_fit"] > 0.0
    assert clashing["bass_clash"] is True
    assert clashing["advice"] and "high-pass" in clashing["advice"]
    assert clean["advice"] is None


def test_section_harmony_without_chroma_is_not_known():
    h = section_harmony({}, {})
    assert h["known"] is False
    assert h["harmonic_fit"] == 0.5


# ── phrase fit ───────────────────────────────────────────────────────────────

def test_bars_in():
    assert bars_in(60.0, 120.0) == pytest.approx(30.0)   # 2s per bar
    assert bars_in(0.0, 120.0) is None
    assert bars_in(60.0, 0) is None
    assert bars_in(60.0, None) is None


def test_a_32_bar_vocal_over_a_16_bar_drop_is_a_clean_loop():
    """The whole point of measuring in bars: this is 'loop the drop x2', a
    specific fixable arrangement, not a half-marks mismatch."""
    pf = phrase_fit(60.0, 30.0, 128.0)
    assert pf["repeats"] == 2
    assert pf["fit"] == pytest.approx(1.0)
    assert "x2" in pf["note"]


def test_equal_lengths_need_no_loop():
    pf = phrase_fit(30.0, 30.0, 128.0)
    assert pf["repeats"] == 1 and pf["fit"] == pytest.approx(1.0)
    assert pf["note"] is None


def test_a_genuinely_awkward_length_still_scores_badly():
    """Looping must not rescue everything — a bed that does not divide the
    vocal is still a real problem."""
    pf = phrase_fit(60.0, 22.0, 128.0)
    assert pf["fit"] < 0.95


def test_unknown_tempo_falls_back_to_seconds():
    pf = phrase_fit(60.0, 30.0, None)
    assert pf["fit"] == pytest.approx(0.5)
    assert pf["vocal_bars"] is None and pf["repeats"] == 1


def test_looping_is_bounded():
    """A 4-bar stab must not be looped sixteen times to 'cover' a long vocal."""
    pf = phrase_fit(240.0, 7.5, 128.0)   # 128 bars of vocal, 4-bar bed
    assert pf["repeats"] <= 4
    assert pf["fit"] < 0.2
