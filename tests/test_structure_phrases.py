"""T3.2 — section boundaries snap to the 8-bar phrase grid.

Pure python: snap_boundaries_to_phrases and apply_phrase_alignment take beat
indices and dicts, so none of this needs librosa or audio.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.structure import (  # noqa: E402
    PHRASE_BEATS, SNAP_TOLERANCE_BEATS, apply_phrase_alignment,
    snap_boundaries_to_phrases,
)

# A 4-minute 128 BPM track: ~512 beats, and a 12s minimum section is ~26 beats.
N_BEATS = 512
MIN_BEATS = 26


def _snap(bounds, phase=0, n_beats=N_BEATS, min_beats=MIN_BEATS):
    return snap_boundaries_to_phrases(bounds, phase, n_beats, min_beats)


def test_late_detections_are_pulled_onto_the_phrase_grid():
    """The novelty curve peaks a beat or two after the edit. Within tolerance
    those land back on the 8-bar line."""
    bounds, snapped = _snap([66, 129, 194, 257])
    assert bounds == [64, 128, 192, 256]
    assert snapped == [True] * 4
    assert all(b % PHRASE_BEATS == 0 for b in bounds)


def test_exact_phrase_boundaries_are_left_alone():
    bounds, snapped = _snap([64, 128, 256])
    assert bounds == [64, 128, 256]
    assert snapped == [True, True, True]


def test_boundary_beyond_tolerance_is_kept_and_flagged():
    """More than 2 bars from the nearest phrase line, the detection is more
    trustworthy than the grid — keep it, but mark it."""
    off = 64 + SNAP_TOLERANCE_BEATS + 1
    bounds, snapped = _snap([off, 256])
    assert bounds == [off, 256]
    assert snapped == [False, True]


def test_tolerance_boundary_is_inclusive():
    bounds, snapped = _snap([64 + SNAP_TOLERANCE_BEATS, 256])
    assert bounds[0] == 64 and snapped[0] is True


def test_phase_shifts_the_whole_grid():
    """beat_phase=2 means bar lines are beats 2, 6, 10 … so phrase lines are
    2, 34, 66 … Snapping to a multiple of 32 would land mid-bar."""
    bounds, snapped = _snap([64, 130], phase=2)
    assert bounds == [66, 130]
    assert snapped == [True, True]
    assert all((b - 2) % PHRASE_BEATS == 0 for b in bounds)


def test_snapping_never_creates_a_section_under_the_minimum():
    """Two detections inside one phrase: snapping both to the same line would
    collapse a section to zero. The second keeps its detected position, or is
    dropped when even that is too close."""
    bounds, snapped = _snap([64, 70])
    assert len(bounds) == len(snapped)
    assert bounds[0] == 64
    prev = 0
    for b in bounds:
        assert b - prev >= MIN_BEATS
        prev = b


def test_boundaries_too_close_to_the_ends_are_dropped():
    assert _snap([4])[0] == []
    assert _snap([N_BEATS - 5])[0] == []


def test_output_is_monotonic_and_within_range():
    bounds, snapped = _snap([40, 66, 130, 200, 260, 321, 400, 470])
    assert bounds == sorted(bounds)
    assert len(bounds) == len(snapped)
    assert all(0 < b < N_BEATS for b in bounds)
    prev = 0
    for b in bounds:
        assert b - prev >= MIN_BEATS
        prev = b


def test_no_boundaries_snaps_to_nothing():
    assert _snap([]) == ([], [])


# ── confidence discount ───────────────────────────────────────────────────────

def test_unsnapped_sections_lose_confidence():
    segs = [
        {"label": "chorus", "confidence": 0.8, "phrase_aligned": True},
        {"label": "chorus", "confidence": 0.8, "phrase_aligned": False},
    ]
    apply_phrase_alignment(segs)
    assert segs[0]["confidence"] == 0.8
    assert segs[1]["confidence"] == 0.6
    # The marker is internal — it must not reach the sections table.
    assert "phrase_aligned" not in segs[0] and "phrase_aligned" not in segs[1]


def test_missing_marker_is_treated_as_aligned():
    segs = [{"label": "verse", "confidence": 0.5}]
    apply_phrase_alignment(segs)
    assert segs[0]["confidence"] == 0.5


def test_structure_output_shape_is_unchanged(monkeypatch):
    """detect_sections must still emit exactly the keys replace_sections writes
    — the phrase marker is consumed before it gets there."""
    pytest.importorskip("librosa")
    from database.models import replace_sections  # noqa: F401
    from analysis.structure import label_segments
    segs = [{"energy": 0.9, "vocal_presence": 0.8, "repetition": 3,
             "start_sec": 0.0, "end_sec": 30.0, "phrase_aligned": False}]
    label_segments(segs, has_vocals=True)
    apply_phrase_alignment(segs)
    assert set(segs[0]) == {"energy", "vocal_presence", "repetition",
                            "start_sec", "end_sec", "label", "confidence"}


# ── lookahead: never trade a whole section for one alignment ─────────────────

def test_snapping_yields_rather_than_crowding_the_next_boundary():
    """Two detections exactly min_beats apart. Pulling the first forward onto a
    phrase line would leave the second under the floor and merge it away, so the
    first stays where it was detected and BOTH survive."""
    first, second = 60, 60 + MIN_BEATS      # 60, 86 — legal as detected
    bounds, snapped = _snap([first, second])
    assert bounds == [first, second], "a section was dropped to buy an alignment"
    assert snapped[0] is False, "the first boundary should have yielded"


def test_lookahead_does_not_block_a_snap_that_has_room():
    """The guard is narrow: with slack before the next boundary, snapping still
    happens."""
    bounds, snapped = _snap([66, 66 + MIN_BEATS * 3])
    assert bounds[0] == 64 and snapped[0] is True


def test_lookahead_ignores_a_next_boundary_that_is_doomed_anyway():
    """If the next detection is too close to the END to survive regardless,
    yielding for it costs an alignment and saves nothing.

    Constructed so the guard would otherwise fire: 474 snaps forward to 480,
    which leaves only 20 beats before 500 — but 500 sits inside the end margin
    (511 - 26 = 485) and cannot be kept either way."""
    doomed = N_BEATS - 12                       # 500, past the end margin
    bounds, snapped = _snap([doomed - MIN_BEATS, doomed])
    assert bounds == [480], "should still snap — the next boundary was lost anyway"
    assert snapped == [True]


def test_last_boundary_has_nothing_to_yield_to():
    bounds, snapped = _snap([66])
    assert bounds == [64] and snapped == [True]
