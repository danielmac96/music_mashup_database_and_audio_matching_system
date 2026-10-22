"""Phrase placement and the drift gate (matcher/phrase_align.py, readme §5.7),
on synthetic click grids: after the stretch the downbeats of a steady pair stay
under the tolerance, a drifting one is rejected, and the pickup survives."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from matcher.phrase_align import (grid_errors, overlay_clicks, phrase_ratio_fit,  # noqa: E402
                                  place, scale_grid, timing_score)


def _section(start, bars, bpm, jitter=None, label="chorus", wobble=0.0):
    """A section whose downbeats are a steady grid at ``bpm`` — or one whose
    tempo wanders by ``wobble`` (fraction) bar to bar."""
    bar = 4 * 60.0 / bpm
    t, dbs = start, []
    for i in range(bars):
        dbs.append(round(t, 5))
        step = bar * (1 + (wobble * np.sin(i / 2.0) if wobble else 0.0))
        t += step
    return {"start_sec": start, "end_sec": t, "downbeats": dbs, "label": label}


def _land(out, vocal, bed, stretch):
    """Where the vocal's phrase downbeat and the bed's landing sit on the
    output timeline once the bed is placed at alignment_offset."""
    base = 100.0
    v = base + (out["alignment_downbeat"] - vocal["start_sec"])
    b = base + out["alignment_offset"] + (out["bed_landing_sec"] - bed["start_sec"]) / stretch
    return v, b


def test_scale_grid_and_errors():
    assert scale_grid([10.0, 12.0], 2.0, 10.0).tolist() == [0.0, 1.0]
    # The mark at the end of the span closes the overlap and is not judged.
    e = grid_errors([0, 1, 2, 3], [0.0, 1.01, 2.02, 3.5], 3.0)
    assert e.max() == pytest.approx(0.02)
    assert grid_errors([0], [0, 1], 2) is None


@pytest.mark.parametrize("v_bpm,b_bpm", [(124, 124), (124, 128), (126, 118), (90, 180)])
def test_stretched_steady_grids_stay_under_tolerance(v_bpm, b_bpm):
    vocal = _section(30.0, 16, v_bpm)
    bed = _section(61.3, 16, b_bpm, label="drop")
    stretch = v_bpm / b_bpm if abs(v_bpm - b_bpm) < 40 else v_bpm / (b_bpm / 2)
    if abs(v_bpm - b_bpm) >= 40:
        # Half time: the bed's every other bar is the vocal's bar; give it the
        # folded grid as the scorer would see it after the fold.
        bed["downbeats"] = bed["downbeats"][::2]
    out = place(vocal, bed, stretch, tol_ms=25)
    assert out["timing_ok"] == 1
    assert out["timing_err_max_ms"] < 25
    v, b = _land(out, vocal, bed, stretch)
    assert v == pytest.approx(b, abs=1e-6)


def test_a_drifting_bed_fails_the_gate():
    vocal = _section(0.0, 16, 124)
    bed = _section(20.0, 16, 124, wobble=0.02)   # ±2 % bar to bar
    out = place(vocal, bed, 1.0, tol_ms=25)
    assert out["timing_ok"] == 0
    assert out["timing_err_max_ms"] > 25
    loose = place(vocal, bed, 1.0, tol_ms=400)
    assert loose["timing_ok"] == 1


def test_the_pickup_is_kept_and_the_downbeat_lands():
    vocal = _section(30.0, 16, 120)
    bed = _section(50.0, 16, 120)
    phrases = [{"downbeat_sec": 34.0, "onset_sec": 33.5, "pickup_sec": -0.5, "bars": 8}]
    out = place(vocal, bed, 1.0, phrases=phrases)
    assert out["alignment_downbeat"] == pytest.approx(34.0)
    assert out["vocal_pickup_ms"] == pytest.approx(-500.0)
    assert out["vocal_phrase_start"] == pytest.approx(33.5)
    v, b = _land(out, vocal, bed, 1.0)
    assert v == pytest.approx(b)
    # The phrase starts two bars (4 s at 120) into the vocal section: the bed
    # is pushed two bars later than the first-downbeat rule
    # (matcher/alignment.align) would put it.
    assert out["alignment_offset"] == pytest.approx(4.0)


def test_meter_mismatch_drifts():
    vocal = _section(0.0, 16, 120)
    bed = dict(_section(0.0, 16, 120))
    bar = 3 * 60.0 / 120                      # a 3/4 bed at the same beat
    bed["downbeats"] = [i * bar for i in range(21)]
    bed["end_sec"] = 21 * bar
    assert place(vocal, bed, 1.0)["timing_ok"] == 0


def test_no_grid_is_unjudged_not_failed():
    out = place({"start_sec": 0, "end_sec": 30}, {"start_sec": 0, "end_sec": 30}, 1.0)
    assert out["timing_ok"] is None and out["alignment_offset"] is None


def test_phrase_ratio_rules():
    ratios = (0.25, 0.5, 1.0, 2.0, 4.0)
    assert phrase_ratio_fit(16, 16, ratios) == 1.0
    assert phrase_ratio_fit(8, 16, ratios) == 0.9
    assert phrase_ratio_fit(1, 16, ratios) == 0.0
    assert phrase_ratio_fit(None, 16, ratios) is None
    assert timing_score(0, 0, 25, 1.0) == 1.0
    assert timing_score(25, 25, 25, 1.0) == pytest.approx(0.25)


def test_landing_moves_to_a_later_phrase_when_the_first_drifts():
    vocal = _section(0.0, 8, 120)
    bed = _section(0.0, 32, 120)
    # The bed's first 8 bars wander; from bar 8 on it is steady.
    dbs = list(bed["downbeats"])
    for i in range(1, 8):
        dbs[i] += 0.06 * np.sin(i)
    bed["downbeats"] = dbs
    out = place(vocal, bed, 1.0)
    assert out["timing_ok"] == 1
    assert out["bed_landing_sec"] == pytest.approx(bed["downbeats"][8])


def test_overlay_clicks_line_up_after_the_offset():
    sr = 8000
    clicks = overlay_clicks([1.0, 3.0], [0.5, 2.5], 0.5, 4.0, sr=sr)
    left = np.flatnonzero(np.abs(clicks[:, 0]) > 0.5)
    right = np.flatnonzero(np.abs(clicks[:, 1]) > 0.5)
    assert abs(int(left[0]) - int(right[0])) <= 2
