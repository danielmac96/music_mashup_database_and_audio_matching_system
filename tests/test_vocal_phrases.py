"""Vocal phrase detection (analysis/phrases.py, readme §5.5): where the singing
starts, snapped to its bar line with the signed pickup kept."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.phrases import (detect_phrases, phrases_in, round_bars,  # noqa: E402
                              snap_to_downbeat, sung_runs)

HOP = 512 / 22050          # the frame the vocal envelope is measured on


def _active(secs, spans):
    """Frame activity: True inside each (start, end) span."""
    n = int(secs / HOP) + 1
    t = np.arange(n) * HOP
    a = np.zeros(n, dtype=bool)
    for s, e in spans:
        a |= (t >= s) & (t < e)
    return a


BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
DOWNBEATS = [i * BAR for i in range(40)]
BEATS = [i * BEAT for i in range(160)]


def test_onset_on_the_bar_has_no_pickup():
    act = _active(80, [(16.0, 30.0)])
    ph = detect_phrases(act, HOP, DOWNBEATS, BEATS)
    assert len(ph) == 1
    assert ph[0]["downbeat_sec"] == pytest.approx(16.0)
    assert abs(ph[0]["pickup_sec"]) < 0.03          # within one frame


def test_anacrusis_one_beat_early_keeps_a_negative_pickup():
    act = _active(80, [(16.0 - BEAT, 30.0)])
    ph = detect_phrases(act, HOP, DOWNBEATS, BEATS)
    assert ph[0]["downbeat_sec"] == pytest.approx(16.0)
    assert ph[0]["pickup_sec"] == pytest.approx(-BEAT, abs=0.03)


def test_late_entry_is_a_positive_pickup_within_50ms():
    act = _active(80, [(24.08, 34.0)])
    ph = detect_phrases(act, HOP, DOWNBEATS, BEATS)
    assert ph[0]["downbeat_sec"] == pytest.approx(24.0)
    assert ph[0]["pickup_sec"] == pytest.approx(0.08, abs=0.03)


def test_a_breath_shorter_than_the_gap_does_not_start_a_phrase():
    # 0.3 s gap inside a line (gap threshold is max(0.6 s, a beat)).
    act = _active(80, [(16.0, 20.0), (20.3, 30.0), (40.0 - BEAT, 50.0)])
    ph = detect_phrases(act, HOP, DOWNBEATS, BEATS)
    assert [p["downbeat_sec"] for p in ph] == pytest.approx([16.0, 40.0])


def test_a_blip_shorter_than_min_sung_is_ignored():
    act = _active(80, [(10.0, 10.2), (16.0, 30.0)])
    ph = detect_phrases(act, HOP, DOWNBEATS, BEATS)
    assert [p["downbeat_sec"] for p in ph] == pytest.approx([16.0])


def test_phrase_bars_are_rounded_to_a_power_of_two():
    act = _active(80, [(16.0, 16.0 + 7.6 * BAR)])
    ph = detect_phrases(act, HOP, DOWNBEATS, BEATS)
    assert ph[0]["bars"] == 8
    assert round_bars(12) == 16 and round_bars(5) == 4 and round_bars(0) == 1


def test_no_grid_no_phrases():
    act = _active(30, [(5.0, 10.0)])
    assert detect_phrases(act, HOP, [], BEATS) == []


def test_melody_rejects_an_unpitched_run():
    act = _active(80, [(16.0, 30.0)])
    step = 0.05
    f0 = [0.0] * int(80 / step)                     # nothing voiced
    assert detect_phrases(act, HOP, DOWNBEATS, BEATS,
                          melody={"f0": f0, "step": step}) == []
    f0v = [220.0 if 16.0 <= i * step < 30.0 else 0.0 for i in range(len(f0))]
    assert len(detect_phrases(act, HOP, DOWNBEATS, BEATS,
                              melody={"f0": f0v, "step": step})) == 1


def test_snap_and_window_helpers():
    assert snap_to_downbeat(15.1, DOWNBEATS) == pytest.approx((16.0, -0.9))
    assert snap_to_downbeat(3.0, []) is None
    phrases = [{"downbeat_sec": 16.0, "onset_sec": 15.5},
               {"downbeat_sec": 32.0, "onset_sec": 32.0}]
    # A pickup sung just before the section starts still belongs to it.
    assert phrases_in(phrases, 15.6, 31.9) == [phrases[0]]
    assert sung_runs([False, True, True, False], 1.0, 0.5, 1.5) == [(1.0, 3.0, float("inf"))]


def test_detection_rate_on_a_synthetic_set():
    """Brief, Phase 2: at least 90% of phrase starts within one beat of the
    true entry and pickups within 50 ms — on 12 synthetic vocals with pickups
    from -1 beat to +80 ms at mixed tempos."""
    rng = np.random.default_rng(7)
    hits = total = 0
    for k in range(12):
        bpm = float(rng.uniform(90, 140))
        beat = 60.0 / bpm
        bar = 4 * beat
        dbs = [i * bar for i in range(80)]
        truth = []
        spans = []
        for bar_i in (4, 12, 20, 28):
            pickup = float(rng.choice([-beat, -beat / 2, 0.0, 0.04, 0.08]))
            onset = dbs[bar_i] + pickup
            truth.append((dbs[bar_i], pickup))
            spans.append((onset, onset + 6 * bar))
        act = _active(dbs[-1], spans)
        ph = detect_phrases(act, HOP, dbs, [i * beat for i in range(320)])
        for db, pickup in truth:
            total += 1
            got = [p for p in ph if abs(p["downbeat_sec"] - db) < 1e-3]
            if got and abs(got[0]["pickup_sec"] - pickup) <= 0.05:
                hits += 1
    assert hits / total >= 0.9
