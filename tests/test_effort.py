"""Phase C — the effort penalty (matcher/effort.py).

The four sub-scores all ask "are these alike?". None asks "what will this cost
me?" — but a 12% stretch and a +5 semitone shift are real damage, and a bed with
a weak beat grid is twenty minutes of manual beatgridding before a note plays.

Pure python + numpy.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from matcher.effort import (  # noqa: E402
    EFFORT_WEIGHTS, dominant_component, effort_block, effort_components,
    effort_label, effort_penalty, is_tempo_fold,
)


def _feat(bpm=120.0, bpm_conf=1.0, key_conf=1.0):
    return {"bpm": bpm, "bpm_confidence": bpm_conf, "key_confidence": key_conf}


# ── The free case ────────────────────────────────────────────────────────────

def test_same_tempo_same_key_confident_costs_nothing():
    total, parts = effort_penalty(_feat(120.0), _feat(120.0),
                                  stretch=1.0, semitones=0)
    assert total == pytest.approx(0.0)
    assert all(v == pytest.approx(0.0) for v in parts.values())
    assert effort_label(total) == "Free"
    assert dominant_component(parts) is None


def test_tiny_stretch_and_one_semitone_are_still_free():
    """Below 2% stretch and ±1 semitone nobody hears it."""
    total, parts = effort_penalty(_feat(120.0), _feat(121.0),
                                  stretch=1.008, semitones=1)
    assert parts["stretch_cost"] == pytest.approx(0.0)
    assert parts["pitch_cost"] == pytest.approx(0.0)
    assert total == pytest.approx(0.0)


# ── Each component ───────────────────────────────────────────────────────────

def test_stretch_cost_ramps_and_saturates():
    def cost(stretch):
        return effort_components(_feat(), _feat(), stretch, 0)["stretch_cost"]
    assert cost(1.0) == pytest.approx(0.0)
    assert cost(1.02) == pytest.approx(0.0)
    assert 0.0 < cost(1.06) < 1.0
    assert cost(1.12) == pytest.approx(1.0)
    assert cost(1.30) == pytest.approx(1.0)
    # Slowing down costs the same as speeding up.
    assert cost(0.94) == pytest.approx(cost(1.06))


def test_pitch_cost_ramps_and_saturates():
    def cost(semis):
        return effort_components(_feat(), _feat(), 1.0, semis)["pitch_cost"]
    assert cost(0) == pytest.approx(0.0)
    assert cost(1) == pytest.approx(0.0)
    assert 0.0 < cost(3) < 1.0
    assert cost(6) == pytest.approx(1.0)
    assert cost(-6) == pytest.approx(cost(6))


def test_pitching_the_vocal_costs_more_than_pitching_the_bed():
    """Formant damage is heard on a voice long before the same shift bothers a
    synth, so the two sides are not interchangeable."""
    bed = effort_components(_feat(), _feat(), 1.0, 3, pitch_side="bed")
    top = effort_components(_feat(), _feat(), 1.0, 3, pitch_side="top")
    assert top["pitch_cost"] > bed["pitch_cost"]
    assert top["pitch_cost"] == pytest.approx(min(1.0, 2.0 * bed["pitch_cost"]))


def test_unknown_stretch_or_shift_costs_the_maximum():
    """An unknown transpose is not a free one."""
    parts = effort_components(_feat(), _feat(), None, None)
    assert parts["stretch_cost"] == pytest.approx(1.0)
    assert parts["pitch_cost"] == pytest.approx(1.0)


def test_halftime_pairing_is_charged_but_not_fatal():
    assert is_tempo_fold(150.0, 75.0) is True
    assert is_tempo_fold(120.0, 122.0) is False
    parts = effort_components(_feat(150.0), _feat(75.0), 1.0, 0)
    assert 0.0 < parts["tempo_fold_cost"] < 1.0


def test_weak_beat_grid_costs_manual_beatgridding():
    parts = effort_components(_feat(bpm_conf=1.0), _feat(bpm_conf=0.3), 1.0, 0)
    assert parts["grid_cost"] == pytest.approx(0.7)


def test_uncertain_key_costs_because_the_shift_is_a_guess():
    parts = effort_components(_feat(key_conf=0.9), _feat(key_conf=0.2), 1.0, 0)
    assert parts["key_certainty_cost"] == pytest.approx(0.8)


def test_missing_confidence_is_not_retroactively_penalised():
    """Tracks analysed before the confidence columns existed must not all
    suddenly read as maximum effort."""
    parts = effort_components({"bpm": 120.0}, {"bpm": 120.0}, 1.0, 0)
    assert parts["grid_cost"] == pytest.approx(0.0)
    assert parts["key_certainty_cost"] == pytest.approx(0.0)


# ── Totals and labels ────────────────────────────────────────────────────────

def test_expensive_pair_scores_high_and_labels_heavy():
    total, parts = effort_penalty(_feat(120.0, 0.4, 0.3), _feat(107.0, 0.5, 0.4),
                                  stretch=1.12, semitones=5)
    # saturated stretch (0.30) + 0.8 pitch (0.24) + 0.6 grid (0.09)
    # + 0.7 key (0.07) = 0.70
    assert total == pytest.approx(0.70)
    assert effort_label(total) == "Heavy"
    assert dominant_component(parts) in ("stretch_cost", "pitch_cost")


def test_labels_partition_the_range():
    assert effort_label(0.0) == "Free"
    assert effort_label(0.20) == "Free"
    assert effort_label(0.21) == "Light"
    assert effort_label(0.50) == "Light"
    assert effort_label(0.51) == "Heavy"


def test_weights_sum_to_one():
    """Otherwise the total is not on the 0-1 scale the label thresholds and the
    score discount both assume."""
    assert sum(EFFORT_WEIGHTS.values()) == pytest.approx(1.0)


# ── Block form agrees with the scalar form ───────────────────────────────────

def test_block_matches_scalar_pair_for_pair():
    """The same drift guard the four sub-scores have: two copies of this
    arithmetic is how the ranked list and the exported plan end up disagreeing
    about what a pair costs."""
    from matcher.match import compute_semitone_shift, compute_stretch_factor

    tops = [
        {"bpm": 120.0, "camelot": "8A", "bpm_confidence": 0.9, "key_confidence": 0.8},
        {"bpm": 150.0, "camelot": "3B", "bpm_confidence": 0.4, "key_confidence": 0.2},
        {"bpm": 96.0, "camelot": "?", "bpm_confidence": 1.0, "key_confidence": 1.0},
    ]
    beds = [
        {"bpm": 121.0, "camelot": "8A", "bpm_confidence": 0.7, "key_confidence": 0.9},
        {"bpm": 75.0, "camelot": "11A", "bpm_confidence": 0.8, "key_confidence": 0.6},
        {"bpm": 107.0, "camelot": "5B", "bpm_confidence": 0.3, "key_confidence": 0.4},
        {"bpm": 0.0, "camelot": "8A", "bpm_confidence": 0.9, "key_confidence": 0.9},
    ]

    top_bpm = np.array([t["bpm"] for t in tops], dtype=np.float64)
    bed_bpm = np.array([b["bpm"] for b in beds], dtype=np.float64)

    stretch = np.zeros((len(tops), len(beds)))
    semis = np.zeros((len(tops), len(beds)))
    known = np.zeros((len(tops), len(beds)), dtype=bool)
    folded = np.zeros((len(tops), len(beds)), dtype=bool)
    for a, t in enumerate(tops):
        for b, bd in enumerate(beds):
            s = compute_stretch_factor(t["bpm"], bd["bpm"])
            stretch[a, b] = 0.0 if s is None else s
            sh = compute_semitone_shift(t["camelot"], bd["camelot"])
            semis[a, b] = 0 if sh is None else sh
            known[a, b] = sh is not None
            folded[a, b] = is_tempo_fold(t["bpm"], bd["bpm"])

    conf = lambda rows, key: np.array([r[key] for r in rows], dtype=np.float64)
    total, parts = effort_block(
        top_bpm[:, None], bed_bpm[None, :], stretch, semis, known,
        conf(tops, "bpm_confidence")[:, None], conf(beds, "bpm_confidence")[None, :],
        conf(tops, "key_confidence")[:, None], conf(beds, "key_confidence")[None, :],
        folded)

    for a, t in enumerate(tops):
        for b, bd in enumerate(beds):
            s = compute_stretch_factor(t["bpm"], bd["bpm"])
            sh = compute_semitone_shift(t["camelot"], bd["camelot"])
            want_total, want_parts = effort_penalty(t, bd, s, sh)
            assert total[a, b] == pytest.approx(want_total, abs=1e-12), \
                f"total mismatch at {a},{b}"
            for name, values in parts.items():
                assert values[a, b] == pytest.approx(want_parts[name], abs=1e-12), \
                    f"{name} mismatch at {a},{b}"


# ── End to end: it must actually reorder the list ────────────────────────────

@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "effort.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


def test_a_free_build_can_outrank_a_better_but_costly_match(db_path):
    """The point of the whole phase: a pair you can build in one drag beats one
    that fits marginally better but needs a destructive stretch and transpose."""
    from database.models import get_conn, init_db, upsert_features, upsert_song
    from matcher.match import score_all_pairs
    init_db(db_path)

    def add(title, bpm, camelot, conf=1.0):
        sid = upsert_song(title, "A", f"u://{title}", 240,
                          status="analysed", db_path=db_path)
        for stem in ("full", "vocals", "instrumental"):
            upsert_features(sid, stem, {
                "bpm": bpm, "key": "C", "mode": "major", "camelot": camelot,
                "loudness_rms": 0.1, "energy": 0.5, "mfcc": [1.0] * 13,
                "spectral_centroid": 2000.0, "spectral_rolloff": 4000.0,
                "zero_crossing_rate": 0.05,
                "bpm_confidence": conf, "key_confidence": conf,
            }, db_path=db_path)
        return sid

    vocal = add("Vocal", 124.0, "8A")
    free = add("Free bed", 124.0, "8A")          # same tempo, same key
    costly = add("Costly bed", 116.0, "10A")     # needs stretch + transpose

    score_all_pairs(db_path=db_path, bpm_max_diff=20.0, key_min_score=0.0)

    conn = get_conn(db_path)
    rows = {r["inst_song_id"]: r for r in conn.execute(
        "SELECT * FROM mashup_candidates WHERE combo_type='vocal_over_instrumental' "
        "AND vocal_song_id=?", (vocal,)).fetchall()}
    conn.close()

    assert rows[free]["score_effort"] == pytest.approx(0.0)
    assert rows[costly]["score_effort"] > 0.2
    assert rows[free]["score_total"] > rows[costly]["score_total"]
    # Components are persisted so the UI can name the dominant cost.
    assert rows[costly]["effort_stretch"] > 0.0
    assert rows[costly]["effort_pitch"] > 0.0
