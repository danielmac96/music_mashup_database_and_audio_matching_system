"""Alignment on the candidate row (P2.4, spec §8).

Until now these numbers only existed at export time, inside render/session.py,
so the ranked list could say a pair was good but not what building it involved.
They are computed from the per-section downbeats P2.1 stores, which means no
audio and every candidate rather than only the ones that reach export.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from matcher.alignment import _first_downbeat, align, describe  # noqa: E402


def sec(start=0.0, end=30.0, downs=None, label="chorus"):
    return {"label": label, "start_sec": start, "end_sec": end,
            "downbeats": downs, "bar_count": 16.0, "beats_per_bar": 4,
            "energy": 0.8, "vocal_presence": 0.8, "energy_trend": "stable"}


# ── first downbeat ───────────────────────────────────────────────────────────

def test_the_first_bar_line_inside_the_section_wins():
    s = sec(start=40.0, end=70.0, downs=[38.0, 40.5, 42.4])
    assert _first_downbeat(s) == 40.5


def test_no_stored_grid_is_none_not_zero():
    """Zero would claim the bar line sits exactly on the section boundary, which
    is the one thing we know we have not established."""
    assert _first_downbeat(sec(downs=None)) is None
    assert _first_downbeat(sec(downs=[])) is None


# ── the offset ───────────────────────────────────────────────────────────────

def test_offset_is_the_difference_in_how_far_each_bar_line_sits_into_its_section():
    v = sec(start=40.0, end=70.0, downs=[40.5])    # 0.5s in
    i = sec(start=20.0, end=50.0, downs=[20.2])    # 0.2s in
    assert align(v, i)["alignment_offset"] == pytest.approx(0.3)


def test_a_bed_that_starts_late_gives_a_negative_offset():
    v = sec(start=40.0, downs=[40.1])
    i = sec(start=20.0, downs=[20.6])
    assert align(v, i)["alignment_offset"] < 0


def test_aligned_sections_have_a_zero_offset():
    v = sec(start=40.0, downs=[40.25])
    i = sec(start=20.0, downs=[20.25])
    assert align(v, i)["alignment_offset"] == pytest.approx(0.0)


def test_the_offset_is_measured_after_the_stretch():
    """The bed's timeline compresses when it is sped up, so its bar line arrives
    proportionally sooner — an offset taken before the stretch is simply wrong."""
    v = sec(start=40.0, downs=[40.4])
    i = sec(start=20.0, downs=[20.8])
    assert align(v, i, stretch=1.0)["alignment_offset"] == pytest.approx(-0.4)
    assert align(v, i, stretch=2.0)["alignment_offset"] == pytest.approx(0.0)


def test_unknown_grid_gives_none_not_a_confident_zero():
    """The exporter has to tell "aligned at the boundary" from "we do not know"."""
    assert align(sec(downs=None), sec(downs=[20.2]))["alignment_offset"] is None
    assert align(sec(downs=[40.5]), sec(downs=None))["alignment_offset"] is None


def test_downbeat_is_reported_in_the_vocal_track_clock():
    v = sec(start=40.0, downs=[40.5])
    assert align(v, sec(downs=[20.2]))["alignment_downbeat"] == 40.5


# ── tempo and pitch ──────────────────────────────────────────────────────────

def test_tempo_adjustment_is_a_percentage():
    assert align(sec(), sec(), stretch=1.05)["tempo_adjustment"] == pytest.approx(5.0)
    assert align(sec(), sec(), stretch=0.94)["tempo_adjustment"] == pytest.approx(-6.0)
    assert align(sec(), sec(), stretch=1.0)["tempo_adjustment"] == 0.0


def test_target_bpm_and_pitch_are_carried_through():
    out = align(sec(), sec(), stretch=1.0, semitones=-3, target_bpm=128.0)
    assert out["target_bpm"] == 128.0
    assert out["pitch_adjustment"] == -3


def test_no_pitch_shift_is_none_not_zero():
    """0 means "measured, and it is zero"; None means nobody asked."""
    assert align(sec(), sec())["pitch_adjustment"] is None
    assert align(sec(), sec(), semitones=0)["pitch_adjustment"] == 0


# ── the reason line ──────────────────────────────────────────────────────────

def test_reason_names_both_sections_and_their_times():
    v = sec(start=72.0, end=104.0, label="chorus")
    i = sec(start=48.0, end=80.0, label="drop")
    line = describe(v, i, {"section_bars_vocal": 16.0}, align(v, i), 128.0, 128.0)
    assert "chorus" in line and "drop" in line
    assert "1:12" in line and "1:44" in line
    assert "16 bars" in line
    assert "128 BPM" in line


def test_reason_shows_a_tempo_change_as_a_move():
    line = describe(sec(), sec(), {}, align(sec(), sec()), 128.0, 120.0)
    assert "120→128 BPM" in line


def test_reason_mentions_looping_only_when_it_happens():
    a = describe(sec(), sec(), {"section_loop_repeats": 2}, align(sec(), sec()))
    b = describe(sec(), sec(), {"section_loop_repeats": 1}, align(sec(), sec()))
    assert "×2" in a
    assert "loop" not in b


def test_reason_mentions_a_nudge_only_when_it_is_audible():
    big = describe(sec(start=40.0, downs=[40.5]), sec(start=20.0, downs=[20.2]), {},
                   align(sec(start=40.0, downs=[40.5]), sec(start=20.0, downs=[20.2])))
    tiny = describe(sec(start=40.0, downs=[40.25]), sec(start=20.0, downs=[20.25]), {},
                    align(sec(start=40.0, downs=[40.25]), sec(start=20.0, downs=[20.25])))
    assert "nudge" in big and "+300 ms" in big
    assert "nudge" not in tiny


def test_reason_survives_a_pair_with_nothing_measured():
    line = describe(sec(downs=None), sec(downs=None), {}, align(sec(), sec()))
    assert line and "over" in line


# ── persistence ──────────────────────────────────────────────────────────────

def test_alignment_reaches_the_candidate_row(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()

    from matcher.sections import top_section_pairs
    v = [dict(sec(start=40.0, end=70.0, downs=[40.5]), section_index=0)]
    i = [dict(sec(start=20.0, end=50.0, downs=[20.2], label="drop"),
              section_index=0, vocal_presence=0.05)]
    pair = top_section_pairs(v, i, stretch=1.0, bpm=128.0, limit=1)[0]

    models.upsert_song(title="V", artist="A", source_url="u://v")
    models.upsert_song(title="I", artist="B", source_url="u://i")
    models.bulk_upsert_candidates([models.candidate_row(
        {"song_id": 1, "title": "V", "artist": "A", "bpm": 128.0, "camelot": "8A"},
        {"song_id": 2, "title": "I", "artist": "B", "bpm": 128.0, "camelot": "8A"},
        {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0, "energy_score": 0.8,
         "timbre_score": 0.7, "collision_score": 0.6},
        section_pair=pair)])

    conn = models.get_conn()
    row = conn.execute("SELECT * FROM mashup_candidates").fetchone()
    conn.close()
    assert row["alignment_downbeat"] == 40.5
    assert row["alignment_offset"] == pytest.approx(0.3)
    assert row["target_bpm"] == 128.0
    assert row["reason"] and "chorus" in row["reason"]
