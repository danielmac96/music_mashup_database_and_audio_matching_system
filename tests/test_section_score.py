"""Spec §7's phrase, rhythm and structure scores (P2.3).

These are properties of a SECTION PAIR, which is why they live beside the other
per-section work rather than in sub_scores — that runs on stem-level feature
dicts inside a vectorised loop over 256-row blocks and has no section to look at.

They ship at zero weight. The tests below therefore check the numbers
themselves, plus the two things that decide whether turning them on is safe:
that missing data scores neutral rather than badly, and that zero weight really
means no effect.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from matcher import patterns as pat  # noqa: E402
from matcher.section_score import (  # noqa: E402
    phrase_score, rhythm_score, section_components, section_structure_score,
)


def sec(label="chorus", start=0.0, end=30.0, bars=16.0, per_bar=4,
        energy=0.7, trend="stable", vocal=0.8, beats=None, downbeats=None):
    return {"label": label, "start_sec": start, "end_sec": end,
            "bar_count": bars, "beats_per_bar": per_bar, "energy": energy,
            "energy_trend": trend, "vocal_presence": vocal,
            "beat_times": beats, "downbeats": downbeats}


def grid(bpm=128.0, bars=4, per_bar=4, start=0.0, skip=()):
    """A steady beat grid; `skip` drops beat positions to shift the emphasis."""
    step = 60.0 / bpm
    beats = [round(start + i * step, 4) for i in range(bars * per_bar)
             if (i % per_bar) not in skip]
    downs = [round(start + i * step, 4) for i in range(bars * per_bar)
             if i % per_bar == 0]
    return beats, downs


# ── phrase ───────────────────────────────────────────────────────────────────

def test_equal_phrase_lengths_score_full_marks():
    assert phrase_score(sec(bars=16.0), sec(bars=16.0)) == 1.0


def test_a_snapped_boundary_within_half_a_bar_still_counts_as_equal():
    """Sections are snapped to an 8-bar grid upstream; a boundary that could not
    be snapped leaves a fractional bar behind and must not be punished for it."""
    assert phrase_score(sec(bars=16.0), sec(bars=15.7)) == 1.0


def test_a_clean_double_is_a_loop_not_a_mismatch():
    """A 16-bar bed under a 32-bar vocal is one drag in a DAW."""
    score = phrase_score(sec(bars=32.0), sec(bars=16.0))
    assert score > 0.8


def test_a_deeper_loop_scores_below_a_shallow_one():
    assert phrase_score(sec(bars=32.0), sec(bars=16.0)) \
        > phrase_score(sec(bars=64.0), sec(bars=8.0))


def test_a_partial_phrase_is_penalised():
    """This is the case that costs real editing — spec §7 asks for it by name."""
    partial = phrase_score(sec(bars=16.0), sec(bars=11.0))
    assert partial < phrase_score(sec(bars=32.0), sec(bars=16.0))
    assert partial < 0.6


def test_stretch_is_applied_to_the_bed():
    """The bed plays at `stretch` to reach the vocal's tempo, so it covers that
    many fewer bars in the same wall clock."""
    assert phrase_score(sec(bars=16.0), sec(bars=32.0), stretch=2.0) == 1.0


def test_missing_bar_counts_fall_back_to_duration():
    """A library analysed before P2.1 has no bar counts, and must not score zero
    — that would read as a bad pair rather than an unmeasured one."""
    v = sec(bars=None, start=0.0, end=30.0)
    i = sec(bars=None, start=0.0, end=30.0)
    assert phrase_score(v, i) == 1.0
    assert phrase_score(v, sec(bars=None, start=0.0, end=15.0)) == pytest.approx(0.5)


def test_degenerate_sections_do_not_raise():
    """A zero bar count is falsy, so it reads as "not measured" and falls back to
    duration — which is the right behaviour, but only if the duration path also
    survives a zero-length section."""
    assert phrase_score(sec(bars=0.0), sec(bars=16.0)) == 1.0   # duration fallback
    assert phrase_score(sec(bars=None, start=0.0, end=0.0),
                        sec(bars=None, start=0.0, end=30.0)) == 0.5
    assert phrase_score(sec(bars=-4.0), sec(bars=16.0)) == 0.0


# ── rhythm ───────────────────────────────────────────────────────────────────

def test_identical_grids_agree_completely():
    beats, downs = grid()
    a = sec(beats=beats, downbeats=downs)
    b = sec(beats=beats, downbeats=downs)
    assert rhythm_score(a, b) == pytest.approx(1.0, abs=1e-3)


def test_different_emphasis_scores_lower_than_identical():
    """One section putting its weight on beats the other leaves empty is the
    pair that needs the vocal nudged."""
    beats_a, downs_a = grid(skip=(1, 3))     # weight on 1 and 3
    beats_b, downs_b = grid(skip=(0, 2))     # weight on 2 and 4
    same = rhythm_score(sec(beats=beats_a, downbeats=downs_a),
                        sec(beats=beats_a, downbeats=downs_a))
    diff = rhythm_score(sec(beats=beats_a, downbeats=downs_a),
                        sec(beats=beats_b, downbeats=downs_b))
    assert diff < same


def test_no_stored_grid_is_neutral_not_zero():
    """No evidence is not evidence against, and a pre-P2.1 library has none."""
    assert rhythm_score(sec(), sec()) == 0.5
    assert rhythm_score(sec(beats=grid()[0], downbeats=grid()[1]), sec()) == 0.5


def test_too_few_beats_is_neutral():
    assert rhythm_score(sec(beats=[0.0, 0.5], downbeats=[0.0]),
                        sec(beats=[0.0, 0.5], downbeats=[0.0])) == 0.5


def test_rhythm_needs_no_audio():
    """matcher must stay importable without librosa, and a scoring run touches
    hundreds of thousands of pairs."""
    import ast

    import matcher.section_score as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"librosa", "soundfile", "numpy", "scipy"}


# ── structure ────────────────────────────────────────────────────────────────

def test_chorus_over_drop_is_the_canonical_shape():
    assert section_structure_score(sec("chorus"), sec("drop")) == 1.0


def test_an_unlisted_pairing_is_neutral_not_zero():
    """The pattern list is a set of good ideas, not an exhaustive grammar. Zero
    would let seven hard-coded shapes veto the rest of the library."""
    assert section_structure_score(sec("outro"), sec("intro")) == 0.5


def test_a_weaker_pattern_scores_below_the_canonical_one():
    assert section_structure_score(sec("verse"), sec("verse")) \
        < section_structure_score(sec("chorus"), sec("drop"))


def test_a_build_is_recognised_by_its_energy_not_its_label():
    """"build" is deliberately not aliased to "breakdown" — a build is rising,
    a breakdown is falling. energy_trend is what actually identifies one."""
    rising = sec("breakdown", trend="increasing")
    falling = sec("breakdown", trend="decreasing")
    assert section_structure_score(sec("verse"), rising) \
        > section_structure_score(sec("verse"), falling)


def test_the_right_shape_with_the_wrong_movement_is_discounted_not_rejected():
    matched = section_structure_score(sec("verse", energy=0.7),
                                      sec("drop", trend="increasing"))
    wrong = section_structure_score(sec("verse", energy=0.7),
                                    sec("drop", trend="decreasing"))
    assert wrong < matched
    assert wrong > 0.5, "still a real idea, just not the one the pattern described"


def test_unmeasured_energy_is_not_punished():
    no_trend = dict(sec("verse", trend=None))
    assert section_structure_score(sec("verse"), dict(no_trend, label="drop")) >= 0.9


def test_custom_patterns_are_honoured():
    only = pat.validate([{"name": "x", "vocal_section_types": ["bridge"],
                          "instrumental_section_types": ["drop"], "weight": 1.0}])
    assert section_structure_score(sec("bridge"), sec("drop"), only) == 1.0
    assert section_structure_score(sec("chorus"), sec("drop"), only) == 0.5


# ── integration with the section fit ─────────────────────────────────────────

def test_components_are_reported_on_every_pair_row():
    """Stored while the weights are still zero, so the numbers can be judged
    against real pairs before they are allowed to move a ranking."""
    from matcher.sections import top_section_pairs
    v = [dict(sec("chorus"), section_index=0)]
    i = [dict(sec("drop", vocal=0.05), section_index=0)]
    row = top_section_pairs(v, i, stretch=1.0, bpm=128.0, limit=1)[0]
    for key in ("score_phrase", "score_rhythm", "score_structure"):
        assert key in row and row[key] is not None


def test_zero_weight_really_means_no_effect(tmp_path, monkeypatch):
    """The safety property behind shipping these off: until a library is
    re-analysed, turning them on would score on data that is not there."""
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    import config
    importlib.reload(config)
    import matcher.sections as sections
    importlib.reload(sections)

    v, i = sec("chorus"), sec("drop", vocal=0.05)
    baseline = sections.score_section_pair(v, i, 1.0, 128.0)

    # Same pair, but with the structure term carrying real weight.
    config.save_settings({"section_weights": {
        "label": 0.4, "duration": 0.35, "voice": 0.25,
        "phrase": 0.0, "rhythm": 0.0, "structure": 0.5}})
    importlib.reload(config)
    importlib.reload(sections)
    try:
        assert sections.score_section_pair(v, i, 1.0, 128.0) != baseline
    finally:
        importlib.reload(config)
        importlib.reload(sections)


def test_new_scores_reach_the_database(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()

    from matcher.sections import top_section_pairs
    pair = top_section_pairs([dict(sec("chorus"), section_index=0)],
                             [dict(sec("drop", vocal=0.05), section_index=0)],
                             stretch=1.0, bpm=128.0, limit=1)[0]
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
    assert row["score_phrase"] == pair["score_phrase"]
    assert row["score_structure"] == pair["score_structure"]
