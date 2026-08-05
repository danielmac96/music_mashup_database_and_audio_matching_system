"""T3.3 — section-level pair selection.

Scoring stores the (vocal section × bed section) a candidate was actually chosen
for, so the preview plays that moment rather than each track's generic hook.
Pure python + sqlite; no audio.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _sec(idx, start, end, label, *, energy=0.7, vp=0.7, conf=0.8):
    return {"section_index": idx, "start_sec": start, "end_sec": end,
            "label": label, "energy": energy, "vocal_presence": vp,
            "repetition": 2, "confidence": conf}


# ── the chooser ───────────────────────────────────────────────────────────────

def test_chorus_over_drop_beats_verse_over_breakdown():
    from matcher.sections import best_section_pair
    vocals = [_sec(0, 0, 30, "verse"), _sec(1, 30, 60, "chorus")]
    beds = [_sec(0, 0, 30, "breakdown"), _sec(1, 30, 60, "drop")]
    best = best_section_pair(vocals, beds, 1.0)
    assert best["vocal_section_idx"] == 1
    assert best["inst_section_idx"] == 1
    assert best["vocal_section_label"] == "chorus"
    assert best["inst_section_label"] == "drop"
    assert 0.0 <= best["score_section"] <= 1.0


def test_duration_fit_breaks_a_label_tie():
    """Two drops of equal standing — take the one that covers the vocal."""
    from matcher.sections import best_section_pair
    vocals = [_sec(0, 0, 30, "chorus")]
    beds = [_sec(0, 0, 8, "drop"), _sec(1, 20, 52, "drop")]
    best = best_section_pair(vocals, beds, 1.0)
    assert best["inst_section_idx"] == 1


def test_stretch_is_applied_to_the_bed_duration():
    """A 60s bed played at 2x covers 30s, matching a 30s vocal exactly; the
    unstretched reading would call the 30s bed the better fit."""
    from matcher.sections import best_section_pair
    vocals = [_sec(0, 0, 30, "chorus")]
    beds = [_sec(0, 0, 30, "drop"), _sec(1, 60, 120, "drop")]
    assert best_section_pair(vocals, beds, 2.0)["inst_section_idx"] == 1
    assert best_section_pair(vocals, beds, 1.0)["inst_section_idx"] == 0


def test_vocal_presence_is_preferred():
    from matcher.sections import best_section_pair
    vocals = [_sec(0, 0, 30, "chorus", vp=0.3), _sec(1, 30, 60, "chorus", vp=0.95)]
    beds = [_sec(0, 0, 30, "drop")]
    assert best_section_pair(vocals, beds, 1.0)["vocal_section_idx"] == 1


def test_intros_and_outros_are_never_chosen():
    from matcher.sections import best_section_pair
    vocals = [_sec(0, 0, 30, "intro"), _sec(1, 30, 60, "verse")]
    beds = [_sec(0, 0, 30, "outro"), _sec(1, 30, 60, "breakdown")]
    best = best_section_pair(vocals, beds, 1.0)
    assert best["vocal_section_label"] == "verse"
    assert best["inst_section_label"] == "breakdown"


def test_silent_vocal_sections_are_filtered_out():
    """_pick_sections drops anything the separator found no voice in — there is
    nothing to lay over the bed there."""
    from matcher.sections import best_section_pair
    vocals = [_sec(0, 0, 30, "chorus", vp=0.05)]
    beds = [_sec(0, 0, 30, "drop")]
    assert best_section_pair(vocals, beds, 1.0) is None


def test_no_sections_on_either_side_returns_none():
    from matcher.sections import best_section_pair
    assert best_section_pair([], [_sec(0, 0, 30, "drop")], 1.0) is None
    assert best_section_pair([_sec(0, 0, 30, "chorus")], [], 1.0) is None
    assert best_section_pair([], [], 1.0) is None


def test_unmeasured_vocal_presence_is_neutral_not_zero():
    """vocal_presence None means the stem was never measured, which is not
    evidence against the section."""
    from matcher.sections import score_section_pair
    unknown = _sec(0, 0, 30, "chorus", vp=None)
    silent = _sec(0, 0, 30, "chorus", vp=0.0)
    bed = _sec(0, 0, 30, "drop")
    assert score_section_pair(unknown, bed, 1.0) > score_section_pair(silent, bed, 1.0)


def test_selection_is_deterministic():
    from matcher.sections import best_section_pair
    vocals = [_sec(i, i * 30, i * 30 + 30, "chorus") for i in range(4)]
    beds = [_sec(i, i * 30, i * 30 + 30, "drop") for i in range(4)]
    first = best_section_pair(vocals, beds, 1.0)
    for _ in range(3):
        assert best_section_pair(vocals, beds, 1.0) == first


def test_duration_fit_edges():
    from matcher.sections import duration_fit
    assert duration_fit(30, 30) == 1.0
    assert duration_fit(30, 15) == 0.5
    assert duration_fit(0, 30) == 0.0
    assert duration_fit(30, 0) == 0.0


# ── persisted onto the candidate row ──────────────────────────────────────────

@pytest.fixture()
def scored(tmp_path, monkeypatch):
    p = tmp_path / "sec.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    from database.models import (
        init_db, replace_sections, upsert_features, upsert_song,
    )
    init_db(p)
    ids = []
    for n, (bpm, cam) in enumerate([(124.0, "8A"), (126.0, "8A")]):
        sid = upsert_song(f"S{n}", "A", f"https://sc/{n}", 200, "Pop",
                          status="analysed", db_path=p)
        ids.append(sid)
        for stem in ("full", "vocals", "instrumental"):
            upsert_features(sid, stem, {
                "bpm": bpm, "key": "A", "mode": "minor", "camelot": cam,
                "loudness_rms": 0.04 + 0.01 * n, "energy": 0.5,
                "mfcc": [190.0] + [float((n * k) % 7) for k in range(12)],
            }, db_path=p)
        replace_sections(sid, [
            _sec(0, 0, 20, "intro", vp=0.05),
            _sec(1, 20, 50, "verse", vp=0.6),
            _sec(2, 50, 80, "chorus", vp=0.9, energy=0.9),
            _sec(3, 80, 110, "drop", vp=0.1, energy=0.95),
        ], db_path=p)

    from matcher.match import score_all_pairs
    score_all_pairs(db_path=p, scorer="heuristic")
    return p, ids


def test_candidate_rows_carry_the_winning_section_pair(scored):
    db_path, _ = scored
    from database.models import get_conn
    conn = get_conn(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchall()]
    conn.close()
    assert rows
    for r in rows:
        assert r["vocal_section_idx"] is not None
        assert r["inst_section_idx"] is not None
        assert r["vocal_section_end"] > r["vocal_section_start"]
        assert r["inst_section_end"] > r["inst_section_start"]
        assert 0.0 <= r["score_section"] <= 1.0
        # Vocal side takes the chorus; bed side takes the drop.
        assert r["vocal_section_idx"] == 2
        assert r["inst_section_idx"] == 3


def test_instrumental_over_instrumental_has_no_section_pair(scored):
    """The top layer there is an instrumental, so the vocal-side filter would be
    asking the wrong question."""
    db_path, _ = scored
    from database.models import get_conn
    conn = get_conn(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM mashup_candidates "
        "WHERE combo_type='instrumental_over_instrumental'").fetchall()]
    conn.close()
    assert rows
    assert all(r["score_section"] is None for r in rows)


def test_enriched_rows_expose_the_section_labels(scored):
    db_path, _ = scored
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(combo_type="vocal_over_instrumental",
                                   db_path=db_path)
    assert rows
    assert rows[0]["vocal_section_label"] == "chorus"
    assert rows[0]["inst_section_label"] == "drop"


def test_section_pair_does_not_change_score_total(scored):
    """T3.3 selects and describes; it must not re-rank. score_total is still the
    weighted composite of the four whole-track sub-scores."""
    db_path, _ = scored
    from config import MATCH_WEIGHTS
    from database.models import get_conn
    conn = get_conn(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM mashup_candidates").fetchall()]
    conn.close()
    for r in rows:
        expected = round(
            r["score_bpm"] * MATCH_WEIGHTS["bpm_score"]
            + r["score_key"] * MATCH_WEIGHTS["key_score"]
            + r["score_energy"] * MATCH_WEIGHTS["energy_score"]
            + r["score_timbre"] * MATCH_WEIGHTS["timbre_score"], 4)
        assert r["score_total"] == pytest.approx(expected, abs=1e-9)


def test_rows_survive_a_library_with_no_structure(tmp_path, monkeypatch):
    """A track analysed but not yet structure-detected still scores; the section
    columns stay NULL and the preview falls back to the track hook."""
    p = tmp_path / "nostruct.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    from database.models import get_conn, init_db, upsert_features, upsert_song
    init_db(p)
    for n in range(2):
        sid = upsert_song(f"S{n}", "A", f"https://sc/n{n}", 200, "Pop",
                          status="analysed", db_path=p)
        for stem in ("full", "vocals", "instrumental"):
            upsert_features(sid, stem, {
                "bpm": 120.0 + n, "key": "A", "mode": "minor", "camelot": "8A",
                "loudness_rms": 0.05, "energy": 0.5, "mfcc": [1.0] * 13,
            }, db_path=p)

    from matcher.match import score_all_pairs
    score_all_pairs(db_path=p, scorer="heuristic")
    conn = get_conn(p)
    rows = [dict(r) for r in conn.execute("SELECT * FROM mashup_candidates")]
    conn.close()
    assert rows
    assert all(r["vocal_section_idx"] is None for r in rows)
    assert all(r["score_section"] is None for r in rows)
