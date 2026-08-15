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
        "WHERE combo_type='vocal_over_instrumental' "
        "ORDER BY score_total DESC").fetchall()]
    conn.close()
    assert rows
    for r in rows:
        assert r["vocal_section_idx"] is not None
        assert r["inst_section_idx"] is not None
        assert r["vocal_section_end"] > r["vocal_section_start"]
        assert r["inst_section_end"] > r["inst_section_start"]
        assert 0.0 <= r["score_section"] <= 1.0
    # The best row still takes the chorus over the drop; E.3 only means the
    # weaker pairings now get rows of their own rather than being discarded.
    assert rows[0]["vocal_section_idx"] == 2
    assert rows[0]["inst_section_idx"] == 3


def test_one_row_per_section_pair(scored):
    """E.3 — the candidate IS the section pair. "chorus over drop" and "verse
    over breakdown" are different ideas about the same two records, and the old
    UNIQUE(combo_type, vocal, inst) collapsed them into one row."""
    from config import MAX_SECTION_PAIRS_PER_SONG_PAIR
    from database.models import get_conn
    db_path, _ = scored
    conn = get_conn(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchall()]
    conn.close()

    by_song_pair: dict = {}
    for r in rows:
        by_song_pair.setdefault(
            (r["vocal_song_id"], r["inst_song_id"]), []).append(r)
    assert any(len(v) > 1 for v in by_song_pair.values()), \
        "no song pair produced more than one section pairing"
    for pairings in by_song_pair.values():
        # Capped, and each (vocal section, bed section) appears exactly once.
        assert len(pairings) <= MAX_SECTION_PAIRS_PER_SONG_PAIR
        keys = [(p["vocal_section_idx"], p["inst_section_idx"]) for p in pairings]
        assert len(keys) == len(set(keys))
        # At most one row per vocal section, so one strong chorus cannot take
        # every slot by pairing with three different bed sections.
        v_idx = [k[0] for k in keys]
        assert len(v_idx) == len(set(v_idx))


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


def test_section_fit_is_part_of_the_total(scored):
    """E.3 inverts T3.3.

    T3.3's rule was "selects and describes; it must not re-rank" — correct while
    the row was a SONG pair and the section was chosen afterwards, because
    folding a post-hoc annotation into the ranking would be double-counting.
    Now the row IS the section pair, so how well those two sections cover each
    other is part of what is being ranked.

    Rows with no section pair (instrumental-over-instrumental, or a track with
    no structure yet) must still be the plain whole-track composite."""
    db_path, _ = scored
    from config import EFFORT_WEIGHT, SECTION_WEIGHT, current_match_weights
    from database.models import get_conn
    conn = get_conn(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM mashup_candidates").fetchall()]
    conn.close()
    assert rows
    saw_section, saw_plain = False, False
    for r in rows:
        # Per-combo weights (P1.3): the vocal path moves timbre onto collision.
        weights = current_match_weights(r["combo_type"])
        whole = sum(r[col] * weights[name] for name, col in (
            ("bpm_score", "score_bpm"), ("key_score", "score_key"),
            ("energy_score", "score_energy"), ("timbre_score", "score_timbre"),
            ("collision_score", "score_collision")))
        if r["score_section"] is None:
            blended = whole
            saw_plain = True
        else:
            blended = ((1.0 - SECTION_WEIGHT) * whole
                       + SECTION_WEIGHT * r["score_section"])
            saw_section = True
        expected = round(blended * (1.0 - EFFORT_WEIGHT * r["score_effort"]), 4)
        assert r["score_total"] == pytest.approx(expected, abs=1e-9)
    assert saw_section and saw_plain, "fixture must cover both shapes"


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


def test_the_list_shows_one_section_pairing_per_song_pair_by_default(scored):
    """E.3 emits a row per section pairing, so without a cap one song pair could
    take three of the top ten with what reads as the same suggestion three
    times. The extra pairings stay in the table and stay reachable by seeding."""
    from database.models import get_candidates_enriched
    db_path, _ = scored
    rows = get_candidates_enriched(combo_type="vocal_over_instrumental",
                                   db_path=db_path)
    pairs = [(r["vocal_song_id"], r["inst_song_id"]) for r in rows]
    assert len(pairs) == len(set(pairs))


def test_seeding_on_a_track_still_reaches_every_section_pairing(scored):
    """The cap is a browsing convenience, not a filter on the data."""
    from database.models import get_candidates_enriched, get_conn
    db_path, _ = scored
    conn = get_conn(db_path)
    row = conn.execute(
        "SELECT vocal_song_id, inst_song_id, COUNT(*) AS n "
        "FROM mashup_candidates WHERE combo_type='vocal_over_instrumental' "
        "GROUP BY vocal_song_id, inst_song_id ORDER BY n DESC LIMIT 1").fetchone()
    conn.close()
    assert row["n"] > 1
    rows = get_candidates_enriched(
        combo_type="vocal_over_instrumental",
        vocal_song_id=row["vocal_song_id"], inst_song_id=row["inst_song_id"],
        max_per_song=0, max_per_song_pair=0, db_path=db_path)
    assert len(rows) == row["n"]


def test_migration_widens_an_existing_candidate_key(tmp_path, monkeypatch):
    """A database created before E.3 carries UNIQUE(combo_type, vocal, inst) as
    a TABLE constraint, which SQLite cannot drop in place. The migration must
    rebuild it — safe because score_all_pairs truncates this table on every run
    and every durable thing the user owns lives in another table."""
    import sqlite3
    p = tmp_path / "legacy.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))

    # A minimal pre-E.3 table with the old key.
    raw = sqlite3.connect(p)
    raw.execute("""CREATE TABLE mashup_candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        combo_type TEXT NOT NULL,
        vocal_song_id INTEGER NOT NULL,
        inst_song_id INTEGER NOT NULL,
        score_total REAL,
        UNIQUE(combo_type, vocal_song_id, inst_song_id))""")
    raw.execute("INSERT INTO mashup_candidates (combo_type, vocal_song_id, "
                "inst_song_id, score_total) VALUES ('v', 1, 2, 0.9)")
    raw.commit()
    raw.close()

    from database.models import get_conn, init_db
    init_db(p)
    conn = get_conn(p)
    ddl = conn.execute("SELECT sql FROM sqlite_master "
                       "WHERE name='mashup_candidates'").fetchone()[0]
    idx = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name='ux_candidates_section_pair'")]
    kept = conn.execute("SELECT score_total FROM mashup_candidates").fetchall()
    conn.close()

    assert "UNIQUE(combo_type, vocal_song_id, inst_song_id)" not in ddl
    assert idx == ["ux_candidates_section_pair"]
    # The rebuild carries existing rows over rather than dropping them.
    assert [r[0] for r in kept] == [0.9]
