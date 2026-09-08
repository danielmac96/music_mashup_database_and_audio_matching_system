"""The three section-score terms that were computed and thrown away.

`score_section` is a weighted sum of label / duration / voice / phrase /
rhythm / structure. Only the last three were ever stored, so a pair card that
wants to explain a score — the handoff's LBL / DUR / VOI / PHR bars — had
values for one of its four bars.

Two things must hold, and they pull against each other:

* `section_terms` must agree with `score_section_pair` to the last decimal,
  because they are two copies of the same arithmetic;
* `score_section_pair` must NOT call it. That function runs once per
  (vocal section x bed section) across the whole library; handing back a dict
  it is about to sum would cost an allocation per candidate pair.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SECTIONS_SRC = (ROOT / "matcher" / "sections.py").read_text(encoding="utf-8")


def _sec(idx, start, end, label, *, energy=0.7, vp=0.7, conf=0.8):
    return {"section_index": idx, "start_sec": start, "end_sec": end,
            "label": label, "energy": energy, "vocal_presence": vp,
            "repetition": 2, "confidence": conf}


# ── the terms themselves ──────────────────────────────────────────────────────

def test_the_terms_reconstruct_the_score():
    """Sum the six weighted terms and you get score_section back. This is the
    assertion that keeps the two copies of the arithmetic honest."""
    from matcher.section_score import section_components
    from matcher.sections import _weights, score_section_pair, section_terms
    v = _sec(0, 50, 80, "chorus", vp=0.9)
    i = _sec(0, 0, 30, "drop", vp=0.1)
    w = _weights()
    terms = dict(section_terms(v, i, 1.0, 124.0))
    terms.update(section_components(v, i, 1.0))
    total = sum(w[name] * terms[f"score_{name}"] for name in
                ("label", "duration", "voice", "phrase", "rhythm", "structure"))
    assert total == pytest.approx(score_section_pair(v, i, 1.0, 124.0), abs=5e-4)


def test_a_missing_vocal_stem_scores_neutral_not_zero():
    """vocal_presence None means the stem was never measured. Scoring it 0
    would mark a section down for a measurement nobody took."""
    from matcher.sections import section_terms
    unmeasured = _sec(0, 0, 30, "chorus")
    unmeasured["vocal_presence"] = None
    assert section_terms(unmeasured, _sec(0, 0, 30, "drop"), 1.0)["score_voice"] == 0.5


def test_every_term_is_a_unit_interval():
    from matcher.sections import section_terms
    for vp, labels in ((0.0, ("intro", "outro")), (1.0, ("chorus", "drop"))):
        t = section_terms(_sec(0, 0, 30, labels[0], vp=vp),
                          _sec(0, 0, 90, labels[1]), 1.0, 128.0)
        assert set(t) == {"score_label", "score_duration", "score_voice"}
        for k, val in t.items():
            assert 0.0 <= val <= 1.0, (k, val)


def test_the_hot_loop_does_not_pay_for_the_dict():
    """score_section_pair runs hundreds of thousands of times per re-score.
    section_terms exists so the STORED rows can explain themselves; wiring it
    into the scoring loop would make every candidate pay for that."""
    fn = SECTIONS_SRC[SECTIONS_SRC.index("def score_section_pair"):]
    fn = fn[:fn.index("\ndef ", 1)]
    assert "section_terms(" not in fn


# ── stored on the row ─────────────────────────────────────────────────────────

def test_the_pair_row_carries_the_terms():
    from matcher.sections import top_section_pairs
    rows = top_section_pairs([_sec(0, 0, 30, "verse"), _sec(1, 30, 60, "chorus")],
                             [_sec(0, 0, 30, "breakdown"), _sec(1, 30, 60, "drop")],
                             1.0, bpm=124.0)
    assert rows
    for r in rows:
        for k in ("score_label", "score_duration", "score_voice"):
            assert r[k] is not None, k
            assert 0.0 <= r[k] <= 1.0


def test_section_pair_columns_binds_them():
    """The P2.0 bug was exactly this: _pair_row computed values that
    SECTION_PAIR_COLUMNS never bound, so they were discarded on every write.
    Computing a term and not listing it here is the same bug again."""
    from database.models import SECTION_PAIR_COLUMNS, _CANDIDATE_INSERT_SQL
    for k in ("score_label", "score_duration", "score_voice"):
        assert k in SECTION_PAIR_COLUMNS, k
        assert k in _CANDIDATE_INSERT_SQL, k
    # candidate_row splats the tuple positionally, so a name added to one side
    # and not the other shifts every later binding silently.
    from database.models import candidate_row
    row = candidate_row({"song_id": 1}, {"song_id": 2},
                        {"total": 0.5, "bpm_score": 1.0, "key_score": 1.0,
                         "energy_score": 1.0, "timbre_score": 1.0})
    assert len(row) == _CANDIDATE_INSERT_SQL.count("?")


def test_a_scored_library_stores_them(tmp_path, monkeypatch):
    p = tmp_path / "terms.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    from database.models import (
        get_conn, init_db, replace_sections, upsert_features, upsert_song,
    )
    init_db(p)
    for n, bpm in enumerate((124.0, 126.0)):
        sid = upsert_song(f"S{n}", "A", f"https://sc/{n}", 200, "Pop",
                          status="analysed", db_path=p)
        for stem in ("full", "vocals", "instrumental"):
            upsert_features(sid, stem, {
                "bpm": bpm, "key": "A", "mode": "minor", "camelot": "8A",
                "loudness_rms": 0.04, "energy": 0.5,
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

    conn = get_conn(p)
    rows = [dict(r) for r in conn.execute(
        "SELECT score_label, score_duration, score_voice FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchall()]
    conn.close()
    assert rows
    assert all(r["score_label"] is not None for r in rows)
    assert all(r["score_voice"] is not None for r in rows)


def test_an_unscored_row_reports_null_not_zero(tmp_path, monkeypatch):
    """A pre-existing row has no terms. NULL means 'unmeasured' and the UI draws
    it as such; 0.0 would claim the section scored badly on every one."""
    p = tmp_path / "old.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    from database.models import get_conn, init_db
    init_db(p)
    conn = get_conn(p)
    conn.execute(
        "INSERT INTO mashup_candidates (combo_type, vocal_song_id, inst_song_id,"
        " score_total) VALUES ('vocal_over_instrumental', 1, 2, 0.8)")
    conn.commit()
    row = dict(conn.execute(
        "SELECT score_label, score_duration, score_voice "
        "FROM mashup_candidates").fetchone())
    conn.close()
    assert row == {"score_label": None, "score_duration": None, "score_voice": None}
