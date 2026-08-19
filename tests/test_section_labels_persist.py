"""Section labels and bar counts must survive the write to mashup_candidates.

matcher.sections._pair_row has always computed vocal_section_label,
inst_section_label, section_bars_vocal, section_bars_bed, section_loop_repeats
and section_note. Until P2.0, SECTION_PAIR_COLUMNS bound only the seven index
and timestamp columns, so every one of them was silently discarded on the way
to the database and the UI had to re-derive the section pair to say anything
about it.

The failure mode this guards is quiet: the insert succeeds, the row looks fine,
and the information is simply gone.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    return models


def _sections(labels_and_spans, vocal=True):
    return [{"section_index": i, "start_sec": a, "end_sec": b, "label": lab,
             "energy": 0.7, "vocal_presence": 0.8 if vocal else 0.05,
             "repetition": 2}
            for i, (lab, a, b) in enumerate(labels_and_spans)]


def test_pair_row_and_best_section_pair_agree_on_shape():
    """Two entry points, one row shape. best_section_pair used to build its own
    dict and omit section_note, so a row's keys depended on how it was made."""
    from matcher.sections import best_section_pair, top_section_pairs

    v = _sections([("intro", 0, 16), ("chorus", 16, 48), ("verse", 48, 80)])
    i = _sections([("intro", 0, 16), ("drop", 16, 48), ("verse", 48, 80)], vocal=False)

    best = best_section_pair(v, i, stretch=1.0, bpm=128.0)
    top = top_section_pairs(v, i, stretch=1.0, bpm=128.0, limit=1)
    assert best is not None and top
    assert set(best) == set(top[0])


def test_every_pair_row_key_is_a_bound_column():
    """The bug in one line: _pair_row emits keys, SECTION_PAIR_COLUMNS binds a
    subset, and the difference is thrown away. Keep the sets equal."""
    from database.models import SECTION_PAIR_COLUMNS
    from matcher.sections import top_section_pairs

    v = _sections([("chorus", 0, 32), ("verse", 32, 64)])
    i = _sections([("drop", 0, 32), ("verse", 32, 64)], vocal=False)
    row = top_section_pairs(v, i, stretch=1.0, bpm=128.0, limit=1)[0]

    dropped = set(row) - set(SECTION_PAIR_COLUMNS)
    assert not dropped, f"_pair_row computes these but nothing binds them: {dropped}"


def test_labels_and_bars_reach_the_database(db):
    from matcher.sections import top_section_pairs

    # 30s at 128bpm is exactly 16 bars (64 beats), the canonical phrase length.
    v = _sections([("chorus", 0.0, 30.0)])
    i = _sections([("drop", 0.0, 30.0)], vocal=False)
    pair = top_section_pairs(v, i, stretch=1.0, bpm=128.0, limit=1)[0]

    vocal = {"song_id": 1, "title": "V", "artist": "A", "bpm": 128.0,
             "camelot": "8A", "loudness_rms": -8.0, "energy": 0.6}
    inst = {"song_id": 2, "title": "I", "artist": "B", "bpm": 128.0,
            "camelot": "8A", "loudness_rms": -8.0, "energy": 0.6}
    scores = {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0,
              "energy_score": 0.8, "timbre_score": 0.7, "collision_score": 0.6}

    db.upsert_song(title="V", artist="A", source_url="u://v")
    db.upsert_song(title="I", artist="B", source_url="u://i")
    db.bulk_upsert_candidates(
        [db.candidate_row(vocal, inst, scores, section_pair=pair)])

    conn = db.get_conn()
    row = conn.execute("SELECT * FROM mashup_candidates").fetchone()
    conn.close()

    assert row["vocal_section_label"] == "chorus"
    assert row["inst_section_label"] == "drop"
    assert row["section_bars_vocal"] == pair["section_bars_vocal"]
    assert row["section_bars_bed"] == pair["section_bars_bed"]
    assert row["section_loop_repeats"] == pair["section_loop_repeats"]
    assert row["section_bars_vocal"] == pytest.approx(16.0, abs=0.1)


def test_a_looped_bed_records_its_repeat_count(db):
    """A 16-bar bed under a 32-bar vocal is buildable by looping it twice, and
    that is the single most useful thing the row can tell a producer."""
    from matcher.sections import top_section_pairs

    v = _sections([("chorus", 0.0, 60.0)])
    i = _sections([("drop", 0.0, 30.0)], vocal=False)
    pair = top_section_pairs(v, i, stretch=1.0, bpm=128.0, limit=1)[0]
    assert pair["section_loop_repeats"] >= 2


def test_upsert_updates_the_new_columns(db):
    """The ON CONFLICT branch has its own column list; forgetting one there
    means the value is right on insert and stale forever after."""
    from matcher.sections import top_section_pairs

    db.upsert_song(title="V", artist="A", source_url="u://v")
    db.upsert_song(title="I", artist="B", source_url="u://i")
    vocal = {"song_id": 1, "title": "V", "artist": "A", "bpm": 128.0, "camelot": "8A"}
    inst = {"song_id": 2, "title": "I", "artist": "B", "bpm": 128.0, "camelot": "8A"}
    scores = {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0,
              "energy_score": 0.8, "timbre_score": 0.7, "collision_score": 0.6}

    first = top_section_pairs(_sections([("chorus", 0.0, 32.0)]),
                              _sections([("drop", 0.0, 32.0)], vocal=False),
                              stretch=1.0, bpm=128.0, limit=1)[0]
    db.bulk_upsert_candidates([db.candidate_row(vocal, inst, scores, section_pair=first)])

    # Same section indices -> same row, different labels.
    second = dict(first, vocal_section_label="verse", inst_section_label="breakdown")
    db.bulk_upsert_candidates([db.candidate_row(vocal, inst, scores, section_pair=second)])

    conn = db.get_conn()
    rows = conn.execute("SELECT * FROM mashup_candidates").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["vocal_section_label"] == "verse"
    assert rows[0]["inst_section_label"] == "breakdown"


def test_enriched_rows_expose_the_labels(db):
    """get_candidates_enriched is SELECT mc.*, so this is really asserting that
    nothing downstream drops them again on the way to the API."""
    from matcher.sections import top_section_pairs

    db.upsert_song(title="V", artist="A", source_url="u://v")
    db.upsert_song(title="I", artist="B", source_url="u://i")
    pair = top_section_pairs(_sections([("chorus", 0.0, 32.0)]),
                             _sections([("drop", 0.0, 32.0)], vocal=False),
                             stretch=1.0, bpm=128.0, limit=1)[0]
    db.bulk_upsert_candidates([db.candidate_row(
        {"song_id": 1, "title": "V", "artist": "A", "bpm": 128.0, "camelot": "8A"},
        {"song_id": 2, "title": "I", "artist": "B", "bpm": 128.0, "camelot": "8A"},
        {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0, "energy_score": 0.8,
         "timbre_score": 0.7, "collision_score": 0.6},
        section_pair=pair)])

    rows = db.get_candidates_enriched(limit=10)
    assert rows and rows[0]["vocal_section_label"] == "chorus"
    assert rows[0]["inst_section_label"] == "drop"


def test_section_pair_none_leaves_the_columns_null(db):
    """A pair with no usable structure on either side is still a valid row."""
    db.upsert_song(title="V", artist="A", source_url="u://v")
    db.upsert_song(title="I", artist="B", source_url="u://i")
    db.bulk_upsert_candidates([db.candidate_row(
        {"song_id": 1, "title": "V", "artist": "A", "bpm": 128.0, "camelot": "8A"},
        {"song_id": 2, "title": "I", "artist": "B", "bpm": 128.0, "camelot": "8A"},
        {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0, "energy_score": 0.8,
         "timbre_score": 0.7, "collision_score": 0.6},
        section_pair=None)])

    conn = db.get_conn()
    row = conn.execute("SELECT * FROM mashup_candidates").fetchone()
    conn.close()
    assert row["vocal_section_label"] is None
    assert row["section_bars_vocal"] is None
