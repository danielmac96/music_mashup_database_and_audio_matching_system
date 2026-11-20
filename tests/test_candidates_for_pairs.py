"""models.candidates_for_pairs: many pairs' live rows in one query, each the
row candidate_for_pair returns alone. (Its first reader was the retired Sets
screen; the dock's playback terms and Segments still read it.)"""
import pytest


def _side(sid, bpm=126.0):
    return {"song_id": sid, "title": f"S{sid}", "artist": "A", "bpm": bpm,
            "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5}


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    db = tmp_path / "pairs.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    from database import models
    models.init_db(db)
    ids = [models.upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200, "House",
                              status="analysed", db_path=db) for n in range(13)]
    vocals, beds = ids[:4], ids[4:]
    for v in vocals:
        for k, b in enumerate(beds):
            models.upsert_candidate(_side(v), _side(b),
                                    {"total": 0.9 - k * 0.01, "bpm_score": 1.0,
                                     "key_score": 1.0, "energy_score": 0.5,
                                     "timbre_score": 0.5}, db_path=db)
    return models, db, vocals, beds


def test_batched_rows_equal_the_per_pair_rows(lib):
    models, db, vocals, beds = lib
    keys = [(v, b, None, None) for v in vocals for b in beds] + [(vocals[0], 999, None, None)]
    got = models.candidates_for_pairs(keys, db_path=db)
    assert len(got) == len(keys)
    for key, row in got.items():
        one = models.candidate_for_pair(*key, db_path=db)
        if one is None:
            assert row is None
        else:
            assert row["score_total"] == one["score_total"]
            assert row["inst_title"] == one["inst_title"]
    assert sum(1 for r in got.values() if r is None) == 1


def test_one_query_for_many_pairs(lib, monkeypatch):
    models, db, vocals, beds = lib
    calls = []
    real = models.get_conn

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(models, "get_conn", counting)
    models.candidates_for_pairs([(v, b, None, None) for v in vocals for b in beds], db_path=db)
    assert len(calls) <= 2, f"{len(calls)} connections for one batch"


def test_best_row_without_sections_and_exact_row_with_them(lib):
    models, db, _, _ = lib
    out = models.candidates_for_pairs([(1, 5, None, None), (1, 5, 0, 0), (1, 999, None, None)],
                                      db_path=db)
    assert out[(1, 5, None, None)]["inst_song_id"] == 5
    assert out[(1, 5, 0, 0)] is None          # the fixture rows carry no sections
    assert out[(1, 999, None, None)] is None
    assert models.candidates_for_pairs([], db_path=db) == {}
