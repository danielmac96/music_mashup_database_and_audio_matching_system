"""Sets that scale (2026-10-10 persona simulation, phase E).

get_set looked up each item's live row with its own listing query — 19 ms
apiece, 1.4 s for a 72-item set on every load and every add — and the Sets
screen was blank until it arrived. One query now serves every item; the
screen shows the set while it loads; the dock offers what comes next.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def _side(sid, bpm=126.0):
    return {"song_id": sid, "title": f"S{sid}", "artist": "A", "bpm": bpm,
            "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5}


@pytest.fixture()
def big_set(tmp_path, monkeypatch):
    db = tmp_path / "sets.db"
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
    s = models.create_set("Big", db_path=db)
    for v in vocals:
        for b in beds:
            models.add_set_item(s["id"], v, b, db_path=db)
    # One item whose pair is no longer scored: it must come back stale.
    gone = models.upsert_song("Gone", "Z", "https://sc/gone", 200, status="analysed",
                              db_path=db)
    conn = models.get_conn(db)
    conn.execute("INSERT INTO set_items (set_id, position, vocal_song_id, inst_song_id, "
                 "payload_json) VALUES (?, 999, ?, ?, '{}')", (s["id"], gone, beds[0]))
    conn.commit()
    conn.close()
    return models, db, s["id"]


def test_batched_rows_equal_the_per_item_rows(big_set):
    models, db, set_id = big_set
    got = models.get_set(set_id, db_path=db)["items"]
    assert len(got) == 37
    for it in got:
        one = models.candidate_for_pair(it["vocal_song_id"], it["inst_song_id"],
                                        it["vocal_section_idx"], it["inst_section_idx"],
                                        db_path=db)
        if one is None:
            assert it["stale"] is True
        else:
            assert not it["stale"]
            assert it["score_total"] == one["score_total"]
            assert it["inst_title"] == one["inst_title"]
    assert sum(1 for it in got if it["stale"]) == 1


def test_a_set_loads_in_a_constant_number_of_queries(big_set, monkeypatch):
    models, db, set_id = big_set
    calls = []
    real = models.get_conn

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(models, "get_conn", counting)
    models.get_set(set_id, db_path=db)
    assert len(calls) <= 3, f"{len(calls)} connections for one set"


def test_best_row_without_sections_and_exact_row_with_them(big_set):
    models, db, _ = big_set
    out = models.candidates_for_pairs([(1, 5, None, None), (1, 5, 0, 0), (1, 999, None, None)],
                                      db_path=db)
    assert out[(1, 5, None, None)]["inst_song_id"] == 5
    assert out[(1, 5, 0, 0)] is None          # the fixture rows carry no sections
    assert out[(1, 999, None, None)] is None
    assert models.candidates_for_pairs([], db_path=db) == {}


def test_the_sets_screen_is_never_blank_while_loading():
    src = _read("components/SetScreen.jsx")
    assert 'if (!data) return <div className="set-screen"><ScreenHeader title="Sets" /></div>;' not in src
    loading = src[src.index("if (!data || data.id !== active.id)"):]
    loading = loading[:loading.index("  }\n")]
    assert "Loading" in loading and "active.name" in loading


def test_adding_updates_the_count_and_offers_what_comes_next():
    sets = _read("hooks/useSets.js")
    assert "item_count: res.items.length" in sets
    assert "requestNextForSet(target.id, target.name)" in sets
    hook = _read("hooks/usePairDock.js")
    assert "onNextForSet(" in hook and "api.getSetNext(nextFor.id, 20)" in hook
    dock = _read("components/PairDock.jsx")
    assert "What comes next in" in dock and "dock.clearNextFor" in dock
