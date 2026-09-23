"""Ranking the pair list by one section term.

The dock draws four bars per card — LBL, DUR, VOI, PHR — and "show me the pairs
whose phrasing agrees" is a different question from "show me the best pairs".
Answering it has to happen in SQL: the dock is handed a top-40 the server
already truncated by score, so re-sorting that page would rank a page, not a
library.

The other thing these pin is NULL handling. A term is NULL when the pair was
scored before the column existed — the hatched bar in the dock — and ordering
it as zero would bury measured-but-mediocre pairs beneath unmeasured ones while
claiming to rank by the term.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "order.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


@pytest.fixture()
def library(db_path):
    """Four pairs whose composite order is the REVERSE of their label order,
    and one with no section terms at all, so a passing sort cannot be the
    score order wearing a different name."""
    from database.models import (
        bulk_upsert_candidates, candidate_row, init_db, upsert_song,
    )
    init_db(db_path)
    songs = [upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200, "Pop",
                         status="analysed", db_path=db_path)
             for n in range(6)]

    def add(v, i, total, terms):
        bulk_upsert_candidates([candidate_row(
            {"song_id": v, "title": f"S{v}", "artist": "A", "bpm": 128.0,
             "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
            {"song_id": i, "title": f"S{i}", "artist": "A", "bpm": 128.0,
             "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
            {"total": total, "bpm_score": 1.0, "key_score": 1.0,
             "energy_score": 0.5, "timbre_score": 0.5},
            section_pair=terms)], db_path=db_path)

    # score descends, every term ascends.
    for n in range(4):
        add(songs[n], songs[n + 1], 0.90 - n * 0.05, {
            "vocal_section_idx": 0, "inst_section_idx": 0,
            "score_label": 0.1 + n * 0.2,
            "score_duration": 0.1 + n * 0.2,
            "score_voice": 0.1 + n * 0.2,
            "score_phrase": 0.1 + n * 0.2,
        })
    # The unmeasured one, scored between the others so a NULL-as-zero bug
    # cannot hide at an end of the list.
    add(songs[0], songs[5], 0.80, {"vocal_section_idx": 1, "inst_section_idx": 1})
    return db_path


TERMS = [("label", "score_label"), ("duration", "score_duration"),
         ("voice", "score_voice"), ("phrase", "score_phrase")]


@pytest.mark.parametrize("order,col", TERMS)
def test_a_term_order_ranks_by_that_term(library, order, col):
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(limit=10, max_per_song=0, order=order,
                                   db_path=library)
    got = [r[col] for r in rows if r[col] is not None]
    assert got == sorted(got, reverse=True), got
    assert got[0] == pytest.approx(0.7), "the best term did not lead"


@pytest.mark.parametrize("order,col", TERMS)
def test_an_unmeasured_term_sorts_last_not_as_zero(library, order, col):
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(limit=10, max_per_song=0, order=order,
                                   db_path=library)
    assert rows[-1][col] is None, "the unmeasured pair is not at the bottom"
    assert all(r[col] is not None for r in rows[:-1])


def test_a_term_order_is_not_the_score_order(library):
    """The fixture is built so it cannot be — asserted so it stays honest."""
    from database.models import get_candidates_enriched
    by_score = get_candidates_enriched(limit=10, max_per_song=0, db_path=library)
    by_label = get_candidates_enriched(limit=10, max_per_song=0, order="label",
                                       db_path=library)
    key = lambda rs: [(r["vocal_song_id"], r["inst_song_id"]) for r in rs]
    assert key(by_score) != key(by_label)


def test_score_is_still_the_default(library):
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(limit=10, max_per_song=0, db_path=library)
    totals = [r["score_total"] for r in rows]
    assert totals == sorted(totals, reverse=True)


def test_the_route_accepts_the_term_orders_and_refuses_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mashups
    importlib.reload(mashups)
    from fastapi import HTTPException

    for order in ("score", "uncertain", "label", "duration", "voice", "phrase"):
        assert mashups.list_candidates(order=order)["order"] == order

    with pytest.raises(HTTPException) as e:
        mashups.list_candidates(order="loudest")
    assert e.value.status_code == 400
    assert "phrase" in e.value.detail, "the message should name what is allowed"


def test_the_sort_buttons_are_built_from_the_bars(tmp_path):
    """One table. A button labelled LBL that ordered by something else is a
    bug you would have to read SQL to notice."""
    from database.models import SECTION_TERM_ORDERS
    src = (ROOT / "frontend" / "src" / "components" / "pairs" / "pairModel.js"
           ).read_text(encoding="utf-8")
    terms = src[src.index("export const SCORE_TERMS"):]
    terms = terms[:terms.index("];")]
    for order, col in SECTION_TERM_ORDERS.items():
        assert f'order: "{order}"' in terms, order
        assert f'key: "{col}"' in terms, col

    hook = (ROOT / "frontend" / "src" / "hooks" / "usePairDock.js"
            ).read_text(encoding="utf-8")
    assert "SCORE_TERMS.map" in hook, "ORDERS should be derived, not retyped"
