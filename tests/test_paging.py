"""C.3 — the ranked list is a library, not a top-50.

`limit` capped at 500 and there was no offset, so everything past the first page
was unreachable except by narrowing filters until it floated up.

The subtlety is the per-song cap. It is a greedy pass — a song only loses a row
once it already has its share of better ones — so it is stateful in rank order.
A SQL OFFSET would start that pass mid-list with empty counts, producing a page
2 that both repeats rows from page 1 and skips others. The offset is therefore
applied AFTER the cap, which means paging re-derives the pages before it.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "test.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


def _library(db_path, n_pairs, *, shared_vocal=False):
    """n_pairs candidates with descending totals.

    shared_vocal puts every pair on ONE vocal, so the per-song cap actually
    bites and the paging has to survive it.
    """
    from database.models import init_db, upsert_candidate, upsert_song

    init_db(db_path)
    side = lambda sid: {                                         # noqa: E731
        "song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
        "key": "A", "mode": "minor", "camelot": "8A",
        "loudness_rms": 0.05, "energy": 0.5,
    }
    v0 = upsert_song("v", "A", "https://sc/v", 200, status="analysed",
                     db_path=db_path)
    for n in range(n_pairs):
        v = v0 if shared_vocal else upsert_song(
            f"v{n}", "A", f"https://sc/v{n}", 200, status="analysed",
            db_path=db_path)
        i = upsert_song(f"i{n}", "A", f"https://sc/i{n}", 200,
                        status="analysed", db_path=db_path)
        upsert_candidate(side(v), side(i), {
            "total": 1.0 - n / (n_pairs * 2.0), "bpm_score": 0.9,
            "key_score": 0.9, "energy_score": 0.9, "timbre_score": 0.9,
        }, section_pair={"vocal_section_idx": n % 3, "inst_section_idx": 0,
                         "score_section": 0.7}, db_path=db_path)


def _ids(rows):
    return [r["id"] for r in rows]


def test_paging_walks_the_list_without_gaps_or_repeats(db_path):
    from database.models import get_candidates_enriched

    _library(db_path, 30)
    everything = _ids(get_candidates_enriched(limit=100, max_per_song=0,
                                              db_path=db_path))
    assert len(everything) == 30

    paged = []
    for offset in (0, 10, 20):
        paged += _ids(get_candidates_enriched(limit=10, offset=offset,
                                              max_per_song=0, db_path=db_path))
    assert paged == everything


def test_paging_survives_the_greedy_per_song_cap(db_path):
    """The case a SQL OFFSET gets wrong. Every pair shares one vocal, so the cap
    drops most of them — and which ones it drops depends on what came before."""
    from database.models import get_candidates_enriched

    _library(db_path, 40, shared_vocal=True)
    capped = _ids(get_candidates_enriched(limit=100, max_per_song=6,
                                          max_per_song_pair=0, db_path=db_path))
    assert 0 < len(capped) < 40, "the cap should actually bite here"

    paged = []
    for offset in range(0, len(capped) + 2, 2):
        paged += _ids(get_candidates_enriched(
            limit=2, offset=offset, max_per_song=6, max_per_song_pair=0,
            db_path=db_path))
    assert paged == capped
    assert len(paged) == len(set(paged)), "a page repeated a row"


def test_an_offset_past_the_end_is_empty_not_an_error(db_path):
    from database.models import get_candidates_enriched

    _library(db_path, 5)
    assert get_candidates_enriched(limit=10, offset=500, max_per_song=0,
                                   db_path=db_path) == []


def test_paging_depth_is_bounded(db_path):
    """Each page re-derives the ones before it, so the depth has to stop
    somewhere; past it the honest answer is 'narrow the filters'."""
    from database.models import MAX_PAGING_DEPTH, get_candidates_enriched

    _library(db_path, 3)
    # Absurd offsets clamp rather than allocating an unbounded pool.
    assert get_candidates_enriched(limit=5, offset=10 ** 9, max_per_song=0,
                                   db_path=db_path) == []
    assert MAX_PAGING_DEPTH > 0


@pytest.fixture()
def route(db_path, monkeypatch):
    """list_candidates against THIS test's database.

    get_candidates_enriched binds db_path as a default argument at import, so
    setting MASHUP_DB_PATH after the fact does not reach it — the route would
    read whatever database the process bound first.
    """
    import api.routes.mashups as mashups
    real = mashups.get_candidates_enriched
    monkeypatch.setattr(
        mashups, "get_candidates_enriched",
        lambda **kw: real(**{**kw, "db_path": db_path}))
    return mashups.list_candidates


def test_the_endpoint_reports_whether_there_is_more(db_path, route):
    _library(db_path, 25)

    first = route(limit=10, max_per_song=0)
    assert first["count"] == 10 and first["has_more"] is True
    assert first["offset"] == 0

    last = route(limit=10, offset=20, max_per_song=0)
    assert last["count"] == 5 and last["has_more"] is False


def test_has_more_is_false_on_an_exactly_full_final_page(db_path, route):
    """A full page at the end still reports more, and the NEXT page is what
    reveals the end — the alternative is a count query on every request."""
    _library(db_path, 20)
    assert route(limit=10, offset=10, max_per_song=0)["has_more"] is True
    assert route(limit=10, offset=20, max_per_song=0)["count"] == 0


def test_a_negative_offset_is_refused(db_path, route):
    from fastapi import HTTPException

    _library(db_path, 3)
    with pytest.raises(HTTPException):
        route(offset=-1)
