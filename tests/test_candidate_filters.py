"""T3.5 — the filter chips: genre, era, energy band, BPM band, vocal-forward.

All of them run in SQL. Filtering a truncated 50 client-side would search the
top of the list rather than the library, which is the opposite of what a filter
is for — so these tests check the filters find rows that are NOT in the
unfiltered page.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "filt.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


SONGS = [
    # (genre, year, bpm, energy, vocal_presence of the best vocal section)
    ("House",   2024,  128.0, 0.90, 0.9),
    ("House",   2023,  126.0, 0.80, 0.8),
    ("Hip Hop", 2015,   92.0, 0.30, 0.7),
    ("Hip Hop", 2012,   95.0, 0.25, 0.2),
    ("Rock",    1998,  140.0, 0.60, 0.5),
    ("Rock",    2005,  145.0, 0.55, 0.1),
]


@pytest.fixture()
def library(db_path):
    from database.models import (
        init_db, replace_sections, upsert_candidate, upsert_song,
    )
    init_db(db_path)
    ids = []
    for n, (genre, year, bpm, energy, vp) in enumerate(SONGS):
        sid = upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200, genre,
                          release_year=year, status="analysed", db_path=db_path)
        ids.append(sid)
        replace_sections(sid, [
            {"start_sec": 0, "end_sec": 30, "label": "intro",
             "energy": 0.2, "vocal_presence": 0.0, "repetition": 1,
             "confidence": 0.6},
            {"start_sec": 30, "end_sec": 60, "label": "chorus",
             "energy": energy, "vocal_presence": vp, "repetition": 2,
             "confidence": 0.8},
        ], db_path=db_path)

    # Every song over every other, scored so the House pairs sit on top —
    # a filter that only searched the visible page would never reach the rest.
    for a, (ga, ya, ba, ea, _) in zip(ids, SONGS):
        for b, (gb, yb, bb, eb, _) in zip(ids, SONGS):
            if a >= b:
                continue
            score = 0.9 if ga == "House" else 0.5 if ga == "Hip Hop" else 0.3
            upsert_candidate(
                {"song_id": a, "title": f"S{a}", "artist": "A", "bpm": ba,
                 "camelot": "8A", "loudness_rms": 0.1, "energy": ea},
                {"song_id": b, "title": f"S{b}", "artist": "A", "bpm": bb,
                 "camelot": "8A", "loudness_rms": 0.1, "energy": eb},
                {"total": score, "bpm_score": 1.0, "key_score": 1.0,
                 "energy_score": 0.5, "timbre_score": 0.5},
                db_path=db_path)
    # Point every row at its chorus so vocal_forward reads the right section.
    from database.models import get_conn
    conn = get_conn(db_path)
    conn.execute("UPDATE mashup_candidates SET vocal_section_idx=1, "
                 "inst_section_idx=1")
    conn.commit()
    conn.close()
    return db_path, ids


def _rows(db_path, **kw):
    from database.models import get_candidates_enriched
    return get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path, **kw)


def test_genre_matches_either_side(library):
    db_path, _ = library
    rows = _rows(db_path, genre="Rock")
    assert rows
    assert all("Rock" in (r["vocal_genre"], r["inst_genre"]) for r in rows)


def test_genre_finds_rows_below_the_unfiltered_page(library):
    """The Rock pairs score lowest, so a client-side filter over the top rows
    would find nothing."""
    db_path, _ = library
    top3 = {r["id"] for r in _rows(db_path)[:3]}
    rock = {r["id"] for r in _rows(db_path, genre="Rock")}
    assert rock - top3


def test_era_matches_either_side(library):
    db_path, _ = library
    rows = _rows(db_path, era="1990s")
    assert rows
    for r in rows:
        assert 1990 <= (r["vocal_year"] or 0) <= 1999 \
            or 1990 <= (r["inst_year"] or 0) <= 1999


def test_unknown_release_year_is_not_treated_as_old(db_path):
    """release_year 0 means the upload date never resolved. 'pre-1990' must not
    scoop up every track whose metadata is missing."""
    from database.models import init_db, upsert_candidate, upsert_song
    init_db(db_path)
    a = upsert_song("A", "A", "https://sc/a", 200, "Pop", release_year=0,
                    status="analysed", db_path=db_path)
    b = upsert_song("B", "B", "https://sc/b", 200, "Pop", release_year=0,
                    status="analysed", db_path=db_path)
    upsert_candidate(
        {"song_id": a, "title": "A", "artist": "A", "bpm": 120.0,
         "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
        {"song_id": b, "title": "B", "artist": "B", "bpm": 120.0,
         "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
        {"total": 0.8, "bpm_score": 1.0, "key_score": 1.0,
         "energy_score": 0.5, "timbre_score": 0.5}, db_path=db_path)
    assert _rows(db_path)
    assert _rows(db_path, era="pre-1990") == []


def test_bpm_band_is_the_target_tempo(library):
    db_path, _ = library
    rows = _rows(db_path, bpm_band="125-134")
    assert rows
    assert all(125.0 <= r["vocal_bpm"] < 135.0 for r in rows)
    assert _rows(db_path, bpm_band="150+") == []


def test_bpm_bands_do_not_overlap(library):
    db_path, _ = library
    from database.models import BPM_BANDS
    seen = set()
    for band in BPM_BANDS:
        ids = {r["id"] for r in _rows(db_path, bpm_band=band)}
        assert not (ids & seen), f"{band} overlaps an earlier band"
        seen |= ids
    assert seen == {r["id"] for r in _rows(db_path)}


def test_energy_band_ranks_within_the_library(library):
    db_path, _ = library
    high = _rows(db_path, energy="high")
    low = _rows(db_path, energy="low")
    assert high and low
    assert not ({r["id"] for r in high} & {r["id"] for r in low})
    assert min(r["inst_energy"] for r in high) >= max(r["inst_energy"] for r in low)


def test_vocal_forward_uses_the_section_that_will_play(library):
    db_path, _ = library
    rows = _rows(db_path, vocal_forward=True)
    assert rows
    # Songs 3 and 5 have an ad-lib-level chorus (0.2 / 0.1) — never the vocal.
    quiet = {ids for ids, s in zip(range(1, len(SONGS) + 1), SONGS) if s[4] < 0.6}
    assert all(r["vocal_song_id"] not in quiet for r in rows)


def test_filters_compose(library):
    db_path, _ = library
    both = _rows(db_path, genre="House", bpm_band="125-134")
    assert both
    assert all("House" in (r["vocal_genre"], r["inst_genre"])
               and 125.0 <= r["vocal_bpm"] < 135.0 for r in both)
    # Composing narrows: never more rows than either filter alone.
    assert len(both) <= min(len(_rows(db_path, genre="House")),
                            len(_rows(db_path, bpm_band="125-134")))


def test_composing_to_nothing_is_empty_not_an_error(library):
    db_path, _ = library
    assert _rows(db_path, genre="House", bpm_band="150+") == []


def test_genre_and_era_may_be_satisfied_by_opposite_sides(library):
    """Both match "either side", so a 2024 House vocal over a 1998 Rock bed
    satisfies genre=House AND era=1990s. That is the intended reading of "show
    me the House pairs" plus "show me the 90s pairs" — the pair qualifies if it
    contains such a record, not only if both of them are."""
    db_path, _ = library
    rows = _rows(db_path, genre="House", era="1990s")
    assert rows
    for r in rows:
        assert "House" in (r["vocal_genre"], r["inst_genre"])
        assert 1990 <= (r["vocal_year"] or 0) <= 1999 \
            or 1990 <= (r["inst_year"] or 0) <= 1999


def test_bad_band_values_raise(library):
    db_path, _ = library
    for kw in ({"era": "medieval"}, {"energy": "nuclear"}, {"bpm_band": "fast"}):
        with pytest.raises(ValueError):
            _rows(db_path, **kw)


def test_filter_options_only_offer_what_exists(library):
    db_path, _ = library
    from database.models import candidate_filter_options
    opts = candidate_filter_options(db_path=db_path)
    genres = {g["genre"] for g in opts["genres"]}
    assert genres == {"House", "Hip Hop", "Rock"}
    assert "2020s" in opts["eras"] and "1990s" in opts["eras"]
    assert "pre-1990" not in opts["eras"]
    assert opts["energy_bands"] == ["low", "mid", "high"]


def test_filters_still_respect_hiding(library):
    db_path, _ = library
    from database.models import hide_pair
    rows = _rows(db_path, genre="House")
    t = rows[0]
    hide_pair(t["vocal_song_id"], t["inst_song_id"], db_path=db_path)
    assert len(_rows(db_path, genre="House")) == len(rows) - 1
