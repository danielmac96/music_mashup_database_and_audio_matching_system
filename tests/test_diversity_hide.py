"""T3.4 — diversity cap, hidden pairs, excluded tracks, and the per-vocal view.

A flat top-50 with no diversity constraint means one vocal that happens to sit
at 128 BPM in 8A owns the page, so fifty rows are worth about eight real
choices. Hiding and excluding are display preferences, deliberately kept out of
pair_feedback: "I'm bored of this track" is not "this pairing sounds bad", and
the model must not learn the difference away.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "div.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


@pytest.fixture()
def library(db_path):
    """One dominant vocal that pairs with everything, plus 5 others.

    Scores descend with the bed index, so the natural ranking puts every one of
    the dominant vocal's pairs above anyone else's — exactly the failure the cap
    exists for."""
    from database.models import init_db, upsert_candidate, upsert_song
    init_db(db_path)
    songs = [upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200, "Pop",
                         status="analysed", db_path=db_path)
             for n in range(8)]
    hog = songs[0]

    def add(v, i, score):
        upsert_candidate(
            {"song_id": v, "title": f"S{v}", "artist": "A", "bpm": 128.0,
             "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
            {"song_id": i, "title": f"S{i}", "artist": "A", "bpm": 128.0,
             "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
            {"total": score, "bpm_score": 1.0, "key_score": 1.0,
             "energy_score": 0.5, "timbre_score": 0.5},
            db_path=db_path)

    # The hog takes the top 7 scores...
    for n, bed in enumerate(songs[1:]):
        add(hog, bed, 0.99 - n * 0.001)
    # ...then everyone else, well below.
    for n, v in enumerate(songs[1:4]):
        for m, i in enumerate(songs[4:]):
            add(v, i, 0.80 - n * 0.01 - m * 0.001)
    return db_path, songs


# ── the cap ───────────────────────────────────────────────────────────────────

def test_uncapped_list_is_dominated_by_one_song(library):
    """The behaviour the cap fixes — asserted so the fixture stays honest."""
    db_path, songs = library
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(limit=10, max_per_song=0, db_path=db_path)
    assert sum(1 for r in rows if songs[0] in (r["vocal_song_id"], r["inst_song_id"])) >= 7


def test_cap_limits_how_often_a_song_appears(library):
    db_path, songs = library
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(limit=50, max_per_song=3, db_path=db_path)
    counts = {}
    for r in rows:
        for sid in (r["vocal_song_id"], r["inst_song_id"]):
            counts[sid] = counts.get(sid, 0) + 1
    assert counts, "no rows returned"
    assert max(counts.values()) <= 3


def test_cap_of_one_is_respected(library):
    db_path, _ = library
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(limit=50, max_per_song=1, db_path=db_path)
    seen = set()
    for r in rows:
        for sid in (r["vocal_song_id"], r["inst_song_id"]):
            assert sid not in seen
            seen.add(sid)


def test_cap_keeps_the_best_rows_first(library):
    """Greedy down the ranked list: the top pair is never the one dropped."""
    db_path, _ = library
    from database.models import get_candidates_enriched
    top = get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path)[0]
    capped = get_candidates_enriched(limit=50, max_per_song=3, db_path=db_path)
    assert capped[0]["id"] == top["id"]
    scores = [r["score_total"] for r in capped]
    assert scores == sorted(scores, reverse=True)


def test_cap_zero_means_uncapped(library):
    """0 restores the pre-T3.4 behaviour exactly: the flat top-N, no filtering."""
    db_path, songs = library
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(limit=5, max_per_song=0, db_path=db_path)
    assert len(rows) == 5
    # All five belong to the hog, which a cap would have prevented.
    assert all(songs[0] in (r["vocal_song_id"], r["inst_song_id"]) for r in rows)


# ── hide / exclude ────────────────────────────────────────────────────────────

def test_hidden_pair_disappears_from_the_list(library):
    db_path, songs = library
    from database.models import get_candidates_enriched, hide_pair
    before = get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path)
    target = before[0]
    hide_pair(target["vocal_song_id"], target["inst_song_id"], db_path=db_path)
    after = get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path)
    assert len(after) == len(before) - 1
    assert all(not (r["vocal_song_id"] == target["vocal_song_id"]
                    and r["inst_song_id"] == target["inst_song_id"]) for r in after)


def test_hiding_is_idempotent_and_reversible(library):
    db_path, _ = library
    from database.models import (
        get_candidates_enriched, hide_pair, list_hidden, unhide_pair,
    )
    rows = get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path)
    t = rows[0]
    hide_pair(t["vocal_song_id"], t["inst_song_id"], db_path=db_path)
    hide_pair(t["vocal_song_id"], t["inst_song_id"], db_path=db_path)
    assert len(list_hidden(db_path=db_path)["pairs"]) == 1
    unhide_pair(t["vocal_song_id"], t["inst_song_id"], db_path=db_path)
    assert list_hidden(db_path=db_path)["pairs"] == []
    assert len(get_candidates_enriched(limit=50, max_per_song=0,
                                       db_path=db_path)) == len(rows)


def test_excluded_track_disappears_from_both_sides(library):
    db_path, songs = library
    from database.models import exclude_track, get_candidates_enriched
    exclude_track(songs[0], db_path=db_path)
    rows = get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path)
    assert rows
    assert all(songs[0] not in (r["vocal_song_id"], r["inst_song_id"]) for r in rows)


def test_excluding_is_reversible(library):
    db_path, songs = library
    from database.models import (
        exclude_track, get_candidates_enriched, include_track, list_hidden,
    )
    before = len(get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path))
    exclude_track(songs[0], db_path=db_path)
    assert len(list_hidden(db_path=db_path)["tracks"]) == 1
    include_track(songs[0], db_path=db_path)
    assert list_hidden(db_path=db_path)["tracks"] == []
    assert len(get_candidates_enriched(limit=50, max_per_song=0,
                                       db_path=db_path)) == before


def test_hidden_pairs_survive_a_rescore(library):
    """clear_candidates truncates mashup_candidates on every 'Score library'.
    A suppression the user has to redo after each re-score is not a feature."""
    db_path, songs = library
    from database.models import (
        clear_candidates, exclude_track, hide_pair, list_hidden,
    )
    hide_pair(songs[0], songs[1], db_path=db_path)
    exclude_track(songs[2], db_path=db_path)
    clear_candidates(db_path=db_path)
    state = list_hidden(db_path=db_path)
    assert len(state["pairs"]) == 1 and len(state["tracks"]) == 1


def test_list_hidden_carries_titles_for_the_undo_ui(library):
    db_path, songs = library
    from database.models import exclude_track, hide_pair, list_hidden
    hide_pair(songs[0], songs[1], db_path=db_path)
    exclude_track(songs[2], db_path=db_path)
    state = list_hidden(db_path=db_path)
    assert state["pairs"][0]["vocal_title"] == "S0"
    assert state["pairs"][0]["inst_title"] == "S1"
    assert state["tracks"][0]["title"] == "S2"


def test_include_hidden_shows_them_again(library):
    db_path, songs = library
    from database.models import get_candidates_enriched, hide_pair
    hide_pair(songs[0], songs[1], db_path=db_path)
    visible = get_candidates_enriched(limit=50, max_per_song=0, db_path=db_path)
    everything = get_candidates_enriched(limit=50, max_per_song=0,
                                         include_hidden=True, db_path=db_path)
    assert len(everything) == len(visible) + 1


# ── the per-vocal view ────────────────────────────────────────────────────────

def test_best_bed_per_vocal_gives_every_vocal_one_row(library):
    db_path, songs = library
    from database.models import best_bed_per_vocal
    rows = best_bed_per_vocal(db_path=db_path)
    vocals = [r["vocal_song_id"] for r in rows]
    assert len(vocals) == len(set(vocals)), "a vocal appears twice"
    # Every vocal that has any candidate at all is represented.
    assert set(vocals) == {songs[0], songs[1], songs[2], songs[3]}


def test_best_bed_per_vocal_picks_the_best_bed(library):
    db_path, songs = library
    from database.models import best_bed_per_vocal, get_candidates_for_song
    rows = {r["vocal_song_id"]: r for r in best_bed_per_vocal(db_path=db_path)}
    for vocal_id, row in rows.items():
        best = max(get_candidates_for_song(vocal_id, role="vocal",
                                           db_path=db_path),
                   key=lambda r: r["score_total"])
        assert row["inst_song_id"] == best["inst_song_id"]


def test_best_bed_per_vocal_can_return_more_than_one(library):
    db_path, _ = library
    from database.models import best_bed_per_vocal
    rows = best_bed_per_vocal(per_vocal=2, db_path=db_path)
    counts = {}
    for r in rows:
        counts[r["vocal_song_id"]] = counts.get(r["vocal_song_id"], 0) + 1
    assert max(counts.values()) == 2


def test_best_bed_per_vocal_respects_hiding(library):
    db_path, songs = library
    from database.models import best_bed_per_vocal, hide_pair
    first = {r["vocal_song_id"]: r["inst_song_id"]
             for r in best_bed_per_vocal(db_path=db_path)}
    hide_pair(songs[0], first[songs[0]], db_path=db_path)
    second = {r["vocal_song_id"]: r["inst_song_id"]
              for r in best_bed_per_vocal(db_path=db_path)}
    assert second[songs[0]] != first[songs[0]]


def test_best_bed_per_vocal_respects_exclusion(library):
    db_path, songs = library
    from database.models import best_bed_per_vocal, exclude_track
    exclude_track(songs[0], db_path=db_path)
    rows = best_bed_per_vocal(db_path=db_path)
    assert rows
    assert all(songs[0] not in (r["vocal_song_id"], r["inst_song_id"])
               for r in rows)
