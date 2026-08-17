"""D.1 / D.2 — the shortlist is the output of a triage session.

Starring was a `useState(new Set())` in the browser: a refresh destroyed it, and
no export path could read it. An hour of listening, distilled to the twelve
pairs worth building, produced nothing you could act on — the only way out of
Discover was "Export top N", driven by filters rather than by the choices just
made by ear.

Two properties this file exists to hold:

* the star is keyed by the SECTION pair, because that is what a candidate row
  has been since E.3 — "that chorus over that drop", not "those two records";
* the star OUTLIVES a re-score. mashup_candidates is truncated on every scoring
  run, so a shortlist joined to it would empty itself exactly when the user
  changed a weight and re-scored.
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
    from database.models import init_db
    init_db(p)
    return p


def _songs(db_path, n=2):
    from database.models import upsert_song
    return [upsert_song(f"Song {k}", f"Artist {k}", f"https://sc/{k}", 200,
                        status="analysed", db_path=db_path) for k in range(n)]


# ── Keyed by the section pair ────────────────────────────────────────────────

def test_two_section_pairs_of_the_same_songs_are_two_entries(db_path):
    """'Chorus over drop' and 'verse over breakdown' are different ideas about
    the same two records, and starring one must not star the other."""
    from database.models import add_to_shortlist, get_shortlist

    v, i = _songs(db_path)
    add_to_shortlist(v, i, 1, 2, db_path=db_path)
    add_to_shortlist(v, i, 3, 0, db_path=db_path)

    rows = get_shortlist(db_path=db_path)
    assert len(rows) == 2
    assert {(r["vocal_section_idx"], r["inst_section_idx"]) for r in rows} == \
           {(1, 2), (3, 0)}


def test_starring_the_same_section_pair_twice_is_one_entry(db_path):
    from database.models import add_to_shortlist, get_shortlist

    v, i = _songs(db_path)
    add_to_shortlist(v, i, 1, 2, harmonic_shift=-2, db_path=db_path)
    add_to_shortlist(v, i, 1, 2, harmonic_shift=3, note="try it", db_path=db_path)

    rows = get_shortlist(db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["harmonic_shift"] == 3
    assert rows[0]["note"] == "try it"


def test_a_section_less_pair_does_not_collide_with_a_sectioned_one(db_path):
    """SQLite treats NULLs as distinct in a UNIQUE, so the index COALESCEs to
    -1. Without that, section-less rows would be free to duplicate."""
    from database.models import add_to_shortlist, get_shortlist

    v, i = _songs(db_path)
    add_to_shortlist(v, i, None, None, db_path=db_path)
    add_to_shortlist(v, i, None, None, db_path=db_path)
    add_to_shortlist(v, i, 0, 0, db_path=db_path)

    assert len(get_shortlist(db_path=db_path)) == 2


def test_unstarring_removes_only_that_section_pair(db_path):
    from database.models import (
        add_to_shortlist, get_shortlist, remove_from_shortlist,
    )

    v, i = _songs(db_path)
    add_to_shortlist(v, i, 1, 2, db_path=db_path)
    add_to_shortlist(v, i, 3, 0, db_path=db_path)

    assert remove_from_shortlist(v, i, 1, 2, db_path=db_path) == 1
    rows = get_shortlist(db_path=db_path)
    assert len(rows) == 1 and rows[0]["vocal_section_idx"] == 3


def test_unstarring_something_unstarred_is_not_an_error(db_path):
    from database.models import remove_from_shortlist
    v, i = _songs(db_path)
    assert remove_from_shortlist(v, i, 1, 2, db_path=db_path) == 0


# ── It outlives a re-score ───────────────────────────────────────────────────

def test_the_shortlist_survives_clear_candidates(db_path):
    """The reason it is its own table. score_all_pairs truncates
    mashup_candidates on every run; a shortlist joined to it would empty itself
    the moment the user changed a weight and re-scored."""
    from database.models import (
        add_to_shortlist, clear_candidates, get_shortlist, upsert_candidate,
    )

    v, i = _songs(db_path)
    side = lambda sid: {                                         # noqa: E731
        "song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
        "key": "A", "mode": "minor", "camelot": "8A",
        "loudness_rms": 0.05, "energy": 0.5,
    }
    upsert_candidate(side(v), side(i), {
        "total": 0.8, "bpm_score": 1.0, "key_score": 1.0,
        "energy_score": 0.9, "timbre_score": 0.9,
    }, section_pair={"vocal_section_idx": 1, "inst_section_idx": 2,
                     "score_section": 0.7}, db_path=db_path)
    add_to_shortlist(v, i, 1, 2, db_path=db_path)

    assert get_shortlist(db_path=db_path)[0]["score_total"] == pytest.approx(0.8)

    clear_candidates(db_path=db_path)
    rows = get_shortlist(db_path=db_path)
    assert len(rows) == 1, "a re-score must not empty the shortlist"
    # The live score is gone, which is honest — the pair is not currently in the
    # scored set — but the star, the sections and the titles remain.
    assert rows[0]["score_total"] is None
    assert rows[0]["vocal_title"] == "Song 0"
    assert rows[0]["vocal_section_idx"] == 1


def test_the_shortlist_carries_what_the_export_needs(db_path):
    """Each entry has to rebuild its own take: the section pair and the
    measured transpose (A.1), not just the two song ids."""
    from api.routes.mashups import _export_pair
    from database.models import add_to_shortlist, get_shortlist

    v, i = _songs(db_path)
    add_to_shortlist(v, i, 1, 2, harmonic_shift=-3, db_path=db_path)

    pair = _export_pair(get_shortlist(db_path=db_path)[0])
    assert pair == {"vocal_song_id": v, "inst_song_id": i,
                    "vocal_section_idx": 1, "inst_section_idx": 2,
                    "harmonic_shift": -3}


def test_clear_shortlist_empties_it(db_path):
    from database.models import add_to_shortlist, clear_shortlist, get_shortlist

    v, i = _songs(db_path)
    add_to_shortlist(v, i, 1, 2, db_path=db_path)
    add_to_shortlist(v, i, 3, 4, db_path=db_path)
    assert clear_shortlist(db_path=db_path) == 2
    assert get_shortlist(db_path=db_path) == []


def test_starring_is_not_a_verdict(db_path):
    """'I want to build this' is not 'this sounded good'. Folding the two
    together would teach the model that everything queued for export was
    loved."""
    from database.models import add_to_shortlist, get_pair_feedback

    v, i = _songs(db_path)
    add_to_shortlist(v, i, 1, 2, db_path=db_path)
    assert get_pair_feedback(db_path=db_path) == []
