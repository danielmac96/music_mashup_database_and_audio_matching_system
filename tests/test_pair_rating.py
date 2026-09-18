"""Per-pair 1-5 ratings, sitting ALONGSIDE the three-way verdict.

The revamped dock collects stars; the learned scorer trains on `verdict`
(matcher/features.py, dataset/). Storing both, with a total mapping in each
direction, is what lets the UI change without touching the training path.

The failures these guard against:

* a star that does not write its verdict — the scorer silently stops seeing
  new judgements while the UI looks like it is collecting them;
* a ✓/~/✗ correction that blanks a star already given;
* a rating write that collapses the section-scoped unique key, which is the
  P2.0 data-loss bug in a new coat.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "rating.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    from database.models import init_db
    init_db(p)
    return p


# ── the mapping ───────────────────────────────────────────────────────────────

def test_the_mapping_is_total_in_both_directions():
    from database.models import (
        VERDICTS, rating_for_verdict, verdict_for_rating,
    )
    assert [verdict_for_rating(n) for n in (1, 2, 3, 4, 5)] == \
        ["no", "no", "ok", "love", "love"]
    # Every verdict has a star, so a row judged before stars existed still
    # renders as one rather than as an empty rating cell.
    for v in VERDICTS:
        assert rating_for_verdict(v) in (1, 3, 5)
    assert verdict_for_rating(None) is None
    assert rating_for_verdict(None) is None


def test_a_derived_star_round_trips_to_its_own_verdict():
    """4 and 2 are unreachable from a verdict, and that is correct — nobody
    said them. What must hold is that deriving back never changes the verdict."""
    from database.models import rating_for_verdict, verdict_for_rating
    for v in ("love", "ok", "no"):
        assert verdict_for_rating(rating_for_verdict(v)) == v


# ── writing ───────────────────────────────────────────────────────────────────

def test_a_star_writes_its_verdict_too(db):
    """The scorer reads `verdict`. If a star did not write one, training data
    would quietly stop arriving the day the UI switched to stars."""
    from database.models import get_pair_feedback, upsert_pair_feedback
    upsert_pair_feedback(1, 2, rating=4, db_path=db)
    row, = get_pair_feedback(db_path=db)
    assert row["rating"] == 4
    assert row["verdict"] == "love"


def test_a_verdict_alone_leaves_the_star_unset_in_the_table(db):
    """...but readers still see one. Storing an invented 5 would claim the user
    picked a star they never picked; deriving it on read does not."""
    from database.models import get_conn, get_pair_feedback, upsert_pair_feedback
    upsert_pair_feedback(1, 2, "ok", db_path=db)
    conn = get_conn(db)
    stored = conn.execute("SELECT rating FROM pair_feedback").fetchone()[0]
    conn.close()
    assert stored is None
    assert get_pair_feedback(db_path=db)[0]["rating"] == 3


def test_a_verdict_correction_does_not_wipe_an_existing_star(db):
    from database.models import get_pair_feedback, upsert_pair_feedback
    upsert_pair_feedback(1, 2, rating=5, db_path=db)
    upsert_pair_feedback(1, 2, "no", db_path=db)     # the ✓/~/✗ path
    row, = get_pair_feedback(db_path=db)
    assert row["verdict"] == "no"
    assert row["rating"] == 5


def test_re_rating_corrects_rather_than_duplicating(db):
    from database.models import get_pair_feedback, upsert_pair_feedback
    upsert_pair_feedback(1, 2, rating=2, vocal_section=6, inst_section=1, db_path=db)
    upsert_pair_feedback(1, 2, rating=5, vocal_section=6, inst_section=1, db_path=db)
    rows = get_pair_feedback(db_path=db)
    assert len(rows) == 1
    assert (rows[0]["rating"], rows[0]["verdict"]) == (5, "love")


def test_ratings_stay_keyed_on_the_section_pair(db):
    """The P2.0 bug, re-checked through the new column: two overlays of the same
    two records are different judgements and must both survive."""
    from database.models import get_pair_feedback, upsert_pair_feedback
    upsert_pair_feedback(1, 2, rating=5, vocal_section=6, inst_section=1, db_path=db)
    upsert_pair_feedback(1, 2, rating=1, vocal_section=10, inst_section=2, db_path=db)
    rows = get_pair_feedback(db_path=db)
    assert len(rows) == 2
    assert {r["rating"] for r in rows} == {1, 5}


def test_a_rating_out_of_range_is_refused(db):
    from database.models import upsert_pair_feedback
    for bad in (0, 6, -1):
        with pytest.raises(ValueError):
            upsert_pair_feedback(1, 2, rating=bad, db_path=db)


def test_neither_a_verdict_nor_a_rating_is_refused(db):
    from database.models import upsert_pair_feedback
    with pytest.raises(ValueError):
        upsert_pair_feedback(1, 2, db_path=db)


# ── the training path is untouched ────────────────────────────────────────────

def test_the_scorer_still_reads_verdicts_not_ratings():
    """matcher/features.py is the training-row builder. If it ever learns about
    `rating`, the 'ratings ride alongside' decision has quietly been undone and
    the two columns can disagree about what the user meant."""
    src = (ROOT / "matcher" / "features.py").read_text(encoding="utf-8")
    assert "verdict" in src
    assert "rating" not in src


def test_starred_rows_reach_the_training_query_as_verdicts(db):
    """A star must be indistinguishable from the equivalent ✓ downstream."""
    from database.models import get_pair_feedback, upsert_pair_feedback
    upsert_pair_feedback(1, 2, rating=5, db_path=db)
    upsert_pair_feedback(3, 4, "love", db_path=db)
    assert {r["verdict"] for r in get_pair_feedback("love", db_path=db)} == {"love"}
    assert len(get_pair_feedback("love", db_path=db)) == 2


# ── clearing ─────────────────────────────────────────────────────────────────
# The one path that takes something away. pair_feedback is otherwise
# append-and-correct, so the delete has to be as precisely keyed as the write.

def test_clearing_removes_the_row_the_four_ids_name(db):
    from database.models import (
        delete_pair_feedback, get_pair_feedback, upsert_pair_feedback,
    )
    upsert_pair_feedback(1, 2, None, vocal_section=0, inst_section=3,
                         rating=4, db_path=db)
    upsert_pair_feedback(1, 2, None, vocal_section=1, inst_section=5,
                         rating=2, db_path=db)

    assert delete_pair_feedback(1, 2, vocal_section=0, inst_section=3,
                                db_path=db) == 1

    left = get_pair_feedback(db_path=db)
    assert len(left) == 1, "the sibling section pair went with it"
    assert (left[0]["vocal_section"], left[0]["inst_section"]) == (1, 5)


def test_clearing_takes_the_verdict_with_the_star(db):
    """`verdict` is NOT NULL, so there is no "rated nothing" state to fall back
    to. A stray 3 wrote verdict='ok' as well, and clearing the star has to
    clear what it implied."""
    from database.models import (
        delete_pair_feedback, get_pair_feedback, upsert_pair_feedback,
    )
    upsert_pair_feedback(7, 8, None, vocal_section=2, inst_section=2,
                         rating=3, db_path=db)
    assert get_pair_feedback(db_path=db)[0]["verdict"] == "ok"

    delete_pair_feedback(7, 8, vocal_section=2, inst_section=2, db_path=db)
    assert get_pair_feedback(db_path=db) == []


def test_clearing_matches_null_sections_the_way_the_index_does(db):
    """The unique index COALESCEs a NULL section to -1. Match it loosely here
    and a NULL-sectioned row survives a clear that reported success."""
    from database.models import (
        delete_pair_feedback, get_pair_feedback, upsert_pair_feedback,
    )
    upsert_pair_feedback(4, 5, None, rating=5, db_path=db)
    assert delete_pair_feedback(4, 5, db_path=db) == 1
    assert get_pair_feedback(db_path=db) == []


def test_clearing_nothing_is_not_an_error(db):
    """The UI clears optimistically and only then tells the server. Asking it
    to forget a pair it never knew is the state the caller wanted."""
    from database.models import delete_pair_feedback
    assert delete_pair_feedback(99, 100, vocal_section=0, inst_section=0,
                                db_path=db) == 0
