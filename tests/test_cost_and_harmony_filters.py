"""B.3 / B.4 — select on the costs and the harmony, not just read them.

The row already displayed all of this: the five effort components (collapsed
into one of three words), the measured harmonic shift and its confidence, the
bass-clash warning, and the collision score. None of it could be filtered on.
The only cost control was a single "Free builds" toggle hardcoded to an effort
cap of 0.25, which can express neither "no transpose, any stretch" nor "any
stretch, no transpose" — different days in the studio with different fixes.
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


def _side(sid):
    return {"song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
            "key": "A", "mode": "minor", "camelot": "8A",
            "loudness_rms": 0.05, "energy": 0.5}


def _library(db_path, specs):
    """One candidate per spec dict of extra score fields, tagged by title."""
    from database.models import upsert_candidate, upsert_song

    out = {}
    for n, spec in enumerate(specs):
        v = upsert_song(f"v{n}", "A", f"https://sc/v{n}", 200,
                        status="analysed", db_path=db_path)
        i = upsert_song(f"i{n}", "A", f"https://sc/i{n}", 200,
                        status="analysed", db_path=db_path)
        upsert_candidate(_side(v), _side(i), {
            "total": 0.8, "bpm_score": 0.9, "key_score": 0.9,
            "energy_score": 0.9, "timbre_score": 0.9, **spec,
        }, db_path=db_path)
        out[n] = (v, i)
    return out


# ── B.3: the two build costs, independently ──────────────────────────────────

def test_no_transpose_keeps_only_pairs_that_need_none(db_path):
    from database.models import get_candidates_enriched

    _library(db_path, [
        {"effort_pitch": 0.0, "effort_stretch": 0.9},   # v0: free pitch, big stretch
        {"effort_pitch": 0.8, "effort_stretch": 0.0},   # v1: wide pitch, free stretch
    ])
    rows = get_candidates_enriched(limit=50, max_pitch_cost=0.0, db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["effort_pitch"] == 0.0
    # The big-stretch pair survives, which is the point: a pitch cap must not
    # silently also cap the stretch.
    assert rows[0]["effort_stretch"] == pytest.approx(0.9)


def test_no_stretch_keeps_the_other_one(db_path):
    """The point of splitting them: the two chips select opposite pairs."""
    from database.models import get_candidates_enriched

    _library(db_path, [
        {"effort_pitch": 0.0, "effort_stretch": 0.9},
        {"effort_pitch": 0.8, "effort_stretch": 0.0},
    ])
    rows = get_candidates_enriched(limit=50, max_stretch_cost=0.0, db_path=db_path)
    assert len(rows) == 1 and rows[0]["effort_stretch"] == 0.0


def test_the_two_cost_caps_compose(db_path):
    from database.models import get_candidates_enriched

    _library(db_path, [
        {"effort_pitch": 0.0, "effort_stretch": 0.0},   # free both ways
        {"effort_pitch": 0.0, "effort_stretch": 0.9},
        {"effort_pitch": 0.8, "effort_stretch": 0.0},
    ])
    rows = get_candidates_enriched(limit=50, max_pitch_cost=0.0,
                                   max_stretch_cost=0.0, db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["effort_pitch"] == 0.0 and rows[0]["effort_stretch"] == 0.0


def test_an_unmeasured_cost_passes_rather_than_hiding_the_library(db_path):
    """NULL means the row predates the column. Treating it as expensive would
    empty the list for anyone who has not re-scored."""
    from database.models import get_candidates_enriched

    _library(db_path, [{}])   # no effort columns at all
    assert len(get_candidates_enriched(limit=50, max_pitch_cost=0.0,
                                       db_path=db_path)) == 1


# ── B.4: harmony and spectral room ───────────────────────────────────────────

def test_measured_harmony_excludes_the_unmeasured(db_path):
    """An unmeasured fit is not a confident one. NULL harmonic_confidence means
    the sections had no stored chroma, so nothing was cross-correlated — those
    rows are ranked on the Camelot wheel and must not pass a filter asking for
    measured agreement."""
    from database.models import get_candidates_enriched

    _library(db_path, [
        {"harmonic_confidence": 0.9, "harmonic_shift": 0},
        {"harmonic_confidence": 0.2, "harmonic_shift": 3},
        {},                                    # never measured
    ])
    rows = get_candidates_enriched(limit=50, min_harmonic_confidence=0.5,
                                   db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["harmonic_confidence"] == pytest.approx(0.9)


def test_bass_clash_can_be_excluded(db_path):
    from database.models import get_candidates_enriched

    _library(db_path, [
        {"bass_clash": 1}, {"bass_clash": 0}, {},
    ])
    rows = get_candidates_enriched(limit=50, exclude_bass_clash=True,
                                   db_path=db_path)
    # A NULL bass_clash is "not measured", which COALESCEs to no clash — the
    # same convention the rest of the filters use for an absent measurement.
    assert len(rows) == 2
    assert all(not r["bass_clash"] for r in rows)


def test_min_collision_keeps_beds_that_leave_room(db_path):
    """The heaviest term on the vocal path, and until B.4 unselectable."""
    from database.models import get_candidates_enriched

    _library(db_path, [
        {"collision_score": 0.85}, {"collision_score": 0.20},
    ])
    rows = get_candidates_enriched(limit=50, min_collision=0.5, db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["score_collision"] == pytest.approx(0.85)


def test_the_new_filters_compose_with_the_old_ones(db_path):
    from database.models import get_candidates_enriched

    _library(db_path, [
        {"effort_pitch": 0.0, "bass_clash": 0, "collision_score": 0.9},
        {"effort_pitch": 0.0, "bass_clash": 1, "collision_score": 0.9},
        {"effort_pitch": 0.9, "bass_clash": 0, "collision_score": 0.9},
        {"effort_pitch": 0.0, "bass_clash": 0, "collision_score": 0.1},
    ])
    rows = get_candidates_enriched(
        limit=50, max_pitch_cost=0.0, exclude_bass_clash=True,
        min_collision=0.5, db_path=db_path)
    assert len(rows) == 1


# ── Stem quality reaches the row (B.2) ───────────────────────────────────────

def test_stem_quality_is_joined_onto_the_candidate_row(db_path):
    """Measured since Phase D and read only by the stem_quality_min cutoff, so a
    0.36 acapella and a 0.95 one looked identical in the list."""
    from database.models import (
        get_candidates_enriched, update_stem_quality, upsert_stem,
    )

    pairs = _library(db_path, [{}])
    v, i = pairs[0]
    upsert_stem(v, "vocals", "/tmp/v.flac", db_path=db_path)
    upsert_stem(i, "instrumental", "/tmp/i.flac", db_path=db_path)
    update_stem_quality(v, "vocals", {
        "quality": 0.42, "bleed": 0.7, "hf_loss": 0.1, "noise_floor": 0.2,
    }, db_path=db_path)

    row = get_candidates_enriched(limit=5, db_path=db_path)[0]
    assert row["vocal_stem_quality"] == pytest.approx(0.42)
    assert row["vocal_stem_bleed"] == pytest.approx(0.7)
    # Never measured on the bed side — absent, not zero.
    assert row["inst_stem_quality"] is None


# ── The query-string contract ────────────────────────────────────────────────

def test_out_of_range_values_are_refused():
    from fastapi import HTTPException
    from api.routes.mashups import list_candidates

    for kwargs in ({"max_pitch_cost": 1.5}, {"max_stretch_cost": -0.1},
                   {"min_harmonic_confidence": 2.0}, {"min_collision": -1.0}):
        with pytest.raises(HTTPException):
            list_candidates(**kwargs)


def test_the_export_can_express_every_new_filter():
    from api.routes.mashups import BatchSessionRequest

    fields = set(BatchSessionRequest.model_fields)
    for name in ("max_pitch_cost", "max_stretch_cost",
                 "min_harmonic_confidence", "exclude_bass_clash",
                 "min_collision"):
        assert name in fields, f"export request cannot express {name}"
