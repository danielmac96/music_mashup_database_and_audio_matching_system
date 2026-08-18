"""D.3 — reach the other ideas about the same two records.

The scorer emits a row per section pairing and the list shows at most one, or
the same two records occupy three rows with what reads as one suggestion
repeated. The others stayed scored, in the table, and unreachable: "chorus over
drop" and "verse over breakdown" are genuinely different ideas, and the only way
to the second was to seed on one track and re-filter.
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


SECTIONS = [
    ("intro", 0.0, 16.0, 0.1),
    ("chorus", 16.0, 48.0, 0.9),
    ("verse", 48.0, 80.0, 0.5),
    ("drop", 80.0, 112.0, 0.95),
    ("breakdown", 112.0, 140.0, 0.4),
]


def _song(db_path, key):
    from database.models import replace_sections, upsert_song

    sid = upsert_song(f"s{key}", "A", f"https://sc/{key}", 200,
                      status="analysed", db_path=db_path)
    replace_sections(sid, [
        {"start_sec": a, "end_sec": b, "label": lab, "energy": e,
         "vocal_presence": 0.9, "repetition": 1, "confidence": 0.9}
        for lab, a, b, e in SECTIONS
    ], db_path=db_path)
    return sid


def _pair(db_path, v, i, v_idx, i_idx, total):
    from database.models import upsert_candidate
    side = lambda sid: {                                         # noqa: E731
        "song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
        "key": "A", "mode": "minor", "camelot": "8A",
        "loudness_rms": 0.05, "energy": 0.5,
    }
    upsert_candidate(side(v), side(i), {
        "total": total, "bpm_score": 0.9, "key_score": 0.9,
        "energy_score": 0.9, "timbre_score": 0.9,
    }, section_pair={"vocal_section_idx": v_idx, "inst_section_idx": i_idx,
                     "score_section": 0.7}, db_path=db_path)


def _three_takes(db_path):
    """One song pair, three scored section pairings."""
    v, i = _song(db_path, "v"), _song(db_path, "i")
    _pair(db_path, v, i, 1, 3, 0.90)   # chorus over drop
    _pair(db_path, v, i, 2, 4, 0.80)   # verse over breakdown
    _pair(db_path, v, i, 1, 1, 0.70)   # chorus over chorus
    return v, i


def test_the_list_still_shows_one_row_per_song_pair(db_path):
    """The reason the others are hidden — three rows of the same two records
    read as one suggestion repeated."""
    from database.models import get_candidates_enriched

    _three_takes(db_path)
    rows = get_candidates_enriched(limit=50, db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["score_total"] == pytest.approx(0.90)


def test_the_other_takes_are_reachable(db_path):
    from database.models import get_candidates_enriched

    v, i = _three_takes(db_path)
    takes = get_candidates_enriched(
        limit=50, vocal_song_id=v, inst_song_id=i,
        max_per_song=0, max_per_song_pair=0, db_path=db_path)
    assert len(takes) == 3
    # Best first, and each names a different pair of sections.
    assert [t["score_total"] for t in takes] == [0.90, 0.80, 0.70]
    assert len({(t["vocal_section_idx"], t["inst_section_idx"])
                for t in takes}) == 3
    labels = {(t["vocal_section_label"], t["inst_section_label"])
              for t in takes}
    assert ("chorus", "drop") in labels
    assert ("verse", "breakdown") in labels


# ── Filtering by the shape of the move ───────────────────────────────────────

def test_the_shape_filter_selects_chorus_over_drop(db_path):
    from database.models import get_candidates_enriched

    v, i = _song(db_path, "v"), _song(db_path, "i")
    v2, i2 = _song(db_path, "v2"), _song(db_path, "i2")
    _pair(db_path, v, i, 1, 3, 0.90)     # chorus over drop
    _pair(db_path, v2, i2, 2, 4, 0.85)   # verse over breakdown

    rows = get_candidates_enriched(limit=50, section_pair="chorus>drop",
                                   db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["vocal_section_label"] == "chorus"
    assert rows[0]["inst_section_label"] == "drop"


def test_half_a_shape_leaves_the_other_side_free(db_path):
    """'chorus>' means any bed section under a chorus."""
    from database.models import get_candidates_enriched

    v, i = _song(db_path, "v"), _song(db_path, "i")
    v2, i2 = _song(db_path, "v2"), _song(db_path, "i2")
    v3, i3 = _song(db_path, "v3"), _song(db_path, "i3")
    _pair(db_path, v, i, 1, 3, 0.90)     # chorus over drop
    _pair(db_path, v2, i2, 1, 4, 0.85)   # chorus over breakdown
    _pair(db_path, v3, i3, 2, 3, 0.80)   # verse over drop

    over_chorus = get_candidates_enriched(limit=50, section_pair="chorus>",
                                          db_path=db_path)
    assert len(over_chorus) == 2
    under_drop = get_candidates_enriched(limit=50, section_pair=">drop",
                                         db_path=db_path)
    assert len(under_drop) == 2


def test_a_shape_nothing_matches_is_empty_not_an_error(db_path):
    from database.models import get_candidates_enriched

    v, i = _song(db_path, "v"), _song(db_path, "i")
    _pair(db_path, v, i, 1, 3, 0.9)
    assert get_candidates_enriched(limit=50, section_pair="outro>intro",
                                   db_path=db_path) == []


def test_the_chip_offers_only_shapes_this_library_contains(db_path):
    from database.models import candidate_filter_options

    v, i = _song(db_path, "v"), _song(db_path, "i")
    v2, i2 = _song(db_path, "v2"), _song(db_path, "i2")
    _pair(db_path, v, i, 1, 3, 0.90)     # chorus over drop
    _pair(db_path, v2, i2, 2, 4, 0.85)   # verse over breakdown

    shapes = {p["value"] for p in
              candidate_filter_options(db_path=db_path)["section_pairs"]}
    assert shapes == {"chorus>drop", "verse>breakdown"}


def test_the_shape_filter_survives_the_export(db_path):
    """Same rule as A.3: a control that changes which rows are on screen has to
    be expressible on the export request."""
    from api.routes.mashups import BatchSessionRequest
    assert "section_pair" in BatchSessionRequest.model_fields
