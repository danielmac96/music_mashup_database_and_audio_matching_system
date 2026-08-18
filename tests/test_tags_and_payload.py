"""E.4 — SoundCloud tags, and not shipping vectors nobody reads.

`songs.tags` was ingested and never used. SoundCloud's `genre` is one free-text
string and often blank, while the tags carry the real description, so both the
genre filter and Phase F's cross-genre contrast term were blind to a large part
of a real library.

Along the way this file pins a bug the tag work surfaced: get_all_features never
selected `genre` or `release_year`, so `_genre_distance` and `_era_distance` saw
None on both sides and returned the neutral 0.5 for EVERY pair. Two of the three
contrast columns were constants in every training vector — the exact "a column
with no signal is only a chance to overfit noise" failure matcher/features.py
warns about two paragraphs above them.
"""
import json
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


def _song(db_path, key, *, genre="", year=0, tags=None, stem="vocals"):
    from database.models import get_conn, upsert_features, upsert_song

    sid = upsert_song(f"s{key}", "A", f"https://sc/{key}", 200, genre,
                      status="analysed", db_path=db_path)
    conn = get_conn(db_path)
    conn.execute("UPDATE songs SET release_year=?, tags=? WHERE id=?",
                 (year, json.dumps(tags) if tags else None, sid))
    conn.commit()
    conn.close()
    upsert_features(sid, stem, {
        "bpm": 128.0, "key": "A", "mode": "minor", "camelot": "8A",
        "loudness_rms": 0.05, "energy": 0.5, "mfcc": [1.0] * 13,
        "spectral_centroid": 2000.0, "spectral_rolloff": 4000.0,
        "zero_crossing_rate": 0.05,
    }, db_path=db_path)
    return sid


# ── The contrast columns were constants ──────────────────────────────────────

def test_the_feature_dict_carries_what_the_contrast_terms_read(db_path):
    from database.models import get_all_features

    _song(db_path, "a", genre="Techno", year=2022, tags=["rave"])
    feat = get_all_features(stem_type="vocals", db_path=db_path)[0]
    for field in ("genre", "release_year", "tags"):
        assert field in feat, f"{field} never reaches pair_features"


def test_a_maximally_cross_genre_cross_era_pair_no_longer_reads_neutral(db_path):
    """Techno/2022 over Soul/1975 is as far apart as this library gets. It
    reported 0.5 — 'we have no idea' — on both axes."""
    from database.models import get_all_features
    from matcher.features import surprise_terms

    _song(db_path, "a", genre="Techno", year=2022, stem="vocals")
    _song(db_path, "b", genre="Soul", year=1975, stem="instrumental")
    v = get_all_features(stem_type="vocals", db_path=db_path)[0]
    i = get_all_features(stem_type="instrumental", db_path=db_path)[0]

    terms = surprise_terms(v, i)
    assert terms["surprise_genre"] == pytest.approx(1.0)
    assert terms["surprise_era"] == pytest.approx(1.0)


def test_a_same_genre_same_era_pair_reads_zero(db_path):
    from database.models import get_all_features
    from matcher.features import surprise_terms

    _song(db_path, "a", genre="House", year=2020, stem="vocals")
    _song(db_path, "b", genre="House", year=2020, stem="instrumental")
    v = get_all_features(stem_type="vocals", db_path=db_path)[0]
    i = get_all_features(stem_type="instrumental", db_path=db_path)[0]

    terms = surprise_terms(v, i)
    assert terms["surprise_genre"] == pytest.approx(0.0)
    assert terms["surprise_era"] == pytest.approx(0.0)


# ── Tags as a genre fallback ─────────────────────────────────────────────────

def test_tags_stand_in_for_a_missing_genre(db_path):
    from matcher.features import _genre_distance

    # No genre on either side, but the tags say they are the same world.
    assert _genre_distance("", "", ["deep house"], ["deep house"]) \
        == pytest.approx(0.0)
    assert _genre_distance("", "", ["techno"], ["soul"]) == pytest.approx(1.0)


def test_tags_never_move_a_genre_that_is_already_set(db_path):
    """Widening a known genre with tags would change a value the model trained
    on. Replacing a 'we don't know' only ever adds information."""
    from matcher.features import _genre_distance

    with_tags = _genre_distance("Techno", "Soul", ["soul", "funk"], ["techno"])
    without = _genre_distance("Techno", "Soul")
    assert with_tags == without == pytest.approx(1.0)


def test_unknown_on_both_sides_is_still_neutral(db_path):
    from matcher.features import _genre_distance
    assert _genre_distance("", "") == pytest.approx(0.5)
    assert _genre_distance("", "", [], []) == pytest.approx(0.5)


def test_malformed_tags_do_not_raise(db_path):
    from matcher.features import _tag_tokens
    assert _tag_tokens("not json") == {"not", "json"}
    assert _tag_tokens('{"a": 1}') == set()
    assert _tag_tokens(None) == set()


# ── Tags in the filter ───────────────────────────────────────────────────────

def _pair(db_path, v, i):
    from database.models import upsert_candidate
    side = lambda sid: {                                         # noqa: E731
        "song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
        "key": "A", "mode": "minor", "camelot": "8A",
        "loudness_rms": 0.05, "energy": 0.5,
    }
    upsert_candidate(side(v), side(i), {
        "total": 0.8, "bpm_score": 0.9, "key_score": 0.9,
        "energy_score": 0.9, "timbre_score": 0.9,
    }, db_path=db_path)


def test_the_genre_filter_matches_tags_too(db_path):
    """An upload with a blank genre and a 'Jersey Club' tag was unreachable."""
    from database.models import get_candidates_enriched

    v = _song(db_path, "a", genre="", tags=["Jersey Club"], stem="vocals")
    i = _song(db_path, "b", genre="House", stem="instrumental")
    v2 = _song(db_path, "c", genre="Ambient", stem="vocals")
    i2 = _song(db_path, "d", genre="Ambient", stem="instrumental")
    _pair(db_path, v, i)
    _pair(db_path, v2, i2)

    rows = get_candidates_enriched(limit=50, genre="Jersey", db_path=db_path)
    assert len(rows) == 1 and rows[0]["vocal_song_id"] == v


def test_the_chip_offers_tags_that_describe_more_than_one_track(db_path):
    from database.models import candidate_filter_options

    v = _song(db_path, "a", genre="", tags=["Jersey Club", "oneoff"],
              stem="vocals")
    i = _song(db_path, "b", genre="", tags=["Jersey Club"], stem="instrumental")
    _pair(db_path, v, i)

    offered = [g["genre"] for g in
               candidate_filter_options(db_path=db_path)["genres"]]
    assert "Jersey Club" in offered
    assert "oneoff" not in offered, "a one-track tag is noise in a cycling chip"


def test_a_tag_duplicating_a_genre_is_not_offered_twice(db_path):
    from database.models import candidate_filter_options

    v = _song(db_path, "a", genre="House", tags=["House"], stem="vocals")
    i = _song(db_path, "b", genre="House", tags=["House"], stem="instrumental")
    _pair(db_path, v, i)

    offered = [g["genre"] for g in
               candidate_filter_options(db_path=db_path)["genres"]]
    assert offered.count("House") == 1


# ── The section payload ──────────────────────────────────────────────────────

def test_sections_do_not_ship_chroma_over_http_by_default(db_path):
    """Four 12-float vectors per section, read by matcher/harmony.py and by no
    screen at all — the bulk of the response on every Studio lane add."""
    from database.models import get_sections, replace_sections

    sid = _song(db_path, "a")
    replace_sections(sid, [{
        "start_sec": 0.0, "end_sec": 30.0, "label": "chorus", "energy": 0.9,
        "vocal_presence": 0.9, "repetition": 2, "confidence": 0.9,
        "chroma": [0.1] * 12, "chroma_vocal": [0.2] * 12,
        "chroma_bed": [0.3] * 12, "bass_chroma": [0.4] * 12,
    }], db_path=db_path)

    lean = get_sections(sid, db_path=db_path, include_chroma=False)[0]
    for key in ("chroma", "chroma_vocal", "chroma_bed", "bass_chroma"):
        assert key not in lean
    # Still there for the matcher, which is the only caller that reads them.
    full = get_sections(sid, db_path=db_path)[0]
    assert full["chroma_vocal"] == [0.2] * 12
    # Everything a screen actually draws survives the trim.
    assert lean["label"] == "chorus" and lean["end_sec"] == 30.0
