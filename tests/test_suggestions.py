"""Tests for the mashup suggestion engine: sections schema, metadata
enrichment, section labelling heuristics, and the actionable plan builder.

Pure python + sqlite (numpy/scipy only for the novelty-boundary test) —
no librosa, demucs, or network needed.
"""
import json
import os
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


@pytest.fixture()
def seeded(db_path):
    """Two analysed songs with features, sections, and one scored candidate."""
    from database.models import (
        init_db, replace_sections, upsert_candidate, upsert_features,
        upsert_song,
    )
    init_db(db_path)
    s1 = upsert_song(
        "Closer", "Halsey", "https://sc/1", 240, "Pop",
        upload_date="20160729", likes=5000, plays=200000,
        tags=json.dumps(["pop", "edm"]), release_year=2016,
        status="analysed", db_path=db_path,
    )
    s2 = upsert_song(
        "Animals", "Martin Garrix", "https://sc/2", 300, "Big Room",
        upload_date="20130617", likes=9000, plays=900000,
        status="analysed", db_path=db_path,
    )
    upsert_features(s1, "vocals", {
        "bpm": 95.0, "key": "G#", "mode": "minor", "camelot": "1A",
        "loudness_rms": 0.1, "energy": 0.5, "mfcc": [1.0] * 13,
    }, db_path=db_path)
    upsert_features(s2, "instrumental", {
        "bpm": 190.0, "key": "F#", "mode": "minor", "camelot": "11A",
        "loudness_rms": 0.12, "energy": 0.6, "mfcc": [1.0] * 13,
    }, db_path=db_path)
    replace_sections(s1, [
        {"start_sec": 0, "end_sec": 20, "label": "intro",
         "energy": 0.3, "vocal_presence": 0.1, "repetition": 1, "confidence": 0.7},
        {"start_sec": 20, "end_sec": 60, "label": "verse",
         "energy": 0.6, "vocal_presence": 0.7, "repetition": 2, "confidence": 0.5},
        {"start_sec": 60, "end_sec": 90, "label": "chorus",
         "energy": 0.9, "vocal_presence": 0.9, "repetition": 3, "confidence": 0.8},
    ], db_path=db_path)
    replace_sections(s2, [
        {"start_sec": 0, "end_sec": 30, "label": "intro",
         "energy": 0.4, "vocal_presence": None, "repetition": 1, "confidence": 0.7},
        {"start_sec": 30, "end_sec": 90, "label": "drop",
         "energy": 0.95, "vocal_presence": None, "repetition": 2, "confidence": 0.6},
    ], db_path=db_path)
    upsert_candidate(
        {"song_id": s1, "title": "Closer", "artist": "Halsey", "bpm": 95.0,
         "key": "G#", "mode": "minor", "camelot": "1A",
         "loudness_rms": 0.1, "energy": 0.5},
        {"song_id": s2, "title": "Animals", "artist": "Martin Garrix",
         "bpm": 190.0, "key": "F#", "mode": "minor", "camelot": "11A",
         "loudness_rms": 0.12, "energy": 0.6},
        {"total": 0.82, "bpm_score": 1.0, "key_score": 0.55,
         "energy_score": 0.9, "timbre_score": 0.8},
        db_path=db_path,
    )
    return db_path, s1, s2


def test_release_year_backfill_from_upload_date(seeded):
    db_path, _, s2 = seeded
    from database.models import get_conn, get_song
    get_conn(db_path).close()  # opening runs migrations + backfill
    assert get_song(s2, db_path=db_path)["release_year"] == 2013


def test_sections_roundtrip(seeded):
    db_path, s1, _ = seeded
    from database.models import get_sections, replace_sections
    sections = get_sections(s1, db_path=db_path)
    assert [s["label"] for s in sections] == ["intro", "verse", "chorus"]
    assert sections[2]["vocal_presence"] == 0.9
    # replace wipes old rows
    replace_sections(s1, sections[:1], db_path=db_path)
    assert len(get_sections(s1, db_path=db_path)) == 1


def test_candidates_enriched_metadata(seeded):
    db_path, _, _ = seeded
    from database.models import get_candidates_enriched
    rows = get_candidates_enriched(db_path=db_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["vocal_genre"] == "Pop" and r["vocal_year"] == 2016
    assert r["inst_genre"] == "Big Room"
    assert r["inst_popularity"] > r["vocal_popularity"]
    assert r["vocal_section_count"] == 3 and r["inst_section_count"] == 2


def test_label_segments_heuristics():
    from analysis.structure import label_segments
    segs = [
        {"energy": 0.3, "vocal_presence": 0.05, "repetition": 1},
        {"energy": 0.55, "vocal_presence": 0.6, "repetition": 2},
        {"energy": 0.85, "vocal_presence": 0.8, "repetition": 3},
        {"energy": 0.9, "vocal_presence": 0.1, "repetition": 2},
        {"energy": 0.35, "vocal_presence": 0.1, "repetition": 1},
    ]
    label_segments(segs, has_vocals=True)
    assert [s["label"] for s in segs] == \
        ["intro", "verse", "chorus", "drop", "outro"]


def test_label_segments_guarantees_a_chorus_with_vocals():
    from analysis.structure import label_segments
    segs = [
        {"energy": 0.5, "vocal_presence": 0.45, "repetition": 2},
        {"energy": 0.5, "vocal_presence": 0.45, "repetition": 2},
    ]
    label_segments(segs, has_vocals=True)
    assert any(s["label"] == "chorus" for s in segs)


def test_effective_inst_bpm_handles_doubletime():
    from matcher.match import effective_bpm
    assert effective_bpm(95.0, 190.0) == 95.0
    assert effective_bpm(150.0, 75.0) == 150.0
    assert effective_bpm(120.0, 122.0) == 122.0


# ── T1.2 semitone shift, computed from the Camelot pair ──────────────────────
# The shift must agree with camelot_score by construction: any pair camelot_score
# rates as compatible must not be handed a destructive transposition.

def test_relative_minor_bed_needs_no_transposition():
    """C major vocal over A minor bed → 0. They share a scale (8B/8A), which is
    why camelot_score rates the pair 0.75. Shifting +3 would drag the bed to C
    minor and clash with the vocal's major third."""
    from matcher.match import compute_semitone_shift
    assert compute_semitone_shift("8B", "8A") == 0


def test_relative_major_bed_needs_no_transposition():
    """The inverse direction: A minor vocal over C major bed → 0."""
    from matcher.match import compute_semitone_shift
    assert compute_semitone_shift("8A", "8B") == 0


def test_identical_key_needs_no_transposition():
    from matcher.match import compute_semitone_shift
    assert compute_semitone_shift("8B", "8B") == 0
    assert compute_semitone_shift("11A", "11A") == 0


def test_unrelated_keys_return_minimal_signed_shift():
    """One Camelot hour = a perfect fifth = 7 semitones, folded into [-6, +6]."""
    from matcher.match import compute_semitone_shift
    assert compute_semitone_shift("8B", "9B") == 5    # G major bed → C major
    assert compute_semitone_shift("8B", "7B") == -5   # F major bed → C major
    assert compute_semitone_shift("8B", "10B") == -2  # D major bed → C major
    assert compute_semitone_shift("1A", "11A") == 2   # F# minor bed → G# minor


def test_semitone_shift_never_exceeds_a_tritone():
    from matcher.match import compute_semitone_shift
    for v in range(1, 13):
        for i in range(1, 13):
            for lv in "AB":
                for li in "AB":
                    shift = compute_semitone_shift(f"{v}{lv}", f"{i}{li}")
                    assert shift is not None and -6 <= shift <= 6


def test_semitone_shift_unknown_key_returns_none():
    from matcher.match import compute_semitone_shift
    assert compute_semitone_shift("?", "8A") is None
    assert compute_semitone_shift("8B", "?") is None
    assert compute_semitone_shift("", "") is None


def test_build_mashup_plan(seeded):
    db_path, s1, s2 = seeded
    from matcher.plan import build_mashup_plan
    plan = build_mashup_plan(s1, s2, db_path=db_path)
    assert plan["semitone_shift"] == 2          # G# over F#
    assert plan["stretch_factor"] == 1.0        # 190 BPM read as halftime 95
    assert plan["target_bpm"] == 95.0
    # chorus is paired before verse, and intro sections are never suggested
    assert plan["pairings"][0]["vocal_label"] == "chorus"
    assert plan["pairings"][0]["inst_label"] == "drop"
    assert all(p["vocal_label"] != "intro" for p in plan["pairings"])
    assert len(plan["steps"]) >= 4


def test_build_mashup_plan_missing_song(db_path):
    from database.models import init_db
    init_db(db_path)
    from matcher.plan import build_mashup_plan
    assert build_mashup_plan(1, 2, db_path=db_path) is None


def test_novelty_boundaries_finds_synthetic_transitions():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from analysis.structure import _novelty_boundaries
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (25, 1))
    b = rng.normal(5, 1, (25, 1))
    X = np.hstack([
        a + rng.normal(0, 0.1, (25, 64)),
        b + rng.normal(0, 0.1, (25, 64)),
        a + rng.normal(0, 0.1, (25, 64)),
    ])
    bounds = _novelty_boundaries(X, min_beats=16, max_sections=14)
    assert any(50 <= x <= 78 for x in bounds)
    assert any(115 <= x <= 142 for x in bounds)


def test_ingest_normalise_tags_and_year():
    from ingest.soundcloud import _normalise
    info = {
        "title": "Track", "uploader": "DJ", "webpage_url": "https://sc/x",
        "duration": 200, "upload_date": "20190501",
        "tags": ["future bass", "remix"], "genre": "Future Bass",
        "like_count": 10, "view_count": 100,
    }
    t = _normalise(info)
    assert json.loads(t["tags"]) == ["future bass", "remix"]
    assert t["release_year"] == 2019
