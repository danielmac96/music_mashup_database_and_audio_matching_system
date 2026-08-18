"""E.1 / E.2 — the plan carries the two measurements that explain a pair.

* **Band occupancy.** collision_score is the heaviest term on the vocal path and
  it is one number: it says "these two fight" without saying WHERE, which is the
  only part you can act on. The 8-band vectors behind it have been measured
  since Phase D and drawn nowhere.
* **Per-section key.** A track has one key only in the sense that an average has
  one value. Real records modulate, so the chorus is frequently not the key the
  whole-track mean reports — the reason a pair that looks compatible on the
  Camelot codes still fights. Stored per section since Phase E, shown nowhere.
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


_SEQ = iter(range(1000))


def _seed(db_path, *, track_camelot, section_camelot, bands):
    """A song whose CHORUS is in a different key from its whole-track estimate.

    The source_url is unique per call: upsert_song keys on it, so two seeds
    sharing one would be the same song and the second would overwrite the first.
    """
    from database.models import replace_sections, upsert_features, upsert_song

    n = next(_SEQ)
    sid = upsert_song(f"s{n}", "A", f"https://sc/{n}", 200,
                      status="analysed", db_path=db_path)
    for stem in ("full", "vocals", "instrumental"):
        upsert_features(sid, stem, {
            "bpm": 128.0, "key": "A", "mode": "minor",
            "camelot": track_camelot, "loudness_rms": 0.05, "energy": 0.5,
            "band_energy": bands,
        }, db_path=db_path)
    replace_sections(sid, [
        {"start_sec": 0.0, "end_sec": 16.0, "label": "intro", "energy": 0.2,
         "vocal_presence": 0.0, "repetition": 1, "confidence": 0.8},
        {"start_sec": 16.0, "end_sec": 48.0, "label": "chorus", "energy": 0.9,
         "vocal_presence": 0.9, "repetition": 2, "confidence": 0.9,
         "key": "C", "mode": "major", "camelot": section_camelot,
         "key_confidence": 0.08},
    ], db_path=db_path)
    return sid


# Vocal sits in the mids, bed owns the low end — complementary, little overlap.
LOW_HEAVY = [0.30, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01, 0.01]
MID_HEAVY = [0.01, 0.02, 0.07, 0.35, 0.35, 0.12, 0.05, 0.03]


def test_the_plan_carries_both_sides_band_occupancy(db_path):
    from matcher.plan import build_mashup_plan
    from analysis.quality import N_BANDS

    v = _seed(db_path, track_camelot="8A", section_camelot="8A", bands=MID_HEAVY)
    i = _seed(db_path, track_camelot="8A", section_camelot="8A", bands=LOW_HEAVY)

    plan = build_mashup_plan(v, i, db_path=db_path)
    assert plan["vocal"]["band_energy"] == MID_HEAVY
    assert plan["inst"]["band_energy"] == LOW_HEAVY
    # Edges label the bars, so there must be one more edge than band.
    assert len(plan["band_edges"]) == N_BANDS + 1


def test_the_band_edges_come_from_the_module_that_measures_them(db_path):
    """A drawn axis that drifts from the vector it labels is worse than no
    axis: it says the clash is somewhere it is not."""
    from analysis.quality import BAND_EDGES
    from matcher.plan import BAND_EDGES as PLAN_EDGES

    assert tuple(PLAN_EDGES) == tuple(BAND_EDGES)


def test_a_stem_analysed_before_phase_d_reports_no_bands(db_path):
    """Absent, not zeros — an unmeasured spectrum must not draw as an empty one."""
    from matcher.plan import build_mashup_plan

    v = _seed(db_path, track_camelot="8A", section_camelot="8A", bands=None)
    i = _seed(db_path, track_camelot="9A", section_camelot="9A", bands=None)
    plan = build_mashup_plan(v, i, db_path=db_path)
    assert plan["vocal"]["band_energy"] is None


def test_the_plan_reports_the_sections_own_key(db_path):
    from matcher.plan import build_mashup_plan

    v = _seed(db_path, track_camelot="8A", section_camelot="3B", bands=MID_HEAVY)
    i = _seed(db_path, track_camelot="8A", section_camelot="8A", bands=LOW_HEAVY)

    plan = build_mashup_plan(v, i, db_path=db_path)
    keys = plan["section_keys"]
    assert keys["vocal"]["camelot"] == "3B"
    assert keys["inst"]["camelot"] == "8A"
    # The interesting bit: this chorus is NOT in the track's key.
    assert keys["vocal"]["differs_from_track"] is True
    assert keys["inst"]["differs_from_track"] is False
    assert keys["vocal"]["key_confidence"] == pytest.approx(0.08)


def test_section_keys_are_for_the_pinned_pair_not_a_re_chosen_one(db_path):
    """Same rule as everything else in A.1: describe the moment the row named."""
    from matcher.plan import build_mashup_plan

    v = _seed(db_path, track_camelot="8A", section_camelot="3B", bands=MID_HEAVY)
    i = _seed(db_path, track_camelot="8A", section_camelot="8A", bands=LOW_HEAVY)

    # Section 0 is the intro, which has no stored key at all.
    plan = build_mashup_plan(v, i, db_path=db_path,
                             vocal_section_idx=0, inst_section_idx=1)
    assert plan["section_keys"]["vocal"]["camelot"] is None
    assert plan["section_keys"]["vocal"]["label"] == "intro"


def test_a_plan_with_no_sections_has_no_section_keys(db_path):
    from database.models import upsert_features, upsert_song
    from matcher.plan import build_mashup_plan

    ids = []
    for n in range(2):
        sid = upsert_song(f"bare{n}", "A", f"https://sc/bare{n}", 200,
                          status="analysed", db_path=db_path)
        upsert_features(sid, "full", {
            "bpm": 128.0, "key": "A", "mode": "minor", "camelot": "8A",
            "loudness_rms": 0.05, "energy": 0.5,
        }, db_path=db_path)
        ids.append(sid)

    plan = build_mashup_plan(ids[0], ids[1], db_path=db_path)
    assert plan["section_keys"] is None
    assert plan["pairings"] == []
