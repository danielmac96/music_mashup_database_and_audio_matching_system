"""Scoring on allin1's sections (readme §5.7): pairs hang off the vocal's
phrase, a drifting bed is dropped by the timing gate, a track allin1 never cut
is scored on its Essentia fallback sections (tagged essentia_fallback), and
every row and every rating records the engine that made it."""
import importlib
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BPM = 124.0
BAR = 4 * 60.0 / BPM
PLAN_V = [("intro", 8, False), ("verse", 16, True), ("chorus", 16, True), ("outro", 8, False)]
PLAN_B = [("intro", 8, False), ("drop", 16, False), ("breakdown", 8, False),
          ("drop", 16, False), ("outro", 8, False)]
CHROMA = [1.0, 0.05, 0.4, 0.4, 0.05, 0.4, 0.05, 0.8, 0.4, 0.05, 0.4, 0.05]


def _secs(plan, role, wobble=0.0):
    out, t = [], 0.0
    for i, (label, bars, sung) in enumerate(plan):
        dbs = []
        for k in range(bars):
            dbs.append(round(t, 4))
            t += BAR * (1 + (wobble * math.sin(k / 2.0) if wobble else 0.0))
        sec = {"section_index": i, "start_sec": dbs[0], "end_sec": round(t, 4),
               "label": label, "energy": 0.9 if label in ("chorus", "drop") else 0.5,
               "repetition": 2, "confidence": 0.9, "bpm": BPM, "bar_count": bars,
               "vocal_presence": 0.85 if sung else 0.05, "downbeats": dbs}
        if role == "vocal" and sung:
            sec["chroma_vocal"] = CHROMA
        if role == "bed" and label in ("drop", "breakdown"):
            sec["chroma_bed"] = CHROMA
        out.append(sec)
    return out


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "dual.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    from database import alignment

    ids = {}
    for n, (name, role) in enumerate((("V", "vocal"), ("B_steady", "bed"),
                                      ("B_drift", "bed"), ("B_essentia_only", "bed"))):
        sid = m.upsert_song(name, f"Artist {n}", f"u://dual/{n}", 200, "House",
                            status="analysed")
        ids[name] = sid
        mfcc = [150.0] + [((n * (k + 3)) % 11) - 5.0 for k in range(12)]
        for stem in ("full", "vocals", "instrumental"):
            m.upsert_features(sid, stem, {
                "bpm": BPM, "key": "A", "mode": "minor", "camelot": "8A",
                "loudness_rms": 0.1, "energy": 0.5, "bpm_confidence": 0.8,
                "key_confidence": 0.05, "mfcc": mfcc,
                "band_energy": [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]})
        plan = PLAN_V if role == "vocal" else PLAN_B
        # The Essentia cut: steady every-4th-beat bar lines for everyone.
        m.replace_sections(sid, _secs(plan, role))
        if name != "B_essentia_only":
            alignment.replace_allin1_sections(
                sid, "unified_4stem",
                _secs(plan, role, wobble=0.03 if name == "B_drift" else 0.0))
    # The vocal sings its verse from a beat before bar 1 of the verse.
    verse = _secs(PLAN_V, "vocal")[1]
    alignment.replace_vocal_phrases(ids["V"], "unified_4stem", "allin1", [
        {"onset_sec": verse["start_sec"] - 60 / BPM, "downbeat_sec": verse["start_sec"],
         "pickup_sec": -60 / BPM, "end_sec": verse["end_sec"], "bars": 16}])
    return m, ids, monkeypatch


def _rows(m):
    """The vocal-over-bed rows with a section pairing (a bed scored as a vocal
    has no sung section, so its rows are section-less, as ever)."""
    conn = m.get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM mashup_candidates WHERE combo_type='vocal_over_instrumental' "
            "AND vocal_section_idx IS NOT NULL")]
    finally:
        conn.close()


def test_allin1_scoring_gates_drift_and_falls_back(lib, monkeypatch):
    m, ids, mp = lib
    from matcher.match import score_all_pairs
    from api.workers import stages
    for name in ("do_stems", "do_analyze", "do_structure", "do_allin1"):
        monkeypatch.setattr(stages, name, lambda *a, **k: pytest.fail("reprocessed"))

    out = score_all_pairs(db_path=m.DB_PATH)
    assert out["_engine"] == "allin1"
    by_bed = {}
    for r in _rows(m):
        by_bed.setdefault(r["inst_song_id"], []).append(r)
    steady = by_bed[ids["B_steady"]]
    assert {r["engine"] for r in steady} == {"allin1"}
    assert all(r["vocal_section_idx"] >= m.ALLIN1_SECTION_BASE for r in steady)
    assert all(r["timing_ok"] == 1 and r["timing_err_max_ms"] < 25 for r in steady)
    verse = [r for r in steady if r["vocal_section_label"] == "verse"]
    assert verse and verse[0]["vocal_pickup_ms"] == pytest.approx(-60000 / BPM, abs=1)
    # The drifting bed fails the gate everywhere: no row at all.
    assert ids["B_drift"] not in by_bed
    assert out["_timing"]["gated"] >= 1
    # No allin1 cut: the Essentia sections, still timed, tagged as the fallback.
    fallback = by_bed[ids["B_essentia_only"]]
    assert {r["engine"] for r in fallback} == {"essentia_fallback"}
    assert all(r["timing_ok"] == 1 for r in fallback)


def test_ratings_record_the_engine_and_the_timing_terms(lib):
    m, ids, mp = lib
    from matcher.match import score_all_pairs
    score_all_pairs(db_path=m.DB_PATH)
    row = next(r for r in _rows(m) if r["engine"] == "allin1")
    m.upsert_pair_feedback(row["vocal_song_id"], row["inst_song_id"], rating=5,
                           vocal_section=row["vocal_section_idx"],
                           inst_section=row["inst_section_idx"])
    fb = m.get_pair_feedback()[0]
    assert fb["engine"] == "allin1" and fb["pipeline"] == "unified_4stem"
    import json
    snap = json.loads(fb["features_json"])
    assert snap["timing_ok"] == 1 and "score_timing" in snap

    # A judgement with no stored row still says which namespace it names.
    m.upsert_pair_feedback(ids["V"], ids["B_essentia_only"], rating=2,
                           vocal_section=1, inst_section=1)
    fb2 = next(f for f in m.get_pair_feedback() if f["inst_song_id"] == ids["B_essentia_only"]
               and f["vocal_section"] == 1)
    assert fb2["engine"] == "essentia"


def test_segments_rank_with_the_timing_gate_in_allin1_mode(lib):
    m, ids, mp = lib
    from matcher.match import score_all_pairs
    from matcher import segment_match
    score_all_pairs(db_path=m.DB_PATH)
    segs = m.list_segments("vocal")
    chorus = next(s for s in segs if s["song_id"] == ids["V"] and s["label"] == "chorus")
    assert chorus["segment_id"] >= m.ALLIN1_SEGMENT_ID_BASE
    assert m.get_segment(chorus["segment_id"])["section_index"] >= m.ALLIN1_SECTION_BASE
    out = segment_match.rank_partners(chorus["segment_id"], "vocal", db_path=m.DB_PATH)
    beds = {r["inst_song_id"] for r in out["rows"]}
    assert ids["B_steady"] in beds and ids["B_drift"] not in beds
    for r in out["rows"]:
        assert r["timing_ok"] in (1, None)


def test_the_listing_keeps_allin1_section_labels(lib):
    """The dock's listing joins labels from `sections`; an allin1 pairing names
    sections_allin1 rows, so the stored label must stand in."""
    m, ids, mp = lib
    from matcher.match import score_all_pairs
    score_all_pairs(db_path=m.DB_PATH)
    rows = m.get_candidates_enriched(vocal_song_id=ids["V"], inst_song_id=ids["B_steady"],
                                     max_per_song=0, limit=50)
    rows = [r for r in rows if r.get("engine") == "allin1"]
    assert rows and all(r["vocal_section_label"] in ("verse", "chorus") for r in rows)
    assert all(r["inst_section_label"] for r in rows)
