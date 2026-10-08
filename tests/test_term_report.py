"""Does each scored term agree with the ratings? (matcher/term_report.py)

A term earns a weight by separating loved pairs from rejected ones on this
library. These pin the statistics, the join from a judgement to its scored
row, and the three Phase 3 terms that are stored at weight 0 to be judged.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── statistics ───────────────────────────────────────────────────────────────

def test_auc_is_mann_whitney_with_ties_counting_half():
    from matcher.term_report import auc
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
    assert auc([], [0.1]) is None


def test_spearman_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    from matcher.term_report import spearman
    x = [1, 2, 2, 3, 5, 4, 1, 5]
    y = [0.1, 0.3, 0.2, 0.5, 0.9, 0.4, 0.3, 0.8]
    assert spearman(x, y) == pytest.approx(stats.spearmanr(x, y).correlation, abs=1e-4)
    assert spearman([1, 1, 1], [0.1, 0.2, 0.3]) is None


# ── the join from a judgement to its row ─────────────────────────────────────

def test_judgements_find_their_section_pair_or_their_song_pair():
    from matcher.term_report import judged_rows
    cands = [
        {"vocal_song_id": 1, "inst_song_id": 2, "vocal_section_idx": 0,
         "inst_section_idx": 1, "score_total": 0.6},
        {"vocal_song_id": 1, "inst_song_id": 2, "vocal_section_idx": 2,
         "inst_section_idx": 3, "score_total": 0.8},
    ]
    fb = [
        {"vocal_song_id": 1, "inst_song_id": 2, "vocal_section": 0, "inst_section": 1},
        {"vocal_song_id": 1, "inst_song_id": 2, "vocal_section": None, "inst_section": None},
        {"vocal_song_id": 1, "inst_song_id": 2, "vocal_section": 0, "inst_section": 1,
         "sections_stale": 1},
        {"vocal_song_id": 1, "inst_song_id": 2, "vocal_section": 9, "inst_section": 9},
    ]
    got = [c["score_total"] for _, c in judged_rows(fb, cands)]
    # exact pair; song pair's best; stale → song pair's best; gone → dropped.
    assert got == [0.6, 0.8, 0.8]


def test_the_report_reads_a_term_that_separates_and_one_that_does_not():
    from matcher.term_report import MIN_PER_CLASS, term_report
    fb, cands = [], []
    for k in range(2 * MIN_PER_CLASS):
        good = k % 2 == 0
        fb.append({"vocal_song_id": k, "inst_song_id": 100 + k, "vocal_section": 0,
                   "inst_section": 0, "rating": 5 if good else 1,
                   "reasons": ["key_clash"] if not good else []})
        cands.append({"vocal_song_id": k, "inst_song_id": 100 + k,
                      "vocal_section_idx": 0, "inst_section_idx": 0,
                      "score_key": 0.9 if good else 0.3,       # separates
                      "score_label": 0.5,                      # does not
                      "score_room_section": None})             # unmeasured
    rep = term_report(fb, cands, section_weights={"label": 0.4},
                      match_weights={"key_score": 0.26})
    by = {t["key"]: t for t in rep["terms"]}
    assert by["score_key"]["auc"] == 1.0 and by["score_key"]["enough"]
    assert by["score_key"]["verdict"] == "agrees with your ears"
    assert by["score_label"]["auc"] == 0.5
    assert by["score_label"]["verdict"].startswith("no clear signal")
    assert by["score_room_section"]["n"] == 0 and by["score_room_section"]["weight"] == 0.0
    [kc] = rep["reasons"]
    assert kc["reason"] == "key_clash" and kc["n"] == MIN_PER_CLASS
    assert kc["suspects"][0] == {"key": "score_key", "mean_tagged": 0.3,
                                 "mean_rest": 0.9, "n_tagged": MIN_PER_CLASS}


# ── the Phase 3 terms ────────────────────────────────────────────────────────

def test_measured_terms_read_the_stored_section_measurements():
    from matcher.section_score import measured_terms
    v = {"band_energy_vocal": [0, 0, 0.5, 0.5, 0, 0, 0, 0],
         "vocal_activity": 0.75, "energy": 0.9}
    b = {"band_energy_bed": [0.5, 0.5, 0, 0, 0, 0, 0, 0], "energy": 0.6}
    t = measured_terms(v, b)
    assert t == {"score_room_section": 1.0, "score_coverage": 0.75,
                 "score_energy_match": 0.7}
    # Unmeasured is None, never zero.
    assert measured_terms({}, {}) == {"score_room_section": None,
                                      "score_coverage": None,
                                      "score_energy_match": None}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.routes.mashups as routes
    importlib.reload(routes)
    import api.server as server
    importlib.reload(server)
    from fastapi.testclient import TestClient
    return TestClient(server.app), models


def test_the_terms_are_stored_on_every_scored_row_and_reported(client):
    c, models = client
    for n in (1, 2):
        sid = models.upsert_song(f"S{n}", "A", f"u://{n}", 200, status="analysed")
        for stem in ("full", "vocals", "instrumental"):
            models.upsert_features(sid, stem, {"bpm": 124.0, "camelot": "8A",
                                               "mfcc": [0.0] * 13,
                                               "band_energy": [0.125] * 8})
        models.replace_sections(sid, [
            {"section_index": 0, "start_sec": 0.0, "end_sec": 30.0, "label": "chorus",
             "energy": 0.9, "vocal_presence": 0.9, "vocal_activity": 0.8,
             "band_energy_vocal": [0, 0, 0.5, 0.5, 0, 0, 0, 0],
             "band_energy_bed": [0.5, 0.5, 0, 0, 0, 0, 0, 0]},
            {"section_index": 1, "start_sec": 30.0, "end_sec": 60.0, "label": "drop",
             "energy": 0.8, "vocal_presence": 0.05,
             "band_energy_bed": [0.5, 0.5, 0, 0, 0, 0, 0, 0]},
        ])
    from matcher.match import score_all_pairs
    rows = score_all_pairs()["vocal_over_instrumental"]
    assert rows and all(r.get("score_coverage") == 0.8 for r in rows)
    stored = models.get_conn().execute(
        "SELECT score_room_section, score_coverage, score_energy_match "
        "FROM mashup_candidates WHERE combo_type='vocal_over_instrumental'").fetchall()
    assert stored and all(tuple(s) == (1.0, 0.8, 0.9) for s in stored)
    r = rows[0]
    c.post("/api/mashups/feedback", json={
        "vocal_song_id": r["vocal_song_id"], "inst_song_id": r["inst_song_id"],
        "vocal_section": r["vocal_section_idx"], "inst_section": r["inst_section_idx"],
        "rating": 5})
    rep = c.get("/api/mashups/term-report").json()
    assert rep["rated"] == 1 and rep["n_good"] == 1
    assert {t["key"] for t in rep["terms"]} >= {"score_room_section", "score_coverage",
                                               "score_energy_match", "score_key"}
