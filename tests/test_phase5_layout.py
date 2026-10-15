"""Phase 5 of the UX plan: the key columns and pair facts fit at 1280–1680 px,
and searches behave the way DJs expect.

The 2026-10-15 simulation: 71 of 100 personas met a Library table hiding
columns behind its own scrollbar; every expanded card read VOI / HARM / RHY /
ENGY and some filled two thirds of a 720 p window; Track detail printed
"confidence 0.020"; a typed key matched only itself; an empty A + B search
could not be scored.
"""
import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


# ── the table fits ────────────────────────────────────────────────────────────

def test_default_columns_make_way_before_the_table_scrolls():
    tt = _read("components/TrackTable.jsx")
    m = re.search(r'export const FIT_DROP = \[([^\]]*)\];', tt)
    drop = re.findall(r'"(\w+)"', m.group(1))
    assert drop[0] == "status" and "title" not in drop and "bpm" not in drop and "key" not in drop
    fit = tt[tt.index("const fitted = useMemo"):tt.index("const visible = fitted.cols;")]
    # Only default columns go: never an attribute, never one the user added.
    assert "!x.attr && x.defaultVisible !== false" in fit
    # The title shrinks first, never below its min.
    assert "titleCol.width - titleCol.min" in tt
    assert "fitted.hidden.length ? `+${fitted.hidden.length}`" in tt


def test_every_drop_column_is_a_default_column():
    tt = _read("components/TrackTable.jsx")
    drop = re.findall(r'"(\w+)"', re.search(r'FIT_DROP = \[([^\]]*)\]', tt).group(1))
    for col in drop:
        line = next(l for l in tt.splitlines() if f'{{ id: "{col}"' in l)
        assert "defaultVisible: false" not in line, col


# ── the expanded card ─────────────────────────────────────────────────────────

def test_meters_are_spelled_out_and_folded():
    card = _read("components/PairCard.jsx")
    meters = card[card.index("export function metersOf"):card.index("// The waveform of one section")]
    for word in ('"Vocal"', '"Harmony"', '"Rhythm"', '"Energy"'):
        assert word in meters
    for jargon in ('"VOI"', '"HARM"', '"RHY"', '"ENGY"'):
        assert jargon not in meters
    assert '<details className="pcx-details">' in card
    folded = card[card.index('<details className="pcx-details">'):card.index("</details>")]
    assert "metersOf(c)" in folded and "termsOf(c)" in folded and "{engine}" in folded
    # The decision stays visible: harmony, bass clash, effort.
    visible = card[card.index('<div className="pcx-side">'):card.index('<details className="pcx-details">')]
    assert "pcx-harm" in visible and "pcx-clash" in visible and "pcx-effort" in visible


def test_key_confidence_reads_in_words():
    theme = _read("theme.js")
    assert "export function keyConfidenceWord(f)" in theme
    detail = _read("components/TrackDetail.jsx")
    assert "`${keyConfidenceWord(f)} confidence`" in detail
    assert "key_confidence.toFixed(3)" not in detail
    assert "key_confidence?.toFixed(3)" not in _read("components/TrackTable.jsx")


# ── search ────────────────────────────────────────────────────────────────────

def test_a_tilde_key_reads_as_its_mixable_neighbours():
    f = _read("hooks/useLibraryFilters.js")
    assert r"/^(\d{1,2})([ab])(~?)$/i" in f
    assert "m[3] ? compatibleKeys(code) : [code]" in f
    assert "Mixes with ${parsed.keyNeighbours.join" in f
    assert "8A~ 124-128" in _read("components/LibraryFilters.jsx")


# ── score this pair anyway ────────────────────────────────────────────────────

@pytest.fixture()
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "f.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    ids = []
    for n, bpm in enumerate((124.0, 150.0)):      # 26 BPM apart: outside the gate
        sid = m.upsert_song(f"Song {n}", f"Art {n}", f"u://{n}", 200, "House",
                            status="analysed")
        ids.append(sid)
        for stem in ("full", "vocals", "instrumental"):
            m.upsert_features(sid, stem, {"bpm": bpm, "key": "A", "mode": "minor",
                                          "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5,
                                          "mfcc": [100.0] + [float(k % 5) for k in range(12)]})
        m.replace_sections(sid, [
            {"start_sec": 0, "end_sec": 16, "label": "intro", "energy": 0.3,
             "vocal_presence": 0.0, "repetition": 1},
            {"start_sec": 16, "end_sec": 48, "label": "chorus", "energy": 0.9,
             "vocal_presence": 0.85, "repetition": 2},
            {"start_sec": 48, "end_sec": 80, "label": "drop", "energy": 0.9,
             "vocal_presence": 0.05, "repetition": 2},
            {"start_sec": 80, "end_sec": 96, "label": "outro", "energy": 0.3,
             "vocal_presence": 0.0, "repetition": 1}])
    from matcher.match import score_all_pairs
    score_all_pairs(db_path=m.DB_PATH)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    mashups = importlib.reload(importlib.import_module("api.routes.mashups"))
    app = FastAPI()
    app.include_router(mashups.router, prefix="/api/mashups")
    return TestClient(app), m, ids


def test_a_pair_outside_the_gate_scores_on_request(lib):
    client, m, (a, b) = lib
    before = client.get("/api/mashups", params={"search": "Song 0 + Song 1"}).json()
    assert before["count"] == 0, "the tempo gate kept them apart"
    r = client.post("/api/mashups/score-pair", json={"vocal_song_id": a, "inst_song_id": b})
    assert r.status_code == 200
    rows = r.json()["candidates"]
    assert rows and all(x["scorer"] == "forced" for x in rows)
    # Through the listing: the recipe and the Heavy label say what it takes.
    assert all(x.get("recipe") and x.get("effort_label") == "Heavy" for x in rows)
    after = client.get("/api/mashups", params={"search": "Song 0 + Song 1"}).json()
    assert after["count"] > 0
    # A second force replaces the first rather than piling up rows.
    again = client.post("/api/mashups/score-pair",
                        json={"vocal_song_id": a, "inst_song_id": b}).json()
    conn = m.get_conn()
    n = conn.execute("SELECT COUNT(*) FROM mashup_candidates WHERE scorer='forced'").fetchone()[0]
    conn.close()
    assert n == len(again["candidates"])


def test_score_pair_refuses_one_song_and_unanalysed_songs(lib):
    client, m, (a, b) = lib
    assert client.post("/api/mashups/score-pair",
                       json={"vocal_song_id": a, "inst_song_id": a}).status_code == 400
    c = m.upsert_song("Bare", "x", "u://bare", 200, status="queued")
    assert client.post("/api/mashups/score-pair",
                       json={"vocal_song_id": a, "inst_song_id": c,
                             "both_ways": False}).status_code == 409


def test_the_dock_offers_score_anyway_and_the_card_says_forced():
    dock = _read("components/PairDock.jsx")
    assert "api.scorePairAnyway(forceTargets[0].id, forceTargets[1].id)" in dock
    assert "tracks={library.tracks}" in _read("App.jsx")
    assert 'c.scorer === "forced"' in _read("components/PairCard.jsx")
    assert ".pc-forced {" in _read("styles.css")
