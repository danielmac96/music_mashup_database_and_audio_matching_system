"""Phase 8 of the UX plan: new imports are useful in minutes, and whole sets
can be arranged.

The 2026-10-15 simulation: a freshly imported track had no pairs until Demucs
and the full analysis finished (minutes per track, an hour behind a big
import); a set opened in Studio as two lanes per pair, capped at twelve pairs.
"""
import importlib
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"

SECTIONS = [
    {"start_sec": 0, "end_sec": 16, "label": "intro", "energy": 0.3, "repetition": 1},
    {"start_sec": 16, "end_sec": 48, "label": "chorus", "energy": 0.9,
     "vocal_presence": 0.85, "repetition": 2},
    {"start_sec": 48, "end_sec": 80, "label": "drop", "energy": 0.9,
     "vocal_presence": 0.05, "repetition": 2},
    {"start_sec": 80, "end_sec": 96, "label": "outro", "energy": 0.3, "repetition": 1},
]


def _feat(bpm=124.0):
    return {"bpm": bpm, "key": "A", "mode": "minor", "camelot": "8A",
            "loudness_rms": 0.1, "energy": 0.5,
            "mfcc": [100.0] + [float(k % 5) for k in range(12)]}


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "p8.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    ids = []
    for n in range(3):
        sid = m.upsert_song(f"Song {n}", f"Art {n}", f"u://{n}", 200, "House", status="analysed")
        ids.append(sid)
        for stem in ("full", "vocals", "instrumental"):
            m.upsert_features(sid, stem, _feat())
        m.replace_sections(sid, SECTIONS)
    from matcher.match import score_all_pairs
    score_all_pairs(db_path=m.DB_PATH)
    return m, ids


def _rows(m, sid):
    conn = m.get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT scorer, vocal_song_id, inst_song_id FROM mashup_candidates "
            "WHERE combo_type='vocal_over_instrumental' AND (vocal_song_id=? OR inst_song_id=?)",
            (sid, sid))]
    finally:
        conn.close()


def _new_quick_track(m):
    """What the quick tier leaves: the mix analysed, provisional sections, no stems."""
    sid = m.upsert_song("Fresh", "New", "u://fresh", 200, "House", status="downloaded")
    m.upsert_features(sid, "full", _feat(125.0))
    m.replace_sections(sid, [{**s, "vocal_presence": None, "provisional": 1} for s in SECTIONS])
    return sid


def test_a_quick_tier_track_pairs_before_its_stems(lib):
    m, ids = lib
    from matcher.match import PROVISIONAL_SCORER, score_song_pairs
    before = len(_rows(m, ids[0]))
    sid = _new_quick_track(m)
    t0 = time.monotonic()
    score_song_pairs(sid, db_path=m.DB_PATH)
    took = time.monotonic() - t0
    rows = _rows(m, sid)
    assert rows, "the mix stands in for both stems"
    assert {r["scorer"] for r in rows} == {PROVISIONAL_SCORER}
    assert {r["vocal_song_id"] for r in rows} >= {sid} and {r["inst_song_id"] for r in rows} >= {sid}
    assert took < 10, f"one track against the library took {took:.1f} s"
    # Nobody else's rows were rewritten; the library re-score marker did not move.
    assert len([r for r in _rows(m, ids[0]) if sid not in (r["vocal_song_id"], r["inst_song_id"])]) == before
    assert m.get_pref("last_score", db_path=m.DB_PATH)["started_at"]


def test_stems_landing_replace_provisional_rows(lib):
    m, ids = lib
    from matcher.match import PROVISIONAL_SCORER, score_song_pairs
    sid = _new_quick_track(m)
    score_song_pairs(sid, db_path=m.DB_PATH)
    for stem in ("vocals", "instrumental"):
        m.upsert_features(sid, stem, _feat(125.0))
    m.replace_sections(sid, SECTIONS)
    score_song_pairs(sid, db_path=m.DB_PATH)
    rows = _rows(m, sid)
    assert rows and PROVISIONAL_SCORER not in {r["scorer"] for r in rows}


def test_a_per_track_run_keeps_forced_rows(lib):
    """A pair forced outside the gate survives the track's re-score (one inside
    the gate is simply scored again, as a normal row)."""
    m, ids = lib
    from matcher.match import FORCED_SCORER, score_pair, score_song_pairs
    far = m.upsert_song("Far", "x", "u://far", 200, "Dubstep", status="analysed")
    for stem in ("full", "vocals", "instrumental"):
        m.upsert_features(far, stem, _feat(160.0))
    m.replace_sections(far, SECTIONS)
    score_pair(ids[0], far, db_path=m.DB_PATH)
    score_song_pairs(ids[0], db_path=m.DB_PATH)
    assert FORCED_SCORER in {r["scorer"] for r in _rows(m, far)}


def test_the_pipeline_scores_after_the_quick_tier_and_after_analysis():
    src = (ROOT / "api" / "workers" / "pipeline_worker.py").read_text(encoding="utf-8")
    assert 'if stage == "quick":\n        _score_track(song_id, "provisional")' in src
    fin = src[src.index("def _finalize"):src.index("def run_stage")]
    assert '_score_track(song_id, "analysed")' in fin
    assert 'c.scorer === "provisional"' in (SRC / "components" / "PairCard.jsx").read_text(encoding="utf-8")


# ── the thread cap ────────────────────────────────────────────────────────────

def test_analysis_workers_share_the_cores(monkeypatch):
    import config
    from analysis import threads
    monkeypatch.setattr(config, "ANALYSIS_WORKERS", 3)
    monkeypatch.setattr(config, "QUICK_WORKERS", 2)
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 20)
    assert threads.worker_thread_cap() == 4
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 2)
    assert threads.worker_thread_cap() == 1
    assert threads.cap_analysis_threads() == 1          # never raises
    src = (ROOT / "api" / "workers" / "stages.py").read_text(encoding="utf-8")
    body = src[src.index("def _analyze_stems"):]
    assert "cap_analysis_threads()" in body[:body.index("def ", 10)]


# ── next playable pair ────────────────────────────────────────────────────────

def test_the_queue_says_when_the_next_track_gets_pairs():
    from api import queue_eta
    rates = {"download": {"per_audio_sec": None, "typical_secs": 20.0},
             "quick": {"per_audio_sec": None, "typical_secs": 25.0}}
    downloading = {"id": "a", "status": "running", "song_id": 1,
                   "stages": {"download": {"state": "running", "started_at": None}}}
    in_quick = {"id": "b", "status": "running", "song_id": 2,
                "stages": {"download": {"state": "done"},
                           "quick": {"state": "running", "started_at": None}}}
    stemming = {"id": "c", "status": "running", "song_id": 3,
                "stages": {"quick": {"state": "done"}, "stems": {"state": "running"}}}
    from datetime import datetime, timezone
    for job in (downloading, in_quick, stemming):
        queue_eta._annotate_one(job, rates, {}, {}, datetime.now(timezone.utc),
                                {p: 0.0 for p in queue_eta.POOL_GROUPS},
                                {p: False for p in queue_eta.POOL_GROUPS})
    assert downloading["playable_eta_secs"] == 45.0      # its download, then the quick tier
    assert in_quick["playable_eta_secs"] == 25.0
    assert stemming["playable_eta_secs"] is None          # it has its pairs already
    assert queue_eta.next_playable_secs([downloading, in_quick, stemming]) == 25.0
    jobs_src = (ROOT / "api" / "routes" / "jobs.py").read_text(encoding="utf-8")
    assert '"next_playable_secs": queue_eta.next_playable_secs(active)' in jobs_src
    assert "next pairs in" in (SRC / "components" / "QueueScreen.jsx").read_text(encoding="utf-8")
    assert "next pairs in" in (SRC / "shell" / "Sidebar.jsx").read_text(encoding="utf-8")


# ── A/B decks ─────────────────────────────────────────────────────────────────

def test_a_set_opens_on_two_deck_rows():
    ms = (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
    import re
    keys = re.search(r"const LANE_KEYS = \[([^\]]*)\];", ms).group(1)
    assert '"deck"' in keys and '"slot"' in keys
    lay = ms[ms.index("const layPairs = useCallback"):ms.index("// Deck clips hold decoded audio")]
    assert 'const deck = decked ? (slot % 2 ? "B" : "A") : null;' in lay
    assert "cursor -= p.overlapSecs" in lay, "each move overlaps the next mashup"
    # Rows group every deck's clips; the panel shows only the selected slot.
    assert "{timelineRows.map((row) => {" in ms and "if (row.deck) return renderDeckRow(row);" in ms
    assert "|| (selLane?.deck && l.slot === selLane.slot)" in ms
    # Deck clips decode near the playhead only, and let go of their audio after.
    assert "await loadLaneMedia(id, track.id, stem, { audio: !lane.deck });" in ms
    assert "releaseStem(url)" in ms
    assert "export function releaseStem(url)" in (SRC / "engine" / "decode.js").read_text(encoding="utf-8")
    sc = (SRC / "components" / "SetScreen.jsx").read_text(encoding="utf-8")
    assert "export const STUDIO_CHAIN_MAX = 64;" in sc
    assert "overlap_secs: k > from ? (f.transitions[k - 1]?.overlap_secs ?? 0) : 0" in sc
    assert "overlapSecs: c.overlap_secs ?? 0" in (SRC / "App.jsx").read_text(encoding="utf-8")
    from render.dsp import MAX_CLIPS
    assert MAX_CLIPS >= 128, "a whole decked set renders as one mixdown"
