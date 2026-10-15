"""Phase 3 of the UX plan: a set's clock, warnings and suggestions match how
it will be played.

The 2026-10-15 simulation found a 41-pair set reading 13½ minutes (each slot
counted only its vocal section), the same vocal three slots running unflagged,
"What comes next" offering a 14% tempo jump as "an easy move", and Studio
opening a whole set as 82 lanes at one tempo.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _pair(iid, bpm=124.0, cam="8A", vocal=None, inst=None, secs=32.0, **kw):
    return {"item_id": iid, "position": iid, "kind": "pair", "vocal_bpm": bpm,
            "target_bpm": bpm, "vocal_camelot": cam, "inst_camelot": cam, "inst_bpm": bpm,
            "vocal_section_start": 0.0, "vocal_section_end": secs,
            "vocal_song_id": vocal or iid, "inst_song_id": inst or 100 + iid,
            "vocal_section_idx": 0, "inst_section_idx": 0, **kw}


# ── the clock ─────────────────────────────────────────────────────────────────

def test_a_41_pair_set_runs_about_41_times_32_bars():
    from matcher.setflow import DEFAULT_PLAY_BARS, flow
    bpms = [118 + (k % 9) for k in range(41)]
    items = [_pair(k + 1, bpm=b, cam="8A") for k, b in enumerate(bpms)]
    f = flow(items)
    # 32 bars a slot, less each blend's overlap — the blends are played, as
    # in a real mix.
    want = (sum(DEFAULT_PLAY_BARS * 240.0 / b for b in bpms)
            - sum(t["overlap_secs"] for t in f["transitions"]))
    assert f["total_secs"] == pytest.approx(want, rel=0.15)
    assert f["total_secs"] > 30 * 60, "41 mashups run over half an hour, not 13 minutes"


def test_a_slot_can_play_longer_or_shorter():
    from matcher.setflow import landing
    assert landing(_pair(1, bpm=120))["secs"] == pytest.approx(64.0)
    assert landing(_pair(1, bpm=120, play_bars=16))["secs"] == pytest.approx(32.0)
    assert landing(_pair(1, bpm=120, play_bars=64))["play_bars"] == 64
    # Without a tempo the vocal section is all there is to go on.
    nob = {**_pair(1), "vocal_bpm": None, "target_bpm": None}
    assert landing(nob)["secs"] == pytest.approx(32.0)


# ── repeats ───────────────────────────────────────────────────────────────────

def test_the_same_vocal_three_slots_running_is_flagged():
    from matcher.setflow import flow
    items = [_pair(1, vocal=7), _pair(2, vocal=7), _pair(3, vocal=7)]
    t = flow(items)["transitions"]
    assert "repeat" not in t[0]["violations"], "a bed swap right after is a move"
    assert "repeat" in t[1]["violations"] and t[1]["fit"] in ("warn", "bad")
    assert t[1]["repeated_song_ids"] == [7]


def test_a_record_coming_back_after_one_other_is_flagged():
    from matcher.setflow import flow
    items = [_pair(1, inst=50), _pair(2), _pair(3, inst=50)]
    assert "repeat" in flow(items)["transitions"][1]["violations"]
    # Past the window it is fine.
    far = [_pair(1, inst=50), _pair(2), _pair(3), _pair(4), _pair(5, inst=50)]
    assert all("repeat" not in t["violations"] for t in flow(far)["transitions"])


# ── what comes next ───────────────────────────────────────────────────────────

def test_what_comes_next_keeps_the_fit_rule():
    from matcher.setflow import FIT_BPM_PCT, next_candidates
    items = [_pair(1, bpm=124.0, cam="8A")]
    pool = [{**_pair(10 + k, bpm=124.0 * (1 + d / 100), cam="8A"),
             "score_percentile": 0.9 - k * 0.01}
            for k, d in enumerate([0, 1, 2, 3, 14, 20])]
    out = next_candidates(items, pool, limit=4)
    assert len(out) == 4
    for r in out:
        assert not r["next_relaxed"]
        assert abs(float(r["next_why"].split("·")[-1].strip().split("%")[0])) <= FIT_BPM_PCT


def test_when_too_few_fit_the_closest_others_come_back_marked():
    from matcher.setflow import next_candidates
    items = [_pair(1, bpm=124.0, cam="8A")]
    pool = [{**_pair(10, bpm=124.0, cam="8A"), "score_percentile": 0.5},
            {**_pair(11, bpm=141.0, cam="8A"), "score_percentile": 0.99},
            {**_pair(12, bpm=124.0, cam="2A"), "score_percentile": 0.9}]
    out = next_candidates(items, pool, limit=6)
    assert out[0]["vocal_song_id"] == 10 and not out[0]["next_relaxed"]
    relaxed = {r["vocal_song_id"]: r["next_violations"] for r in out if r["next_relaxed"]}
    assert relaxed[11] == ["bpm"] and relaxed[12] == ["key"]


# ── storage and the route ─────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    sets = importlib.reload(importlib.import_module("api.routes.sets"))
    importlib.reload(importlib.import_module("api.routes.mashups"))
    app = FastAPI()
    app.include_router(sets.router, prefix="/api/sets")
    return TestClient(app), m


def test_a_slots_play_length_is_stored_and_reset(client):
    c, m = client
    a = m.upsert_song("A", "x", "u://a", 200, status="analysed")
    b = m.upsert_song("B", "y", "u://b", 200, status="analysed")
    s = c.post("/api/sets", json={"name": "t"}).json()
    d = c.post(f"/api/sets/{s['id']}/items",
               json={"kind": "pair", "vocal_song_id": a, "inst_song_id": b}).json()
    iid = d["added"]["id"]
    d = c.patch(f"/api/sets/{s['id']}/items/{iid}", json={"play_bars": 64}).json()
    assert d["items"][0]["play_bars"] == 64
    d = c.patch(f"/api/sets/{s['id']}/items/{iid}", json={"play_bars": None}).json()
    assert d["items"][0]["play_bars"] is None
    assert c.patch(f"/api/sets/{s['id']}/items/{iid}", json={"play_bars": 2}).status_code == 422


def test_the_cue_sheet_prints_running_time():
    src = (ROOT / "render" / "exports.py").read_text(encoding="utf-8")
    assert "running time" in src and "of vocal sections" not in src


# ── the screen ────────────────────────────────────────────────────────────────

def test_the_sets_screen_edits_play_length_and_names_repeats():
    sc = (SRC / "components" / "SetScreen.jsx").read_text(encoding="utf-8")
    assert "function PlayLength(" in sc and "onSave({ play_bars:" in sc
    assert "<PlayLength item={selItem} land={land} onSave={(patch) => editSlot(selItem, patch)} />" in sc
    assert 'repeat: "record repeats"' in sc and "a record comes back too soon" in sc
    assert "c.next_relaxed" in sc


def test_studio_gets_each_slots_landing_tempo_and_at_most_twelve_pairs():
    sc = (SRC / "components" / "SetScreen.jsx").read_text(encoding="utf-8")
    # Phase 8's A/B decks lifted the cap from 12 (tests/test_phase8_throughput.py).
    assert "export const STUDIO_CHAIN_MAX = 64;" in sc
    body = sc[sc.index("const openChain = () =>"):sc.index("const editSlot")]
    assert "set_bpm: it.set_bpm ?? f.landings[k]?.bpm ?? null" in body
    assert "pairs.slice(0, STUDIO_CHAIN_MAX)" in body
    # MixStudio already lays each pair at its own targetBpm, unsynced.
    studio = (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
    assert "const at = p.targetBpm || bpm;" in studio
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    assert "targetBpm: c.set_bpm ?? null" in app
