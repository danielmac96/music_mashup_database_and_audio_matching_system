"""Set slots that are not pairs, the energy curve and gap suggestions
(revamp B3, readme §5.12).

A slot is a pair, a single track, a break or a gap. Breaks and gaps are never
graded but advance the running time; a gap is filled from /suggest — pairs that
fit BOTH neighbours (±1 Camelot step, relative = 0; ±4% tempo, half/double
allowed), ranked by distance to the energy curve, then score.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _pair(iid, bpm, cam, secs=32.0, vocal=None, inst=None, **kw):
    return {"item_id": iid, "position": iid, "kind": "pair", "vocal_bpm": bpm,
            "target_bpm": bpm, "vocal_camelot": cam, "inst_camelot": cam, "inst_bpm": bpm,
            "vocal_section_start": 0.0, "vocal_section_end": secs,
            "vocal_song_id": vocal or iid, "inst_song_id": inst or 100 + iid,
            "vocal_section_idx": 0, "inst_section_idx": 0, **kw}


def _slot(iid, kind, secs):
    return {"item_id": iid, "position": iid, "kind": kind, "duration_secs": secs}


# ── the fit rule ────────────────────────────────────────────────────────────

def test_key_fits_is_one_step_with_relative_free():
    from matcher.setflow import key_fits
    assert key_fits("8A", "8A") and key_fits("8A", "9A") and key_fits("8A", "7A")
    assert key_fits("8A", "8B"), "relative major/minor is no step"
    assert not key_fits("8A", "9B"), "a diagonal is two moves"
    assert not key_fits("8A", "10A")
    assert key_fits("12A", "1A"), "the wheel wraps"
    assert key_fits(None, "8A") is None


def test_each_move_carries_the_fit_dot():
    from matcher.setflow import flow
    t = flow([_pair(1, 124, "8A"), _pair(2, 128, "9A")])["transitions"][0]
    assert t["fit"] == "ok" and t["violations"] == []          # +3.2%, one step
    t = flow([_pair(1, 124, "8A"), _pair(2, 132, "9A")])["transitions"][0]
    assert t["fit"] == "warn" and t["violations"] == ["bpm"]
    t = flow([_pair(1, 124, "8A"), _pair(2, 132, "3B")])["transitions"][0]
    assert t["fit"] == "bad" and set(t["violations"]) == {"key", "bpm"}
    t = flow([_pair(1, 87, "8A"), _pair(2, 174, "8A")])["transitions"][0]
    assert t["fit"] == "ok", "half/double time is no tempo change"


# ── breaks, gaps and tracks in the flow ─────────────────────────────────────

def test_breaks_and_gaps_are_not_graded_but_take_time():
    from matcher.setflow import flow
    items = [_pair(1, 124, "8A", secs=40), _slot(2, "break", 30),
             _pair(3, 124, "8A", secs=40), _slot(4, "gap", 90)]
    f = flow(items)
    assert [t["grade"] for t in f["transitions"]] == ["none", "none", "none"]
    assert all(t["move"] is None and t["overlap_secs"] == 0 for t in f["transitions"])
    assert f["starts"] == [0, 40, 70, 110]
    assert f["total_secs"] == 200
    assert f["grades"]["smooth"] == 0


def test_a_track_lands_at_its_own_tempo_and_key():
    from matcher.setflow import flow
    track = {"item_id": 2, "kind": "track", "song_id": 9, "bpm": 126.0,
             "camelot": "9A", "song_duration": 200.0}
    f = flow([_pair(1, 124, "8A"), track])
    assert f["landings"][1] == {"bpm": 126.0, "camelot": "9A", "secs": 200.0, "bed_shift": None}
    t = f["transitions"][0]
    assert t["grade"] == "smooth" and t["fit"] == "ok"
    # Two tracks never look like a bed or vocal swap (None == None).
    other = {**track, "item_id": 3}
    assert flow([track, other])["transitions"][0]["move"]["type"] != "bed_swap"


def test_the_tempo_curve_skips_breaks():
    from matcher.setflow import flow
    items = [_pair(1, 118, "8A"), _slot(2, "break", 20), _pair(3, 130, "8A")]
    f = flow(items, {"start_bpm": 120, "end_bpm": 124})
    assert [l["bpm"] for l in f["landings"]] == [120, None, 124]


def test_auto_order_keeps_breaks_in_place():
    from matcher.setflow import suggest_order
    items = [_pair(1, 130, "8A"), _slot(2, "break", 20), _pair(3, 120, "8A"),
             _pair(4, 125, "8A")]
    order = suggest_order(items)
    assert order[1] == 2, "the break stays second"
    assert order[0] == 3, "the slowest pair opens"


# ── the energy curve and gap suggestions ────────────────────────────────────

def test_energy_target_interpolates_and_holds():
    from matcher.setflow import energy_target
    plan = [{"t": 0.2, "e": 0.2}, {"t": 0.6, "e": 0.8}]
    assert energy_target(None, 0.5) is None
    assert energy_target(plan, 0.0) == 0.2
    assert energy_target(plan, 0.4) == pytest.approx(0.5)
    assert energy_target(plan, 1.0) == 0.8


def _cand(v, b, bpm, cam, pct):
    return {"vocal_song_id": v, "inst_song_id": b, "vocal_section_idx": 0,
            "inst_section_idx": 0, "vocal_bpm": bpm, "inst_bpm": bpm, "target_bpm": bpm,
            "vocal_camelot": cam, "inst_camelot": cam, "score_percentile": pct,
            "vocal_section_start": 0.0, "vocal_section_end": 30.0}


def test_gap_candidates_fit_both_sides_ranked_by_energy():
    from matcher.setflow import gap_candidates
    items = [_pair(1, 124, "8A", secs=60), _slot(2, "gap", 60), _pair(3, 126, "9A", secs=60)]
    pool = [
        _cand(10, 20, 125, "8A", 0.99),   # fits, energy far from the curve
        _cand(11, 21, 125, "9A", 0.70),   # fits, energy on the curve
        _cand(12, 22, 140, "8A", 1.00),   # tempo breaks both sides
        _cand(13, 23, 125, "3B", 1.00),   # key breaks both sides
        _cand(1, 101, 124, "8A", 1.00),   # already in the set
    ]
    energy = {20: 0.1, 21: 0.9, 22: 0.5, 23: 0.5}
    out = gap_candidates(items, 1, pool, energy, energy_plan=[{"t": 0, "e": 0.9}, {"t": 1, "e": 0.9}])
    assert not out["relaxed"]
    assert [(c["vocal_song_id"], c["inst_song_id"]) for c in out["candidates"]] == [(11, 21), (10, 20)]
    top = out["candidates"][0]
    assert top["fit"]["from_prev"]["key_ok"] and top["fit"]["into_next"]["bpm_ok"]
    assert out["neighbours"] == {"prev": 1, "next": 3}
    # Without a curve the better score wins.
    out = gap_candidates(items, 1, pool, energy)
    assert out["candidates"][0]["vocal_song_id"] == 10


def test_when_nothing_fits_the_closest_come_back_marked():
    from matcher.setflow import gap_candidates
    items = [_pair(1, 124, "8A"), _slot(2, "gap", 60), _pair(3, 124, "8A")]
    pool = [_cand(12, 22, 140, "8A", 0.9), _cand(13, 23, 140, "3B", 0.99)]
    out = gap_candidates(items, 1, pool, {})
    assert out["relaxed"]
    assert out["candidates"][0]["violations"] == ["bpm"], "fewest broken rules first"
    assert set(out["candidates"][1]["violations"]) == {"bpm", "key"}


# ── migration ───────────────────────────────────────────────────────────────

def test_old_set_items_are_rebuilt_with_nullable_pairs(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE sets (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            note TEXT DEFAULT '', created_at TEXT, updated_at TEXT, UNIQUE(name));
        CREATE TABLE set_items (id INTEGER PRIMARY KEY AUTOINCREMENT, set_id INTEGER NOT NULL,
            position INTEGER NOT NULL, vocal_song_id INTEGER NOT NULL, inst_song_id INTEGER NOT NULL,
            vocal_section INTEGER, inst_section INTEGER, payload_json TEXT, added_at TEXT);
        INSERT INTO sets(name) VALUES ('kept');
        INSERT INTO set_items(set_id, position, vocal_song_id, inst_song_id, payload_json)
            VALUES (1, 0, 5, 6, '{}'), (1, 1, 7, 8, '{}');
    """)
    conn.commit(); conn.close()
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    import database.models as models
    models.init_db(db)
    conn = sqlite3.connect(db)
    info = {r[1]: r for r in conn.execute("PRAGMA table_info(set_items)")}
    assert {"kind", "song_id", "label", "duration_secs", "transition_json"} <= set(info)
    assert info["vocal_song_id"][3] == 0, "pair ids are nullable now"
    rows = conn.execute("SELECT id, kind, vocal_song_id FROM set_items ORDER BY id").fetchall()
    assert rows == [(1, "pair", 5), (2, "pair", 7)]
    assert "energy_plan_json" in {r[1] for r in conn.execute("PRAGMA table_info(sets)")}
    idx = {r[1] for r in conn.execute("PRAGMA index_list(set_items)")}
    assert {"idx_set_items_set", "ux_set_items_pair"} <= idx
    conn.close()
    # Running it again is a no-op.
    models._INITIALIZED_PATHS.discard(str(db))
    models.init_db(db)


# ── the API ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    for name in ("mashups", "sets"):
        mod = importlib.reload(importlib.import_module(f"api.routes.{name}"))
        app.include_router(mod.router, prefix=f"/api/{name}")
    ids = [models.upsert_song(f"S{n}", "A", f"u://{n}", 200, status="analysed")
           for n in range(6)]
    for n, sid in enumerate(ids):
        models.upsert_features(sid, "full", {"bpm": 124.0, "camelot": "8A", "key": "A",
                                             "mode": "minor", "energy": 0.1 * (n + 1)})
    side = lambda sid, bpm, cam="8A": {"song_id": sid, "title": f"S{sid}", "artist": "A",
                                       "bpm": bpm, "camelot": cam, "loudness_rms": 0.1,
                                       "energy": 0.5}
    for v, b, bpm in ((ids[0], ids[1], 124.0), (ids[2], ids[3], 125.0), (ids[4], ids[5], 124.0)):
        models.upsert_candidate(side(v, bpm), side(b, bpm),
                                {"total": 0.8, "bpm_score": 1.0, "key_score": 1.0,
                                 "energy_score": 0.5, "timbre_score": 0.5})
    return TestClient(app), models, ids


def test_slots_through_the_api(client):
    c, models, ids = client
    s = c.post("/api/sets", json={"name": "Night"}).json()
    sid = s["id"]
    s = c.post(f"/api/sets/{sid}/items", json={"vocal_song_id": ids[0], "inst_song_id": ids[1]}).json()
    s = c.post(f"/api/sets/{sid}/items", json={"kind": "track", "song_id": ids[2]}).json()
    # A break inserted between the two, and a gap at the end.
    s = c.post(f"/api/sets/{sid}/items", json={"kind": "break", "label": "Talk", "position": 1}).json()
    s = c.post(f"/api/sets/{sid}/items", json={"kind": "gap"}).json()
    kinds = [(i["kind"], i["position"]) for i in s["items"]]
    assert kinds == [("pair", 0), ("break", 1), ("track", 2), ("gap", 3)]
    brk, track, gap = s["items"][1], s["items"][2], s["items"][3]
    assert brk["label"] == "Talk" and brk["duration_secs"] == 30
    assert gap["duration_secs"] == 90
    assert track["title"] == "S2" and track["bpm"] == 124 and track["camelot"] == "8A"
    assert track["energy"] is not None and s["items"][0]["energy"] is not None
    assert [t["grade"] for t in s["flow"]["transitions"]] == ["none", "none", "none"]

    s = c.patch(f"/api/sets/{sid}/items/{brk['item_id']}", json={"duration_secs": 45}).json()
    assert s["items"][1]["duration_secs"] == 45
    assert c.patch(f"/api/sets/{sid}/items/99999", json={"label": "x"}).status_code == 404

    # Validation: a pair needs both ids, a track its song.
    assert c.post(f"/api/sets/{sid}/items", json={"kind": "pair", "vocal_song_id": 1}).status_code == 422
    assert c.post(f"/api/sets/{sid}/items", json={"kind": "track"}).status_code == 422
    assert c.post(f"/api/sets/{sid}/items", json={"kind": "solo"}).status_code == 422

    # The energy curve round-trips, sorted by t; an empty list clears it.
    s = c.patch(f"/api/sets/{sid}", json={"energy_plan": [{"t": 0.8, "e": 0.9}, {"t": 0.1, "e": 0.3}]}).json()
    assert s["energy_plan"] == [{"t": 0.1, "e": 0.3}, {"t": 0.8, "e": 0.9}]
    assert c.patch(f"/api/sets/{sid}", json={"energy_plan": [{"t": 2, "e": 0}]}).status_code == 422
    assert c.patch(f"/api/sets/{sid}", json={"energy_plan": []}).json()["energy_plan"] is None

    # Suggestions for the gap, with the dock's playback terms on them.
    out = c.get(f"/api/sets/{sid}/suggest", params={"position": 3}).json()
    assert out["neighbours"]["prev"] == track["item_id"] and out["neighbours"]["next"] is None
    assert out["candidates"] and "semitone_shift" in out["candidates"][0]
    assert all((x["vocal_song_id"], x["inst_song_id"]) != (ids[0], ids[1]) for x in out["candidates"])
    assert c.get(f"/api/sets/{sid}/suggest", params={"position": 99}).status_code == 400

    # Exports: the break is a cue, the gap is not written, rekordbox skips both.
    csv = c.get(f"/api/sets/{sid}/export?format=csv").text.splitlines()
    assert csv[0].endswith(",kind")
    assert [line.rsplit(",", 1)[1] for line in csv[1:]] == ["pair", "break", "track"]
    cue = c.get(f"/api/sets/{sid}/export?format=cue").text
    assert "— Talk (0:45) —" in cue and "A - S2" in cue
    rb = c.get(f"/api/sets/{sid}/export?format=rekordbox")
    assert rb.status_code == 200 and "Talk" not in rb.text

    # Removing keeps the positions dense.
    s = c.delete(f"/api/sets/{sid}/items/{brk['item_id']}").json()
    assert [i["position"] for i in s["items"]] == [0, 1, 2]


def test_fl_export_queues_the_pairs_in_order(client, monkeypatch):
    c, models, ids = client
    from api.workers import session_worker
    seen = {}
    monkeypatch.setattr(session_worker, "run_batch", lambda job_id, pairs: seen.update(pairs=pairs))
    sid = c.post("/api/sets", json={"name": "FL"}).json()["id"]
    c.post(f"/api/sets/{sid}/items", json={"vocal_song_id": ids[2], "inst_song_id": ids[3]})
    c.post(f"/api/sets/{sid}/items", json={"kind": "break"})
    c.post(f"/api/sets/{sid}/items", json={"vocal_song_id": ids[0], "inst_song_id": ids[1]})
    r = c.get(f"/api/sets/{sid}/export?format=fl").json()
    assert r["pair_count"] == 2 and r["archive_url"].endswith("/archive") and r["job_id"]
    assert [(p["vocal_song_id"], p["inst_song_id"]) for p in seen["pairs"]] == \
        [(ids[2], ids[3]), (ids[0], ids[1])]
    empty = c.post("/api/sets", json={"name": "Empty"}).json()["id"]
    assert c.get(f"/api/sets/{empty}/export?format=fl").status_code == 400


# ── B4: hearing a transition ────────────────────────────────────────────────

def _sec_pair(iid, bpm, vocal, inst, **kw):
    return {**_pair(iid, bpm, "8A", vocal=vocal, inst=inst), "inst_section_start": 60.0,
            "inst_section_end": 120.0, "vocal_section_start": 30.0,
            "vocal_section_end": 90.0, **kw}


def test_transition_clips_tail_into_head_with_the_overlap():
    from render.transition import transition_clips
    a = _sec_pair(1, 120, 10, 11)
    b = _sec_pair(2, 120, 12, 13, alignment_offset=0.25)
    out = transition_clips(a, b, 120.0, 120.0, overlap_bars=4, bars=8)
    va, ba, vb, bb = out["clips"]
    # 8 bars at 120 BPM = 16 s: the last 16 s of A's sections, the first of B's.
    assert (va["start_sec"], va["end_sec"]) == (74.0, 90.0)
    assert (ba["start_sec"], ba["end_sec"]) == (104.0, 120.0)
    assert (vb["start_sec"], vb["end_sec"]) == (30.0, 46.0)
    assert va["stem"] == "vocals" and ba["stem"] == "instrumental"
    # B comes in 4 bars (8 s) before A ends; its bed on its own bar line.
    assert out["join_sec"] == 8.0 and vb["offset_sec"] == 8.0 and bb["offset_sec"] == 8.25
    assert out["secs"] == pytest.approx(8.0 + 16.25)
    # A cut is back to back.
    assert transition_clips(a, b, 120.0, 120.0, 0, 8)["join_sec"] == 16.0


def test_transition_clips_land_each_side_at_its_tempo():
    from render.transition import transition_clips
    a = _sec_pair(1, 120, 10, 11)
    b = _sec_pair(2, 63, 12, 13)          # a 63 BPM vocal lands at 126 as double time
    out = transition_clips(a, b, 120.0, 126.0, 0, 4)
    vb = out["clips"][2]
    assert vb["rate"] == pytest.approx(1.0) and vb["end_sec"] - vb["start_sec"] == pytest.approx(4 * 240 / 126, abs=1e-3)
    va = out["clips"][0]
    assert va["rate"] == 1.0
    track = {"kind": "track", "song_id": 5, "bpm": 128.0, "song_duration": 200.0}
    out = transition_clips(track, a, 128.0, 120.0, 0, 8)
    t = out["clips"][0]
    assert t["stem"] == "full" and t["end_sec"] == 200.0 and t["start_sec"] == pytest.approx(185.0)
    assert transition_clips({"kind": "break"}, a, 0, 120.0)["error"]
    no_sections = {**a, "vocal_section_start": None}
    assert "re-score" in transition_clips(no_sections, b, 120.0, 120.0)["error"]


def test_the_transition_route_renders_once_and_caches(client, monkeypatch, tmp_path):
    c, models, ids = client
    import render.mixdown as mixdown
    calls = []

    def fake(token, clips, on_progress=None, db_path=None):
        calls.append(clips)
        path = mixdown.mixdown_path(token)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF0000WAVE")
        return path
    monkeypatch.setattr(mixdown, "build_mixdown", fake)
    sid = c.post("/api/sets", json={"name": "T"}).json()["id"]
    c.post(f"/api/sets/{sid}/items", json={"vocal_song_id": ids[0], "inst_song_id": ids[1]})
    s = c.post(f"/api/sets/{sid}/items", json={"vocal_song_id": ids[4], "inst_song_id": ids[5]}).json()
    # The fixture's rows have no section spans: a clear 422, not a crash.
    second = s["items"][1]["item_id"]
    r = c.get(f"/api/sets/{sid}/transition/{second}/audio")
    assert r.status_code == 422 and "re-score" in r.json()["detail"]
    assert c.get(f"/api/sets/{sid}/transition/{s['items'][0]['item_id']}/audio").status_code == 409

    # Give both pairs sections, then it renders, and the second call is cached.
    from render import transition as tr
    real = tr.transition_clips
    monkeypatch.setattr(tr, "transition_clips", lambda a, b, *x: real(
        {**a, "vocal_section_start": 0, "vocal_section_end": 30, "inst_section_start": 0,
         "inst_section_end": 30}, {**b, "vocal_section_start": 0, "vocal_section_end": 30,
                                   "inst_section_start": 0, "inst_section_end": 30}, *x))
    r = c.get(f"/api/sets/{sid}/transition/{second}/audio?bars=4")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert float(r.headers["x-join-sec"]) >= 0
    c.get(f"/api/sets/{sid}/transition/{second}/audio?bars=4")
    assert len(calls) == 1, "a second listen is served from the cache"

    # A break on the boundary has nothing to blend.
    s = c.post(f"/api/sets/{sid}/items", json={"kind": "break"}).json()
    brk = s["items"][2]["item_id"]
    assert c.get(f"/api/sets/{sid}/transition/{brk}/audio").status_code == 409
