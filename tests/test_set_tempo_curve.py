"""Sets with a tempo curve and real transitions (readme §5.12, phase 4).

A set used to land every mashup at its vocal's own tempo and grade the move
between them. Now a set can carry a tempo curve — each mashup lands at its
point between start and end BPM, vocal and bed both stretched to it — and
each move is a musical one (bed swap, vocal swap, blend, echo out, cut),
suggested and overridable, with the overlap counted in the running time.
"""
import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FRONT = ROOT / "frontend" / "src"


def _item(iid, bpm, cam, start=0.0, end=32.0, vocal=None, inst=None, **kw):
    return {"item_id": iid, "position": iid, "vocal_bpm": bpm, "target_bpm": bpm,
            "vocal_camelot": cam, "inst_camelot": cam, "inst_bpm": bpm,
            "vocal_section_start": start, "vocal_section_end": end,
            "vocal_song_id": vocal or iid, "inst_song_id": inst or 100 + iid, **kw}


# ── the curve ────────────────────────────────────────────────────────────────

def test_targets_run_linearly_from_start_to_end():
    from matcher.setflow import tempo_targets
    assert tempo_targets(3, None) == [None] * 3
    assert tempo_targets(5, {"start_bpm": 120, "end_bpm": 128}) == [120, 122, 124, 126, 128]
    assert tempo_targets(3, {"start_bpm": 124}) == [124, 124, 124], "no end holds the tempo"
    assert tempo_targets(1, {"start_bpm": 120, "end_bpm": 128}) == [120]


def test_on_a_curve_every_mashup_lands_on_it():
    from matcher.setflow import flow
    items = [_item(1, 118, "8A"), _item(2, 131, "8A"), _item(3, 126, "8A")]
    f = flow(items, {"start_bpm": 122, "end_bpm": 126})
    assert [l["bpm"] for l in f["landings"]] == [122, 124, 126]
    # Gentle curve, same key: every move smooth even though the vocals'
    # own tempos jump around.
    assert f["grades"]["smooth"] == 2
    # A vocal stretched to land is shorter or longer in time: 32 s at 118
    # played at 122 lasts 32 × 118/122.
    assert f["landings"][0]["secs"] == pytest.approx(32 * 118 / 122)
    # Without the curve the old landings come back.
    assert [l["bpm"] for l in flow(items, {})["landings"]] == [118, 131, 126]


def test_the_recipe_stretches_the_vocal_to_the_curve():
    from matcher.recipe import pair_recipe
    r = pair_recipe(_item(1, 120, "8A", stretch_factor=1.0), target_bpm=126)
    assert r["vocal_rate"] == pytest.approx(1.05) and r["bed_rate"] == pytest.approx(1.05)
    adj = {a["key"]: a for a in r["adjustments"]}
    assert adj["vocal_stretch"]["text"] == "vocal tempo +5.0%"
    assert adj["vocal_stretch"]["level"] == "heavy", "a voice turns heavy above 4%"
    assert "vocal_stretch" in {w["key"] for w in r["warnings"]}
    # Folded: a 63 BPM vocal lands at 126 as double time, not a 100% stretch.
    r2 = pair_recipe(_item(1, 63, "8A", stretch_factor=1.0), target_bpm=126)
    assert r2["vocal_fold"] == "double" and r2["vocal_rate"] == 1.0
    # No target: the vocal plays native, as before.
    assert pair_recipe(_item(1, 120, "8A", stretch_factor=1.0))["vocal_rate"] == 1.0


# ── the moves ────────────────────────────────────────────────────────────────

def test_a_shared_record_suggests_a_swap():
    from matcher.setflow import flow
    f = flow([_item(1, 124, "8A", vocal=7), _item(2, 124, "3A", vocal=7)])
    assert f["transitions"][0]["move"]["type"] == "bed_swap"
    f = flow([_item(1, 124, "8A", inst=9), _item(2, 124, "8A", inst=9)])
    assert f["transitions"][0]["move"]["type"] == "vocal_swap"


def test_the_grade_picks_the_move_otherwise():
    from matcher.setflow import flow
    def move(a, b):
        return flow([a, b])["transitions"][0]["move"]
    assert move(_item(1, 124, "8A"), _item(2, 124, "9A"))["type"] == "blend"
    assert move(_item(1, 124, "8A"), _item(2, 124, "9A"))["bars"] == 8
    assert move(_item(1, 124, "8A"), _item(2, 128, "10A"))["bars"] == 4
    assert move(_item(1, 124, "8A"), _item(2, 128, "2A"))["type"] == "echo_out"
    assert move(_item(1, None, "8A"), _item(2, 128, "8A"))["type"] == "cut"


def test_a_chosen_move_wins_and_remembers_the_suggestion():
    from matcher.setflow import flow
    b = _item(2, 124, "9A", transition={"type": "cut", "bars": None})
    m = flow([_item(1, 124, "8A"), b])["transitions"][0]["move"]
    assert m["type"] == "cut" and not m["suggested"]
    assert m["suggestion"]["type"] == "blend"
    assert flow([_item(1, 124, "8A"), b])["transitions"][0]["overlap_secs"] == 0.0


def test_overlap_shortens_the_running_time_but_never_swallows_a_mashup():
    from matcher.setflow import flow
    # 8 bars at 120 BPM = 16 s; both sections 32 s → full 16 s overlap.
    f = flow([_item(1, 120, "8A"), _item(2, 120, "8A")])
    assert f["transitions"][0]["overlap_secs"] == 16.0
    assert f["starts"] == [0.0, 16.0] and f["total_secs"] == 48.0
    # A 10 s section caps the overlap at 5 s.
    f = flow([_item(1, 120, "8A", end=10.0), _item(2, 120, "8A")])
    assert f["transitions"][0]["overlap_secs"] == 5.0


def test_next_candidates_rank_fit_and_move_together():
    from matcher.setflow import next_candidates
    items = [_item(1, 124, "8A", vocal=7)]
    pool = [
        {**_item(10, 124, "3A"), "score_percentile": 0.95},        # great, far key
        {**_item(11, 124, "8A"), "score_percentile": 0.80},        # good, same key
        {**_item(12, 124, "9A", vocal=7), "score_percentile": 0.75},  # bed swap
        {**_item(1, 124, "8A", vocal=7), "score_percentile": 0.99},   # already in
    ]
    for p in pool:
        p["vocal_section_idx"] = p["inst_section_idx"] = None
    items[0]["vocal_section_idx"] = items[0]["inst_section_idx"] = None
    out = next_candidates(items, pool)
    ids = [r["vocal_song_id"] for r in out]
    assert 1 not in [r["item_id"] for r in out], "already in the set"
    assert ids[0] in (11, 7) and ids[-1] == 10
    swap = next(r for r in out if r["vocal_song_id"] == 7)
    assert swap["next_move"] == "bed_swap" and swap["next_why"].startswith("bed swap")


# ── the API ──────────────────────────────────────────────────────────────────

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
           for n in range(4)]
    side = lambda sid, bpm: {"song_id": sid, "title": f"S{sid}", "artist": "A",
                             "bpm": bpm, "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5}
    for v, b, bpm in ((ids[0], ids[1], 120.0), (ids[2], ids[3], 128.0), (ids[0], ids[3], 120.0)):
        models.upsert_candidate(side(v, bpm), side(b, bpm),
                                {"total": 0.8, "bpm_score": 1.0, "key_score": 1.0,
                                 "energy_score": 0.5, "timbre_score": 0.5})
    return TestClient(app), models, ids


def test_the_set_api_carries_the_curve_and_the_moves(client):
    c, models, ids = client
    s = c.post("/api/sets", json={"name": "Vol 1"}).json()
    for v, b in ((ids[0], ids[1]), (ids[2], ids[3])):
        s = c.post(f"/api/sets/{s['id']}/items",
                   json={"vocal_song_id": v, "inst_song_id": b}).json()
    assert s["tempo_plan"] is None
    s = c.patch(f"/api/sets/{s['id']}", json={"tempo_plan": {"start_bpm": 122, "end_bpm": 126}}).json()
    assert s["tempo_plan"] == {"start_bpm": 122, "end_bpm": 126}
    assert [l["bpm"] for l in s["flow"]["landings"]] == [122, 126]
    # Every item's recipe is re-priced at its point on the curve.
    assert s["items"][0]["recipe"]["target_bpm"] == 122
    assert s["items"][1]["recipe"]["vocal_rate"] == pytest.approx(126 / 128, abs=1e-4)

    second = s["items"][1]["item_id"]
    s = c.put(f"/api/sets/{s['id']}/items/{second}/transition",
              json={"type": "echo_out"}).json()
    assert s["flow"]["transitions"][0]["move"]["type"] == "echo_out"
    assert c.put(f"/api/sets/{s['id']}/items/{second}/transition",
                 json={"type": "teleport"}).status_code == 400
    s = c.put(f"/api/sets/{s['id']}/items/{second}/transition", json={}).json()
    assert s["flow"]["transitions"][0]["move"]["suggested"]

    nxt = c.get(f"/api/sets/{s['id']}/next").json()["candidates"]
    assert [(r["vocal_song_id"], r["inst_song_id"]) for r in nxt] == [(ids[0], ids[3])]

    csv = c.get(f"/api/sets/{s['id']}/export?format=csv").text
    assert "move_in" in csv.splitlines()[0]
    s = c.patch(f"/api/sets/{s['id']}", json={"clear_tempo": True}).json()
    assert s["tempo_plan"] is None and s["flow"]["landings"][0]["bpm"] == 120


def test_migration_adds_the_set_columns_to_an_old_database(tmp_path, monkeypatch):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE sets (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            note TEXT DEFAULT '', created_at TEXT, updated_at TEXT, UNIQUE(name));
        CREATE TABLE set_items (id INTEGER PRIMARY KEY AUTOINCREMENT, set_id INTEGER NOT NULL,
            position INTEGER NOT NULL, vocal_song_id INTEGER NOT NULL, inst_song_id INTEGER NOT NULL,
            vocal_section INTEGER, inst_section INTEGER, payload_json TEXT, added_at TEXT);
        INSERT INTO sets(name) VALUES ('kept');
    """)
    conn.commit(); conn.close()
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    import database.models as models
    models.init_db(db)
    conn = sqlite3.connect(db)
    assert "tempo_plan_json" in {r[1] for r in conn.execute("PRAGMA table_info(sets)")}
    assert "transition_json" in {r[1] for r in conn.execute("PRAGMA table_info(set_items)")}
    assert conn.execute("SELECT name FROM sets").fetchone()[0] == "kept"


# ── the screen ───────────────────────────────────────────────────────────────

def test_the_screen_offers_exactly_the_servers_moves():
    from matcher.setflow import TRANSITIONS
    src = (FRONT / "components" / "SetScreen.jsx").read_text(encoding="utf-8")
    block = src[src.index("export const MOVES"):]
    block = block[:block.index("];")]
    assert set(re.findall(r'\["(\w+)",', block)) == set(TRANSITIONS)


def test_the_screen_has_the_curve_the_timeline_and_what_comes_next():
    src = (FRONT / "components" / "SetScreen.jsx").read_text(encoding="utf-8")
    for piece in ("<TempoControl plan={data.tempo_plan}", "<SetTimeline items={data.items}",
                  "<NextUp setId={data.id}", "api.setItemTransition", "api.getSetNext"):
        assert piece in src, piece
    css = (FRONT / "styles.css").read_text(encoding="utf-8")
    for cls in (".set-move", ".set-tempo", ".set-timeline", ".set-tl-block", ".set-next-row"):
        assert cls in css, cls
    # Studio lays each mashup at its own point on the curve.
    assert "targetBpm: c.set_bpm" in (FRONT / "App.jsx").read_text(encoding="utf-8")
    assert "const at = p.targetBpm || bpm" in (FRONT / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
