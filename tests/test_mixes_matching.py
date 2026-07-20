"""Mixes manual-matching layer: schema migration, re-import preservation,
role/match assignment API. No network, no audio stack.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh config + models bound to a scratch DB, plus a mixes-router client."""
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))

    import config
    import database.models as models
    for mod in (config, models):
        importlib.reload(mod)
    models.init_db()

    import api.routes.mixes as mixes_routes
    importlib.reload(mixes_routes)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(mixes_routes.router, prefix="/api/mixes")
    return {"client": TestClient(app), "models": models,
            "db": tmp_path / "mashup.db"}


PASTE = (
    "Test Mix Vol 1\n"
    "1. [0:00] Kanye West - Stronger\n"
    "w/ [0:45] Whitney Houston - I Wanna Dance With Somebody\n"
    "2. [2:10] Avicii - Levels\n"
    "w/ ID - ID\n"
    "3. [4:05] Zedd & Grey - The Middle (Dzeko Remix)\n"
)


def _import(env, content=PASTE, url="https://example.com/test-mix"):
    res = env["client"].post("/api/mixes/import-paste",
                             json={"content": content, "url": url})
    assert res.status_code == 200, res.text
    return res.json()


# ── schema migration (Checkpoint 2) ───────────────────────────────────────────

def test_migration_upgrades_pre_matching_db(tmp_path, monkeypatch):
    """A DB created with the pre-matching mix schema gains the new columns,
    the vocal-uniqueness index, and keeps its rows."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE mixes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT,
            source_url TEXT UNIQUE, dj TEXT, import_method TEXT,
            raw_snapshot_path TEXT, imported_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE mix_tracks (id INTEGER PRIMARY KEY AUTOINCREMENT,
            mix_id INTEGER NOT NULL, entry_index INTEGER,
            position INTEGER NOT NULL, is_overlay INTEGER DEFAULT 0,
            artist TEXT, title TEXT, cue_secs REAL, link_url TEXT,
            link_platform TEXT, resolve_status TEXT DEFAULT 'unresolved',
            song_id INTEGER, UNIQUE(mix_id, entry_index, position));
        CREATE TABLE mashup_pairs (id INTEGER PRIMARY KEY AUTOINCREMENT,
            mix_id INTEGER NOT NULL, inst_mix_track_id INTEGER NOT NULL,
            vocal_mix_track_id INTEGER NOT NULL, cue_secs REAL,
            UNIQUE(inst_mix_track_id, vocal_mix_track_id));
        INSERT INTO mixes (title, source_url) VALUES ('old', 'u://m');
        INSERT INTO mix_tracks (mix_id, position, artist, title) VALUES (1, 0, 'A', 'B');
        INSERT INTO mashup_pairs (mix_id, inst_mix_track_id, vocal_mix_track_id)
            VALUES (1, 1, 2);
        -- a historically double-assigned vocal: migration must dedupe it
        INSERT INTO mashup_pairs (mix_id, inst_mix_track_id, vocal_mix_track_id)
            VALUES (1, 3, 2);
    """)
    conn.commit()
    conn.close()

    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    import database.models as models
    for mod in (config, models):
        importlib.reload(mod)
    conn = models.get_conn()

    track_cols = {r[1] for r in conn.execute("PRAGMA table_info(mix_tracks)")}
    assert {"raw_label", "is_id", "remixer", "mashup_parts",
            "parse_confidence", "role", "role_assigned_at"} <= track_cols
    pair_cols = {r[1] for r in conn.execute("PRAGMA table_info(mashup_pairs)")}
    assert {"origin", "created_at"} <= pair_cols

    # data survived; duplicate-vocal pair deduped to the earliest; index exists
    assert conn.execute("SELECT COUNT(*) FROM mix_tracks").fetchone()[0] == 1
    pairs = conn.execute(
        "SELECT inst_mix_track_id FROM mashup_pairs").fetchall()
    assert [p[0] for p in pairs] == [1]
    idx = {r[1] for r in conn.execute("PRAGMA index_list(mashup_pairs)")}
    assert "ux_mashuppairs_vocal" in idx
    conn.close()


# ── import persists the new parser fields ─────────────────────────────────────

def test_import_persists_parser_fields(env):
    mix = _import(env)
    by_title = {t["title"]: t for t in mix["tracks"]}
    zedd = by_title["The Middle (Dzeko Remix)"]
    assert zedd["raw_label"] == "3. [4:05] Zedd & Grey - The Middle (Dzeko Remix)"
    assert zedd["remixer"] == "Dzeko"
    assert zedd["parse_confidence"] == 1.0
    assert by_title["ID"]["is_id"] == 1
    # parsed 'w/' pairs seeded with origin='parsed'
    conn = env["models"].get_conn()
    origins = {r[0] for r in conn.execute(
        "SELECT origin FROM mashup_pairs WHERE mix_id=?", (mix["id"],))}
    conn.close()
    assert origins == {"parsed"}


# ── re-import preservation ────────────────────────────────────────────────────

def test_reimport_preserves_links_roles_and_manual_pairs(env):
    mix = _import(env)
    tracks = {t["title"]: t for t in mix["tracks"]}
    stronger, levels = tracks["Stronger"], tracks["Levels"]
    whitney = tracks["I Wanna Dance With Somebody"]

    # user work after first import: a resolved link, roles, one manual match
    env["client"].post(f"/api/mixes/tracks/{stronger['id']}/resolve",
                       json={"url": "https://soundcloud.com/kw/stronger"})
    conn = env["models"].get_conn()
    conn.execute("UPDATE mix_tracks SET role='instrumental', "
                 "role_assigned_at=datetime('now') WHERE id=?", (levels["id"],))
    conn.execute("UPDATE mix_tracks SET role='vocal' WHERE id=?", (whitney["id"],))
    conn.execute("DELETE FROM mashup_pairs WHERE vocal_mix_track_id=?",
                 (whitney["id"],))
    conn.execute("INSERT INTO mashup_pairs (mix_id, inst_mix_track_id, "
                 "vocal_mix_track_id, origin) VALUES (?,?,?,'manual')",
                 (mix["id"], levels["id"], whitney["id"]))
    conn.commit()
    conn.close()

    # re-import the same URL with one line edited and one appended
    edited = PASTE.replace("(Dzeko Remix)", "(Dzeko Extended Remix)") + \
        "4. Fisher - Losing It\n"
    mix2 = _import(env, content=edited)
    assert mix2["id"] == mix["id"]
    t2 = {t["title"]: t for t in mix2["tracks"]}

    # resolved link survived
    assert t2["Stronger"]["link_url"] == "https://soundcloud.com/kw/stronger"
    assert t2["Stronger"]["resolve_status"] == "manual"
    # roles survived
    assert t2["Levels"]["role"] == "instrumental"
    assert t2["I Wanna Dance With Somebody"]["role"] == "vocal"
    # the edited line is a new entry (no carried state), the appended one exists
    assert t2["The Middle (Dzeko Extended Remix)"]["role"] == "unassigned"
    assert "Losing It" in t2

    # the manual pair survived, still manual, remapped to the new track ids
    conn = env["models"].get_conn()
    pair = conn.execute(
        "SELECT * FROM mashup_pairs WHERE origin='manual' AND mix_id=?",
        (mix["id"],)).fetchone()
    conn.close()
    assert pair is not None
    assert pair["inst_mix_track_id"] == t2["Levels"]["id"]
    assert pair["vocal_mix_track_id"] == t2["I Wanna Dance With Somebody"]["id"]


def test_vocal_uniqueness_enforced(env):
    mix = _import(env)
    tracks = {t["title"]: t for t in mix["tracks"]}
    conn = env["models"].get_conn()
    with pytest.raises(sqlite3.IntegrityError):
        # Whitney already rides on Stronger via the parsed pair — a second bed
        # for the same vocal must be rejected by ux_mashuppairs_vocal.
        conn.execute(
            "INSERT INTO mashup_pairs (mix_id, inst_mix_track_id, "
            "vocal_mix_track_id, origin) VALUES (?,?,?,'manual')",
            (mix["id"], tracks["Levels"]["id"],
             tracks["I Wanna Dance With Somebody"]["id"]))
    conn.close()


# ── assignments endpoint (Phase 3) ────────────────────────────────────────────

def test_assignments_bulk_roles_and_matches(env):
    mix = _import(env)
    t = {x["title"]: x for x in mix["tracks"]}
    res = env["client"].post(f"/api/mixes/{mix['id']}/assignments", json={
        "roles": [
            {"track_id": t["Levels"]["id"], "role": "instrumental"},
            {"track_id": t["Stronger"]["id"], "role": "instrumental"},
        ],
        "matches": [
            {"vocal_track_id": t["I Wanna Dance With Somebody"]["id"],
             "inst_track_id": t["Levels"]["id"]},
        ],
    })
    assert res.status_code == 200, res.text
    out = res.json()
    t2 = {x["title"]: x for x in out["tracks"]}
    assert t2["Levels"]["role"] == "instrumental"
    # match forced the vocal role on the dragged track
    assert t2["I Wanna Dance With Somebody"]["role"] == "vocal"
    # the vocal was re-homed: parsed pair (on Stronger) replaced by the manual one
    manual = [p for p in out["pairs"] if p["origin"] == "manual"]
    assert len(manual) == 1
    assert manual[0]["inst_mix_track_id"] == t["Levels"]["id"]
    assert all(p["vocal_mix_track_id"] != t["I Wanna Dance With Somebody"]["id"]
               for p in out["pairs"] if p["origin"] == "parsed")


def test_assignments_unmatch_and_validation(env):
    mix = _import(env)
    t = {x["title"]: x for x in mix["tracks"]}
    whit = t["I Wanna Dance With Somebody"]["id"]

    # unmatch: null bed clears the vocal's pair (parsed one included)
    res = env["client"].post(f"/api/mixes/{mix['id']}/assignments", json={
        "matches": [{"vocal_track_id": whit, "inst_track_id": None}]})
    assert res.status_code == 200
    assert all(p["vocal_mix_track_id"] != whit for p in res.json()["pairs"])

    # foreign track id → 400, self-match → 400, bad role → 400
    assert env["client"].post(f"/api/mixes/{mix['id']}/assignments", json={
        "roles": [{"track_id": 99999, "role": "vocal"}]}).status_code == 400
    assert env["client"].post(f"/api/mixes/{mix['id']}/assignments", json={
        "matches": [{"vocal_track_id": whit, "inst_track_id": whit}]
    }).status_code == 400
    assert env["client"].post(f"/api/mixes/{mix['id']}/assignments", json={
        "roles": [{"track_id": whit, "role": "lead-guitar"}]}).status_code == 400


def test_role_change_away_drops_dependent_manual_matches(env):
    mix = _import(env)
    t = {x["title"]: x for x in mix["tracks"]}
    whit, levels = t["I Wanna Dance With Somebody"]["id"], t["Levels"]["id"]
    env["client"].post(f"/api/mixes/{mix['id']}/assignments", json={
        "matches": [{"vocal_track_id": whit, "inst_track_id": levels}]})
    # demote the bed to unassigned → its manual match must go
    res = env["client"].post(f"/api/mixes/{mix['id']}/assignments", json={
        "roles": [{"track_id": levels, "role": "unassigned"}]})
    assert res.status_code == 200
    assert all(p["origin"] != "manual" for p in res.json()["pairs"])


def test_import_paste_idempotent_on_url(env):
    one = _import(env)
    two = _import(env)
    assert one["id"] == two["id"]
    assert one["track_count"] == two["track_count"]
    conn = env["models"].get_conn()
    n_mixes = conn.execute("SELECT COUNT(*) FROM mixes").fetchone()[0]
    conn.close()
    assert n_mixes == 1
