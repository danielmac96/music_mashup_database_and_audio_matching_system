"""Migration 11 removes Sets (master plan 3D): a verified JSON backup first,
and with any failure the tables stay where they were."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _at_version_10(path: Path) -> sqlite3.Connection:
    """A database as Phase 2 left it, holding two sets."""
    from database import schema

    conn = sqlite3.connect(path)
    keep = schema.MIGRATIONS
    schema.MIGRATIONS = tuple(m for m in keep if m.version <= 10)
    try:
        schema.apply(conn)
    finally:
        schema.MIGRATIONS = keep
    assert schema.current_version(conn) == 10
    conn.execute("INSERT INTO sets(id, name, note, tempo_plan_json) VALUES "
                 "(1, 'Friday', 'warm-up', '{\"start_bpm\": 120}'), (2, 'Empty', '', NULL)")
    payload = json.dumps({"vocal_title": "V", "inst_title": "B", "score_total": 0.71})
    conn.execute("INSERT INTO set_items(set_id, position, vocal_song_id, inst_song_id,"
                 " vocal_section, inst_section, payload_json, kind) VALUES "
                 "(1, 0, 10, 20, 1002, 1004, ?, 'pair')", (payload,))
    conn.execute("INSERT INTO set_items(set_id, position, kind, label, duration_secs) "
                 "VALUES (1, 1, 'break', 'mic', 12.0)")
    conn.execute("INSERT INTO set_items(set_id, position, kind, song_id) "
                 "VALUES (1, 2, 'track', 30)")
    conn.commit()
    return conn


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_upgrade_writes_a_verified_backup_then_drops_the_tables(tmp_path):
    from database import schema

    conn = _at_version_10(tmp_path / "lib.db")
    try:
        assert schema.apply(conn) == [11, 12]
        assert not ({"sets", "set_items"} & _tables(conn))
        files = list((tmp_path / "exports").glob("sets_backup_*.json"))
        assert len(files) == 1
        doc = json.loads(files[0].read_text(encoding="utf-8"))
        assert doc["counts"] == {"sets": 2, "set_items": 3}
        friday = next(s for s in doc["sets"] if s["name"] == "Friday")
        assert [i["kind"] for i in friday["items"]] == ["pair", "break", "track"]
        # The frozen pair row comes back parsed, with its four-id key.
        pair = friday["items"][0]
        assert pair["payload"]["score_total"] == 0.71
        assert (pair["vocal_song_id"], pair["inst_song_id"],
                pair["vocal_section"], pair["inst_section"]) == (10, 20, 1002, 1004)
        pref = json.loads(conn.execute("SELECT value FROM app_prefs WHERE key=?",
                                       (schema.SETS_BACKUP_PREF,)).fetchone()[0])
        assert pref == {"path": str(files[0]), "sets": 2, "items": 3, "seen": False}
        # Idempotent: nothing left to do, and no second file.
        schema._remove_sets(conn)
        assert len(list((tmp_path / "exports").glob("sets_backup_*.json"))) == 1
    finally:
        conn.close()


def test_a_failed_backup_leaves_the_tables_in_place(tmp_path, monkeypatch):
    from database import schema

    conn = _at_version_10(tmp_path / "lib.db")
    try:
        # exports/ is a file, so the backup cannot be written.
        (tmp_path / "exports").write_text("in the way", encoding="utf-8")
        with pytest.raises(RuntimeError, match=r"migration 11 \(remove_sets\)"):
            schema.apply(conn)
        assert schema.current_version(conn) == 10
        assert {"sets", "set_items"} <= _tables(conn)
        assert conn.execute("SELECT COUNT(*) FROM set_items").fetchone()[0] == 3
    finally:
        conn.close()


def test_a_short_backup_is_refused_and_nothing_is_dropped(tmp_path, monkeypatch):
    """The file is read back and counted against the tables: a write that
    lost a row aborts the drop."""
    from database import schema

    real_dump = json.dump

    def lossy(doc, fh, **kw):
        doc = {**doc, "sets": [{**s, "items": s["items"][:-1]} for s in doc["sets"]]}
        return real_dump(doc, fh, **kw)

    conn = _at_version_10(tmp_path / "lib.db")
    try:
        monkeypatch.setattr(json, "dump", lossy)
        with pytest.raises(RuntimeError, match="nothing was dropped"):
            schema.apply(conn)
        assert schema.current_version(conn) == 10
        assert {"sets", "set_items"} <= _tables(conn)
    finally:
        conn.close()


def test_a_library_without_sets_writes_no_file(tmp_path):
    from database import schema

    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        schema.apply(conn)
        assert not ({"sets", "set_items"} & _tables(conn))
        assert not (tmp_path / "exports").exists()
    finally:
        conn.close()


def test_the_ui_is_told_where_the_backup_is_once(tmp_path, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    from database import schema

    db = tmp_path / "lib.db"
    _at_version_10(db).close()
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()                                  # runs migration 11
    import api.server as server
    importlib.reload(server)
    client = TestClient(server.app)
    got = client.get("/api/prefs/sets-backup").json()
    assert got["sets"] == 2 and got["items"] == 3
    assert Path(got["path"]).parent == tmp_path / "exports"
    assert client.post("/api/prefs/sets-backup/seen").json() == {"ok": True}
    assert client.get("/api/prefs/sets-backup").json() == {"path": None}
    assert models.get_pref(schema.SETS_BACKUP_PREF)["path"] == got["path"], "the record is kept"
    # No Sets route survives.
    assert client.get("/api/sets").status_code in (404, 405) or \
        "text/html" in client.get("/api/sets").headers.get("content-type", "")


# Identifiers only the Sets feature used. SoundCloud's own "sets" (its
# playlists) are a different thing and stay.
_GONE = ("setflow", "set_items", "SetScreen", "useSets", "/api/sets", "addToSet",
         "AddToSet", "render.transition", "render/transition", "+ Set<",
         "getSetNext", "chainToStudio", "nextForSet", "[\"sets\", \"Sets\"", "a +set")


def test_no_route_component_or_module_of_sets_remains():
    root = Path(__file__).parent.parent
    for gone in ("api/routes/sets.py", "matcher/setflow.py", "render/transition.py",
                 "frontend/src/components/SetScreen.jsx", "frontend/src/hooks/useSets.js"):
        assert not (root / gone).exists(), gone
    hits = []
    for folder in ("api", "matcher", "render", "analysis", "frontend/src", "scripts"):
        for f in (root / folder).rglob("*"):
            if f.suffix not in (".py", ".js", ".jsx", ".css") or "node_modules" in f.parts:
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            hits += [f"{f.relative_to(root)}: {g}" for g in _GONE if g in text]
    # database/models.py only names the tables in its key comments; schema.py
    # drops them. Neither may define a Sets helper.
    models = (root / "database" / "models.py").read_text(encoding="utf-8")
    for fn in ("def create_set", "def get_set", "def add_set_item", "def list_sets"):
        assert fn not in models, fn
    assert not hits, hits
