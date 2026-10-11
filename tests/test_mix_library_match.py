"""A pasted or scraped tracklist links its rows to songs already in the library.

Found by the 2026-10-10 persona simulation: a set naming eight library tracks
parsed into "0 matched", every row "needs link" and every documented pair "not
in library" — rows only reached library songs through online Auto-link and
Ingest. matching runs at import, offline, on the import's own connection.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SET = """1. Lumen Park - Midnight Signal
w/ Cass Rowan - Fireproof (Acapella)
2. Two Rooms - Bassline Theory
3. Lumen Park - Midnight Signal (Kade Monroe Remix)
4. ID - ID
"""


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mixes
    importlib.reload(mixes)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(mixes.router, prefix="/api/mixes")
    return {"client": TestClient(app), "mixes": mixes, "models": models}


def _song(models, title, artist, slug):
    return models.upsert_song(title, artist, f"https://soundcloud.com/x/{slug}", 200,
                              status="analysed")


def _paste(env, content=SET, url="https://www.1001tracklists.com/tracklist/t/set.html"):
    res = env["client"].post("/api/mixes/import-paste", json={"content": content, "url": url})
    assert res.status_code == 200, res.text
    return res.json()


def _rows(mix):
    return {(t["artist"], t["title"]): t for t in mix["tracks"]}


def test_a_library_track_is_linked_at_import(env):
    m = env["models"]
    signal = _song(m, "Midnight Signal", "Lumen Park", "signal")
    mix = _paste(env)
    row = _rows(mix)[("Lumen Park", "Midnight Signal")]
    assert row["song_id"] == signal
    assert row["resolve_status"] == "library"
    assert row["trusted"] is True
    assert row["resolved_url"] == "https://soundcloud.com/x/signal"


def test_an_acapella_line_links_to_the_original(env):
    m = env["models"]
    fire = _song(m, "Fireproof", "Cass Rowan", "fire")
    mix = _paste(env)
    assert _rows(mix)[("Cass Rowan", "Fireproof (Acapella)")]["song_id"] == fire


def test_a_remix_never_links_to_the_original(env):
    m = env["models"]
    signal = _song(m, "Midnight Signal", "Lumen Park", "signal")
    mix = _paste(env)
    rows = _rows(mix)
    remix = rows[("Lumen Park", "Midnight Signal (Kade Monroe Remix)")]
    assert remix["song_id"] is None
    assert rows[("Lumen Park", "Midnight Signal")]["song_id"] == signal


def test_the_plain_original_wins_over_its_extended_mix(env):
    m = env["models"]
    fire = _song(m, "Fireproof", "Cass Rowan", "fire")
    _song(m, "Fireproof (Extended Mix)", "Cass Rowan", "fire-ext")
    mix = _paste(env)
    assert _rows(mix)[("Cass Rowan", "Fireproof (Acapella)")]["song_id"] == fire


def test_two_equal_candidates_leave_the_row_unlinked(env):
    m = env["models"]
    _song(m, "Bassline Theory (Extended Mix)", "Two Rooms", "a")
    _song(m, "Bassline Theory (Radio Edit)", "Two Rooms", "b")
    mix = _paste(env)
    assert _rows(mix)[("Two Rooms", "Bassline Theory")]["song_id"] is None


def test_id_rows_are_never_linked(env):
    m = env["models"]
    _song(m, "ID", "ID", "id")
    mix = _paste(env)
    ids = [t for t in mix["tracks"] if t["title"] == "ID"]
    assert ids and all(t["song_id"] is None for t in ids)


def test_documented_pairs_reach_the_engine_view(env):
    m = env["models"]
    _song(m, "Midnight Signal", "Lumen Park", "signal")
    _song(m, "Fireproof", "Cass Rowan", "fire")
    mix = _paste(env)
    pair = mix["pairs"][0]
    assert pair["vocal_song_id"] and pair["inst_song_id"]
    assert pair["engine_state"] != "not in library"


def test_find_in_library_links_rows_after_the_fact(env):
    m = env["models"]
    mix = _paste(env)
    assert _rows(mix)[("Two Rooms", "Bassline Theory")]["song_id"] is None
    theory = _song(m, "Bassline Theory", "Two Rooms", "theory")
    res = env["client"].post(f"/api/mixes/{mix['id']}/match-library")
    assert res.status_code == 200
    out = res.json()
    assert out["linked"] == 1
    assert _rows(out)[("Two Rooms", "Bassline Theory")]["song_id"] == theory
    assert env["client"].post("/api/mixes/9999/match-library").status_code == 404


def test_a_manual_link_is_kept_on_reimport(env):
    """Carry-over still wins: a row already pointed at a song keeps it."""
    m = env["models"]
    other = _song(m, "Something Else", "Nobody", "else")
    mix = _paste(env)
    row = _rows(mix)[("Two Rooms", "Bassline Theory")]
    conn = m.get_conn()
    conn.execute("UPDATE mix_tracks SET song_id=?, resolve_status='manual' WHERE id=?",
                 (other, row["id"]))
    conn.commit()
    conn.close()
    _song(m, "Bassline Theory", "Two Rooms", "theory")
    again = _paste(env)
    row = _rows(again)[("Two Rooms", "Bassline Theory")]
    assert row["song_id"] == other and row["resolve_status"] == "manual"


def test_matching_uses_only_the_callers_connection(env, monkeypatch):
    """§7: inside _persist_mix's write transaction, a helper that opened its own
    connection would block on the lock."""
    m, mixes = env["models"], env["mixes"]
    _song(m, "Midnight Signal", "Lumen Park", "signal")
    calls = []
    real = mixes.get_conn

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    conn = real()
    conn.execute("INSERT INTO mixes (title, source_url, import_method) VALUES ('t', NULL, 'paste')")
    mix_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("INSERT INTO mix_tracks (mix_id, entry_index, position, is_overlay, artist, title) "
                 "VALUES (?, 0, 0, 0, 'Lumen Park', 'Midnight Signal')", (mix_id,))
    monkeypatch.setattr(mixes, "get_conn", counting)
    assert mixes.match_library(conn, mix_id) == 1
    assert calls == []
    conn.commit()
    conn.close()


def test_library_links_count_as_trusted():
    from database.models import is_trusted_link
    assert is_trusted_link("library", None, None) is True


def test_the_board_offers_find_in_library():
    src = (ROOT / "frontend" / "src" / "components" / "MixImporter.jsx").read_text(encoding="utf-8")
    assert "api.matchMixToLibrary(detail.id)" in src and "Find in library" in src
    api = (ROOT / "frontend" / "src" / "api.js").read_text(encoding="utf-8")
    assert "/match-library" in api
