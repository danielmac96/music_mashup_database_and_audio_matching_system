"""The read-only database browser (⚙ Settings → database).

Every whitelisted table must page without error. It used to ORDER BY id, and
track_excluded has no id column (it is keyed by song_id), so browsing it
answered 500 — found by the persona simulation."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "browse.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import importlib
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    import api.routes.database as routes
    importlib.reload(routes)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/db")
    models.init_db(models.DB_PATH)
    sid = models.upsert_song("T", "A", "https://sc/t", 200, "", db_path=models.DB_PATH)
    models.exclude_track(sid, db_path=models.DB_PATH)
    return TestClient(app), routes


def test_every_browsable_table_pages(client):
    c, routes = client
    names = [t["name"] for t in c.get("/api/db/tables").json()["tables"]]
    assert "track_excluded" in names
    for name in routes._TABLES:
        r = c.get(f"/api/db/tables/{name}?limit=5")
        assert r.status_code == 200, (name, r.text)
    rows = c.get("/api/db/tables/track_excluded").json()["rows"]
    assert len(rows) == 1


def test_an_unknown_table_is_a_404(client):
    c, _ = client
    assert c.get("/api/db/tables/sqlite_master").status_code == 404
