"""/api/prefs: the Library view and filter presets — the response contracts the
redesigned Library builds against."""
import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.routes.prefs as prefs
    importlib.reload(prefs)
    import api.server as server
    importlib.reload(server)
    return TestClient(server.app), models


EMPTY = {"columns": [], "order": [], "sort": {"id": None, "dir": None}}


# ── library view ─────────────────────────────────────────────────────────────

def test_nothing_stored_is_the_empty_view(app):
    client, _ = app
    assert client.get("/api/prefs/library-view").json() == EMPTY


def test_put_round_trips_the_exact_shape(app):
    client, models = app
    body = {"columns": [{"id": "title", "visible": True, "width": 320},
                        {"id": "attr:lufs", "visible": False, "width": None}],
            "order": ["attr:lufs", "title"],
            "sort": {"id": "title", "dir": "desc"}}
    r = client.put("/api/prefs/library-view", json=body)
    assert r.status_code == 200 and r.json() == body
    assert client.get("/api/prefs/library-view").json() == body
    assert models.get_pref("library_view") == body      # app_prefs.library_view


def test_contract_key_sets_and_types(app):
    client, _ = app
    client.put("/api/prefs/library-view",
               json={"columns": [{"id": "bpm"}], "sort": {"id": "bpm", "dir": "asc"}})
    view = client.get("/api/prefs/library-view").json()
    assert set(view) == {"columns", "order", "sort"}
    assert set(view["sort"]) == {"id", "dir"}
    col = view["columns"][0]
    assert set(col) == {"id", "visible", "width"}
    assert col == {"id": "bpm", "visible": True, "width": None}
    assert view["order"] == []


def test_unknown_keys_are_ignored(app):
    client, _ = app
    r = client.put("/api/prefs/library-view", json={
        "columns": [{"id": "bpm", "visible": True, "width": 90, "pinned": True}],
        "order": ["bpm"], "sort": {"id": None, "dir": None, "nulls": "last"},
        "density": "compact"})
    assert r.status_code == 200
    assert r.json() == {"columns": [{"id": "bpm", "visible": True, "width": 90}],
                        "order": ["bpm"], "sort": {"id": None, "dir": None}}


@pytest.mark.parametrize("body", [
    [],
    {"columns": "title"},
    {"columns": ["title"]},
    {"columns": [{"visible": True}]},
    {"columns": [{"id": "a", "visible": "yes"}]},
    {"columns": [{"id": "a", "width": "wide"}]},
    {"order": [1, 2]},
    {"sort": {"id": "a", "dir": "up"}},
    {"sort": "title"},
])
def test_a_wrong_shape_is_400(app, body):
    client, _ = app
    assert client.put("/api/prefs/library-view", json=body).status_code == 400
    assert client.get("/api/prefs/library-view").json() == EMPTY


def test_columns_are_capped_at_200(app):
    client, _ = app
    cols = [{"id": f"c{i}"} for i in range(201)]
    assert client.put("/api/prefs/library-view", json={"columns": cols}).status_code == 400
    assert client.put("/api/prefs/library-view",
                      json={"columns": cols[:200]}).status_code == 200


def test_a_corrupt_stored_view_reads_as_empty(app):
    client, models = app
    models.set_pref("library_view", {"columns": "nope"})
    assert client.get("/api/prefs/library-view").json() == EMPTY


def test_attribute_visibility_is_untouched(app):
    client, models = app
    before = client.get("/api/analysis/attributes").json()["visibility"]
    client.put("/api/prefs/library-view", json={"columns": [{"id": "attr:lufs"}]})
    assert client.get("/api/analysis/attributes").json()["visibility"] == before
    assert models.get_pref("attribute_visibility") is None


# ── filter presets ───────────────────────────────────────────────────────────

def test_presets_start_empty(app):
    client, _ = app
    assert client.get("/api/prefs/filter-presets").json() == {"presets": []}


def test_create_list_delete(app):
    client, models = app
    filters = {"genre": "House", "bpm": [120, 130], "nested": {"a": None}}
    r = client.post("/api/prefs/filter-presets",
                    json={"name": "Peak time", "filters": filters})
    assert r.status_code == 200
    preset = r.json()
    assert set(preset) == {"id", "name", "filters", "created_at"}
    assert isinstance(preset["id"], str) and preset["id"]
    assert preset["name"] == "Peak time" and preset["filters"] == filters
    from datetime import datetime
    datetime.fromisoformat(preset["created_at"])

    assert client.get("/api/prefs/filter-presets").json() == {"presets": [preset]}
    assert "presets" in models.get_pref("filter_presets")

    assert client.delete(f"/api/prefs/filter-presets/{preset['id']}").status_code == 200
    assert client.get("/api/prefs/filter-presets").json() == {"presets": []}
    assert client.delete(f"/api/prefs/filter-presets/{preset['id']}").status_code == 404


def test_names_are_unique_case_insensitively(app):
    client, _ = app
    assert client.post("/api/prefs/filter-presets",
                       json={"name": "Warm Up", "filters": {}}).status_code == 200
    assert client.post("/api/prefs/filter-presets",
                       json={"name": "  warm up ", "filters": {"a": 1}}).status_code == 409
    assert len(client.get("/api/prefs/filter-presets").json()["presets"]) == 1


def test_filters_over_16kb_are_refused(app):
    client, _ = app
    r = client.post("/api/prefs/filter-presets",
                    json={"name": "big", "filters": {"x": "y" * (16 * 1024)}})
    assert r.status_code == 413
    assert client.get("/api/prefs/filter-presets").json() == {"presets": []}


@pytest.mark.parametrize("body", [
    {"filters": {}}, {"name": "  ", "filters": {}}, {"name": "a"},
    {"name": "a", "filters": [1]}, "nope",
])
def test_a_bad_preset_is_400(app, body):
    client, _ = app
    assert client.post("/api/prefs/filter-presets", json=body).status_code == 400
