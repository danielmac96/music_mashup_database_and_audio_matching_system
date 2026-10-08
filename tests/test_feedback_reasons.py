"""Why a pair got its star.

A 1-star says the pair failed; it does not say which term failed it. Reasons
are picked on the card after rating and stored beside the verdict — never
instead of it, never changing it — so Phase 3 can check each section term
against the faults people actually hear.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FRONT = ROOT / "frontend" / "src"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import importlib
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.routes.mashups as routes
    importlib.reload(routes)
    import api.server as server
    importlib.reload(server)
    from fastapi.testclient import TestClient
    return TestClient(server.app), models


KEY = {"vocal_song_id": 1, "inst_song_id": 2, "vocal_section": 3, "inst_section": 4}


def test_a_reason_needs_a_rating(client):
    c, _ = client
    r = c.post("/api/mashups/feedback/reasons", json={**KEY, "reasons": ["boring"]})
    assert r.status_code == 404


def test_reasons_sit_beside_the_verdict(client):
    c, models = client
    assert c.post("/api/mashups/feedback", json={**KEY, "rating": 1}).status_code == 200
    r = c.post("/api/mashups/feedback/reasons",
               json={**KEY, "reasons": ["key_clash", "bass_mud", "key_clash"]})
    assert r.status_code == 200 and r.json()["reasons"] == ["key_clash", "bass_mud"]
    [row] = c.get("/api/mashups/feedback").json()["feedback"]
    assert row["reasons"] == ["key_clash", "bass_mud"]
    assert row["verdict"] == "no" and row["rating"] == 1

    # Re-rating keeps the reasons (the upsert never touches them) …
    c.post("/api/mashups/feedback", json={**KEY, "rating": 2})
    [row] = c.get("/api/mashups/feedback").json()["feedback"]
    assert row["reasons"] == ["key_clash", "bass_mud"] and row["rating"] == 2
    # … an empty list clears them, and clearing the star takes them with it.
    c.post("/api/mashups/feedback/reasons", json={**KEY, "reasons": []})
    assert c.get("/api/mashups/feedback").json()["feedback"][0]["reasons"] == []
    c.request("DELETE", "/api/mashups/feedback", json=KEY)
    assert c.get("/api/mashups/feedback").json()["feedback"] == []


def test_unknown_reasons_are_refused(client):
    c, _ = client
    c.post("/api/mashups/feedback", json={**KEY, "rating": 5})
    r = c.post("/api/mashups/feedback/reasons", json={**KEY, "reasons": ["vibes"]})
    assert r.status_code == 400


def test_the_ui_offers_exactly_the_stored_vocabulary(client):
    _, models = client
    model = (FRONT / "components" / "pairs" / "pairModel.js").read_text(encoding="utf-8")
    block = model[model.index("export const VERDICT_REASONS"):]
    block = block[:block.index("];")]
    ui = dict(re.findall(r'key: "(\w+)",[^}]*tone: "(good|bad)"', block))
    assert ui == models.FEEDBACK_REASONS


def test_the_focused_rated_card_asks_why():
    card = (FRONT / "components" / "PairCard.jsx").read_text(encoding="utf-8")
    assert "focused && rating && onReason" in card
    assert "reasonsFor(rating)" in card
    dock = (FRONT / "components" / "PairDock.jsx").read_text(encoding="utf-8")
    assert "ratings.toggleReason(c, r)" in dock
    hook = (FRONT / "hooks" / "useRatings.js").read_text(encoding="utf-8")
    assert "api.savePairReasons" in hook and "reasonsOf" in hook
