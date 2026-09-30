import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

TAGS = {"genre": [{"label": "Electronic---Tropical House", "p": 0.41},
                  {"label": "Electronic---House", "p": 0.2}],
        "genre_parent": "Electronic", "voice": 0.9, "female": 0.7, "danceable": 0.8,
        "tonal": 0.6, "bright": 0.4,
        "mood": {"happy": 0.5, "sad": 0.1, "aggressive": 0.2, "relaxed": 0.3,
                 "party": 0.9, "acoustic": 0.05, "electronic": 0.95},
        "moodtheme": [{"label": "summer", "p": 0.3}],
        "instruments": [{"label": "synthesizer", "p": 0.7}]}
FULL = {"bpm": 128.0, "camelot": "8A", "key": "A", "mode": "minor", "lufs": -7.5,
        "tags_json": json.dumps(TAGS), "analyzer": "essentia"}
VOCALS = {"melody_json": json.dumps({"median_midi": 64.2, "range_st": 11.5, "voiced": 0.6}),
          "stem_quality": 0.93}
INST = {"residual_vocal_ratio": 0.04, "stem_quality": 0.88}


def test_ids_are_unique_and_every_attribute_has_a_known_category():
    from analysis.attributes import ATTRIBUTES, CATEGORIES
    ids = [a.id for a in ATTRIBUTES]
    assert len(ids) == len(set(ids))
    assert all(a.category in CATEGORIES for a in ATTRIBUTES)
    assert all(len(a.short) <= 6 for a in ATTRIBUTES)


def test_the_essentia_genre_is_style_never_genre():
    from analysis.attributes import BY_ID
    assert BY_ID["style"].short == "STYLE" and "genre" not in BY_ID


def test_values_come_from_the_three_rows():
    from analysis.attributes import extract
    v = extract(FULL, VOCALS, INST)
    assert v["bpm"] == 128.0 and v["key"] == "8A" and v["lufs"] == -7.5
    assert v["style"] == "Tropical House" and v["genre_parent"] == "Electronic"
    assert v["styles"][0] == {"label": "Tropical House", "p": 0.41}
    assert v["party"] == 0.9 and v["voice"] == 0.9 and v["female"] == 0.7
    assert v["sung_range"] == 11.5
    assert v["vocal_quality"] == 0.93 and v["bed_quality"] == 0.88
    assert v["residual_vocal"] == 0.04
    assert "instruments" in v and "moodtheme" in v


def test_unmeasured_is_absent_not_zero():
    from analysis.attributes import extract
    v = extract({"bpm": 120.0}, None, None)
    assert v == {"bpm": 120.0}
    assert extract(None, None, None) == {}


def test_distributions():
    from analysis.attributes import BY_ID, distribution
    d = distribution(BY_ID["lufs"], [-14.0, -10.0, -8.0, -7.0, -6.0])
    assert d["min"] == -14.0 and d["max"] == -6.0 and d["median"] == -8.0
    assert sum(d["hist"]) == 5 and len(d["hist"]) == 12 and len(d["edges"]) == 13
    c = distribution(BY_ID["genre_parent"], ["Electronic"] * 3 + ["Pop"])
    assert c["top"][0] == {"label": "Electronic", "count": 3}
    t = distribution(BY_ID["instruments"], [[{"label": "synthesizer", "p": .7}],
                                             [{"label": "synthesizer", "p": .6}]])
    assert t["top"][0] == {"label": "synthesizer", "count": 2}
    assert distribution(BY_ID["lufs"], []) == {}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "m.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "s"))
    import config, database.models as models, api.routes.tracks, api.routes.analysis
    for m in (config, models, api.routes.tracks, api.routes.analysis):
        importlib.reload(m)
    models.init_db()
    for n, lufs in enumerate((-8.0, -6.0), start=1):
        sid = models.upsert_song(title=f"T{n}", artist="A", source_url=f"http://x/{n}")
        models.update_song_status(sid, "analysed")
        models.upsert_features(sid, "full", {"bpm": 128.0, "key": "A", "mode": "minor",
                                             "camelot": "8A"})
        models.update_features_extras(sid, "full", {"lufs": lufs, "tags_json": json.dumps(TAGS)})
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(api.routes.tracks.router, prefix="/api/tracks")
    app.include_router(api.routes.analysis.router, prefix="/api/analysis")
    return TestClient(app)


def test_tracks_carry_their_attributes(client):
    t = client.get("/api/tracks").json()["tracks"][0]
    assert t["attrs"]["style"] == "Tropical House" and t["attrs"]["lufs"] == -8.0


def test_the_catalogue_reports_coverage_and_distribution(client):
    body = client.get("/api/analysis/attributes").json()
    lufs = next(a for a in body["attributes"] if a["id"] == "lufs")
    assert lufs["coverage"] == {"n": 2, "total": 2} and lufs["dist"]["median"] == -7.0
    tuning = next(a for a in body["attributes"] if a["id"] == "tuning_hz")
    assert tuning["coverage"]["n"] == 0 and tuning["dist"] == {}
    assert body["visibility"]["library"] == [] and "style" in body["visibility"]["detail"]


def test_visibility_is_saved_and_validated(client):
    r = client.put("/api/analysis/attributes/visibility",
                   json={"library": ["style", "lufs"], "detail": ["voice"]})
    assert r.status_code == 200
    assert client.get("/api/analysis/attributes").json()["visibility"] == \
        {"library": ["style", "lufs"], "detail": ["voice"]}
    bad = client.put("/api/analysis/attributes/visibility", json={"library": ["nope"], "detail": []})
    assert bad.status_code == 400 and "nope" in bad.json()["detail"]
