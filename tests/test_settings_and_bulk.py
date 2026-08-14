"""Settings exposure + bulk reprocessing.

Phases D and E added features that only exist on tracks analysed since, and
several knobs that only existed as module constants. Neither was reachable from
the app: the features needed a per-track ⟳ across the whole library, and the
knobs needed a text editor and a restart.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """An app on a throwaway DB *and* a throwaway settings file.

    The settings file matters: these tests save settings, and writing to the
    real ~/.config file would silently reconfigure the developer's own library."""
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    # config already supports this override for exactly this reason; writing to
    # the real ~/.config file would silently reconfigure the developer's library.
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "cfg"))

    # get_conn's default db_path binds at function definition, so the routes
    # would otherwise talk to whichever database was configured when the module
    # first imported. Reloading is the pattern the other route tests use.
    import config
    import database.models as models
    for mod in (config, models):
        importlib.reload(mod)
    models.init_db()

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import settings as settings_routes
    from api.routes import tracks as track_routes
    for mod in (settings_routes, track_routes):
        importlib.reload(mod)

    app = FastAPI()
    app.include_router(track_routes.router, prefix="/api/tracks")
    app.include_router(settings_routes.router, prefix="/api/settings")
    return TestClient(app), tmp_path / "t.db"


def _add(db_path, k, *, analysed=True, bands=False, quality=False,
         chroma=False, sections=True, drums=False):
    """A song with selectable generations of feature data present."""
    from database.models import (
        get_conn, replace_sections, upsert_features, upsert_song, upsert_stem,
        update_stem_quality,
    )
    sid = upsert_song(f"S{k}", "A", f"u://{k}", 200,
                      status="analysed" if analysed else "downloaded",
                      db_path=db_path)
    for stem in ("full", "vocals", "instrumental"):
        feats = {"bpm": 120.0, "key": "C", "mode": "major", "camelot": "8A",
                 "loudness_rms": 0.1, "energy": 0.5, "mfcc": [1.0] * 13}
        if bands:
            feats["band_energy"] = [0.125] * 8
        upsert_features(sid, stem, feats, db_path=db_path)
        upsert_stem(sid, stem, f"/x/{k}_{stem}.flac", db_path=db_path)
        if quality and stem != "full":
            update_stem_quality(sid, stem, {"quality": 0.8}, db_path=db_path)
    if drums:
        upsert_stem(sid, "drums", f"/x/{k}_drums.flac", db_path=db_path)
    if sections:
        sec = {"start_sec": 0.0, "end_sec": 30.0, "label": "chorus",
               "energy": 0.8, "vocal_presence": 0.8, "repetition": 2,
               "confidence": 0.9}
        if chroma:
            sec["chroma"] = [0.1] * 12
        replace_sections(sid, [sec], db_path=db_path)
    return sid


# ── Settings exposure ────────────────────────────────────────────────────────

def test_every_tunable_knob_is_reported(client):
    """If a knob is not in this payload the UI cannot render it, so it may as
    well not exist."""
    c, _ = client
    got = c.get("/api/settings").json()
    for key in ("stem_mode", "stem_separator", "effort_weight", "section_weight",
                "stem_quality_min", "bpm_max_diff", "key_min_score",
                "bpm_max_diff_model", "max_section_pairs", "match_weights"):
        assert key in got, f"{key} missing from /api/settings"
        assert "value" in got[key] and "source" in got[key]


def test_saving_a_knob_applies_without_a_restart(client):
    """The whole point of the live-read: turn a knob, re-score, hear the
    difference — not restart the server."""
    import config
    c, _ = client
    before = config.current_float("effort_weight")
    out = c.post("/api/settings", json={"effort_weight": 0.6}).json()
    assert out["saved"] and out["restart_required"] is False
    assert config.current_float("effort_weight") == pytest.approx(0.6)
    assert config.current_float("effort_weight") != before


def test_weights_do_not_have_to_add_up(client):
    """A user dragging five sliders should not have to make them sum to 1;
    un-normalised weights would rescale every score in the library and make the
    Min-match slider meaningless."""
    import config
    c, _ = client
    c.post("/api/settings", json={"match_weights": {
        "bpm_score": 2, "key_score": 2, "energy_score": 2,
        "timbre_score": 2, "collision_score": 2}})
    w = config.current_match_weights()
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(0.2) for v in w.values())


def test_all_zero_weights_fall_back_rather_than_zeroing_the_library(client):
    c, _ = client
    r = c.post("/api/settings", json={"match_weights": {
        "bpm_score": 0, "key_score": 0, "energy_score": 0,
        "timbre_score": 0, "collision_score": 0}})
    assert r.status_code == 400


@pytest.mark.parametrize("payload", [
    {"effort_weight": 5},
    {"effort_weight": -1},
    {"max_section_pairs": 99},
    {"stem_mode": "eight"},
    {"match_weights": {"bogus": 1}},
    {"match_weights": {"bpm_score": -1}},
])
def test_invalid_settings_are_rejected(client, payload):
    c, _ = client
    assert c.post("/api/settings", json=payload).status_code == 400


def test_four_stem_needs_demucs(client):
    """MDX is a two-stem model. Saving the pair together would leave the user
    with a setting that silently does nothing."""
    c, _ = client
    c.post("/api/settings", json={"stem_separator": "mdx"})
    r = c.post("/api/settings", json={"stem_mode": "four"})
    assert r.status_code == 400
    assert "demucs" in r.json()["detail"].lower()
    # Switching both at once is fine.
    assert c.post("/api/settings", json={"stem_mode": "four",
                                         "stem_separator": "demucs"}).status_code == 200


def test_env_pinned_knobs_report_their_source(client, monkeypatch):
    """A control the user cannot actually change must not look editable."""
    monkeypatch.setenv("MASHUP_EFFORT_WEIGHT", "0.9")
    c, _ = client
    got = c.get("/api/settings").json()
    assert got["effort_weight"]["source"] == "env"
    assert got["effort_weight"]["value"] == pytest.approx(0.9)


# ── Staleness ────────────────────────────────────────────────────────────────

def test_staleness_counts_each_missing_feature_group(client):
    """Reported per group so the UI can say WHAT is missing, not just how many."""
    c, db = client
    _add(db, 1, bands=True, quality=True, chroma=True)     # current
    _add(db, 2, bands=False, quality=True, chroma=True)     # pre-Phase-D bands
    _add(db, 3, bands=True, quality=True, chroma=False)     # pre-Phase-E chroma
    _add(db, 4, bands=True, quality=False, chroma=True)     # no stem quality

    got = c.get("/api/tracks/staleness").json()
    assert got["total_analysed"] == 4
    assert got["missing_band_energy"] == 1
    assert got["missing_section_chroma"] == 1
    assert got["missing_stem_quality"] == 1
    assert got["needs_analysis"] == 3      # only song 1 is current


def test_a_current_library_reports_nothing_stale(client):
    c, db = client
    _add(db, 1, bands=True, quality=True, chroma=True)
    got = c.get("/api/tracks/staleness").json()
    assert got["needs_analysis"] == 0


def test_tracks_with_no_sections_are_counted_separately(client):
    """Those need structure detection, not a re-analysis of something that never
    ran — conflating them would overstate the work."""
    c, db = client
    _add(db, 1, bands=True, quality=True, sections=False)
    got = c.get("/api/tracks/staleness").json()
    assert got["missing_sections"] == 1
    assert got["missing_section_chroma"] == 0


def test_four_stem_staleness_only_counts_when_four_stem_is_on(client):
    """A user who does not want four-stem must not be told their library needs
    hours of work."""
    c, db = client
    _add(db, 1, bands=True, quality=True, chroma=True, drums=False)
    assert c.get("/api/tracks/staleness").json()["missing_four_stems"] == 0

    c.post("/api/settings", json={"stem_mode": "four", "stem_separator": "demucs"})
    got = c.get("/api/tracks/staleness").json()
    assert got["stem_mode"] == "four"
    assert got["missing_four_stems"] == 1


# ── Bulk reprocessing ────────────────────────────────────────────────────────

def test_bulk_stale_queues_only_what_needs_it(client):
    """So adding one track does not mean re-analysing the whole library."""
    from database.models import get_conn
    c, db = client
    current = _add(db, 1, bands=True, quality=True, chroma=True)
    stale = _add(db, 2, bands=False, quality=True, chroma=True)

    r = c.post("/api/tracks/bulk", json={"action": "analyze", "scope": "stale"})
    assert r.status_code == 200
    assert r.json()["count"] == 1

    conn = get_conn(db)
    rows = {x["id"]: x["status"] for x in
            conn.execute("SELECT id, status FROM songs").fetchall()}
    conn.close()
    # Re-analysis rewinds to 'stemmed' — the stems are kept, only the analysis
    # stage re-runs.
    assert rows[stale] == "stemmed"
    assert rows[current] == "analysed"


def test_bulk_all_queues_everything(client):
    c, db = client
    _add(db, 1, bands=True, quality=True, chroma=True)
    _add(db, 2, bands=True, quality=True, chroma=True)
    r = c.post("/api/tracks/bulk", json={"action": "analyze", "scope": "all"})
    assert r.json()["count"] == 2


def test_bulk_ids_scope_targets_a_selection(client):
    c, db = client
    a = _add(db, 1, bands=True, quality=True, chroma=True)
    _add(db, 2, bands=True, quality=True, chroma=True)
    r = c.post("/api/tracks/bulk",
               json={"action": "analyze", "scope": "ids", "song_ids": [a]})
    assert r.json()["count"] == 1


def test_bulk_rejects_unknown_ids(client):
    c, db = client
    _add(db, 1)
    r = c.post("/api/tracks/bulk",
               json={"action": "analyze", "scope": "ids", "song_ids": [999]})
    assert r.status_code == 404


def test_bulk_separate_rewinds_further_than_analyze(client):
    """Re-separation is hours of Demucs where re-analysis is minutes, so the two
    are deliberately different actions."""
    from database.models import get_conn
    c, db = client
    sid = _add(db, 1, bands=True, quality=True, chroma=True)
    c.post("/api/tracks/bulk",
           json={"action": "separate", "scope": "ids", "song_ids": [sid]})
    conn = get_conn(db)
    status = conn.execute("SELECT status FROM songs WHERE id=?", (sid,)).fetchone()[0]
    conn.close()
    assert status == "downloaded"


def test_nothing_stale_says_so_rather_than_queueing_nothing(client):
    c, db = client
    _add(db, 1, bands=True, quality=True, chroma=True)
    r = c.post("/api/tracks/bulk", json={"action": "analyze", "scope": "stale"})
    assert r.status_code == 404
    assert "stale" in r.json()["detail"].lower()


def test_bulk_rejects_a_bad_action_or_scope(client):
    c, _ = client
    assert c.post("/api/tracks/bulk", json={"action": "delete"}).status_code == 400
    assert c.post("/api/tracks/bulk",
                  json={"action": "analyze", "scope": "sideways"}).status_code == 400
