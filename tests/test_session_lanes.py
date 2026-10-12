"""Studio lane state in the FL session export: optional per-lane `lanes` replace
the plan's values, are recorded in session.json and the README, and their
absence changes nothing."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_session_export import SR, _seed_song, env  # noqa: E402,F401

LANES = [
    {"role": "vocal", "gain": 0.5, "fade_in": 0.5},
    {"role": "bed", "semitones": 3, "rate": 1.1, "gain": 0.4, "offset_ms": 250.0,
     "fade_out": 2.0, "highpass_hz": 120.0, "lowpass_hz": 9000.0},
]


def _pair(tmp_path, db_path):
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=126.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))
    return vocal, inst


def _session(out):
    return json.loads((out / "session.json").read_text(encoding="utf-8"))


def test_without_lanes_nothing_changes(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal, inst = _pair(tmp_path, db_path)
    plain = _session(build_session("abcdef10", vocal, inst, db_path=db_path))
    empty = _session(build_session("abcdef11", vocal, inst, db_path=db_path, lanes=[]))
    assert plain["lanes"] is None and empty["lanes"] is None
    assert plain["clips"] == empty["clips"]
    for c in plain["clips"]:
        assert set(c) == {"song_id", "stem", "offset_sec", "rate", "semitones", "gain"}
        assert c["offset_sec"] == 0.0
    assert plain["clips"][0]["rate"] == 1.0 and plain["clips"][0]["semitones"] == 0
    assert plain["stretch_factor"] == pytest.approx(120.0 / 126.0, abs=1e-3)
    readme = (tmp_path / "previews" / "fl_session_abcdef10" / "README.txt").read_text(
        encoding="utf-8")
    assert "FROM YOUR STUDIO LANES" not in readme
    assert "vocal        : native tempo, no transpose" in readme


def test_lanes_replace_the_plans_values(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal, inst = _pair(tmp_path, db_path)
    plain = build_session("abcdef12", vocal, inst, db_path=db_path)
    out = build_session("abcdef13", vocal, inst, db_path=db_path, lanes=LANES)
    data = _session(out)

    assert data["lanes"] == {
        "vocal": {"gain": 0.5, "fade_in": 0.5},
        "bed": {"semitones": 3.0, "rate": 1.1, "gain": 0.4, "offset_ms": 250.0,
                "fade_out": 2.0, "highpass_hz": 120.0, "lowpass_hz": 9000.0}}
    assert data["semitone_shift"] == 3 and data["stretch_factor"] == pytest.approx(1.1)
    v_clip, b_clip = data["clips"]
    assert v_clip == {"song_id": vocal, "stem": "vocals", "offset_sec": 0.0,
                      "rate": 1.0, "semitones": 0, "gain": 0.5, "fade_in": 0.5}
    assert b_clip == {"song_id": inst, "stem": "instrumental", "offset_sec": 0.25,
                      "rate": 1.1, "semitones": 3, "gain": 0.4, "fade_out": 2.0,
                      "hp_hz": 120.0, "lp_hz": 9000.0}
    assert data["inst"]["conformed"]["rate"] == pytest.approx(1.1)
    assert data["inst"]["conformed"]["semitones"] == 3

    # The nudge is baked in: the bed starts 250 ms later than it did.
    bed, _ = sf.read(str(out / "instrumental.wav"))
    before, _ = sf.read(str(plain / "instrumental.wav"))
    lead = int(0.25 * SR)
    assert float(np.max(np.abs(bed[:lead - 8]))) == 0.0
    assert float(np.max(np.abs(bed[lead:lead + SR]))) > 0.01
    assert float(np.max(np.abs(before[:lead]))) > 0.01
    # Level, fades and filters are NOT baked: the vocal file is the plan's.
    v_new, _ = sf.read(str(out / "vocals.wav"))
    v_old, _ = sf.read(str(plain / "vocals.wav"))
    n = min(len(v_new), len(v_old))
    assert np.allclose(v_new[:n], v_old[:n], atol=1e-6)


def test_the_readme_says_what_was_baked_and_what_is_left(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal, inst = _pair(tmp_path, db_path)
    out = build_session("abcdef14", vocal, inst, db_path=db_path, lanes=LANES)
    text = (out / "README.txt").read_text(encoding="utf-8")
    assert "FROM YOUR STUDIO LANES" in text
    assert "×1.1000 time-stretch, +3 semitones, nudged +250 ms" in text
    assert "level -8.0 dB, fade out 2.00 s, high-pass 120 Hz, low-pass 9000 Hz" in text
    assert "level -6.0 dB, fade in 0.50 s" in text
    assert "NOT baked in" in text


def test_a_negative_nudge_trims_the_head_and_a_vocal_lane_can_move(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal, inst = _pair(tmp_path, db_path)
    out = build_session("abcdef15", vocal, inst, db_path=db_path, lanes=[
        {"role": "bed", "offset_ms": -100.0},
        {"role": "vocal", "semitones": -2, "rate": 0.98}])
    data = _session(out)
    assert data["clips"][1]["offset_sec"] == pytest.approx(-0.1)
    assert data["clips"][0]["semitones"] == -2 and data["clips"][0]["rate"] == 0.98
    assert data["semitone_shift"] == 0          # the bed kept the plan's transpose
    text = (out / "README.txt").read_text(encoding="utf-8")
    assert "vocal        : ×0.9800 time-stretch, -2 semitones" in text
    v, _ = sf.read(str(out / "vocals.wav"))
    b, _ = sf.read(str(out / "instrumental.wav"))
    assert len(v) == len(b)                     # still one aligned set of files


def test_session_json_with_lanes_still_renders_as_a_mixdown(env, monkeypatch):
    from render.mixdown import build_mixdown
    from render.session import build_session
    import render.mixdown as mixdown
    tmp_path, db_path = env
    vocal, inst = _pair(tmp_path, db_path)
    out = build_session("abcdef16", vocal, inst, db_path=db_path, lanes=LANES)
    monkeypatch.setattr(mixdown, "PREVIEWS_DIR", tmp_path / "previews")
    assert build_mixdown("abcdef17", _session(out)["clips"], db_path=db_path) is not None


def test_malformed_lanes_are_dropped_not_fatal():
    from render.session import lane_overrides
    assert lane_overrides(None) == {}
    assert lane_overrides([{"role": "drums", "gain": 1}, "nope", {"gain": 1}]) == {}
    assert lane_overrides([{"role": "bed", "gain": "loud", "rate": 1.05, "x": 1},
                           {"role": "bed", "rate": 2.0}]) == {"bed": {"rate": 1.05}}
    assert lane_overrides([{"role": "vocal", "gain": -1, "offset_ms": 1e9}]) == \
        {"vocal": {"gain": 0.0, "offset_ms": 60000.0}}


def test_batch_pairs_carry_their_own_lanes(env):
    from render.session import build_session_batch
    tmp_path, db_path = env
    vocal, inst = _pair(tmp_path, db_path)
    out = build_session_batch("abcdef18", [
        {"vocal_song_id": vocal, "inst_song_id": inst,
         "lanes": [{"role": "bed", "semitones": 2}]},
        {"vocal_song_id": inst, "inst_song_id": vocal},
    ], db_path=db_path)
    folders = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(folders) == 2
    assert _session(folders[0])["semitone_shift"] == 2
    assert _session(folders[0])["lanes"] == {"bed": {"semitones": 2.0}}
    assert _session(folders[1])["lanes"] is None


# ── the worker call path: no db_path, as the app calls it ────────────────────

def test_the_worker_passes_lanes_through_without_a_db_path(env, monkeypatch):
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    import render.session as session
    import api.jobs as jobs
    import api.workers.session_worker as worker
    tmp_path, db_path = env
    assert models.DB_PATH == db_path
    monkeypatch.setattr(session, "PREVIEWS_DIR", tmp_path / "previews")
    vocal, inst = _pair(tmp_path, db_path)

    job_id = jobs.new_job(kind="session")
    worker.run(job_id, vocal, inst, None, None, LANES)
    job = jobs.get(job_id)
    assert job["status"] == "completed", job
    data = _session(Path(job["result"]["folder"]))
    assert data["semitone_shift"] == 3 and data["lanes"]["bed"]["offset_ms"] == 250.0

    batch_id = jobs.new_job(kind="session")
    worker.run_batch(batch_id, [{"vocal_song_id": vocal, "inst_song_id": inst,
                                 "lanes": [{"role": "bed", "semitones": -1}]}])
    job = jobs.get(batch_id)
    assert job["status"] == "completed", job
    folder = next(p for p in Path(job["result"]["folder"]).iterdir() if p.is_dir())
    assert _session(folder)["semitone_shift"] == -1


# ── routes ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "routes.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.routes.studio as studio
    importlib.reload(studio)
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.server as server
    importlib.reload(server)
    calls = []
    monkeypatch.setattr(studio.session_worker, "run", lambda *a: calls.append(("one", a)))
    monkeypatch.setattr(studio.session_worker, "run_batch",
                        lambda *a: calls.append(("batch", a)))
    ids = [models.upsert_song(f"S{n}", "A", f"u://{n}", 200, status="analysed")
           for n in range(2)]
    return TestClient(server.app), ids, calls


def test_session_route_accepts_optional_lanes(client):
    c, (v, i), calls = client
    r = c.post("/api/studio/session", json={"vocal_song_id": v, "inst_song_id": i})
    assert r.status_code == 200 and set(r.json()) == {"job_id", "archive_url"}
    assert calls[-1][1][-1] is None                         # absent = as before

    r = c.post("/api/studio/session", json={
        "vocal_song_id": v, "inst_song_id": i, "vocal_section_idx": 1,
        "inst_section_idx": 2, "lanes": LANES})
    assert r.status_code == 200
    _job, vv, ii, vs, bs, lanes = calls[-1][1]
    assert (vv, ii, vs, bs) == (v, i, 1, 2)
    assert lanes == [{"role": "vocal", "gain": 0.5, "fade_in": 0.5},
                     {"role": "bed", "semitones": 3, "rate": 1.1, "gain": 0.4,
                      "offset_ms": 250.0, "fade_out": 2.0, "highpass_hz": 120.0,
                      "lowpass_hz": 9000.0}]


@pytest.mark.parametrize("lanes,code", [
    ([{"role": "drums"}], 422),
    ([{"role": "bed", "rate": 0}], 422),
    ([{"role": "bed", "semitones": 40}], 422),
    ([{"role": "bed"}, {"role": "bed"}], 400),
])
def test_session_route_rejects_bad_lanes(client, lanes, code):
    c, (v, i), calls = client
    r = c.post("/api/studio/session",
               json={"vocal_song_id": v, "inst_song_id": i, "lanes": lanes})
    assert r.status_code == code and not calls


def test_batch_route_takes_explicit_pairs_with_lanes(client):
    c, (v, i), calls = client
    r = c.post("/api/mashups/session/batch", json={"pairs": [
        {"vocal_song_id": v, "inst_song_id": i, "vocal_section_idx": 0,
         "inst_section_idx": 1, "lanes": [{"role": "bed", "semitones": 2}]},
        {"vocal_song_id": i, "inst_song_id": v}]})
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"job_id", "pair_count", "archive_url"}
    assert r.json()["pair_count"] == 2
    kind, (_job, pairs) = calls[-1]
    assert kind == "batch"
    assert pairs == [
        {"vocal_song_id": v, "inst_song_id": i, "vocal_section_idx": 0,
         "inst_section_idx": 1, "lanes": [{"role": "bed", "semitones": 2}]},
        {"vocal_song_id": i, "inst_song_id": v, "vocal_section_idx": None,
         "inst_section_idx": None}]

    bad = c.post("/api/mashups/session/batch", json={"pairs": [
        {"vocal_song_id": v, "inst_song_id": i, "lanes": [{"role": "bed", "rate": 99}]}]})
    assert bad.status_code == 400
    assert c.post("/api/mashups/session/batch", json={"pairs": []}).status_code == 400
    # No pairs: the filtered top N, as before (nothing scored here → 404).
    assert c.post("/api/mashups/session/batch", json={}).status_code == 404
