"""Pipeline timings persist to analysis_runs: every stage and every analysis /
structure step, failures included, readable as a summary over the API.

Phase 0 of the analysis overhaul — "which part of the pipeline is slow" had no
answer beyond the in-memory job timeline, which a restart erased."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))

    import config
    import database.models as models
    import api.jobs
    import api.workers.stages
    import api.routes.jobs
    for mod in (config, models, api.jobs, api.workers.stages, api.routes.jobs):
        importlib.reload(mod)
    models.init_db()

    app = FastAPI()
    app.include_router(api.routes.jobs.router, prefix="/api/jobs")
    return models, api.workers.stages, TestClient(app), tmp_path


def _runs(models):
    conn = models.get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM analysis_runs ORDER BY id").fetchall()]
    finally:
        conn.close()


def _wav(path: Path, secs: float, bpm: float = 120.0, sr: int = 22050) -> Path:
    import soundfile as sf
    n = int(secs * sr)
    y = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(np.arange(0, secs, 60.0 / bpm)):
        s = int(t * sr)
        e = min(n, s + int(0.08 * sr))
        tt = np.arange(e - s) / sr
        y[s:e] += (1.0 if i % 4 == 0 else 0.5) * np.sin(2 * np.pi * 80 * tt) * np.exp(-tt * 30)
    # a tone that changes every 16 beats gives the segmenter something to find
    for k, t in enumerate(np.arange(0, secs, 16 * 60.0 / bpm)):
        s, e = int(t * sr), min(n, int((t + 16 * 60.0 / bpm) * sr))
        tt = np.arange(e - s) / sr
        y[s:e] += 0.2 * np.sin(2 * np.pi * (220 if k % 2 else 330) * tt)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y / np.abs(y).max(), sr)
    return path


# ── The table and its summary ─────────────────────────────────────────────────

def test_summary_reports_medians_failures_and_real_time_factor(env):
    models, _stages, _client, _tmp = env
    for ms in (100, 200, 300):
        models.record_analysis_run("analysis.tempo", ms, song_id=1,
                                   stem_type="full", audio_secs=100)
    models.record_analysis_run("analysis.tempo", 999, song_id=2,
                               stem_type="full", ok=False, error="boom")
    models.record_analysis_run("stems", 60_000, song_id=1, audio_secs=200)

    rows = {(r["grp"], r["stem_type"]): r for r in models.analysis_timing_summary()}
    tempo = rows[("analysis.tempo", "full")]
    # The failed run is counted but never pollutes the timing it failed to make.
    assert tempo["runs"] == 3 and tempo["failed"] == 1
    assert tempo["median_ms"] == 200 and tempo["p90_ms"] == 300
    assert tempo["median_rtf"] == pytest.approx(0.002)
    assert rows[("stems", None)]["median_rtf"] == pytest.approx(0.3)


def test_recording_never_raises(env, tmp_path):
    models, *_ = env
    # A path whose parent is a file cannot be opened as a database.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    models.record_analysis_run("stems", 1.0, db_path=blocker / "db.sqlite")


def test_timings_route_is_not_swallowed_by_job_id(env):
    models, _stages, client, _tmp = env
    models.record_analysis_run("download", 1500, song_id=1)
    resp = client.get("/api/jobs/timings")
    assert resp.status_code == 200
    assert resp.json()["timings"][0]["grp"] == "download"


# ── Stages record themselves ──────────────────────────────────────────────────

def test_a_failed_download_is_recorded_as_a_failed_run(env, monkeypatch):
    models, stages, _client, _tmp = env
    import downloader.download as dl

    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")

    def _refuse(**_kw):
        raise dl.DownloadError("Go+ only")
    monkeypatch.setattr(dl, "download_track", _refuse)

    with pytest.raises(stages.StageError):
        stages.do_download(song)
    (run,) = _runs(models)
    assert run["grp"] == "download" and run["ok"] == 0 and "Go+" in run["error"]


def test_reused_stems_are_not_filed_as_a_demucs_timing(env, monkeypatch):
    models, stages, _client, tmp = env
    import stems.separate as sep

    raw = _wav(tmp / "raw.wav", 2.0)
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "downloaded", raw_path=str(raw))
    monkeypatch.setattr(sep, "separate", lambda **_kw: {
        "vocals": raw, "instrumental": raw, "separator": None})

    stages.do_stems(song)
    (run,) = _runs(models)
    # A reuse costs nothing and would drag the Demucs median toward zero.
    assert run["grp"] == "stems.reused" and run["ok"] == 1
    assert run["audio_secs"] == pytest.approx(2.0, abs=0.01)


def test_analysis_records_each_step_per_stem_and_marks_failed_steps(env, monkeypatch):
    models, stages, _client, tmp = env
    import analysis.analyze as analyze
    import analysis.quality as quality

    raw = _wav(tmp / "raw.wav", 1.0)
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "stemmed", raw_path=str(raw))
    models.upsert_stem(song, "full", str(raw))

    def _fake(path, trim_secs=None, on_progress=None, timings=None, cache=None):
        timings.update({"load": 5.0, "tempo": 10.0, "key": 20.0,
                        "audio_secs": 1.0, "failed_steps": ["key"]})
        return {"bpm": 120.0}
    monkeypatch.setattr(analyze, "analyze_file", _fake)
    monkeypatch.setattr(quality, "band_energy", lambda _p: [0.125] * 8)
    monkeypatch.setattr(stages, "_measure_stem_quality", lambda *a, **k: None)

    stages.do_analyze(song)
    runs = {r["grp"]: r for r in _runs(models) if r["stem_type"] == "full"}
    assert runs["analysis.tempo"]["ms"] == 10.0 and runs["analysis.tempo"]["ok"] == 1
    assert runs["analysis.key"]["ok"] == 0
    assert runs["analysis.tempo"]["audio_secs"] == 1.0
    assert "analysis.bands" in runs
    grps = {r["grp"] for r in _runs(models)}
    assert {"analysis", "quality"} <= grps


def test_structure_records_the_stage_and_its_phases(env, monkeypatch):
    models, stages, _client, tmp = env
    import analysis.structure as structure
    import api.workers.hook_worker as hook_worker

    raw = _wav(tmp / "raw.wav", 1.0)
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "analysed", raw_path=str(raw))
    models.upsert_stem(song, "full", str(raw))

    def _fake(full, vocals=None, inst_path=None, bass_path=None,
              on_progress=None, timings=None, grid=None):
        timings.update({"load": 3.0, "beats": 7.0, "audio_secs": 1.0})
        return [{"start_sec": 0.0, "end_sec": 1.0, "label": "verse",
                 "energy": 1.0, "vocal_presence": None, "repetition": 1,
                 "confidence": 0.3}]
    monkeypatch.setattr(structure, "detect_sections", _fake)
    monkeypatch.setattr(hook_worker, "warm_hooks", lambda *a, **k: {})

    stages.do_structure(song)
    runs = {r["grp"]: r for r in _runs(models)}
    assert runs["structure.beats"]["ms"] == 7.0
    assert runs["structure"]["audio_secs"] == 1.0
    assert "hooks" in runs


# ── The analysers fill the timing dict for real ───────────────────────────────

def test_analyze_file_times_every_step(tmp_path):
    from analysis.analyze import STEPS, analyze_file
    timings: dict = {}
    feats = analyze_file(_wav(tmp_path / "t.wav", 8.0), timings=timings)
    assert feats.get("bpm")
    assert {"load", *STEPS} <= set(timings)
    assert timings["audio_secs"] == pytest.approx(8.0, abs=0.05)
    assert timings["failed_steps"] == []
    assert all(timings[s] >= 0 for s in STEPS)


def test_detect_sections_times_its_phases(tmp_path):
    from analysis.structure import detect_sections
    timings: dict = {}
    sections = detect_sections(_wav(tmp_path / "t.wav", 60.0), timings=timings)
    assert sections
    assert {"load", "beats", "features", "boundaries", "sections"} <= set(timings)
    # No stems given: the phases that need them never ran and are absent, not 0.
    assert "vocal" not in timings
    assert timings["audio_secs"] == pytest.approx(60.0, abs=0.05)
