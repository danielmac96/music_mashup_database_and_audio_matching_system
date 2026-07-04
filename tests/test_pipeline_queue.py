"""Tests for the Phase-1 auto-processing pipeline: the auto-chain worker and the
ingest-time enqueue. The heavy stages (yt-dlp/Demucs/librosa) are mocked so this
runs in milliseconds with no GPU/network.

Asserts:
  - pipeline_worker walks a track download -> stems -> analyse -> structure and
    marks the job completed.
  - A fatal stage failure contains to that one track (error_* status, failed
    job) and never blocks the rest.
  - A structure failure is non-fatal: the track is still 'analysed' and the job
    completes.
  - POST /api/playlists/ingest enqueues exactly one pipeline job per track.
"""
from pathlib import Path
import importlib
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point audio/DB at tmp_path and reload — in dependency order — the modules
    that bind config paths / helper functions at import time. Using
    importlib.reload (rather than pop + re-import) keeps a single stable module
    object for each, so every module shares one consistent jobs/stages/models
    graph and one clean DB per test."""
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_PIPELINE_WORKERS", "1")

    import config
    import database.models as models
    import api.jobs
    import api.workers.stages
    import api.workers.pipeline_worker
    for mod in (config, models, api.jobs,
                api.workers.stages, api.workers.pipeline_worker):
        importlib.reload(mod)

    models.init_db()
    return models


def _mock_stages(monkeypatch, *, fail_stems_for=None, fail_structure_for=None):
    import api.workers.stages as stages
    import database.models as models

    def dl(sid, on_progress=None):
        models.update_song_status(sid, "downloaded", raw_path=f"/f/{sid}.mp3")
        return {"path": f"/f/{sid}.mp3"}

    def st(sid, on_progress=None):
        if fail_stems_for and sid in fail_stems_for:
            models.update_song_status(sid, "error_stems")
            raise stages.StageError("boom stems")
        models.update_song_status(sid, "stemmed")
        return {}

    def an(sid, on_progress=None):
        models.update_song_status(sid, "analysed")
        return {}

    def sc(sid, on_progress=None):
        if fail_structure_for and sid in fail_structure_for:
            raise stages.StageError("boom structure")
        models.replace_sections(sid, [{"start_sec": 0, "end_sec": 5, "label": "intro"}])
        return {"section_count": 1}

    monkeypatch.setattr(stages, "do_download", dl)
    monkeypatch.setattr(stages, "do_stems", st)
    monkeypatch.setattr(stages, "do_analyze", an)
    monkeypatch.setattr(stages, "do_structure", sc)


def test_pipeline_worker_chains_to_analysed(env, monkeypatch):
    models = env
    _mock_stages(monkeypatch)
    import api.workers.pipeline_worker as pw
    import api.jobs as jobs

    sid = models.upsert_song(title="T", artist="A", source_url="http://x/1", status="queued")
    job_id = jobs.new_job(kind="pipeline", song_id=sid)
    pw.run(job_id, sid)

    assert models.get_song(sid)["status"] == "analysed"
    assert len(models.get_sections(sid)) == 1
    assert jobs.get(job_id)["status"] == "completed"


def test_stage_failure_contains_to_one_track(env, monkeypatch):
    models = env
    good = models.upsert_song(title="Good", artist="A", source_url="http://x/g", status="queued")
    bad = models.upsert_song(title="Bad", artist="B", source_url="http://x/b", status="queued")
    _mock_stages(monkeypatch, fail_stems_for={bad})
    import api.workers.pipeline_worker as pw
    import api.jobs as jobs

    for sid in (good, bad):
        jid = jobs.new_job(kind="pipeline", song_id=sid)
        pw.run(jid, sid)

    assert models.get_song(good)["status"] == "analysed"   # unaffected
    assert models.get_song(bad)["status"] == "error_stems"  # contained


def test_structure_failure_is_non_fatal(env, monkeypatch):
    models = env
    sid = models.upsert_song(title="T", artist="A", source_url="http://x/1", status="queued")
    _mock_stages(monkeypatch, fail_structure_for={sid})
    import api.workers.pipeline_worker as pw
    import api.jobs as jobs

    jid = jobs.new_job(kind="pipeline", song_id=sid)
    pw.run(jid, sid)

    assert models.get_song(sid)["status"] == "analysed"     # still analysed
    assert len(models.get_sections(sid)) == 0               # no sections
    assert jobs.get(jid)["status"] == "completed"           # job succeeds


def test_ingest_enqueues_one_job_per_track(env, monkeypatch):
    # Don't actually run the pipeline: record enqueue calls instead.
    import api.queue_runner as queue_runner
    import api.routes.playlists as playlists_route
    enqueued = []
    monkeypatch.setattr(queue_runner, "enqueue_song",
                        lambda sid: (enqueued.append(sid) or f"job-{sid}"))
    monkeypatch.setattr(playlists_route, "enqueue_song", queue_runner.enqueue_song, raising=False)
    # Avoid network: enrichment "fails" so flat metadata is used.
    monkeypatch.setattr(playlists_route, "enrich_track", lambda url: None)

    from fastapi.testclient import TestClient
    import api.server as server
    # Prevent the real queue/threads from starting during lifespan.
    monkeypatch.setattr(server.queue_runner, "start", lambda *a, **k: None)
    monkeypatch.setattr(server.queue_runner, "resume_pending", lambda *a, **k: 0)

    with TestClient(server.app) as client:
        resp = client.post("/api/playlists/ingest", json={"tracks": [
            {"title": "One", "artist": "A", "source_url": "http://x/1"},
            {"title": "Two", "artist": "B", "source_url": "http://x/2"},
        ]})
    body = resp.json()
    assert resp.status_code == 200
    assert body["count"] == 2
    assert len(body["job_ids"]) == 2
    assert len(enqueued) == 2
