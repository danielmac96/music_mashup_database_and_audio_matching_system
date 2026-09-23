"""The Queue screen's backend: the pool snapshot, place in line, busy counts,
and single-stage jobs that say which track they belong to."""
import importlib
import sys
from pathlib import Path

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
    import api.workers.pipeline_worker
    import api.queue_runner
    import api.routes.jobs
    # Dependency order; reloading api.jobs and api.queue_runner also empties the
    # job registry and the stage queues for this test.
    for mod in (config, models, api.jobs, api.workers.stages,
                api.workers.pipeline_worker, api.queue_runner, api.routes.jobs):
        importlib.reload(mod)
    models.init_db()

    app = FastAPI()
    app.include_router(api.routes.jobs.router, prefix="/api/jobs")
    return models, TestClient(app)


def _song(models, n, status="queued"):
    return models.upsert_song(title=f"T{n}", artist="A",
                              source_url=f"http://x/{n}", status=status)


def test_queue_route_reports_each_tracks_place_in_line(env):
    models, client = env
    import api.jobs as jobs
    import api.queue_runner as queue_runner

    job_ids = [queue_runner.enqueue_song(_song(models, n)) for n in range(3)]
    queue_runner.enqueue_song(_song(models, 9, status="downloaded"))

    resp = client.get("/api/jobs/queue")
    # Declared before /{job_id}: otherwise "queue" is looked up as a job and 404s.
    assert resp.status_code == 200
    body = resp.json()
    assert body["stages"]["download"]["waiting"] == 3
    assert body["stages"]["stems"]["waiting"] == 1
    assert body["stages"]["download"]["workers"] >= 1
    assert [body["positions"][j]["position"] for j in job_ids] == [1, 2, 3]
    assert {body["positions"][j]["stage"] for j in job_ids} == {"download"}
    # The wait is on the job's timeline too, stamped.
    rec = jobs.get(job_ids[0])["stages"]["download"]
    assert rec["state"] == "waiting" and rec["enqueued_at"]


def test_busy_is_counted_from_running_stages_not_job_status(env):
    """A pipeline job is 'running' from its first stage to its last, including
    while it waits in the next queue — counting status would call it busy."""
    _models, client = env
    import api.jobs as jobs

    between = jobs.new_job(kind="pipeline", song_id=1)
    jobs.update(between, status="running")
    jobs.stage_start(between, "download")
    jobs.stage_finish(between, "download", "done")

    structuring = jobs.new_job(kind="pipeline", song_id=2)
    jobs.update(structuring, status="running")
    jobs.stage_start(structuring, "structure")

    stages = client.get("/api/jobs/queue").json()["stages"]
    assert stages["download"]["running"] == 0
    # Structure is the trailing pass of the analysis pool.
    assert stages["analysis"]["running"] == 1


def test_job_snapshots_do_not_share_stage_records(env):
    import api.jobs as jobs

    jid = jobs.new_job(kind="pipeline", song_id=1)
    jobs.stage_start(jid, "download")
    snap = jobs.get(jid)
    jobs.stage_finish(jid, "download", "done")
    assert snap["stages"]["download"]["state"] == "running"
    assert jobs.get(jid)["stages"]["download"]["state"] == "done"


def test_single_stage_jobs_belong_to_their_track(env, monkeypatch):
    """The row menu's Download / Separate jobs carry song_id, so the Queue can
    put them on the right row."""
    models, _client = env
    import api.jobs as jobs
    import api.routes.tracks as tracks
    importlib.reload(tracks)
    monkeypatch.setattr(tracks.download_worker, "run", lambda *a, **k: None)
    monkeypatch.setattr(tracks.stems_worker, "run", lambda *a, **k: None)

    app = FastAPI()
    app.include_router(tracks.router, prefix="/api/tracks")
    client = TestClient(app)

    sid = _song(models, 1)
    models.update_song_status(sid, "downloaded", raw_path="/f/1.mp3")

    for action, kind in (("download", "download"), ("separate", "separate")):
        resp = client.post(f"/api/tracks/{sid}/{action}")
        assert resp.status_code == 200, resp.text
        job = jobs.get(resp.json()["job_id"])
        assert job["kind"] == kind
        assert job["song_id"] == sid
