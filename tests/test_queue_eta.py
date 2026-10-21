"""Queue readability (api/queue_eta.py): eta_secs and a one-sentence summary per
job in /api/jobs and /api/jobs/queue, and clears_in_secs per stage pool."""
import importlib
import sys
from datetime import datetime, timedelta, timezone
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
    import api.queue_eta
    import api.routes.jobs
    for mod in (config, models, api.jobs, api.workers.stages,
                api.workers.pipeline_worker, api.queue_runner, api.queue_eta,
                api.routes.jobs):
        importlib.reload(mod)
    models.init_db()

    app = FastAPI()
    app.include_router(api.routes.jobs.router, prefix="/api/jobs")
    return models, TestClient(app)


def _song(models, n, status="queued", duration=200.0):
    return models.upsert_song(title=f"T{n}", artist="A", source_url=f"http://x/{n}",
                              duration_secs=duration, status=status)


def _history(models):
    """Stems cost 0.5 s per audio second; a download typically 10 s (no audio
    length is recorded for one); analysis 0.2 and structure 0.05 per second."""
    for _ in range(3):
        models.record_analysis_run("stems", 100_000, audio_secs=200.0)
        models.record_analysis_run("download", 10_000)
        models.record_analysis_run("analysis", 40_000, audio_secs=200.0)
        models.record_analysis_run("structure", 10_000, audio_secs=200.0)
    # Never part of an estimate: cached runs, per-stem steps, failures.
    models.record_analysis_run("stems.reused", 5, audio_secs=200.0)
    models.record_analysis_run("analysis.tempo", 900_000, stem_type="full",
                               audio_secs=200.0)
    models.record_analysis_run("stems", 900_000, audio_secs=200.0, ok=False)


def _ago(secs):
    return (datetime.now(timezone.utc) - timedelta(seconds=secs)).isoformat()


# ── contract ─────────────────────────────────────────────────────────────────

def test_contract_shapes(env):
    models, client = env
    import api.queue_runner as queue_runner
    _history(models)
    waiting = queue_runner.enqueue_song(_song(models, 1))

    body = client.get("/api/jobs/queue").json()
    assert set(body) == {"stages", "positions", "running", "next_playable_secs"}
    for pool in ("download", "quick", "stems", "analysis"):
        assert set(body["stages"][pool]) == {"workers", "running", "waiting",
                                            "clears_in_secs"}
        clears = body["stages"][pool]["clears_in_secs"]
        assert clears is None or isinstance(clears, (int, float))
    pos = body["positions"][waiting]
    assert set(pos) == {"stage", "position", "eta_secs", "summary"}
    assert isinstance(pos["summary"], str) and pos["summary"]
    assert isinstance(body["running"], list)

    for job in client.get("/api/jobs").json()["jobs"]:
        assert "eta_secs" in job and isinstance(job["summary"], str)
        assert job["eta_secs"] is None or isinstance(job["eta_secs"], (int, float))
    one = client.get(f"/api/jobs/{waiting}").json()
    assert one["summary"] == pos["summary"] and one["eta_secs"] == pos["eta_secs"]


def test_running_entries_carry_the_stage_and_its_eta(env):
    models, client = env
    import api.jobs as jobs
    _history(models)
    jid = jobs.new_job(kind="pipeline", song_id=_song(models, 1))
    jobs.update(jid, status="running")
    jobs.stage_start(jid, "stems")
    body = client.get("/api/jobs/queue").json()
    assert len(body["running"]) == 1
    run = body["running"][0]
    assert set(run) == {"job_id", "song_id", "stage", "eta_secs", "summary"}
    assert run["job_id"] == jid and run["stage"] == "stems"
    assert run["eta_secs"] == pytest.approx(100.0, abs=3)
    rec = client.get(f"/api/jobs/{jid}").json()["stages"]["stems"]
    assert rec["eta_secs"] == pytest.approx(100.0, abs=3)


# ── estimates ────────────────────────────────────────────────────────────────

def test_a_running_stage_is_rate_times_length_minus_elapsed(env):
    models, _client = env
    import api.jobs as jobs
    import api.queue_eta as queue_eta
    _history(models)
    jid = jobs.new_job(kind="pipeline", song_id=_song(models, 1, duration=400.0))
    jobs.update(jid, status="running")
    jobs.stage_start(jid, "stems")
    jobs._stage_merge(jid, "stems", {"started_at": _ago(80)})

    job = jobs.get(jid)
    pools = queue_eta.annotate([job], {"stems": {"workers": 1, "waiting": []}})
    # 0.5 s per audio second × 400 s = 200 s, 80 s in.
    assert job["eta_secs"] == pytest.approx(120.0, abs=2)
    assert job["stages"]["stems"]["eta_secs"] == job["eta_secs"]
    assert job["summary"] == "Separating vocals — about 2 min left"
    assert pools["stems"]["clears_in_secs"] == pytest.approx(120.0, abs=2)
    assert pools["download"]["clears_in_secs"] == 0


def test_an_overrunning_stage_never_goes_negative(env):
    models, _client = env
    import api.jobs as jobs
    import api.queue_eta as queue_eta
    _history(models)
    jid = jobs.new_job(kind="pipeline", song_id=_song(models, 1))
    jobs.update(jid, status="running")
    jobs.stage_start(jid, "stems")
    jobs._stage_merge(jid, "stems", {"started_at": _ago(5000)})
    job = jobs.get(jid)
    queue_eta.annotate([job], {})
    assert job["eta_secs"] == 0


def test_a_waiting_job_adds_the_line_in_front_of_it(env):
    models, client = env
    import api.queue_runner as queue_runner
    _history(models)
    ids = [queue_runner.enqueue_song(_song(models, n, status="downloaded"))
           for n in range(3)]
    # Downloaded tracks wait for the quick tier, which has no history here.
    body = client.get("/api/jobs/queue").json()
    assert body["positions"][ids[2]]["eta_secs"] is None
    assert body["positions"][ids[2]]["summary"] == "Waiting for quick analysis (#3 in line)"
    assert body["stages"]["quick"]["clears_in_secs"] is None

    dl = [queue_runner.enqueue_song(_song(models, 10 + n)) for n in range(3)]
    body = client.get("/api/jobs/queue").json()
    workers = body["stages"]["download"]["workers"]
    etas = [body["positions"][j]["eta_secs"] for j in dl]
    # Its own 10 s, plus position × the typical 10 s shared across the workers.
    assert etas == [pytest.approx(10 + (i + 1) * 10 / workers) for i in range(3)]
    assert body["positions"][dl[1]]["summary"] == "Waiting for download (#2 in line)"
    assert body["stages"]["download"]["clears_in_secs"] == pytest.approx(30 / workers)


def test_the_analysis_pool_counts_analysis_and_structure(env):
    models, _client = env
    import api.queue_eta as queue_eta
    _history(models)
    rates = queue_eta.stage_rates()
    assert queue_eta.estimate(queue_eta.POOL_GROUPS["analysis"], rates, 200.0) \
        == pytest.approx(50.0)
    assert queue_eta.estimate(queue_eta.RECORD_GROUPS["analysis"], rates, 200.0) \
        == pytest.approx(40.0)
    # No length: fall back to the typical wall time.
    assert queue_eta.estimate(("stems",), rates, None) == pytest.approx(100.0)


def test_no_history_is_null_and_never_raises(env):
    models, client = env
    import api.jobs as jobs
    import api.queue_runner as queue_runner
    waiting = queue_runner.enqueue_song(_song(models, 1))
    running = jobs.new_job(kind="pipeline", song_id=_song(models, 2))
    jobs.update(running, status="running")
    jobs.stage_start(running, "stems")

    body = client.get("/api/jobs/queue")
    assert body.status_code == 200
    body = body.json()
    assert body["positions"][waiting]["eta_secs"] is None
    assert body["positions"][waiting]["summary"] == "Waiting for download (#1 in line)"
    assert body["running"][0]["eta_secs"] is None
    assert body["running"][0]["summary"] == "Separating vocals…"
    assert body["stages"]["download"]["clears_in_secs"] is None
    assert body["stages"]["stems"]["clears_in_secs"] is None
    assert body["stages"]["analysis"]["clears_in_secs"] == 0     # nothing queued
    assert client.get("/api/jobs").status_code == 200


def test_a_broken_timings_table_degrades_to_null(env):
    models, client = env
    import api.queue_runner as queue_runner
    jid = queue_runner.enqueue_song(_song(models, 1))
    conn = models.get_conn()
    conn.execute("DROP TABLE analysis_runs")
    conn.commit()
    conn.close()
    r = client.get("/api/jobs/queue")
    assert r.status_code == 200
    assert r.json()["positions"][jid]["eta_secs"] is None
    assert client.get("/api/jobs").status_code == 200


# ── sentences ────────────────────────────────────────────────────────────────

def test_failed_and_done_sentences(env):
    models, client = env
    import api.jobs as jobs
    failed = jobs.new_job(kind="pipeline", song_id=_song(models, 1))
    jobs.update(failed, status="running")
    jobs.stage_start(failed, "download")
    jobs.stage_finish(failed, "download", "failed", error="SoundCloud preview only")
    jobs.fail(failed, "SoundCloud preview only\nTraceback…")

    done = jobs.new_job(kind="pipeline", song_id=_song(models, 2))
    jobs.stage_start(done, "download")
    jobs._stage_merge(done, "download", {"started_at": _ago(252)})
    jobs.stage_finish(done, "download", "done")
    jobs.done(done)

    scoring = jobs.new_job(kind="match", message="Scoring pairs 40%")
    jobs.update(scoring, status="running")

    by_id = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert by_id[failed]["summary"] == \
        "Failed downloading: SoundCloud preview only — Retry"
    assert by_id[done]["summary"] == "Done in 4 m 12 s"
    assert by_id[scoring]["summary"] == "Scoring pairs 40%"
    assert all(by_id[j]["eta_secs"] is None for j in (failed, done, scoring))


@pytest.mark.parametrize("secs,want", [(8, "about 8 s"), (89, "about 89 s"),
                                       (120, "about 2 min"), (5400, "about 1.5 h")])
def test_about(secs, want):
    from api.queue_eta import about
    assert about(secs) == want


@pytest.mark.parametrize("secs,want", [(38, "38 s"), (252, "4 m 12 s"),
                                       (3780, "1 h 3 m")])
def test_took(secs, want):
    from api.queue_eta import took
    assert took(secs) == want
