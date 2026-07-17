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


def test_stage_queues_route_track_through_pipeline(env, monkeypatch):
    """A queued track hops download → stems → analysis queues and completes.
    Queues are drained synchronously here (no worker threads) so the routing
    logic is tested without daemon-thread/monkeypatch lifetime hazards."""
    models = env
    _mock_stages(monkeypatch)
    import api.queue_runner as queue_runner
    importlib.reload(queue_runner)  # rebind the reloaded api.jobs from env
    import api.workers.pipeline_worker as pw
    import api.jobs as jobs

    sid = models.upsert_song(title="T", artist="A", source_url="http://x/1", status="queued")
    jid = queue_runner.enqueue_song(sid)

    for expected in ("download", "stems", "analysis"):
        job_id, song_id = queue_runner._QUEUES[expected].get_nowait()
        assert song_id == sid
        outcome = pw.run_stage(job_id, song_id, expected)
        if outcome == "next":
            queue_runner._dispatch(job_id, song_id)

    assert outcome == "done"
    assert models.get_song(sid)["status"] == "analysed"
    assert len(models.get_sections(sid)) == 1
    assert jobs.get(jid)["status"] == "completed"
    assert queue_runner.queued_count() == 0


def test_dispatch_resumes_mid_pipeline_track_at_right_stage(env, monkeypatch):
    """A 'downloaded' track (server restarted mid-pipeline) enters at stems,
    not download — the status-derived resume contract."""
    models = env
    _mock_stages(monkeypatch)
    import api.queue_runner as queue_runner
    importlib.reload(queue_runner)

    sid = models.upsert_song(title="T", artist="A", source_url="http://x/2",
                             status="downloaded")
    queue_runner.enqueue_song(sid)
    assert queue_runner._QUEUES["download"].qsize() == 0
    assert queue_runner._QUEUES["stems"].qsize() == 1


def test_error_records_last_error_and_progress_clears_it(env):
    models = env
    import api.workers.stages as stages

    # A queued song with no downloaded file: do_stems must fail and record why.
    sid = models.upsert_song(title="T", artist="A", source_url="http://x/1", status="queued")
    with pytest.raises(stages.StageError):
        stages.do_stems(sid)
    song = models.get_song(sid)
    assert song["status"] == "error_stems"
    assert song["last_error"] and "Download it first" in song["last_error"]

    # Advancing to a non-error status clears the stale reason.
    models.update_song_status(sid, "downloaded", raw_path="/f/1.mp3")
    assert models.get_song(sid)["last_error"] is None


def test_reverify_replaces_stale_preview(tmp_path, monkeypatch):
    import downloader.download as dl
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path)
    out = tmp_path / f"{dl._safe('Song')}_{dl._safe('Artist')}.mp3"
    out.write_bytes(b"preview")
    monkeypatch.setattr(dl, "_get_duration", lambda p: 30.0)  # on-disk file is a preview
    monkeypatch.setattr(dl, "download_track",
                        lambda *a, **k: dl.DownloadResult(out, 180.0))  # YT fallback full

    res = dl.reverify_track(1, "Song", "http://x", artist="Artist")
    assert res.replaced is True
    assert res.duration_secs == 180.0


def test_reverify_keeps_full_file_without_redownload(tmp_path, monkeypatch):
    import downloader.download as dl
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path)
    out = tmp_path / f"{dl._safe('Song')}_{dl._safe('Artist')}.mp3"
    out.write_bytes(b"full")
    monkeypatch.setattr(dl, "_get_duration", lambda p: 200.0)  # already full-length
    calls = {"n": 0}
    monkeypatch.setattr(dl, "download_track",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), None)[1])

    res = dl.reverify_track(1, "Song", "http://x", artist="Artist")
    assert res.replaced is False
    assert res.duration_secs == 200.0
    assert calls["n"] == 0  # no re-download when the file is already full


def _song_with_feats(models, title, url, full_bpm, camelot):
    sid = models.upsert_song(title=title, artist="A", source_url=url, status="analysed")
    for stem in ("full", "vocals", "instrumental"):
        models.upsert_features(sid, stem, {
            "bpm": full_bpm, "camelot": camelot, "key": "A", "mode": "minor",
            "energy": 0.5, "loudness_rms": 0.5,
        })
    return sid


def test_score_filter_width_and_deterministic_clear(env):
    models = env
    from matcher.match import score_all_pairs

    _song_with_feats(models, "A", "http://x/a", 120.0, "8A")
    _song_with_feats(models, "B", "http://x/b", 132.0, "9A")  # 12 BPM apart, adjacent key

    # Wide pre-filter: the 12-BPM gap qualifies → 2 vocal/inst + 1 inst/inst.
    score_all_pairs(bpm_max_diff=16, key_min_score=0.4)
    assert len(models.get_candidates(limit=100)) == 3

    # Tighter pre-filter: 12 > 10 BPM, so nothing qualifies AND the prior
    # (now stale) pairs are cleared — re-score is deterministic.
    score_all_pairs(bpm_max_diff=10, key_min_score=0.55)
    assert len(models.get_candidates(limit=100)) == 0


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
