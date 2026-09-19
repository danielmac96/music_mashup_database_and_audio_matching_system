"""POST /api/mixes/{id}/ingest and the worker behind it.

The bug this file exists for: ingesting a 206-track mix saved exactly ONE track
and then answered 500, forever. The route held one open connection across the
whole loop while ``upsert_song`` wrote on its own connection, so the first
``UPDATE mix_tracks`` took SQLite's single writer lock and the next
``upsert_song`` waited out busy_timeout and raised "database is locked".

These tests run the worker inline (no BackgroundTasks) against a real temp DB,
with the network (``enrich_track``) and the pipeline queue stubbed out.
"""
import importlib

import pytest
from fastapi import BackgroundTasks, HTTPException


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import playlists
    importlib.reload(playlists)
    from api.routes import mixes
    importlib.reload(mixes)
    from api.workers import mix_ingest_worker
    importlib.reload(mix_ingest_worker)

    # No network: every link enriches to itself. ingest_rows reaches
    # enrich_track through _resolve_metadata for rows not marked hydrated.
    monkeypatch.setattr(playlists, "enrich_track",
                        lambda url: {"title": "Enriched", "artist": "A",
                                     "source_url": url, "duration_secs": 200.0})
    # No pipeline: record what would have been queued.
    queued: list[int] = []
    monkeypatch.setattr(playlists.queue_runner, "enqueue_song",
                        lambda sid: (queued.append(sid), f"job-{sid}")[1])
    return mixes, mix_ingest_worker, models, queued


def _mk(mixes, links):
    """A persisted mix whose tracks carry `links` (None = unlinked)."""
    rows = [{"entry_index": i + 1, "cue_secs": None, "is_overlay": False,
             "artist": f"A{i}", "title": f"T{i}", "raw_label": f"{i+1}. A{i} - T{i}",
             "is_id": 0, "remixer": None, "mashup_parts": [], "parse_confidence": 1.0}
            for i in range(len(links))]
    detail = mixes._persist_mix("M", "https://src/set", rows, method="paste")
    conn = mixes.get_conn()
    try:
        for track, link in zip(detail["tracks"], links):
            if link:
                conn.execute("UPDATE mix_tracks SET link_url=?, link_platform='soundcloud',"
                             " resolve_status='manual' WHERE id=?", (link, track["id"]))
        conn.commit()
    finally:
        conn.close()
    return detail


def _ingest(mixes, worker, mix_id):
    """Call the route, then run its job inline, and return the job."""
    from api import jobs
    out = mixes.ingest_mix(mix_id, BackgroundTasks())
    worker.run_ingest(out["job_id"], mix_id)
    return out, jobs.get(out["job_id"])


def _links(n):
    return [f"https://soundcloud.com/a/t{i}" for i in range(n)]


# ── the reported bug ──────────────────────────────────────────────────────────

def test_every_linked_track_is_ingested_not_just_the_first(tmp_path, monkeypatch):
    # THE regression: 206 linked tracks used to yield exactly 1 saved song and a
    # 500. Well past the worker's CHUNK of 25, so the batching is exercised too.
    mixes, worker, models, queued = _setup(tmp_path, monkeypatch)
    n = 60
    detail = _mk(mixes, _links(n))

    out, job = _ingest(mixes, worker, detail["id"])

    assert out["queued"] == n
    assert job["status"] == "completed", job.get("error")
    assert job["result"]["count"] == n
    assert len(queued) == n            # one pipeline job per track
    assert len(models.get_all_songs()) == n


def test_every_mix_track_is_pointed_at_its_song(tmp_path, monkeypatch):
    mixes, worker, models, _q = _setup(tmp_path, monkeypatch)
    detail = _mk(mixes, _links(30))

    _ingest(mixes, worker, detail["id"])

    after = mixes.get_mix(detail["id"])
    assert after["ingested_count"] == 30
    assert all(t["song_id"] for t in after["tracks"])
    assert all(t["resolve_status"] == "resolved" for t in after["tracks"])


def test_a_second_ingest_is_a_no_op_and_never_500s(tmp_path, monkeypatch):
    # The second symptom: clicking Ingest again answered 500 and queued nothing.
    mixes, worker, models, queued = _setup(tmp_path, monkeypatch)
    detail = _mk(mixes, _links(30))
    _ingest(mixes, worker, detail["id"])
    queued.clear()

    out, job = _ingest(mixes, worker, detail["id"])

    assert out["queued"] == 0
    assert out["already_ingested"] == 30
    assert job["status"] == "completed", job.get("error")
    assert job["result"]["count"] == 0
    assert queued == []                       # nothing re-processed
    assert len(models.get_all_songs()) == 30  # nothing duplicated


def test_a_second_ingest_does_not_reset_an_analysed_song(tmp_path, monkeypatch):
    # upsert_song's ON CONFLICT does `status=excluded.status`, so re-upserting an
    # already-ingested track would knock it back to 'queued' and re-run the whole
    # download/stems/analysis pipeline. Dedup has to happen before that.
    mixes, worker, models, _q = _setup(tmp_path, monkeypatch)
    detail = _mk(mixes, _links(3))
    _ingest(mixes, worker, detail["id"])
    for song in models.get_all_songs():
        models.update_song_status(song["id"], "analysed")

    _ingest(mixes, worker, detail["id"])

    assert [s["status"] for s in models.get_all_songs()] == ["analysed"] * 3


def test_the_same_link_twice_makes_one_song_and_one_job(tmp_path, monkeypatch):
    # A track that appears as both a bed and a 'w/' overlay carries the same
    # link on two rows. songs.source_url is UNIQUE, so the second upsert used to
    # collapse onto the first and enqueue it a second time — two pipeline jobs
    # racing on one song.
    mixes, worker, models, queued = _setup(tmp_path, monkeypatch)
    dupe = "https://soundcloud.com/a/same"
    detail = _mk(mixes, [dupe, "https://soundcloud.com/a/other", dupe])

    _ingest(mixes, worker, detail["id"])

    assert len(models.get_all_songs()) == 2
    assert len(queued) == 2
    after = mixes.get_mix(detail["id"])
    ids = [t["song_id"] for t in after["tracks"]]
    assert ids[0] == ids[2] and ids[0] != ids[1]   # both rows point at one song


def test_unlinked_tracks_are_left_alone(tmp_path, monkeypatch):
    mixes, worker, models, queued = _setup(tmp_path, monkeypatch)
    detail = _mk(mixes, ["https://soundcloud.com/a/t0", None, "https://soundcloud.com/a/t2"])

    out, job = _ingest(mixes, worker, detail["id"])

    assert out["queued"] == 2
    assert job["result"]["count"] == 2
    after = mixes.get_mix(detail["id"])
    assert [bool(t["song_id"]) for t in after["tracks"]] == [True, False, True]


def test_a_mix_with_no_links_is_a_400(tmp_path, monkeypatch):
    mixes, _worker, _m, _q = _setup(tmp_path, monkeypatch)
    detail = _mk(mixes, [None, None])
    with pytest.raises(HTTPException) as ei:
        mixes.ingest_mix(detail["id"], BackgroundTasks())
    assert ei.value.status_code == 400


def test_unknown_mix_is_a_404(tmp_path, monkeypatch):
    mixes, _worker, _m, _q = _setup(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as ei:
        mixes.ingest_mix(9999, BackgroundTasks())
    assert ei.value.status_code == 404


# ── the mechanism that broke ──────────────────────────────────────────────────

def test_the_worker_holds_no_write_lock_while_saving(tmp_path, monkeypatch):
    """The root cause, pinned directly.

    While ``upsert_song`` runs, nothing may be holding an open write transaction
    on another connection — SQLite has one writer, and the second connection
    waits out busy_timeout and then raises "database is locked". Here a probe
    inside upsert_song asserts a *separate* connection can still write."""
    mixes, worker, models, _q = _setup(tmp_path, monkeypatch)
    from api.routes import playlists
    detail = _mk(mixes, _links(5))

    real_upsert = playlists.upsert_song
    probes: list[bool] = []

    def probing_upsert(*args, **kwargs):
        probe = models.get_conn()
        try:
            probe.execute("UPDATE songs SET updated_at=updated_at")
            probe.commit()
            probes.append(True)
        finally:
            probe.close()
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(playlists, "upsert_song", probing_upsert)
    _out, job = _ingest(mixes, worker, detail["id"])

    assert job["status"] == "completed", job.get("error")
    assert probes == [True] * 5


def test_a_failure_mid_run_keeps_what_was_already_saved(tmp_path, monkeypatch):
    # The old route committed only at the very end, so a failure rolled back
    # every mix_tracks.song_id it had written — the mix looked untouched even
    # though the songs were in the library. Batches commit as they go.
    mixes, worker, models, _q = _setup(tmp_path, monkeypatch)
    from api.routes import playlists
    detail = _mk(mixes, _links(worker.CHUNK * 2))

    calls = {"n": 0}
    real_rows = playlists.ingest_rows

    def failing_rows(tracks, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom")
        return real_rows(tracks, **kw)

    monkeypatch.setattr(playlists, "ingest_rows", failing_rows)
    out = mixes.ingest_mix(detail["id"], BackgroundTasks())
    worker.run_ingest(out["job_id"], detail["id"])

    from api import jobs
    job = jobs.get(out["job_id"])
    assert job["status"] == "failed"
    assert "boom" in job["error"]
    # The first batch survived and is still pointed at its songs.
    after = mixes.get_mix(detail["id"])
    assert after["ingested_count"] == worker.CHUNK
