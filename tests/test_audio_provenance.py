"""Provenance: the link a track was imported from survives a substitute download,
and tracks whose audio may not be their record are found and re-downloaded.

Regression: "Massive" (soundcloud.com/octobersveryown/drake-massive, 5:37) was
downloaded as a 3:00 YouTube remix, and the download then overwrote source_url
and duration_secs with the remix's — erasing both the link that was asked for
and the length that would have exposed the mismatch.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SC_URL = "https://soundcloud.com/octobersveryown/drake-massive"
YT_REMIX = "https://youtube.com/watch?v=ArTTyLYpi70"
YT_RECORD = "https://www.youtube.com/watch?v=ay1l_u6vltY"
SC_ART = "https://i1.sndcdn.com/artworks-Y2iukGk0BWR0-0-t500x500.jpg"
LENGTH = 336.947


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    return models


def _raw_song(models, **cols):
    """Insert a row the way an older build left it — no origin columns set."""
    conn = models.get_conn()
    keys = ", ".join(cols)
    conn.execute(f"INSERT INTO songs ({keys}) VALUES ({','.join('?' * len(cols))})",
                 tuple(cols.values()))
    conn.commit()
    sid = conn.execute("SELECT MAX(id) FROM songs").fetchone()[0]
    conn.close()
    return sid


def _massive_as_the_bug_left_it(models, status="analysed", url=YT_REMIX):
    return _raw_song(models, title="Massive", artist="octobersveryown",
                     source_url=url, source="youtube", duration_secs=180.27,
                     track_id="1289061475", thumbnail=SC_ART, status=status)


# ── the origin columns ──────────────────────────────────────────────────────

def test_insert_records_the_import_link_and_length_once(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Massive", artist="Drake", source_url=SC_URL,
                        duration_secs=LENGTH, source="soundcloud")
    row = m.get_song(sid)
    assert row["origin_url"] == SC_URL
    assert row["origin_duration_secs"] == pytest.approx(LENGTH)

    m.upsert_song(title="Massive", artist="Drake", source_url=SC_URL, duration_secs=30)
    assert m.get_song(sid)["origin_duration_secs"] == pytest.approx(LENGTH)


def test_a_preview_length_is_not_the_records_length(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Snip", source_url="https://soundcloud.com/a/snip",
                        duration_secs=30)
    assert m.get_song(sid)["origin_duration_secs"] is None


def test_migration_backfills_every_link_a_fallback_did_not_overwrite(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sc = _raw_song(m, title="SC", source_url="https://soundcloud.com/a/b",
                   source="soundcloud", duration_secs=200, track_id="123", thumbnail=SC_ART)
    yt = _raw_song(m, title="YT", source_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
                   source="youtube", duration_secs=212, track_id="dQw4w9WgXcQ",
                   thumbnail="https://i.ytimg.com/vi/x.jpg")
    bug = _massive_as_the_bug_left_it(m)

    conn = m.get_conn()
    m._migrate_songs_columns(conn)
    conn.commit()
    conn.close()

    assert m.get_song(sc)["origin_url"] == "https://soundcloud.com/a/b"
    assert m.get_song(sc)["origin_duration_secs"] == pytest.approx(200)
    assert m.get_song(yt)["origin_url"] == "https://youtube.com/watch?v=dQw4w9WgXcQ"
    # Guessing here would record the remix as what was asked for.
    assert m.get_song(bug)["origin_url"] is None
    assert m.get_song(bug)["origin_duration_secs"] is None


def test_dedup_finds_a_track_by_the_link_it_was_imported_from(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Massive", source_url=SC_URL, duration_secs=LENGTH)
    conn = m.get_conn()
    conn.execute("UPDATE songs SET source_url=? WHERE id=?", (YT_REMIX, sid))
    conn.commit()
    conn.close()
    assert m.get_song_by_url(SC_URL)["id"] == sid
    assert m.get_song_by_url(YT_REMIX)["id"] == sid
    ident = m.songs_by_identity(source_urls=[SC_URL])
    assert ident["by_url"][SC_URL]["id"] == sid


def test_changing_the_link_keeps_the_origin_and_records_provenance(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Massive", source_url=SC_URL, duration_secs=LENGTH)
    m.update_song_url(sid, YT_RECORD, provenance={"via": "manual", "url": YT_RECORD})
    row = m.get_song(sid)
    assert row["origin_url"] == SC_URL
    assert json.loads(row["audio_provenance"])["via"] == "manual"
    m.update_song_url(sid, SC_URL)
    assert m.get_song(sid)["audio_provenance"] is None


def test_set_song_origin_restores_link_length_and_credited_artist(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = _massive_as_the_bug_left_it(m)
    m.set_song_origin(sid, SC_URL, LENGTH, artist="Drake")
    row = m.get_song(sid)
    assert (row["origin_url"], row["artist"]) == (SC_URL, "Drake")
    assert row["origin_duration_secs"] == pytest.approx(LENGTH)
    m.set_song_origin(sid, SC_URL, 30, artist="")          # a preview, no artist
    row = m.get_song(sid)
    assert row["origin_duration_secs"] == pytest.approx(LENGTH)
    assert row["artist"] == "Drake"


# ── the download stage ──────────────────────────────────────────────────────

def _stages(tmp_path, monkeypatch, result_factory, seen):
    import downloader.download as dl
    from api.workers import stages
    importlib.reload(stages)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")

    def fake(song_id, title, source_url, artist="", on_progress=None, **kw):
        seen.update(kw, artist=artist)
        return result_factory(dl, audio)

    monkeypatch.setattr(dl, "download_track", fake)
    return stages


def test_download_checks_against_the_origin_length_not_the_overwritten_one(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Massive", artist="Drake", source_url=SC_URL,
                        duration_secs=LENGTH, source="soundcloud")
    m.update_song_duration(sid, 180.27)        # what the old fallback wrote
    seen = {}
    stages = _stages(tmp_path, monkeypatch,
                     lambda dl, audio: dl.DownloadResult(audio, 337.5), seen)
    stages.do_download(sid)
    assert seen["expected_duration"] == pytest.approx(LENGTH)
    assert json.loads(m.get_song(sid)["audio_provenance"]) == \
        {"via": "soundcloud", "url": SC_URL}


def test_fallback_download_records_what_it_substituted(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Massive", artist="Drake", source_url=SC_URL,
                        duration_secs=LENGTH, source="soundcloud")
    prov = {"via": "youtube_fallback", "url": YT_RECORD, "title": "Drake - Massive",
            "uploader": "Drake", "score": 1.0, "duration_secs": 337.5,
            "expected_secs": 336.9}
    stages = _stages(tmp_path, monkeypatch,
                     lambda dl, audio: dl.DownloadResult(audio, 337.5, YT_RECORD, prov), {})
    stages.do_download(sid)
    row = m.get_song(sid)
    recorded = json.loads(row["audio_provenance"])
    assert recorded["via"] == "youtube_fallback" and recorded["title"] == "Drake - Massive"
    assert recorded["url"] == "https://youtube.com/watch?v=ay1l_u6vltY"
    assert row["source_url"] == "https://youtube.com/watch?v=ay1l_u6vltY"
    assert row["origin_url"] == SC_URL


def test_redownloading_a_manual_pick_keeps_saying_it_was_yours(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Massive", source_url=SC_URL, duration_secs=LENGTH)
    pick = "https://youtube.com/watch?v=ay1l_u6vltY"
    m.update_song_url(sid, pick, provenance={"via": "manual", "url": pick, "title": "Drake - Massive"})
    stages = _stages(tmp_path, monkeypatch,
                     lambda dl, audio: dl.DownloadResult(audio, 337.5), {})
    stages.do_download(sid)
    assert json.loads(m.get_song(sid)["audio_provenance"])["title"] == "Drake - Massive"


# ── suspect audio ───────────────────────────────────────────────────────────

def test_suspect_audio_finds_unchecked_and_wrong_length_substitutes(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    from api.workers import bulk_worker

    bug = _massive_as_the_bug_left_it(m)
    wrong_len = m.upsert_song(title="Wrong", source_url="https://soundcloud.com/a/wrong",
                              duration_secs=300, status="analysed")
    m.update_song_url(wrong_len, "https://youtube.com/watch?v=aaaaaaaaaaa")
    m.update_song_status(wrong_len, "analysed")
    m.update_song_duration(wrong_len, 200)

    mine = m.upsert_song(title="Mine", source_url="https://soundcloud.com/a/mine",
                         duration_secs=300)
    m.update_song_url(mine, "https://youtube.com/watch?v=bbbbbbbbbbb",
                      provenance={"via": "manual", "url": "https://youtube.com/watch?v=bbbbbbbbbbb"})
    m.update_song_status(mine, "analysed")
    m.update_song_duration(mine, 200)

    fine = m.upsert_song(title="Fine", source_url="https://soundcloud.com/a/fine",
                         duration_secs=300, status="analysed")
    _massive_as_the_bug_left_it(m, status="queued",           # not downloaded yet
                                url="https://youtube.com/watch?v=ccccccccccc")

    assert bulk_worker.suspect_audio_ids() == [bug, wrong_len]
    assert bulk_worker.staleness()["suspect_audio"] == 2
    assert fine not in bulk_worker.stale_song_ids("redownload_suspect")


def _capture_job(monkeypatch):
    from api import jobs, queue_runner
    done, queued = {}, []
    monkeypatch.setattr(jobs, "update", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "fail", lambda job_id, msg, *a: done.update(failed=msg))
    monkeypatch.setattr(jobs, "done", lambda job_id, result: done.update(result))
    monkeypatch.setattr(queue_runner, "enqueue_song", lambda sid, **_kw: queued.append(sid) or "j")
    return done, queued


def test_redownload_restores_the_soundcloud_link_and_credited_artist(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    from api.workers import bulk_worker
    from ingest import soundcloud_browse

    sid = _massive_as_the_bug_left_it(m)
    remix = tmp_path / "Massive_octobersveryown.mp3"
    remix.write_bytes(b"the remix")
    conn = m.get_conn()
    conn.execute("UPDATE songs SET raw_path=? WHERE id=?", (str(remix), sid))
    conn.commit()
    conn.close()

    asked = []
    monkeypatch.setattr(soundcloud_browse, "get_tracks", lambda ids, **k: asked.extend(ids) or [
        {"track_id": "1289061475", "source_url": SC_URL, "duration_secs": LENGTH,
         "artist": "Drake"}])
    done, queued = _capture_job(monkeypatch)

    bulk_worker.run("job", "redownload_suspect", [sid])

    row = m.get_song(sid)
    assert asked == ["1289061475"]
    assert (row["source_url"], row["origin_url"], row["artist"]) == (SC_URL, SC_URL, "Drake")
    assert row["source"] == "soundcloud"
    assert row["origin_duration_secs"] == pytest.approx(LENGTH)
    assert row["status"] == "queued" and row["audio_provenance"] is None
    assert not remix.exists()
    assert queued == [sid] and done["queued"] == 1 and done["failed"] == 0
    assert bulk_worker.suspect_audio_ids() == []


def test_redownload_reports_a_track_whose_link_cannot_be_recovered(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    from api.workers import bulk_worker
    from ingest import soundcloud_browse

    sid = _massive_as_the_bug_left_it(m)
    monkeypatch.setattr(soundcloud_browse, "get_tracks", lambda ids, **k: [])
    done, queued = _capture_job(monkeypatch)

    bulk_worker.run("job", "redownload_suspect", [sid])

    assert queued == [] and done["failed"] == 1
    assert "SoundCloud link" in done["reasons"][0]
    assert m.get_song(sid)["source_url"] == YT_REMIX      # untouched, not guessed


# ── routes ──────────────────────────────────────────────────────────────────

def _tracks(tmp_path, monkeypatch):
    m = _setup(tmp_path, monkeypatch)
    from api import queue_runner
    monkeypatch.setattr(queue_runner, "enqueue_song", lambda sid, **_kw: f"job-{sid}")
    from api.routes import tracks
    importlib.reload(tracks)
    return m, tracks


def test_audio_candidates_route_annotates_against_the_origin_length(tmp_path, monkeypatch):
    m, tracks = _tracks(tmp_path, monkeypatch)
    import downloader.download as dl
    sid = m.upsert_song(title="Massive", artist="Drake", source_url=SC_URL,
                        duration_secs=LENGTH)
    m.update_song_url(sid, "https://youtube.com/watch?v=ay1l_u6vltY")
    seen = {}

    def fake(title, artist, expected=None, **kw):
        seen.update(title=title, artist=artist, expected=expected, **kw)
        return [{"url": YT_RECORD, "title": "Drake - Massive", "uploader": "Drake",
                 "duration_secs": 338.0, "score": 1.0, "passes": True, "reason": "",
                 "duration_delta": 1.1},
                {"url": "https://www.youtube.com/watch?v=ArTTyLYpi70",
                 "title": "Drake - Massive (OCTANE Remix)", "uploader": "Clebi",
                 "duration_secs": 181.0, "score": 0.85, "passes": False,
                 "reason": "a remix or rework, not the original record",
                 "duration_delta": -155.9}]

    monkeypatch.setattr(dl, "youtube_candidates", fake)
    out = tracks.audio_candidates(sid)
    assert seen == {"title": "Massive", "artist": "Drake",
                    "expected": pytest.approx(LENGTH), "exhaustive": True}
    assert out["expected_duration"] == pytest.approx(LENGTH)
    assert out["origin_url"] == SC_URL
    assert [c["in_use"] for c in out["candidates"]] == [True, False]

    with pytest.raises(HTTPException) as err:
        tracks.audio_candidates(9999)
    assert err.value.status_code == 404


def test_picking_an_upload_records_it_as_manual(tmp_path, monkeypatch):
    m, tracks = _tracks(tmp_path, monkeypatch)
    sid = m.upsert_song(title="Massive", artist="Drake", source_url=SC_URL,
                        duration_secs=LENGTH)
    tracks.change_url(sid, tracks.UrlUpdate(
        source_url=YT_RECORD,
        pick={"title": "Drake - Massive", "uploader": "Drake", "duration_secs": 338.0,
              "ignored": "x"}))
    row = m.get_song(sid)
    assert row["origin_url"] == SC_URL
    assert json.loads(row["audio_provenance"]) == {
        "via": "manual", "url": "https://youtube.com/watch?v=ay1l_u6vltY",
        "title": "Drake - Massive", "uploader": "Drake", "duration_secs": 338.0}


# ── "✓ Sounds right" ────────────────────────────────────────────────────────

def test_confirming_unverified_audio_settles_it(tmp_path, monkeypatch):
    m, tracks = _tracks(tmp_path, monkeypatch)
    from api.workers import bulk_worker

    sid = _massive_as_the_bug_left_it(m)
    assert bulk_worker.suspect_audio_ids() == [sid]

    with pytest.raises(HTTPException) as err:            # nothing listened to yet
        tracks.confirm_track_audio(sid)
    assert err.value.status_code == 409

    m.update_song_status(sid, "analysed", raw_path="/data/audio/full_song/m.mp3")
    out = tracks.confirm_track_audio(sid)
    assert out["audio_provenance"] == {"via": "youtube", "url": YT_REMIX, "confirmed": True}
    assert bulk_worker.suspect_audio_ids() == []

    with pytest.raises(HTTPException) as err:
        tracks.confirm_track_audio(9999)
    assert err.value.status_code == 404


def test_a_confirmation_keeps_the_substitute_details_and_survives_the_same_link(tmp_path, monkeypatch):
    m, tracks = _tracks(tmp_path, monkeypatch)
    yt = "https://youtube.com/watch?v=ay1l_u6vltY"
    sid = m.upsert_song(title="Massive", artist="Drake", source_url=SC_URL,
                        duration_secs=LENGTH)
    m.update_song_url(sid, yt, provenance={"via": "youtube_fallback", "url": yt,
                                           "title": "Drake - Massive"})
    m.update_song_status(sid, "analysed", raw_path="/data/audio/full_song/m.mp3")

    prov = tracks.confirm_track_audio(sid)["audio_provenance"]
    assert (prov["via"], prov["title"], prov["confirmed"]) == \
        ("youtube_fallback", "Drake - Massive", True)

    # Downloading the SAME link again is the same audio: still confirmed.
    stages = _stages(tmp_path, monkeypatch,
                     lambda dl, audio: dl.DownloadResult(audio, 337.1), {})
    stages.do_download(sid)
    assert json.loads(m.get_song(sid)["audio_provenance"])["confirmed"] is True

    # A different link is different audio: the confirmation does not carry over.
    m.update_song_url(sid, SC_URL)
    stages.do_download(sid)
    assert "confirmed" not in json.loads(m.get_song(sid)["audio_provenance"])
