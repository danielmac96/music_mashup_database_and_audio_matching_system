"""Recovering metadata a throttled ingest never fetched.

Regression: a 206-track mix import ran ENRICH_WORKERS parallel yt-dlp metadata
fetches against SoundCloud, most of them were throttled, and 119 songs were
saved with genre='', plays=0, release_year=0, track_id='' and
metadata_partial=1. Their audio, stems and analysis all succeeded, so nothing
ever complained — the library simply showed four empty columns, and nothing in
the app ever went back for them.

Covers the repair (a metadata-only writer and the bulk action that drives it),
and the ways the same rows were being produced silently.
"""
import importlib
import sys
import time as _time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SC_URL = "https://soundcloud.com/audien/audien-hindsight-preview"
YT_URL = "https://youtube.com/watch?v=nmy7ZsKE0Jc"
TAGS_JSON = '["progressive"]'


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    return models


def _partial_song(models, url=SC_URL, **cols):
    """A row exactly as a throttled ingest leaves it: real link and audio, no
    descriptive metadata, flagged partial."""
    row = dict(title="Hindsight", artist="audien", source_url=url,
               source="soundcloud", duration_secs=222.0, genre="",
               status="analysed", raw_path="/audio/full_song/hindsight.mp3",
               artist_id="", track_id="", upload_date="", likes=0, reposts=0,
               comments=0, plays=0, thumbnail="", metadata_partial=1, tags="",
               release_year=0, origin_url=url, origin_duration_secs=222.0)
    row.update(cols)
    conn = models.get_conn()
    cur = conn.execute(
        "INSERT INTO songs (%s) VALUES (%s)"
        % (",".join(row), ",".join("?" * len(row))),
        list(row.values()))
    conn.commit()
    song_id = cur.lastrowid
    conn.close()
    return song_id


FETCHED = {
    "title": "Hindsight", "artist": "Audien", "artist_id": "audien",
    "track_id": "141929808", "duration_secs": 222.0, "duration_str": "3:42",
    "source_url": SC_URL, "upload_date": "20140130", "likes": 18794,
    "reposts": 900, "comments": 120, "plays": 1088929,
    "thumbnail": "https://i1.sndcdn.com/a.jpg", "genre": "Progressive",
    "tags": TAGS_JSON, "release_year": 2014,
}


def _row(models, song_id):
    conn = models.get_conn()
    r = dict(conn.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone())
    conn.close()
    return r


# ── The writer ────────────────────────────────────────────────────────────────

def test_update_song_metadata_fills_the_blanks(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models)

    models.update_song_metadata(sid, FETCHED)

    row = _row(models, sid)
    assert row["genre"] == "Progressive"
    assert row["plays"] == 1088929
    assert row["likes"] == 18794
    assert row["release_year"] == 2014
    assert row["track_id"] == "141929808"
    assert row["upload_date"] == "20140130"
    assert row["tags"] == TAGS_JSON
    assert row["metadata_partial"] == 0


def test_update_song_metadata_never_touches_audio_or_status(tmp_path, monkeypatch):
    """The whole reason this is not upsert_song: that sets status='queued' and
    would send an analysed library back through Demucs."""
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models)
    before = _row(models, sid)

    # A fetch that disagrees about everything the pipeline owns.
    models.update_song_metadata(sid, dict(
        FETCHED, source_url="https://soundcloud.com/other/thing",
        duration_secs=30.0, status="queued", raw_path="", last_error="boom"))

    after = _row(models, sid)
    for col in ("status", "source_url", "source", "raw_path", "duration_secs",
                "origin_url", "origin_duration_secs", "last_error"):
        assert after[col] == before[col], col


def test_a_blank_fetched_value_never_overwrites_a_stored_one(tmp_path, monkeypatch):
    """unknown is not bad: a re-fetch that returns no genre means the upload has
    none, not that the genre we hold should be thrown away."""
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models, genre="Hip Hop", plays=500, release_year=2019,
                        metadata_partial=0)

    models.update_song_metadata(sid, {"genre": "", "plays": 0, "likes": 0,
                                      "release_year": 0, "tags": ""})

    row = _row(models, sid)
    assert row["genre"] == "Hip Hop"
    assert row["plays"] == 500
    assert row["release_year"] == 2019


def test_a_genreless_upload_still_clears_the_partial_flag(tmp_path, monkeypatch):
    """Plenty of SoundCloud uploads set no genre. The fetch succeeded, so the
    row must stop queueing itself for the same backfill forever."""
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models)

    models.update_song_metadata(sid, dict(FETCHED, genre="", tags=""))

    row = _row(models, sid)
    assert row["genre"] == ""
    assert row["plays"] == 1088929
    assert row["metadata_partial"] == 0
    assert models.partial_metadata_song_ids() == []


def test_title_and_artist_are_only_filled_when_blank(tmp_path, monkeypatch):
    """A mix tracklist's credited artist beats a SoundCloud uploader handle."""
    models = _setup(tmp_path, monkeypatch)
    kept = _partial_song(models, title="Hindsight", artist="Audien")
    filled = _partial_song(models, url=SC_URL + "2", title="Unknown", artist="")

    noisy = dict(FETCHED, artist="audienmusic",
                 title="AUDIEN - HINDSIGHT (PREVIEW)")
    models.update_song_metadata(kept, noisy)
    models.update_song_metadata(filled, noisy)

    assert _row(models, kept)["artist"] == "Audien"
    assert _row(models, kept)["title"] == "Hindsight"
    assert _row(models, filled)["artist"] == "audienmusic"
    assert _row(models, filled)["title"] == "AUDIEN - HINDSIGHT (PREVIEW)"


def test_partial_metadata_song_ids_finds_exactly_the_flagged_rows(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    a = _partial_song(models)
    _partial_song(models, url=SC_URL + "2", metadata_partial=0, genre="House")
    b = _partial_song(models, url=SC_URL + "3")

    assert models.partial_metadata_song_ids() == [a, b]


# ── upsert_song no longer undoes the repair ───────────────────────────────────

def test_a_sparse_reupsert_cannot_blank_recovered_metadata(tmp_path, monkeypatch):
    """A mix re-ingest, a flat playlist seed and a legacy crate payload all
    re-upsert with '' / 0 for every descriptive column. That used to wipe the
    row — silently undoing this whole backfill."""
    models = _setup(tmp_path, monkeypatch)
    models.upsert_song(title="Hindsight", artist="Audien", source_url=SC_URL,
                       genre="Progressive", plays=1088929, likes=18794,
                       reposts=900, comments=120, upload_date="20140130",
                       track_id="141929808", artist_id="audien",
                       thumbnail="https://i1.sndcdn.com/a.jpg",
                       tags=TAGS_JSON, release_year=2014)

    # The sparse shape: title + artist + url and nothing else.
    sid = models.upsert_song(title="Hindsight", artist="Audien", source_url=SC_URL)

    row = _row(models, sid)
    assert row["genre"] == "Progressive"
    assert row["plays"] == 1088929
    assert row["likes"] == 18794
    assert row["reposts"] == 900
    assert row["comments"] == 120
    assert row["upload_date"] == "20140130"
    assert row["track_id"] == "141929808"
    assert row["artist_id"] == "audien"
    assert row["thumbnail"] == "https://i1.sndcdn.com/a.jpg"
    assert row["release_year"] == 2014


def test_a_rich_reupsert_still_updates_metadata(tmp_path, monkeypatch):
    """The guard must not freeze a row: a real fetch still corrects it."""
    models = _setup(tmp_path, monkeypatch)
    models.upsert_song(title="X", source_url=SC_URL, genre="House", plays=10)
    sid = models.upsert_song(title="X", source_url=SC_URL, genre="Progressive",
                             plays=1088929)
    row = _row(models, sid)
    assert row["genre"] == "Progressive"
    assert row["plays"] == 1088929


# ── The bulk action ───────────────────────────────────────────────────────────

def _bulk(monkeypatch):
    from api import jobs
    importlib.reload(jobs)
    from api.workers import bulk_worker
    importlib.reload(bulk_worker)
    monkeypatch.setattr(bulk_worker.time, "sleep", lambda s: None)
    return bulk_worker, jobs


def test_staleness_counts_and_selects_the_partial_rows(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    a = _partial_song(models)
    _partial_song(models, url=SC_URL + "2", metadata_partial=0)
    bulk_worker, _ = _bulk(monkeypatch)

    assert bulk_worker.staleness()["missing_metadata"] == 1
    assert bulk_worker.stale_song_ids("metadata") == [a]
    # "all" means every partial row too: re-fetching metadata a track already
    # has is a round trip that changes nothing.
    assert bulk_worker.all_song_ids("metadata") == [a]


def test_backfill_refreshes_what_it_can_and_survives_the_rest(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    ok = _partial_song(models)
    empty = _partial_song(models, url=SC_URL + "2")
    boom = _partial_song(models, url=SC_URL + "3")
    bulk_worker, jobs = _bulk(monkeypatch)

    def fake_enrich(url):
        if url.endswith("2"):
            return None                      # SoundCloud returned nothing
        if url.endswith("3"):
            raise RuntimeError("connection reset")
        return FETCHED

    monkeypatch.setattr("ingest.soundcloud.enrich_track", fake_enrich)

    job_id = jobs.new_job(kind="bulk", message="")
    bulk_worker.run(job_id, "metadata", [ok, empty, boom])

    job = jobs.get(job_id)
    assert job["status"] == "completed"
    assert job["result"]["updated"] == 1
    assert job["result"]["failed"] == 2
    assert len(job["result"]["reasons"]) == 2

    assert _row(models, ok)["genre"] == "Progressive"
    assert _row(models, ok)["metadata_partial"] == 0
    # A track that could not be fetched keeps its flag, so the next run retries it.
    assert _row(models, empty)["metadata_partial"] == 1
    assert _row(models, boom)["metadata_partial"] == 1


def test_a_403_falls_back_to_the_v2_resolver(tmp_path, monkeypatch):
    """yt-dlp gets a flat 403 on a slice of the SoundCloud catalogue — 20 of the
    119 rows this was written for. The v2 browse layer answers for those same
    links, and already returns the canonical row shape."""
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models)
    bulk_worker, jobs = _bulk(monkeypatch)

    monkeypatch.setattr("ingest.soundcloud.enrich_track", lambda url: None)
    monkeypatch.setattr("ingest.soundcloud_browse.resolve",
                        lambda url: {"kind": "track", "item": FETCHED})

    bulk_worker.run(jobs.new_job(kind="bulk", message=""), "metadata", [sid])

    row = _row(models, sid)
    assert row["genre"] == "Progressive"
    assert row["plays"] == 1088929
    assert row["metadata_partial"] == 0


def test_the_v2_fallback_is_only_for_soundcloud_and_only_for_tracks(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    bulk_worker, _ = _bulk(monkeypatch)
    called = []

    def spy(url):
        called.append(url)
        return {"kind": "playlist", "item": {"title": "a set"}}

    monkeypatch.setattr("ingest.soundcloud_browse.resolve", spy)

    # A YouTube link never reaches the SoundCloud API at all.
    assert bulk_worker._metadata_via_v2(YT_URL) is None
    assert called == []
    # A permalink that resolves to a set is not a track's metadata.
    assert bulk_worker._metadata_via_v2(SC_URL) is None
    assert called == [SC_URL]


def test_a_failing_v2_fallback_is_just_no_fallback(tmp_path, monkeypatch):
    """The breaker tripping must report the track, not sink the batch."""
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models)
    bulk_worker, jobs = _bulk(monkeypatch)

    monkeypatch.setattr("ingest.soundcloud.enrich_track", lambda url: None)

    def boom(url):
        raise RuntimeError("circuit breaker open")

    monkeypatch.setattr("ingest.soundcloud_browse.resolve", boom)

    job_id = jobs.new_job(kind="bulk", message="")
    bulk_worker.run(job_id, "metadata", [sid])

    assert jobs.get(job_id)["status"] == "completed"
    assert jobs.get(job_id)["result"]["failed"] == 1
    assert _row(models, sid)["metadata_partial"] == 1


def test_backfill_refetches_the_imported_link_not_the_substitute(tmp_path, monkeypatch):
    """A YouTube substitute rewrites source_url; the SoundCloud metadata lives
    on origin_url, the link that was actually imported."""
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models, source_url=YT_URL, origin_url=SC_URL)
    bulk_worker, jobs = _bulk(monkeypatch)

    seen = []

    def fake_enrich(url):
        seen.append(url)
        return FETCHED

    monkeypatch.setattr("ingest.soundcloud.enrich_track", fake_enrich)
    bulk_worker.run(jobs.new_job(kind="bulk", message=""), "metadata", [sid])

    assert seen == [SC_URL]
    # …and the substitute link itself is untouched.
    assert _row(models, sid)["source_url"] == YT_URL


def test_backfill_does_not_requeue_anything(tmp_path, monkeypatch):
    """The failure this whole change exists to avoid: an analysed library sent
    back through download → Demucs → analysis."""
    models = _setup(tmp_path, monkeypatch)
    sid = _partial_song(models)
    bulk_worker, jobs = _bulk(monkeypatch)

    enqueued = []
    monkeypatch.setattr(bulk_worker.queue_runner, "enqueue_song", enqueued.append)
    monkeypatch.setattr("ingest.soundcloud.enrich_track", lambda url: FETCHED)

    bulk_worker.run(jobs.new_job(kind="bulk", message=""), "metadata", [sid])

    assert enqueued == []
    assert _row(models, sid)["status"] == "analysed"


# ── Root cause: the fetch itself ──────────────────────────────────────────────

def test_a_throttled_fetch_is_retried(monkeypatch):
    import ingest.soundcloud as sc
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    attempts = []

    def flaky(url):
        attempts.append(url)
        if len(attempts) < 3:
            return [], "ERROR: unable to download: HTTP Error 429: Too Many Requests"
        return [{"title": "ok"}], ""

    monkeypatch.setattr(sc, "_fetch_via_ytdlp_once", flaky)
    assert sc._fetch_via_ytdlp("u") == [{"title": "ok"}]
    assert len(attempts) == 3


def test_a_removed_track_is_not_retried(monkeypatch):
    """Sleeping through three rounds of backoff per dead track would make a
    large import unusable."""
    import ingest.soundcloud as sc
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    attempts = []

    def gone(url):
        attempts.append(url)
        return [], "ERROR: [soundcloud] 1: Unable to download JSON metadata: HTTP Error 404: Not Found"

    monkeypatch.setattr(sc, "_fetch_via_ytdlp_once", gone)
    assert sc._fetch_via_ytdlp("u") == []
    assert len(attempts) == 1


@pytest.mark.parametrize("stderr, transient", [
    ("ERROR: HTTP Error 429: Too Many Requests", True),
    ("ERROR: HTTP Error 503: Service Unavailable", True),
    ("ERROR: The read operation timed out", True),
    ("ERROR: HTTP Error 404: Not Found", False),
    ("ERROR: HTTP Error 403: Forbidden", False),
    ("ERROR: this track is private", False),
    ("", False),
])
def test_transient_classification(stderr, transient):
    import ingest.soundcloud as sc
    assert sc._is_transient(stderr) is transient


# ── Root cause: the silent producers ──────────────────────────────────────────

def test_a_failed_hydration_is_not_mistaken_for_real_metadata(tmp_path, monkeypatch):
    """The hydrator marks a row hydrated even when the fetch returned nothing,
    so the session can complete. Ingest used to read that as "is rich" and save
    the blank row with metadata_partial=0 — invisible to any backfill."""
    _setup(tmp_path, monkeypatch)
    from api.routes import playlists
    importlib.reload(playlists)

    flat = {"source_url": SC_URL, "title": "Hindsight", "genre": "", "plays": 0,
            "hydrated": True, "enriched": False}
    monkeypatch.setattr(playlists, "enrich_track", lambda url: None)
    monkeypatch.setattr(playlists.preview_hydrator, "cache_get", lambda url: None)

    _, is_rich = playlists._resolve_metadata(dict(flat))
    assert is_rich is False

    # A row that really was enriched still skips the refetch…
    _, is_rich = playlists._resolve_metadata(dict(flat, enriched=True))
    assert is_rich is True
    # …as does a canonical Discover/crate row, which sets no `enriched` key.
    bare = dict(flat)
    bare.pop("enriched")
    _, is_rich = playlists._resolve_metadata(bare)
    assert is_rich is True


def test_hydrator_records_whether_it_actually_got_anything(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from api import preview_hydrator
    importlib.reload(preview_hydrator)
    monkeypatch.setattr("ingest.soundcloud.enrich_track", lambda url: None)

    pid = preview_hydrator.start([{"source_url": SC_URL, "title": "Hindsight"}])
    session = preview_hydrator.get(pid)
    for _ in range(200):
        if session["done"]:
            break
        _time.sleep(0.02)
        session = preview_hydrator.get(pid)

    assert session["done"] is True
    row = session["tracks"][0]
    assert row["hydrated"] is True
    assert row["enriched"] is False


def test_a_legacy_crate_payload_is_not_stamped_as_canonical():
    """crate_payloads rebuilds a pre-payload_json item from three columns.
    Stamping that hydrated would save it blank AND unflagged."""
    from api.routes import crates
    assert crates._is_canonical(
        {"source_url": SC_URL, "title": "x", "artist": "y"}) is False
    assert crates._is_canonical(
        {"source_url": SC_URL, "title": "x", "artist": "y",
         "genre": "", "plays": 0, "release_year": 0}) is True


# ── Frontend contract ─────────────────────────────────────────────────────────

def test_bulk_bar_offers_the_metadata_refresh():
    jsx = (REPO_ROOT / "frontend/src/components/BulkReprocess.jsx").read_text(
        encoding="utf-8")
    assert "missing_metadata" in jsx
    assert 'run("metadata", "stale")' in jsx
    assert 'badge("metadata")' in jsx
    # The bar must not render on a library with nothing to fix.
    assert "!noMeta" in jsx
