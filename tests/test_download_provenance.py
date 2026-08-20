"""When SoundCloud won't serve a track, the downloader searches YouTube and
keeps that audio instead. These cover recording *where the file actually came
from*, so the row stops claiming SoundCloud provenance for YouTube audio (which
also sent every later re-download back to the URL that never worked).
"""
import importlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from downloader.download import _resolved_youtube_url  # noqa: E402

# Real yt-dlp output for `ytsearch3:...` — the "[youtube] Extracting URL" line is
# the only record of which upload a search actually resolved to.
_REAL_SEARCH_OUTPUT = """[youtube:search] Extracting URL: ytsearch3:jeonghyeon On The World
[download] Downloading playlist: jeonghyeon On The World
[youtube:search] Playlist jeonghyeon On The World: Downloading 1 items
[download] Downloading item 1 of 1
[youtube] Extracting URL: https://www.youtube.com/watch?v=9zcfYTxeTIo
[youtube] 9zcfYTxeTIo: Downloading webpage
[info] 9zcfYTxeTIo: Downloading 1 format(s): 299+251
"""


def test_reads_the_resolved_video_from_search_output():
    assert _resolved_youtube_url(_REAL_SEARCH_OUTPUT) == \
        "https://www.youtube.com/watch?v=9zcfYTxeTIo"


def test_falls_back_to_the_bare_video_id_line():
    assert _resolved_youtube_url("[youtube] dQw4w9WgXcQ: Downloading webpage") == \
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_returns_none_when_no_video_is_named():
    # A SoundCloud download names no YouTube video — the caller must then leave
    # the recorded source_url alone rather than invent one.
    assert _resolved_youtube_url(
        "[soundcloud] gudvibrations/feeling-gud: Downloading info JSON") is None
    assert _resolved_youtube_url("") is None


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.workers import stages
    importlib.reload(stages)
    return models, stages


def _add_song(models, url, title="T", artist="A"):
    conn = models.get_conn()
    conn.execute("INSERT INTO songs (title, artist, source_url, source) VALUES (?,?,?,?)",
                 (title, artist, url, "soundcloud"))
    conn.commit()
    sid = conn.execute("SELECT id FROM songs WHERE source_url=?", (url,)).fetchone()["id"]
    conn.close()
    return sid


def _fake_download(stages, monkeypatch, tmp_path, actual_url):
    """Stub download_track so do_download runs without touching the network.
    ``actual_url`` is what the downloader reports the audio really came from."""
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    import downloader.download as dl

    def fake(song_id, title, source_url, artist="", on_progress=None):
        return dl.DownloadResult(audio, 200.0, actual_url)

    monkeypatch.setattr(dl, "download_track", fake)
    return audio


def _source_of(models, sid):
    conn = models.get_conn()
    row = conn.execute("SELECT source_url, source FROM songs WHERE id=?", (sid,)).fetchone()
    conn.close()
    return row["source_url"], row["source"]


def test_youtube_fallback_updates_the_song_source(tmp_path, monkeypatch):
    models, stages = _setup(tmp_path, monkeypatch)
    sid = _add_song(models, "https://soundcloud.com/x/gated")
    _fake_download(stages, monkeypatch, tmp_path,
                   "https://www.youtube.com/watch?v=9zcfYTxeTIo")

    stages.do_download(sid)

    url, source = _source_of(models, sid)
    # Stored normalised, the same shape ingest writes, so dedup by URL still hits.
    assert url == "https://youtube.com/watch?v=9zcfYTxeTIo"
    assert source == "youtube"


def test_source_untouched_when_download_reports_no_url(tmp_path, monkeypatch):
    models, stages = _setup(tmp_path, monkeypatch)
    sid = _add_song(models, "https://soundcloud.com/x/fine")
    _fake_download(stages, monkeypatch, tmp_path, None)

    stages.do_download(sid)

    assert _source_of(models, sid)[0] == "https://soundcloud.com/x/fine"


def test_collision_with_another_song_keeps_the_original_url(tmp_path, monkeypatch):
    """songs.source_url is UNIQUE. If the fallback URL already belongs to another
    song the download still succeeded — keep the old URL rather than fail."""
    models, stages = _setup(tmp_path, monkeypatch)
    _add_song(models, "https://youtube.com/watch?v=9zcfYTxeTIo", title="Other")
    sid = _add_song(models, "https://soundcloud.com/x/gated")
    _fake_download(stages, monkeypatch, tmp_path,
                   "https://www.youtube.com/watch?v=9zcfYTxeTIo")

    out = stages.do_download(sid)           # must not raise

    assert out["path"]
    assert _source_of(models, sid)[0] == "https://soundcloud.com/x/gated"
