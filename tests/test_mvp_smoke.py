"""
End-to-end smoke test for the MVP pipeline.

Mocks every external (yt-dlp, Demucs, librosa) so the test runs in a few
seconds on any laptop with no GPU and no network.

Asserts:
  - Every track lands in `analysed` status.
  - DB rows exist in songs / stems / features.
  - The expected files are written under the overridden audio root.
  - A second run is fully idempotent (no calls to the externals).
  - A track that fails at one stage gets the matching error_* status without
    killing the rest of the run.
"""
from pathlib import Path
import importlib
import sys

import pytest


# Ensure the repo root is on sys.path so `import pipeline` works under pytest.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point the audio library and DB at tmp_path; reload config so the
    new env vars take effect, then return the (audio_root, db_path) pair."""
    audio_root = tmp_path / "audio"
    db_path    = tmp_path / "mashup.db"
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(audio_root))
    monkeypatch.setenv("MASHUP_DB_PATH",    str(db_path))

    # Reset cached modules so that config.py re-reads env vars on import.
    for mod in (
        "config", "pipeline",
        "database.models",
        "ingest.soundcloud", "downloader.download",
        "stems.separate", "analysis.analyze",
    ):
        sys.modules.pop(mod, None)

    return audio_root, db_path


@pytest.fixture
def fake_tracks():
    return [
        {
            "title": "Sunrise", "artist": "Test Artist A",
            "source_url": "https://soundcloud.com/test/sunrise",
            "duration_secs": 200.0, "duration_str": "3:20",
            "genre": "Electronic", "artist_id": "tA", "track_id": "1",
            "upload_date": "20250101",
            "likes": 10, "reposts": 2, "comments": 1, "plays": 100,
            "thumbnail": "",
        },
        {
            "title": "Midnight", "artist": "Test Artist B",
            "source_url": "https://soundcloud.com/test/midnight",
            "duration_secs": 180.0, "duration_str": "3:00",
            "genre": "Hip Hop", "artist_id": "tB", "track_id": "2",
            "upload_date": "20250102",
            "likes": 5, "reposts": 1, "comments": 0, "plays": 50,
            "thumbnail": "",
        },
    ]


@pytest.fixture
def call_log():
    return {"download": 0, "separate": 0, "analyze": 0, "ingest": 0}


@pytest.fixture
def mocked_pipeline(isolated_env, monkeypatch, fake_tracks, call_log):
    """Import pipeline AFTER env is set, then monkeypatch each external."""
    audio_root, db_path = isolated_env
    pipeline = importlib.import_module("pipeline")

    # 1) Ingest: stub fetch_playlist to return our fixture tracks.
    def fake_fetch_playlist(url):
        call_log["ingest"] += 1
        return list(fake_tracks)

    monkeypatch.setattr(pipeline, "fetch_playlist", fake_fetch_playlist)

    # 2) Download: write a tiny fake MP3 to the expected location.
    from downloader.download import DownloadResult
    from config import RAW_DIR

    def fake_download_track(song_id, title, source_url, artist=""):
        call_log["download"] += 1
        safe = "".join(c if c.isalnum() else "_" for c in title)[:40]
        safe_artist = "".join(c if c.isalnum() else "_" for c in artist)[:30]
        out = RAW_DIR / f"{safe}_{safe_artist}.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xfb" + b"\x00" * 1024)  # tiny stub
        return DownloadResult(out, duration_secs=200.0)

    monkeypatch.setattr(pipeline, "download_track", fake_download_track)

    # 3) Stems: write tiny WAV stubs and return the dict separate() would.
    from config import VOCALS_DIR, INSTRUMENTALS_DIR
    import re

    def fake_separate(song_id, title, audio_path, artist=""):
        call_log["separate"] += 1
        safe_title  = re.sub(r'[^\w]', '_', title)[:40]
        safe_artist = re.sub(r'[^\w]', '_', artist)[:30]
        name = f"{safe_title}_{safe_artist}"
        v = VOCALS_DIR        / f"{name}_vocals.wav"
        i = INSTRUMENTALS_DIR / f"{name}_instrumental.wav"
        v.parent.mkdir(parents=True, exist_ok=True)
        i.parent.mkdir(parents=True, exist_ok=True)
        v.write_bytes(b"RIFF" + b"\x00" * 100)
        i.write_bytes(b"RIFF" + b"\x00" * 100)
        return {"vocals": v, "instrumental": i}

    monkeypatch.setattr(pipeline, "separate", fake_separate)

    # 4) Analysis: return a fully populated stub features dict.
    def fake_analyze_file(audio_path, trim_secs=None):
        call_log["analyze"] += 1
        return {
            "bpm": 120.0, "bpm_confidence": 0.5,
            "key": "C", "mode": "major", "camelot": "8B",
            "loudness_rms": 0.1, "energy": 5.0,
            "mfcc": [0.0] * 13,
            "spectral_centroid": 2000.0,
            "spectral_rolloff":  4000.0,
            "zero_crossing_rate": 0.05,
        }

    monkeypatch.setattr(pipeline, "analyze_file", fake_analyze_file)

    return pipeline


# ── Tests ───────────────────────────────────────────────────────────────────


def test_pipeline_end_to_end(mocked_pipeline, isolated_env, call_log):
    audio_root, db_path = isolated_env
    pipeline = mocked_pipeline

    pipeline.run_ingest("https://soundcloud.com/fake/playlist")
    pipeline.run_download()
    pipeline.run_stems()
    pipeline.run_analysis()

    from database.models import get_all_songs, get_features_for_song

    songs = get_all_songs()
    assert len(songs) == 2, f"expected 2 songs, got {len(songs)}"
    assert all(s["status"] == "analysed" for s in songs), \
        f"statuses: {[s['status'] for s in songs]}"

    # Files on disk
    assert db_path.exists()
    raws = list((audio_root / "full_song").glob("*.mp3"))
    vocals = list((audio_root / "vocals").glob("*.wav"))
    insts = list((audio_root / "instrumentals").glob("*.wav"))
    assert len(raws) == 2 and len(vocals) == 2 and len(insts) == 2

    # Features rows for each stem of each song
    for s in songs:
        for stem in ("full", "vocals", "instrumental"):
            feat = get_features_for_song(s["id"], stem_type=stem)
            assert feat is not None, f"no features row for song {s['id']} stem {stem}"
            assert feat["bpm"] == 120.0
            assert feat["camelot"] == "8B"

    # Externals were called the expected number of times
    assert call_log["download"] == 2
    assert call_log["separate"] == 2
    assert call_log["analyze"] == 6  # 2 songs × 3 stems


def test_idempotent_rerun(mocked_pipeline, call_log):
    pipeline = mocked_pipeline

    pipeline.run_ingest("https://soundcloud.com/fake/playlist")
    pipeline.run_download()
    pipeline.run_stems()
    pipeline.run_analysis()

    before = dict(call_log)

    # Second run: every stage should be a no-op.
    pipeline.run_download()
    pipeline.run_stems()
    pipeline.run_analysis()

    assert call_log["download"] == before["download"], "download re-ran"
    assert call_log["separate"] == before["separate"], "separate re-ran"
    assert call_log["analyze"]  == before["analyze"],  "analyze re-ran"


def test_per_track_failure_does_not_crash_run(mocked_pipeline, monkeypatch, call_log):
    """A failing download for one track should leave the other track unaffected."""
    pipeline = mocked_pipeline

    real_download = pipeline.download_track

    def flaky_download(song_id, title, source_url, artist=""):
        if "Midnight" in title:
            return None  # simulate failure
        return real_download(song_id, title, source_url, artist)

    monkeypatch.setattr(pipeline, "download_track", flaky_download)

    pipeline.run_ingest("https://soundcloud.com/fake/playlist")
    pipeline.run_download()
    pipeline.run_stems()
    pipeline.run_analysis()

    from database.models import get_all_songs
    by_title = {s["title"]: s for s in get_all_songs()}
    assert by_title["Sunrise"]["status"] == "analysed"
    assert by_title["Midnight"]["status"] == "error_download"
