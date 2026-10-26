"""analysis/timeseries.py — the one frame grid and its .npz sidecars."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    """An isolated database and sidecar folder.

    `config.DB_PATH` binds at import, so setting MASHUP_DB_PATH here would do
    nothing and a call without `db_path=` would write to the user's real
    library. Every call below passes this path explicitly instead."""
    from analysis import timeseries as ts
    from database import models

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    path = tmp_path / "frames.db"
    models.get_conn(path).close()          # runs the whole ledger on it
    return path


def test_step_is_the_shared_essentia_grid():
    from analysis import timeseries as ts

    # essentia.spectral runs 2048/512 at 22.05 kHz, decimated by 2;
    # TonalExtractor runs 4096/2048 at 44.1 kHz. Both are 2048/44100.
    assert ts.STEP_SECS == pytest.approx(2048 / 44100.0, abs=1e-9)
    assert ts.STEP_SECS == pytest.approx(2 * 512 / 22050.0, abs=1e-9)


def test_sync_by_time_matches_librosa_sync():
    """The one numerical contract: structure.py's beat aggregation is unchanged."""
    librosa = pytest.importorskip("librosa")
    from analysis import timeseries as ts

    rng = np.random.default_rng(7)
    step = ts.STEP_SECS
    M = rng.random((12, 400)).astype(np.float32)
    beat_times = np.arange(0, 400 * step, 0.5)          # a 120 BPM grid

    idx = np.asarray(np.round(beat_times / step), dtype=int)
    idx = idx[(idx >= 0) & (idx < M.shape[1])]
    want = librosa.util.sync(M, idx, aggregate=np.median)
    got = ts.sync_by_time(M, step, beat_times, np.median)

    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-5)


def test_write_read_round_trip(db):
    from analysis import timeseries as ts

    rng = np.random.default_rng(3)
    frames = ts.StemFrames(
        step=ts.STEP_SECS,
        chroma=rng.random((12, 50)).astype(np.float32),
        mfcc=(rng.random((13, 50)) * 200 - 100).astype(np.float32),
        bands=(rng.random((8, 50)) * 1e6).astype(np.float32),
        curves=(rng.random((5, 50)) * 5000).astype(np.float32),
    )
    meta = ts.write(1, "vocals", "unified_4stem", frames, "abc123", db_path=db)
    assert meta["n_frames"] == 50
    assert meta["matrices"] == {"chroma": 12, "mfcc": 13, "bands": 8, "curves": 5}
    assert ts.path_for(1, "vocals", "unified_4stem").exists()

    back = ts.read(1, "vocals", "unified_4stem", db_path=db)
    assert back is not None
    assert back.step == pytest.approx(ts.STEP_SECS)
    assert back.chroma.shape == (12, 50)
    assert np.allclose(back.chroma, frames.chroma, atol=1e-3)
    assert np.allclose(back.mfcc, frames.mfcc, rtol=2e-3, atol=0.2)
    # bands are scaled by their own max before float16, so compare relatively.
    assert np.allclose(back.bands, frames.bands, rtol=5e-3, atol=1.0)
    assert np.allclose(back.curves, frames.curves, rtol=1e-6)


def test_a_matrix_left_out_reads_back_as_none_never_zeros(db):
    from analysis import timeseries as ts

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.ones((12, 10), dtype=np.float32))
    ts.write(5, "full", "unified_4stem", frames, "h", db_path=db)

    back = ts.read(5, "full", "unified_4stem", db_path=db)
    assert back.chroma is not None
    assert back.mfcc is None and back.bands is None
    assert back.curves is None and back.f0 is None
    assert back.matrices() == {"chroma": 12}


def test_a_wrong_row_count_is_refused_and_nothing_is_written(db):
    from analysis import timeseries as ts
    from database import alignment

    bad = ts.StemFrames(step=ts.STEP_SECS,
                        chroma=np.ones((11, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        ts.write(6, "full", "unified_4stem", bad, "h", db_path=db)
    assert alignment.get_frames(6, db_path=db) == []
    assert not ts.path_for(6, "full", "unified_4stem").exists()


def test_a_corrupt_sidecar_reads_as_not_measured(db):
    from analysis import timeseries as ts

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.ones((12, 10), dtype=np.float32))
    ts.write(7, "full", "unified_4stem", frames, "h", db_path=db)
    ts.path_for(7, "full", "unified_4stem").write_bytes(b"not an npz")

    assert ts.read(7, "full", "unified_4stem", db_path=db) is None


def test_have_is_false_for_another_hash_or_a_missing_file(db):
    from analysis import timeseries as ts

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(2, "full", "unified_4stem", frames, "hash-one", db_path=db)

    assert ts.have(2, "full", "unified_4stem", "hash-one", db_path=db)
    assert not ts.have(2, "full", "unified_4stem", "hash-two", db_path=db)
    assert not ts.have(2, "full", "unified_4stem", None, db_path=db)

    ts.path_for(2, "full", "unified_4stem").unlink()
    assert not ts.have(2, "full", "unified_4stem", "hash-one", db_path=db)


def test_have_is_false_for_another_format_version(db, monkeypatch):
    from analysis import timeseries as ts

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(8, "full", "unified_4stem", frames, "h", db_path=db)
    assert ts.have(8, "full", "unified_4stem", "h", db_path=db)

    monkeypatch.setattr(ts, "VERSION", ts.VERSION + 1)
    assert not ts.have(8, "full", "unified_4stem", "h", db_path=db)


def test_rewriting_a_stem_replaces_its_row(db):
    from analysis import timeseries as ts
    from database import alignment

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(9, "full", "unified_4stem", frames, "old", db_path=db)
    ts.write(9, "full", "unified_4stem", frames, "new", db_path=db)

    rows = alignment.get_frames(9, db_path=db)
    assert len(rows) == 1 and rows[0]["content_hash"] == "new"


def test_load_track_returns_only_the_stems_that_exist(db):
    from analysis import timeseries as ts

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(3, "vocals", "unified_4stem", frames, "h", db_path=db)

    track = ts.load_track(3, "unified_4stem", db_path=db)
    assert set(track.stems) == {"vocals"}
    assert track.get("vocals") is not None
    assert track.get("instrumental") is None


def test_drop_removes_files_and_rows(db):
    from analysis import timeseries as ts
    from database import alignment

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(4, "full", "unified_4stem", frames, "h", db_path=db)
    ts.write(4, "vocals", "unified_4stem", frames, "h", db_path=db)

    assert ts.drop(4, "unified_4stem", db_path=db) == 2
    assert not ts.path_for(4, "full", "unified_4stem").exists()
    assert alignment.get_frames(4, db_path=db) == []


def test_drop_leaves_the_other_pipelines_frames(db):
    from analysis import timeseries as ts

    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(10, "full", "unified_4stem", frames, "h", db_path=db)
    ts.write(10, "full", "legacy_2stem", frames, "h", db_path=db)

    assert ts.drop(10, "unified_4stem", db_path=db) == 1
    assert ts.have(10, "full", "legacy_2stem", "h", db_path=db)


# ── The table is created by migration 4, not by alignment._DDL ───────────────

def _tables(conn) -> set:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def test_a_database_built_by_get_conn_has_analysis_frames(tmp_path):
    """The live path: get_conn -> schema.apply. `alignment.migrate` runs only
    inside migration 1, so a table added to `_DDL` alone would pass every test
    that builds a fresh database and never exist on the 206-track library."""
    from database import models, schema

    conn = models.get_conn(tmp_path / "fresh.db")
    try:
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
        assert "analysis_frames" in _tables(conn)
        index_names = {r[1] for r in conn.execute(
            "PRAGMA index_list(analysis_frames)")}
        assert "idx_analysis_frames_song" in index_names
    finally:
        conn.close()


def test_migration_4_reaches_a_database_already_at_version_3(tmp_path, monkeypatch):
    """An existing library: migrations 1-3 have run, 4 has not."""
    from database import schema

    real = schema.MIGRATIONS
    assert real[3].name == "analysis_frames"
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.row_factory = sqlite3.Row
    try:
        monkeypatch.setattr(schema, "MIGRATIONS", real[:3])
        assert schema.apply(conn) == [1, 2, 3]
        monkeypatch.setattr(schema, "MIGRATIONS", real)
        assert schema.current_version(conn) == 3
        assert "analysis_frames" not in _tables(conn)

        assert schema.apply(conn) == [4]
        assert "analysis_frames" in _tables(conn)
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
    finally:
        conn.close()


def test_ensure_frames_schema_is_idempotent(tmp_path):
    """downgrade.py resets user_version to 0 and replays every migration."""
    from database import alignment, models

    conn = models.get_conn(tmp_path / "again.db")
    try:
        alignment.ensure_frames_schema(conn)
        alignment.ensure_frames_schema(conn)
        assert "analysis_frames" in _tables(conn)
    finally:
        conn.close()
