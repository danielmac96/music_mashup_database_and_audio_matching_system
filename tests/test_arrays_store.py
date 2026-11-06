"""database/arrays.py — float arrays as BLOBs, out of the hot rows."""
from __future__ import annotations

import sqlite3

import pytest


def test_pack_unpack_round_trip():
    from database import arrays

    vals = [0.0, 1.5, -2.25, 1e6]
    blob = arrays.pack(vals)
    assert isinstance(blob, bytes)
    assert len(blob) == 4 * len(vals)
    assert arrays.unpack(blob) == pytest.approx(vals)

    # Unmeasured is absent, never a zero-length blob.
    assert arrays.pack(None) is None
    assert arrays.pack([]) is None
    assert arrays.unpack(None) == []


def test_put_get_and_delete(tmp_path):
    from database import arrays
    from database.models import get_conn

    conn = get_conn(tmp_path / "a.db")
    try:
        arrays.put(conn, 7, "full", "beat_times", [0.0, 0.5, 1.0])
        assert arrays.get(conn, 7, "full", "beat_times") == pytest.approx([0.0, 0.5, 1.0])
        assert arrays.get(conn, 7, "full", "waveform_rms") == []

        # Writing None removes it rather than storing an empty array.
        arrays.put(conn, 7, "full", "beat_times", None)
        assert arrays.get(conn, 7, "full", "beat_times") == []

        arrays.put(conn, 7, "vocals", "waveform_rms", [0.1, 0.2])
        assert arrays.delete_for(conn, 7) == 1
    finally:
        conn.close()


def test_get_many_and_song_ids_with(tmp_path):
    from database import arrays
    from database.models import get_conn

    conn = get_conn(tmp_path / "many.db")
    try:
        arrays.put(conn, 1, "full", "beat_times", [1.0, 2.0])
        arrays.put(conn, 2, "full", "beat_times", [3.0])
        arrays.put(conn, 2, "full", "waveform_rms", [0.5])
        arrays.put(conn, 2, "vocals", "beat_times", [9.0])

        got = arrays.get_many(conn, [1, 2, 3], "full", ["beat_times"])
        assert set(got) == {(1, "beat_times"), (2, "beat_times")}
        assert got[(2, "beat_times")] == pytest.approx([3.0])
        assert arrays.get_many(conn, [], "full", ["beat_times"]) == {}

        # Presence without reading a single blob: what the track list's
        # "waveform measured" flag needs for every row at once.
        assert arrays.song_ids_with(conn, "full", "waveform_rms") == {2}
        assert arrays.song_ids_with(conn, "full", "beat_times") == {1, 2}
    finally:
        conn.close()


def test_features_round_trip_through_the_store(tmp_path):
    from database.models import (get_features_for_song, upsert_features,
                                 upsert_song)

    db = tmp_path / "f.db"
    sid = upsert_song(title="T", artist="A", source_url="u://1", db_path=db)
    beats = [0.0, 0.47, 0.94]
    upsert_features(sid, "full", {"bpm": 128.0, "beat_times": beats,
                                  "waveform_rms": [0.1, 0.2, 0.3],
                                  "mfcc": [0.0] * 13}, db_path=db)
    got = get_features_for_song(sid, "full", db_path=db)
    assert got["bpm"] == 128.0
    assert got["beat_times"] == pytest.approx(beats)
    assert got["waveform_rms"] == pytest.approx([0.1, 0.2, 0.3])

    # A re-analysis that measures no grid clears the old one, as the NULL column did.
    upsert_features(sid, "full", {"bpm": 128.0, "mfcc": [0.0] * 13}, db_path=db)
    got = get_features_for_song(sid, "full", db_path=db)
    assert "beat_times" not in got and "waveform_rms" not in got


def test_deleting_a_song_takes_its_arrays(tmp_path):
    from database.models import (delete_song, get_conn, upsert_features,
                                 upsert_song)

    db = tmp_path / "d.db"
    sid = upsert_song(title="T", artist="A", source_url="u://1", db_path=db)
    upsert_features(sid, "full", {"bpm": 120.0, "beat_times": [0.0, 0.5]}, db_path=db)
    delete_song(sid, db_path=db)
    conn = get_conn(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM feature_arrays").fetchone()[0] == 0
    finally:
        conn.close()


def test_the_scoring_projection_leaves_the_arrays_behind(tmp_path):
    from database.models import (FEATURES_SCORING_COLUMNS, get_all_features,
                                 upsert_features, upsert_song)

    db = tmp_path / "p.db"
    sid = upsert_song(title="T", artist="A", source_url="u://1", db_path=db)
    upsert_features(sid, "full", {"bpm": 128.0, "beat_times": [0.0, 0.5],
                                  "mfcc": [0.0] * 13}, db_path=db)

    rows = get_all_features(stem_type="full", columns=FEATURES_SCORING_COLUMNS,
                            db_path=db)
    assert rows and rows[0]["bpm"] == 128.0
    assert "beat_times" not in rows[0]
    assert "waveform_rms" not in rows[0]
    # The joined columns the scorer and dedup need are still there.
    for key in ("title", "artist", "variant_cluster", "mfcc"):
        assert key in rows[0], key

    # The default projection is every column of `features`, minus the arrays.
    wide = get_all_features(stem_type="full", db_path=db)
    assert "spectral_centroid" in wide[0] and "beat_times" not in wide[0]

    # `song_ids` narrows the rows.
    assert get_all_features(stem_type="full", song_ids=[sid + 99], db_path=db) == []
    assert len(get_all_features(stem_type="full", song_ids=[sid], db_path=db)) == 1


def test_a_keyboard_interrupt_rolls_back_and_is_not_wrapped(tmp_path, monkeypatch):
    """Only an `Exception` is renamed "schema migration N failed"; Ctrl-C during
    a long migration still rolls back, and still reaches the caller as itself."""
    from database import schema

    def interrupted(conn):
        conn.execute("CREATE TABLE half_done (x)")
        raise KeyboardInterrupt

    monkeypatch.setattr(schema, "MIGRATIONS",
                        (schema.Migration(1, "interrupted", interrupted),))
    conn = sqlite3.connect(tmp_path / "k.db")
    try:
        with pytest.raises(KeyboardInterrupt):
            schema.apply(conn)
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='half_done'"
                            ).fetchone() is None
        assert schema.current_version(conn) == 0
        assert not conn.in_transaction
    finally:
        conn.close()
