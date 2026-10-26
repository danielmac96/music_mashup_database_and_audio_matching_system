"""models.connection(): one connection for a batch, without the deadlock."""
from __future__ import annotations

import sqlite3

import pytest


def test_connection_commits_on_clean_exit(tmp_path):
    from database.models import connection, get_conn

    db = tmp_path / "s.db"
    with connection(db) as conn:
        conn.execute("INSERT INTO songs (title, source_url) VALUES ('T','u://1')")
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 1
    finally:
        check.close()


def test_connection_rolls_back_on_an_exception(tmp_path):
    from database.models import connection, get_conn

    db = tmp_path / "r.db"
    with pytest.raises(RuntimeError):
        with connection(db) as conn:
            conn.execute("INSERT INTO songs (title, source_url) VALUES ('T','u://1')")
            raise RuntimeError("boom")
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 0
    finally:
        check.close()


def test_a_helper_given_a_connection_does_not_commit(tmp_path):
    """The convention: the caller owns the transaction. A helper that commits
    mid-batch makes a partial import durable, which is the bug this replaces."""
    from database.models import connection, upsert_song

    db = tmp_path / "t.db"
    with connection(db) as conn:
        upsert_song(title="T", artist="A", source_url="u://1", conn=conn)
        assert conn.in_transaction, "upsert_song committed the caller's transaction"
        upsert_song(title="U", artist="B", source_url="u://2", conn=conn)
        assert conn.in_transaction


def test_helpers_still_work_without_a_connection(tmp_path):
    """Every existing caller passes no connection and must be unaffected."""
    from database.models import get_song_by_url, songs_by_identity, upsert_song

    db = tmp_path / "u.db"
    sid = upsert_song(title="T", artist="A", source_url="u://1", track_id="99",
                      db_path=db)
    assert get_song_by_url("u://1", db_path=db)["id"] == sid
    found = songs_by_identity(source_urls=["u://1"], db_path=db)
    assert found and sid in {v["id"] for v in found["by_url"].values()}


def test_one_connection_serves_a_whole_batch(tmp_path):
    from database.models import connection, get_song_by_url, upsert_song

    db = tmp_path / "b.db"
    with connection(db) as conn:
        for i in range(25):
            url = f"u://{i}"
            assert get_song_by_url(url, conn=conn) is None
            upsert_song(title=f"T{i}", source_url=url, conn=conn)
        assert get_song_by_url("u://7", conn=conn) is not None
    from database.models import get_conn
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 25
    finally:
        check.close()


def test_upsert_returns_the_id_before_any_commit(tmp_path):
    """The id of a freshly inserted row must be readable inside the caller's
    still-open transaction, and an update must return the existing row's id."""
    from database.models import connection, get_conn, upsert_song

    db = tmp_path / "i.db"
    with connection(db) as conn:
        a = upsert_song(title="T", source_url="u://1", conn=conn)
        b = upsert_song(title="U", source_url="u://2", conn=conn)
        again = upsert_song(title="T2", source_url="u://1", conn=conn)
        assert a != b and again == a
        assert conn.execute("SELECT title FROM songs WHERE id=?",
                            (a,)).fetchone()[0] == "T2"
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 2
    finally:
        check.close()


def test_an_exception_after_helpers_leaves_nothing_behind(tmp_path):
    """The partial-import failure: rows written through the scoped connection
    vanish together when the batch raises."""
    from database.models import connection, get_conn, upsert_song

    db = tmp_path / "p.db"
    with pytest.raises(RuntimeError):
        with connection(db) as conn:
            for i in range(5):
                upsert_song(title=f"T{i}", source_url=f"u://{i}", conn=conn)
            raise RuntimeError("boom")
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 0
    finally:
        check.close()


def test_read_helpers_and_relink_given_a_connection_do_not_commit(tmp_path):
    from database.models import (connection, get_conn, relink_crate_songs,
                                 songs_by_identity, upsert_song)

    db = tmp_path / "c.db"
    seed = get_conn(db)
    try:
        seed.execute("INSERT INTO crates (name) VALUES ('c')")
        cid = seed.execute("SELECT id FROM crates").fetchone()[0]
        seed.execute("INSERT INTO crate_items (crate_id, source_url, title, artist, position) "
                     "VALUES (?, 'u://1', 'T', 'A', 0)", (cid,))
        seed.commit()
    finally:
        seed.close()

    with connection(db) as conn:
        sid = upsert_song(title="T", source_url="u://1", track_id="7", conn=conn)
        found = songs_by_identity(source_urls=["u://1"], track_ids=["7"], conn=conn)
        assert found["by_url"]["u://1"]["id"] == sid
        assert found["by_track_id"]["7"]["id"] == sid
        assert relink_crate_songs(conn=conn) == 1
        assert conn.in_transaction, "a helper committed the caller's transaction"
        assert conn.execute("SELECT song_id FROM crate_items").fetchone()[0] == sid


def test_helpers_without_a_connection_still_commit(tmp_path):
    """conn=None is unchanged: open, write, commit, close — durable on return."""
    from database.models import get_conn, relink_crate_songs, upsert_song

    db = tmp_path / "d.db"
    sid = upsert_song(title="T", source_url="u://1", db_path=db)
    seed = get_conn(db)
    try:
        seed.execute("INSERT INTO crates (name) VALUES ('c')")
        cid = seed.execute("SELECT id FROM crates").fetchone()[0]
        seed.execute("INSERT INTO crate_items (crate_id, source_url, title, artist, position) "
                     "VALUES (?, 'u://1', 'T', 'A', 0)", (cid,))
        seed.commit()
    finally:
        seed.close()
    assert relink_crate_songs(db_path=db) == 1
    check = get_conn(db)
    try:
        assert check.execute("SELECT song_id FROM crate_items").fetchone()[0] == sid
    finally:
        check.close()
