"""ingest_rows: ask the library before asking the network, and dedup on identity."""
from __future__ import annotations

import importlib
import sqlite3
from types import SimpleNamespace

import pytest


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A temp library, the network stubbed, and a count of enrichment calls.

    `config.DB_PATH` binds at import, so the DB is pointed at the temp folder
    and the modules reloaded (the order the other ingest tests use). Without
    that, `ingest_rows` would write to whatever database `config` resolved to
    at collection time: the live one.
    """
    db = tmp_path / "i.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import playlists
    importlib.reload(playlists)

    calls: list[str] = []

    def fake_enrich(url):
        calls.append(url)
        return {"source_url": url, "title": f"T {url[-1]}", "artist": "A",
                "duration_secs": 200.0, "track_id": url[-1], "hydrated": True}

    monkeypatch.setattr(playlists, "enrich_track", fake_enrich)
    monkeypatch.setattr(playlists.queue_runner, "enqueue_song",
                        lambda sid, **kw: f"job-{sid}")
    monkeypatch.setattr(playlists.preview_hydrator, "cache_get", lambda url: None)
    monkeypatch.setattr(playlists.preview_hydrator, "cache_put", lambda url, row: None)
    return SimpleNamespace(db=db, models=models, playlists=playlists, enriched=calls)


def test_a_row_already_in_the_library_is_never_enriched(lib):
    lib.models.upsert_song(title="Owned", source_url="https://soundcloud.com/a/1",
                           track_id="1", db_path=lib.db)

    out = lib.playlists.ingest_rows([{"source_url": "https://soundcloud.com/a/1"},
                                     {"source_url": "https://soundcloud.com/a/2"}])

    assert lib.enriched == ["https://soundcloud.com/a/2"], lib.enriched
    assert out["skipped_count"] == 1
    assert out["count"] == 1
    assert out["enriched_count"] == 1


def test_the_same_record_under_two_links_imports_once(lib):
    """A share link and the canonical link are one record, one pipeline run."""
    sid = lib.models.upsert_song(title="Owned", source_url="https://soundcloud.com/a/7",
                                 track_id="7", db_path=lib.db)

    out = lib.playlists.ingest_rows(
        [{"source_url": "https://soundcloud.com/a/7?secret_token=xyz",
          "track_id": "7", "title": "Owned", "hydrated": True}])

    assert out["count"] == 0
    assert out["skipped_count"] == 1
    assert out["ordered_song_ids"] == [sid]


def test_ordered_song_ids_stays_positionally_aligned(lib):
    owned = lib.models.upsert_song(title="Owned", source_url="https://soundcloud.com/a/1",
                                   track_id="1", db_path=lib.db)

    out = lib.playlists.ingest_rows([
        {"source_url": "https://soundcloud.com/a/1"},   # owned
        {"source_url": ""},                              # unresolvable
        {"source_url": "https://soundcloud.com/a/3"},   # new
    ])

    ids = out["ordered_song_ids"]
    assert len(ids) == 3
    assert ids[0] == owned
    assert ids[1] is None
    assert ids[2] in out["inserted_ids"]


def test_the_batch_runs_on_one_connection(lib, monkeypatch):
    opens = {"n": 0}
    real_get_conn = lib.models.get_conn

    def counting(*a, **k):
        opens["n"] += 1
        return real_get_conn(*a, **k)

    monkeypatch.setattr(lib.models, "get_conn", counting)
    lib.playlists.ingest_rows([{"source_url": f"https://soundcloud.com/a/{i}"}
                               for i in range(20)])
    # Exactly one: the batch's. No group was asked for, so no crate step opens
    # another, and nothing per row does (the pre-D5 code opened 41 for 20 rows).
    # A bound looser than this would let a per-batch regression through.
    assert opens["n"] == 1, opens["n"]


def test_a_record_repeated_inside_one_import_is_saved_once(lib):
    """The library check runs once, up front, so it cannot see a row the same
    import is about to write. Without a second look after enrichment the repeat
    upserts onto the first row, resets it to `queued` and is queued twice."""
    out = lib.playlists.ingest_rows([
        {"source_url": "https://soundcloud.com/a/5"},
        {"source_url": "https://soundcloud.com/a/5"},
    ])

    assert out["count"] == 1
    assert out["skipped_count"] == 1
    first, second = out["ordered_song_ids"]
    assert first == second == out["inserted_ids"][0]


def test_one_unsaveable_row_does_not_lose_the_batch(lib, monkeypatch):
    import sqlite3
    real = lib.playlists.upsert_song

    def flaky(**kw):
        if kw["source_url"].endswith("/2"):
            raise sqlite3.IntegrityError("boom")
        return real(**kw)

    monkeypatch.setattr(lib.playlists, "upsert_song", flaky)
    out = lib.playlists.ingest_rows([{"source_url": f"https://soundcloud.com/a/{i}"}
                                     for i in (1, 2, 3)])

    assert out["count"] == 2
    assert out["unresolvable_count"] == 1
    ids = out["ordered_song_ids"]
    assert ids[0] and ids[1] is None and ids[2]
    assert lib.models.get_song_by_url("https://soundcloud.com/a/3",
                                      db_path=lib.db) is not None


def test_the_same_record_under_two_links_is_caught_after_enrichment(lib):
    """The mix ingester sends title/artist and no `track_id`, so the up-front
    check has only the URL to go on and two links to one record both pass it.
    Enrichment is what reveals the shared track id; the second look is the only
    thing between a 206-track mix and two downloads of the same record."""
    out = lib.playlists.ingest_rows([{"source_url": "https://soundcloud.com/a/9"},
                                     {"source_url": "https://soundcloud.com/b/9"}])

    assert out["count"] == 1
    assert out["skipped_count"] == 1
    first, second = out["ordered_song_ids"]
    assert first == second == out["inserted_ids"][0]


def test_a_lost_transaction_fails_the_import_instead_of_enqueueing(lib, monkeypatch):
    """SQLITE_FULL and SQLITE_IOERR roll the whole transaction back by
    themselves. Carrying on would return ids of rows that no longer exist and
    start pipeline jobs against them."""
    import sqlite3
    real = lib.playlists.upsert_song
    enqueued: list[int] = []
    monkeypatch.setattr(lib.playlists.queue_runner, "enqueue_song",
                        lambda sid, **kw: enqueued.append(sid))

    def disk_full(**kw):
        if kw["source_url"].endswith("/3"):
            kw["conn"].rollback()   # what SQLite does to the transaction itself
            raise sqlite3.OperationalError("database or disk is full")
        return real(**kw)

    monkeypatch.setattr(lib.playlists, "upsert_song", disk_full)

    with pytest.raises(sqlite3.OperationalError):
        lib.playlists.ingest_rows([{"source_url": f"https://soundcloud.com/a/{i}"}
                                   for i in (1, 2, 3, 4)])

    assert enqueued == []
    for i in (1, 2, 3, 4):
        assert lib.models.get_song_by_url(f"https://soundcloud.com/a/{i}",
                                          db_path=lib.db) is None


# ---------------------------------------------------------------------------
# The rule `models.connection` states in prose, as tests (readme §7): nothing
# inside the batch's `with` block may open a connection of its own. A caller
# sitting on an uncommitted write while a helper opens another connection is
# what made a 206-track mix save one track and answer HTTP 500 forever.
# `get_or_create_crate`, `add_songs_to_crate` and `library_groups` (reached by
# `_save_as_group`) all do exactly that, and none takes a `conn`.
# ---------------------------------------------------------------------------

# Functions of the playlists module the block may call: neither touches the DB.
_PURE_LOCAL_HELPERS = {"_existing_ids", "_resolve_counting"}


def _batch_block_violations(playlists, source: str) -> tuple[list[str], set[str]]:
    """(violations, names called) for the `with models_connection()` block of
    the function whose source this is."""
    import ast
    import inspect

    tree = ast.parse(source)
    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.With)
              and any(isinstance(i.context_expr, ast.Call)
                      and getattr(i.context_expr.func, "id", None) == "models_connection"
                      for i in n.items)]
    assert len(blocks) == 1, "expected exactly one `with models_connection()` block"

    violations: list[str] = []
    called: set[str] = set()
    # The body only: the `with` header's own models_connection() is the opener.
    for node in (n for stmt in blocks[0].body for n in ast.walk(stmt)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            root = func
            while isinstance(root, ast.Attribute):
                root = root.value
            # The pipeline queue must only be fed after the commit.
            if getattr(root, "id", None) == "queue_runner":
                violations.append(f"line {node.lineno}: queue_runner call inside "
                                  "the transaction")
            continue
        name = getattr(func, "id", None)
        target = getattr(playlists, name, None) if name else None
        if not inspect.isfunction(target):
            continue   # builtins, classes, the pool: not ours to police
        called.add(name)
        if target.__module__ == "database.models":
            if "conn" not in inspect.signature(target).parameters:
                violations.append(f"line {node.lineno}: {name}() takes no `conn` "
                                  "and opens its own connection")
            elif not any(k.arg == "conn" for k in node.keywords):
                violations.append(f"line {node.lineno}: {name}() is not passed "
                                  "`conn=`")
        elif target.__module__ == playlists.__name__ and name not in _PURE_LOCAL_HELPERS:
            violations.append(f"line {node.lineno}: {name}() is a playlists helper "
                              "that is not known to leave the DB alone")
    return violations, called


def test_nothing_in_the_batch_transaction_opens_its_own_connection(lib):
    import inspect
    violations, called = _batch_block_violations(
        lib.playlists, inspect.getsource(lib.playlists.ingest_rows))

    assert violations == []
    # Not vacuous: the block really does the writing it is meant to police.
    assert {"upsert_song", "relink_crate_songs", "_existing_ids"} <= called, called


def test_the_block_check_catches_the_group_step_moved_inside(lib):
    """The detector itself: `_save_as_group` and the crate helpers inside the
    block must be reported, or the test above guards nothing."""
    import inspect
    src = inspect.getsource(lib.playlists.ingest_rows).replace(
        "        if inserted_ids:\n            relink_crate_songs(conn=conn)",
        "        if inserted_ids:\n            relink_crate_songs(conn=conn)\n"
        "        _save_as_group(group_name, ordered_song_ids)\n"
        "        get_or_create_crate('x')\n"
        "        queue_runner.enqueue_song(1)")
    assert "get_or_create_crate('x')" in src

    violations, _ = _batch_block_violations(lib.playlists, src)

    joined = "\n".join(violations)
    assert "_save_as_group" in joined
    assert "get_or_create_crate() takes no `conn`" in joined
    assert "queue_runner" in joined


def test_no_second_connection_opens_while_the_batch_has_written(lib, monkeypatch):
    """The same rule observed rather than read: a connection requested while the
    batch's own connection is mid-transaction is the deadlock. The group step
    runs, so a `_save_as_group` moved inside the block opens one here."""
    real_get_conn = lib.models.get_conn
    batch: dict = {}
    offences: list[str] = []

    def watching(*a, **k):
        try:
            mid_write = batch["conn"].in_transaction if "conn" in batch else False
        except sqlite3.ProgrammingError:   # the batch connection is closed
            mid_write = False
        if mid_write:
            offences.append("opened while the batch transaction held a write")
        conn = real_get_conn(*a, **k)
        batch.setdefault("conn", conn)   # the first open is the batch's
        return conn

    monkeypatch.setattr(lib.models, "get_conn", watching)
    out = lib.playlists.ingest_rows(
        [{"source_url": f"https://soundcloud.com/a/{i}"} for i in (1, 2, 3)],
        group_name="Probe")

    assert offences == []
    assert out["count"] == 3
    assert out["group"] is not None   # the group step really ran
