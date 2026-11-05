"""scripts/ingest_urls.py — a library from a URL list, through the one ingest path."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ingest_urls.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("ingest_urls", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_read_urls_keeps_order_and_drops_comments(tmp_path):
    mod = _load_script()

    listing = tmp_path / "urls.txt"
    listing.write_text(
        "# a comment\n"
        "https://soundcloud.com/a/one\n"
        "\n"
        "  https://soundcloud.com/a/two  \n"
        "# another\n"
        "https://soundcloud.com/a/three\n", encoding="utf-8")

    assert mod.read_urls(str(listing)) == [
        "https://soundcloud.com/a/one",
        "https://soundcloud.com/a/two",
        "https://soundcloud.com/a/three",
    ]


@pytest.mark.parametrize("text, expected", [
    # A BOM'd URL on the first line must come back clean.
    ("﻿https://soundcloud.com/a/one\r\n# c\r\nhttps://soundcloud.com/a/two\r\n",
     ["https://soundcloud.com/a/one", "https://soundcloud.com/a/two"]),
    # A BOM'd '#' comment on the first line must still be a comment, not a URL.
    ("﻿# c\r\nhttps://soundcloud.com/a/one\r\n",
     ["https://soundcloud.com/a/one"]),
])
def test_read_urls_survives_a_utf8_bom_and_crlf(tmp_path, text, expected):
    # Windows PowerShell 5.1 writes a BOM and CRLF line endings by default.
    mod = _load_script()

    listing = tmp_path / "urls.txt"
    listing.write_bytes(text.encode("utf-8"))

    assert mod.read_urls(str(listing)) == expected


def test_read_urls_strips_a_bom_from_stdin(monkeypatch):
    # The docstring's `- --process < urls.txt` form feeds a BOM'd file on stdin.
    mod = _load_script()
    monkeypatch.setattr(sys, "stdin",
                        io.StringIO("﻿# c\r\nhttps://soundcloud.com/a/one\r\n"))

    assert mod.read_urls("-") == ["https://soundcloud.com/a/one"]


def test_main_runs_when_stdout_has_no_reconfigure(tmp_path):
    # --help exits in argparse, before anything touches config or the database.
    # Under a plain StringIO stdout the reconfigure call must not raise.
    mod = _load_script()

    with contextlib.redirect_stdout(io.StringIO()) as out:
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])

    assert exc.value.code == 0
    assert "--process" in out.getvalue()


def test_no_process_says_nothing_is_working_on_the_tracks(tmp_path, monkeypatch):
    # The message must not promise that a running server will pick the tracks up:
    # this process starts no workers, and only a server restart re-queues them.
    # ingest_rows is stubbed, so the real ingest path (network, database) never runs.
    import api.routes.playlists as playlists

    def fake_ingest_rows(rows, group_name=None):
        return {"count": 1, "skipped_count": 0, "unresolvable_count": 0,
                "partial_count": 0, "enriched_count": 0, "unresolvable": [],
                "skipped": [], "inserted_ids": [123]}

    monkeypatch.setattr(playlists, "ingest_rows", fake_ingest_rows)
    mod = _load_script()
    listing = tmp_path / "urls.txt"
    listing.write_text("https://soundcloud.com/a/one\n", encoding="utf-8")

    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = mod.main([str(listing)])

    text = out.getvalue()
    assert code == 0
    assert "Saved 1 track(s) at status 'queued'" in text
    assert "nothing is processing them" in text
    assert "--process" in text
    assert "restart the server" in text
    assert "server's workers" not in text
    assert "Queue screen" not in text


def test_it_goes_through_ingest_rows_not_its_own_path():
    # Drift this guards: a second write path into `songs`. Every song row must
    # come through api.routes.playlists.ingest_rows, so dedup, the partial-metadata
    # flag and crate relinking cannot drift from the app. A literal upsert_song
    # import, a raw INSERT, or a sqlite3 connection would each be a second path.
    # Checked on the AST, not by running main(): the ingest path reaches SoundCloud
    # and writes the configured database.
    import re

    write_sql = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE|UPSERT)\b")
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))

    # 1. No write statement in any string literal (the docstring included).
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not write_sql.search(node.value), (
                f"line {node.lineno}: write SQL in a string literal")

    # 2. Only get_conn (read-back) comes from database.models, and nothing from
    #    database or sqlite3 is imported any other way.
    storage_imports = set()
    from_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            from_imports |= {(node.module, a.name) for a in node.names}
            if node.module.split(".")[0] in ("database", "sqlite3"):
                storage_imports |= {(node.module, a.name) for a in node.names}
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in ("database", "sqlite3"):
                    storage_imports.add(("import", a.name))
    assert storage_imports == {("database.models", "get_conn")}, storage_imports

    # 3. The positive direction: ingest_rows is imported from the one path and
    #    actually called.
    assert ("api.routes.playlists", "ingest_rows") in from_imports
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "ingest_rows"]
    assert calls, "the script must call ingest_rows, the one ingest path"
