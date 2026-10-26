"""scripts/ingest_urls.py — a library from a URL list, through the one ingest path."""
from __future__ import annotations

import pytest


def test_read_urls_keeps_order_and_drops_comments(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ingest_urls", Path("scripts/ingest_urls.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

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


def test_it_goes_through_ingest_rows_not_its_own_path():
    from pathlib import Path

    src = Path("scripts/ingest_urls.py").read_text(encoding="utf-8")
    assert "ingest_rows" in src
    # A second upsert path is how the two ingesters drift apart.
    assert "upsert_song" not in src
