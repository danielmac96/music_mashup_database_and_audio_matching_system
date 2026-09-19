import importlib

from api.routes import mixes

importlib.reload(mixes)


def test_bed_and_overlay_shape():
    scraped = [
        {"position": "01", "artist": "Dr. Dre", "title": "Still D.R.E.",
         "is_overlay": False, "tl_track_url": "https://x/track/1/index.html"},
        {"position": "w/", "artist": "Eminem", "title": "Without Me",
         "is_overlay": True, "tl_track_url": ""},
    ]
    rows = mixes._scraped_rows_to_persist_rows(scraped)
    assert rows[0]["is_overlay"] is False
    assert rows[0]["entry_index"] == 1
    assert rows[0]["artist"] == "Dr. Dre" and rows[0]["title"] == "Still D.R.E."
    assert rows[0]["tl_track_url"].endswith("/index.html")
    assert rows[1]["is_overlay"] is True
    assert rows[1]["entry_index"] is None


def test_remixer_derived_via_parse_line():
    scraped = [{"position": "02", "artist": "Martin Garrix",
                "title": "Hurricane (N3RI Remix)", "is_overlay": False, "tl_track_url": ""}]
    rows = mixes._scraped_rows_to_persist_rows(scraped)
    assert rows[0]["remixer"] and "N3RI" in rows[0]["remixer"]


def test_an_unparseable_row_is_dropped_not_persisted_as_an_empty_dict():
    # parse_line returns None for a line it can't read (here a title that is one
    # of its skip prefixes). That used to become {}, and _persist_mix then did
    # r["artist"] — a KeyError and a 500 on an import Firecrawl had already been
    # paid for.
    scraped = [
        {"artist": "", "title": "www.example.com", "is_overlay": False,
         "tl_track_url": ""},
        {"artist": "Kanye West", "title": "Runaway", "is_overlay": False,
         "tl_track_url": ""},
    ]
    rows = mixes._scraped_rows_to_persist_rows(scraped)
    assert len(rows) == 1
    assert rows[0]["title"] == "Runaway"
    # The dropped bed does not leave a hole in the numbering.
    assert rows[0]["entry_index"] == 1
    assert all("artist" in r and "title" in r for r in rows)


def test_a_mix_of_unparseable_rows_still_persists(tmp_path, monkeypatch):
    # The end of that path: whatever survives must reach _persist_mix without
    # raising. Exercised against a real temp DB.
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    importlib.reload(mixes)

    scraped = [
        {"artist": "", "title": "Follow us", "is_overlay": False, "tl_track_url": ""},
        {"artist": "Two Friends", "title": "Big Bootie", "is_overlay": False,
         "tl_track_url": ""},
        {"artist": "", "title": "w", "is_overlay": True, "tl_track_url": ""},
    ]
    rows = mixes._scraped_rows_to_persist_rows(scraped)
    detail = mixes._persist_mix("M", "https://src/bb", rows, method="scrape")
    assert detail["track_count"] == len(rows) >= 1
