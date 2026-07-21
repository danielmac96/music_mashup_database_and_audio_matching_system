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
