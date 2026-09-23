"""Importing a tracklist captured from the page by the browser bookmarklet.

Firecrawl's stealth proxy rendered 1001tracklists until 2026-09-19 and has been
refused by Cloudflare Turnstile since. The capture is the way in that depends on
nobody's proxy: your own browser renders the page, the bookmarklet reads the
tracklist out of it, and it emits THE SAME MARKDOWN a scrape returned — so one
parser serves the scrape, the capture and the plain paste.

What these pin:

* the per-track /track/ID/ link survives, because that is what "Scrape link"
  needs later and it is the whole reason a capture beats a plain paste;
* the capture is cached under the set's URL, so a later URL import re-parses it
  offline instead of paying Firecrawl for a page it cannot render;
* cue times reach mashup_pairs, which the scrape path never managed.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TRACK = "https://www.1001tracklists.com/track"

# One line per track, exactly what grabTracklist.js emits.
CAPTURE = "\n".join([
    rf"1. [0:00] Swedish House Mafia \- Greyhound[open track page]({TRACK}/aaa/index.html)",
    rf"w/ [0:40] Dua Lipa \- Levitating[open track page]({TRACK}/bbb/index.html)",
    rf"2. [2:15] Fred again.. \- Turn On The Lights again..[open track page]({TRACK}/ccc/index.html)",
    rf"w/ [2:50] Rihanna \- We Found Love[open track page]({TRACK}/ddd/index.html)",
    rf"3. [4:30] Skrillex \- Rumble[open track page]({TRACK}/eee/index.html)",
])

SET_URL = "https://www.1001tracklists.com/tracklist/abc123/two-friends-bbm-26.html"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mixes
    importlib.reload(mixes)

    # Point the markdown cache at tmp by PATCHING the global, not by reloading
    # the module. firecrawl_scrape binds MARKDOWN_CACHE_DIR at import, but a
    # reload swaps the module object — and then the exception classes this file
    # created are no longer the ones test_firecrawl_scrape.py imported, so its
    # pytest.raises stop matching. markdown_cache_path reads the global at call
    # time, so a patch is enough and leaves module identity alone.
    from ingest import firecrawl_scrape
    monkeypatch.setattr(firecrawl_scrape, "MARKDOWN_CACHE_DIR",
                        tmp_path / "data" / "tracklist_cache")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(mixes.router, prefix="/api/mixes")
    return {"client": TestClient(app), "mixes": mixes, "models": models,
            "fc": firecrawl_scrape}


def _import(env, markdown=CAPTURE, url=SET_URL):
    res = env["client"].post("/api/mixes/import-markdown",
                             json={"markdown": markdown, "url": url})
    assert res.status_code == 200, res.text
    return res.json()


# ── the rows ─────────────────────────────────────────────────────────────────

def test_beds_and_overlays_come_through(env):
    mix = _import(env)
    assert mix["track_count"] == 5
    assert mix["import_method"] == "bookmarklet"

    detail = env["client"].get(f"/api/mixes/{mix['id']}").json()
    roles = [t["role"] for t in detail["tracks"]]
    assert roles == ["instrumental", "vocal", "instrumental", "vocal", "unassigned"]


def test_every_row_keeps_its_track_page_link(env):
    """The point of capturing rather than pasting: 'Scrape link' reads these."""
    mix = _import(env)
    detail = env["client"].get(f"/api/mixes/{mix['id']}").json()
    urls = [t["tl_track_url"] for t in detail["tracks"]]
    assert all(u and "/track/" in u for u in urls), urls
    assert urls[0].endswith("/track/aaa/index.html")


def test_the_documented_overlays_become_pairs_with_their_cue(env):
    """cue_secs is where in the set the overlay lands. The scrape path threw it
    away by rebuilding the line without it; a capture has it, so it is kept."""
    mix = _import(env)
    pairs = env["client"].get(f"/api/mixes/{mix['id']}").json()["pairs"]
    assert len(pairs) == 2
    assert sorted(p["cue_secs"] for p in pairs) == [40.0, 170.0]


def test_a_relative_track_href_is_made_absolute(env):
    md = r"1. A \- B[open track page](/track/rel/index.html)"
    mix = _import(env, markdown=md, url="")
    detail = env["client"].get(f"/api/mixes/{mix['id']}").json()
    assert detail["tracks"][0]["tl_track_url"].startswith("https://www.1001tracklists.com/")


# ── the cache, which is what makes a later URL import free ───────────────────

def test_the_capture_is_cached_under_the_set_url(env):
    _import(env)
    assert env["fc"].markdown_cache_path(SET_URL).exists()


def test_a_later_url_import_reparses_the_capture_without_the_network(env):
    """scrape_tracklist checks the cache before any request, so the URL button
    starts working offline for a page you have captured once."""
    _import(env)

    def explode(*a, **k):
        raise AssertionError("a cached capture must not hit Firecrawl")

    rows = env["fc"].scrape_tracklist(SET_URL, api_key="x", _post=explode)
    assert len(rows) == 5
    assert rows[1]["is_overlay"] is True
    assert rows[1]["cue"] == "0:40"


# ── refusals ─────────────────────────────────────────────────────────────────

def test_text_with_no_track_links_is_a_422(env):
    """Plain pasted text belongs in /import-paste. Accepting it here would drop
    every link and look like a successful capture."""
    res = env["client"].post("/api/mixes/import-markdown",
                             json={"markdown": "1. Artist - Title\nw/ Other - Thing",
                                   "url": ""})
    assert res.status_code == 422
    assert "track" in res.json()["detail"].lower()


def test_a_track_link_in_the_url_field_is_refused(env):
    res = env["client"].post("/api/mixes/import-markdown",
                             json={"markdown": CAPTURE,
                                   "url": "https://soundcloud.com/a/b"})
    assert res.status_code == 400
    assert "tracklist page" in res.json()["detail"]


# ── re-capture ───────────────────────────────────────────────────────────────

def test_re_capturing_the_same_url_replaces_rather_than_duplicates(env):
    one = _import(env)
    two = _import(env)
    assert one["id"] == two["id"]
    assert len(env["client"].get("/api/mixes").json()["mixes"]) == 1


def test_a_capture_carries_over_a_resolved_link(env):
    """Re-capturing after fixing a link must not cost you the fix — the same
    _persist_mix contract the paste and scrape paths rely on."""
    mix = _import(env)
    detail = env["client"].get(f"/api/mixes/{mix['id']}").json()
    tid = detail["tracks"][0]["id"]
    env["client"].post(f"/api/mixes/tracks/{tid}/resolve",
                       json={"url": "https://soundcloud.com/x/greyhound"})

    _import(env)
    after = env["client"].get(f"/api/mixes/{mix['id']}").json()
    assert after["tracks"][0]["link_url"] == "https://soundcloud.com/x/greyhound"
