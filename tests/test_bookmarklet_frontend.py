"""The browser-capture bookmarklet, pinned from Python like every other frontend
contract in this repo (there is no JS test runner).

The bookmarklet runs on a page I cannot see and cannot test against, so what is
checkable here is the part that WILL rot: the selector strategy, the shape of the
markdown it emits, and the routing that decides which endpoint a paste goes to.
"""
import json
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"

BM = (SRC / "bookmarklet" / "grabTracklist.js").read_text(encoding="utf-8")
GEN = (SRC / "bookmarklet" / "generated.js").read_text(encoding="utf-8")
IMPORTER = (SRC / "components" / "MixImporter.jsx").read_text(encoding="utf-8")
API = (SRC / "api.js").read_text(encoding="utf-8")


def _code(src: str) -> str:
    """The source minus comments — a rule named in prose is not a rule."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.split("\n") if not l.strip().startswith("//"))


# ── the selector strategy ────────────────────────────────────────────────────

def test_the_capture_keys_off_the_track_link_not_class_names():
    """1001tracklists' class names are obfuscated and change. The one durable
    fact — every track row links to /track/<id>/ — is what every page Firecrawl
    ever returned proves, so it is the only thing worth selecting on."""
    code = _code(BM)
    assert "a[href*='/track/']" in code
    assert not re.search(r"""querySelector\w*\(\s*['"][.#]""", code), \
        "no class or id selectors — they will rot"
    assert "className" not in code


def test_the_row_walk_stops_before_the_next_track():
    """Without the guard the walk keeps climbing until one 'row' holds the whole
    tracklist, and every track gets the same text."""
    code = _code(BM)
    walk = code[code.index("function rowFor"):]
    walk = walk[:walk.index("\n  }")]
    assert "length > 1" in walk and "break" in walk


# ── the markdown it emits ────────────────────────────────────────────────────

def test_it_emits_the_markdown_the_server_already_parses():
    """The capture is deliberately NOT a new format: it is what Firecrawl
    returned, so ingest/firecrawl_scrape.parse_markdown_tracklist serves the
    scrape, the capture and (via the line parser) the paste."""
    code = _code(BM)
    assert "[open track page](" in code
    assert '"w/ "' in code          # overlays carry the marker inline
    assert '". "' in code           # beds are numbered


def test_the_emitted_lines_round_trip_through_the_real_parser():
    """The one end-to-end check available without a browser: build a line the
    way the bookmarklet builds it and run the server's parser over it."""
    from ingest.firecrawl_scrape import parse_markdown_tracklist
    url = "https://www.1001tracklists.com/track/aaa/index.html"
    md = (rf"1. [0:00] Artist One \- Title One[open track page]({url})"
          "\n"
          rf"w/ [0:40] Artist Two \- Title Two[open track page]({url})")
    rows = parse_markdown_tracklist(md)
    assert [r["artist"] for r in rows] == ["Artist One", "Artist Two"]
    assert [r["is_overlay"] for r in rows] == [False, True]
    assert [r["cue"] for r in rows] == ["0:00", "0:40"]


def test_nothing_is_sent_anywhere_from_the_page():
    """The capture is clipboard-only. A bookmarklet that POSTed to the app would
    need CORS opened to a third-party origin, on a server running locally."""
    code = _code(BM)
    for banned in ("fetch(", "XMLHttpRequest", "localhost", "8000"):
        assert banned not in code, banned


def test_it_shows_what_it_found_before_you_import_it():
    code = _code(BM)
    assert "textarea" in code
    assert "captured " in BM          # the count, in the panel header
    assert "outerHTML" in code, "the debug view is how a wrong selector is found"


# ── the generated URL ────────────────────────────────────────────────────────

def test_the_generated_bookmarklet_is_a_usable_javascript_url():
    url = json.loads(GEN[GEN.index("=") + 1:].strip().rstrip(";").strip())
    assert url.startswith("javascript:")
    # Browsers stop honouring bookmarklets somewhere past ~8KB.
    assert len(url) < 8000, len(url)
    assert "%20" in url or "%7B" in url, "must be percent-encoded for an href"


def _generated_source() -> str:
    url = json.loads(GEN[GEN.index("=") + 1:].strip().rstrip(";").strip())
    return urllib.parse.unquote(url[len("javascript:"):])


def test_the_generated_file_is_in_step_with_its_source():
    """It is committed so a clean checkout builds, which means it can go stale.
    Markers from the source must survive minification and percent-encoding."""
    code = _generated_source()
    assert "open track page" in code
    assert "/track/" in code
    assert "outerHTML" in code


# ── routing: one box, two endpoints ──────────────────────────────────────────

def test_the_paste_box_routes_a_capture_to_the_markdown_endpoint():
    code = _code(IMPORTER)
    assert "CAPTURE_RE" in code
    # A JS regex literal escapes its slashes, so match the escaped form.
    pattern = code[code.index("const CAPTURE_RE"):][:300]
    assert r"\/track\/" in pattern, pattern[:120]
    assert "api.importMixMarkdown(paste" in code
    assert "api.importMixPaste(paste" in code, "plain text still has its route"


def test_both_import_calls_exist_and_hit_different_endpoints():
    assert '"/api/mixes/import-markdown"' in API
    assert '"/api/mixes/import-paste"' in API
