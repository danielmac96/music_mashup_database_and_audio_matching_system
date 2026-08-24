"""The crate-badge wiring in the JSX, pinned from Python.

This repo has no JS test runner and adding one is a separate decision, so the
frontend invariants that matter are asserted by reading the source — the same
approach tests/test_scraped_rows.py and tests/test_stale_frontend.py take.

What is worth pinning here is not that a chip renders prettily; it is that
*both* Discover panes are wired, because the two are easy to drift apart. A
suggestion row excludes library-owned tracks, so its crate chip is the only
membership signal it has.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def test_track_row_renders_a_crate_chip():
    src = _read("components/ScRows.jsx")
    assert "crate-chip" in src
    assert "crates" in src.split("export function PlaylistRow")[0]


def test_the_chip_is_not_a_button():
    """Read-only was the explicit decision — adding stays on the tick-box plus
    the bulk CrateAddButton path."""
    src = _read("components/ScRows.jsx")
    chip_line = [l for l in src.splitlines() if "crate-chip" in l]
    assert chip_line, "no crate-chip in ScRows.jsx"
    assert not any("<button" in l for l in chip_line), chip_line


def test_both_discover_panes_use_the_membership_hook():
    for pane in ("components/SoundCloudBrowser.jsx", "components/Suggestions.jsx"):
        src = _read(pane)
        assert "useCrateMembership" in src, pane
        assert "crates={" in src, pane


def test_the_hook_refires_on_the_crate_refresh_counter():
    """crateRefresh is bumped after a successful add; that is what makes the
    badge appear without a reload."""
    for pane in ("components/SoundCloudBrowser.jsx", "components/Suggestions.jsx"):
        src = _read(pane)
        assert "useCrateMembership(" in src
        call = src.split("useCrateMembership(")[1].split(")")[0]
        assert "crateRefresh" in call, (pane, call)


def test_the_hook_guards_against_a_stale_response():
    src = _read("hooks/useCrateMembership.js")
    assert "useRef" in src and "token" in src


def test_the_api_client_posts_membership():
    src = _read("api.js")
    assert "crateMembership" in src
    assert "/api/crates/membership" in src


def test_the_chip_has_styles_distinct_from_the_in_library_flag():
    css = "\n".join(p.read_text(encoding="utf-8")
                    for p in SRC.rglob("*.css"))
    assert ".crate-chip" in css
    assert ".sc-crates" in css
