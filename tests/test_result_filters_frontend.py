"""The Discover filter/sort bar, pinned from Python.

Same approach as tests/test_crate_badges_frontend.py: this repo has no JS test
runner and adding one is a separate decision, so the invariants that would
actually hurt if they drifted are asserted by reading the source.

The two that matter:

* both panes must render `visible`, not `items` — otherwise the bar is
  decorative and filtering does nothing;
* selection must derive from `visible` too, or "select all" selects rows the
  user cannot see and the bulk import sends them.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"
PANES = ("components/SoundCloudBrowser.jsx", "components/Suggestions.jsx")


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def test_both_panes_render_the_bar():
    for pane in PANES:
        src = _read(pane)
        assert "ResultFilters" in src, pane
        assert "useResultFilters" in src, pane


def test_both_panes_map_over_visible_not_items():
    for pane in PANES:
        src = _read(pane)
        assert "visible.map(" in src, pane
        assert "items.map(" not in src, pane
        assert "rows.map(" not in src, pane


def test_selection_derives_from_visible():
    """Filtering a row out must not leave it selected for import — the hazard
    run()'s clear() comment already calls out."""
    for pane in PANES:
        src = _read(pane)
        call = src.split("useRowSelection(")[1].split(")")[0]
        assert "visible" in call, (pane, call)


def test_filters_reset_when_the_listing_changes():
    for pane in PANES:
        assert "resetFilters" in _read(pane), pane


def test_the_filter_module_makes_no_network_call():
    """Scope is what is loaded. The browse layer shares one scraped client_id
    with the frozen mixes resolver; spending its rate limit on a nicer sort is
    the trade this codebase refuses."""
    for rel in ("hooks/useResultFilters.js", "components/ResultFilters.jsx"):
        src = _read(rel)
        for banned in ("fetch(", "api.", "await "):
            assert banned not in src, (rel, banned)


def test_sorting_defaults_to_unsorted():
    """SoundCloud's own relevance order is meaningful and must survive until the
    user asks for something else."""
    src = _read("hooks/useResultFilters.js")
    assert re.search(r"sort:\s*\"\"", src)


def test_apply_filters_is_exported_as_a_plain_function():
    src = _read("hooks/useResultFilters.js")
    assert "export function applyFilters(" in src


def test_genre_is_a_dropdown_not_free_text():
    """SoundCloud genre strings are unbounded user input."""
    bar = _read("components/ResultFilters.jsx")
    assert "genresIn" in bar
    assert 'type="text"' not in bar


def test_numeric_sorts_put_missing_last_in_both_directions():
    """A row with no play count is unknown, not unpopular."""
    src = _read("hooks/useResultFilters.js")
    cmp_body = src.split("function makeCmp(")[1].split("\n}")[0]
    assert "if (!av) return 1;" in cmp_body
    assert "if (!bv) return -1;" in cmp_body


def test_the_bar_says_how_many_of_how_many():
    bar = _read("components/ResultFilters.jsx")
    assert "of ${rows.length} loaded" in bar


def test_the_bar_has_styles():
    css = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.css"))
    assert ".sc-filterbar" in css
