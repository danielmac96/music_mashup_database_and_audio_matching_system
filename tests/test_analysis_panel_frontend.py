"""The Analysis panel's frontend contract (readme §9, C): read the source."""
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "frontend" / "src"


def _read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def test_attributes_are_fetched_once_in_app():
    app = _read("App.jsx")
    assert app.count("useAttributes()") == 1
    for screen in ("LibraryScreen", "TrackDetail", "AnalysisScreen"):
        assert re.search(rf"<{screen}[^>]*attributes={{attributes}}", app, re.S), screen


def test_the_api_has_both_calls():
    api = _read("api.js")
    assert "/api/analysis/attributes" in api and "/api/analysis/attributes/visibility" in api


def test_attribute_columns_are_keyed_by_id():
    # Ids are "attr:<id>", so a dragged width or a moved column follows its
    # attribute rather than a position.
    js = _read("components/LibraryScreen.jsx")
    assert "`attr:${a.id}`" in js


def test_the_rail_has_analysis_between_queue_and_discover():
    nav = _read("shell/Sidebar.jsx")
    order = [m.group(1) for m in re.finditer(r'\["(\w+)", "', nav)]
    assert order.index("analysis") == order.index("queue") + 1
    assert order.index("discovery") == order.index("analysis") + 1
    assert 'route === "analysis"' in _read("App.jsx")


def test_every_panel_class_exists_in_the_stylesheet():
    css = _read("styles.css")
    jsx = _read("components/AnalysisScreen.jsx")
    for cls in set(re.findall(r'className="([^"]+)"', jsx)):
        for c in cls.split():
            assert f".{c}" in css, c


def test_the_panel_toggles_library_and_detail_and_never_recomputes():
    jsx = _read("components/AnalysisScreen.jsx")
    assert 'toggle("library"' in jsx and 'toggle("detail"' in jsx
    assert "bulk" not in jsx and "reanalys" not in jsx.lower()


def test_attribute_columns_come_from_the_catalogue_and_the_panel_switches():
    """Any catalogue attribute can be a Library column; whether it shows is the
    Analysis panel's Library switch, so the Columns menu and the panel agree."""
    screen = _read("components/LibraryScreen.jsx")
    assert "const attrColumn = (a)" in screen and "id: `attr:${a.id}`" in screen
    assert 'attributes.toggle("library", c.attr.id)' in screen
    view = _read("hooks/useColumnWidths.js")
    assert "const visible = c.attr ? attrSet.has(id)" in view


def test_attribute_sort_keys_are_understood():
    js = _read("hooks/useLibraryFilters.js")
    assert 'startsWith("attr:")' in js


def test_track_detail_shows_the_detail_attributes():
    td = _read("components/TrackDetail.jsx")
    assert "visibility.detail" in td and "fmtAttr(" in td
    css = _read("styles.css")
    for cls in set(re.findall(r'className="(attr-[^"\s]+)', td)):
        assert f".{cls}" in css, cls


def test_table_cells_are_compact_and_small_numbers_stay_readable():
    """Seen in the browser: '-10.4 LU…' (the unit repeated in every cell, the
    header already says LUFS), 'Hands…' (a style name in a narrow column) and
    Energy as 0.000 (values around 2e-4 at three decimals)."""
    js = _read("attributes.js")
    assert "withUnit" in js and "toPrecision(" in js
    screen = _read("components/LibraryScreen.jsx")
    assert 'w: a.kind === "number" ? 70 : 110' in screen      # text columns are wider
    tt = _read("components/TrackTable.jsx")
    assert "fmtAttr(col.attr, v, false)" in tt


def test_compare_says_whether_the_two_tracks_mash():
    """Comparing two tracks stopped at their attributes; it now shows the scored
    pair both ways round, with a loop and a Studio hand-off."""
    src = _read("components/AnalysisScreen.jsx")
    assert "function PairVerdict(" in src
    assert 'comboType: "vocal_over_instrumental"' in src
    assert "one(a, b), one(b, a)" in src
    assert "onPlay(c)" in src and "onOpen(c)" in src
    app = _read("App.jsx")
    assert "onPlayPair={dock.play} onOpenPair={pairToStudio}" in app
