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
    js = _read("attributes.js")
    assert '"attr:" + ' in js or "`attr:${" in js


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
