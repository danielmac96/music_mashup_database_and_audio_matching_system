"""The sidebar shell, pinned from Python.

This repo has no JS test runner — the invariants that would hurt if they drifted
are asserted by reading the source, the same way
tests/test_studio_timing_pills_frontend.py does.

What matters here is one claim: **the pair dock is not behind a tab.** That is
the entire point of the revamp. If the dock ever goes back to being a route, the
app still works and still looks fine, and the thing it was rebuilt for is gone.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


APP = _read("App.jsx")
SIDEBAR = _read("shell/Sidebar.jsx")
CSS = _read("styles.css")


def test_the_dock_lives_beside_the_library_not_behind_a_tab():
    """PairDock is rendered inside the library route, next to LibraryScreen —
    not as a route of its own."""
    # From the JSX, not from the keyboard gate that mentions the same route.
    lib = APP[APP.index('{route === "library" && ('):]
    lib = lib[:lib.index('{route === "track" && (')]
    assert "<LibraryScreen" in lib
    assert "<PairDock" in lib
    # The player bar is deliberately NOT in here — see
    # test_the_player_bar_is_not_inside_a_route.
    assert "<PlayerBar" not in lib
    # And there is no nav entry that would take you to it.
    nav = SIDEBAR[SIDEBAR.index("const NAV = ["):]
    assert "pair" not in nav[:nav.index("]")].lower()


def test_the_library_layout_is_main_plus_a_404px_dock():
    block = CSS[CSS.index(".lib-layout {"):]
    block = block[:block.index("}")]
    assert "404px" in block, "the dock is no longer the width the design fixes"
    # The transport is no longer a row of this grid — it belongs to the app, so
    # a grid-column rule here would only ever be a leftover.
    # Anchored: `.module.transport` is an unrelated audition-era rule.
    tr = CSS[CSS.index("\n.transport {"):]
    tr = tr[:tr.index("}")]
    assert "grid-column" not in tr
    assert "flex: none" in tr


def test_every_route_renders_something():
    for route in ("library", "queue", "track", "discovery", "mixes", "studio"):
        assert f'route === "{route}"' in APP, route


def test_a_track_screen_keeps_library_lit_in_the_rail():
    """The detail view is a place inside Library, not a fifth destination — it
    is reached from a row and dismissed with escape."""
    assert 'route === "track" ? "library" : route' in APP


def test_the_library_is_fetched_once_at_the_top():
    """Four screens read the same unpaginated endpoint. Four fetches of it would
    be four answers that can disagree while the pipeline is running."""
    assert "const library = useLibrary()" in APP
    assert "const ratings = useRatings()" in APP
    # Nothing below App re-fetches the whole track list for the library screen.
    assert "api.getTracks()" not in _read("components/LibraryScreen.jsx")
    assert "api.getTracks()" not in _read("components/TrackTable.jsx")


def test_each_screen_registers_its_own_rail_block():
    """The rail does not know four screens' internals; each screen hands it a
    block. A rail that reached into every screen would have to change whenever
    any of them did."""
    assert "onRailSlot" in APP
    assert "onRailSlot" in _read("components/LibraryScreen.jsx")
    assert "onRailSlot" in _read("components/Discovery.jsx")
    assert "rail-slot" in CSS


def test_the_header_status_contract_survives():
    """Every screen still reports `{locked, text}` and the shell still renders
    it — the top bar is gone, not the readout."""
    assert "setHeaderStatus" in APP
    assert "float-status" in APP and ".float-status" in CSS


def test_the_shell_does_not_scroll():
    """The rail must not scroll away with the page. Only the main column moves."""
    block = CSS[CSS.index(".app-shell {"):]
    block = block[:block.index("}")]
    assert "height: 100vh" in block
    assert "overflow: hidden" in block


def test_the_first_run_wizard_still_has_a_way_in():
    """Nothing is configured yet, so there is nothing for a rail to navigate."""
    assert 'className="app-shell setup"' in APP
    assert "<SetupWizard" in APP
    assert ".app-shell.setup" in CSS


def test_no_dead_player_bar_rules_remain():
    """The DELETED PlayerBar emitted .player-* while the stylesheet only ever
    defined .pb-*, so its whole interior was unstyled.

    There is a PlayerBar again — the global transport — and this test is what
    stops it repeating that mistake: it emits `.transport` / `.tr-*`, which are
    the classes that actually exist, and neither dead family came back.
    """
    bar = _read("components/PlayerBar.jsx")
    assert 'className="transport' in bar or "className={`transport" in bar
    assert "player-" not in bar
    assert not re.search(r"^\.pb-", CSS, re.M)
    assert not re.search(r"^\.player-bar\b", CSS, re.M)


def test_chip_active_is_defined():
    """MashupSuggestions writes `chip active` in four places. Only .chip.on
    existed, so those chips silently showed no on-state at all."""
    assert ".chip.active" in CSS
    assert "chip${" in _read("components/MashupSuggestions.jsx")
