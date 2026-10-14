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
    """PairDock is rendered beside the Library (and its Track detail) — not as
    a route of its own."""
    lib = APP[APP.index('{(route === "library" || route === "track") && ('):]
    lib = lib[:lib.index('{route === "queue" && (')]
    assert "<LibraryScreen" in lib
    assert "<PairDock" in lib
    # The player bar is deliberately NOT in here — see
    # test_the_player_bar_is_not_inside_a_route.
    assert "<PlayerBar" not in lib
    # And there is no nav entry that would take you to it.
    nav = SIDEBAR[SIDEBAR.index("const NAV = ["):]
    assert "pair" not in nav[:nav.index("]")].lower()


def test_the_library_layout_is_main_plus_a_resizable_dock():
    block = CSS[CSS.index(".lib-layout {"):]
    block = block[:block.index("}")]
    assert "grid-template-columns: minmax(0, 1fr) auto" in block
    assert "style={{ width }}" in _read("components/PairDock.jsx")
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
    """Chips built with a template literal write `chip active`. Only .chip.on
    existed once, so those chips silently showed no on-state at all."""
    assert ".chip.active" in CSS
    writers = [p.name for p in SRC.rglob("*.jsx") if "chip${" in p.read_text(encoding="utf-8")]
    assert writers, "nothing builds a chip class dynamically any more"


# ── revamp step 0: tokens + shell ────────────────────────────────────────────

THEME = _read("theme.js")
PALETTE = _read("shell/CommandPalette.jsx")


def test_one_token_source_in_root():
    """Every revamp token is a :root variable; components read var(--x)."""
    root = CSS[CSS.index(":root {"):]
    root = root[:root.index("\n}")]
    for name in ("--bg:", "--bg-sunk:", "--well:", "--bar:", "--card:", "--panel:",
                 "--raised:", "--line:", "--line-2:", "--line-soft:", "--fg:", "--fg-2:",
                 "--muted:", "--faint:", "--disabled:", "--vocal:", "--bed:", "--accent:",
                 "--ok:", "--warn:", "--fail:", "--run:", "--star:", "--playhead:",
                 "--tier-strong:", "--tier-good:", "--tier-maybe:", "--tier-long:"):
        assert name in root, name
    assert "--vocal: #ff5fae" in root and "--bed: #2dd4bf" in root
    assert "--accent: #5b8cff" in root
    assert ":focus-visible { outline: 2px solid var(--accent)" in CSS


def test_new_shell_files_carry_no_hex():
    """Colours come from the tokens. The shell and the primitives are the
    first files built on them and must not carry a hex of their own."""
    for rel in ("shell/Sidebar.jsx", "shell/CommandPalette.jsx", "shell/Styleguide.jsx",
                "components/ui/primitives.jsx", "components/PlayerBar.jsx",
                "components/LibraryScreen.jsx", "components/LibraryFilters.jsx",
                "components/TrackTable.jsx", "components/KeyChip.jsx",
                "components/PairDock.jsx", "components/PairCard.jsx",
                "components/TrackDetail.jsx", "components/StructureStrip.jsx",
                "components/SectionTable.jsx"):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", _read(rel)), rel


def test_camelot_colour_is_the_mixed_in_key_wheel():
    assert "(165 - (num - 1) * 30)" in THEME
    assert "hsl(${hue} 66% 58%)" in THEME and "hsl(${hue} 80% 76%)" in THEME
    assert "color: var(--key-ink)" in CSS


def test_the_rail_collapses_and_remembers():
    assert '"mashup.rail.v1"' in SIDEBAR
    assert ".app-shell.rail-collapsed { grid-template-columns: 56px" in CSS
    assert "rail-collapsed" in APP and "useRailCollapsed()" in APP
    # The pipeline readout opens the queue.
    assert 'onRoute("queue")' in SIDEBAR and "usePipelineReadout(" in APP


def test_cmd_k_palette_has_every_group_and_its_keys():
    for group in ('"Go to"', '"Actions"', '"Saved views"', '"Tracks"', '"Pairs"'):
        assert group in PALETTE, group
    assert "needle.length >= 2" in PALETTE           # tracks from two characters
    for key in ('"ArrowDown"', '"ArrowUp"', '"Enter"', '"Escape"'):
        assert key in PALETTE, key
    assert 'e.key === "k" || e.key === "K"' in APP and "<CommandPalette" in APP


def test_slash_and_space_are_global():
    assert 'e.key === "/"' in APP and "focusSearch" in APP
    assert 'data-search="library"' in _read("components/LibraryFilters.jsx")
    assert 'data-search="discover"' in _read("components/SoundCloudBrowser.jsx")
    assert 'e.code !== "Space"' in APP


def test_styleguide_is_a_dev_only_route():
    assert "import.meta.env.DEV" in APP
    assert 'has("styleguide")' in APP
    assert 'route === "styleguide" && STYLEGUIDE' in APP


def test_every_global_key_is_in_help():
    help_ = _read("components/HelpPanel.jsx")
    for k in ("⌘K", "/ focus search", "space play/pause", "esc back one layer"):
        assert k in help_, k
