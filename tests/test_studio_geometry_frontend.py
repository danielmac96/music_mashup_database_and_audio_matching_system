"""The Studio's timeline geometry, pinned from Python.

Same approach as tests/test_studio_timing_pills_frontend.py: this repo has no JS
test runner, so the invariants that would actually hurt if they drifted are
asserted by reading the source.

The one that matters here is arithmetic, not styling. `HEADER_W` converts pixels
to seconds and positions the playhead; `.studio-grid`'s first column plus its
column-gap is where the lane canvases actually begin. If those two disagree,
every clip is drawn at the wrong time and the playhead sits over the wrong
moment — and nothing about the page looks broken, which is why it needs a test
rather than an eye.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"
STUDIO = (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
RAIL = (SRC / "components" / "StudioRail.jsx").read_text(encoding="utf-8")
CSS = (SRC / "styles.css").read_text(encoding="utf-8")


def _header_w() -> int:
    m = re.search(r"^const HEADER_W = (\d+)", STUDIO, re.M)
    assert m, "HEADER_W is gone from MixStudio.jsx"
    return int(m.group(1))


def _grid_gutter() -> int:
    """The first column plus the column gap — where the lane canvases start."""
    block = CSS[CSS.index(".studio-grid {"):]
    block = block[:block.index("}")]
    col = re.search(r"grid-template-columns:\s*(\d+)px", block)
    gap = re.search(r"column-gap:\s*(\d+)px", block)
    assert col, "the grid's first column is no longer a fixed px width"
    return int(col.group(1)) + (int(gap.group(1)) if gap else 0)


def test_the_gutter_arithmetic_agrees():
    """The single most load-bearing number on the screen."""
    assert _header_w() == _grid_gutter()


def test_the_playhead_starts_at_the_gutter():
    """One cursor over the ruler, both lanes and the bar grid — which only works
    if it is offset by exactly the gutter the canvases begin at."""
    assert "left: HEADER_W + playheadX" in STUDIO


def test_the_lane_cards_are_border_box():
    """The prototype's bug: a card measuring its padding on top of its width
    pushes every canvas out of line with the ruler above it. `.studio-grid > *`
    covers the cards, the corner and the lanes in one rule."""
    block = CSS[CSS.index(".studio-grid > * {"):]
    assert "box-sizing: border-box" in block[:block.index("}")]


def test_all_four_rows_share_one_axis():
    """Ruler, both lanes and the bar grid are painted from the same viewStart
    and pps. A row painted from its own view would drift as soon as you scroll."""
    paint = STUDIO[STUDIO.index("// ── painting"):]
    paint = paint[:paint.index("}, [lanes, viewStart")]
    assert "paintLane(c, l, viewStart, pps" in paint
    assert "paintRuler(rulerRef.current, viewStart, pps" in paint
    assert "paintRuler(barGridRef.current, viewStart, pps" in paint


def test_both_tracks_stay_loaded_in_full():
    """The suggestion is a REGION, not a cut. placementFor sets clipStart and
    clipEnd — a trim window over the whole stem — and never shortens rawDur, so
    the rest of each song is still there to scroll to."""
    fn = STUDIO[STUDIO.index("export function placementFor"):]
    fn = fn[:fn.index("\n}")]
    assert "clipStart" in fn and "clipEnd" in fn
    assert "rawDur" not in fn


def test_the_rail_draws_the_matchers_suggestion_as_a_tick():
    """A manual edit has to read as a divergence from the recipe, which means
    the suggested value stays visible after you move away from it."""
    assert "knob-tick" in RAIL
    # ...and a suggestion that does not exist is NOT drawn at zero: null means
    # the matcher had nothing to say, which is not the same as "zero".
    assert "suggest == null ? null" in RAIL


def test_no_grid_is_not_a_measured_zero():
    """alignment_offset is null when neither side has a stored downbeat grid.
    Printing '0 ms' for that would claim a measurement nobody took."""
    assert "no grid" in STUDIO
    assert "unmeasured, not zero" in RAIL


def test_the_lane_card_keeps_only_what_fits():
    """138px holds a name, a tempo, a key and solo/mute. Everything else moved
    to the rail; leaving it on the card is how the column overflows."""
    card = STUDIO[STUDIO.index("{/* lane card — 138px"):]
    card = card[:card.index("{/* lane canvas */}")]
    for gone in ("lh-stem-seg", "lh-gain", "lh-rate", "⚡ key", "⇥ grid"):
        assert gone not in card, f"{gone} is back on the 138px card"
    for moved in ("stemOrder", "⚡ key", "⇥ grid", "SELECTED LANE"):
        assert moved in RAIL, f"{moved} did not make it to the rail"


def test_studio_still_exports_an_fl_session():
    """The design does not draw it, which is not the same as removing it."""
    assert "handleSessionExport" in STUDIO
    assert "FL session" in STUDIO
