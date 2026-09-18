"""The way into the track detail screen, pinned from Python.

Screen 1b was built in full during the revamp and then reachable only by
double-clicking a library row, with a `title=` tooltip as its whole affordance.
A screen nobody can find is the same as a screen nobody built, and that failure
is silent: the app works, the tests pass, and the feature is gone.

Two invariants:

* the title block is a LINK — a visible one — and clicking it does not also
  toggle the row's selection, because the row still belongs to the pair dock;
* a partner in the rail walks to ITS track, flipping which side of the pair is
  being asked about on the way. Opening a bed as a vocal would silently answer a
  different question than the one that was clicked.
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
TABLE = _read("components/TrackTable.jsx")
DETAIL = _read("components/TrackDetail.jsx")
RAIL = _read("components/PartnersRail.jsx")
CSS = _read("styles.css")


def _row(src: str) -> str:
    """The TrackRow component's body — not the TrackTable wrapper above it,
    which mentions the same prop names."""
    return src[src.index("function TrackRow("):]


# ── the library row ────────────────────────────────────────────────────────

def test_the_title_is_a_button_that_opens_the_track():
    row = _row(TABLE)
    block = row[row.index('className="tt-name"'):]
    block = block[:block.index("</button>")]
    assert "<button" in row[:row.index('className="tt-name"')][-200:], \
        "tt-name must be a button, not a div — a div is not a link"
    assert "onOpen(t.id)" in block


def test_opening_a_track_does_not_also_toggle_the_row():
    """Without stopPropagation one click navigates AND re-scopes the dock, so
    you arrive on the detail screen having silently changed what the dock was
    showing behind it."""
    row = _row(TABLE)
    block = row[row.index('className="tt-name"'):]
    block = block[:block.index("</button>")]
    assert "e.stopPropagation()" in block


def test_the_row_itself_still_scopes_the_pair_dock():
    """The README's interaction contract: selecting a library row re-scopes the
    dock. The link is additive — it must not have taken that gesture over."""
    row = _row(TABLE)
    head = row[:row.index('className="tt-play"')]
    assert "onClick={() => onSelect(t.id)}" in head
    # And the double-click fallback the revamp shipped with still works.
    assert "onDoubleClick={() => onOpen(t.id)}" in head


def test_the_link_is_visible_before_it_is_clicked():
    """The bug was an invisible affordance, so the hover state IS the fix."""
    assert ".tt-go" in CSS
    assert ".tt-row:hover .tt-go" in CSS
    assert ".tt-name:hover .tt-text" in CSS
    # A button in a grid cell reverts to centred text with default padding,
    # which breaks the title column's ellipsis.
    block = CSS[CSS.index("\n.tt-name {"):]
    block = block[:block.index("}")]
    assert "text-align: left" in block
    assert "border: 0" in block


def test_a_long_title_does_not_swallow_its_own_chevron():
    """Most titles overflow this column. With the chevron inside the ellipsised
    element, the affordance is clipped on exactly the rows that need it — and
    that is what the browser check caught."""
    for line, text, go in ((".tt-title", ".tt-text", ".tt-go"),
                           (".partner-line", ".pc-text", ".partner-go")):
        row = CSS[CSS.index(f"\n{line} {{"):]
        row = row[:row.index("}")]
        assert "display: flex" in row, line
        cell = CSS[CSS.index(f"\n{text} {{"):]
        cell = cell[:cell.index("}")]
        assert "text-overflow: ellipsis" in cell, text
        chev = CSS[CSS.index(f"\n{go} {{"):]
        chev = chev[:chev.index("}")]
        assert "flex: none" in chev, go


# ── the partners rail ──────────────────────────────────────────────────────

def test_a_partner_walks_to_its_own_track():
    assert "onOpenTrack" in RAIL
    block = RAIL[RAIL.index('className="partner-open"'):]
    block = block[:block.index("</button>")]
    # songId is the PARTNER's id — the other side of the pair, not the track
    # whose screen you are already on.
    assert "onOpenTrack(songId)" in block
    assert "e.stopPropagation()" in block, \
        "the partner row loops on click; opening must not also start audio"


def test_opening_a_partner_flips_the_role():
    fn = DETAIL[DETAIL.index("const openPartner ="):]
    fn = fn[:fn.index("};")]
    assert "onRole(" in fn and "instrumental" in fn and "vocal" in fn
    assert "onOpenTrack(id)" in fn
    assert "onOpenTrack={openPartner}" in DETAIL


def test_the_rails_link_treatment_is_scoped_to_the_rail():
    """.pc-title is shared with PairCard, where nothing is clickable. A hover
    rule on the bare class would put a phantom link on every pair in the dock."""
    assert ".partner-open" in CSS
    assert not re.search(r"^\.pc-title:hover", CSS, re.M)
    assert not re.search(r"^\.pc-title\b[^{]*:hover", CSS, re.M)


def test_the_way_out_is_not_buried_under_the_status_pill():
    """.float-status is absolutely positioned top-right of the main column,
    which on this screen is `song #N` and the esc button. Measured in the
    browser: the pill covered both. It has pointer-events:none so esc still
    worked, but a dismiss control you cannot see is the same bug as a way in
    you cannot see."""
    block = CSS[CSS.index("\n.float-status {"):]
    block = block[:block.index("}")]
    assert "position: absolute" in block and "right: 16px" in block
    assert "onStatus(null)" in DETAIL, \
        "the detail screen must not publish a float status over its own header"
    assert "scored pairings" not in DETAIL


def test_walking_to_a_partner_starts_at_the_top_of_the_page():
    """The screen re-renders rather than remounting, so scroll survives — and
    you land mid-page under somebody else's hero."""
    assert "mainRef" in DETAIL
    assert 'className="detail-main" ref={mainRef}' in DETAIL
    assert "mainRef.current?.scrollTo" in DETAIL


# ── the wiring ─────────────────────────────────────────────────────────────

def test_app_hands_the_detail_screen_a_way_to_change_tracks():
    block = APP[APP.index("<TrackDetail"):]
    block = block[:block.index("/>")]
    assert "onOpenTrack={setSelectedTrackId}" in block, \
        "the raw setter, not selectTrack — that one toggles, and opening a " \
        "partner you are already scoped to would clear the selection instead"


def test_the_whole_name_cell_opens_the_track():
    """The button spans the column's WIDTH already; at content height the strips
    above and below the two lines fell through to the row, so a click half a row
    from the title scoped the dock instead of opening the track."""
    block = CSS[CSS.index(".tt-name {"):]
    block = block[:block.index("}")]
    assert "height: 100%" in block
    assert "display: flex" in block and "flex-direction: column" in block
    # ...and the row still owns everything else.
    assert "onClick={() => onSelect(t.id)}" in _row(TABLE)
