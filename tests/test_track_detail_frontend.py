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
CSS = _read("styles.css")


def _row(src: str) -> str:
    """The TrackRow component's body — not the TrackTable wrapper above it,
    which mentions the same prop names."""
    return src[src.index("const Row = memo("):]


# ── the library row ────────────────────────────────────────────────────────

def test_the_title_is_a_button_that_opens_the_track():
    cell = TABLE[TABLE.index('case "title": {'):]
    cell = cell[:cell.index('case "bpm":')]
    block = cell[cell.index('<button className="tt-title"'):]
    block = block[:block.index("</button>")]
    assert "ctx.onOpen(t.id)" in block, "a span is not a link — keep it a button"


def test_opening_a_track_does_not_also_toggle_the_row():
    """Without stopPropagation one click navigates AND re-scopes the dock, so
    you arrive on the detail screen having silently changed what the dock was
    showing behind it."""
    cell = TABLE[TABLE.index('<button className="tt-title"'):]
    cell = cell[:cell.index("</button>")]
    assert "e.stopPropagation()" in cell


def test_the_row_itself_still_scopes_the_pair_dock():
    """The README's interaction contract: selecting a library row re-scopes the
    dock. The link is additive — it must not have taken that gesture over."""
    row = _row(TABLE)
    assert "onClick={() => ctx.select(t.id)}" in row
    assert "onDoubleClick={() => ctx.onOpen(t.id)}" in row


def test_the_link_is_visible_before_it_is_clicked():
    """The bug was an invisible affordance, so the hover state IS the fix — and
    a button reverts to centred text with a border unless told otherwise."""
    assert ".tt-title:hover { text-decoration: underline; }" in CSS
    block = CSS[CSS.index("\n.tt-title {"):]
    block = block[:block.index("}")]
    assert "text-align: left" in block and "border: 0" in block
    assert "text-overflow: ellipsis" in block


def test_a_long_partner_name_truncates_instead_of_pushing_the_row():
    block = CSS[CSS.index("\n.dt-partnername {"):]
    block = block[:block.index("}")]
    assert "text-overflow: ellipsis" in block and "white-space: nowrap" in block
    grid = CSS[CSS.index("\n.dt-partner {"):]
    assert "minmax(0, 1fr)" in grid[:grid.index("}")]


# ── partners, and the screen itself ────────────────────────────────────────

def test_a_partner_walks_to_its_own_track():
    """The Partners group: a partner's name opens THAT track (the other side of
    the pair, not the one whose screen you are on); ▶ loops the pair and ◫
    opens it in Studio."""
    block = DETAIL[DETAIL.index('className="dt-partnername"'):]
    block = block[:block.index("</button>")]
    assert "openPartner(pid, as)" in block
    assert "onPlayPair(c)" in DETAIL and "onStudio(c)" in DETAIL


def test_opening_a_partner_flips_the_role():
    """You got here looking at this track as the vocal, so the partner you
    clicked is a bed — it opens as one."""
    fn = DETAIL[DETAIL.index("const openPartner ="):]
    fn = fn[:fn.index("};")]
    assert "onRole(asRole)" in fn and "onOpenTrack(id)" in fn
    assert 'as: "instrumental"' in DETAIL and 'as: "vocal"' in DETAIL


def test_the_way_out_is_not_buried_under_the_status_pill():
    """.float-status sits in flow in the rail, never over a screen's header."""
    block = CSS[CSS.index("\n.float-status {"):]
    block = block[:block.index("}")]
    assert "position: absolute" not in block
    assert "rail-status" in _read("shell/Sidebar.jsx")
    assert "onStatus(null)" in DETAIL
    assert '← Library</button>' in DETAIL


def test_walking_to_a_partner_starts_at_the_top_of_the_page():
    """The screen re-renders rather than remounting, so scroll survives — and
    you land mid-page under somebody else's hero."""
    assert 'className="detail" ref={mainRef}' in DETAIL
    assert "mainRef.current?.scrollTo" in DETAIL


def test_the_detail_replaces_the_table_and_the_dock_stays():
    """Track detail renders inside the Library column with the pair dock beside
    it, and the Library stays mounted (hidden) underneath — so esc lands back
    on the same filters, selection and scroll."""
    lib = APP[APP.index('{(route === "library" || route === "track") && ('):]
    lib = lib[:lib.index('{route === "queue" && (')]
    assert "<LibraryScreen" in lib and "<TrackDetail" in lib and "<PairDock" in lib
    assert 'className={`lib-pane${route === "track" ? " away" : ""}`}' in lib
    css = CSS[CSS.index("\n.lib-pane.away {"):]
    assert "visibility: hidden" in css[:css.index("}")]


def test_five_disclosures_remembered():
    assert '"mashup.detailGroups.v1"' in DETAIL
    for gid in ('id="attrs"', 'id="stems"', 'id="src"', 'id="partners"', 'id="actions"'):
        assert gid in DETAIL, gid
    prim = _read("components/ui/primitives.jsx")
    assert "localStorage.setItem(storageKey" in prim
    # Actions: re-run each stage, BPM ×2 / ÷2, edit key.
    for call in ("api.startSeparate", "api.startAnalyze", "api.startStructure",
                 "fixBpm(f.bpm * 2)", "fixBpm(f.bpm / 2)", "<TrackEditor"):
        assert call in DETAIL, call


def test_space_plays_here_and_the_dock_keeps_its_other_keys():
    assert 'e.code === "Space"' in DETAIL and "playHook()" in DETAIL
    app = APP
    assert '{ space: route === "library" }' in app
    hook = _read("hooks/usePairDock.js")
    assert "if (!space) return;" in hook


# ── the wiring ─────────────────────────────────────────────────────────────

def test_app_hands_the_detail_screen_a_way_to_change_tracks():
    block = APP[APP.index("<TrackDetail"):]
    block = block[:block.index("/>")]
    assert "onOpenTrack={setSelectedTrackId}" in block, \
        "the raw setter, not selectTrack — that one toggles, and opening a " \
        "partner you are already scoped to would clear the selection instead"


def test_the_artist_and_the_rest_of_the_row_select():
    """Only the title opens the track; the artist beside it and every other
    cell belong to the row, which scopes the dock."""
    cell = TABLE[TABLE.index('case "title": {'):]
    cell = cell[:cell.index('case "bpm":')]
    artist = cell[cell.index('className="tt-artist"') - 10:]
    assert "onClick" not in artist[:artist.index("</span>")]


