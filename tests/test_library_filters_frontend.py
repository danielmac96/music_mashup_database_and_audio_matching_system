"""The library's filters and sort, pinned from Python.

The rule this file exists for is the same one tests/test_result_filters_frontend.py
guards for Discover: **filtering must not fetch.** GET /api/tracks returns every
song unpaginated, so narrowing the list is arithmetic over rows already in
memory. A filter that quietly re-queries turns "show me fewer" into "spend a
request", and once one control does it the rest follow.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


HOOK = _read("hooks/useLibraryFilters.js")
BAR = _read("components/LibraryFilters.jsx")
TABLE = _read("components/TrackTable.jsx")
THEME = _read("theme.js")


def test_filtering_makes_no_request():
    for rel, src in (("hooks/useLibraryFilters.js", HOOK),
                     ("components/LibraryFilters.jsx", BAR),
                     ("components/TrackTable.jsx", TABLE)):
        for banned in ("fetch(", "api.", "XMLHttpRequest"):
            assert banned not in src, (rel, banned)


def test_the_table_renders_what_the_filter_returned():
    """`visible` is the filtered, sorted list. Rendering the raw rows next to a
    count that describes the filtered ones is how a filter appears to do
    nothing."""
    screen = _read("components/LibraryScreen.jsx")
    assert "tracks={visible}" in screen
    assert "visible.length.toLocaleString()" in screen


def test_unsorted_is_the_default_and_a_header_cycles_back_to_it():
    """Rows arrive in import order, which is the order you added them, and that
    is often exactly what you are looking for: a header click cycles first
    direction → the other → off."""
    fn = HOOK[HOOK.index("export function cycleSort"):]
    fn = fn[:fn.index("\n}")]
    assert "return { id, dir: first };" in fn
    assert "return { id, dir: other };" in fn
    assert "return { id: null, dir: null };" in fn
    view = _read("hooks/useColumnWidths.js")
    assert "sort: { id: null, dir: null }" in view


def test_missing_values_sort_last_in_both_directions():
    """A track with no play count is unknown, not unpopular. The direction sign
    is applied only after the null checks, so nulls sink either way."""
    fn = HOOK[HOOK.index("function compare("):]
    fn = fn[:fn.index("\n// Unsorted")]
    assert "if (x == null) return 1;" in fn
    assert "if (y == null) return -1;" in fn
    assert fn.index("if (x == null) return 1;") < fn.index("* sign")


def test_a_range_filter_excludes_rows_with_no_value():
    """An unanalysed track has no tempo and an undated upload has no year.
    Keeping them in every band would put un-mixable rows in all of them, and
    silently treat "unknown" as "in range"."""
    apply_fn = HOOK[HOOK.index("export function applyLibraryFilters"):]
    assert "if (f0.bpm == null) return false;" in apply_fn
    assert "if (!y) return false;" in apply_fn


def test_the_genre_facet_comes_from_the_library():
    """SoundCloud genre strings are unbounded user input. The old toolbar
    shipped a hard-coded Pop / Hip Hop / Rap / EDM and hid everything else."""
    assert "export function facetsOf" in HOOK
    assert "facets.genres" in BAR
    for hard_coded in ('"Pop"', '"Hip Hop"', '"EDM"'):
        assert hard_coded not in HOOK and hard_coded not in BAR


def test_the_key_filter_walks_the_camelot_wheel():
    """8A +/-1 has to mean what a DJ means by it: 8A, 8B, 7A and 9A all mix, and
    none of them needs a pitch shift. Distance is over the wheel POSITION, so
    the relative major/minor counts as the same place."""
    fn = HOOK[HOOK.index("export function wheelDistance"):]
    fn = fn[:fn.index("\n}")]
    assert "parseCamelot" in fn
    assert "12" in fn and "6" in fn          # folded to the shorter way round
    assert "letter" not in fn                # A vs B must not add distance


def test_the_filter_chips_cannot_shrink():
    """The row is only ~830px at this frame width. A chip allowed to shrink
    turns 122-136 into an ellipsis, and the number is the whole chip."""
    css = _read("styles.css")
    block = css[css.index(".fchip {"):]
    block = block[:block.index("}")]
    assert "flex: none" in block
    assert "white-space: nowrap" in block
    bar = css[css.index("\n.filter-bar {"):]
    assert "flex-wrap: wrap" in bar[:bar.index("}")]


def test_the_views_are_defined_once():
    """The rail counts them and the table filters by them. Two definitions of
    'ready to mash' would let the sidebar and the list disagree."""
    for name in ("isReadyToMash", "needsAttention", "isRecentlyAdded"):
        assert f"export function {name}" in THEME or f"export const {name}" in THEME
    assert "isReadyToMash" in HOOK


def test_a_track_star_is_the_best_pairing_it_appears_in():
    """There is no per-song rating store, and this deliberately does not add
    one: the thing being judged is a pairing."""
    ratings = _read("hooks/useRatings.js")
    fn = ratings[ratings.index("const bySong = useMemo"):]
    fn = fn[:fn.index("}, [rows])")]
    assert "f.vocal_song_id, f.inst_song_id" in fn
    assert "stars > out[id]" in fn


# ── columns ──────────────────────────────────────────────────────────────────
# The table is CSS-grid divs, so the column model IS the grid template. Head and
# rows share one string; two copies would drift and every cell below the header
# would draw under the wrong column.

def test_the_head_and_the_rows_share_one_width():
    """Header cells and row cells are sized by the same widthOf(c), so a cell
    can never draw under the wrong column."""
    assert TABLE.count("style={{ width: widthOf(c) }}") == 1          # the header
    assert "visible.map((c) => [c, widthOf(c)])" in TABLE             # the rows
    assert "style={{ width: w }}" in TABLE


def test_every_column_declares_an_id_and_a_width():
    """Widths and order are stored per id. An index would silently re-point
    every stored width the first time a column moved, and the symptom — one
    narrow column somewhere else — looks nothing like the cause."""
    cols = TABLE[TABLE.index("export const COLUMNS = ["):]
    cols = cols[:cols.index("];")]
    entries = [l for l in cols.split("\n") if l.strip().startswith("{ id:")]
    ids = [e.split('"')[1] for e in entries]
    assert ids[:14] == ["art", "title", "bpm", "key", "energy", "genre", "year", "len",
                        "vox", "best", "pairs", "plays", "rating", "status"], ids
    for line in entries:
        assert " w: " in line and " min: " in line, line


def test_the_title_takes_the_slack():
    """Widths are fixed px; on a wide window the title takes what is left over
    rather than the table leaving a gap at the right."""
    assert "const slack = Math.max(0, box.w - sum);" in TABLE
    assert 'c.width + (c.id === "title" ? slack - titleCut : 0)' in TABLE


def test_a_grip_drag_never_reaches_the_sort_header():
    grip = TABLE[TABLE.index("const onGrab"):]
    grip = grip[:grip.index("const handlers")]
    assert "e.stopPropagation()" in grip
    assert "resizing.current = true" in grip, "a resize must not start a column drag"
    assert 'className="tt-grip"' in TABLE
    assert "onResetWidth(c.id)" in TABLE, "double-click should restore the default"
    assert "if (resizing.current) { e.preventDefault(); return; }" in TABLE


def test_the_table_is_virtualised_with_fixed_rows():
    """5,000 tracks: only the rows in (and just around) the scroll window are in
    the DOM, at a fixed 32px."""
    assert "export const ROW_H = 32;" in TABLE
    assert "tracks.slice(first, first + count)" in TABLE
    assert "height: tracks.length * ROW_H" in TABLE
    assert "const Row = memo(" in TABLE
    css = _read("styles.css")
    row = css[css.index("\n.tt-row {"):]
    assert "height: 32px" in row[:row.index("}")]


def test_the_view_lives_on_the_server():
    """Columns, order, widths and sort go to /api/prefs/library-view; the local
    copy is only a cache."""
    view = _read("hooks/useColumnWidths.js")
    assert "api.getLibraryView()" in view and "api.putLibraryView(" in view
    assert "export function useLibraryView" in view


def test_the_search_box_parses_filters():
    """'8A 124-128 house 2015+ vox' reads as a key, a BPM range, a genre this
    library has, a year and vocal-forward; ⏎ pins them."""
    fn = HOOK[HOOK.index("export function parseQuery"):]
    fn = fn[:fn.index("\n}")]
    for token in ("([ab])(~?)$/i", "(\\d{2,3})[-–](\\d{2,3})$", "bpm$/i", "\\+$/",
                  '"vox"', 'lo === "stems"'):
        assert token in fn, token
    assert "onPin()" in BAR and 'e.key === "Enter"' in BAR


def test_presets_are_saved_on_the_server():
    presets = _read("hooks/useFilterPresets.js")
    assert "api.saveFilterPreset(" in presets and "api.deleteFilterPreset(" in presets
    assert 'label: "Undo"' in presets           # a delete can be undone
    for banned in ("fetch(", "api."):
        assert banned not in BAR


def test_stored_widths_survive_a_column_change():
    hook = _read("hooks/useColumnWidths.js")
    assert "localStorage" in hook
    # Unknown ids ignored, missing ones fall back — neither may throw.
    assert "try {" in hook and "catch" in hook
    assert "Number.isFinite" in hook, "a half-written value must not collapse a column"


# ── the class filter is gone ─────────────────────────────────────────────────

def test_no_class_filter():
    """Every row in the library is a whole track, and playback already toggles
    vocal/bed — the chip filtered on a dominant section class that answered a
    question the screen does not ask."""
    for src in (HOOK, BAR):
        assert "track_class" not in src
        assert "CLASSES" not in src
        assert "cls:" not in src


def test_year_length_and_plays_are_hidden_by_default():
    """The design's full column set needs 1270px; beside the 400px dock that
    scrolled the table at 1440. Year, Length and Plays start hidden (Columns ▾
    brings them back, and a stored view keeps whatever the user chose)."""
    cols = TABLE[TABLE.index("export const COLUMNS = ["):]
    cols = cols[:cols.index("];")]
    hidden = {l.split('"')[1] for l in cols.split("\n")
              if l.strip().startswith("{ id:") and "defaultVisible: false" in l}
    assert {"year", "len", "plays"} <= hidden
    assert not hidden & {"title", "bpm", "key", "energy", "genre", "vox", "best", "pairs", "rating", "status"}
