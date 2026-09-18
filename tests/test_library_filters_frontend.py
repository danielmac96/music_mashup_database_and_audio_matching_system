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
    assert "visible.map(" in screen
    assert "tracks.map(" not in screen


def test_unsorted_is_the_default():
    """Rows arrive in import order, which is the order you added them, and that
    is often exactly what you are looking for."""
    assert re.search(r'primary:\s*""', HOOK)
    assert HOOK.index('["", "unsorted"]') < HOOK.index('["added"')


def test_missing_values_sort_last_in_both_directions():
    """A track with no play count is unknown, not unpopular. The direction sign
    is applied only after the null checks, so nulls sink either way."""
    fn = HOOK[HOOK.index("function compare("):]
    fn = fn[:fn.index("\n// Unsorted")]
    assert fn.count("if (x == null) return 1;") >= 2
    assert fn.count("if (y == null) return -1;") >= 2


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
    bar = css[css.index(".filter-bar {"):]
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

def test_the_head_and_the_rows_share_one_template():
    assert TABLE.count("gridTemplateColumns: template") == 1
    assert "gridTemplateColumns: cols" in TABLE
    assert "<TrackRow key={t.id} t={t} cols={template}" in TABLE


def test_every_column_declares_an_id_and_a_width():
    """Widths are stored per id. An index would silently re-point every stored
    width the first time a column moved, and the symptom — one narrow column
    somewhere else — looks nothing like the cause."""
    heads = TABLE[TABLE.index("const HEADS = ["):]
    heads = heads[:heads.index("];")]
    entries = [l for l in heads.split("\n") if l.strip().startswith("{ id:")]
    assert len(entries) == 11, entries
    for line in entries:
        assert " w: " in line, line
    # Each entry stays on ONE line: test_sc_preview_frontend parses this block
    # line-by-line to assert PIPE carries no sort key.
    assert "\n" not in "".join(e for e in entries if e.count("{") != e.count("}"))


def test_title_and_genre_both_flex():
    """Title used to be the only 1fr, so it absorbed every pixel of slack on a
    wide desktop while GENRE sat at its floor and truncated the one thing it
    exists to show."""
    heads = TABLE[TABLE.index("const HEADS = ["):]
    heads = heads[:heads.index("];")]
    name = next(l for l in heads.split("\n") if '{ id: "name"' in l)
    genre = next(l for l in heads.split("\n") if '{ id: "genre"' in l)
    assert "fr)" in name and "minmax(" in name
    assert "fr)" in genre and "minmax(" in genre


def test_a_grip_drag_never_reaches_the_sort_header():
    grip = TABLE[TABLE.index("const onGrab"):]
    grip = grip[:grip.index("return (")]
    assert "e.stopPropagation()" in grip
    assert "getBoundingClientRect" in grip, "a fr column has no stored width to read"
    assert 'className="tt-grip"' in TABLE
    assert "resetColumn" in TABLE, "double-click should restore the default"


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
