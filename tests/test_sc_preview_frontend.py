"""Discover previews a track without leaving the app — and without spending the
client_id. Pinned from Python, like the rest of this repo's frontend contracts.

Discover's ▶ used to be an <a target="_blank">, because "this app never streams
external audio". Auditioning something you were considering importing meant a
new tab and losing your place in the list.

It plays here now, and HOW it plays is the load-bearing part. Measured against a
live search, only 3 of 10 results expose a `progressive` mp3; the rest are HLS
and most also carry DRM-encrypted variants. Resolving those ourselves would mean
hls.js AND one extra api-v2 request per play against the scraped client_id that
the FROZEN mixes resolver shares — the one rate limit this codebase refuses to
spend. SoundCloud's own widget costs zero api-v2 requests and plays everything.

So the rule this file exists to enforce: **nothing under frontend/src may ever
learn about api-v2, client_id or transcodings.** That is the door this feature
must not open.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"
WIDGET = (SRC / "hooks" / "useScWidget.js").read_text(encoding="utf-8")
ROWS = (SRC / "components" / "ScRows.jsx").read_text(encoding="utf-8")
PANES = ("components/SoundCloudBrowser.jsx", "components/Suggestions.jsx")
CSS = (SRC / "styles.css").read_text(encoding="utf-8")


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


_LINE_COMMENT = re.compile(r"^\s*//.*$", re.M)
_JSX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.S)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _strip(src: str) -> str:
    """Comments out. Several of these files EXPLAIN the client_id constraint at
    length, and a test hunting for the string would match the paragraph saying
    why we do not do it."""
    src = _JSX_COMMENT.sub("", src)
    src = _BLOCK_COMMENT.sub("", src)
    return _LINE_COMMENT.sub("", src)


def test_the_browser_never_touches_the_scraped_api():
    """The whole reason the widget was chosen over resolving a stream."""
    for path in list(SRC.rglob("*.js")) + list(SRC.rglob("*.jsx")):
        src = _strip(path.read_text(encoding="utf-8"))
        for banned in ("api-v2", "client_id", "transcodings"):
            assert banned not in src, f"{path.name} mentions {banned}"


def test_the_widget_api_is_loaded_lazily():
    """At module scope it would be a network request on every cold start, in an
    app that otherwise runs entirely against localhost."""
    assert "w.soundcloud.com/player/api.js" in WIDGET
    # The <script> is only appended inside the loader, not at import time.
    top = WIDGET[:WIDGET.index("function loadApi()")]
    assert "createElement" not in top
    assert "appendChild" not in top
    assert "let scriptPromise = null;" in WIDGET


def test_a_track_that_cannot_be_embedded_keeps_the_old_link():
    """`embeddable_by` is 'me' on a real share of results. Answering that before
    the click is the difference between a link and a button that fails."""
    assert 'row.embeddable !== false' in ROWS
    assert "permalink_url" in ROWS
    assert 'target="_blank"' in ROWS
    assert "embeddable" in _read("hooks/usePlayer.js")


def test_the_play_button_is_a_button_and_still_stops_propagation():
    """The row's own click shortlists the track. Without stopPropagation, ▶
    would play it AND silently add it to the shortlist."""
    block = ROWS[ROWS.index('className="sc-open"'):]
    block = block[:block.index("<TrackArt")]
    assert "stopPropagation" in block
    assert "onPlay(row)" in block


def test_both_panes_pass_playback_down():
    for rel in PANES:
        src = _read(rel)
        assert "onPlay={playRow}" in src, rel
        # The source object is built once, in ScRows, and shared. Both panes held
        # a byte-identical copy of it plus a byte-identical "is this row the one"
        # test - and the copy compared track_id.
        assert "scSource(row)" in src, rel
        assert "scRowState(player, row)" in src, rel
        assert 'kind: "sc"' not in _strip(src), rel


def test_row_identity_is_the_players_own_key():
    """`player.source.trackId === row.track_id` is `undefined === undefined` for
    a row SoundCloud returned without an id, so every id-less row claimed to be
    playing whenever any id-less row was - and `toggle` keys those same rows by
    permalink, so the highlight and the pause/resume identity disagreed on
    exactly the rows that break."""
    assert "sourceKey(scSource(row)) === player.key" in ROWS
    assert "player.source.trackId" not in _strip(ROWS)
    for rel in PANES:
        assert "player.source.trackId" not in _strip(_read(rel)), rel


def test_a_row_says_both_loaded_and_sounding():
    """A paused row used to render a plain play glyph with no hint that it was
    the bar's source at all, so you could start a preview and then lose which of
    fifty rows it came from."""
    assert '" current"' in ROWS
    assert '" playing"' in ROWS
    for cls in (".sc-row.current", ".sc-row.playing"):
        assert cls in CSS, cls


# -- the widget has to report itself ------------------------------------------

def test_a_fresh_iframe_per_track():
    """SC's Widget(frame) hands back the SAME wrapper for an element it has
    already seen. Re-pointing frame.src therefore left the PREVIOUS track's
    handlers registered, bound against the previous token - whose guard then
    discarded every PLAY_PROGRESS and PAUSE. The first row worked perfectly and
    every row after it showed a frozen clock and a dead pause button while the
    audio really was playing."""
    code = _strip(WIDGET)
    assert "frame.remove()" in code
    play = code[code.index("async play(permalink)"):]
    play = play[:play.index("async pause()")]
    assert "unmount()" in play
    assert "mount(" in play


def test_transport_commands_survive_being_early():
    """The iframe carries auto_play=true and the bar appears on click, so there
    is a window in which audio is already sounding while `widget` is still null.
    Every control was a silent no-op through all of it."""
    code = _strip(WIDGET)
    for cmd in ("async pause()", "async resume()", "async seek(secs)"):
        assert cmd in code, cmd
    assert "await live()" in code


def test_position_is_polled_not_only_pushed():
    """PLAY_PROGRESS is push-only and stops entirely when paused or after a
    paused seek, which is "skipping around works but the bar does not track
    it"."""
    code = _strip(WIDGET)
    assert "setInterval" in code
    assert "getPosition" in code
    # The widget, not an inferred flag, is the authority on being paused.
    assert "isPaused" in code


def test_the_watchdog_still_waits_for_the_position_to_move():
    """Unchanged contract: SoundCloud emits PLAY and then silence on part of the
    major-label catalogue, with no ERROR. A rising position is the only evidence
    of audio - and now the event and the poll feed the same single test."""
    code = _strip(WIDGET)
    advanced = code[code.index("const advanced = "):]
    advanced = advanced[:advanced.index("};")]
    assert "disarmWatchdog()" in advanced
    assert "secs > 0" in advanced


def test_the_footer_no_longer_says_play_opens_soundcloud():
    src = _read("components/SoundCloudBrowser.jsx")
    assert "open on SoundCloud</span>" not in src
    # The import promise is still true — a preview streams, it does not download.
    assert "nothing downloads until you import" in src


# ── sorting ──────────────────────────────────────────────────────────────────

def test_every_discover_column_with_a_value_is_sortable():
    """The headers were inert <div>s while a whole sort engine sat one file
    away, reachable only from a dropdown."""
    heads = ROWS[ROWS.index("const SC_HEADS = ["):]
    heads = heads[:heads.index("];")]
    for label, key in (("TITLE", "title"), ("UPLOADER", "artist"),
                       ("GENRE", "genre"), ("PLAYS", "plays"),
                       ("YEAR", "year"), ("LIKES", "likes"),
                       ("TIME", "duration")):
        assert f'"{label}"' in heads, label
        assert f'"{key}"' in heads, key
    assert "ScHeader({ filters, onChange })" in ROWS

    sorts = _read("hooks/useResultFilters.js")
    block = sorts[sorts.index("  track: {"):]
    block = block[:block.index("  },")]
    for key in ("title", "artist", "genre", "plays", "likes", "duration", "year"):
        assert f"{key}:" in block, key


def test_the_year_column_sorts_by_the_number_it_displays():
    """The YEAR cell shows release_year, falling back to the upload year. The
    only pre-existing date key was `upload`, which is the raw upload date — so
    sorting on it would have ordered the rows by a value the column is not
    showing. One accessor, imported by both."""
    theme = _read("theme.js")
    assert "export const yearOf" in theme
    assert ROWS.count("const yearOf") == 0
    assert "yearOf" in ROWS                      # imported, not redefined
    assert "yearOf" in _read("hooks/useResultFilters.js")


def test_the_library_table_sorts_the_same_way():
    table = _read("components/TrackTable.jsx")
    assert "<SortHead" in table
    for key in ("title", "artist", "genre", "year", "plays", "bpm", "key",
                "rating", "duration"):
        assert f'"{key}"' in table, key
    # PIPE is four assembled booleans, not a value — it stays inert.
    heads = table[table.index("const HEADS = ["):]
    heads = heads[:heads.index("];")]
    pipe = [ln for ln in heads.splitlines() if "PIPE" in ln][0]
    assert "key:" not in pipe


def test_a_third_click_returns_to_the_natural_order():
    """Unsorted is a real choice in both tables — SoundCloud's relevance order
    in one, import order in the other. A two-state toggle would make either
    unreachable without hunting for the dropdown."""
    head = _read("components/SortHead.jsx")
    assert 'return { sort: "", dir: "asc" };' in head
    # Numbers open descending: "most plays first" is what clicking PLAYS means.
    assert 'const first = numeric ? "desc" : "asc";' in head


def test_sorting_still_fetches_nothing():
    """The bar says "showing 12 of 47 loaded" and means it. A header that
    quietly paged the API to make its sort look global would spend the rate
    limit the frozen resolver shares."""
    head = _read("components/SortHead.jsx")
    for banned in ("fetch(", "api.", "await "):
        assert banned not in head, banned
    assert not re.search(r"\bfetch\s*\(", ROWS.split("export function TrackRow")[0])
