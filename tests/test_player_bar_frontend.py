"""One player, one bar — pinned from Python.

This repo has no JS test runner and adding one is a separate decision, so the
invariants that would hurt if they drifted are asserted by reading the source,
the same way tests/test_shell_frontend.py does.

The claim being defended: **exactly one thing in this app makes a sound, and the
bar at the bottom is showing it.** Before, four players ran side by side — a
hidden <audio> in LibraryScreen, a second one in TrackDetail, a MashupEngine in
the pair dock and a THIRD AudioContext in Discover's mashup pane — and the bar
was wired only to the dock. So playing a song from a library row had no
play/pause, no scrub and no time readout, and navigating away unmounted the
element and killed it silently.

Every assertion here is one of those failures. They are all silent: the app
builds, the tests pass, and the audio is simply out of reach.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


_LINE_COMMENT = re.compile(r"^\s*//.*$", re.M)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _code(rel: str) -> str:
    """Source with the comments stripped.

    These files EXPLAIN the players they replaced, so a test looking for the
    string "<audio" would match the paragraph saying there is no longer one.
    Assert against what runs, not against what it says about itself.
    """
    src = _BLOCK_COMMENT.sub("", _read(rel))
    return _LINE_COMMENT.sub("", src)


APP = _read("App.jsx")
PLAYER = _read("hooks/usePlayer.js")
BAR = _read("components/PlayerBar.jsx")
CSS = _read("styles.css")

# Every screen that used to own audio of its own.
FORMER_OWNERS = (
    "components/LibraryScreen.jsx",
    "components/TrackDetail.jsx",
)


# ── the element cannot belong to a screen ────────────────────────────────────

def test_no_screen_renders_its_own_audio_element():
    """The <audio> was rendered by the screen that started it, so leaving the
    screen unmounted the element and stopped the song with no way to tell."""
    for rel in FORMER_OWNERS:
        src = _code(rel)
        assert "<audio" not in src, rel
        assert "new Audio(" not in src, rel


def test_the_element_is_a_ref_not_jsx():
    """usePlayer holds it imperatively for the same reason: the bar returns null
    when idle, and an element rendered by the bar would die with it."""
    assert "new Audio(" in PLAYER
    assert "<audio" not in _code("hooks/usePlayer.js")


def test_only_the_player_builds_the_engine():
    """useHookAudition constructs a MashupEngine and an AudioContext. Two call
    sites meant two engines that could play over each other; Studio keeps its
    own because it is an arranger, not a preview."""
    callers = []
    for path in sorted(SRC.rglob("*.js")) + sorted(SRC.rglob("*.jsx")):
        if path.name in ("useHookAudition.js",):
            continue
        rel = path.relative_to(SRC).as_posix()
        if "useHookAudition(" in _code(rel):
            callers.append(path.name)
    assert callers == ["usePlayer.js"], callers


def test_the_dock_borrows_the_shared_player():
    """The dock used to build a MashupEngine of its own, which is how the app
    ended up with several players and one bar that belonged to none of them."""
    src = _code("hooks/usePairDock.js")
    assert "useHookAudition" not in src
    assert "player." in src


def test_starting_one_source_silences_the_others():
    """Mutual exclusion is the point of the hook. Without it the element, the
    widget and the engine can all sound at once — which they could."""
    block = PLAYER[PLAYER.index("const silence = useCallback("):]
    block = block[:block.index("}, [pair]);")]
    for backend in ('keep !== "element"', 'keep !== "sc"', 'keep !== "pair"'):
        assert backend in block, backend


# ── the bar is the app's, not a route's ──────────────────────────────────────

def test_the_player_bar_is_not_inside_a_route():
    """setRoute clears the header status and the rail slot. The player is the
    one thing that must survive navigation, so it is mounted after the route
    switch rather than inside it."""
    lib = APP[APP.index('{route === "library" && ('):]
    lib = lib[:lib.index('{route === "track" && (')]
    assert "<PlayerBar" not in lib

    # It IS mounted, once, at the end of .app-main.
    assert APP.count("<PlayerBar") == 1
    assert 'route !== "studio" && (' in APP


def test_studio_stops_the_global_player():
    """Studio drives an engine of its own and its timeline IS its transport.
    Two of them would fight over the space bar and over the speakers."""
    assert 'if (route === "studio") player.stop();' in APP


def test_the_strip_is_a_control_not_a_readout():
    """It had no handler at all — which is literally "I cannot scrub through the
    song or pause or play"."""
    assert "onPointerDown" in BAR
    assert "player.seek(" in BAR
    assert 'role="slider"' in BAR
    # ...and the CSS has to let a pointer land on it.
    strip = CSS[CSS.index(".tr-strip {"):]
    assert "cursor: pointer" in strip[:strip.index("}")]


def test_the_bar_shows_every_kind_of_source():
    for kind in ("track", "pair", "sc"):
        assert f'"{kind}"' in PLAYER, kind


# -- a section is a window, not a file of its own -----------------------------

def test_the_clip_backend_is_gone():
    """A section used to be served as a WAV holding ONLY that section, which is
    a file that cannot be scrubbed anywhere else in the record - the whole of
    "clicking the bar just resets the loop". It is a loop window on the real file
    now, so no screen may reintroduce the fourth backend."""
    assert 'kind === "clip"' not in _code("hooks/usePlayer.js")
    for path in sorted(SRC.rglob("*.js")) + sorted(SRC.rglob("*.jsx")):
        rel = path.relative_to(SRC).as_posix()
        assert 'kind: "clip"' not in _code(rel), rel
    # ...and the amber tint follows the loop rather than the departed kind.
    assert ".transport.k-clip" not in CSS
    assert ".transport.is-looping" in CSS


def test_the_native_loop_is_never_used_for_a_window():
    """el.loop loops the whole FILE. A window inside one has to be wrapped by
    hand, or a looped section plays on to the end of the record."""
    assert "el.loop = false" in PLAYER
    assert "el.loop = true" not in PLAYER
    # timeupdate fires ~4x a second, i.e. up to 250ms of overshoot past the loop.
    assert "requestAnimationFrame" in PLAYER


def test_the_strip_and_the_seek_agree_on_one_unit():
    """The bar subtracted `source.start` to display and then handed a
    span-relative second back to seek(), which subtracted the start AGAIN - so
    clicking anywhere on a looping section landed at the section start. Both
    sides are absolute song seconds now."""
    assert "source.start" not in _code("components/PlayerBar.jsx")
    assert "source.start" not in _code("hooks/usePlayer.js")
    seek = PLAYER[PLAYER.index("const seek = useCallback("):]
    seek = seek[:seek.index("}, [kind, pair")]
    # Seeking out of an armed loop releases it rather than snapping back, which
    # would read as the drag having failed.
    assert "loop: null" in seek


def test_the_stem_is_a_live_control_not_an_identity():
    """Switching Full/Vox/Bed keeps the clock, the loop and whether it was
    playing. If the stem were in the key, every row highlight in the app would
    drop the moment you switched layer."""
    assert "switchStem" in PLAYER
    assert "switchStem" in BAR
    key = PLAYER[PLAYER.index("export function sourceKey("):]
    key = key[:key.index("\n}")]
    track = key[key.index('s.kind === "track"'):key.index('s.kind === "sc"')]
    assert "s.stem" not in track
    assert "s.loop" in track


def test_the_bar_marks_the_loop_it_is_inside():
    """The strip spans the whole record while a section loops - otherwise there
    is nowhere to scrub TO - so it has to draw where the loop is."""
    assert "tr-loop" in BAR
    assert ".tr-loop {" in CSS


def test_pair_only_controls_are_gated():
    """Stem solo, the star and Open in Studio are meaningless for a single song,
    and the bar used to render nothing BUT those."""
    assert "isPair && (" in BAR
    assert "<StarRating" in BAR


def test_the_dead_transport_rules_are_gone():
    """`.transport .play-btn` was an audition-era leftover carrying MixStudio's
    green. Nothing rendered it, and it would have silently ambushed any
    .play-btn this bar added."""
    assert ".transport .play-btn" not in CSS
    assert ".transport .loop-btn" not in CSS


def test_the_bar_emits_classes_that_exist():
    """The reason the previous PlayerBar was deleted: it emitted .player-* while
    the stylesheet only ever defined .pb-*."""
    for cls in re.findall(r'className="([a-z][a-z0-9 -]*)"', BAR):
        for name in cls.split():
            assert f".{name}" in CSS, name
