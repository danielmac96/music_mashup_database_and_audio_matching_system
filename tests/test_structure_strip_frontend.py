"""The structure strip draws everything on ONE axis — pinned from Python.

The reported bug: "the colored bars for sections of the song should align with
the waveform sections they are part of, most songs this becomes misaligned
throughout the song so the bar start and end does not align with the loop
highlighted section".

The cause was not a tuning problem. The section blocks and the dividers were
flex children sized by `flex: bar_count`, while the waveform, the loop window
and the playhead were positioned as a percentage of TIME. Two axes in one box
cannot be made to line up, and the error grows left to right — which is exactly
"misaligned throughout the song". The loop highlight did not land on the section
you had just clicked to create it.

Bars were a deliberate choice (the strip's own header argued for them: "what can
I loop over what" is a question about bars) and the information is kept — on each
block's tooltip, and in the header's total. But a block has to sit over the audio
it describes.

Two second-order contributors are asserted here too, because both are invisible
and both come back the moment someone re-adds a flex gap:

  * `flex: n` expands to `flex: n 1 0%`, so the 1px gaps between blocks and the
    1px divider borders were subtracted from the distributable space BEFORE the
    proportional split, while the percentage-positioned playhead paid nothing.
  * the axis LENGTH was `songs.duration_secs` — yt-dlp container metadata — while
    the section times and the stem envelopes are both measured on the decoded
    full mix, which is also the timebase `<audio>` reports as currentTime.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"
STRIP = (SRC / "components" / "StructureStrip.jsx").read_text(encoding="utf-8")
DETAIL = (SRC / "components" / "TrackDetail.jsx").read_text(encoding="utf-8")
CSS = (SRC / "styles.css").read_text(encoding="utf-8")

_LINE_COMMENT = re.compile(r"^\s*//.*$", re.M)
_JSX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.S)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _strip(src: str) -> str:
    """Comments out. This component EXPLAINS the bars axis it replaced at
    length, so a test looking for "flex" would match the paragraph saying there
    is no longer one."""
    src = _JSX_COMMENT.sub("", src)
    src = _BLOCK_COMMENT.sub("", src)
    return _LINE_COMMENT.sub("", src)


CODE = _strip(STRIP)


def _css_block(selector: str) -> str:
    i = CSS.index(selector)
    return CSS[i:CSS.index("}", i)]


# ── one axis ─────────────────────────────────────────────────────────────────

def test_nothing_in_the_strip_is_weighted_by_bars():
    """`weightOf` returned a bar count — and, for a section analysed before P2.1
    that has none, `seconds / 2`. Two incompatible units inside one flex
    container, on top of being the wrong axis entirely."""
    assert "weightOf" not in CODE
    assert "flex:" not in CODE
    # bar_count still feeds the header total and each block's tooltip, which is
    # where that information belongs. What it may never do again is decide
    # GEOMETRY, so it must not appear on a line that sizes or places anything.
    for line in CODE.splitlines():
        if "bar_count" in line:
            for geom in ("style", "flex", "width", "left", "pct("):
                assert geom not in line, line.strip()


def test_blocks_ruler_loop_and_playhead_share_one_axis():
    """Every x on the canvas — downbeats, section blocks, key band, loop window,
    playhead — goes through the one time accessor X(t)."""
    assert "const X = (t) => (t / total) * W;" in CODE
    for use in ("const x = X(t);", "X(s.start_sec), x1 = X(s.end_sec)",
                "X(loop.start), x1 = X(loop.end)", "X(position)"):
        assert use in CODE, use


def test_the_axis_length_is_the_decoded_timebase_not_container_metadata():
    """Section times come from librosa's decode of the full mix (structure.py
    forces the last section's end to that decode's duration) and the stem
    envelopes are measured on stems of the same decode. `songs.duration_secs` is
    yt-dlp container metadata — the one number here on a different clock. It
    stays as the fallback for a track whose structure was never detected."""
    total = CODE[CODE.index("const total ="):]
    total = total[:total.index(";")]
    # sections first, duration second.
    assert total.index("sections[sections.length - 1].end_sec") < total.index("duration")


def test_the_strip_is_one_156px_canvas():
    """The design's strip: one canvas, ruler / blocks / waveform / envelopes /
    key band in fixed rows."""
    assert "const H = 156;" in CODE
    assert '<canvas ref={canvasRef} className="struct-canvas"' in CODE
    block = CSS[CSS.index("\n.struct-canvas {"):]
    assert "height: 156px" in block[:block.index("}")]


def test_the_bar_counts_are_not_lost():
    """Bars are numbered on the ruler (every phrase start) and each block's
    tooltip counts its own."""
    assert "c.fillText(String(i + 1)" in CODE
    assert "Math.round(hovered.bar_count)} bars" in CODE


# ── the playhead is a control (1c) ───────────────────────────────────────────

def test_the_wave_can_be_clicked_and_dragged():
    """On the section blocks a click loops that section; anywhere else it seeks
    and a drag keeps seeking."""
    assert "onPointerDown" in CODE and "onPointerMove" in CODE and "onSeek" in CODE
    down = CODE[CODE.index("const onPointerDown"):]
    down = down[:down.index("};")]
    assert "y >= BLOCKS[0] && y <= BLOCKS[1]" in down and "onPickSection(s)" in down
    assert "seekAt(e.clientX)" in down
    assert "if (e.buttons === 1) seekAt(e.clientX);" in CODE
    block = CSS[CSS.index("\n.struct-canvas {"):]
    block = block[:block.index("}")]
    assert "cursor: pointer" in block and "touch-action: none" in block


def test_clicking_an_idle_track_starts_it_there():
    """Anything else makes a click on a waveform a dead end."""
    seek = DETAIL[DETAIL.index("const seekTo = (secs) =>"):]
    seek = seek[:seek.index("\n  };")]
    assert "player.seek(secs)" in seek
    assert "startAt: secs" in seek


# ── which stem you are hearing (1e) ──────────────────────────────────────────

def test_the_selected_stem_is_the_prominent_envelope():
    """Which waveform reads as the one you hear follows the selected stem; the
    other stays visible, dimmed, never at zero."""
    block = CODE[CODE.index("const WAVE_ALPHA = {"):]
    block = block[:block.index("};")]
    for stem in ("full", "vocals", "instrumental"):
        assert stem in block, stem
    vals = [float(v) for v in re.findall(r"(?:bed|vox): ([0-9.]+)", block)]
    assert all(v > 0 for v in vals), "a stem at zero opacity is a hidden stem"
    vocals = re.search(r"vocals: \{ bed: ([0-9.]+), vox: ([0-9.]+) \}", block)
    assert float(vocals.group(2)) > float(vocals.group(1))
    assert "stem={stem}" in DETAIL


def test_the_detail_screen_and_the_bar_cannot_disagree_about_the_stem():
    """Two controls for one fact (1b): the hero segment switches the live audio,
    and mirrors the bar's own switch back into local state."""
    assert "const pickStem = (id) => {" in DETAIL
    assert "player.switchStem(id)" in DETAIL
    assert "if (onThisTrack && src.stem && src.stem !== stem) setStem(src.stem);" in DETAIL
