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


def test_blocks_dividers_loop_and_playhead_share_one_accessor():
    """If any one of the four is positioned by something else, it is back to two
    axes and nothing on screen looks broken — it just quietly disagrees."""
    # The section blocks and the dividers both go through `span()`...
    assert "const span = (s) => ({" in CODE
    assert CODE.count("style={span(s)}") == 2, "labels and dividers"
    # ...which is built from the same pct() the loop and the playhead use.
    span = CODE[CODE.index("const span = (s) => ({"):]
    span = span[:span.index("});")]
    assert "pct(s.start_sec)" in span
    assert "pct(s.end_sec)" in span
    for el in ("struct-loop", "struct-playhead"):
        block = CODE[CODE.index(el):]
        assert "pct(" in block[:260], el


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


def test_the_flex_gap_that_stole_width_is_gone():
    """`flex: n` is `flex: n 1 0%`, so the gaps and the divider borders came out
    of the distributable space before the proportional split while the
    percentage-positioned playhead paid nothing: ~3.3% of the width on a 700px
    strip with 12 sections, worst at the end of the track."""
    labels = _css_block(".struct-labels {")
    assert "gap:" not in labels
    assert "position: relative" in labels
    dividers = _css_block(".struct-dividers {")
    assert "gap:" not in dividers
    assert "display: flex" not in dividers
    # Absolutely positioned children, and the hairline no longer takes up space.
    child = _css_block(".struct-dividers > div {")
    assert "position: absolute" in child
    assert "border-right" not in child
    assert "position: absolute" in _css_block(".struct-label {")


def test_the_bar_counts_are_not_lost():
    """They were the reason the old axis existed, so they have to stay readable
    somewhere."""
    assert "bars" in CODE[CODE.index("const tip = (s) =>"):][:400]
    assert "bars unmeasured" in CODE          # the header total


# ── the playhead is a control (1c) ───────────────────────────────────────────

def test_the_wave_can_be_clicked_and_dragged():
    """The red line was a readout with nothing listening to it: there was no
    pointer handler anywhere on the strip."""
    assert "onPointerDown" in CODE
    assert "onPointerMove" in CODE
    assert "onSeek" in CODE
    wave = _css_block(".struct-wave {")
    assert "cursor: pointer" in wave
    # A drag must not scroll the page out from under the playhead.
    assert "touch-action: none" in wave
    # A 1.5px line is not a grab target.
    assert ".struct-grab" in CSS
    # The dividers sit over the wave and would swallow the pointer.
    assert "pointer-events: none" in _css_block(".struct-dividers {")


def test_clicking_an_idle_track_starts_it_there():
    """Anything else makes a click on a waveform a dead end."""
    seek = DETAIL[DETAIL.index("const seekTo = (secs) =>"):]
    seek = seek[:seek.index("\n  };")]
    assert "player.seek(secs)" in seek
    assert "startAt: secs" in seek


# ── which stem you are hearing (1e) ──────────────────────────────────────────

def test_the_selected_stem_is_the_prominent_envelope():
    """And the other one stays VISIBLE. The whole value of the overlay is seeing
    where the vocal sits against the bed; hiding one throws that away."""
    fills = CODE[CODE.index("const WAVE_FILL = {"):]
    fills = fills[:fills.index("};")]
    for stem in ("full", "vocals", "instrumental"):
        assert stem in fills, stem
    vals = [float(v) for v in re.findall(r"0\.\d+", fills)]
    assert all(v > 0 for v in vals), "a stem at zero opacity is a hidden stem"
    # vocals emphasises vox over bed; instrumental the other way round.
    vocals = fills[fills.index("vocals:"):fills.index("instrumental:")]
    bed, vox = (float(re.search(rf"{k}: (0\.\d+)", vocals).group(1))
                for k in ("bed", "vox"))
    assert vox > bed
    assert "stem={stem}" in DETAIL


def test_the_detail_screen_and_the_bar_cannot_disagree_about_the_stem():
    """Two controls for one fact (1b): the hero segment switches the live audio,
    and mirrors the bar's own switch back into local state."""
    assert "const pickStem = (id) => {" in DETAIL
    assert "player.switchStem(id)" in DETAIL
    assert "if (onThisTrack && src.stem && src.stem !== stem) setStem(src.stem);" in DETAIL
