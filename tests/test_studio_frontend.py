"""Studio (revamp step 7): contracts read from the source."""
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


STUDIO = _read("components/MixStudio.jsx")
PANEL = _read("components/StudioPanel.jsx")
CSS = _read("styles.css")
ENGINE = _read("engine/MashupEngine.js")


def test_a_trimmed_or_extended_clip_renders_as_it_plays():
    """The engine plays clipStart/clipEnd, fades and filters; the mixdown and
    the FL session are sent the same trim, fades and filters."""
    for k in ("clipStartSec", "clipEndSec", "fadeInSec", "hpHz"):
        assert k in ENGINE, k
    clip = STUDIO[STUDIO.index("const clipFor = (l) =>"):][:1600]
    assert "...(trimmed ? { clip_start: cs, clip_end: ce } : {})" in clip
    assert "offset_sec: trimmed ? l.offsetSec + cs / (l.rate || 1) : l.offsetSec" in clip
    lanes = STUDIO[STUDIO.index("const sessionLanes = () =>"):][:1400]
    for k in ("clip_start: cs, clip_end: ce", "fade_in:", "hp_hz:", "offset_ms:"):
        assert k in lanes, k
    assert "armed, sessionLanes())" in STUDIO
    assert "...(lanes?.length ? { lanes } : {})" in _read("api.js")


def test_render_menu_has_mixdown_stems_and_session():
    for item in ("<b>WAV mixdown</b>", "<b>Stems per lane</b>", "<b>FL session</b>"):
        assert item in STUDIO, item
    assert "api.startMixdown([clipFor(l)])" in STUDIO


def test_edges_trim_or_extend_and_alt_moves_freely():
    assert "if (trimmed || selected)" in STUDIO, "the whole-song ghost shows on the selected lane"
    assert "alt || me.altKey ? { offset: raw, snapped: false }" in STUDIO
    assert "if (alt && !moved) claimDownbeat(" in STUDIO, "an alt-click still sets bar 1"
    for k in ('onExtend(lane, edge)', '"◂ +1 bar"', '"+1 bar ▸"', "⟷ section"):
        assert k.strip('"') in PANEL, k


def test_the_adjustments_panel():
    assert 'const PANEL_KEY = "mashup.studio.panel.v1"' in STUDIO
    assert "PANEL_MIN = 120, TIMELINE_MIN = 96" in STUDIO
    assert "onDoubleClick={onFit}" in PANEL
    for label in ('label="Tempo"', 'label="Pitch"', 'label="Offset"', 'label="Gain"'):
        assert label in PANEL, label
    assert "↺ all to suggestion" in PANEL and "Next pair ⏎" in PANEL
    assert ".sp-card" in CSS and "width: 370px" in CSS[CSS.index(".sp-card {"):][:200]
    # Per-lane suggestions come from the plan's lanes block (B5).
    assert "pairPlan?.lanes?.vocal?.gain" in STUDIO and "pairPlan?.lanes?.bed?.gain" in STUDIO


def test_the_keys():
    keys = STUDIO[STUDIO.index("// ── keyboard"):][:3500]
    for k in ('e.code === "Space"', "e.shiftKey ? beat * 4 : beat", 'e.key === "m"',
              'e.key === "s"', 'e.key === "Enter" && onNextPair', 'e.key === "Delete"'):
        assert k in keys, k
    assert '["Studio", "space play · ←/→ nudge the selected lane' in _read("components/HelpPanel.jsx")


def test_clear_offers_undo_and_lanes_are_64px():
    assert 'label: "Undo"' in STUDIO[STUDIO.index("const clearWithUndo"):][:600]
    head = CSS[CSS.index(".studio-lanehead {"):][:200]
    lane = CSS[CSS.index(".studio-lane {"):][:200]
    assert "height: 64px" in head and "height: 64px" in lane


def test_no_hex():
    for src in (STUDIO, PANEL):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", src)
