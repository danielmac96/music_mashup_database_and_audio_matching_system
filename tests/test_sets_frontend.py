"""Sets screen (revamp step 6): contracts read from the source."""
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


SCREEN = _read("components/SetScreen.jsx")
API = _read("api.js")
CSS = _read("styles.css")


def test_every_slot_kind_can_be_added_after_the_selection():
    for kind in ('kind: "pair"', 'kind: "track"', 'kind: "break"', 'kind: "gap"'):
        assert kind in SCREEN, kind
    assert "const after = selIdx >= 0 ? selIdx + 1 : items.length;" in SCREEN
    assert "api.addSetSlot(" in SCREEN and '`/api/sets/${id}/items`' in API


def test_the_fit_dot_reads_the_servers_verdict():
    # The screen draws what matcher/setflow decided; it does not re-derive it.
    assert "FIT[t.fit]" in SCREEN
    assert "ssKeyStep" not in SCREEN and "compatibleKeys" not in SCREEN
    for cls in (".ss-dot.ok", ".ss-dot.warn", ".ss-dot.bad"):
        assert cls in CSS, cls


def test_slot_width_is_its_duration():
    assert "width: Math.max(MIN_SLOT_PX, layout.w[k] * pps)" in SCREEN
    assert "f.landings[k]?.secs" in SCREEN


def test_energy_curve_is_drag_points_saved_to_the_set():
    assert "energy_plan: pts" in SCREEN
    assert "e.shiftKey" in SCREEN and "onDoubleClick={onDbl}" in SCREEN
    for name in ("Warm-up", "Peak time", "Journey"):
        assert f'["{name}"' in SCREEN, name


def test_gap_fill_and_transition_preview_use_the_b3_b4_routes():
    assert "api.getSetSuggest(setId, position)" in SCREEN
    assert "api.setTransitionAudioUrl(data.id, b.item_id, 8)" in SCREEN
    assert "/suggest?position=" in API and "/transition/${itemId}/audio?bars=" in API
    # The preview plays through the one shared player, as a URL source.
    player = _read("hooks/usePlayer.js")
    assert "el.src = next.url || api.audioUrl(" in player
    assert "if (s.url) return `track:url:${s.url}`;" in player


def test_the_keys_and_undo():
    for key in ('e.key === "ArrowRight" || e.key === "ArrowLeft"', "e.altKey",
                'e.key === "t"', 'e.key === "g"', 'e.key === "Delete" || e.key === "Backspace"'):
        assert key in SCREEN, key
    # A removal is undone by putting the slot back where it was.
    remove = SCREEN[SCREEN.index("const remove = async (item)"):][:1400]
    assert 'label: "Undo"' in remove and "slotBody(item, k)" in remove
    help_ = _read("components/HelpPanel.jsx")
    assert '["Sets", "←/→ select a slot' in help_


def test_four_export_formats_including_fl():
    for fmt in ('exportAs("cue")', 'exportAs("rekordbox")', 'exportAs("fl")', 'exportAs("csv")'):
        assert fmt in SCREEN, fmt
    assert "api.exportSetFl(data.id)" in SCREEN and "export?format=fl" in API


def test_the_rail_slot_stays_empty_and_one_set_is_active():
    assert "onRailSlot?.(null)" in SCREEN
    assert "<RailSection" not in SCREEN
    assert 'aria-label="Active set"' in SCREEN


def test_no_hex_and_no_confirm():
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", SCREEN)
    assert "window.confirm" not in SCREEN and "window.prompt" not in SCREEN
