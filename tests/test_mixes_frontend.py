"""Mixes screen (revamp step 5): contracts read from the source, like the
other *_frontend tests — there is no JS test runner."""
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


IMPORTER = _read("components/MixImporter.jsx")
BOARD = _read("components/MixMatchBoard.jsx")
API = _read("api.js")


def test_every_mixes_route_is_still_used():
    """The redesign may move a button; it may not drop a capability."""
    block = API[API.index("  importMix:"):API.index("matchMixToLibrary:") + 200]
    names = re.findall(r"^\s{2}(\w+):", block, re.M)
    names = [n for n in names if "Mix" in n or n.startswith("mix")]
    assert len(names) >= 18, names
    both = IMPORTER + BOARD
    for name in names:
        assert f"api.{name}(" in both, name


def test_no_confirm_dialogs_undo_instead():
    for src in (IMPORTER, BOARD):
        assert "window.confirm" not in src
    # Unlink and remove wait behind an Undo toast before they are sent.
    assert 'label: "Undo"' in IMPORTER
    assert "setTimeout(run, UNDO_MS)" in IMPORTER
    assert "api.unlinkMixTracks(mixId, trackIds)" in IMPORTER
    assert "api.deleteMixTrack(mixId, t.id)" in IMPORTER
    # Reset is undone by writing the old matches back.
    assert 'label: "Undo"' in BOARD and "api.saveMixAssignments(mix.id, roleDiff, matchDiff)" in BOARD


def test_waiting_changes_are_sent_before_anything_reads_the_mix():
    """A pending removal must land before a reorder, auto-link or ingest sends
    the server a picture of the mix that still contains it."""
    for fn in ("const autoResolve =", "const findInLibrary =", "const relink =", "const ingest =",
               "const onReorder =", "const selectMix ="):
        body = IMPORTER[IMPORTER.index(fn):][:400]
        assert "await flushPending()" in body, fn


def test_match_is_pick_then_click():
    assert "setPick(on ? null : v.id)" in BOARD
    assert "matchVocal(pick, b.id)" in BOARD
    assert "DndContext" not in BOARD
    # Saves stay debounced and batched.
    assert "SAVE_DEBOUNCE_MS" in BOARD


def test_filter_chips_and_two_tone_mix_cards():
    for chip in ('"In library"', '"Linked"', '"Check ⚠"', '"Needs link"'):
        assert chip in IMPORTER, chip
    assert 'className="mx-bar"' in IMPORTER
    assert "m.summary" in IMPORTER


def test_documented_pairs_carry_the_engine_tier():
    assert "e.confidence_tier && <TierBadge" in IMPORTER


def test_no_hex_in_the_mixes_components():
    for src in (IMPORTER, BOARD):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", src)
