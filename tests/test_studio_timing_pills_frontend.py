"""The Studio's timing pills, pinned from Python.

Same approach as tests/test_result_filters_frontend.py: this repo has no JS
test runner and adding one is a separate decision, so the invariants that
would actually hurt if they drifted are asserted by reading the source.

The ones that matter here:

* the options must be FETCHED from the pair, not carried on the seed — the seed
  is consumed on arrival, so a seed-carried list disappears the moment you
  leave the tab, which is the whole complaint this feature answers;
* a pill must move, trim AND loop, or "cycle the suggestions" means scrubbing;
* verdicts must be keyed on the section indexes, or judging one overlay
  overwrites your verdict on the others;
* the two lanes must be resolved by songId, not by position.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"
STUDIO = "components/MixStudio.jsx"
# The pair model, shared by every surface that can hand a pair to Studio: the
# dock, the track detail's partners rail and the player bar.
PAIR_MODEL = "components/pairs/pairModel.js"
APP = "App.jsx"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def test_the_pair_handed_across_carries_its_scored_option():
    """Studio offers the row's own pair as a pill. top_section_pairs is capped,
    so a row scored under different weights need not be in the plan's six -
    without this the pill you arrived on could be missing from its own list.

    scoredOptionOf lives in the pair model rather than one screen: App's
    pairToStudio is reached from the dock, the partners rail and the bar."""
    assert "scoredOption: scoredOptionOf(c)" in _read(APP)
    src = _read(PAIR_MODEL)
    fn = src[src.index("export function scoredOptionOf"):]
    fn = fn[:fn.index("\n}")]
    for key in ("vocal_section_idx", "inst_section_idx",
                "vocal_section_start", "vocal_section_end",
                "inst_section_start", "inst_section_end",
                "alignment_offset", "score_section"):
        assert key in fn, key



def test_studio_fetches_the_options_from_the_pair():
    src = _read(STUDIO)
    assert "usePlan(pairCtx?.vocalSongId, pairCtx?.instSongId)" in src
    assert "section_options" in src


def test_a_pill_moves_trims_and_loops():
    """placementFor is the whole contract: an offset, a trim on both sides, and
    a loop. Drop any one of them and cycling options stops being hands-free."""
    src = _read(STUDIO)
    body = src[src.index("export function placementFor"):]
    body = body[:body.index("\n}")]
    for token in ("offsetSec", "clipStart", "clipEnd", "loop",
                  "alignment_offset"):
        assert token in body, token
    assert "applyTimingOption" in src
    assert "setLoop(place.loop)" in src


def test_the_loop_is_the_intersection_of_the_two_trims():
    """MashupEngine._armVoice only loops natively while the loop window sits
    inside the trim. Sized to the longer side, the shorter one plays once and
    falls silent — which reads as a bug in the suggestion, not in the loop."""
    src = _read(STUDIO)
    body = src[src.index("export function placementFor"):]
    body = body[:body.index("\n}")]
    assert "Math.min(vLen, bLen)" in body


def test_the_bar_nudge_is_unknown_not_zero_when_there_is_no_grid():
    src = _read(STUDIO)
    body = src[src.index("export function placementFor"):]
    body = body[:body.index("\n}")]
    assert "opt.alignment_offset ?? 0" in body, \
        "null must fall back to no nudge, and never be read as a measured 0"


def test_verdicts_are_keyed_on_the_section_pair():
    """pair_feedback's unique key includes the sections. Sending null sections
    from here would collapse every overlay of two records onto one row and
    destroy the earlier verdict."""
    src = _read(STUDIO)
    assert "savePairFeedback" in src
    call = src[src.index("api.savePairFeedback"):]
    call = call[:call.index("});")]
    assert "vocalSection: opt.vocal_section_idx" in call
    assert "instSection: opt.inst_section_idx" in call


def test_the_pair_lanes_are_resolved_by_song_id():
    """Lanes can be reordered, removed, or joined by a third, at which point
    index 0/1 stops meaning 'the vocal' and 'the bed'."""
    src = _read(STUDIO)
    apply_fn = src[src.index("const applyTimingOption"):]
    apply_fn = apply_fn[:apply_fn.index("}, [pairCtx]);")]
    assert "l.songId === pairCtx?.vocalSongId" in apply_fn
    assert "l.songId === pairCtx?.instSongId" in apply_fn
    assert not re.search(r"lanes\[[01]\]", apply_fn)


def test_the_pair_survives_a_reload_but_the_options_do_not():
    """pairCtx is persisted so the pills come back; the option list is not, so
    a re-analysis can never leave stale timings on screen."""
    src = _read(STUDIO)
    payload = src[src.index("const payload = {"):]
    payload = payload[:payload.index("};")]
    assert "pairCtx" in payload
    assert "activeOptionKey" in payload
    assert "timingOptions" not in payload
    assert "saved.pairCtx" in src


def test_the_pills_are_keyboard_reachable():
    src = _read(STUDIO)
    assert "cycleOption" in src
    assert 'e.key === "["' in src
