"""Phase 2 of the UX plan: every rating can carry a reason, and the user can
see their ratings counting toward something.

The 2026-10-15 persona simulation rated a collapsed card 9 times and never saw
a WHY chip; a double-click on an expanded card collapsed it instead of opening
Studio; and an empty Keepers list told users to loosen filters they had not
set. There is no JS test runner — these read the source (readme §8).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def test_the_tuning_counter_uses_the_term_reports_thresholds():
    from matcher import term_report
    model = _read("components/pairs/pairModel.js")
    m = re.search(r"export const TUNING_MIN_PER_CLASS = (\d+);", model)
    assert m and int(m.group(1)) == term_report.MIN_PER_CLASS
    ratings = _read("hooks/useRatings.js")
    assert f"n >= {term_report.GOOD}" in ratings and f"n <= {term_report.BAD}" in ratings
    assert "tuning }" in ratings or "tuning};" in ratings or ", tuning" in ratings


def test_the_rail_shows_ratings_toward_tuning():
    side = _read("shell/Sidebar.jsx")
    assert "rail-tuning" in side and "tuning.need" in side
    assert "tuning={ratings.tuning}" in _read("App.jsx")
    css = _read("styles.css")
    assert ".rail-tuning {" in css and ".rail-tuning.ready" in css


def test_a_collapsed_card_just_rated_shows_its_why_chips():
    card = _read("components/PairCard.jsx")
    assert "justRated = false" in card
    assert "!expanded && justRated && rating && onReason" in card
    # One row for both places, so the inline chips and the expanded ones agree.
    assert card.count("<WhyRow ") == 2 and "function WhyRow(" in card
    assert ".pc-why {" in _read("styles.css")


def test_the_dock_remembers_the_pair_just_rated_and_its_keys_tag_reasons():
    hook = _read("hooks/usePairDock.js")
    assert "setLastRated({ key: keyOf(candidate), candidate, n })" in hook
    assert "setTimeout(() => setLastRated(null), 8000)" in hook
    # Rating by key and by star both note it.
    rate = hook[hook.index("const rateAndAdvance"):hook.index("const unhide")]
    assert "noteRated(candidate, n)" in rate
    dock = _read("components/PairDock.jsx")
    assert "dock.noteRated?.(c, n)" in dock and "justRated={dock.lastRated?.key === k}" in dock
    # The reason keys act on the pair just rated, not the cursor's.
    assert "REASON_KEYS.indexOf(e.key.toLowerCase())" in hook
    assert "ratings.toggleReason(lastRated.candidate, reason.key)" in hook


def test_reason_keys_do_not_collide_with_the_docks_other_keys():
    model = _read("components/pairs/pairModel.js")
    keys = re.search(r'export const REASON_KEYS = "([a-z]+)";', model).group(1)
    taken = set("jkcavbh12345 ")
    assert not (set(keys) & taken)
    from_reasons = re.findall(r'\{ key: "(\w+)", label', model)
    assert len(keys) >= 7 and len(from_reasons) >= 7


def test_double_click_on_an_expanded_card_opens_studio():
    card = _read("components/PairCard.jsx")
    assert "onClick={onClick} onDoubleClick={onDoubleClick}" in card
    body = card[card.index("const onClick = (e) =>"):card.index("const onDoubleClick")]
    assert "e.detail > 1" in body and "setTimeout(() => onToggle?.(), DBL_CLICK_MS)" in body
    assert "clearTimeout(clickTimer.current); onStudio?.();" in card


def test_an_empty_keepers_list_names_the_real_cause():
    dock = _read("components/PairDock.jsx")
    assert "No keepers yet." in dock
    assert 'filters.rated === "loved" && nServer === 1' in dock
    msg = dock[dock.index("No pairs match these filters."):][:10]
    assert msg
    assert "!keepersOnly && (nServer > 0 || filters.search)" in dock


def test_the_key_strip_and_help_name_the_reason_keys():
    assert "q–u why" in _read("components/PairDock.jsx")
    assert "q–u tag why" in _read("components/HelpPanel.jsx")
