"""Mashup patterns as configuration (P2.2, spec §6).

The point of this file is that the patterns became the single source of truth
for what pairs with what, WITHOUT changing how the existing library ranks. The
two things worth guarding are that equivalence and the aliasing, because both
fail silently.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from matcher import patterns as pat  # noqa: E402

# The hard-coded dicts matcher/plan.py carried before patterns existed.
LEGACY_VOCAL = {"chorus": 0, "verse": 1, "bridge": 2, "drop": 3,
                "breakdown": 4, "intro": 5, "outro": 6}
LEGACY_INST = {"drop": 0, "chorus": 1, "verse": 2, "breakdown": 3,
               "bridge": 4, "intro": 5, "outro": 6}

# What _pick_sections actually keeps — intro and outro are dropped outright, so
# their rank cannot affect any result.
USED = ("chorus", "verse", "bridge", "drop", "breakdown")


def _order(priority, labels=USED):
    return [lab for lab in sorted(labels, key=lambda x: priority[x])]


# ── the ordering is preserved ────────────────────────────────────────────────

def test_derived_priority_matches_the_legacy_order_for_labels_that_are_used():
    """Patterns replaced two hard-coded dicts. If that reordered the labels
    _pick_sections keeps, every section choice in the library would shift on an
    upgrade that was supposed to be a refactor."""
    assert _order(pat.priority_for(True)) == _order(LEGACY_VOCAL)
    assert _order(pat.priority_for(False)) == _order(LEGACY_INST)


def test_plan_still_exports_the_dicts_its_importers_expect():
    from matcher.plan import _INST_LABEL_PRIORITY, _VOCAL_LABEL_PRIORITY
    assert _order(_VOCAL_LABEL_PRIORITY) == _order(LEGACY_VOCAL)
    assert _order(_INST_LABEL_PRIORITY) == _order(LEGACY_INST)


def test_every_known_label_gets_a_rank():
    """A label the patterns never mention must still be rankable — dropping it
    here would remove sections from consideration rather than deprioritise
    them, which is a far bigger decision than ordering."""
    for side in (True, False):
        priority = pat.priority_for(side)
        for label in pat.KNOWN_LABELS:
            assert label in priority


# ── aliasing ─────────────────────────────────────────────────────────────────

def test_spec_vocabulary_resolves_to_stored_labels():
    """A user may write patterns in the spec's words; the database speaks its
    own. Renaming the analyser's labels would invalidate every stored section."""
    assert pat.canonical("break") == "breakdown"
    assert pat.canonical("pre_chorus") == "verse"
    assert pat.canonical("instrumental") == "verse"
    assert pat.canonical("CHORUS") == "chorus"


def test_build_is_not_aliased_to_breakdown():
    """A build is high energy RISING into a drop; a breakdown is the
    arrangement falling away. Aliasing them promoted every breakdown above
    choruses as a bed, on the strength of a pattern about a section the
    analyser cannot emit."""
    assert pat.canonical("build") == "build"
    inst = pat.priority_for(False)
    assert inst["breakdown"] > inst["chorus"]


def test_defaults_are_aliased_too():
    """The defaults are written in the spec's vocabulary, so skipping the
    aliasing pass on them would leave labels in the priority map that no
    library ever emits."""
    for p in pat.current_patterns():
        for label in p["vocal_section_types"] + p["instrumental_section_types"]:
            assert label not in pat.ALIASES, f"{label} should have been aliased"


# ── matching ─────────────────────────────────────────────────────────────────

def test_chorus_over_drop_is_the_strongest_pattern():
    match = pat.matching("chorus", "drop")
    assert match is not None
    assert match["name"] == "chorus_over_drop"
    assert match["weight"] == 1.0


def test_an_unlisted_pairing_matches_nothing():
    assert pat.matching("outro", "intro") is None


def test_the_best_pattern_wins_when_several_apply():
    custom = [
        {"name": "weak", "vocal_section_types": ["chorus"],
         "instrumental_section_types": ["drop"], "weight": 0.2},
        {"name": "strong", "vocal_section_types": ["chorus"],
         "instrumental_section_types": ["drop"], "weight": 0.9},
    ]
    assert pat.matching("chorus", "drop", pat.validate(custom))["name"] == "strong"


def test_matching_accepts_spec_words_on_either_side():
    assert pat.matching("pre_chorus", "drop") is not None


# ── validation ───────────────────────────────────────────────────────────────

def test_a_malformed_pattern_is_dropped_not_raised():
    """A typo in settings.json must cost one idea, not the whole ranked list."""
    out = pat.validate([
        {"name": "good", "vocal_section_types": ["chorus"],
         "instrumental_section_types": ["drop"]},
        {"name": "no sides"},
        "not a dict",
        {"vocal_section_types": ["verse"]},          # no instrumental side
    ])
    assert [p["name"] for p in out] == ["good"]


def test_unknown_relationships_fall_back_to_any():
    out = pat.validate([{
        "vocal_section_types": ["chorus"], "instrumental_section_types": ["drop"],
        "preferred_bar_relationship": "sideways", "energy_relationship": "purple",
    }])
    assert out[0]["preferred_bar_relationship"] == "any"
    assert out[0]["energy_relationship"] == "any"


def test_weights_are_clamped_and_bad_ones_defaulted():
    out = pat.validate([
        {"vocal_section_types": ["chorus"], "instrumental_section_types": ["drop"],
         "weight": 9.0},
        {"vocal_section_types": ["verse"], "instrumental_section_types": ["drop"],
         "weight": -3},
        {"vocal_section_types": ["bridge"], "instrumental_section_types": ["drop"],
         "weight": "loud"},
    ])
    assert [p["weight"] for p in out] == [1.0, 0.0, 0.5]


def test_an_entirely_unusable_list_falls_back_to_the_defaults():
    """Scoring with no patterns at all would quietly turn the structure score
    into a constant, which looks like it is working."""
    assert pat.validate(["junk", 42]) == pat.validate(pat.DEFAULT_PATTERNS)
    assert pat.validate("not a list") == pat.validate(pat.DEFAULT_PATTERNS)


def test_a_pattern_without_a_name_gets_one():
    out = pat.validate([{"vocal_section_types": ["chorus"],
                         "instrumental_section_types": ["drop"]}])
    assert out[0]["name"] == "chorus_over_drop"


# ── live config ──────────────────────────────────────────────────────────────

def test_patterns_come_from_settings_when_present(tmp_path, monkeypatch):
    """Live-read like the scoring weights, so editing patterns and re-scoring
    does not need a restart."""
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    import config
    importlib.reload(config)
    config.save_settings({"mashup_patterns": [{
        "name": "only_bridges", "vocal_section_types": ["bridge"],
        "instrumental_section_types": ["breakdown"], "weight": 1.0}]})

    importlib.reload(pat)
    try:
        active = pat.current_patterns()
        assert [p["name"] for p in active] == ["only_bridges"]
        # And the derived priority follows the edit.
        assert pat.priority_for(True)["bridge"] == 0
    finally:
        importlib.reload(config)
        importlib.reload(pat)
