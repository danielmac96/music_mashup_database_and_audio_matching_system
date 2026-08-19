"""
matcher/patterns.py — Mashup patterns as configuration, not as code.

A pattern says which shape of vocal section belongs over which shape of bed
section, and what the two should be doing to each other in bar length and
energy. Spec §6 asks for exactly this and asks for it to be data, so that
someone who builds verse-over-build mashups can say so without editing Python.

Before this, the only expression of the idea was two hard-coded priority dicts
in matcher/plan.py, which said what to PREFER but nothing about what pairs
sensibly with what. They are now derived from the patterns below (see
`priority_for`), so the existing callers keep working while the patterns become
the single source of truth.

Two things this file deliberately does NOT do:

* It does not rename the analyser's labels. The database stores
  intro|verse|chorus|drop|breakdown|bridge|outro; the spec's vocabulary adds
  pre_chorus, build, break and instrumental. Renaming would invalidate every
  stored section and every trained model, so spec-worded patterns are resolved
  through ALIASES instead. Widening what label_segments can emit is a separate
  change, and when it happens this map shrinks rather than the patterns moving.
* It does not score. `section_structure_score` in matcher/section_score.py asks
  these patterns a question; the patterns themselves stay declarative.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# Spec vocabulary -> what the analyser actually emits today. A pattern may be
# written in either, so a user editing settings.json can use the words the spec
# uses and still match a library labelled the old way.
ALIASES = {
    # A "break" and a "breakdown" are the same idea: the arrangement drops out.
    "break": "breakdown",
    # A pre-chorus is sung, and an instrumental section is a bed — both behave
    # like a verse for the purpose of choosing what to layer.
    "pre_chorus": "verse",
    "instrumental": "verse",
    "unknown": "verse",
}

# Deliberately NOT aliased: "build". A build is high energy RISING into a drop;
# a breakdown is the arrangement falling away. Mapping one to the other would
# promote every breakdown above choruses as a bed, on the strength of a pattern
# about a section the analyser cannot currently emit. Patterns naming "build"
# therefore stay inert on labels alone — energy_trend (P2.1) is what actually
# identifies one, and section_structure_score is where that gets asked.
UNMAPPED_LABELS = ("build",)

# Every label the matcher will consider, after aliasing.
KNOWN_LABELS = ("intro", "verse", "chorus", "drop", "breakdown", "bridge", "outro")

# Sections that are never worth layering: an intro is an intro because nothing
# is happening yet, and an outro because it has stopped happening.
EXCLUDED_LABELS = ("intro", "outro")

# Bar relationships a pattern can ask for. 1:1 means "same phrase length";
# "multiple" accepts a clean 2:1 / 1:2 / 4:1 as well, which is what looping a
# 16-bar bed under a 32-bar vocal actually is.
BAR_RELATIONSHIPS = ("equal", "multiple", "any")

# What the energy should be doing across the pair.
ENERGY_RELATIONSHIPS = ("rising", "matched", "falling", "any")


# The defaults, transcribed from spec §6. `weight` is how strongly a pattern
# pulls a pair towards the top when it matches — a chorus over a drop is the
# canonical mashup, a verse over a verse is merely valid.
DEFAULT_PATTERNS: List[Dict] = [
    {"name": "chorus_over_drop",
     "vocal_section_types": ["chorus"],
     "instrumental_section_types": ["drop"],
     "preferred_bar_relationship": "equal",
     "energy_relationship": "matched",
     "weight": 1.0},
    {"name": "verse_over_drop",
     "vocal_section_types": ["verse"],
     "instrumental_section_types": ["drop"],
     "preferred_bar_relationship": "equal",
     "energy_relationship": "rising",
     "weight": 0.9},
    {"name": "verse_over_build",
     "vocal_section_types": ["verse"],
     "instrumental_section_types": ["build"],
     "preferred_bar_relationship": "equal",
     "energy_relationship": "rising",
     "weight": 0.85},
    {"name": "chorus_over_chorus",
     "vocal_section_types": ["chorus"],
     "instrumental_section_types": ["chorus"],
     "preferred_bar_relationship": "equal",
     "energy_relationship": "matched",
     "weight": 0.8},
    {"name": "verse_over_verse",
     "vocal_section_types": ["verse"],
     "instrumental_section_types": ["verse"],
     "preferred_bar_relationship": "equal",
     "energy_relationship": "matched",
     "weight": 0.7},
    {"name": "bridge_over_break",
     "vocal_section_types": ["bridge"],
     "instrumental_section_types": ["break"],
     "preferred_bar_relationship": "multiple",
     "energy_relationship": "falling",
     "weight": 0.65},
    {"name": "intro_over_build",
     "vocal_section_types": ["intro"],
     "instrumental_section_types": ["build"],
     "preferred_bar_relationship": "multiple",
     "energy_relationship": "rising",
     "weight": 0.5},
]


def canonical(label: Optional[str]) -> str:
    """A section label as the database spells it."""
    lab = (label or "").strip().lower()
    return ALIASES.get(lab, lab) or "verse"


def _labels(pattern: Dict, key: str) -> List[str]:
    return [canonical(x) for x in (pattern.get(key) or [])]


def validate(patterns) -> List[Dict]:
    """Keep only well-formed patterns, silently dropping the rest.

    A typo in settings.json must not take scoring down: a bad pattern is one
    idea lost, an exception is the whole ranked list lost. Returns the defaults
    when nothing usable survives, because scoring with no patterns at all would
    quietly turn section_structure_score into a constant.
    """
    if not isinstance(patterns, list):
        return validate(DEFAULT_PATTERNS)
    out = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        vocal = _labels(p, "vocal_section_types")
        inst = _labels(p, "instrumental_section_types")
        if not vocal or not inst:
            continue
        bar = p.get("preferred_bar_relationship", "any")
        energy = p.get("energy_relationship", "any")
        try:
            weight = float(p.get("weight", 0.5))
        except (TypeError, ValueError):
            weight = 0.5
        out.append({
            "name": str(p.get("name") or f"{vocal[0]}_over_{inst[0]}"),
            "vocal_section_types": vocal,
            "instrumental_section_types": inst,
            "preferred_bar_relationship": bar if bar in BAR_RELATIONSHIPS else "any",
            "energy_relationship": energy if energy in ENERGY_RELATIONSHIPS else "any",
            "weight": max(0.0, min(1.0, weight)),
        })
    if out:
        return out
    # Guard against infinite recursion if DEFAULT_PATTERNS itself is broken.
    return [] if patterns is DEFAULT_PATTERNS else validate(DEFAULT_PATTERNS)


def current_patterns() -> List[Dict]:
    """The active pattern list: settings.json if it has one, else the defaults.

    Read live, like the scoring weights, so editing patterns and re-scoring does
    not need a restart."""
    try:
        from config import _load_settings
        saved = _load_settings().get("mashup_patterns")
    except Exception:  # noqa: BLE001 — config is optional for unit tests
        saved = None
    # The defaults go through validate() too: they are written in the spec's
    # vocabulary ("build", "break"), and skipping the aliasing pass would leave
    # labels in the priority map that no library ever emits.
    return validate(saved if saved else DEFAULT_PATTERNS)


def matching(vocal_label: Optional[str], inst_label: Optional[str],
             patterns: Optional[List[Dict]] = None) -> Optional[Dict]:
    """The highest-weighted pattern this pairing satisfies, or None."""
    v, i = canonical(vocal_label), canonical(inst_label)
    best = None
    for p in (patterns if patterns is not None else current_patterns()):
        if v in p["vocal_section_types"] and i in p["instrumental_section_types"]:
            if best is None or p["weight"] > best["weight"]:
                best = p
    return best


def priority_for(vocal_side: bool,
                 patterns: Optional[List[Dict]] = None) -> Dict[str, int]:
    """Label -> rank, derived from the patterns rather than hard-coded.

    Replaces matcher/plan.py's _VOCAL_LABEL_PRIORITY / _INST_LABEL_PRIORITY as
    the source of truth while keeping their exact shape, so _pick_sections and
    usable_sections need no changes. A label the patterns never mention still
    gets a rank — below every label they do — because dropping it here would
    silently remove sections from consideration rather than merely deprioritise
    them, and that is a much bigger decision than ordering.
    """
    key = "vocal_section_types" if vocal_side else "instrumental_section_types"
    best: Dict[str, float] = {}
    for p in (patterns if patterns is not None else current_patterns()):
        for label in p[key]:
            best[label] = max(best.get(label, 0.0), p["weight"])

    ranked = sorted(best, key=lambda lab: (-best[lab], lab))
    priority = {lab: rank for rank, lab in enumerate(ranked)}
    for lab in KNOWN_LABELS:
        priority.setdefault(lab, len(priority))
    return priority
