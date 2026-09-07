"""The mashup plan carries the SCORED timing options, not just the recipe.

Discover's Plan expander and the Studio's timing pills both have to offer the
same overlays, and both have to agree with the ranked row they were opened
from. That only holds while `section_options` is literally
matcher.sections.top_section_pairs — the engine the candidate row is built
from — rather than a second, parallel idea of what a good moment is.

The other half of this file guards the additive-ness of the change:
render/session.py trims the FL session export from `plan["pairings"][0]`, so
`pairings` must keep both its place and its key names.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    return m


def _sections(spans, *, vocal=True, downbeats=True):
    """A structure analysis. `downbeats=False` models a library analysed before
    P2.1: the grid is unknown, which align() must report as None."""
    out = []
    for i, (label, a, b) in enumerate(spans):
        s = {"section_index": i, "start_sec": float(a), "end_sec": float(b),
             "label": label, "energy": 0.7,
             "vocal_presence": 0.8 if vocal else 0.05,
             "repetition": 2, "confidence": 0.9}
        if downbeats:
            # A bar line a quarter-second into the section on the vocal side and
            # a tenth on the bed's, so the measured nudge is a real number.
            s["downbeats"] = [float(a) + (0.25 if vocal else 0.1)]
        out.append(s)
    return out


def _pair(models, *, v_downbeats=True, i_downbeats=True, sectioned=True):
    """Two analysed songs, returned as (vocal_id, inst_id)."""
    v_id = models.upsert_song(title="V", artist="A", source_url="u://v")
    i_id = models.upsert_song(title="I", artist="B", source_url="u://i")
    for sid, stem in ((v_id, "vocals"), (i_id, "instrumental")):
        models.upsert_features(sid, stem, {
            "bpm": 128.0, "key": "C", "mode": "major", "camelot": "8B"})
        models.upsert_features(sid, "full", {
            "bpm": 128.0, "key": "C", "mode": "major", "camelot": "8B"})
    if sectioned:
        models.replace_sections(v_id, _sections(
            [("intro", 0, 16), ("verse", 16, 48), ("chorus", 48, 80),
             ("verse", 80, 112), ("chorus", 112, 144), ("outro", 144, 160)],
            vocal=True, downbeats=v_downbeats))
        models.replace_sections(i_id, _sections(
            [("intro", 0, 16), ("drop", 16, 48), ("breakdown", 48, 80),
             ("drop", 80, 112), ("verse", 112, 144), ("outro", 144, 160)],
            vocal=False, downbeats=i_downbeats))
    return v_id, i_id


def test_plan_offers_scored_timing_options_best_first(models):
    from matcher.plan import build_mashup_plan, SECTION_OPTION_LIMIT

    plan = build_mashup_plan(*_pair(models))
    opts = plan["section_options"]

    assert opts, "an analysed pair must offer at least one timing option"
    assert len(opts) <= SECTION_OPTION_LIMIT
    fits = [o["score_section"] for o in opts]
    assert fits == sorted(fits, reverse=True), "options must be best-first"
    # intro/outro are not mashable and usable_sections drops them; if they ever
    # start appearing the pills would offer silence as a suggestion.
    labels = {o["vocal_section_label"] for o in opts}
    assert not labels & {"intro", "outro"}


def test_an_option_is_exactly_a_pair_row(models):
    """The Studio treats a plan option and the seeded candidate pair as the same
    kind of thing. That only works while both speak _pair_row's vocabulary, so
    pin the key set against the function itself rather than listing names."""
    from matcher.plan import build_mashup_plan
    from matcher.sections import top_section_pairs

    plan = build_mashup_plan(*_pair(models))
    reference = top_section_pairs(
        plan["vocal_sections"], plan["inst_sections"],
        plan["stretch_factor"] or 1.0, bpm=plan["target_bpm"], limit=1)[0]

    for opt in plan["section_options"]:
        assert set(opt) == set(reference)

    # The fields the pills actually place and label with, spelled out once so a
    # rename upstream fails here rather than silently emptying the toolbar.
    first = plan["section_options"][0]
    for key in ("vocal_section_idx", "inst_section_idx",
                "vocal_section_start", "vocal_section_end",
                "inst_section_start", "inst_section_end",
                "vocal_section_label", "inst_section_label",
                "score_section", "section_bars_vocal",
                "alignment_offset", "reason",
                # The four terms the pair card draws as bars. LBL/DUR/VOI were
                # computed and discarded until they were added here; PHR was
                # always stored. Dropping any of them empties a bar rather than
                # failing, which is why they are pinned.
                "score_label", "score_duration", "score_voice", "score_phrase"):
        assert key in first


def test_alignment_offset_is_measured_when_both_sides_have_a_grid(models):
    from matcher.plan import build_mashup_plan

    plan = build_mashup_plan(*_pair(models))
    offsets = [o["alignment_offset"] for o in plan["section_options"]]
    assert all(o is not None for o in offsets)
    # Vocal bar line sits 0.25s into its section, the bed's 0.1s, no stretch.
    assert offsets[0] == pytest.approx(0.15)


def test_alignment_offset_is_none_when_a_side_has_no_grid(models):
    """None means "we never established where the bar line is". The Studio adds
    the offset to the bed's placement, so a 0.0 here would be a silent claim
    that the sections are already aligned."""
    from matcher.plan import build_mashup_plan

    plan = build_mashup_plan(*_pair(models, i_downbeats=False))
    assert plan["section_options"]
    assert all(o["alignment_offset"] is None for o in plan["section_options"])


def test_a_pair_with_no_structure_yields_no_options(models):
    """No pills, and no exception — Audition on an un-analysed pair must still
    open the arranger."""
    from matcher.plan import build_mashup_plan

    plan = build_mashup_plan(*_pair(models, sectioned=False))
    assert plan["section_options"] == []


def test_pairings_is_untouched(models):
    """render/session.py:322 trims the FL export from plan["pairings"][0].
    section_options is ADDITIVE; it does not replace that list."""
    from matcher.plan import build_mashup_plan

    plan = build_mashup_plan(*_pair(models))
    assert plan["pairings"], "the recipe's pairings must survive"
    for key in ("vocal_start", "vocal_end", "inst_start", "inst_end",
                "vocal_label", "inst_label", "note"):
        assert key in plan["pairings"][0]
