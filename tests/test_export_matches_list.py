"""A.2 / A.3 — the export must describe the list it was launched from.

Two independent ways that failed:

* **Two section choosers.** matcher.plan.build_pairings ranked by label
  priority and a seconds-based duration fit; matcher.sections.top_section_pairs
  — which produced the candidate row — ranks by label, vocal presence and a
  bars-based phrase fit. Both answers reached the user at once: the ranked row
  showed one section pair and the Plan expander directly beneath it showed
  another, while the FL export silently rendered the second.

* **Dropped filters.** "Export top N" sent min_score and the taste filters but
  not max_effort, order, adventure or the sort. Turning on "Free builds",
  sorting by Effort and pressing the button gave you the ten best-SCORING pairs
  out of the unfiltered list.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── A.2: one chooser ─────────────────────────────────────────────────────────

def _sections(spec):
    return [
        {"section_index": n, "start_sec": s, "end_sec": e, "label": lab,
         "energy": en, "vocal_presence": vp, "repetition": 1, "confidence": 0.9}
        for n, (s, e, lab, en, vp) in enumerate(spec)
    ]


VOCAL_SECTIONS = _sections([
    (0.0, 8.0, "intro", 0.2, 0.0),
    (8.0, 24.0, "verse", 0.5, 0.9),
    (24.0, 56.0, "chorus", 0.9, 0.95),
])
BED_SECTIONS = _sections([
    (0.0, 16.0, "breakdown", 0.4, 0.0),
    (16.0, 48.0, "drop", 0.95, 0.02),
    (48.0, 64.0, "verse", 0.6, 0.05),
])


def test_build_pairings_agrees_with_the_scorer_chooser():
    """The Plan's top pairing must be the candidate row's section pair."""
    from matcher.plan import build_pairings
    from matcher.sections import top_section_pairs

    pairings = build_pairings(VOCAL_SECTIONS, BED_SECTIONS, 1.0, bpm=128.0)
    rows = top_section_pairs(VOCAL_SECTIONS, BED_SECTIONS, stretch=1.0,
                             bpm=128.0, limit=4)

    assert pairings and rows
    assert pairings[0]["vocal_start"] == rows[0]["vocal_section_start"]
    assert pairings[0]["inst_start"] == rows[0]["inst_section_start"]
    # And every pairing, in the same order — not just the winner.
    assert [p["vocal_section_idx"] for p in pairings] == \
           [r["vocal_section_idx"] for r in rows]


def test_build_pairings_carries_the_section_indices():
    """Without them a caller has to match pairings back on a rounded float."""
    from matcher.plan import build_pairings

    for p in build_pairings(VOCAL_SECTIONS, BED_SECTIONS, 1.0, bpm=128.0):
        assert p["vocal_section_idx"] is not None
        assert p["inst_section_idx"] is not None


def test_build_pairings_with_no_usable_sections_is_empty():
    from matcher.plan import build_pairings
    assert build_pairings([], BED_SECTIONS, 1.0) == []
    assert build_pairings(VOCAL_SECTIONS, [], 1.0) == []


def test_a_pinned_plan_offers_only_the_pinned_pairing():
    """Three alternatives under a pinned row put back the ambiguity the pin
    exists to remove."""
    from matcher.plan import _pinned_pairing

    p = _pinned_pairing(VOCAL_SECTIONS, BED_SECTIONS, 1, 2, 1.0)
    assert p is not None
    assert p["vocal_start"] == 8.0 and p["inst_start"] == 48.0
    assert p["pinned"] is True


def test_a_pin_at_a_missing_index_resolves_to_nothing():
    from matcher.plan import _pinned_pairing
    assert _pinned_pairing(VOCAL_SECTIONS, BED_SECTIONS, 99, 0, 1.0) is None
    assert _pinned_pairing(VOCAL_SECTIONS, BED_SECTIONS, 0, 99, 1.0) is None


# ── A.3: the export sees the same page ───────────────────────────────────────

def _rows(*specs):
    """Minimal candidate-row dicts: (id, score_total, score_effort, pop)."""
    return [
        {"id": i, "score_total": total, "score_effort": effort,
         "vocal_popularity": pop, "inst_popularity": 0.0,
         "vocal_song_id": i, "inst_song_id": 100 + i,
         "vocal_section_idx": 1, "inst_section_idx": 2, "harmonic_shift": -1}
        for i, (total, effort, pop) in enumerate(specs)
    ]


def test_effort_sort_puts_the_cheapest_build_first():
    from api.routes.mashups import _apply_sort

    rows = _rows((0.90, 0.80, 0.1), (0.70, 0.10, 0.2), (0.80, 0.40, 0.3))
    assert [r["id"] for r in _apply_sort(rows, "effort")] == [1, 2, 0]


def test_effort_sort_puts_an_unmeasured_cost_last_not_first():
    """An unknown cost is not a free one."""
    from api.routes.mashups import _apply_sort

    rows = _rows((0.90, None, 0.1), (0.70, 0.10, 0.2))
    assert [r["id"] for r in _apply_sort(rows, "effort")] == [1, 0]


def test_popularity_sort_adds_both_sides():
    from api.routes.mashups import _apply_sort

    rows = _rows((0.90, 0.1, 0.1), (0.70, 0.2, 0.9))
    assert [r["id"] for r in _apply_sort(rows, "popularity")] == [1, 0]


def test_score_and_uncertain_keep_the_server_order():
    from api.routes.mashups import _apply_sort

    rows = _rows((0.10, 0.9, 0.1), (0.90, 0.1, 0.9))
    for mode in ("score", "uncertain", ""):
        assert [r["id"] for r in _apply_sort(rows, mode)] == [0, 1]


def test_export_pair_carries_the_rows_own_choices():
    """A.1's pin has to survive the trip from the ranked row to the worker."""
    from api.routes.mashups import _export_pair

    row = _rows((0.9, 0.1, 0.1))[0]
    assert _export_pair(row) == {
        "vocal_song_id": 0, "inst_song_id": 100,
        "vocal_section_idx": 1, "inst_section_idx": 2, "harmonic_shift": -1,
    }


def test_batch_request_defaults_do_not_silently_drop_controls():
    """Every control that changes WHICH rows are on screen must be a field on
    the export request, or the export runs a different query."""
    from api.routes.mashups import BatchSessionRequest

    fields = set(BatchSessionRequest.model_fields)
    for name in ("max_effort", "order", "adventure", "sort", "limit",
                 "min_score", "max_per_song", "genre", "era", "energy",
                 "bpm_band", "vocal_forward", "vocal_song_id", "inst_song_id"):
        assert name in fields, f"export request cannot express {name}"


def test_batch_request_rejects_a_bogus_sort():
    from fastapi import HTTPException
    from api.routes.mashups import _validate_list_params

    with pytest.raises(HTTPException):
        _validate_list_params("vocal_over_instrumental", 3, "score", 0.0,
                              "loudest", "", "", "")
