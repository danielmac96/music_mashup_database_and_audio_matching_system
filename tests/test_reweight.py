"""C.2 — try a different balance without re-scoring the library.

Every part of the composite is already on the candidate row, so re-weighting is
arithmetic. Before this it meant Settings → Save → "Score library" → minutes of
walking the whole pair matrix again, which is why nobody ever tried a different
balance.

The load-bearing property is that the re-weight and the scorer agree: feeding
the saved weights back in must reproduce the total the scorer stored, or the
two are separate implementations of the composite and will drift.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "test.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "cfg"))
    return p


def _side(sid, **over):
    base = {"song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
            "key": "A", "mode": "minor", "camelot": "8A",
            "loudness_rms": 0.05, "energy": 0.5}
    base.update(over)
    return base


def _library(db_path, specs, combo="vocal_over_instrumental"):
    """specs: list of dicts of sub-scores (+ optional score_section/effort)."""
    from database.models import init_db, upsert_candidate, upsert_song

    init_db(db_path)
    ids = []
    for n in range(len(specs)):
        for role in ("v", "i"):
            ids.append(upsert_song(f"{role}{n}", "A", f"https://sc/{role}{n}",
                                   200, "Pop", status="analysed",
                                   db_path=db_path))
    for n, spec in enumerate(specs):
        v, i = ids[2 * n], ids[2 * n + 1]
        section_pair = None
        if spec.get("score_section") is not None:
            section_pair = {"vocal_section_idx": 0, "inst_section_idx": 0,
                            "score_section": spec["score_section"]}
        upsert_candidate(_side(v), _side(i), dict(spec), combo_type=combo,
                         section_pair=section_pair, db_path=db_path)
    return ids


def _scores(bpm, key, energy, timbre, collision, *, total=None,
            effort=None, section=None):
    """One candidate's parts, with the composite the scorer would store."""
    from config import current_match_weights, current_float

    parts = {"bpm_score": bpm, "key_score": key, "energy_score": energy,
             "timbre_score": timbre, "collision_score": collision}
    if total is None:
        w = current_match_weights("vocal_over_instrumental")
        whole = sum(parts[k] * w[k] for k in w)
        if section is not None:
            sw = current_float("section_weight")
            whole = (1 - sw) * whole + sw * section
        total = whole * (1 - current_float("effort_weight") * (effort or 0.0))
    out = {**parts, "total": round(total, 4)}
    if effort is not None:
        out["score_effort"] = effort
    if section is not None:
        out["score_section"] = section
    return out


# ── The agreement that keeps the two implementations from drifting ───────────

def test_reweighting_with_the_saved_weights_reproduces_the_stored_total(db_path):
    from config import current_match_weights
    from database.models import get_candidates_enriched

    _library(db_path, [
        _scores(0.90, 0.40, 0.70, 0.30, 0.80),
        _scores(0.30, 0.95, 0.55, 0.85, 0.20, effort=0.4),
        _scores(0.60, 0.60, 0.60, 0.60, 0.60, effort=0.1, section=0.75),
    ])

    stored = {r["id"]: r["score_total"]
              for r in get_candidates_enriched(limit=50, db_path=db_path)}
    same = {r["id"]: r["score_total"] for r in get_candidates_enriched(
        limit=50, weights=current_match_weights(), db_path=db_path)}

    assert stored and same.keys() == stored.keys()
    for rid, total in stored.items():
        assert same[rid] == pytest.approx(total, abs=1e-3), (
            "the re-weight and matcher.match compute the composite differently")


def test_a_heavier_weight_promotes_the_pair_that_is_strong_on_it(db_path):
    """The whole point: 'tempo matters more than key tonight' reorders the list."""
    from database.models import get_candidates_enriched

    _library(db_path, [
        _scores(0.95, 0.10, 0.5, 0.5, 0.5),   # tempo-strong, key-weak
        _scores(0.10, 0.95, 0.5, 0.5, 0.5),   # key-strong, tempo-weak
    ])
    only = {"bpm_score": 0.0, "key_score": 0.0, "energy_score": 0.0,
            "timbre_score": 0.0, "collision_score": 0.0}

    tempo = get_candidates_enriched(limit=50, db_path=db_path,
                                    weights={**only, "bpm_score": 1.0})
    key = get_candidates_enriched(limit=50, db_path=db_path,
                                  weights={**only, "key_score": 1.0})

    assert tempo[0]["score_bpm"] == pytest.approx(0.95)
    assert key[0]["score_key"] == pytest.approx(0.95)
    assert tempo[0]["id"] != key[0]["id"]


def test_the_percentile_is_recomputed_so_min_match_still_means_something(db_path):
    """The row displays the percentile and Min match gates on it. Leaving it
    ranked against the OLD totals filters by a ranking the user just changed."""
    from database.models import get_candidates_enriched

    _library(db_path, [
        _scores(0.95, 0.10, 0.5, 0.5, 0.5),
        _scores(0.60, 0.50, 0.5, 0.5, 0.5),
        _scores(0.10, 0.95, 0.5, 0.5, 0.5),
    ])
    only_tempo = {"bpm_score": 1.0, "key_score": 0.0, "energy_score": 0.0,
                  "timbre_score": 0.0, "collision_score": 0.0}

    rows = get_candidates_enriched(limit=50, db_path=db_path,
                                   weights=only_tempo)
    # Ordered by the new total, and the displayed percentile agrees with it.
    assert [r["score_bpm"] for r in rows] == [0.95, 0.60, 0.10]
    assert [r["score_percentile"] for r in rows] == sorted(
        (r["score_percentile"] for r in rows), reverse=True)
    assert all(r.get("reweighted") for r in rows)


def test_reweighting_reranks_the_whole_table_not_the_visible_page(db_path):
    """A pair the old weights buried must be able to reach the top.

    Re-sorting the returned fifty would answer the wrong question: the pairs a
    different balance promotes are mostly not in the old top fifty.
    """
    from database.models import get_candidates_enriched

    # 40 mediocre pairs, then one that is superb on tempo and awful elsewhere,
    # so the old composite puts it dead last.
    specs = [_scores(0.55, 0.90, 0.9, 0.9, 0.9) for _ in range(40)]
    specs.append(_scores(1.00, 0.01, 0.01, 0.01, 0.01))
    _library(db_path, specs)

    baseline = get_candidates_enriched(limit=5, max_per_song=0, db_path=db_path)
    assert all(r["score_bpm"] < 1.0 for r in baseline)

    tempo = get_candidates_enriched(
        limit=5, max_per_song=0, db_path=db_path,
        weights={"bpm_score": 1.0, "key_score": 0.0, "energy_score": 0.0,
                 "timbre_score": 0.0, "collision_score": 0.0})
    assert tempo[0]["score_bpm"] == pytest.approx(1.0)


def test_an_unmeasured_subscore_counts_as_neutral_not_zero(db_path):
    """A row scored before Phase D has no collision value. Reading NULL as 0
    would silently demote every pre-Phase-D pair the moment collision is given
    any weight."""
    from database.models import UNMEASURED_SUBSCORE, get_candidates_enriched

    # total is given explicitly: a NULL collision cannot go through the
    # scorer's own weighted sum, which is exactly the case under test.
    _library(db_path, [_scores(0.5, 0.5, 0.5, 0.5, None, total=0.5)])
    rows = get_candidates_enriched(
        limit=5, db_path=db_path,
        weights={"bpm_score": 0.0, "key_score": 0.0, "energy_score": 0.0,
                 "timbre_score": 0.0, "collision_score": 1.0})
    assert rows[0]["score_collision"] is None
    assert rows[0]["score_total"] == pytest.approx(UNMEASURED_SUBSCORE, abs=1e-3)


def test_model_scored_rows_are_left_alone(db_path):
    """A learned total is a probability, not a weighted sum of these five —
    re-weighting it would be inventing a number."""
    from database.models import get_candidates_enriched, init_db, upsert_candidate

    init_db(db_path)
    from database.models import upsert_song
    v = upsert_song("v", "A", "https://sc/v", 200, status="analysed", db_path=db_path)
    i = upsert_song("i", "A", "https://sc/i", 200, status="analysed", db_path=db_path)
    upsert_candidate(_side(v), _side(i),
                     _scores(0.1, 0.1, 0.1, 0.1, 0.1, total=0.87),
                     scorer="model", model_version="v1", db_path=db_path)

    rows = get_candidates_enriched(
        limit=5, db_path=db_path,
        weights={"bpm_score": 1.0, "key_score": 0.0, "energy_score": 0.0,
                 "timbre_score": 0.0, "collision_score": 0.0})
    assert rows[0]["score_total"] == pytest.approx(0.87)
    assert not rows[0].get("reweighted")


def test_weights_are_normalised_so_sliders_need_not_add_up(db_path):
    from database.models import get_candidates_enriched, normalise_weights

    assert normalise_weights({"bpm_score": 2, "key_score": 2}) == {
        "bpm_score": 0.5, "key_score": 0.5, "energy_score": 0.0,
        "timbre_score": 0.0, "collision_score": 0.0}
    assert normalise_weights({}) is None
    assert normalise_weights({"bpm_score": 0}) is None
    assert normalise_weights(None) is None
    assert normalise_weights({"bpm_score": "nonsense"}) is None

    _library(db_path, [_scores(0.8, 0.2, 0.5, 0.5, 0.5)])
    doubled = get_candidates_enriched(
        limit=5, db_path=db_path,
        weights={"bpm_score": 2.0, "key_score": 2.0, "energy_score": 2.0,
                 "timbre_score": 2.0, "collision_score": 2.0})
    plain = get_candidates_enriched(
        limit=5, db_path=db_path,
        weights={"bpm_score": 1.0, "key_score": 1.0, "energy_score": 1.0,
                 "timbre_score": 1.0, "collision_score": 1.0})
    assert doubled[0]["score_total"] == pytest.approx(plain[0]["score_total"])


def test_the_vocal_path_redistribution_still_applies(db_path):
    """Timbre's weight moves onto collision on the vocal path (config._for_combo).
    That is a property of the path, not of the user's preference, so a
    user-supplied weight set must still go through it."""
    from database.models import get_candidates_enriched

    # Timbre 1.0 on a vocal row: _for_combo moves it all onto collision, so the
    # total should read the COLLISION score, not the timbre one.
    _library(db_path, [_scores(0.1, 0.1, 0.1, 0.2, 0.9)])
    rows = get_candidates_enriched(
        limit=5, db_path=db_path,
        weights={"bpm_score": 0.0, "key_score": 0.0, "energy_score": 0.0,
                 "timbre_score": 1.0, "collision_score": 0.0})
    assert rows[0]["score_total"] == pytest.approx(0.9, abs=1e-3)


def test_instrumental_rows_keep_their_own_weighting(db_path):
    from database.models import get_candidates_enriched

    _library(db_path, [_scores(0.1, 0.1, 0.1, 0.2, 0.9)],
             combo="instrumental_over_instrumental")
    rows = get_candidates_enriched(
        limit=5, combo_type="instrumental_over_instrumental", db_path=db_path,
        weights={"bpm_score": 0.0, "key_score": 0.0, "energy_score": 0.0,
                 "timbre_score": 1.0, "collision_score": 0.0})
    assert rows[0]["score_total"] == pytest.approx(0.2, abs=1e-3)


# ── The query-string contract ────────────────────────────────────────────────

def test_bad_weight_payloads_are_refused_not_ignored():
    """The user just dragged a slider. Quietly ranking by something else is
    worse than saying no."""
    from fastapi import HTTPException
    from api.routes.mashups import parse_weights

    assert parse_weights(None) is None
    assert parse_weights("") is None
    assert parse_weights('{"bpm_score": 1}') == {"bpm_score": 1}

    for bad in ("{not json", '["bpm_score"]', '{"loudness": 1}',
                '{"bpm_score": 0}'):
        with pytest.raises(HTTPException):
            parse_weights(bad)


def test_export_can_reapply_the_same_weights():
    """A re-weighted list is a different ranking; its top N is a different N."""
    from api.routes.mashups import BatchSessionRequest

    fields = set(BatchSessionRequest.model_fields)
    for name in ("weights", "effort_weight", "section_weight"):
        assert name in fields
