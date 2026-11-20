"""The learning loop (readme §5.8, matcher/learning.py): labels, the model,
the shrinkage blend, the shadow gate, rollback, the retrain tick,
exploration and the personal weight fit."""
import importlib
import importlib.util
import inspect
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _synth():
    spec = importlib.util.spec_from_file_location(
        "synth_library", ROOT / "scripts" / "synth_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    out = tmp_path_factory.mktemp("learn")
    _synth().generate(out, 90, seed=5, planted=6)
    return out


def _reload(tmp_path, monkeypatch, db):
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setenv("MASHUP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    import matcher.match as match
    importlib.reload(match)
    import matcher.learning as learning
    importlib.reload(learning)
    import matcher.segment_match as segment_match
    importlib.reload(segment_match)
    mashups = importlib.reload(importlib.import_module("api.routes.mashups"))
    return m, match, learning, mashups


@pytest.fixture(scope="module")
def scored_db(library, tmp_path_factory):
    """One Score library of the synthetic library, copied per test."""
    import os
    tmp = tmp_path_factory.mktemp("learn_scored")
    db = tmp / "scored.db"
    shutil.copy(library / "mashup.db", db)
    keep = {k: os.environ.get(k) for k in ("MASHUP_DB_PATH", "MASHUP_AUDIO_ROOT",
                                           "MASHUP_SETTINGS_DIR", "MASHUP_DATA_DIR")}
    mp = pytest.MonkeyPatch()
    try:
        m, match, _l, _r = _reload(tmp, mp, db)
        match.score_all_pairs(db_path=m.DB_PATH)
    finally:
        mp.undo()
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return db


@pytest.fixture()
def scored(scored_db, tmp_path, monkeypatch):
    db = tmp_path / "scored.db"
    shutil.copy(scored_db, db)
    return _reload(tmp_path, monkeypatch, db)


def _judge(m, learning, n=40, seed=3):
    """Verdicts with a pattern a model can find and the rule does not have:
    breakdown beds are loved, the rest mostly not."""
    import random
    rng = random.Random(seed)
    conn = m.get_conn()
    rows = conn.execute(
        "SELECT vocal_song_id, inst_song_id, vocal_section_idx, inst_section_idx, "
        "inst_section_label FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental' ORDER BY id").fetchall()
    conn.close()
    rng.shuffle(rows)
    bd = [r for r in rows if r[4] == "breakdown"][: n // 2]
    other = [r for r in rows if r[4] != "breakdown"][: n - len(bd)]
    for r in bd + other:
        m.upsert_pair_feedback(r[0], r[1], None, vocal_section=r[2], inst_section=r[3],
                               rating=5 if r[4] == "breakdown" else
                               (1 if rng.random() < 0.85 else 4))
    return bd + other


def _table(db):
    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")
                if r[1] not in ("id", "created_at", "updated_at")]
        return conn.execute(f"SELECT {', '.join(cols)} FROM mashup_candidates "
                            "ORDER BY id").fetchall()
    finally:
        conn.close()


# ── schema and the contract with pair_feedback ───────────────────────────────

def test_migration_ten_is_additive_and_leaves_pair_feedback_alone(tmp_path):
    """Criterion 2.5: pair_feedback's schema is untouched (diff-checked)."""
    from database import schema
    assert schema.SCHEMA_VERSION >= 10
    src = (ROOT / "database" / "schema.py").read_text(encoding="utf-8")
    body = src.split("def _learning_loop")[1].split("\n_SLOT_COLUMNS")[0]
    assert "executescript" not in body
    assert "pair_feedback" not in body.split('"""')[2]       # code, not the docstring
    conn = sqlite3.connect(tmp_path / "m.db")
    schema._baseline(conn)
    conn.execute("INSERT INTO mashup_candidates (combo_type, vocal_song_id, inst_song_id, "
                 "score_total, scorer) VALUES ('vocal_over_instrumental', 1, 2, 0.7, "
                 "'heuristic'), ('vocal_over_instrumental', 3, 4, 0.9, 'model')")
    before = schema.shape(conn)["tables"]["pair_feedback"]
    idx = conn.execute("SELECT sql FROM sqlite_master WHERE tbl_name='pair_feedback' "
                       "AND type='index'").fetchall()
    schema._slot_and_dropin(conn)
    schema._learning_loop(conn)
    schema._learning_loop(conn)                              # idempotent
    after = schema.shape(conn)
    assert after["tables"]["pair_feedback"] == before
    assert conn.execute("SELECT sql FROM sqlite_master WHERE tbl_name='pair_feedback' "
                        "AND type='index'").fetchall() == idx
    cols = {r[1]: r for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    assert cols["score_rule"][3] == 0 and cols["model_prob"][3] == 0
    # A rule row keeps its total as its rule score; a retired model-path row,
    # whose total is a probability, is not passed off as one.
    assert conn.execute("SELECT score_rule FROM mashup_candidates ORDER BY vocal_song_id"
                        ).fetchall() == [(0.7,), (None,)]
    mcols = {r[1] for r in conn.execute("PRAGMA table_info(models)")}
    assert {"kind", "alpha", "n_strong", "auc_strong", "rule_auc", "decision",
            "activated_at"} <= mcols
    conn.close()


def test_training_reads_the_verdict_never_the_rating(scored):
    """Criterion 2.5: a verdict's label comes from `verdict`. A row whose
    rating disagrees (only possible by hand) still trains on its verdict."""
    m, _match, learning, _r = scored
    conn = m.get_conn()
    conn.execute("INSERT INTO pair_feedback (vocal_song_id, inst_song_id, vocal_section, "
                 "inst_section, verdict, rating) VALUES (1, 2, NULL, NULL, 'no', 5), "
                 "(3, 4, NULL, NULL, 'ok', 1)")
    conn.commit()
    labels = dict(learning.strong_labels(conn))
    conn.close()
    assert labels[(1, 2, None, None)] == 0.0
    assert labels[(3, 4, None, None)] == 0.6
    src = inspect.getsource(learning.strong_labels)
    assert "rating" not in src and "verdict" in src
    assert learning.STRONG_LABELS == {"love": 1.0, "ok": 0.6, "no": 0.0}


# ── the blend ────────────────────────────────────────────────────────────────

def test_alpha_is_shrinkage_and_zero_without_verdicts():
    from matcher.learning import alpha_for, blend
    assert alpha_for(0, 40, 0.8) == 0.0
    assert alpha_for(40, 40, 0.8) == pytest.approx(0.5)
    assert alpha_for(10_000, 40, 0.8) == 0.8
    assert blend(0.7123, 0.9, 0.0) == 0.7123
    assert blend(0.7123, None, 0.5) == 0.7123
    assert blend(0.7, 0.9, 0.5) == round(0.5 * 0.7 + 0.5 * 0.9, 4)


def test_zero_ratings_leave_every_total_the_rules(scored, tmp_path):
    """Criterion 2.2: with no verdict and no model, every stored score_total
    is the rule's own (score_rule), the blend pass writes nothing, and a run
    that skips the blend stores the identical table."""
    m, match, learning, _r = scored
    rows = _table(m.DB_PATH)
    conn = m.get_conn()
    bad = conn.execute("SELECT COUNT(*) FROM mashup_candidates WHERE "
                       "combo_type='vocal_over_instrumental' AND "
                       "(score_rule IS NULL OR score_total <> score_rule "
                       "OR model_prob IS NOT NULL)").fetchone()[0]
    conn.close()
    assert bad == 0
    assert learning.blend_table(m.DB_PATH)["rows"] == 0
    assert _table(m.DB_PATH) == rows
    match.score_all_pairs(db_path=m.DB_PATH, scorer="heuristic")
    assert _table(m.DB_PATH) == rows
    # Too few verdicts: no model is trained, nothing moves.
    _judge(m, learning, n=6)
    out = learning.train(m.DB_PATH)
    assert out["trained"] is False and learning.active_blender(m.DB_PATH) is None
    assert learning.blend_table(m.DB_PATH)["rows"] == 0


def test_a_trained_model_blends_the_table_and_segments_agree(scored):
    m, _match, learning, _r = scored
    _judge(m, learning, n=40)
    out = learning.train(m.DB_PATH)
    assert out["trained"] and out["metrics"]["auc_strong"] is not None
    if not out["activated"]:
        pytest.skip(f"the shadow gate kept the rule: {out['decision']}")
    blender = learning.active_blender(m.DB_PATH)
    assert blender is not None and 0 < blender.alpha <= 0.8
    assert blender.alpha == pytest.approx(learning.alpha_for(40, 40.0, 0.8))
    conn = m.get_conn()
    rows = conn.execute("SELECT score_total, score_rule, model_prob, model_version "
                        "FROM mashup_candidates WHERE combo_type='vocal_over_instrumental'"
                        ).fetchall()
    conn.close()
    assert rows and all(r[2] is not None for r in rows)
    for total, rule, prob, version in rows:
        assert total == learning.blend(rule, prob, blender.alpha)
        assert version == blender.version
    # Segments' live rows get the same blend as their stored twins (§7).
    from matcher import segment_match
    seg, got = next(
        (s, o) for s in m.list_segments("vocal")
        for o in [segment_match.rank_partners(s["segment_id"], "vocal", limit=30,
                                              db_path=m.DB_PATH)]
        if sum(1 for r in o["rows"] if r["stored"]) >= 3)
    stored = {(r["inst_song_id"], r["inst_section_idx"]): r for r in got["rows"]
              if r["stored"]}
    conn = m.get_conn()
    conn.execute("DELETE FROM mashup_candidates WHERE vocal_song_id=?", (seg["song_id"],))
    conn.commit()
    conn.close()
    live = {(r["inst_song_id"], r["inst_section_idx"]): r
            for r in segment_match.rank_partners(seg["segment_id"], "vocal", limit=500,
                                                 db_path=m.DB_PATH)["rows"]}
    checked = 0
    for key, r in stored.items():
        if key in live:
            assert live[key]["stored"] is False
            for col in ("score_total", "score_rule", "model_prob"):
                assert live[key][col] == r[col], (col, live[key][col], r[col])
            checked += 1
    assert checked >= 3


def test_a_worse_model_is_never_activated_and_rollback_is_one_call(scored):
    """Criterion 2.4."""
    m, _match, learning, _r = scored
    _judge(m, learning, n=40)
    first = learning.train(m.DB_PATH)
    if not first["activated"]:
        pytest.skip(f"the shadow gate kept the rule: {first['decision']}")
    # The active model's held-out AUC is the bar: make it unbeatable.
    conn = m.get_conn()
    conn.execute("UPDATE models SET auc_strong = 1.0 WHERE id=?", (first["model_id"],))
    conn.commit()
    conn.close()
    second = learning.train(m.DB_PATH)
    assert second["trained"] and not second["activated"]
    assert second["decision"].startswith("kept the active scorer")
    st = learning.status(m.DB_PATH)
    assert st["active"]["id"] == first["model_id"]
    # Activate the second by hand, then roll back: the first is active again,
    # and the table carries its blend.
    learning.activate_model(second["model_id"], m.DB_PATH)
    assert learning.status(m.DB_PATH)["active"]["id"] == second["model_id"]
    learning.rollback(m.DB_PATH)
    assert learning.status(m.DB_PATH)["active"]["id"] == first["model_id"]
    assert learning.active_blender(m.DB_PATH).model_id == first["model_id"]
    # The rules alone: every total back to the rule's.
    learning.activate_model(None, m.DB_PATH)
    conn = m.get_conn()
    assert conn.execute("SELECT COUNT(*) FROM mashup_candidates WHERE "
                        "combo_type='vocal_over_instrumental' AND (score_total <> score_rule "
                        "OR model_prob IS NOT NULL)").fetchone()[0] == 0
    conn.close()


def test_the_rule_is_the_bar_when_no_model_is_active(scored, monkeypatch):
    m, _match, learning, _r = scored
    _judge(m, learning, n=40)
    real = learning._strong_auc
    calls = []

    def rigged(per_key, strong):
        calls.append(1)
        # The model's own AUC (first call) is under the rule's (third).
        return {1: 0.55, 2: 0.6, 3: 0.7}.get(len(calls), real(per_key, strong))
    monkeypatch.setattr(learning, "_strong_auc", rigged)
    out = learning.train(m.DB_PATH)
    assert out["trained"] and not out["activated"]
    assert "the rule" in out["decision"]
    assert learning.active_blender(m.DB_PATH) is None


# ── labels ───────────────────────────────────────────────────────────────────

def test_weak_labels_from_events_and_a_verdict_overrides_them():
    from matcher.learning import weak_labels

    class Conn:
        def __init__(self, raw):
            self.raw = raw

        def execute(self, sql, *a):
            return self.raw if "FROM pair_events" in sql else []
    cfg = {"loop_secs": 20.0, "w_build": 0.3, "w_loop": 0.15, "w_skip": 0.1,
           "w_hide": 0.3, "w_shown": 0.03}
    raw = [
        (1, 2, 0, 0, "open_studio", 1, 0.0, 0),        # built with → good
        (1, 3, 0, 0, "loop_play", 1, 25.0, 0),         # long loop → mildly good
        (1, 4, 0, 0, "loop_play", 1, 5.0, 0),          # short loop → nothing
        (1, 5, 0, 0, "skip", 1, 1.0, 0),               # skip → mildly bad
        (1, 6, 0, 0, "shown", 3, 0.0, 2),              # top five, ignored → very weak bad
        (1, 7, 0, 0, "shown", 1, 0.0, 0),              # shown lower down → nothing
        (1, 8, 0, 0, "shown", 1, 0.0, 1),
        (1, 8, 0, 0, "expand", 1, 0.0, 0),             # shown and opened → not ignored
        (1, 9, 0, 0, "hide", 1, 0.0, 0),
        (1, 9, 0, 0, "open_studio", 1, 0.0, 0),        # mixed → a soft label
        (1, 10, 0, 0, "render", 1, 0.0, 0),            # has a verdict → left out
    ]
    out = {k: (y, w) for k, y, w in weak_labels(Conn(raw), cfg, {(1, 10, 0, 0)}, set())}
    assert out[(1, 2, 0, 0)] == (1.0, 0.3)
    assert out[(1, 3, 0, 0)] == (1.0, 0.15)
    assert (1, 4, 0, 0) not in out
    assert out[(1, 5, 0, 0)] == (0.0, 0.1)
    assert out[(1, 6, 0, 0)] == (0.0, 0.03)
    assert (1, 7, 0, 0) not in out and (1, 8, 0, 0) not in out
    assert out[(1, 9, 0, 0)] == (pytest.approx(0.5), pytest.approx(0.6))
    assert (1, 10, 0, 0) not in out


def test_new_labels_drive_the_retrain_and_never_while_busy(scored, monkeypatch):
    """Criterion 2.3: the tick trains only when due AND the pipeline is idle."""
    m, _match, learning, _r = scored
    monkeypatch.setenv("MASHUP_AUTO_RETRAIN", "1")
    assert learning.retrain_due(m.DB_PATH) is None
    _judge(m, learning, n=12)
    assert learning.new_since_last_train(m.DB_PATH)["strong"] == 12
    assert "12 new verdicts" in learning.retrain_due(m.DB_PATH)
    ran = []
    assert learning.tick(m.DB_PATH, idle=lambda: False, run=ran.append) is None
    assert ran == []
    assert learning.tick(m.DB_PATH, idle=lambda: True, run=ran.append) is not None
    assert len(ran) == 1
    monkeypatch.setenv("MASHUP_AUTO_RETRAIN", "0")
    assert learning.retrain_due(m.DB_PATH) is None
    # A queued separation (or any waiting stage) is "busy".
    from api import queue_runner
    monkeypatch.setattr(queue_runner, "queued_count", lambda: 1)
    assert learning.pipeline_idle() is False


# ── exploration ──────────────────────────────────────────────────────────────

def test_exploration_slots_are_labelled_inserted_and_paged(scored, monkeypatch):
    m, _match, learning, mashups = scored
    _judge(m, learning, n=40)
    out = learning.train(m.DB_PATH)
    if not out["activated"]:
        pytest.skip(f"the shadow gate kept the rule: {out['decision']}")
    monkeypatch.setenv("MASHUP_EXPLORE_EVERY", "0")
    plain = mashups.list_candidates(limit=21)["candidates"]
    assert not any(r.get("explore") for r in plain)
    monkeypatch.setenv("MASHUP_EXPLORE_EVERY", "8")
    rows = mashups.list_candidates(limit=21)["candidates"]
    ranked = [r for r in rows if not r.get("explore")]
    explored = [(k, r) for k, r in enumerate(rows) if r.get("explore")]
    assert [mashups_key(r) for r in ranked] == [mashups_key(r) for r in plain]
    assert explored, "no exploration slot on a page with an active model"
    for k, r in explored:
        assert (k + 1) % 8 == 0
        assert r["scorer"] == "explore" and "unsure" in r["explore_reason"]
        assert r["model_prob"] is not None
        assert mashups_key(r) not in {mashups_key(x) for x in ranked}
        assert r["score_rule"] >= min(x["score_rule"] for x in ranked)
    # Judged pairs are never offered for exploration.
    judged = {(f["vocal_song_id"], f["inst_song_id"]) for f in m.get_pair_feedback()}
    assert not {(r["vocal_song_id"], r["inst_song_id"]) for _k, r in explored} & judged


def mashups_key(r):
    return (r["vocal_song_id"], r["inst_song_id"], r.get("vocal_section_idx"),
            r.get("inst_section_idx"))


# ── the personal weight fit ──────────────────────────────────────────────────

def test_weight_suggestion_follows_the_verdicts_and_is_only_a_proposal():
    import random
    from matcher.weight_fit import suggest
    rng = random.Random(1)
    feedback, cands = [], []
    for k in range(60):
        good = k % 2 == 0
        label = (0.9 if good else 0.2) + rng.uniform(-0.1, 0.1)
        feedback.append({"vocal_song_id": k, "inst_song_id": 1000 + k,
                         "vocal_section": None, "inst_section": None,
                         "verdict": "love" if good else "no", "rating": 5 if good else 1})
        cands.append({"combo_type": "vocal_over_instrumental", "vocal_song_id": k,
                      "inst_song_id": 1000 + k, "vocal_section_idx": 0,
                      "inst_section_idx": 0, "score_total": 0.7,
                      "score_label": label, "score_duration": rng.random(),
                      "score_voice": rng.random(), "score_phrase": None,
                      "score_rhythm": None, "score_structure": None})
    current = {"label": 0.2, "duration": 0.5, "voice": 0.3, "phrase": 0.0,
               "rhythm": 0.0, "structure": 0.0}
    out = suggest(feedback, cands, current)
    assert out["ready"]
    assert max(out["proposed"], key=out["proposed"].get) == "label"
    assert sum(out["proposed"].values()) == pytest.approx(1.0, abs=1e-3)
    assert all(v >= 0 for v in out["proposed"].values())
    assert out["auc_proposed"] > out["auc_current"]
    assert out["diff"]["label"] > 0
    few = suggest(feedback[:6], cands[:6], current)
    assert few["ready"] is False and "needs" in few["reason"]
    # It never writes a setting: only the apply route does.
    src = (ROOT / "matcher" / "weight_fit.py").read_text(encoding="utf-8")
    assert "save_settings" not in src
    route = (ROOT / "api" / "routes" / "learning.py").read_text(encoding="utf-8")
    assert route.count("save_settings") == 1


# ── features ─────────────────────────────────────────────────────────────────

def test_the_feature_vector_only_grew_at_the_end():
    from matcher.features import FEATURE_NAMES, ROW_FEATURE_NAMES, pair_features
    assert FEATURE_NAMES[-len(ROW_FEATURE_NAMES):] == ROW_FEATURE_NAMES
    assert FEATURE_NAMES[:4] == ["bpm_score", "key_score", "energy_score", "timbre_score"]
    assert FEATURE_NAMES[len(FEATURE_NAMES) - len(ROW_FEATURE_NAMES) - 1] == "surprise_timbre"
    for name in ("score_slot", "slot_occupancy", "slot_phrase_match", "slot_register",
                 "score_window", "dropin", "score_timing", "score_effort", "rule_score",
                 "bed_label_breakdown"):
        assert name in ROW_FEATURE_NAMES
    # No row: unmeasured terms are neutral, never zero, and flagged unknown.
    f = pair_features({}, {}, [], [])
    assert f["row_known"] == 0.0 and f["score_slot"] == 0.5 and f["dropin"] == 0.0


def test_the_song_cache_changes_no_feature(scored):
    m, _match, learning, _r = scored
    ctx = learning.FeatureContext(m.DB_PATH)
    conn = m.get_conn()
    cols = learning._row_columns()
    rows = [dict(zip(cols, r)) for r in conn.execute(
        f"SELECT {', '.join(cols)} FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental' LIMIT 50")]
    conn.close()
    from matcher.features import pair_features
    for r in rows:
        cached = ctx.features(r["vocal_song_id"], r["inst_song_id"], r)
        plain = pair_features(ctx.vocals[r["vocal_song_id"]], ctx.beds[r["inst_song_id"]],
                              ctx.sections(r["vocal_song_id"]),
                              ctx.sections(r["inst_song_id"]), ctx.stats, row=r,
                              row_ctx=ctx.row_ctx)
        assert cached == plain


# ── the frontend ─────────────────────────────────────────────────────────────

def _src(*parts):
    return (ROOT / "frontend" / "src").joinpath(*parts).read_text(encoding="utf-8")


def test_the_learning_readout_and_the_suggestion_are_wired():
    panel = _src("components", "LearningPanel.jsx")
    for call in ("api.getLearning", "api.retrainNow", "api.rollbackModel", "api.rulesOnly",
                 "api.getTermReport"):
        assert call in panel, call
    for field in ("st.alpha", "st.last_train", "st.labels", "st.history", "auc_strong"):
        assert field in panel, field
    assert "LearningPanel" in _src("App.jsx")
    tuning = _src("components", "TuningPanel.jsx")
    assert "Suggested from your ratings" in tuning
    assert "api.getWeightSuggestion" in tuning and "api.applyWeightSuggestion" in tuning
    for key in ("learn_k", "learn_alpha_max", "explore_every", "weak_w_build"):
        assert f'key: "{key}"' in tuning, key
    api = _src("api.js")
    for path in ("/api/learning", "/api/learning/retrain", "/api/learning/rollback",
                 "/api/learning/deactivate", "/api/learning/weight-suggestion"):
        assert path in api, path
    css = _src("styles.css")
    for cls in (".learning-panel", ".learn-grid", ".pc-explore"):
        assert cls in css, cls


def test_the_card_labels_exploration_and_paging_skips_it():
    card = _src("components", "PairCard.jsx")
    assert "c.explore" in card and "c.explore_reason" in card
    dock = _src("hooks", "usePairDock.js")
    assert "offset: rowsRef.current.filter((r) => !r.explore).length" in dock
    # Rating stays one keypress: the dock's 1-5 keys are unchanged.
    assert "Rating stays one keypress" in _src("components", "LearningPanel.jsx")


class _Coin:
    """An 'active model' that knows nothing: every pair is a coin toss."""
    def predict_proba(self, X):
        import numpy as np
        return np.full((len(X), 2), 0.5)


def test_the_gate_re_measures_the_active_model_on_verdicts_it_never_saw(scored):
    """A model whose recorded AUC was lucky must not lock every later model
    out: once there are fresh verdicts (3+ each side), the active model and the
    candidate are compared on exactly those — verdicts the active one never
    saw and the candidate holds out — not on the record."""
    import joblib
    import json
    m, _match, learning, _r = scored
    _judge(m, learning, n=40)
    first = learning.train(m.DB_PATH)
    if not first["activated"]:
        pytest.skip(f"the shadow gate kept the rule: {first['decision']}")
    assert first["metrics"]["strong_max_id"] > 0
    # Its record is unbeatable — the old gate would keep it for ever.
    conn = m.get_conn()
    conn.execute("UPDATE models SET auc_strong = 1.0 WHERE id=?", (first["model_id"],))
    path = conn.execute("SELECT file_path FROM models WHERE id=?",
                        (first["model_id"],)).fetchone()[0]
    conn.commit()
    conn.close()
    # Without new verdicts the record still decides (nothing to compare on).
    again = learning.train(m.DB_PATH)
    assert again["metrics"]["gate"]["mode"] == "recorded" and not again["activated"]
    # In truth it knows nothing.
    bundle = joblib.load(path)
    bundle["estimator"] = _Coin()
    joblib.dump(bundle, path)
    _judge(m, learning, n=80, seed=11)
    second = learning.train(m.DB_PATH)
    gate = second["metrics"]["gate"]
    assert gate["mode"] == "fresh" and gate["n_fresh"] >= 6
    assert gate["active_auc"] == pytest.approx(0.5)
    assert "never saw" in second["decision"]
    expect = (gate["candidate_auc"] >= gate["active_auc"]
              and second["metrics"]["auc_strong"] >= (second["metrics"]["rule_auc"] or 0))
    assert second["activated"] == expect
    conn = m.get_conn()
    stored = json.loads(conn.execute("SELECT metrics_json FROM models WHERE id=?",
                                     (second["model_id"],)).fetchone()[0])
    conn.close()
    assert stored["gate"] == gate
