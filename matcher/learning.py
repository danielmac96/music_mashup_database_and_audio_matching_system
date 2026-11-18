"""matcher/learning.py — the learning loop (readme §5.8).

The ranking learns from what you say about a pair (a verdict) and from what
you do with it (pair events), retrains on its own, and never ranks worse than
the rules:

  * **Labels.** Strong: every pair_feedback verdict, unchanged — love = 1,
    ok = 0.6, no = 0 (training reads `verdict`, never `rating`, readme §7).
    Weak: pair_events — building with a pair (open_studio, render, fl_export,
    export) is good, a loop_play of `loop_positive_secs` or more is mildly
    good, a skip or a hide mildly bad, and being shown in the top five with
    nothing else done a very weak bad. A verdict on the same pairing
    overrides every event about it. Documented mix pairs stay positives, as
    they always were.
  * **The model.** `pair_features` (matcher/features.py, one function for
    train and serve) on each label's scored row → a HistGradientBoosting
    classifier with sample weights (a soft label y counts as y of a positive
    and 1 − y of a negative), cross-validated with GroupKFold by vocal song,
    calibrated. A logistic regression is fitted beside it for comparison.
  * **The shrinkage blend.** score_total = (1 − α)·rule + α·prob, with
    α = n_strong / (n_strong + k) capped at `learn_alpha_max`, n_strong the
    verdicts the active model was trained on. α = 0 — no model, or no
    verdicts behind it — leaves every total the rule's, to the bit. The
    rule picks the rows (the gate, the pruning, the cap and each track's
    reserve are untouched); the blend re-orders them. `blend_table` is the
    one writer of a blended total; Segments blends its live rows with the
    same `Blender`.
  * **Shadow evaluation.** A trained model is activated only when its
    held-out AUC on your verdicts is at least the active model's (as measured
    when that one was trained), and at least the rule's on the same verdicts.
    Every model records its feature names, counts and metrics; rollback
    re-activates the one active before it.
  * **Auto-retrain.** `tick()` — debounced, run by the server every minute —
    retrains when `retrain_min_strong` new verdicts or `retrain_min_weak` new
    non-`shown` events have arrived since the last train, and only while the
    pipeline queues are empty (never beside a separation).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

STRONG_LABELS = {"love": 1.0, "ok": 0.6, "no": 0.0}
BUILD_EVENTS = ("open_studio", "render", "fl_export", "export")
TOP_SHOWN = 5                 # `shown` counts as a weak negative only at ranks 1-5
MIN_STRONG = 10               # verdicts before a model is trained at all
MIN_PER_CLASS = 3             # ... with at least this many good and bad
CV_SPLITS = 5
SAMPLED_NEG_RATIO = 5         # sampled negatives per documented mix positive
MODEL_NAME = "ranker"
_PREF = "learning_state"

_TRAIN_LOCK = threading.Lock()


def _db(db_path=None) -> Path:
    from database import models
    return Path(db_path) if db_path is not None else models.DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def settings() -> Dict:
    """The loop's knobs, read once per call site (never per row)."""
    from config import current_auto_retrain, current_float, current_int
    return {
        "k": current_float("learn_k"),
        "alpha_max": current_float("learn_alpha_max"),
        "w_build": current_float("weak_w_build"),
        "w_loop": current_float("weak_w_loop"),
        "loop_secs": current_float("loop_positive_secs"),
        "w_skip": current_float("weak_w_skip"),
        "w_hide": current_float("weak_w_hide"),
        "w_shown": current_float("weak_w_shown"),
        "min_strong": current_int("retrain_min_strong"),
        "min_weak": current_int("retrain_min_weak"),
        "explore_every": current_int("explore_every"),
        "auto": current_auto_retrain(),
    }


def alpha_for(n_strong: int, k: float, alpha_max: float) -> float:
    """α = n / (n + k), capped. 0 without a verdict."""
    n = max(0, int(n_strong or 0))
    if n == 0:
        return 0.0
    return float(min(alpha_max, n / (n + max(k, 1e-9))))


def blend(rule: float, prob: Optional[float], alpha: float) -> float:
    """The one formula: (1 − α)·rule + α·prob, at the stored precision.
    α = 0 or no probability returns the rule's own value untouched."""
    if prob is None or not alpha:
        return rule
    return round((1.0 - alpha) * float(rule) + alpha * float(prob), 4)


# ── Labels ───────────────────────────────────────────────────────────────────

Key = Tuple[int, int, Optional[int], Optional[int]]


def strong_labels(conn) -> List[Tuple[Key, float]]:
    """Every verdict as (four-id key, label). A judgement whose sections could
    not be carried onto a re-cut structure judges the song pair only."""
    rows = conn.execute(
        """SELECT vocal_song_id, inst_song_id,
                  CASE WHEN COALESCE(sections_stale, 0) = 1 THEN NULL
                       ELSE vocal_section END,
                  CASE WHEN COALESCE(sections_stale, 0) = 1 THEN NULL
                       ELSE inst_section END,
                  verdict
           FROM pair_feedback
           WHERE vocal_song_id IS NOT NULL AND inst_song_id IS NOT NULL
             AND vocal_song_id <> inst_song_id""").fetchall()
    return [((r[0], r[1], r[2], r[3]), STRONG_LABELS[r[4]])
            for r in rows if r[4] in STRONG_LABELS]


def weak_labels(conn, cfg: Dict, overridden: set, overridden_pairs: set
                ) -> List[Tuple[Key, float, float]]:
    """(key, soft label, weight) per pairing from its events, raw and folded.

    Per pairing the good weights (built with, looped long enough) and the bad
    ones (skipped, hidden, shown in the top five and left alone) are summed:
    label = good / (good + bad), weight = good + bad. A pairing with a verdict
    (or whose song pair has a section-less verdict) is left out."""
    agg: Dict[Key, Dict[str, float]] = {}

    def add(key, event, n, vmax, n_top5):
        d = agg.setdefault(key, {})
        d[event] = d.get(event, 0) + (n or 0)
        d[event + ":max"] = max(d.get(event + ":max", 0.0), vmax or 0.0)
        d[event + ":top5"] = d.get(event + ":top5", 0) + (n_top5 or 0)

    for r in conn.execute(
            f"""SELECT vocal_song_id, inst_song_id, vocal_section, inst_section,
                       event, COUNT(*), MAX(COALESCE(value, 0)),
                       SUM(CASE WHEN rank_shown BETWEEN 1 AND {TOP_SHOWN}
                                THEN 1 ELSE 0 END)
                FROM pair_events GROUP BY 1, 2, 3, 4, 5"""):
        add((r[0], r[1], r[2], r[3]), r[4], r[5], r[6], r[7])
    # Folded rows keep a count, the number shown in the top five and the sum
    # of values — a loop's mean length stands in for its longest.
    for r in conn.execute(
            """SELECT vocal_song_id, inst_song_id, vocal_section, inst_section,
                      event, SUM(n), SUM(value_sum), SUM(n_top5)
               FROM pair_event_summary GROUP BY 1, 2, 3, 4, 5"""):
        n = r[5] or 0
        add((r[0], r[1], r[2], r[3]), r[4], n, (r[6] or 0.0) / n if n else 0.0, r[7])

    out = []
    for key, d in agg.items():
        if key[0] == key[1] or key in overridden or (key[0], key[1]) in overridden_pairs:
            continue
        good = bad = 0.0
        built = any(d.get(e) for e in BUILD_EVENTS)
        looped = d.get("loop_play:max", 0.0) >= cfg["loop_secs"]
        if built:
            good += cfg["w_build"]
        if looped:
            good += cfg["w_loop"]
        if d.get("skip") and not looped:
            bad += cfg["w_skip"]
        if d.get("hide"):
            bad += cfg["w_hide"]
        touched = any(d.get(e) for e in ("loop_play", "skip", "expand", "open_studio",
                                         "render", "fl_export", "export", "hide",
                                         "compare", "rate"))
        if d.get("shown:top5") and not touched:
            bad += cfg["w_shown"]
        if good + bad <= 0:
            continue
        out.append((key, good / (good + bad), good + bad))
    return out


# ── Rows and features ────────────────────────────────────────────────────────

_ROW_COLS = None


def _row_columns() -> Tuple[str, ...]:
    """The stored row columns pair_features and the blend read."""
    global _ROW_COLS
    if _ROW_COLS is None:
        _ROW_COLS = (
            "id", "vocal_song_id", "inst_song_id", "vocal_section_idx",
            "inst_section_idx", "vocal_camelot", "inst_camelot", "score_total",
            "score_rule", "model_prob", "score_key", "score_section", "score_label",
            "score_duration", "score_voice", "score_phrase", "score_room_section",
            "score_coverage", "score_energy_match", "score_effort",
            "harmonic_shift", "harmonic_confidence", "bass_clash", "score_timing",
            "timing_ok", "timing_err_max_ms", "score_slot", "slot_occupancy",
            "slot_phrase_match", "slot_register", "score_window", "dropin_mask",
            "camelot_shift", "inst_section_label", "vocal_section_label", "scorer")
    return _ROW_COLS


class FeatureContext:
    """What pair_features reads beside a row, loaded once: each song's
    feature row with the scoring tempo (match._full_by_song, the one builder),
    the library stats, the sections of the engine in use, and what the
    Drop-in verdict reads (row_ctx)."""

    def __init__(self, db_path=None, song_ids: Optional[Iterable[int]] = None):
        from config import current_segmentation_engine
        from database import models
        from matcher.compat import slot_floor
        from matcher.match import _full_by_song, _with_full_bpm, get_library_stats
        from matcher.recipe import harmony_unsure_below
        db = _db(db_path)
        self.db = db
        cols = models.FEATURES_SCORING_COLUMNS
        ids = sorted(set(song_ids)) if song_ids is not None else None
        full = _full_by_song(models.get_all_features(
            "full", columns=cols, song_ids=ids, db_path=db), db_path=db)
        self.vocals = {f["song_id"]: _with_full_bpm(f, full) for f in
                       models.get_all_features("vocals", columns=cols,
                                               song_ids=ids, db_path=db)}
        self.beds = {f["song_id"]: _with_full_bpm(f, full) for f in
                     models.get_all_features("instrumental", columns=cols,
                                             song_ids=ids, db_path=db)}
        self.stats = get_library_stats(db_path=db)
        self.engine = current_segmentation_engine()
        self._sections: Dict[int, list] = {}
        # The song-level half of pair_features, per song pair (bounded).
        self._song_cache: Dict[tuple, dict] = {}
        self.row_ctx = {"slot_floor": slot_floor(db_path=db),
                        "unsure_below": harmony_unsure_below(db_path=db)}

    def load_sections(self, song_ids: Iterable[int]) -> None:
        from database import models
        want = set(song_ids) - set(self._sections)
        if want:
            got = models.sections_for_songs(want, db_path=self.db, engine=self.engine)
            for sid in want:
                self._sections[sid] = got.get(sid, [])

    def sections(self, song_id: int) -> list:
        if song_id not in self._sections:
            self.load_sections([song_id])
        return self._sections[song_id]

    def features(self, vocal_id: int, inst_id: int, row: Optional[dict],
                 v_sec: Optional[int] = None, i_sec: Optional[int] = None
                 ) -> Optional[dict]:
        from matcher.features import pair_features
        top, bed = self.vocals.get(vocal_id), self.beds.get(inst_id)
        if top is None or bed is None:
            return None
        if len(self._song_cache) > 50_000:
            self._song_cache.clear()
        return pair_features(top, bed, self.sections(vocal_id), self.sections(inst_id),
                             self.stats, top_section_idx=v_sec, bed_section_idx=i_sec,
                             row=row, row_ctx=self.row_ctx, song_cache=self._song_cache)


def _stored_rows(conn, song_pairs: set) -> Dict[Tuple[int, int], List[dict]]:
    """Every stored vocal-over-bed row of the given song pairs, in one query
    (a temporary table of the pairs, joined)."""
    out: Dict[Tuple[int, int], List[dict]] = {}
    if not song_pairs:
        return out
    cols = _row_columns()
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _learn_pairs "
                 "(v INTEGER, i INTEGER, PRIMARY KEY (v, i)) WITHOUT ROWID")
    conn.execute("DELETE FROM _learn_pairs")
    conn.executemany("INSERT OR IGNORE INTO _learn_pairs VALUES (?, ?)", sorted(song_pairs))
    # CROSS JOIN fixes the loop order: each pair probes the candidates'
    # (combo_type, vocal, inst, …) unique index, never a scan per pair.
    for r in conn.execute(
            f"""SELECT {', '.join('mc.' + c for c in cols)}
                FROM _learn_pairs p CROSS JOIN mashup_candidates mc
                WHERE mc.combo_type = 'vocal_over_instrumental'
                  AND mc.vocal_song_id = p.v AND mc.inst_song_id = p.i"""):
        d = dict(zip(cols, r))
        out.setdefault((d["vocal_song_id"], d["inst_song_id"]), []).append(d)
    conn.execute("DELETE FROM _learn_pairs")
    return out


def _row_for(key: Key, by_pair: Dict) -> Optional[dict]:
    """The stored row a label names: that pairing, or for a section-less key
    the song pair's best row by the rule."""
    rows = by_pair.get((key[0], key[1])) or []
    if key[2] is None and key[3] is None:
        return max(rows, key=lambda r: (r.get("score_rule") if r.get("score_rule")
                                        is not None else r.get("score_total") or 0.0),
                   default=None)
    for r in rows:
        if r["vocal_section_idx"] == key[2] and r["inst_section_idx"] == key[3]:
            return r
    return None


# ── Training ─────────────────────────────────────────────────────────────────

def training_set(db_path=None, cfg: Optional[Dict] = None) -> Dict:
    """Labels → (X, y, w, groups) plus which rows are verdicts.

    Each label becomes up to two rows: y·w as a positive and (1 − y)·w as a
    negative, so a soft label ("ok" = 0.6, a mixed bag of events) is learned
    as exactly that share."""
    from database.models import get_conn
    from matcher.features import FEATURE_NAMES, _documented_pairs, features_to_row
    cfg = cfg or settings()
    db = _db(db_path)
    conn = get_conn(db)
    try:
        strong = strong_labels(conn)
        overridden = {k for k, _y in strong}
        overridden_pairs = {(k[0], k[1]) for k, _y in strong if k[2] is None and k[3] is None}
        weak = weak_labels(conn, cfg, overridden, overridden_pairs)
        doc_pos, doc_all = _documented_pairs(conn)
        judged_pairs = {(k[0], k[1]) for k, _y in strong}
        doc = [((v, i, None, None), 1.0, 1.0) for v, i, _m in doc_pos
               if (v, i) not in judged_pairs]
        pairs = ({(k[0], k[1]) for k, _y in strong} | {(k[0], k[1]) for k, _y, _w in weak}
                 | {(k[0], k[1]) for k, _y, _w in doc})
        by_pair = _stored_rows(conn, pairs)
        sampled = []
        if doc:
            # Documented positives come with sampled negatives, as they always
            # have — undocumented, unjudged stored rows, at a weak weight.
            n_want = SAMPLED_NEG_RATIO * len(doc)
            skip = pairs | doc_all
            cand = conn.execute(
                "SELECT vocal_song_id, inst_song_id, vocal_section_idx, inst_section_idx "
                "FROM mashup_candidates WHERE combo_type='vocal_over_instrumental' "
                "ORDER BY id").fetchall()
            rng = np.random.default_rng(42)
            cand = [tuple(c) for c in cand if (c[0], c[1]) not in skip]
            if cand:
                pick = rng.choice(len(cand), size=min(n_want, len(cand)), replace=False)
                sampled = [(cand[int(j)], 0.0, cfg["w_skip"]) for j in sorted(pick)]
                by_pair.update(_stored_rows(conn, {(k[0], k[1]) for k, _y, _w in sampled}))
    finally:
        conn.close()

    labelled: List[Tuple[Key, float, float, str]] = (
        [(k, y, 1.0, "strong") for k, y in strong]
        + [(k, y, w, "weak") for k, y, w in weak]
        + [(k, y, w, "doc") for k, y, w in doc]
        + [(k, y, w, "sampled") for k, y, w in sampled])
    songs = {x for k, *_ in labelled for x in (k[0], k[1])}
    ctx = FeatureContext(db, song_ids=songs)
    ctx.load_sections(songs)

    X, y, w, groups, src, key_of, rule = [], [], [], [], [], [], []
    n_by = {"strong": 0, "weak": 0, "doc": 0, "sampled": 0}
    for key, label, weight, kind in labelled:
        row = _row_for(key, by_pair)
        if row is None and kind != "strong":
            continue              # only a verdict is worth learning without its row
        feats = ctx.features(key[0], key[1], row,
                             v_sec=key[2] if row is None else None,
                             i_sec=key[3] if row is None else None)
        if feats is None:
            continue
        vec = features_to_row(feats)
        n_by[kind] += 1
        r_score = feats["rule_score"]
        for target, share in ((1, label), (0, 1.0 - label)):
            if share * weight <= 0:
                continue
            X.append(vec)
            y.append(target)
            w.append(share * weight)
            groups.append(key[0])
            src.append(kind)
            key_of.append(key)
            rule.append(r_score)
    return {"X": np.asarray(X, dtype=np.float64), "y": np.asarray(y, dtype=np.int64),
            "w": np.asarray(w, dtype=np.float64), "groups": np.asarray(groups),
            "src": np.asarray(src), "keys": key_of, "rule": np.asarray(rule),
            "feature_names": list(FEATURE_NAMES), "counts": n_by,
            "strong_labels": dict(strong)}


try:
    from sklearn.base import BaseEstimator, ClassifierMixin
except Exception:  # noqa: BLE001 — scikit-learn absent: nothing here trains
    BaseEstimator = ClassifierMixin = object


class QuantizedHGB(ClassifierMixin, BaseEstimator):
    """HistGradientBoostingClassifier on columns cut to at most 255 values.

    The gradient booster bins every column into 255 bins anyway; given sample
    weights, scikit-learn finds those bins with a weighted percentile, which
    took ~9 s of a 10 s fit on 20,000 × 69 (criterion 2.3's budget is 60 s for
    the whole retrain). Cutting each column at its own unweighted quantiles
    first leaves at most 255 distinct values, which the booster bins by
    their midpoints in milliseconds. The cut is fitted here and applied at
    predict time, so train and serve see the same columns."""

    def __init__(self, max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                 min_samples_leaf=10, l2_regularization=1.0, random_state=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _cut(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for j, edges in enumerate(self.edges_):
            out[:, j] = np.searchsorted(edges, X[:, j], side="right")
        return out

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = np.asarray(X, dtype=np.float64)
        qs = np.linspace(0.0, 1.0, 256)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization, early_stopping=False,
            random_state=self.random_state)
        self.model_.fit(self._cut(X), y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._cut(X))

    def decision_function(self, X):
        return self.model_.decision_function(self._cut(X))

    def predict(self, X):
        return self.model_.predict(self._cut(X))


def _hgb():
    return QuantizedHGB()


def _logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, C=0.5))])


def _fit(est, X, y, w):
    if hasattr(est, "steps"):
        return est.fit(X, y, clf__sample_weight=w)
    return est.fit(X, y, sample_weight=w)


def _auc(scores: Sequence[float], good: Sequence[bool]) -> Optional[float]:
    from matcher.term_report import auc
    s = list(scores)
    g = [v for v, ok in zip(s, good) if ok]
    b = [v for v, ok in zip(s, good) if not ok]
    if not g or not b:
        return None
    return auc(g, b)


def _strong_auc(per_key: Dict[Key, float], strong: Dict[Key, float]) -> Optional[float]:
    keys = [k for k in strong if k in per_key]
    return _auc([per_key[k] for k in keys], [strong[k] >= 0.6 for k in keys])


def _oof(make, data: Dict) -> Dict[Key, float]:
    """Out-of-fold probability per verdict key, folds grouped by vocal song."""
    from sklearn.model_selection import GroupKFold
    X, y, w, groups = data["X"], data["y"], data["w"], data["groups"]
    n_groups = len(set(groups.tolist()))
    out: Dict[Key, float] = {}
    if n_groups < 2:
        return out
    for tr, te in GroupKFold(n_splits=min(CV_SPLITS, n_groups)).split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        est = _fit(make(), X[tr], y[tr], w[tr])
        p = est.predict_proba(X[te])[:, 1]
        for j, prob in zip(te, p):
            if data["src"][j] == "strong":
                out.setdefault(data["keys"][j], float(prob))
    return out


def _calibrated(X, y, w, groups):
    """The final model: HGB on everything, calibrated on grouped folds —
    isotonic with enough rows, else Platt."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold
    n_groups = len(set(groups.tolist()))
    if n_groups >= 3:
        splits = [(tr, te) for tr, te in GroupKFold(n_splits=3).split(X, y, groups)
                  if len(set(y[tr].tolist())) == 2 and len(set(y[te].tolist())) == 2]
        if len(splits) >= 2:
            method = "isotonic" if len(y) >= 1000 else "sigmoid"
            cal = CalibratedClassifierCV(_hgb(), method=method, cv=splits)
            cal.fit(X, y, sample_weight=w)
            return cal
    return _fit(_hgb(), X, y, w)


def _active_row(conn) -> Optional[dict]:
    r = conn.execute(f"SELECT * FROM models WHERE active=1 AND kind=? "
                     "ORDER BY id DESC LIMIT 1", (MODEL_NAME,)).fetchone()
    return dict(r) if r else None


def _counts(conn) -> Dict[str, int]:
    return {
        "strong": conn.execute("SELECT COUNT(*) FROM pair_feedback").fetchone()[0],
        "strong_max_id": conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM pair_feedback").fetchone()[0],
        "weak_max_id": conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM pair_events").fetchone()[0],
    }


def train(db_path=None, reason: str = "manual", activate: bool = True,
          progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    """Train a ranker on every label, shadow-evaluate it, activate it if it
    earns it, and re-blend the stored table. Returns what happened."""
    from database.models import get_conn, get_pref, set_pref
    from config import MODELS_DIR
    with _TRAIN_LOCK:
        cfg = settings()
        db = _db(db_path)
        t0 = time.time()
        if progress:
            progress(5, "Collecting labels…")
        conn = get_conn(db)
        try:
            counts_now = _counts(conn)
        finally:
            conn.close()
        data = training_set(db, cfg)
        t_feat = time.time() - t0
        strong = data["strong_labels"]
        n_strong = data["counts"]["strong"]
        n_good = sum(1 for k, v in strong.items() if v >= 0.6)
        n_bad = sum(1 for k, v in strong.items() if v < 0.6)
        state = {"last_train_at": _now(), "reason": reason,
                 "strong_max_id": counts_now["strong_max_id"],
                 "weak_max_id": counts_now["weak_max_id"],
                 "n_strong": n_strong, "n_weak": data["counts"]["weak"]}
        if n_strong < MIN_STRONG or n_good < MIN_PER_CLASS or n_bad < MIN_PER_CLASS:
            state["result"] = (f"waiting for verdicts: {n_strong} of {MIN_STRONG}, "
                               f"needs {MIN_PER_CLASS}+ good and bad")
            set_pref(_PREF, state, db_path=db)
            return {"trained": False, "activated": False, "reason": state["result"],
                    "counts": data["counts"]}
        if progress:
            progress(25, "Cross-validating…")
        oof = _oof(_hgb, data)
        auc_strong = _strong_auc(oof, strong)
        auc_logreg = _strong_auc(_oof(_logreg, data), strong)
        rule_of: Dict[Key, float] = {}
        for j, k in enumerate(data["keys"]):
            if data["src"][j] == "strong":
                rule_of.setdefault(k, float(data["rule"][j]))
        rule_auc = _strong_auc(rule_of, strong)
        if progress:
            progress(60, "Fitting and calibrating…")
        est = _calibrated(data["X"], data["y"], data["w"], data["groups"])
        t_train = time.time() - t0

        conn = get_conn(db)
        try:
            active = _active_row(conn)
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM models WHERE name=?",
                (MODEL_NAME,)).fetchone()[0]
            alpha = alpha_for(n_strong, cfg["k"], cfg["alpha_max"])
            bar = None
            if active is not None and active.get("auc_strong") is not None:
                bar, bar_from = float(active["auc_strong"]), f"{MODEL_NAME} v{active['version']}"
            elif rule_auc is not None:
                bar, bar_from = float(rule_auc), "the rule"
            ok = (auc_strong is not None and (bar is None or auc_strong >= bar)
                  and (rule_auc is None or auc_strong >= rule_auc))
            if auc_strong is None:
                decision = "kept the active scorer: no held-out AUC (one class per fold)"
            elif ok:
                decision = (f"activated: held-out AUC {auc_strong:.3f} ≥ {bar_from} "
                            f"{bar:.3f}" if bar is not None else
                            f"activated: held-out AUC {auc_strong:.3f}")
            else:
                decision = (f"kept the active scorer: held-out AUC {auc_strong:.3f} "
                            f"< {bar_from} {bar:.3f}")
            label = f"{MODEL_NAME} v{version}"
            metrics = {"auc_strong": auc_strong, "rule_auc": rule_auc,
                       "logreg_auc": auc_logreg, "roc_auc": auc_strong,
                       "n_rows": int(len(data["y"])), "counts": data["counts"],
                       "n_good": n_good, "n_bad": n_bad,
                       "cv": {"scheme": f"GroupKFold({CV_SPLITS}) by vocal song",
                              "n_groups": int(len(set(data["groups"].tolist())))},
                       "secs_features": round(t_feat, 2),
                       "secs_train": round(t_train, 2), "reason": reason}
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / f"{MODEL_NAME}_v{version}.joblib"
            import joblib
            joblib.dump({"estimator": est, "feature_names": data["feature_names"],
                         "version": label, "metrics": metrics, "algo": "hgb",
                         "name": MODEL_NAME, "n_strong": n_strong, "alpha": alpha}, path)
            cur = conn.execute(
                """INSERT INTO models (name, version, dataset_id, algo, metrics_json,
                       feature_names_json, file_path, active, kind, alpha, n_strong,
                       n_weak, auc_strong, rule_auc, decision)
                   VALUES (?,?,NULL,'hgb',?,?,?,0,?,?,?,?,?,?,?)""",
                (MODEL_NAME, version, json.dumps(metrics),
                 json.dumps(data["feature_names"]), str(path), MODEL_NAME, alpha,
                 n_strong, data["counts"]["weak"], auc_strong, rule_auc, decision))
            model_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()
        state.update(result=decision, model_id=model_id, auc_strong=auc_strong,
                     rule_auc=rule_auc, secs=round(time.time() - t0, 2))
        set_pref(_PREF, state, db_path=db)
        log.info("learning: trained %s on %d verdicts / %d weak in %.1f s — %s",
                 label, n_strong, data["counts"]["weak"], time.time() - t0, decision)
        out = {"trained": True, "model_id": model_id, "version": label,
               "activated": False, "decision": decision, "metrics": metrics,
               "alpha": alpha, "secs": round(time.time() - t0, 2)}
    if ok and activate:
        if progress:
            progress(80, "Re-ranking the stored pairs…")
        activate_model(model_id, db)
        out["activated"] = True
    return out


# ── Activation, rollback, the blend ──────────────────────────────────────────

def activate_model(model_id: Optional[int], db_path=None, reblend: bool = True) -> Dict:
    """Make `model_id` the active ranker (None: the rule alone), then
    re-blend the stored table so every total reflects it."""
    from database.models import get_conn
    db = _db(db_path)
    conn = get_conn(db)
    try:
        conn.execute("UPDATE models SET active=0 WHERE active=1")
        if model_id is not None:
            if not conn.execute("SELECT 1 FROM models WHERE id=?", (model_id,)).fetchone():
                raise ValueError(f"model {model_id} not found")
            conn.execute("UPDATE models SET active=1, activated_at=? WHERE id=?",
                         (_now(), model_id))
        conn.commit()
    finally:
        conn.close()
    _BLENDERS.clear()
    return blend_table(db) if reblend else {}


def rollback(db_path=None) -> Dict:
    """One click back: the model active before this one, or the rule alone
    when there was none."""
    from database.models import get_conn
    db = _db(db_path)
    conn = get_conn(db)
    try:
        active = _active_row(conn)
        prev = conn.execute(
            "SELECT id FROM models WHERE kind=? AND activated_at IS NOT NULL "
            "AND id <> ? ORDER BY activated_at DESC, id DESC LIMIT 1",
            (MODEL_NAME, active["id"] if active else -1)).fetchone()
    finally:
        conn.close()
    target = prev[0] if prev else None
    res = activate_model(target, db)
    return {"active_model_id": target, **(res or {})}


class Blender:
    """The active model and its α, loaded once, applied to rows."""

    def __init__(self, bundle: Dict, alpha: float, version: str, model_id: int):
        self.bundle, self.alpha = bundle, float(alpha)
        self.version, self.model_id = version, model_id

    def probs(self, feats: List[dict]) -> List[float]:
        from matcher.model_scorer import model_score_batch
        return model_score_batch(feats, self.bundle) if feats else []


_BLENDERS: Dict[str, Tuple[tuple, Optional[Blender]]] = {}


def active_blender(db_path=None) -> Optional[Blender]:
    """The active ranker as a Blender, or None (no ranker active, its file is
    gone, it was trained on other feature names, or its α is 0)."""
    from database.models import get_conn
    from matcher.model_scorer import _feature_names_match
    db = _db(db_path)
    try:
        conn = get_conn(db)
        try:
            row = _active_row(conn)
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — no models table yet, or none readable
        return None
    if row is None or not row.get("file_path"):
        return None
    cfg = settings()
    path = Path(row["file_path"])
    try:
        sig = (row["id"], path.stat().st_mtime, cfg["k"], cfg["alpha_max"])
    except OSError:
        log.warning("active model file missing: %s", path)
        return None
    hit = _BLENDERS.get(str(db))
    if hit and hit[0] == sig:
        return hit[1]
    blender = None
    try:
        import joblib
        bundle = joblib.load(path)
        if _feature_names_match(bundle):
            alpha = alpha_for(row.get("n_strong") or 0, cfg["k"], cfg["alpha_max"])
            if alpha > 0:
                blender = Blender(bundle, alpha, f"{MODEL_NAME} v{row['version']}", row["id"])
    except Exception:  # noqa: BLE001
        log.exception("could not load the active model")
    _BLENDERS[str(db)] = (sig, blender)
    return blender


def blend_rows(rows: List[dict], ctx_features: Callable[[dict], Optional[dict]],
               blender: Optional[Blender]) -> None:
    """Blend scored rows in place: score_rule keeps the rule's total,
    model_prob the model's, score_total becomes the blend. The same arithmetic
    blend_table stores, so a live row (Segments) and its stored twin agree."""
    if blender is None:
        return
    feats, idx = [], []
    for j, r in enumerate(rows):
        f = ctx_features(r)
        if f is not None:
            feats.append(f)
            idx.append(j)
    for j, p in zip(idx, blender.probs(feats)):
        r = rows[j]
        rule = r.get("score_rule")
        rule = r["score_total"] if rule is None else rule
        r["score_rule"] = rule
        r["model_prob"] = round(float(p), 4)
        r["score_total"] = blend(rule, r["model_prob"], blender.alpha)
        r["model_version"] = blender.version


def blend_table(db_path=None, song_ids: Optional[Iterable[int]] = None,
                pairs: Optional[Iterable[Tuple[int, int]]] = None,
                progress=None, chunk: int = 20000) -> Dict:
    """See _blend_table. Holds the scorer's lock (re-entrant: a scoring run
    calls this from inside it), so a model switch never re-blends a table a
    Score library is rewriting."""
    from matcher.match import _SCORE_LOCK
    with _SCORE_LOCK:
        return _blend_table(db_path, song_ids, pairs, progress, chunk)


def _blend_table(db_path=None, song_ids: Optional[Iterable[int]] = None,
                 pairs: Optional[Iterable[Tuple[int, int]]] = None,
                 progress=None, chunk: int = 20000) -> Dict:
    """Write the active model's blend onto stored vocal-over-bed rows — all of
    them, or those touching `song_ids`, or those of `pairs`. With no model
    (or α = 0) a row that carries a blend is put back to its rule total;
    a row that never had one is not touched, so a library with no verdicts
    is never rewritten."""
    from database import models
    from database.models import get_conn
    db = _db(db_path)
    blender = active_blender(db)
    where, args = ["combo_type='vocal_over_instrumental'", "score_rule IS NOT NULL"], []
    if song_ids is not None:
        ids = sorted({int(x) for x in song_ids})
        if not ids:
            return {"rows": 0, "alpha": 0.0}
        marks = ",".join("?" * len(ids))
        where.append(f"(vocal_song_id IN ({marks}) OR inst_song_id IN ({marks}))")
        args += ids + ids
    if pairs is not None:
        pl = sorted({(int(a), int(b)) for a, b in pairs})
        if not pl:
            return {"rows": 0, "alpha": 0.0}
        where.append("(" + " OR ".join("(vocal_song_id=? AND inst_song_id=?)"
                                       for _ in pl) + ")")
        args += [x for p in pl for x in p]
    t0 = time.time()
    conn = get_conn(db)
    try:
        if blender is None:
            cur = conn.execute(
                f"UPDATE mashup_candidates SET score_total = score_rule, model_prob = NULL, "
                f"model_version = NULL WHERE {' AND '.join(where)} "
                f"AND (model_prob IS NOT NULL OR score_total <> score_rule)", args)
            conn.commit()
            n = cur.rowcount
            if n:
                models._RANKS_CACHE.clear()
            return {"rows": n, "alpha": 0.0, "version": None,
                    "secs": round(time.time() - t0, 2)}
        cols = _row_columns()
        rows = [dict(zip(cols, r)) for r in conn.execute(
            f"SELECT {', '.join(cols)} FROM mashup_candidates "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY vocal_song_id, inst_song_id, id", args)]
    finally:
        conn.close()
    songs = {x for r in rows for x in (r["vocal_song_id"], r["inst_song_id"])}
    ctx = FeatureContext(db, song_ids=songs if song_ids is not None or pairs is not None
                         else None)
    ctx.load_sections(songs)
    updates = []
    n = len(rows)
    for start in range(0, n, chunk):
        part = rows[start:start + chunk]
        blend_rows(part, lambda r: ctx.features(r["vocal_song_id"], r["inst_song_id"], r),
                   blender)
        updates += [(r["score_total"], r.get("model_prob"), r.get("model_version"), r["id"])
                    for r in part]
        if progress:
            progress(min(start + chunk, n), n)
    conn = get_conn(db)
    try:
        conn.executemany("UPDATE mashup_candidates SET score_total=?, model_prob=?, "
                         "model_version=? WHERE id=?", updates)
        conn.commit()
    finally:
        conn.close()
    models._RANKS_CACHE.clear()
    secs = round(time.time() - t0, 2)
    log.info("learning: blended %d rows with %s (α=%.3f) in %.1f s",
             n, blender.version, blender.alpha, secs)
    return {"rows": n, "alpha": blender.alpha, "version": blender.version, "secs": secs}


# ── Auto-retrain ─────────────────────────────────────────────────────────────

def new_since_last_train(db_path=None) -> Dict[str, int]:
    """Verdicts and non-`shown` events that arrived since the last train."""
    from database.models import get_conn, get_pref
    db = _db(db_path)
    state = get_pref(_PREF, db_path=db) or {}
    conn = get_conn(db)
    try:
        strong = conn.execute(
            "SELECT COUNT(*) FROM pair_feedback WHERE id > ?",
            (int(state.get("strong_max_id") or 0),)).fetchone()[0]
        # A re-rating rewrites its row in place; it counts once a train has
        # passed it by (its id is old). New verdicts drive retraining.
        weak = conn.execute(
            "SELECT COUNT(*) FROM pair_events WHERE id > ? AND event <> 'shown'",
            (int(state.get("weak_max_id") or 0),)).fetchone()[0]
    finally:
        conn.close()
    return {"strong": int(strong), "weak": int(weak)}


# Jobs that mean the machine is busy with the pipeline or a library-wide run.
_BUSY_KINDS = ("pipeline", "download", "separate", "analyze", "structure",
               "bulk", "match", "mix_ingest", "learn")


def pipeline_idle() -> bool:
    """No track waiting in any stage queue and no pipeline job running — the
    retrain never takes CPU from a separation (criterion 2.3)."""
    try:
        from api import jobs, queue_runner
        if queue_runner.queued_count() > 0:
            return False
        return not any(j.get("kind") in _BUSY_KINDS
                       for j in jobs.list_jobs(active_only=True))
    except Exception:  # noqa: BLE001 — cannot tell: do not retrain now
        return False


def retrain_due(db_path=None, cfg: Optional[Dict] = None) -> Optional[str]:
    """Why a retrain is due now, or None."""
    cfg = cfg or settings()
    if not cfg["auto"]:
        return None
    new = new_since_last_train(db_path)
    if new["strong"] >= cfg["min_strong"]:
        return f"{new['strong']} new verdicts"
    if new["weak"] >= cfg["min_weak"]:
        return f"{new['weak']} new pair events"
    return None


_TICK_LOCK = threading.Lock()


def tick(db_path=None, idle: Optional[Callable[[], bool]] = None,
         run: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """One debounced check: when a retrain is due and the pipeline is idle,
    start it (through `run`, the server's job wrapper, or inline). Returns the
    reason it started, else None. Never raises."""
    if not _TICK_LOCK.acquire(blocking=False):
        return None
    try:
        why = retrain_due(db_path)
        if why is None or not (idle or pipeline_idle)():
            return None
        if _TRAIN_LOCK.locked():
            return None
        if run is not None:
            run(why)
        else:
            train(db_path, reason=why)
        return why
    except Exception:  # noqa: BLE001
        log.exception("learning tick failed")
        return None
    finally:
        _TICK_LOCK.release()


# ── What the Settings readout shows ──────────────────────────────────────────

def status(db_path=None) -> Dict:
    """Labels collected, the current α, the last retrain, the AUC trend and
    which terms agree with your ears."""
    from database.models import get_conn, get_pref
    db = _db(db_path)
    cfg = settings()
    conn = get_conn(db)
    try:
        verdicts = {r[0]: r[1] for r in conn.execute(
            "SELECT verdict, COUNT(*) FROM pair_feedback GROUP BY verdict")}
        events = {r[0]: r[1] for r in conn.execute(
            "SELECT event, COUNT(*) FROM pair_events GROUP BY event")}
        history = [dict(r) for r in conn.execute(
            "SELECT id, version, created_at, activated_at, active, alpha, n_strong, "
            "n_weak, auc_strong, rule_auc, decision FROM models WHERE kind=? "
            "ORDER BY id DESC LIMIT 20", (MODEL_NAME,))]
        active = _active_row(conn)
        blended = conn.execute(
            "SELECT COUNT(*) FROM mashup_candidates WHERE model_prob IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    state = get_pref(_PREF, db_path=db) or {}
    blender = active_blender(db)
    return {
        "labels": {"strong": sum(verdicts.values()), "verdicts": verdicts,
                   "events": events, "weak_events": sum(v for k, v in events.items()
                                                       if k != "shown")},
        "alpha": blender.alpha if blender else 0.0,
        "alpha_next": alpha_for(sum(verdicts.values()), cfg["k"], cfg["alpha_max"]),
        "k": cfg["k"], "alpha_max": cfg["alpha_max"],
        "active": ({"id": active["id"], "version": f"{MODEL_NAME} v{active['version']}",
                    "auc_strong": active.get("auc_strong"),
                    "rule_auc": active.get("rule_auc"),
                    "n_strong": active.get("n_strong"),
                    "activated_at": active.get("activated_at")} if active else None),
        "blended_rows": blended,
        "last_train": state or None,
        "new_since": new_since_last_train(db),
        "thresholds": {"strong": cfg["min_strong"], "weak": cfg["min_weak"]},
        "auto": cfg["auto"], "explore_every": cfg["explore_every"],
        "min_strong": MIN_STRONG,
        "history": list(reversed(history)),
    }
