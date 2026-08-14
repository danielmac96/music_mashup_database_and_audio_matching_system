"""matcher/model_scorer.py — train, load, and serve the learned pairwise scorer
(Phase 5).

Consumes a dataset built by matcher.features.build_dataset, fits a scikit-learn
classifier on the FEATURE_NAMES columns, and saves a joblib bundle registered in
the ``models`` table. At serve time matcher.match.score_all_pairs calls
load_active_model() + model_score(); both degrade gracefully (return None / fall
back to the heuristic) so a missing or broken model never breaks scoring.

The bundle dict shape (also what api/routes/mashups.scorer_status reads):
    {estimator, feature_names, version, metrics, algo, dataset_id, name}
where ``metrics`` carries ``roc_auc`` (the badge/AUC column).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from config import MODELS_DIR
from database.models import DB_PATH, get_conn

log = logging.getLogger(__name__)

_ALGOS = ("logreg", "gbm")

# Two mashups lifted from the same Big Bootie set are not independent samples:
# the DJ's taste, the era, the tempo range and often the key are shared. A
# random split puts siblings on both sides of the fold and reports an AUC that
# says "I recognise this mix", not "I can rank a pair". Grouping by mix is the
# only honest split, and the same reasoning makes every user judgment one group.
CV_SPLITS = 5


def _build_estimator(algo: str):
    """A fresh, unfitted estimator. logreg is scaled + class-balanced (robust on
    small imbalanced sets); gbm is a gradient-boosted tree ensemble."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if algo == "gbm":
        # HistGradientBoosting over the older GradientBoosting: it handles the
        # NaN-free-but-sparse feature matrix faster and is what T2.4 asks for.
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=42)
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def _metrics(estimator, X_test, y_test) -> dict:
    """roc_auc / pr_auc / precision / recall / f1 on a held-out (or in-sample)
    split. Single-class targets can't yield an AUC — those come back None."""
    from sklearn.metrics import (
        average_precision_score, f1_score, precision_score, recall_score,
        roc_auc_score,
    )
    proba = estimator.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out: dict = {
        "n_eval": int(len(y_test)),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }
    if len(set(y_test.tolist())) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_test, proba)), 4)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def train(dataset_id: int, algo: str = "logreg", db_path: Path = DB_PATH) -> dict:
    """Train a model on a built dataset, save it, and register it (inactive).

    Raises ValueError for actionable states (unknown dataset, missing file,
    too little data, single-class labels)."""
    if algo not in _ALGOS:
        raise ValueError(f"algo must be one of {_ALGOS}")

    conn = get_conn(db_path)
    try:
        ds = conn.execute(
            "SELECT id, name, version, file_path FROM datasets WHERE id=?",
            (dataset_id,)).fetchone()
        if not ds:
            raise ValueError(f"dataset {dataset_id} not found")
        path = Path(ds["file_path"])
        if not path.exists():
            raise ValueError(f"dataset file missing: {path}")

        X, y, feature_names, groups = _load_dataset(path)

        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos < 1 or n_neg < 1:
            raise ValueError(
                f"dataset needs both classes to train (got {n_pos} pos / "
                f"{n_neg} neg) — import more mixes or widen negatives.")
        if len(y) < 8:
            raise ValueError(
                f"only {len(y)} examples — ingest more documented mixes before "
                "training (need a handful of positives).")

        # Held-out evaluation when both classes have room to spare; else
        # evaluate in-sample (small-data regime) and flag it honestly.
        in_sample = n_pos < 2 or n_neg < 2 or len(y) < 16
        estimator = _build_estimator(algo)
        if in_sample:
            estimator.fit(X, y)
            metrics = _metrics(estimator, X, y)
            metrics["in_sample"] = True
            metrics["cv"] = None
        else:
            metrics = _grouped_cv_metrics(algo, X, y, groups)
            metrics["in_sample"] = False
            # Refit on all data for the deployed artifact (more signal to serve).
            estimator.fit(X, y)

        # Calibrate so the displayed percentage is a real probability. Without
        # this a "82%" is an arbitrary monotone score and the Min-match slider
        # means nothing across models.
        estimator = _calibrate(estimator, X, y, in_sample)

        name = f"pairwise_{ds['name']}"
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM models WHERE name=?",
            (name,)).fetchone()
        version = row["v"]
        version_label = f"{name} v{version}"

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        bundle = {
            "estimator": estimator,
            "feature_names": feature_names,
            "version": version_label,
            "metrics": metrics,
            "algo": algo,
            "dataset_id": dataset_id,
            "name": name,
        }
        import joblib
        file_path = MODELS_DIR / f"{name}_v{version}.joblib"
        joblib.dump(bundle, file_path)

        cur = conn.execute(
            """INSERT INTO models
                   (name, version, dataset_id, algo, metrics_json,
                    feature_names_json, file_path, active)
               VALUES (?,?,?,?,?,?,?,0)""",
            (name, version, dataset_id, algo, json.dumps(metrics),
             json.dumps(feature_names), str(file_path)))
        conn.commit()
        model_id = cur.lastrowid
        log.info("Trained %s (%s) roc_auc=%s on %d pos / %d neg → %s",
                 version_label, algo, metrics.get("roc_auc"), n_pos, n_neg, file_path)
        return {
            "id": model_id, "name": name, "version": version,
            "algo": algo, "dataset_id": dataset_id,
            "metrics": metrics, "auc": metrics.get("roc_auc"),
            "file_path": str(file_path), "active": False,
        }
    finally:
        conn.close()


# ── Serve ─────────────────────────────────────────────────────────────────────
#
# Cache the active bundle keyed by (file_path, mtime) so re-scoring the library
# doesn't re-read the joblib file per call, while a freshly-trained/activated
# model is still picked up (its path or mtime differs).
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, float], dict] = {}


def _feature_names_match(bundle: dict) -> bool:
    """Refuse a model trained on a different feature vector.

    model_score orders the row by the bundle's own feature_names, so a stale
    name simply is not present in the dict and _coerce turns it into 0.0 — the
    model would keep scoring, on zeros, with no error anywhere. Falling back to
    the heuristic is the honest failure. Retrain to re-enable the model.
    """
    try:
        from matcher.features import FEATURE_NAMES
    except Exception:  # noqa: BLE001
        return True                      # cannot verify; do not block scoring
    trained = list(bundle.get("feature_names") or [])
    if trained == list(FEATURE_NAMES):
        return True
    log.warning(
        "Active model was trained on a different feature set — falling back to "
        "the heuristic. Rebuild the dataset and retrain. missing=%s unexpected=%s",
        sorted(set(FEATURE_NAMES) - set(trained)),
        sorted(set(trained) - set(FEATURE_NAMES)))
    return False


def load_active_model(db_path: Path = DB_PATH) -> Optional[dict]:
    """Load the active model's bundle, or None when none is active or loading
    fails. Never raises — scoring falls back to the heuristic on None."""
    try:
        conn = get_conn(db_path)
        try:
            row = conn.execute(
                "SELECT file_path FROM models WHERE active=1 "
                "ORDER BY id DESC LIMIT 1").fetchone()
        finally:
            conn.close()
        if not row or not row["file_path"]:
            return None
        path = Path(row["file_path"])
        if not path.exists():
            log.warning("Active model file missing: %s", path)
            return None
        key = (str(path), path.stat().st_mtime)
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return cached
        import joblib
        bundle = joblib.load(path)
        if not _feature_names_match(bundle):
            return None
        with _CACHE_LOCK:
            _CACHE[key] = bundle
        return bundle
    except Exception:  # noqa: BLE001 — model loading must never break scoring
        log.exception("load_active_model failed")
        return None


def model_score(feats: dict, bundle: dict) -> float:
    """Probability in [0,1] that ``feats`` is a good mashup, per the model. Orders
    ``feats`` by the bundle's own feature_names so a model always reads the columns
    it was trained on."""
    names = bundle["feature_names"]
    row = np.asarray([[_coerce(feats.get(n)) for n in names]], dtype=np.float64)
    proba = bundle["estimator"].predict_proba(row)[0][1]
    return float(np.clip(proba, 0.0, 1.0))


def model_score_batch(feats_list: list, bundle: dict) -> list:
    """model_score for many pairs at once — one predict_proba call instead of
    one per pair. Same ordering rule (the bundle's own feature_names) and the
    same clipping, so a batch of one is identical to model_score.

    Library-wide scoring calls this tens of thousands of times' worth of rows;
    per-row predict_proba spends nearly all of its time in scikit-learn's
    validation and dispatch, not in the model."""
    names = bundle["feature_names"]
    if not feats_list:
        return []
    X = np.asarray([[_coerce(f.get(n)) for n in names] for f in feats_list],
                   dtype=np.float64)
    proba = bundle["estimator"].predict_proba(X)[:, 1]
    return np.clip(proba, 0.0, 1.0).tolist()


def _coerce(val) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def _grouped_cv_metrics(algo: str, X, y, groups) -> dict:
    """Cross-validated metrics, grouped so siblings never straddle a fold.

    Falls back to a stratified split when there are too few groups to split on
    — reported in the returned dict rather than silently, because "AUC 0.9 on
    two groups" and "AUC 0.9 on seventeen" are very different claims.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    n_groups = len(set(groups)) if groups else 0
    use_groups = groups is not None and n_groups >= 2
    n_splits = min(CV_SPLITS, n_groups if use_groups else 5,
                   int(min((y == 1).sum(), (y == 0).sum())))
    n_splits = max(2, n_splits)

    if use_groups:
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
        split_iter = splitter.split(X, y, groups=np.asarray(groups))
        scheme = f"GroupKFold({min(n_splits, n_groups)}) by mix"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)
        scheme = f"StratifiedKFold({n_splits}) — too few groups to split by mix"

    folds = []
    for tr, te in split_iter:
        if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:
            continue
        est = _build_estimator(algo)
        est.fit(X[tr], y[tr])
        folds.append(_metrics(est, X[te], y[te]))

    if not folds:
        est = _build_estimator(algo)
        est.fit(X, y)
        out = _metrics(est, X, y)
        out["cv"] = {"scheme": "none — no fold had both classes", "n_folds": 0,
                     "n_groups": n_groups}
        return out

    def _mean(key):
        vals = [f[key] for f in folds if f.get(key) is not None]
        return round(float(sum(vals) / len(vals)), 4) if vals else None

    return {
        "n_eval": sum(f["n_eval"] for f in folds),
        "precision": _mean("precision"), "recall": _mean("recall"),
        "f1": _mean("f1"), "roc_auc": _mean("roc_auc"), "pr_auc": _mean("pr_auc"),
        "cv": {"scheme": scheme, "n_folds": len(folds), "n_groups": n_groups},
    }


def _calibrate(estimator, X, y, in_sample: bool):
    """Wrap the fitted estimator so predict_proba is a real probability.

    A raw margin is monotone but arbitrary: "82%" from a logistic regression and
    "82%" from a boosted tree mean different things, and the Min-match slider
    has to mean the same thing across models. Skipped in the small-data regime,
    where calibration would fit noise.
    """
    if in_sample:
        return estimator
    try:
        from sklearn.calibration import CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
        calibrated.fit(X, y)
        return calibrated
    except Exception:  # noqa: BLE001
        log.warning("calibration failed; serving the uncalibrated estimator",
                    exc_info=True)
        return estimator


def feature_contributions(feats: dict, bundle: dict, top_n: int = 3) -> list:
    """The features pushing this pair up or down, for the row's "why".

    Without this you cannot tell a well-ranked list from a plausible-looking
    one, and you will not trust it enough to skip auditioning. Returns
    [{feature, direction, weight}], strongest first. Empty when the model shape
    does not expose usable coefficients — better nothing than a fabricated
    explanation.
    """
    import numpy as np

    names = bundle.get("feature_names") or []
    if not names:
        return []
    x = np.asarray([float(feats.get(n) or 0.0) for n in names], dtype=np.float64)

    est = bundle.get("estimator")
    coef = None
    # Unwrap calibration and pipeline layers to find a linear model.
    for candidate in (est, getattr(est, "estimator", None)):
        if candidate is None:
            continue
        inner = candidate
        if hasattr(inner, "named_steps"):
            inner = list(inner.named_steps.values())[-1]
        if hasattr(inner, "coef_"):
            coef = np.asarray(inner.coef_).ravel()
            break
    if coef is None or coef.shape[0] != x.shape[0]:
        return []

    contrib = coef * x
    order = np.argsort(-np.abs(contrib))[:top_n]
    return [
        {"feature": names[i],
         "direction": "up" if contrib[i] >= 0 else "down",
         "weight": round(float(abs(contrib[i])), 4)}
        for i in order if abs(contrib[i]) > 1e-9
    ]


def _load_dataset(path: Path):
    """(X, y, feature_names, groups) from a dataset artifact.

    CSV is what build_dataset writes now (T2.5). .npz is still read so datasets
    built before that keep training — a stored artifact is a record of what a
    model was trained on, and silently refusing to load one would strand it.
    """
    import numpy as np

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return (np.asarray(data["X"], dtype=np.float64),
                np.asarray(data["y"], dtype=np.int64),
                [str(n) for n in data["feature_names"].tolist()],
                ([str(g) for g in data["groups"].tolist()]
                 if "groups" in data.files else None))

    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"dataset {path.name} has no rows")
    header = rows[0]
    if header[-2:] != ["label", "group"]:
        raise ValueError(f"dataset {path.name} is missing the label/group columns")
    feature_names = header[:-2]
    X, y, groups = [], [], []
    for r in rows[1:]:
        X.append([float(v) for v in r[:len(feature_names)]])
        y.append(int(float(r[-2])))
        groups.append(r[-1])
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64),
            feature_names, groups)
