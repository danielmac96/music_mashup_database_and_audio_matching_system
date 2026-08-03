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


def _build_estimator(algo: str):
    """A fresh, unfitted estimator. logreg is scaled + class-balanced (robust on
    small imbalanced sets); gbm is a gradient-boosted tree ensemble."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if algo == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=42)
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

        data = np.load(path, allow_pickle=True)
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int64)
        feature_names = [str(n) for n in data["feature_names"].tolist()]

        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos < 1 or n_neg < 1:
            raise ValueError(
                f"dataset needs both classes to train (got {n_pos} pos / "
                f"{n_neg} neg) — import more mixes or widen negatives.")
        if len(y) < 8:
            raise ValueError(
                f"only {len(y)} examples — ingest more documented mixes before "
                "training (need a handful of positives).")

        # Held-out split when both classes have room to spare; else evaluate
        # in-sample (small-data regime) and flag it.
        in_sample = n_pos < 2 or n_neg < 2 or len(y) < 16
        estimator = _build_estimator(algo)
        if in_sample:
            estimator.fit(X, y)
            metrics = _metrics(estimator, X, y)
            metrics["in_sample"] = True
        else:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42)
            estimator.fit(X_tr, y_tr)
            metrics = _metrics(estimator, X_te, y_te)
            metrics["in_sample"] = False
            # Refit on all data for the deployed artifact (more signal to serve).
            estimator.fit(X, y)

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


def _coerce(val) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0
