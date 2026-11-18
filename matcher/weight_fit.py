"""matcher/weight_fit.py — a section-weight set fitted to your verdicts
(readme §5.8, "Suggested from your ratings").

The term report (matcher/term_report.py) says which section terms agree with
your ears; this turns that into weights. On every judged pair that still has a
scored row, a logistic fit of the verdict (love/ok vs no) on the six section
terms, with every weight held ≥ 0 — a term can be switched off, never turned
upside down — normalised to sum to 1 as Settings stores them.

The expected change is measured honestly: out of fold (folds grouped by vocal
song), the AUC of the section fit under the proposed weights against the AUC
of the same fit under your current weights, on the same verdicts. It is a
proposal only: Tuning shows the diff and applies it on one click; nothing here
writes a setting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SECTION_TERMS = (("label", "score_label"), ("duration", "score_duration"),
                 ("voice", "score_voice"), ("phrase", "score_phrase"),
                 ("rhythm", "score_rhythm"), ("structure", "score_structure"))
MIN_PER_CLASS = 10          # the term report's floor: below it an AUC is noise
FOLDS = 5
_NEUTRAL = 0.5


def _matrix(pairs: Sequence[Tuple[Dict, Dict]]):
    X, y, g = [], [], []
    for f, c in pairs:
        verdict = f.get("verdict")
        if verdict not in ("love", "ok", "no"):
            continue
        X.append([float(c.get(col)) if c.get(col) is not None else _NEUTRAL
                  for _k, col in SECTION_TERMS])
        y.append(1 if verdict in ("love", "ok") else 0)
        g.append(f["vocal_song_id"])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(g)


def _fit(X: np.ndarray, y: np.ndarray, l2: float = 0.05) -> np.ndarray:
    """Non-negative logistic regression (with a free intercept), balanced."""
    from scipy.optimize import minimize
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    sw = np.where(y == 1, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    k = X.shape[1]

    def loss(theta):
        w, b = theta[:k], theta[k]
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-9
        ll = -(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).mean()
        grad_z = sw * (p - y) / len(y)
        return ll + l2 * float(w @ w), np.concatenate([X.T @ grad_z + 2 * l2 * w,
                                                       [grad_z.sum()]])
    x0 = np.concatenate([np.full(k, 0.5), [0.0]])
    res = minimize(loss, x0, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * k + [(None, None)])
    return np.clip(res.x[:k], 0.0, None)


def _normalised(w: np.ndarray) -> Optional[np.ndarray]:
    s = float(w.sum())
    return w / s if s > 1e-9 else None


def _auc(scores: np.ndarray, y: np.ndarray) -> Optional[float]:
    from matcher.term_report import auc
    good, bad = scores[y == 1].tolist(), scores[y == 0].tolist()
    if not good or not bad:
        return None
    return auc(good, bad)


def suggest(feedback: Sequence[Dict], candidates: Sequence[Dict],
            current: Dict[str, float]) -> Dict:
    """The proposed section weights, the diff from `current`, and the expected
    AUC of the section fit before and after (out of fold)."""
    from matcher.term_report import judged_rows
    pairs = judged_rows(feedback, candidates)
    X, y, g = _matrix(pairs)
    n_good, n_bad = int(y.sum()), int(len(y) - y.sum())
    base = {"n_good": n_good, "n_bad": n_bad, "min_per_class": MIN_PER_CLASS,
            "current": {k: round(float(current.get(k, 0.0)), 4) for k, _c in SECTION_TERMS}}
    if n_good < MIN_PER_CLASS or n_bad < MIN_PER_CLASS:
        return {**base, "ready": False,
                "reason": f"needs {MIN_PER_CLASS}+ loved/ok and {MIN_PER_CLASS}+ rejected "
                          f"pairs that are still scored (have {n_good} and {n_bad})"}
    cur = np.asarray([float(current.get(k, 0.0)) for k, _c in SECTION_TERMS])
    cur = _normalised(cur) if _normalised(cur) is not None else np.full(len(cur), 1 / len(cur))

    # Out of fold: fit on the other vocal songs, score the held-out ones.
    oof = np.zeros(len(y))
    groups = sorted(set(g.tolist()))
    folds = min(FOLDS, len(groups))
    if folds >= 2:
        from sklearn.model_selection import GroupKFold
        for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
            if len(set(y[tr].tolist())) < 2:
                oof[te] = X[te] @ cur
                continue
            w = _normalised(_fit(X[tr], y[tr]))
            oof[te] = X[te] @ (w if w is not None else cur)
    else:
        oof = X @ cur
    w_all = _normalised(_fit(X, y))
    if w_all is None:
        return {**base, "ready": False,
                "reason": "no section term separates your verdicts — keep the current weights"}
    auc_now = _auc(X @ cur, y)
    auc_new = _auc(oof, y)
    proposed = {k: round(float(v), 4) for (k, _c), v in zip(SECTION_TERMS, w_all)}
    return {**base, "ready": True, "proposed": proposed,
            "diff": {k: round(proposed[k] - base["current"][k], 4) for k, _c in SECTION_TERMS},
            "auc_current": auc_now, "auc_proposed": auc_new,
            "auc_change": (round(auc_new - auc_now, 4)
                           if auc_now is not None and auc_new is not None else None),
            "measured": f"section fit AUC on {len(y)} judged pairs, out of fold by vocal song"}
