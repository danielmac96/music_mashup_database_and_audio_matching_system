"""Background workers for the learned scorer: build a dataset, train a model.

Both used to run synchronously inside the request (T2.5/T2.6). At the scale the
runbook plans for — ~1,000 documented positives across 17 mixes, plus every
sampled negative — a build walks the whole feature table and a train
cross-validates five times over it. That is minutes, not milliseconds, and an
HTTP request is the wrong place for it.
"""
from __future__ import annotations

import logging

from api import jobs

log = logging.getLogger(__name__)


def build(job_id: str, name: str, neg_ratio: int, seed: int) -> None:
    jobs.update(job_id, status="running", message="Building training dataset…")
    try:
        from matcher.features import build_dataset
    except Exception as exc:  # noqa: BLE001
        jobs.fail(job_id, f"Feature stack unavailable: {type(exc).__name__}: {exc}")
        return
    try:
        ds = build_dataset(name=name, neg_ratio=neg_ratio, seed=seed)
    except ValueError as exc:
        # No trainable positives yet — expected and actionable, not a crash.
        jobs.fail(job_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("build_dataset raised")
        jobs.fail(job_id, f"Dataset error: {type(exc).__name__}: {exc}")
        return

    jobs.done(job_id, {
        **ds,
        "summary": (f"{ds['n_pos']} positives "
                    f"({ds['n_pos_mixes']} from mixes, {ds['n_pos_user']} yours) "
                    f"· {ds['n_neg']} negatives "
                    f"({ds['n_neg_user']} rejected by ear) "
                    f"· {ds['n_groups']} CV groups"),
    })


def train(job_id: str, dataset_id: int, algo: str) -> None:
    jobs.update(job_id, status="running", message=f"Training ({algo})…")
    try:
        from matcher.model_scorer import train as _train
    except Exception as exc:  # noqa: BLE001
        jobs.fail(job_id, f"Scorer stack unavailable: {type(exc).__name__}: {exc}")
        return
    try:
        model = _train(dataset_id, algo=algo)
    except ValueError as exc:
        jobs.fail(job_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("train raised")
        jobs.fail(job_id, f"Training error: {type(exc).__name__}: {exc}")
        return

    metrics = model.get("metrics") or {}
    cv = metrics.get("cv") or {}
    jobs.done(job_id, {
        **model,
        "summary": (f"AUC {metrics.get('roc_auc')} "
                    f"({cv.get('scheme') or 'in-sample'})"),
    })
