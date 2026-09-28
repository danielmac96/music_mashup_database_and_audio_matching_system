"""
analysis/cache.py — reuse a feature group's result while its audio, version and
parameters are unchanged (analysis/registry.py says which is which).

Before this, re-analysing a track after any change — or no change at all —
decoded and recomputed everything: a bulk re-analyse of the library was minutes
per track even when one step had been touched. Now:

  * ``content_hash(path)`` — blake2b of the file's bytes, remembered per
    (path, size, mtime) so a file is read for hashing once per process.
  * ``lookup`` / ``store`` — one group's payload for one hash, via the
    ``feature_cache`` table. A row whose version or params_hash differs from
    the registry's is a miss.
  * ``StepCache`` — the adapter analyze_file takes: step name → group.
  * ``cached(group, key, compute)`` — the same for any other group (bands,
    residual, structure).

``config.current_analysis_cache()`` off turns every lookup into a miss; results
are still stored, so switching it back on serves the newest ones.

Everything here degrades: a database or hashing error is logged and treated as
a miss — the cache must never be the reason an analysis fails.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from analysis.registry import GROUPS, STEP_GROUPS, FeatureGroup

log = logging.getLogger(__name__)

_HASH_CHUNK = 1 << 20
_HASHES: Dict[tuple, str] = {}
_HASH_LOCK = threading.Lock()


def content_hash(path) -> Optional[str]:
    """Hex blake2b-128 of the file's bytes, or None when it cannot be read."""
    try:
        path = Path(path)
        st = os.stat(path)
        key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
        with _HASH_LOCK:
            if key in _HASHES:
                return _HASHES[key]
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                h.update(chunk)
        digest = h.hexdigest()
        with _HASH_LOCK:
            _HASHES[key] = digest
        return digest
    except OSError:
        log.warning("could not hash %s", path, exc_info=True)
        return None


def combo_hash(parts: Dict[str, Optional[str]],
               required: Tuple[str, ...] = ()) -> Optional[str]:
    """One key for a group over several files. Each role is named, so a missing
    optional stem ('vocals': None) is a different key from one that is present;
    a ``required`` role without a hash makes the whole key None (no caching)."""
    if any(not parts.get(role) for role in required):
        return None
    items = sorted(parts.items())
    blob = ";".join(f"{role}={h or '-'}" for role, h in items)
    return "combo:" + hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def _json_default(obj: Any):
    # numpy scalars and arrays that slipped into a result.
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _enabled() -> bool:
    try:
        from config import current_analysis_cache
        return current_analysis_cache()
    except Exception:  # noqa: BLE001
        return True


def lookup(group: FeatureGroup, key: Optional[str]) -> Optional[Any]:
    """The stored payload, or None on a miss (absent, stale, disabled, error)."""
    if not key or not _enabled():
        return None
    try:
        from database.models import get_feature_cache
        row = get_feature_cache(key, group.name)
        if (not row or row["version"] != group.version
                or row["params_hash"] != group.params_hash()):
            return None
        return json.loads(row["payload_json"])
    except Exception:  # noqa: BLE001
        log.warning("feature cache lookup failed for %s/%s", key, group.name,
                    exc_info=True)
        return None


def store(group: FeatureGroup, key: Optional[str], payload: Any,
          ms: Optional[float] = None) -> None:
    if not key:
        return
    try:
        from database.models import put_feature_cache
        put_feature_cache(key, group.name, group.version, group.params_hash(),
                          json.dumps(payload, default=_json_default),
                          analyzer=group.analyzer, ms=ms)
    except Exception:  # noqa: BLE001
        log.warning("feature cache store failed for %s/%s", key, group.name,
                    exc_info=True)


def cached(group_name: str, key: Optional[str],
           compute: Callable[[], Any]) -> Tuple[Any, bool, float]:
    """(payload, hit, ms). On a miss ``compute()`` runs and its result is stored
    unless it is None (nothing measured — try again next time). Exceptions from
    ``compute`` propagate; nothing is stored for them."""
    group = GROUPS[group_name]
    t0 = time.perf_counter()
    hit = lookup(group, key)
    if hit is not None:
        return hit, True, (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    value = compute()
    ms = (time.perf_counter() - t0) * 1000.0
    if value is not None:
        store(group, key, value, ms)
    return value, False, ms


class StepCache:
    """An analyser's view of the cache for one file: ``get(step)`` /
    ``put(step, result, ms)`` over the step's registry group. ``groups`` maps
    step names to groups — the librosa steps (analyze_file) by default, or
    ESSENTIA_STEP_GROUPS for analyze_file_essentia."""

    def __init__(self, key: Optional[str],
                 groups: Optional[Dict[str, FeatureGroup]] = None):
        self.key = key
        self.groups = STEP_GROUPS if groups is None else groups

    def get(self, step: str) -> Optional[dict]:
        group = self.groups.get(step)
        if group is None:
            return None
        hit = lookup(group, self.key)
        return hit if isinstance(hit, dict) else None

    def put(self, step: str, result: dict, ms: float) -> None:
        group = self.groups.get(step)
        if group is not None:
            store(group, self.key, result, ms)
