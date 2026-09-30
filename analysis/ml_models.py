"""
analysis/ml_models.py — the Essentia TensorFlow models behind the genre and
tag estimates (readme §9, phase 5).

Discogs-EffNet turns 16 kHz mono audio into one 1280-d embedding per ~1 s
patch; each classification head reads those embeddings. Files come from
essentia.upf.edu on first use into <data_dir>/essentia_models, each .pb with
its .json metadata. The metadata, not this file, names the input/output nodes
and the classes — they differ by head. No trusted hashes were published to pin
in advance, so each file's sha256 is recorded on first download (manifest.json)
for anyone who wants to check a copy later.

Offline, or before the download finishes, the models are *unavailable*, which
the analysis treats as "skip the tags", never as a failed track.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.request
from collections import namedtuple
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

BASE_URL = "https://essentia.upf.edu/models/"
EFFNET = "discogs-effnet-bs64-1"
EFFNET_PATH = "feature-extractors/discogs-effnet/"

Head = namedtuple("Head", "model key kind positive")
# kind: "genre" (the 400 Discogs styles), "binary" (keep `positive`'s
# probability), "multi" (multi-label sigmoid: keep the top labels).
HEADS = (
    Head("genre_discogs400", "genre", "genre", None),
    Head("voice_instrumental", "voice", "binary", "voice"),
    Head("gender", "female", "binary", "female"),
    Head("danceability", "danceable", "binary", "danceable"),
    Head("tonal_atonal", "tonal", "binary", "tonal"),
    Head("timbre", "bright", "binary", "bright"),
    Head("mood_happy", "happy", "binary", "happy"),
    Head("mood_sad", "sad", "binary", "sad"),
    Head("mood_aggressive", "aggressive", "binary", "aggressive"),
    Head("mood_relaxed", "relaxed", "binary", "relaxed"),
    Head("mood_party", "party", "binary", "party"),
    Head("mood_acoustic", "acoustic", "binary", "acoustic"),
    Head("mood_electronic", "electronic", "binary", "electronic"),
    Head("mtg_jamendo_moodtheme", "moodtheme", "multi", None),
    Head("mtg_jamendo_instrument", "instruments", "multi", None),
)

# After a failed download, wait this long before trying again: a track
# analysed offline should not stall on a network timeout, and the next one
# should not either.
RETRY_AFTER_SECS = 600

_LOCK = threading.Lock()
_PREDICTORS: Dict[str, object] = {}
_STATE = {"failed_at": 0.0, "error": None}


class ModelsUnavailable(RuntimeError):
    pass


def head_id(head: Head) -> str:
    return f"{head.model}-discogs-effnet-1"


def catalogue_ids() -> list:
    return [EFFNET] + [head_id(h) for h in HEADS]


def _url(model_id: str, ext: str) -> str:
    if model_id == EFFNET:
        return f"{BASE_URL}{EFFNET_PATH}{model_id}{ext}"
    head = model_id.rsplit("-discogs-effnet-1", 1)[0]
    return f"{BASE_URL}classification-heads/{head}/{model_id}{ext}"


def models_dir() -> Path:
    from config import DATA_DIR
    return Path(DATA_DIR) / "essentia_models"


def _fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())


def _missing() -> list:
    d = models_dir()
    return [(m, ext) for m in catalogue_ids() for ext in (".pb", ".json")
            if not (d / f"{m}{ext}").exists()]


def ensure_models(download: bool = True) -> bool:
    """True when every catalogued model is on disk, downloading what is
    missing unless ``download`` is False or a recent attempt failed."""
    if not _missing():
        return True
    if not download or time.time() - _STATE["failed_at"] < RETRY_AFTER_SECS:
        return False
    with _LOCK:
        todo = _missing()
        if not todo:
            return True
        d = models_dir()
        d.mkdir(parents=True, exist_ok=True)
        manifest_path = d / "manifest.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists() else {})
        try:
            for model_id, ext in todo:
                dest = d / f"{model_id}{ext}"
                part = dest.with_name(dest.name + ".part")
                _fetch(_url(model_id, ext), part)
                part.replace(dest)
                manifest[dest.name] = hashlib.sha256(dest.read_bytes()).hexdigest()
                log.info("fetched essentia model %s", dest.name)
        except Exception as exc:  # noqa: BLE001 — offline is not an error
            _STATE.update(failed_at=time.time(), error=f"{type(exc).__name__}: {exc}")
            log.warning("essentia models unavailable: %s", _STATE["error"])
            return False
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True),
                                     encoding="utf-8")
        _STATE.update(failed_at=0.0, error=None)
        return True


def metadata(model_id: str) -> dict:
    return json.loads((models_dir() / f"{model_id}.json").read_text(encoding="utf-8"))


def io_nodes(meta: dict, purpose: str = "predictions") -> tuple:
    """(input node, output node) for ``purpose`` from a model's metadata."""
    schema = meta.get("schema") or {}
    inp = (schema.get("inputs") or [{}])[0].get("name")
    out = next((o.get("name") for o in schema.get("outputs") or []
                if o.get("output_purpose") == purpose), None)
    return inp, out


def status() -> dict:
    missing = _missing()
    return {"available": not missing, "dir": str(models_dir()),
            "missing": [f"{m}{e}" for m, e in missing], "error": _STATE["error"]}


def _predictor(model_id: str):
    """One loaded graph per model for the life of the process: loading costs
    far more than predicting (measured ~0.7 s per head, most of it the load)."""
    pred = _PREDICTORS.get(model_id)
    if pred is not None:
        return pred
    import essentia.standard as es
    meta = metadata(model_id)
    graph = str(models_dir() / f"{model_id}.pb")
    if model_id == EFFNET:
        _inp, out = io_nodes(meta, "embeddings")
        pred = es.TensorflowPredictEffnetDiscogs(graphFilename=graph, output=out)
    else:
        inp, out = io_nodes(meta)
        pred = es.TensorflowPredict2D(graphFilename=graph, input=inp, output=out)
    _PREDICTORS[model_id] = pred
    return pred


def predict(audio16: np.ndarray) -> Dict[str, np.ndarray]:
    """{"embeddings": (P, 1280), <head model>: (P, n_classes), ...}. Serialised:
    the predictors are shared across analysis threads."""
    if not ensure_models():
        raise ModelsUnavailable(status().get("error") or "models not downloaded")
    with _LOCK:
        emb = np.asarray(_predictor(EFFNET)(np.ascontiguousarray(audio16, dtype=np.float32)))
        out = {"embeddings": emb}
        for h in HEADS:
            out[h.model] = np.asarray(_predictor(head_id(h))(emb))
    return out


def classes(model: str) -> list:
    """Class labels of a head, by its short name (``genre_discogs400``)."""
    return list(metadata(f"{model}-discogs-effnet-1").get("classes") or [])
