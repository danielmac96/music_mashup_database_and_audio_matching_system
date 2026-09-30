"""analysis/ml_models.py — the Essentia model files: catalogue, download,
manifest and node names. The TensorFlow calls themselves need Essentia and are
exercised in tests/test_essentia_analyzer.py."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mm(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import analysis.ml_models as mod
    importlib.reload(mod)
    fetched = []

    def fake_fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        meta = {"schema": {"inputs": [{"name": "model/Placeholder"}],
                           "outputs": [{"name": "model/Softmax", "output_purpose": "predictions"},
                                       {"name": "model/dense/BiasAdd", "output_purpose": ""}]},
                "classes": ["a", "b"]}
        dest.write_bytes(json.dumps(meta).encode() if url.endswith(".json") else b"pb:" + url.encode())
    monkeypatch.setattr(mod, "_fetch", fake_fetch)
    return mod, fetched


def test_the_catalogue_is_effnet_and_the_decided_heads(mm):
    mod, _ = mm
    ids = mod.catalogue_ids()
    assert ids[0] == "discogs-effnet-bs64-1"
    for h in ("genre_discogs400", "voice_instrumental", "gender", "danceability",
              "tonal_atonal", "timbre", "mood_happy", "mood_sad", "mood_aggressive",
              "mood_relaxed", "mood_party", "mood_acoustic", "mood_electronic",
              "mtg_jamendo_moodtheme", "mtg_jamendo_instrument"):
        assert f"{h}-discogs-effnet-1" in ids
    assert not any("approachability" in i or "engagement" in i for i in ids)


def test_models_are_fetched_once_with_a_sha256_manifest(mm):
    mod, fetched = mm
    assert mod.ensure_models() is True
    n = len(fetched)
    assert n == 2 * len(mod.catalogue_ids())          # a .pb and a .json each
    manifest = json.loads((mod.models_dir() / "manifest.json").read_text(encoding="utf-8"))
    pb = mod.models_dir() / "discogs-effnet-bs64-1.pb"
    assert manifest["discogs-effnet-bs64-1.pb"] == hashlib.sha256(pb.read_bytes()).hexdigest()
    assert mod.ensure_models() is True and len(fetched) == n     # nothing refetched
    assert not list(mod.models_dir().glob("*.part"))


def test_a_failed_download_backs_off_and_is_not_an_error(mm, monkeypatch):
    mod, fetched = mm

    def boom(url, dest):
        fetched.append(url)
        raise OSError("offline")
    monkeypatch.setattr(mod, "_fetch", boom)
    assert mod.ensure_models() is False
    n = len(fetched)
    assert mod.ensure_models() is False and len(fetched) == n   # backing off
    assert mod.status()["available"] is False and "offline" in mod.status()["error"]


def test_without_download_missing_models_are_just_unavailable(mm):
    mod, fetched = mm
    assert mod.ensure_models(download=False) is False and fetched == []


def test_node_names_come_from_the_metadata(mm):
    mod, _ = mm
    meta = {"schema": {"inputs": [{"name": "serving_default_model_Placeholder"}],
                       "outputs": [{"name": "PartitionedCall:0", "output_purpose": "predictions"}]}}
    assert mod.io_nodes(meta) == ("serving_default_model_Placeholder", "PartitionedCall:0")
    eff = {"schema": {"inputs": [{"name": "serving_default_melspectrogram"}],
                      "outputs": [{"name": "PartitionedCall:0", "output_purpose": "predictions"},
                                  {"name": "PartitionedCall:1", "output_purpose": "embeddings"}]}}
    assert mod.io_nodes(eff, "embeddings")[1] == "PartitionedCall:1"


import numpy as np


def _labels():
    from analysis.ml_models import HEADS
    # Binary heads: [other, positive], so a [0.5, 0.5] prediction reads 0.5.
    lab = {h.model: ["other", h.positive] if h.kind == "binary" else ["x", "y"]
           for h in HEADS}
    lab["genre_discogs400"] = ["Electronic---House", "Electronic---Tropical House",
                               "Hip Hop---Trap", "Rock---Indie Rock"]
    lab["voice_instrumental"] = ["instrumental", "voice"]
    lab["gender"] = ["female", "male"]
    lab["mood_party"] = ["non_party", "party"]
    lab["mtg_jamendo_moodtheme"] = ["dark", "energetic", "happy"]
    lab["mtg_jamendo_instrument"] = ["drums", "synthesizer", "voice"]
    return lab


def _preds(voice: float):
    from analysis.ml_models import HEADS
    p = {h.model: np.tile([0.5, 0.5], (3, 1)) for h in HEADS}
    p["genre_discogs400"] = np.tile([0.3, 0.25, 0.4, 0.05], (3, 1))
    p["voice_instrumental"] = np.tile([1 - voice, voice], (3, 1))
    p["gender"] = np.tile([0.7, 0.3], (3, 1))
    p["mood_party"] = np.array([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    p["mtg_jamendo_moodtheme"] = np.tile([0.1, 0.6, 0.3], (3, 1))
    p["mtg_jamendo_instrument"] = np.tile([0.5, 0.7, 0.2], (3, 1))
    p["embeddings"] = np.ones((3, 1280))
    return p


def test_tags_summarise_every_head_over_the_track():
    from analysis.essentia_groups import summarise_heads
    t = summarise_heads(_preds(voice=0.9), _labels())
    assert [g["label"] for g in t["genre"][:2]] == ["Hip Hop---Trap", "Electronic---House"]
    # The parent genre sums its styles: Electronic 0.55 beats Hip Hop 0.40.
    assert t["genre_parent"] == "Electronic"
    assert t["voice"] == pytest.approx(0.9)
    assert t["female"] == pytest.approx(0.7)
    assert t["mood"]["party"] == pytest.approx(0.7)             # mean over patches
    assert [m["label"] for m in t["moodtheme"]] == ["energetic", "happy", "dark"]
    assert t["instruments"][0] == {"label": "synthesizer", "p": pytest.approx(0.7)}


def test_gender_is_not_claimed_for_an_instrumental():
    from analysis.essentia_groups import summarise_heads
    assert summarise_heads(_preds(voice=0.2), _labels())["female"] is None
