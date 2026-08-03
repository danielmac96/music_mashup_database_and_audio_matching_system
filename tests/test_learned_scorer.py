"""End-to-end tests for the learned pairwise scorer: build_dataset (matcher/
features.py) → train (matcher/model_scorer.py) → serve (load_active_model /
model_score).

sqlite + numpy + scikit-learn only — no audio, no network. Documented mashups are
seeded directly (mixes + mix_tracks + mashup_pairs) the way api/routes/mixes.py
would after import + ingest.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "test.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    # Keep dataset/model artifacts inside the tmp dir.
    import matcher.features as features
    import matcher.model_scorer as model_scorer
    monkeypatch.setattr(features, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(model_scorer, "MODELS_DIR", tmp_path / "models")
    return p


def _add_song(db_path, k, *, bpm, camelot):
    """A song with both a vocals and an instrumental feature row (analysed)."""
    from database.models import upsert_features, upsert_song
    sid = upsert_song(f"Song {k}", f"Artist {k}", f"u://{k}", 240,
                      status="analysed", db_path=db_path)
    for stem in ("vocals", "instrumental"):
        upsert_features(sid, stem, {
            "bpm": bpm, "key": "C", "mode": "major", "camelot": camelot,
            "loudness_rms": 0.1 + 0.01 * k, "energy": 0.5,
            "mfcc": [float(k)] * 13, "spectral_centroid": 2000.0,
            "spectral_rolloff": 4000.0, "zero_crossing_rate": 0.05,
        }, db_path=db_path)
    return sid


def _add_mix_track(conn, mix_id, pos, song_id, is_overlay, *,
                   status="manual", score=None, dur=None):
    cur = conn.execute(
        "INSERT INTO mix_tracks (mix_id, position, is_overlay, artist, title, "
        "link_url, resolve_status, resolve_score, resolve_duration_secs, song_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (mix_id, pos, int(is_overlay), "A", f"T{pos}", "u://link",
         status, score, dur, song_id))
    return cur.lastrowid


def _document_pair(conn, mix_id, pos, vocal_song, inst_song, *,
                   vocal_status="manual", vocal_score=None, vocal_dur=None):
    """Insert a bed + overlay mix_track and the mashup_pairs row linking them."""
    bed_id = _add_mix_track(conn, mix_id, pos, inst_song, is_overlay=False)
    overlay_id = _add_mix_track(conn, mix_id, pos + 1, vocal_song, is_overlay=True,
                                status=vocal_status, score=vocal_score, dur=vocal_dur)
    conn.execute(
        "INSERT INTO mashup_pairs (mix_id, inst_mix_track_id, vocal_mix_track_id) "
        "VALUES (?,?,?)", (mix_id, bed_id, overlay_id))
    return bed_id, overlay_id


@pytest.fixture()
def seeded(db_path):
    """6 analysed songs, 3 trusted documented positives, 1 low-confidence
    (excluded) documented pair."""
    from database.models import get_conn, init_db
    init_db(db_path)
    ids = {k: _add_song(db_path, k, bpm=120.0 + k, camelot="8A") for k in range(1, 7)}

    conn = get_conn(db_path)
    try:
        mix_id = conn.execute(
            "INSERT INTO mixes (title, import_method) VALUES ('BBM 27','paste')"
        ).lastrowid
        # Trusted positives (manual links).
        _document_pair(conn, mix_id, 0, ids[1], ids[2])
        _document_pair(conn, mix_id, 2, ids[3], ids[4])
        _document_pair(conn, mix_id, 4, ids[5], ids[6])
        # Low-confidence auto link on the vocal side → excluded from positives,
        # and (crucially) also excluded from the negative pool.
        _document_pair(conn, mix_id, 6, ids[1], ids[4],
                       vocal_status="auto", vocal_score=0.40, vocal_dur=200.0)
        conn.commit()
    finally:
        conn.close()
    return db_path, ids


# ── build_dataset ─────────────────────────────────────────────────────────────

def test_build_dataset_counts_and_gate(seeded):
    from matcher.features import FEATURE_NAMES, build_dataset
    db_path, _ = seeded
    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    # 3 trusted positives; the low-confidence pair is NOT one of them.
    assert ds["n_pos"] == 3
    # Pool = 6×6 − 6 same − 4 documented = 26; capped at neg_ratio×n_pos = 15.
    assert ds["n_neg"] == 15
    assert Path(ds["file_path"]).exists()
    assert ds["feature_names"] == FEATURE_NAMES


def test_build_dataset_registers_row(seeded):
    from database.models import get_conn
    from matcher.features import build_dataset
    db_path, _ = seeded
    ds = build_dataset(name="bbm", db_path=db_path)
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (ds["id"],)).fetchone()
    conn.close()
    assert row["name"] == "bbm" and row["n_pos"] == 3


def test_build_dataset_no_positives_raises(db_path):
    """Only a low-confidence documented pair → nothing trainable yet."""
    from database.models import get_conn, init_db
    from matcher.features import build_dataset
    init_db(db_path)
    v = _add_song(db_path, 1, bpm=120.0, camelot="8A")
    i = _add_song(db_path, 2, bpm=121.0, camelot="8A")
    conn = get_conn(db_path)
    mix_id = conn.execute(
        "INSERT INTO mixes (title, import_method) VALUES ('m','paste')").lastrowid
    _document_pair(conn, mix_id, 0, v, i,
                   vocal_status="auto", vocal_score=0.3, vocal_dur=200.0)
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="No trainable mashup pairs"):
        build_dataset(db_path=db_path)


# ── train + serve ─────────────────────────────────────────────────────────────

def test_train_and_score_roundtrip(seeded):
    from database.models import get_conn
    from matcher.features import build_dataset, pair_features
    from matcher.model_scorer import load_active_model, model_score, train
    db_path, _ = seeded

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    model = train(ds["id"], algo="logreg", db_path=db_path)
    assert model["id"] and Path(model["file_path"]).exists()
    assert "roc_auc" in model["metrics"]

    # No model is active yet.
    assert load_active_model(db_path=db_path) is None

    conn = get_conn(db_path)
    conn.execute("UPDATE models SET active=1 WHERE id=?", (model["id"],))
    conn.commit()
    conn.close()

    bundle = load_active_model(db_path=db_path)
    assert bundle is not None
    from matcher.features import FEATURE_NAMES
    assert bundle["feature_names"] == FEATURE_NAMES

    feats = pair_features(
        {"bpm": 120, "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5,
         "mfcc": [1.0] * 13, "spectral_centroid": 2000, "spectral_rolloff": 4000,
         "zero_crossing_rate": 0.05},
        {"bpm": 121, "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5,
         "mfcc": [1.0] * 13, "spectral_centroid": 2000, "spectral_rolloff": 4000,
         "zero_crossing_rate": 0.05}, [], [])
    p = model_score(feats, bundle)
    assert 0.0 <= p <= 1.0


def test_train_rejects_unknown_dataset(db_path):
    from database.models import init_db
    from matcher.model_scorer import train
    init_db(db_path)
    with pytest.raises(ValueError, match="not found"):
        train(999, db_path=db_path)
