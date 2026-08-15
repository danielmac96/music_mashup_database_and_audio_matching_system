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


def _activate_model(db_path):
    """Build → train → activate, returning the loaded bundle."""
    from database.models import get_conn
    from matcher.features import build_dataset
    from matcher.model_scorer import load_active_model, train

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    model = train(ds["id"], algo="logreg", db_path=db_path)
    conn = get_conn(db_path)
    conn.execute("UPDATE models SET active=1 WHERE id=?", (model["id"],))
    conn.commit()
    conn.close()
    return load_active_model(db_path=db_path)


def test_model_score_batch_matches_one_at_a_time(seeded):
    """The bulk scorer is only safe if a batch of N is N calls to model_score.
    Library-wide scoring uses the batch form for every pair in the BPM window."""
    from matcher.features import pair_features
    from matcher.model_scorer import model_score, model_score_batch
    db_path, ids = seeded
    bundle = _activate_model(db_path)

    from database.models import get_all_features
    vocals = get_all_features(stem_type="vocals", db_path=db_path)
    inst = get_all_features(stem_type="instrumental", db_path=db_path)
    feats = [pair_features(v, i, [], []) for v in vocals for i in inst
             if v["song_id"] != i["song_id"]]
    assert len(feats) >= 6

    batched = model_score_batch(feats, bundle)
    assert batched == pytest.approx([model_score(f, bundle) for f in feats],
                                    abs=1e-12)
    assert model_score_batch([], bundle) == []


def test_score_all_pairs_with_an_active_model(seeded):
    """The model path must persist the model's probability as score_total, tag
    the rows, and widen the gate to the BPM window — a documented mashup often
    breaks the key rule, which is exactly what the model is there to learn."""
    from database.models import get_conn
    from matcher.features import pair_features
    from matcher.match import score_all_pairs
    from matcher.model_scorer import model_score
    db_path, ids = seeded

    # A key-incompatible but tempo-compatible track: 3B against 8A scores 0.25.
    odd = _add_song(db_path, 7, bpm=122.0, camelot="3B")
    bundle = _activate_model(db_path)

    # The key gate is passed explicitly here. Since P1.1 it defaults OFF for the
    # heuristic too (effort already prices the transpose, so gating on key as
    # well charged twice for it), and comparing two ungated runs would not test
    # anything. The claim under test is that the MODEL path ignores the gate
    # whatever it is set to.
    heuristic = score_all_pairs(db_path=db_path, scorer="heuristic",
                                key_min_score=0.55)
    modelled = score_all_pairs(db_path=db_path, scorer="model",
                               key_min_score=0.55)
    assert modelled["_scorer"] == "model"
    assert modelled["_model_version"] == bundle["version"]
    assert len(modelled["vocal_over_instrumental"]) > \
        len(heuristic["vocal_over_instrumental"])
    assert any(r["inst_song_id"] == odd or r["vocal_song_id"] == odd
               for r in modelled["vocal_over_instrumental"])

    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchall()
    ii = conn.execute(
        "SELECT DISTINCT scorer FROM mashup_candidates "
        "WHERE combo_type='instrumental_over_instrumental'").fetchall()
    conn.close()
    assert rows and all(r["scorer"] == "model" for r in rows)
    assert all(r["model_version"] == bundle["version"] for r in rows)
    # instrumental↔instrumental has no training signal — it stays heuristic.
    assert [r["scorer"] for r in ii] == ["heuristic"]

    from database.models import get_all_features
    from matcher.match import get_library_stats
    stats = get_library_stats(db_path=db_path, refresh=True)
    by_song = {(f["song_id"], f["stem_type"]): f
               for stem in ("vocals", "instrumental")
               for f in get_all_features(stem_type=stem, db_path=db_path)}
    for r in rows:
        v = by_song[(r["vocal_song_id"], "vocals")]
        i = by_song[(r["inst_song_id"], "instrumental")]
        expected = model_score(pair_features(v, i, [], [], stats), bundle)
        assert r["score_total"] == pytest.approx(round(expected, 4), abs=1e-9)
        # The four heuristic sub-scores are still written, for display.
        assert 0.0 <= r["score_key"] <= 1.0


def test_train_rejects_unknown_dataset(db_path):
    from database.models import init_db
    from matcher.model_scorer import train
    init_db(db_path)
    with pytest.raises(ValueError, match="not found"):
        train(999, db_path=db_path)


# ── A.1: pair_feedback as a training source ──────────────────────────────────

def test_feedback_verdicts_become_training_rows(seeded):
    """The user's ✓/✗ judgments are the highest-signal labels in the system.
    They must reach the dataset, with 'no' as an explicit hard negative rather
    than merely being withheld from the negative pool."""
    from database.models import get_conn, upsert_pair_feedback
    from matcher.features import build_dataset
    import json

    db_path, ids = seeded
    # 3 loves the mixes do not document, and 2 rejections.
    upsert_pair_feedback(ids[2], ids[1], "love", db_path=db_path)
    upsert_pair_feedback(ids[4], ids[3], "ok", db_path=db_path)
    upsert_pair_feedback(ids[6], ids[5], "love", db_path=db_path)
    upsert_pair_feedback(ids[2], ids[3], "no", db_path=db_path)
    upsert_pair_feedback(ids[4], ids[5], "no", db_path=db_path)

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    # 3 documented + 3 user positives.
    assert ds["n_pos_mixes"] == 3
    assert ds["n_pos_user"] == 3
    assert ds["n_pos"] == 6
    # Both rejections are carried as hard negatives, on top of the sampled ones.
    assert ds["n_neg_user"] == 2
    assert ds["n_neg"] == ds["n_neg_user"] + ds["n_neg_sampled"]

    conn = get_conn(db_path)
    row = conn.execute("SELECT config_json FROM datasets WHERE id=?",
                       (ds["id"],)).fetchone()
    conn.close()
    cfg = json.loads(row["config_json"])
    assert cfg["n_pos_user"] == 3 and cfg["n_neg_user"] == 2


def test_rejected_pair_is_never_a_positive(seeded):
    """A documented mashup the user rejected by ear is not a positive. Their
    taste is the target; a contradictory label pair teaches nothing."""
    from database.models import upsert_pair_feedback
    from matcher.features import build_dataset
    db_path, ids = seeded
    # ids[1] over ids[2] is a trusted documented positive in the fixture.
    upsert_pair_feedback(ids[1], ids[2], "no", db_path=db_path)

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    assert ds["n_pos_mixes"] == 2          # was 3
    assert ds["n_neg_user"] == 1


def _read_dataset(path):
    """The dataset as a list of dicts. T2.5 writes CSV: a human can open it,
    check the label balance and sort by a feature without a Python session."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_judged_pairs_never_sampled_as_negatives(seeded):
    """A pair with a real label must not also be drawn as a random negative.

    neg_ratio is set high enough to exhaust the pool, so n_neg_sampled reports
    the pool size exactly: 6×6 − 6 self − 4 documented = 26, and one more comes
    off for the judged pair."""
    from database.models import upsert_pair_feedback
    from matcher.features import build_dataset
    db_path, ids = seeded

    baseline = build_dataset(name="a", neg_ratio=100, seed=1, db_path=db_path)
    assert baseline["n_neg_sampled"] == 26

    upsert_pair_feedback(ids[2], ids[1], "love", db_path=db_path)
    ds = build_dataset(name="b", neg_ratio=100, seed=1, db_path=db_path)
    assert ds["n_neg_sampled"] == 25

    rows = _read_dataset(ds["file_path"])
    assert len(rows) == ds["n_pos"] + ds["n_neg"]


def test_groups_identify_mix_and_user_sources(seeded):
    """GroupKFold needs a group per row: mashups from one mix are not
    independent samples, and neither are the user's own judgments."""
    from database.models import upsert_pair_feedback
    from matcher.features import build_dataset
    db_path, ids = seeded
    upsert_pair_feedback(ids[2], ids[1], "love", db_path=db_path)

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    groups = [r["group"] for r in _read_dataset(ds["file_path"])]
    assert any(g.startswith("mix:") for g in groups)
    assert "user" in set(groups)


def test_feedback_only_library_is_trainable(db_path):
    """No documented mixes at all, but the user has judged pairs — that is a
    trainable dataset. Previously this raised."""
    from database.models import init_db, upsert_pair_feedback
    from matcher.features import build_dataset
    init_db(db_path)
    a = _add_song(db_path, 1, bpm=120.0, camelot="8A")
    b = _add_song(db_path, 2, bpm=121.0, camelot="8A")
    upsert_pair_feedback(a, b, "love", db_path=db_path)

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    assert ds["n_pos"] == 1 and ds["n_pos_user"] == 1 and ds["n_pos_mixes"] == 0


def test_pinned_sections_drive_the_feature_vector():
    """A verdict is about the moment that was auditioned, so the section terms
    must describe those sections, not the ones build_pairings would pick."""
    from matcher.features import pair_features
    top = {"bpm": 120.0, "camelot": "8A"}
    bed = {"bpm": 120.0, "camelot": "8A"}
    top_sections = [
        {"section_index": 0, "start_sec": 0.0, "end_sec": 30.0,
         "label": "chorus", "energy": 0.9, "vocal_presence": 0.9},
        {"section_index": 1, "start_sec": 30.0, "end_sec": 60.0,
         "label": "verse", "energy": 0.2, "vocal_presence": 0.4},
    ]
    bed_sections = [
        {"section_index": 0, "start_sec": 0.0, "end_sec": 30.0,
         "label": "drop", "energy": 0.9, "vocal_presence": 0.0},
        {"section_index": 1, "start_sec": 30.0, "end_sec": 60.0,
         "label": "verse", "energy": 0.1, "vocal_presence": 0.0},
    ]
    default = pair_features(top, bed, top_sections, bed_sections)
    pinned = pair_features(top, bed, top_sections, bed_sections,
                           top_section_idx=1, bed_section_idx=1)
    # Default picks chorus-over-drop (vocal_presence 0.9); the pin picks the
    # quiet verse pair (0.4).
    assert default["top_section_vocal_presence"] == 0.9
    assert pinned["top_section_vocal_presence"] == 0.4
    assert pinned["hook_energy_delta"] != default["hook_energy_delta"]


def test_unresolvable_pin_falls_back_to_default_pick():
    """A stale section index (structure was re-detected since the verdict) must
    not blank the section terms."""
    from matcher.features import pair_features
    top = {"bpm": 120.0, "camelot": "8A"}
    bed = {"bpm": 120.0, "camelot": "8A"}
    top_sections = [{"section_index": 0, "start_sec": 0.0, "end_sec": 30.0,
                     "label": "chorus", "energy": 0.9, "vocal_presence": 0.9}]
    bed_sections = [{"section_index": 0, "start_sec": 0.0, "end_sec": 30.0,
                     "label": "drop", "energy": 0.8, "vocal_presence": 0.0}]
    default = pair_features(top, bed, top_sections, bed_sections)
    stale = pair_features(top, bed, top_sections, bed_sections,
                          top_section_idx=99, bed_section_idx=99)
    assert stale == default


# ── Phase F: grouped CV, calibration, reasons, wider gate ────────────────────

def test_cv_is_grouped_by_mix_not_random(seeded):
    """Two mashups from one Big Bootie set are not independent samples: the
    DJ's taste, era, tempo range and often key are shared. A random split puts
    siblings on both sides and reports "I recognise this mix", not "I can rank
    a pair"."""
    from database.models import get_conn, upsert_pair_feedback
    from matcher.features import build_dataset
    from matcher.model_scorer import train
    db_path, ids = seeded
    # Enough user rows for a second group and a real held-out split.
    for v, i in ((ids[2], ids[1]), (ids[4], ids[3]), (ids[6], ids[5])):
        upsert_pair_feedback(v, i, "love", db_path=db_path)

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    model = train(ds["id"], algo="logreg", db_path=db_path)
    cv = model["metrics"].get("cv")
    assert cv is not None
    assert cv["n_groups"] >= 2
    assert cv["n_folds"] >= 1
    assert "GroupKFold" in cv["scheme"] or "too few groups" in cv["scheme"]


def test_metrics_report_whether_they_are_in_sample(seeded):
    """An honest badge: "AUC 0.9 in-sample" and "AUC 0.9 cross-validated over
    17 mixes" are very different claims."""
    from matcher.features import build_dataset
    from matcher.model_scorer import train
    db_path, _ = seeded
    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    model = train(ds["id"], algo="logreg", db_path=db_path)
    assert "in_sample" in model["metrics"]


def test_scores_stay_probabilities(seeded):
    from matcher.features import pair_features
    from matcher.model_scorer import model_score
    db_path, _ = seeded
    bundle = _activate_model(db_path)
    from database.models import get_all_features
    v = get_all_features(stem_type="vocals", db_path=db_path)[0]
    i = get_all_features(stem_type="instrumental", db_path=db_path)[0]
    p = model_score(pair_features(v, i, [], []), bundle)
    assert 0.0 <= p <= 1.0


def test_feature_contributions_explain_a_row(seeded):
    """Without a why, a plausible-looking list is indistinguishable from a good
    one — and you will not trust it enough to skip auditioning."""
    from matcher.features import pair_features
    from matcher.model_scorer import feature_contributions
    db_path, _ = seeded
    bundle = _activate_model(db_path)
    from database.models import get_all_features
    v = get_all_features(stem_type="vocals", db_path=db_path)[0]
    i = get_all_features(stem_type="instrumental", db_path=db_path)[0]
    reasons = feature_contributions(pair_features(v, i, [], []), bundle)
    assert len(reasons) <= 3
    for r in reasons:
        assert r["feature"] in bundle["feature_names"]
        assert r["direction"] in ("up", "down")
        assert r["weight"] >= 0


def test_contributions_are_empty_rather_than_invented():
    """A model shape with no usable coefficients must produce no explanation
    rather than a fabricated one."""
    from matcher.model_scorer import feature_contributions
    assert feature_contributions({}, {}) == []
    assert feature_contributions({"a": 1.0}, {"feature_names": ["a"],
                                              "estimator": object()}) == []


def test_surprise_terms_measure_distance_not_similarity():
    """Compatibility and contrast are different axes: tight on tempo and
    harmony, far on genre and era."""
    from matcher.features import surprise_terms
    same = surprise_terms({"genre": "House", "release_year": 2015},
                          {"genre": "House", "release_year": 2015})
    far = surprise_terms({"genre": "Indie Rock", "release_year": 2003},
                         {"genre": "Techno", "release_year": 2023})
    assert same["surprise_genre"] == pytest.approx(0.0)
    assert same["surprise_era"] == pytest.approx(0.0)
    assert far["surprise_genre"] == pytest.approx(1.0)
    assert far["surprise_era"] == pytest.approx(1.0)


def test_related_genres_are_near_not_far():
    """Free-text SoundCloud genres: "Future House" and "House" are neighbours."""
    from matcher.features import surprise_terms
    s = surprise_terms({"genre": "Future House"}, {"genre": "House"})
    assert 0.0 < s["surprise_genre"] < 1.0


def test_unknown_genre_or_era_is_neutral():
    from matcher.features import surprise_terms
    s = surprise_terms({}, {"genre": "House", "release_year": 2015})
    assert s["surprise_genre"] == 0.5 and s["surprise_era"] == 0.5


def test_model_path_widens_the_tempo_gate(seeded):
    """The gate bounds the matrix; it must not express taste the model is meant
    to learn. A pair the heuristic rejects on tempo must still reach the model."""
    from config import BPM_MAX_DIFF, BPM_MAX_DIFF_MODEL
    assert BPM_MAX_DIFF_MODEL > BPM_MAX_DIFF


def test_dataset_is_csv_with_a_readable_header(seeded):
    """T2.5 — a dataset is the one artifact a human might want to open."""
    from matcher.features import FEATURE_NAMES, build_dataset
    db_path, _ = seeded
    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    assert Path(ds["file_path"]).suffix == ".csv"
    rows = _read_dataset(ds["file_path"])
    assert list(rows[0].keys()) == [*FEATURE_NAMES, "label", "group"]
    assert {r["label"] for r in rows} == {"0", "1"}


def test_legacy_npz_datasets_still_train(seeded, tmp_path):
    """A stored artifact is the record of what a model was trained on; silently
    refusing to load a pre-CSV one would strand it."""
    import numpy as np
    from database.models import get_conn
    from matcher.features import FEATURE_NAMES, build_dataset
    from matcher.model_scorer import train
    db_path, _ = seeded

    ds = build_dataset(name="bbm", neg_ratio=5, seed=1, db_path=db_path)
    rows = _read_dataset(ds["file_path"])
    X = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, X=X, y=y, groups=np.asarray([r["group"] for r in rows]),
             feature_names=np.asarray(FEATURE_NAMES))

    conn = get_conn(db_path)
    conn.execute("UPDATE datasets SET file_path=? WHERE id=?",
                 (str(legacy), ds["id"]))
    conn.commit()
    conn.close()

    model = train(ds["id"], algo="logreg", db_path=db_path)
    assert model["id"] and "roc_auc" in model["metrics"]
