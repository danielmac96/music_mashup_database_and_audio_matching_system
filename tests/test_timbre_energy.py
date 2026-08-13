"""T2.2 — make timbre and energy actually discriminate.

Together they carry 45% of the composite weight while being close to constant
(timbre) or dominated by an artefact (energy). They are also the learned
scorer's input features, and a model cannot recover information that is not in
its features — so this is a prerequisite for the ML work, not an alternative.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from matcher import match
    importlib.reload(match)
    return models, match


def _seed_library(models, n=12, seed=0):
    """A library with genuinely varied timbre and per-stem loudness levels."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        sid = models.upsert_song(f"S{i}", "A", f"https://sc/{i}", 200, "Pop",
                                 status="analysed")
        for stem, base in (("vocals", 0.08), ("instrumental", 0.17)):
            mfcc = [-160 + rng.normal(0, 90)]                  # c0: loudness term
            mfcc += list(rng.normal(0, 20, 12))                # c1-12: timbre
            models.upsert_features(sid, stem, {
                "bpm": 120.0, "camelot": "8B", "key": "C", "mode": "major",
                "loudness_rms": float(base * np.exp(rng.normal(0, 0.3))),
                "energy": 20.0, "mfcc": mfcc,
            })


# ── timbre ───────────────────────────────────────────────────────────────────

def test_timbre_ignores_the_loudness_coefficient(tmp_path, monkeypatch):
    """MFCC[0] is a large same-sign loudness term. Two tracks that differ ONLY
    in overall level must not thereby look like different timbres."""
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models)
    stats = match.get_library_stats(refresh=True)

    body = list(np.linspace(-20, 20, 12))
    quiet = {"mfcc": [-300.0] + body}
    loud = {"mfcc": [-40.0] + body}

    assert match.timbre_score(quiet, loud, stats) == pytest.approx(1.0, abs=1e-6)


def test_timbre_separates_different_timbres(tmp_path, monkeypatch):
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models)
    stats = match.get_library_stats(refresh=True)

    a = {"mfcc": [-160.0] + [30.0] * 12}
    same = {"mfcc": [-160.0] + [30.0] * 12}
    opposite = {"mfcc": [-160.0] + [-30.0] * 12}

    assert match.timbre_score(a, same, stats) > match.timbre_score(a, opposite, stats)


def test_timbre_spans_a_usable_range_across_a_library(tmp_path, monkeypatch):
    """The acceptance criterion: real spread, not a cluster above 0.9."""
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models, n=16, seed=3)
    stats = match.get_library_stats(refresh=True)

    rows = [f for f in models.get_all_features(stem_type="vocals")]
    beds = [f for f in models.get_all_features(stem_type="instrumental")]
    vals = [match.timbre_score(a, b, stats) for a in rows for b in beds]

    assert max(vals) - min(vals) > 0.3, f"range only {max(vals) - min(vals):.3f}"


def test_timbre_is_unknown_rather_than_confident_without_mfcc(tmp_path, monkeypatch):
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models)
    stats = match.get_library_stats(refresh=True)
    assert match.timbre_score({}, {"mfcc": [1.0] * 13}, stats) == 0.5


# ── energy ───────────────────────────────────────────────────────────────────

def test_energy_compares_each_stem_against_its_own_kind(tmp_path, monkeypatch):
    """Vocal stems are systematically ~2x quieter than instrumental stems, so a
    raw ratio mostly measures which stem type it is. A vocal that is typical for
    a vocal over a bed that is typical for a bed is a good energy match."""
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models, n=20, seed=1)
    stats = match.get_library_stats(refresh=True)

    med_v = float(np.exp(stats.loudness["vocals"][0]))
    med_i = float(np.exp(stats.loudness["instrumental"][0]))
    typical = match.energy_match(
        {"loudness_rms": med_v, "stem_type": "vocals"},
        {"loudness_rms": med_i, "stem_type": "instrumental"}, stats)

    assert typical > 0.9, f"two typical stems scored {typical:.3f}"
    # and the raw ratio would have called that a poor match
    assert match.energy_score(med_v, med_i) < typical


def test_energy_penalises_a_genuine_mismatch(tmp_path, monkeypatch):
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models, n=20, seed=1)
    stats = match.get_library_stats(refresh=True)

    lo_mu, lo_sd = stats.loudness["vocals"]
    hi_mu, hi_sd = stats.loudness["instrumental"]
    loud_vocal = {"loudness_rms": float(np.exp(lo_mu + 2.5 * lo_sd)), "stem_type": "vocals"}
    quiet_bed = {"loudness_rms": float(np.exp(hi_mu - 2.5 * hi_sd)), "stem_type": "instrumental"}

    assert match.energy_match(loud_vocal, quiet_bed, stats) < 0.5


# ── the contract sub_scores holds ────────────────────────────────────────────

def test_sub_scores_returns_exactly_the_weighted_terms(tmp_path, monkeypatch):
    """sub_scores must produce one value per MATCH_WEIGHTS key and nothing else.

    Asserting against the config rather than a hardcoded list is what makes this
    a real guard: adding a weight without a sub-score (or the reverse) would
    otherwise silently contribute zero to every pair in the library.
    collision_score joined the four in Phase D.
    """
    from config import MATCH_WEIGHTS
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models)
    a = models.get_all_features(stem_type="vocals")[0]
    b = models.get_all_features(stem_type="instrumental")[0]

    s = match.sub_scores(a, b)
    assert set(s) == set(MATCH_WEIGHTS)
    assert all(0.0 <= v <= 1.0 for v in s.values())


def test_sub_scores_is_deterministic_for_identical_input(tmp_path, monkeypatch):
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models)
    a = models.get_all_features(stem_type="vocals")[0]
    b = models.get_all_features(stem_type="instrumental")[0]
    assert match.sub_scores(a, b) == match.sub_scores(a, b)


def test_candidates_expose_a_library_percentile_alongside_the_raw_score(tmp_path, monkeypatch):
    """A raw composite reads ~78% for everything, which tells the user nothing
    about whether THIS pair is good *for their library*. The percentile is the
    headline; the raw value stays available in the Plan expander."""
    models, _ = _setup(tmp_path, monkeypatch)
    ids = []
    for i in range(5):
        v = models.upsert_song(f"V{i}", "A", f"https://sc/pv{i}", 200, "Pop", status="analysed")
        b = models.upsert_song(f"B{i}", "A", f"https://sc/pb{i}", 200, "EDM", status="analysed")
        ids.append((v, b))
    for n, (v, b) in enumerate(ids):
        models.upsert_candidate(
            {"song_id": v, "title": f"V{n}", "artist": "A", "bpm": 120.0, "key": "C",
             "mode": "major", "camelot": "8B", "loudness_rms": 0.1, "energy": 0.5},
            {"song_id": b, "title": f"B{n}", "artist": "A", "bpm": 120.0, "key": "C",
             "mode": "major", "camelot": "8B", "loudness_rms": 0.1, "energy": 0.5},
            {"total": 0.50 + n * 0.05, "bpm_score": 1.0, "key_score": 1.0,
             "energy_score": 0.5, "timbre_score": 0.5},
        )

    rows = models.get_candidates_enriched(limit=10)
    assert all("score_percentile" in r for r in rows)
    best = max(rows, key=lambda r: r["score_total"])
    worst = min(rows, key=lambda r: r["score_total"])
    assert best["score_percentile"] == pytest.approx(1.0)
    assert worst["score_percentile"] == pytest.approx(0.0)
    assert best["score_total"] == pytest.approx(0.70)   # raw value still there


def test_percentile_ranks_within_combo_type_not_across_it(tmp_path, monkeypatch):
    """The ranked list is segmented by combo type, so ranking a vocal-over-bed
    pair against instrumental-over-instrumental ones makes the best visible row
    read as ~84th percentile instead of the top."""
    models, _ = _setup(tmp_path, monkeypatch)

    def cand(n, combo, total):
        v = models.upsert_song(f"V{combo}{n}", "A", f"https://sc/{combo}v{n}", 200,
                               "Pop", status="analysed")
        b = models.upsert_song(f"B{combo}{n}", "A", f"https://sc/{combo}b{n}", 200,
                               "EDM", status="analysed")
        models.upsert_candidate(
            {"song_id": v, "title": "V", "artist": "A", "bpm": 120.0, "key": "C",
             "mode": "major", "camelot": "8B", "loudness_rms": 0.1, "energy": 0.5},
            {"song_id": b, "title": "B", "artist": "A", "bpm": 120.0, "key": "C",
             "mode": "major", "camelot": "8B", "loudness_rms": 0.1, "energy": 0.5},
            {"total": total, "bpm_score": 1.0, "key_score": 1.0,
             "energy_score": 0.5, "timbre_score": 0.5},
            combo_type=combo)

    # instrumental pairs all score ABOVE every vocal pair
    for n, t in enumerate([0.90, 0.95, 0.99]):
        cand(n, "instrumental_over_instrumental", t)
    for n, t in enumerate([0.50, 0.60, 0.70]):
        cand(n, "vocal_over_instrumental", t)

    vi = models.get_candidates_enriched(combo_type="vocal_over_instrumental", limit=10)
    top = max(vi, key=lambda r: r["score_total"])
    assert top["score_percentile"] == pytest.approx(1.0), \
        "best vocal-over-bed pair must be the top of ITS list, not dragged down by I/I pairs"


def test_library_stats_are_computed_once_not_per_pair(tmp_path, monkeypatch):
    """Scoring a 900-song library must not re-read every features row per pair."""
    models, match = _setup(tmp_path, monkeypatch)
    _seed_library(models)
    match.get_library_stats(refresh=True)

    calls = {"n": 0}
    real = models.get_all_features

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(match, "get_all_features", counting)
    a = models.get_all_features(stem_type="vocals")[0]
    b = models.get_all_features(stem_type="instrumental")[0]
    for _ in range(50):
        match.sub_scores(a, b)

    assert calls["n"] == 0, "sub_scores re-read the library instead of using cached stats"
