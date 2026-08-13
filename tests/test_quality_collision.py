"""Phase D — stem quality and arrangement collision (analysis/quality.py).

Two questions the ranked list could not previously ask: is this stem usable at
all, and do the two sides fight for the same frequency space? Both are invisible
to the four similarity sub-scores, and both sink mashups those sub-scores like.

Real audio via soundfile; no demucs, no network.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from analysis.quality import (  # noqa: E402
    BAND_EDGES, N_BANDS, band_energy, collision_block, collision_score,
    quiet_windows_for, residual_vocal_ratio, stem_quality,
)

SR = 22050


def _tone(path: Path, freq: float, secs: float = 4.0, amp: float = 0.3,
          sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(secs * sr), dtype=np.float32) / sr
    sf.write(str(path), (amp * np.sin(2 * np.pi * freq * t)).astype("float32"), sr)
    return path


def _noise(path: Path, secs: float = 4.0, amp: float = 0.3, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sf.write(str(path), (amp * rng.standard_normal(int(secs * sr))).astype("float32"), sr)
    return path


# ── Band occupancy ───────────────────────────────────────────────────────────

def test_band_energy_is_a_distribution(tmp_path):
    bands = band_energy(_tone(tmp_path / "a.wav", 440.0))
    assert len(bands) == N_BANDS
    assert sum(bands) == pytest.approx(1.0, abs=1e-3)
    assert all(v >= 0 for v in bands)


def test_band_energy_locates_the_tone(tmp_path):
    """A 440 Hz tone must land in the 400-1000 Hz band, not somewhere else."""
    bands = band_energy(_tone(tmp_path / "a.wav", 440.0))
    idx = next(i for i, (lo, hi) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:]))
               if lo <= 440.0 < hi)
    assert bands[idx] == max(bands)


def test_band_energy_of_an_unreadable_file_is_zeros(tmp_path):
    assert band_energy(tmp_path / "nope.wav") == [0.0] * N_BANDS


# ── Collision ────────────────────────────────────────────────────────────────

def test_disjoint_bands_do_not_collide():
    a = [1.0] + [0.0] * (N_BANDS - 1)
    b = [0.0] * (N_BANDS - 1) + [1.0]
    assert collision_score(a, b) == pytest.approx(1.0)


def test_identical_bands_collide_completely():
    a = [1.0 / N_BANDS] * N_BANDS
    assert collision_score(a, a) == pytest.approx(0.0)


def test_partial_overlap_is_in_between():
    a = [0.5, 0.5] + [0.0] * (N_BANDS - 2)
    b = [0.0, 0.5, 0.5] + [0.0] * (N_BANDS - 3)
    s = collision_score(a, b)
    assert 0.0 < s < 1.0


def test_unmeasured_bands_score_neutral_not_flattering():
    """"We did not measure it" is not "they are complementary"."""
    a = [1.0 / N_BANDS] * N_BANDS
    assert collision_score(None, a) == 0.5
    assert collision_score(a, []) == 0.5
    assert collision_score([0.0] * N_BANDS, a) == 0.5


def test_collision_block_matches_the_scalar_form():
    """Same drift guard the other sub-scores have."""
    rng = np.random.default_rng(3)
    A = rng.random((4, N_BANDS))
    B = rng.random((5, N_BANDS))
    A[2] = 0.0          # an unmeasured row on each side
    B[1] = 0.0
    block = collision_block(A, B)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            want = collision_score(list(A[i]) if A[i].sum() > 0 else None,
                                   list(B[j]) if B[j].sum() > 0 else None)
            assert block[i, j] == pytest.approx(want, abs=1e-12)


def test_a_real_mid_heavy_pair_collides_more_than_a_split_pair(tmp_path):
    """The point of the feature: two records in the same band fight, even when
    every similarity score likes them."""
    mid_a = band_energy(_tone(tmp_path / "m1.wav", 700.0))
    mid_b = band_energy(_tone(tmp_path / "m2.wav", 900.0))
    low = band_energy(_tone(tmp_path / "low.wav", 80.0))
    high = band_energy(_tone(tmp_path / "high.wav", 7000.0))
    assert collision_score(mid_a, mid_b) < collision_score(low, high)


# ── Residual vocal ───────────────────────────────────────────────────────────

def test_residual_vocal_ratio_reports_the_share_of_voice(tmp_path):
    """A bed that still carries its own topline is not a usable bed, and nothing
    in the four sub-scores can see that."""
    full = _tone(tmp_path / "full.wav", 440.0, amp=0.4)
    loud = _tone(tmp_path / "v_loud.wav", 440.0, amp=0.4)
    quiet = _tone(tmp_path / "v_quiet.wav", 440.0, amp=0.04)
    assert residual_vocal_ratio(loud, full) == pytest.approx(1.0, abs=0.05)
    assert residual_vocal_ratio(quiet, full) < 0.05


def test_residual_vocal_ratio_missing_input_is_none(tmp_path):
    full = _tone(tmp_path / "full.wav", 440.0)
    assert residual_vocal_ratio(None, full) is None
    assert residual_vocal_ratio(full, None) is None


# ── Stem quality ─────────────────────────────────────────────────────────────

def test_clean_separation_scores_higher_than_bleeding_one(tmp_path):
    full = _noise(tmp_path / "full.wav")
    # A "clean" stem: uncorrelated with its complement.
    clean = _tone(tmp_path / "clean.wav", 440.0)
    other_clean = _tone(tmp_path / "other_clean.wav", 3000.0)
    # A "bleeding" stem: the complement is the same signal.
    bleed = _tone(tmp_path / "bleed.wav", 440.0)
    other_bleed = _tone(tmp_path / "other_bleed.wav", 440.0)

    q_clean = stem_quality(clean, full, other_path=other_clean)
    q_bleed = stem_quality(bleed, full, other_path=other_bleed)
    assert q_bleed["bleed"] > q_clean["bleed"]
    assert q_clean["quality"] > q_bleed["quality"]


def test_missing_metrics_are_dropped_not_guessed(tmp_path):
    """A stem where nothing could be measured scores the neutral 0.5, the same
    convention the sub-scores use for unknown inputs."""
    q = stem_quality(tmp_path / "nope.wav", tmp_path / "also-nope.wav")
    assert q["quality"] == 0.5
    assert q["bleed"] is None and q["hf_loss"] is None


def test_hf_loss_detects_a_smeared_top_end(tmp_path):
    """The classic MDX artefact. Measured against the full mix so a genuinely
    dark record is not mistaken for a damaged stem."""
    bright = _noise(tmp_path / "full.wav")            # broadband
    dark = _tone(tmp_path / "dark.wav", 200.0)        # no top end at all
    q = stem_quality(dark, bright)
    assert q["hf_loss"] is not None and q["hf_loss"] > 0.8


def test_quiet_windows_are_the_sections_with_no_voice():
    sections = [
        {"start_sec": 0.0, "end_sec": 10.0, "vocal_presence": 0.0},
        {"start_sec": 10.0, "end_sec": 20.0, "vocal_presence": 0.9},
        {"start_sec": 20.0, "end_sec": 30.0, "vocal_presence": None},
        {"start_sec": 30.0, "end_sec": 40.0, "vocal_presence": 0.05},
    ]
    assert quiet_windows_for(sections) == [(0.0, 10.0), (30.0, 40.0)]


# ── The hard filter ──────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "q.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


def test_an_unusable_top_stem_is_not_offered(db_path):
    """However well it matches. A bleeding, smeared acapella near the top of the
    list is what stops the list being trusted."""
    from database.models import (
        get_conn, init_db, update_stem_quality, upsert_features, upsert_song,
        upsert_stem,
    )
    from matcher.match import score_all_pairs
    init_db(db_path)

    def add(title, quality):
        sid = upsert_song(title, "A", f"u://{title}", 240,
                          status="analysed", db_path=db_path)
        for stem in ("full", "vocals", "instrumental"):
            upsert_features(sid, stem, {
                "bpm": 124.0, "key": "C", "mode": "major", "camelot": "8A",
                "loudness_rms": 0.1, "energy": 0.5, "mfcc": [1.0] * 13,
                "spectral_centroid": 2000.0, "spectral_rolloff": 4000.0,
                "zero_crossing_rate": 0.05,
            }, db_path=db_path)
            upsert_stem(sid, stem, f"/nonexistent/{title}_{stem}.wav",
                        db_path=db_path)
        if quality is not None:
            for stem in ("vocals", "instrumental"):
                update_stem_quality(sid, stem, {"quality": quality},
                                    db_path=db_path)
        return sid

    good = add("Good", 0.9)
    bad = add("Bad", 0.10)      # below STEM_QUALITY_MIN
    unmeasured = add("Old", None)

    score_all_pairs(db_path=db_path, bpm_max_diff=20.0, key_min_score=0.0)
    conn = get_conn(db_path)
    tops = {r["vocal_song_id"] for r in conn.execute(
        "SELECT vocal_song_id FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchall()}
    conn.close()

    assert good in tops
    assert bad not in tops
    # A library analysed before Phase D must not vanish from the list.
    assert unmeasured in tops
