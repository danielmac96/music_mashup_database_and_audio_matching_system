"""P0 — the three measurements that were quietly wrong.

P0.1  `bpm_confidence` was `len(beats) / n_frames`, i.e. beats-per-frame, i.e.
      `bpm / 2580`. It ranged 0.027-0.067 and never meant anything. Everything
      downstream read it as a 0-1 confidence, which put a constant ~0.24 floor
      under `effort_penalty` and made `effort_label`'s "Free" bucket (<= 0.20)
      unreachable for every pair in every library.

P0.2  Section chroma was measured on the FULL MIX for both sides of a pair, so
      the "vocal" side's harmony was dominated by an arrangement that gets
      thrown away. The measured transposition described a record nobody hears.

P0.3  The `key_min_score` pre-filter gated on the key of an isolated ACAPELLA —
      a whole-track mean chroma over one voice, correlated against Krumhansl.
      Pairs died on the least reliable number in the database, before Phase E
      ever got to measure the real harmony.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "test.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


# ── P0.1  beat-grid confidence is a confidence ───────────────────────────────

def _grid(np, bpm, n=64, jitter=0.0, seed=0):
    """Beat times at `bpm` with a relative jitter on each interval."""
    rng = np.random.default_rng(seed)
    ibi = 60.0 / bpm
    steps = ibi * (1.0 + rng.normal(0.0, jitter, n)) if jitter else np.full(n, ibi)
    return np.concatenate([[0.0], np.cumsum(np.abs(steps))])


def test_locked_grid_is_confident_at_every_tempo():
    """The old formula returned bpm/2580 — 0.027 at 70 BPM, 0.067 at 174. A
    machine-perfect grid must read as near-certain regardless of tempo."""
    np = pytest.importorskip("numpy")
    from analysis.analyze import beat_grid_confidence

    for bpm in (70, 100, 128, 174):
        assert beat_grid_confidence(_grid(np, bpm)) > 0.9, bpm


def test_confidence_falls_as_the_grid_wanders():
    np = pytest.importorskip("numpy")
    from analysis.analyze import beat_grid_confidence

    locked = beat_grid_confidence(_grid(np, 128, jitter=0.0))
    loose = beat_grid_confidence(_grid(np, 128, jitter=0.03, seed=1))
    lost = beat_grid_confidence(_grid(np, 128, jitter=0.25, seed=2))

    assert locked > loose > lost
    assert lost < 0.2


def test_too_few_beats_is_no_confidence_rather_than_full_confidence():
    np = pytest.importorskip("numpy")
    from analysis.analyze import beat_grid_confidence

    assert beat_grid_confidence(np.array([0.0, 0.5, 1.0])) == 0.0
    assert beat_grid_confidence([]) == 0.0


def test_salience_discounts_a_grid_that_misses_the_transients():
    """A perfectly even grid sitting between the kicks is still the wrong grid."""
    np = pytest.importorskip("numpy")
    from analysis.analyze import beat_grid_confidence

    beat_times = _grid(np, 120, n=64)
    frames = np.arange(len(beat_times)) * 10
    env = np.ones(int(frames[-1]) + 8, dtype=float)

    on_beat = env.copy()
    on_beat[frames] = 8.0                      # all the energy is on the beats

    assert beat_grid_confidence(beat_times, on_beat, frames) > 0.9
    # A flat envelope is no evidence the beats are where the music is.
    assert beat_grid_confidence(beat_times, env, frames) == 0.0


# ── P0.1  the consequence: "Free" has to be reachable ────────────────────────

def test_an_effortless_pair_is_labelled_free():
    """The acceptance criterion. Same tempo, same key, best-in-library grids:
    there is nothing to do in the DAW, so the effort bucket must say so.

    Before P0.1 the floor was 0.95*0.15 + 0.98*0.10 = 0.2405, so this pair — and
    every other pair in every library — was labelled "Light" at best.
    """
    from matcher.effort import effort_label, effort_penalty

    ideal = {"bpm": 128.0, "camelot": "8A",
             "bpm_confidence": 0.95, "key_confidence": 0.9}
    total, _parts = effort_penalty(ideal, dict(ideal), stretch=1.0, semitones=0)

    assert total < 0.20
    assert effort_label(total) == "Free"


def test_library_relative_confidence_keeps_the_effort_axis_alive():
    """Even on an estimator whose absolute scale is tiny, the BEST tracks in the
    library must cost nothing — that is what conf_pct buys."""
    import numpy as np

    from matcher.effort import effort_label, effort_penalty
    from matcher.match import LibraryStats

    # A library whose key confidences all sit around 0.01, as the Krumhansl
    # estimator actually produces.
    stats = LibraryStats(conf={
        "bpm": np.sort(np.linspace(0.2, 0.95, 32)),
        "key": np.sort(np.linspace(0.001, 0.02, 32)),
    })
    norm = lambda kind, v: float(stats.conf_pct(kind, v))  # noqa: E731

    best = {"bpm": 128.0, "camelot": "8A",
            "bpm_confidence": 0.95, "key_confidence": 0.02}
    worst = {"bpm": 128.0, "camelot": "8A",
             "bpm_confidence": 0.2, "key_confidence": 0.001}

    good, _ = effort_penalty(best, dict(best), 1.0, 0, conf_norm=norm)
    bad, _ = effort_penalty(worst, dict(worst), 1.0, 0, conf_norm=norm)

    assert effort_label(good) == "Free"
    # Same tempo and same key, so there is no stretch and no transpose to pay
    # for — the whole difference is that these two grids are the worst in the
    # library, which is beatgridding time rather than damaged audio.
    assert effort_label(bad) == "Light"
    assert bad > good


def test_conf_pct_is_a_percentile_and_degrades_to_the_raw_value():
    import numpy as np

    from matcher.match import LibraryStats

    stats = LibraryStats(conf={"bpm": np.sort(np.linspace(0.0, 1.0, 100))})
    assert stats.conf_pct("bpm", 0.0) == pytest.approx(0.01, abs=0.02)
    assert stats.conf_pct("bpm", 0.5) == pytest.approx(0.5, abs=0.02)
    assert stats.conf_pct("bpm", 1.0) == pytest.approx(1.0)

    # A library too small to have a distribution hands the raw value back rather
    # than collapsing every track onto the same percentile.
    tiny = LibraryStats(conf={"bpm": np.array([0.3, 0.4])})
    assert float(tiny.conf_pct("bpm", 0.37)) == pytest.approx(0.37)
    assert float(LibraryStats().conf_pct("key", 0.42)) == pytest.approx(0.42)


# ── P0.2  harmony is measured on the stems that get layered ──────────────────

def _chroma(np, *pitch_classes):
    v = np.zeros(12)
    for pc in pitch_classes:
        v[pc % 12] = 1.0
    return list(v / np.linalg.norm(v))


def test_section_harmony_prefers_the_stem_chroma_over_the_full_mix():
    """The vocal side must be judged on what is SUNG, not on the record it was
    lifted from. Here the full mixes agree and the stems do not, so reading the
    full mix would report a perfect fit for a pair that clashes."""
    np = pytest.importorskip("numpy")
    from matcher.harmony import section_harmony

    agree = _chroma(np, 0, 4, 7)          # C major triad, in both full mixes
    vocal_sings = _chroma(np, 1, 5, 8)    # the acapella is a semitone up
    bed_plays = _chroma(np, 0, 4, 7)

    both_full_mix = section_harmony({"chroma": agree}, {"chroma": agree})
    from_stems = section_harmony(
        {"chroma": agree, "chroma_vocal": vocal_sings},
        {"chroma": agree, "chroma_bed": bed_plays})

    assert both_full_mix["shift"] == 0
    # Measured on the stems, the bed needs moving to meet the vocal.
    assert from_stems["shift"] != 0
    assert from_stems["known"] is True


def test_section_harmony_falls_back_to_full_mix_chroma_when_stems_are_absent():
    """A library analysed before P0.2 must rank exactly as it did."""
    np = pytest.importorskip("numpy")
    from matcher.harmony import section_harmony

    c = _chroma(np, 0, 4, 7)
    old = section_harmony({"chroma": c}, {"chroma": c})

    assert old["known"] is True
    assert old["shift"] == 0
    assert old["harmonic_fit"] > 0.9


def test_sections_round_trip_the_per_stem_chroma(db_path):
    np = pytest.importorskip("numpy")
    from database.models import get_sections, init_db, replace_sections, upsert_song

    init_db(db_path)
    sid = upsert_song("T", "A", "https://sc/p02", 200, "Pop",
                      status="analysed", db_path=db_path)
    replace_sections(sid, [{
        "start_sec": 0.0, "end_sec": 30.0, "label": "chorus",
        "energy": 0.8, "vocal_presence": 0.7, "repetition": 2, "confidence": 0.9,
        "chroma": _chroma(np, 0, 4, 7),
        "chroma_vocal": _chroma(np, 0, 7),
        "chroma_bed": _chroma(np, 0, 3, 7),
        "bass_chroma": _chroma(np, 0),
    }], db_path=db_path)

    s = get_sections(sid, db_path=db_path)[0]
    assert len(s["chroma_vocal"]) == 12
    assert len(s["chroma_bed"]) == 12
    assert s["chroma_vocal"] != s["chroma_bed"]


def test_sections_written_without_stem_chroma_read_back_absent(db_path):
    np = pytest.importorskip("numpy")
    from database.models import get_sections, init_db, replace_sections, upsert_song

    init_db(db_path)
    sid = upsert_song("T", "A", "https://sc/p02b", 200, "Pop",
                      status="analysed", db_path=db_path)
    replace_sections(sid, [{
        "start_sec": 0.0, "end_sec": 30.0, "label": "chorus",
        "energy": 0.8, "vocal_presence": 0.7, "repetition": 2, "confidence": 0.9,
        "chroma": _chroma(np, 0, 4, 7),
    }], db_path=db_path)

    s = get_sections(sid, db_path=db_path)[0]
    assert "chroma_vocal" not in s
    assert "chroma_bed" not in s
    assert len(s["chroma"]) == 12


# ── P0.3  the gate reads the full-mix key ────────────────────────────────────

def test_full_mix_key_replaces_the_stem_key_for_matching():
    from matcher.match import _with_full_bpm

    vocal_stem = {"song_id": 1, "stem_type": "vocals", "bpm": 63.0,
                  "camelot": "3B", "key": "F", "mode": "major",
                  "key_confidence": 0.002, "mfcc": [1.0] * 13,
                  "loudness_rms": 0.02}
    full = {"song_id": 1, "bpm": 126.0, "bpm_confidence": 0.9,
            "camelot": "8A", "key": "A", "mode": "minor",
            "key_confidence": 0.31}

    out = _with_full_bpm(vocal_stem, {1: full})

    assert out["bpm"] == 126.0
    assert out["camelot"] == "8A"
    assert out["key_confidence"] == 0.31
    # The stem's own readings are kept, not lost.
    assert out["stem_bpm"] == 63.0
    assert out["stem_camelot"] == "3B"
    # Timbre and loudness stay stem-derived — those are measured, not estimated,
    # and they are what is actually heard layered.
    assert out["mfcc"] == [1.0] * 13
    assert out["loudness_rms"] == 0.02


def test_a_pair_the_acapella_key_would_have_rejected_now_survives_the_gate():
    """The point of P0.3: the pre-filter must not run on the acapella's key."""
    from matcher.match import _passes_filter, _with_full_bpm, camelot_score

    vocal_stem = {"song_id": 1, "bpm": 128.0, "camelot": "2B"}   # noisy estimate
    vocal_full = {"song_id": 1, "bpm": 128.0, "camelot": "8A"}   # the real key
    bed = {"song_id": 2, "bpm": 128.0, "camelot": "8A"}

    assert camelot_score("2B", "8A") < 0.55        # would have been rejected
    assert not _passes_filter(vocal_stem, bed, 10.0, 0.55)

    fixed = _with_full_bpm(vocal_stem, {1: vocal_full})
    assert _passes_filter(fixed, bed, 10.0, 0.55)


def test_full_mix_row_without_a_key_leaves_the_stem_key_alone():
    """A track whose full-mix key step failed must not have its key blanked."""
    from matcher.match import _with_full_bpm

    stem = {"song_id": 1, "bpm": 120.0, "camelot": "5A", "key": "C"}
    out = _with_full_bpm(stem, {1: {"song_id": 1, "bpm": 121.0}})

    assert out["bpm"] == 121.0
    assert out["camelot"] == "5A"
    assert "stem_camelot" not in out
