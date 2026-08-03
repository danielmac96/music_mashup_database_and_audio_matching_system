"""T1.3 — key detection must report how much to trust itself.

Key carries the heaviest score weight (config.py KEY_WEIGHT) and is the least
reliable number in the DB, so _step_key now returns the margin between the best
key profile correlation and the runner-up. Presenting an unreliable key as fact
is what makes a bad semitone shift look authoritative.
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


def _tone(np, freqs, sr=22050, secs=5.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (y / max(abs(y).max(), 1e-9)).astype("float32")


# ── the analysis step ────────────────────────────────────────────────────────

def test_step_key_returns_confidence_in_unit_range():
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    from analysis.analyze import _step_key

    out = _step_key(_tone(np, (261.63, 329.63, 392.00)), 22050, 512)
    assert "key_confidence" in out
    assert 0.0 <= out["key_confidence"] <= 1.0


def test_unambiguous_triad_is_more_confident_than_white_noise():
    """The whole point of the number: a clear tonal centre must outrank noise."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    from analysis.analyze import _step_key

    sr = 22050
    triad = _tone(np, (261.63, 329.63, 392.00), sr=sr)
    noise = np.random.default_rng(0).normal(0, 0.3, len(triad)).astype("float32")

    assert (_step_key(triad, sr, 512)["key_confidence"]
            > _step_key(noise, sr, 512)["key_confidence"])


def test_step_key_still_reports_key_mode_and_camelot():
    """Additive change — the existing contract must not move."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    from analysis.analyze import _step_key

    out = _step_key(_tone(np, (261.63, 329.63, 392.00)), 22050, 512)
    assert out["mode"] in ("major", "minor")
    assert out["key"] in ("C", "C#", "D", "D#", "E", "F",
                          "F#", "G", "G#", "A", "A#", "B")
    assert out["camelot"] != "?"


# ── persistence ──────────────────────────────────────────────────────────────

def test_features_table_migrates_to_carry_key_confidence(db_path):
    from database.models import get_conn, init_db
    init_db(db_path)
    conn = get_conn(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(features)")}
    conn.close()
    assert "key_confidence" in cols


def test_key_confidence_round_trips_through_upsert(db_path):
    from database.models import (
        get_features_for_song, init_db, upsert_features, upsert_song,
    )
    init_db(db_path)
    sid = upsert_song("T", "A", "https://sc/kc", 200, "Pop",
                      status="analysed", db_path=db_path)
    upsert_features(sid, "full", {
        "bpm": 120.0, "key": "C", "mode": "major", "camelot": "8B",
        "key_confidence": 0.1234,
    }, db_path=db_path)

    assert get_features_for_song(sid, "full", db_path=db_path)["key_confidence"] \
        == pytest.approx(0.1234)


def test_ranked_list_rows_carry_key_confidence_for_both_sides(db_path):
    """The ranked list must be able to flag an uncertain key without a re-score,
    so it is joined from features rather than frozen onto the candidate row."""
    from database.models import (
        get_candidates_enriched, init_db, upsert_candidate, upsert_features,
        upsert_song,
    )
    init_db(db_path)
    v = upsert_song("V", "A", "https://sc/v", 200, "Pop",
                    status="analysed", db_path=db_path)
    i = upsert_song("I", "B", "https://sc/i", 200, "EDM",
                    status="analysed", db_path=db_path)
    upsert_features(v, "vocals", {"bpm": 120.0, "key": "C", "mode": "major",
                                  "camelot": "8B", "key_confidence": 0.011},
                    db_path=db_path)
    upsert_features(i, "instrumental", {"bpm": 120.0, "key": "C", "mode": "major",
                                        "camelot": "8B", "key_confidence": 0.42},
                    db_path=db_path)
    upsert_candidate(
        {"song_id": v, "title": "V", "artist": "A", "bpm": 120.0, "key": "C",
         "mode": "major", "camelot": "8B", "loudness_rms": 0.1, "energy": 0.5},
        {"song_id": i, "title": "I", "artist": "B", "bpm": 120.0, "key": "C",
         "mode": "major", "camelot": "8B", "loudness_rms": 0.1, "energy": 0.5},
        {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0,
         "energy_score": 0.9, "timbre_score": 0.9},
        db_path=db_path,
    )

    row = get_candidates_enriched(limit=1, db_path=db_path)[0]
    assert row["vocal_key_confidence"] == pytest.approx(0.011)
    assert row["inst_key_confidence"] == pytest.approx(0.42)


def test_features_written_without_key_confidence_read_back_as_none(db_path):
    """Tracks analysed before this change must still load."""
    from database.models import (
        get_features_for_song, init_db, upsert_features, upsert_song,
    )
    init_db(db_path)
    sid = upsert_song("T", "A", "https://sc/old", 200, "Pop",
                      status="analysed", db_path=db_path)
    upsert_features(sid, "full", {"bpm": 120.0, "key": "C", "mode": "major"},
                    db_path=db_path)

    assert get_features_for_song(sid, "full", db_path=db_path)["key_confidence"] is None
