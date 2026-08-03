"""T1.4 — bar 1 must actually be bar 1.

beat_times were stored, but every consumer assumed "every 4th beat from the
first detected beat" is a downbeat. When librosa latches mid-bar the whole grid
is 1–2 beats off, which throws hook boundaries and snapping. _pick_beat_phase
finds which of the 4 candidate phases carries the accents.
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


def _envelope(np, beat_frames, accent_at_phase, strong=10.0, weak=1.0):
    """Onset envelope that is loud only on beats whose index % 4 == phase."""
    env = np.zeros(int(beat_frames[-1]) + 8, dtype=float)
    for i, f in enumerate(beat_frames):
        env[int(f)] = strong if i % 4 == accent_at_phase else weak
    return env


# ── phase detection ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_picks_the_phase_carrying_the_accents(phase):
    np = pytest.importorskip("numpy")
    from analysis.analyze import _pick_beat_phase

    beat_frames = np.arange(0, 32) * 10          # 32 evenly spaced beats
    env = _envelope(np, beat_frames, accent_at_phase=phase)

    assert _pick_beat_phase(env, beat_frames) == phase


def test_phase_is_zero_when_no_beat_is_accented():
    """A flat envelope has no information; default to 0 rather than inventing one."""
    np = pytest.importorskip("numpy")
    from analysis.analyze import _pick_beat_phase

    beat_frames = np.arange(0, 32) * 10
    env = np.ones(int(beat_frames[-1]) + 8, dtype=float)

    assert _pick_beat_phase(env, beat_frames) == 0


def test_phase_degrades_to_zero_on_too_few_beats():
    np = pytest.importorskip("numpy")
    from analysis.analyze import _pick_beat_phase

    assert _pick_beat_phase(np.ones(10), np.array([])) == 0
    assert _pick_beat_phase(np.ones(10), np.array([0, 3])) == 0


def test_phase_is_always_a_valid_bar_position():
    np = pytest.importorskip("numpy")
    from analysis.analyze import _pick_beat_phase

    rng = np.random.default_rng(7)
    for _ in range(20):
        beat_frames = np.arange(0, 40) * 8
        env = rng.random(int(beat_frames[-1]) + 8)
        assert _pick_beat_phase(env, beat_frames) in (0, 1, 2, 3)


# ── persistence ──────────────────────────────────────────────────────────────

def _reload_tracks_route(tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import tracks
    importlib.reload(tracks)
    return tracks, models


def test_track_payload_exposes_beat_phase_and_key_confidence(tmp_path, monkeypatch):
    """_FEATURE_FIELDS is a whitelist — a column the UI reads but which is not
    listed here is silently always undefined, so the ⚠ chips never render."""
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/fp", 200, "Pop",
                             status="analysed")
    models.upsert_features(sid, "full", {
        "bpm": 128.0, "key": "C", "mode": "major", "camelot": "8B",
        "key_confidence": 0.012, "beat_phase": 3,
    })

    feats = tracks._features_by_song("full")[sid]
    assert feats["key_confidence"] == pytest.approx(0.012)
    assert feats["beat_phase"] == 3


def test_waveform_beat_phase_follows_the_same_stem_as_the_beats(tmp_path, monkeypatch):
    """The vocals stem falls back to the full track's beats when its own tempo is
    low-confidence. The phase must fall back with them — a phase read off a
    different beat array points at the wrong beat."""
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/wf", 200, "Pop",
                             status="analysed")
    # Vocals: beats present but confidence below the fallback threshold.
    models.upsert_features(sid, "vocals", {
        "bpm": 128.0, "bpm_confidence": 0.0,
        "beat_times": [0.1, 0.6, 1.1, 1.6], "beat_phase": 1,
    })
    models.upsert_features(sid, "full", {
        "bpm": 128.0, "bpm_confidence": 0.9,
        "beat_times": [0.0, 0.5, 1.0, 1.5], "beat_phase": 2,
    })

    out = tracks.get_waveform(sid, stem="vocals")
    assert out["beat_source"] == "full"
    assert out["beat_times"] == [0.0, 0.5, 1.0, 1.5]
    assert out["beat_phase"] == 2, "phase must come from the stem that supplied the beats"


def test_override_sets_phase_on_the_stem_the_user_was_looking_at(tmp_path, monkeypatch):
    """alt+click declares 'this beat is bar 1'. Detection is a guess; the ear
    is not, so the manual value must stick to the stem being viewed."""
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/ov", 200, "Pop",
                             status="analysed")
    models.upsert_features(sid, "full", {"bpm": 128.0, "beat_phase": 0})
    models.upsert_features(sid, "vocals", {"bpm": 128.0, "beat_phase": 0})

    out = tracks.set_beat_phase(sid, tracks.BeatPhaseUpdate(stem="vocals", phase=3))

    assert out["beat_phase"] == 3
    assert models.get_features_for_song(sid, "vocals")["beat_phase"] == 3
    assert models.get_features_for_song(sid, "full")["beat_phase"] == 0, \
        "override must not leak onto a stem the user was not looking at"


@pytest.mark.parametrize("bad", [-1, 4, 99])
def test_override_rejects_a_position_outside_the_bar(tmp_path, monkeypatch, bad):
    from fastapi import HTTPException
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/ov2", 200, "Pop",
                             status="analysed")
    models.upsert_features(sid, "full", {"bpm": 128.0})

    with pytest.raises(HTTPException) as e:
        tracks.set_beat_phase(sid, tracks.BeatPhaseUpdate(stem="full", phase=bad))
    assert e.value.status_code == 400


def test_override_404s_when_the_stem_has_no_features(tmp_path, monkeypatch):
    from fastapi import HTTPException
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/ov3", 200, "Pop",
                             status="analysed")

    with pytest.raises(HTTPException) as e:
        tracks.set_beat_phase(sid, tracks.BeatPhaseUpdate(stem="full", phase=1))
    assert e.value.status_code == 404


def test_beat_phase_round_trips_and_defaults_to_zero(db_path):
    from database.models import (
        get_features_for_song, init_db, upsert_features, upsert_song,
    )
    init_db(db_path)
    sid = upsert_song("T", "A", "https://sc/bp", 200, "Pop",
                      status="analysed", db_path=db_path)

    upsert_features(sid, "full", {"bpm": 120.0, "beat_phase": 2}, db_path=db_path)
    assert get_features_for_song(sid, "full", db_path=db_path)["beat_phase"] == 2

    # A track analysed before this existed must still render on phase 0.
    sid2 = upsert_song("T2", "A", "https://sc/bp2", 200, "Pop",
                       status="analysed", db_path=db_path)
    upsert_features(sid2, "full", {"bpm": 120.0}, db_path=db_path)
    assert get_features_for_song(sid2, "full", db_path=db_path)["beat_phase"] == 0
