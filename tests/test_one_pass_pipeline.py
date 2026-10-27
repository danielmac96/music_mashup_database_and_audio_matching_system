"""One download, one separation, one analysis, one cut, one score.

The analysis stage is the one place that already decodes every stem, so it is
where the per-frame sidecars (analysis/timeseries.py) are written. It also
gains the bass stem, analysed for its chroma alone — no `features` row, because
nothing reads one for the bass.

Isolation: `config.DB_PATH` binds at import, so setting MASHUP_DB_PATH without
reloading would do nothing and the stage — which calls `upsert_features` and
`timeseries.write` on its own, with no `db_path=` to give it — would write to
the user's real library. The `env` fixture sets the paths and reloads the chain
that binds them, the way tests/test_quick_tier.py does.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_PATH_VARS = ("MASHUP_AUDIO_ROOT", "MASHUP_DB_PATH", "MASHUP_SETTINGS_DIR")


def _has_essentia() -> bool:
    try:
        from analysis.essentia_groups import available
        return available()
    except Exception:  # noqa: BLE001
        return False


needs_essentia = pytest.mark.skipif(not _has_essentia(),
                                    reason="essentia not installed")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """An isolated library: paths set, then every module that binds one
    reloaded. Restored at teardown so a later test file in the same invocation
    does not inherit a tmp_path pytest will delete."""
    prev = {k: os.environ.get(k) for k in _PATH_VARS}
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    import database.models as models
    import api.workers.stages as stages
    for mod in (config, models, stages):
        importlib.reload(mod)
    models.init_db()
    from analysis import decode
    decode.clear()
    try:
        yield models, stages, tmp_path / "mashup.db"
    finally:
        for key, val in prev.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        for mod in (config, models, stages):
            importlib.reload(mod)
        decode.clear()


def _essentia_payload(stem_type: str) -> dict:
    """The minimum essentia_core_complete accepts, per stem."""
    return {
        "rhythm": {"bpm": 124.0, "bpm_confidence": 0.8, "beat_times": [0.0, 0.48],
                   "beat_phase": 0},
        "tonal": {"key": "A", "mode": "minor", "camelot": "8A",
                  "key_confidence": 0.5, "hpcp": [0.0] * 12},
        "loudness": {"loudness_rms": 0.1},
        "spectral": {"mfcc": [0.0] * 13, "spectral_centroid": 1000.0,
                     "spectral_rolloff": 4000.0, "zero_crossing_rate": 0.05,
                     "energy": 0.01, "band_energy": [0.125] * 8,
                     "waveform_rms": [0.1] * 360},
    }


@pytest.fixture
def seeded_song(env, tmp_path, monkeypatch):
    """One song row with four silent stem files on disk, and the sidecars
    pointed at tmp_path."""
    import soundfile as sf

    from analysis import timeseries as ts
    models, _stages, db = env
    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")

    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    paths = {}
    for stem in ("full", "vocals", "instrumental", "bass"):
        p = tmp_path / f"{stem}.wav"
        sf.write(str(p), y, sr)
        paths[stem] = str(p)
    song_id = models.upsert_song(title="T", artist="A", source_url="u://1",
                                 raw_path=paths["full"], duration_secs=2,
                                 db_path=db)
    for stem, p in paths.items():
        models.upsert_stem(song_id, stem, p,
                           separator="demucs:htdemucs:4:unified", db_path=db)
    return song_id, paths


def _fake_essentia(captured: dict, fill_frames: bool = True):
    """A stand-in `_run_essentia` that records the stems it was asked for and
    the arguments that decide what Essentia is asked to compute."""
    from analysis import timeseries as ts

    def fake(song_id_, stem_type, path, key, on_progress, extra_steps=(),
             frames_out=None, force_steps=(), only_steps=()):
        captured.setdefault("stems", []).append(stem_type)
        captured.setdefault("calls", {})[stem_type] = {
            "extra_steps": tuple(extra_steps), "force_steps": tuple(force_steps),
            "only_steps": tuple(only_steps), "frames": frames_out is not None}
        if frames_out is not None and fill_frames:
            frames_out["chroma"] = np.full((12, 20), 0.2887, dtype=np.float32)
            frames_out["chroma_step"] = ts.STEP_SECS
            frames_out["mfcc"] = np.zeros((13, 20), dtype=np.float32)
            frames_out["bands"] = np.ones((8, 20), dtype=np.float32)
            frames_out["curves"] = np.zeros((5, 20), dtype=np.float32)
            frames_out["spectral_step"] = ts.STEP_SECS
        return _essentia_payload(stem_type), False

    return fake


def _essentia_mode(stages, monkeypatch, captured, fill_frames=True):
    monkeypatch.setattr(stages, "_run_essentia",
                        _fake_essentia(captured, fill_frames))
    monkeypatch.setattr(stages, "effective_analyzer",
                        lambda: ("essentia", "essentia"))
    monkeypatch.setattr(stages, "require_analyzer", lambda: None)


# ── The sidecars ──────────────────────────────────────────────────────────────

def test_analysis_writes_a_sidecar_per_stem(monkeypatch, seeded_song, env):
    """The analysis stage is the only place that decodes every stem, so it is
    where the frames are kept."""
    from analysis import timeseries as ts
    models, stages, db = env
    song_id, stem_paths = seeded_song

    captured: dict = {}
    _essentia_mode(stages, monkeypatch, captured)

    stages._analyze_stems(song_id, stem_paths, stages._ANALYSIS_STEM_ORDER, None)

    assert captured["stems"] == list(stages._ANALYSIS_STEM_ORDER)
    for stem in ("full", "vocals", "instrumental", "bass"):
        assert ts.read(song_id, stem, "legacy_2stem") is not None, stem
    # The bass stem is analysed for its chroma only — no features row.
    assert models.get_features_for_song(song_id, "bass", db_path=db) is None
    assert models.get_features_for_song(song_id, "full", db_path=db) is not None


def test_bass_is_analysed_tonal_only():
    from api.workers import stages

    assert "bass" in stages._ANALYSIS_STEM_ORDER
    assert stages._TONAL_ONLY_STEMS == ("bass",)


def test_a_tonal_only_stem_asks_for_the_tonal_group_alone(monkeypatch,
                                                          seeded_song, env):
    """`only_steps` is what keeps the bass's rhythm, loudness, spectral and
    effnet groups from running for a row nothing reads."""
    _models, stages, _db = env
    song_id, stem_paths = seeded_song

    captured: dict = {}
    _essentia_mode(stages, monkeypatch, captured)
    stages._analyze_stems(song_id, stem_paths, stages._ANALYSIS_STEM_ORDER, None)

    assert captured["calls"]["bass"]["only_steps"] == ("tonal",)
    assert captured["calls"]["bass"]["extra_steps"] == ()
    assert captured["calls"]["full"]["only_steps"] == ()
    assert captured["calls"]["full"]["extra_steps"] == ("effnet",)
    assert captured["calls"]["vocals"]["extra_steps"] == ("melody",)


def test_a_missing_sidecar_forces_the_two_steps_that_carry_frames(monkeypatch,
                                                                  seeded_song,
                                                                  env):
    """The payloads can be cached while the sidecar is not on disk. The frame
    steps are then re-run; a stem that already has its sidecar forces nothing."""
    from analysis import timeseries as ts
    _models, stages, _db = env
    song_id, stem_paths = seeded_song

    captured: dict = {}
    _essentia_mode(stages, monkeypatch, captured)
    stages._analyze_stems(song_id, stem_paths, ("full",), None)
    assert captured["calls"]["full"]["force_steps"] == ("tonal", "spectral")
    assert ts.read(song_id, "full", "legacy_2stem") is not None

    captured.clear()
    stages._analyze_stems(song_id, stem_paths, ("full",), None)
    assert captured["calls"]["full"]["force_steps"] == ()


def test_the_quick_tier_writes_no_sidecars(monkeypatch, seeded_song, env):
    """The quick tier exists to put BPM and key in the library in seconds: no
    frames, and no extras either."""
    from analysis import timeseries as ts
    _models, stages, _db = env
    song_id, stem_paths = seeded_song

    captured: dict = {}
    _essentia_mode(stages, monkeypatch, captured)
    stages._analyze_stems(song_id, stem_paths, ("full",), None,
                          stage_grp="quick", gate="quick")

    assert captured["calls"]["full"]["frames"] is False
    assert captured["calls"]["full"]["extra_steps"] == ()
    assert ts.read(song_id, "full", "legacy_2stem") is None


def test_keep_frames_off_writes_no_sidecars(monkeypatch, seeded_song, env):
    from analysis import timeseries as ts
    _models, stages, _db = env
    song_id, stem_paths = seeded_song

    monkeypatch.setenv("MASHUP_KEEP_ANALYSIS_FRAMES", "0")
    captured: dict = {}
    _essentia_mode(stages, monkeypatch, captured)
    stages._analyze_stems(song_id, stem_paths, ("full",), None)

    assert captured["calls"]["full"]["frames"] is False
    assert ts.read(song_id, "full", "legacy_2stem") is None


def test_keep_frames_defaults_on():
    import config

    assert config.current_keep_frames() is True


def test_a_frames_failure_does_not_fail_the_stem(monkeypatch, seeded_song, env):
    """The sidecar is an optimisation: a stem whose frames cannot be built is
    still analysed."""
    from analysis import timeseries as ts
    models, stages, db = env
    song_id, stem_paths = seeded_song

    captured: dict = {}
    _essentia_mode(stages, monkeypatch, captured)
    monkeypatch.setattr(ts, "write",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))

    analysed, failed = stages._analyze_stems(song_id, stem_paths, ("full",), None)

    assert analysed == ["full"] and failed == []
    assert models.get_features_for_song(song_id, "full", db_path=db) is not None


# ── Against Essentia itself ───────────────────────────────────────────────────

@needs_essentia
def test_only_steps_runs_one_group_and_still_fills_the_frames(tmp_path):
    """A tonal-only stem: one group's payload, the chroma frames, and nothing
    else computed or cached."""
    from analysis.essentia_groups import analyze_file_essentia

    sr = 44100
    secs = 6.0
    t = np.arange(int(secs * sr)) / sr
    y = sum(0.2 * np.sin(2 * np.pi * 261.63 * 2 ** (iv / 12) * t)
            for iv in (0, 4, 7)).astype(np.float32)
    wav = tmp_path / "tonal.wav"
    import soundfile as sf
    sf.write(str(wav), y, sr)

    class _MemCache:
        def __init__(self):
            self.store = {}

        def get(self, step):
            return self.store.get(step)

        def put(self, step, payload, ms):
            self.store[step] = payload

    cache = _MemCache()
    frames: dict = {}
    timings: dict = {}
    out = analyze_file_essentia(wav, cache=cache, timings=timings,
                                frames_out=frames, only_steps=("tonal",))

    assert set(out) == {"tonal"}
    assert set(cache.store) == {"tonal"}
    assert "rhythm" not in timings and "spectral" not in timings
    assert frames["chroma"].shape[0] == 12 and frames["chroma"].shape[1] > 0
    assert "mfcc" not in frames


@needs_essentia
def test_the_bass_stem_gets_a_chroma_sidecar_and_no_features_row(monkeypatch,
                                                                 seeded_song,
                                                                 env, tmp_path):
    """The tonal-only branch against Essentia itself: the bass's sidecar holds
    the chroma alone — the spectral group never ran — and no `features` row is
    written for it."""
    import soundfile as sf

    from analysis import timeseries as ts
    models, stages, db = env
    song_id, stem_paths = seeded_song
    monkeypatch.setenv("MASHUP_ANALYZER", "essentia")

    # Silence measures nothing; a bass line does. 44.1 kHz, as a stem is.
    sr = 44100
    t = np.arange(int(8.0 * sr)) / sr
    y = sum(0.3 * np.sin(2 * np.pi * 55.0 * 2 ** (iv / 12) * t)
            for iv in (0, 7, 3)).astype(np.float32)
    bass = tmp_path / "bass_real.wav"
    sf.write(str(bass), y, sr)
    stem_paths = {"bass": str(bass)}
    models.upsert_stem(song_id, "bass", str(bass), db_path=db)

    analysed, failed = stages._analyze_stems(song_id, stem_paths, ("bass",), None)

    assert analysed == ["bass"] and failed == []
    frames = ts.read(song_id, "bass", "legacy_2stem")
    assert frames is not None
    assert frames.matrices() == {"chroma": 12}        # spectral never ran
    assert frames.step == pytest.approx(ts.STEP_SECS)
    assert models.get_features_for_song(song_id, "bass", db_path=db) is None
