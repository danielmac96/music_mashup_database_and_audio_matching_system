"""Phase 2 of the analysis overhaul: the Essentia analyser's groups, how they
project into the features table, and the librosa | shadow | essentia switch.

Tests that need Essentia itself skip where it is not installed (native Windows,
and the Windows CI leg); the projection, key/chroma conventions and the
switch's fallback are pure and run everywhere."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _has_essentia() -> bool:
    try:
        from analysis.essentia_groups import available
        return available()
    except Exception:  # noqa: BLE001
        return False


needs_essentia = pytest.mark.skipif(not _has_essentia(), reason="essentia not installed")


def _track(path: Path, secs: float = 50.0, bpm: float = 120.0, sr: int = 44100,
           root_hz: float = 261.63) -> Path:
    """Kick on every beat (louder on the bar), a major triad on ``root_hz``."""
    import soundfile as sf
    n = int(secs * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for i, bt in enumerate(np.arange(0, secs, 60.0 / bpm)):
        s = int(bt * sr)
        e = min(n, s + int(0.12 * sr))
        tt = np.arange(e - s) / sr
        y[s:e] += (1.0 if i % 4 == 0 else 0.55) * np.sin(
            2 * np.pi * (55 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 25)
    for iv in (0, 4, 7):
        y += 0.12 * np.sin(2 * np.pi * root_hz * 2 ** (iv / 12) * t)
    y = (y / np.abs(y).max() * 0.8).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([y, y], axis=1), sr)
    return path


# ── Pure: conventions and projection ─────────────────────────────────────────

def test_hpcp_is_folded_to_twelve_bins_starting_at_c():
    from analysis.essentia_groups import hpcp_to_chroma
    frames = np.zeros((4, 36))
    frames[:, 9] = 1.0          # A-referenced, 3 bins/semitone: bin 9 = semitone 3 = C
    frames[:, 10] = 0.5         # its upper neighbour folds into the same pitch
    chroma = hpcp_to_chroma(frames)
    assert chroma.shape == (12,) and int(np.argmax(chroma)) == 0
    assert np.isclose(np.linalg.norm(chroma), 1.0)
    frames12 = np.zeros((2, 12))
    frames12[:, 0] = 1.0        # A in a 12-bin HPCP
    assert int(np.argmax(hpcp_to_chroma(frames12))) == 9


def test_keys_are_spelled_as_the_library_spells_them():
    from analysis.essentia_groups import normalise_key
    assert normalise_key("Bb") == "A#"
    assert normalise_key("Eb") == "D#"
    assert normalise_key("C") == "C"
    assert normalise_key("nonsense") is None


def test_true_peak_finds_the_full_oversampled_maximum():
    from scipy.signal import resample_poly
    from analysis.essentia_groups import true_peak_dbtp
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((44100 * 3, 2)) * 0.1).astype(np.float32)
    x[5000, 0] = 0.95
    x[5001, 0] = -0.9          # an inter-sample overshoot between these two
    full = max(float(np.max(np.abs(resample_poly(x[:, c].astype(float), 4, 1))))
               for c in range(2))
    assert true_peak_dbtp(x) == pytest.approx(20 * np.log10(full), abs=0.01)


def test_onset_rate_counts_the_bursts():
    from analysis.essentia_groups import HOP, SR_FRAMES, onset_rate_from_envelope
    secs = 10.0
    env = np.zeros(int(secs * SR_FRAMES / HOP))
    env[:: int(0.5 * SR_FRAMES / HOP)] = 1.0     # two onsets a second
    assert onset_rate_from_envelope(env, secs) == pytest.approx(2.0, abs=0.15)


def _payloads():
    return {
        "rhythm": {"bpm": 124.0, "bpm_confidence": 0.8, "beat_times": [0.5, 1.0],
                   "beat_phase": 1, "bpm_candidates": {"percival": 124.1},
                   "danceability": 1.4, "onset_rate": 3.2, "rhythm_confidence": 0.0},
        "tonal": {"key": "A", "mode": "minor", "camelot": "8A", "key_confidence": 0.5,
                  "key_strength": 0.7, "key_consensus": 0.75,
                  "key_candidates": {"edma": ["A", "minor", 0.7]}, "tuning_hz": 440.1,
                  "chords": {"key": "A"}, "hpcp": [0.1] * 12},
        "loudness": {"loudness_rms": 0.1, "lufs": -9.5, "lra": 4.0, "true_peak": -0.3,
                     "replay_gain": -6.0, "dynamic_complexity": 2.1, "crest_db": 9.0,
                     "stereo_width": 0.2},
        "spectral": {"mfcc": [0.0] * 13, "spectral_centroid": 2000.0,
                     "spectral_rolloff": 4000.0, "zero_crossing_rate": 0.05,
                     "energy": 1e-4, "band_energy": [0.125] * 8, "bands3": [0.3, 0.6, 0.1],
                     "waveform_rms": [0.5] * 360, "spectral": {"dissonance": 0.3}},
    }


def test_essentia_projects_onto_exactly_the_librosa_columns(tmp_path):
    """core_from_essentia must fill every key analyze_file (+ band occupancy)
    fills, and nothing else: upsert_features writes whatever it is given."""
    from analysis.analyze import analyze_file
    from analysis.project import core_from_essentia
    import soundfile as sf
    p = tmp_path / "t.wav"
    sf.write(str(p), (np.random.default_rng(1).standard_normal(22050 * 6) * 0.1)
             .astype(np.float32), 22050)
    librosa_keys = set(analyze_file(p)) | {"band_energy"}
    assert set(core_from_essentia(_payloads())) == librosa_keys


def test_extras_are_unmeasured_when_essentia_did_not_run():
    from analysis.project import EXTRA_COLUMNS, extras_from_essentia
    ex = extras_from_essentia(_payloads(), "librosa")
    assert set(ex) == set(EXTRA_COLUMNS)
    assert ex["lufs"] == -9.5 and ex["analyzer"] == "librosa"
    none = extras_from_essentia({}, "librosa")
    assert none["analyzer"] == "librosa"
    assert all(v is None for k, v in none.items() if k != "analyzer")


# ── The switch ────────────────────────────────────────────────────────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    import database.models as models
    import api.jobs
    import api.workers.stages
    import api.routes.analysis
    import api.routes.settings
    for mod in (config, models, api.jobs, api.workers.stages,
                api.routes.analysis, api.routes.settings):
        importlib.reload(mod)
    models.init_db()
    from analysis import decode
    decode.clear()
    import api.workers.hook_worker as hook_worker
    monkeypatch.setattr(hook_worker, "warm_hooks", lambda *a, **k: {})
    monkeypatch.setattr(api.workers.stages, "_measure_stem_quality", lambda *a, **k: None)
    app = FastAPI()
    app.include_router(api.routes.analysis.router, prefix="/api/analysis")
    app.include_router(api.routes.settings.router, prefix="/api/settings")
    return models, api.workers.stages, TestClient(app), tmp_path


def _song(models, tmp):
    mix = _track(tmp / "mix.wav")
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "stemmed", raw_path=str(mix))
    models.upsert_stem(song, "full", str(mix))
    return song


def _full_row(models, song):
    return models.get_features_for_song(song, "full")


def test_config_reads_the_analyzer_live(env, monkeypatch):
    import config
    monkeypatch.delenv("MASHUP_ANALYZER", raising=False)
    assert config.current_analyzer() == "essentia"        # the default
    monkeypatch.setenv("MASHUP_ANALYZER", "Shadow")
    assert config.current_analyzer() == "shadow"
    monkeypatch.setenv("MASHUP_ANALYZER", "nonsense")
    assert config.current_analyzer() == "essentia"
    assert config.current_essentia_key_profile() == "edma"
    assert config.current_essentia_rhythm_method() == "degara"


def test_without_essentia_analysis_is_refused_not_degraded(env, monkeypatch):
    models, stages, client, tmp = env
    import analysis.essentia_groups as eg
    monkeypatch.setattr(eg, "available", lambda: False)
    monkeypatch.setenv("MASHUP_ANALYZER", "essentia")
    assert stages.effective_analyzer() == ("essentia", "essentia")
    song = _song(models, tmp)
    with pytest.raises(stages.StageError, match="Docker or WSL2"):
        stages.do_analyze(song)
    row = models.get_song(song)
    assert row["status"] == "error_analysis" and "Docker or WSL2" in row["last_error"]
    assert _full_row(models, song) is None                 # no librosa row slipped in
    status = client.get("/api/analysis/status").json()
    assert status["analyzer"]["blocked"] is True
    assert status["models"]["available"] is False       # and says why tags are missing

    monkeypatch.setenv("MASHUP_ANALYZER", "librosa")       # comparison mode still works
    stages.do_analyze(song)
    assert _full_row(models, song)["analyzer"] == "librosa"

    monkeypatch.delenv("MASHUP_ANALYZER")
    r = client.post("/api/settings", json={"analyzer": "shadow"})
    assert r.status_code == 400 and "Essentia" in r.json()["detail"]
    assert client.post("/api/settings", json={"analyzer": "bogus"}).status_code == 400


def test_the_analysis_cache_switch_can_be_saved_off(env):
    import config
    _models, _stages, client, _tmp = env
    assert client.post("/api/settings", json={"analysis_cache": False}).status_code == 200
    assert config.current_analysis_cache() is False
    client.post("/api/settings", json={"analysis_cache": True})
    assert config.current_analysis_cache() is True


@needs_essentia
def test_groups_measure_a_known_track(tmp_path):
    from analysis.essentia_groups import analyze_file_essentia
    out = analyze_file_essentia(_track(tmp_path / "t.wav", bpm=120.0, root_hz=261.63))
    r, t, lo, sp = out["rhythm"], out["tonal"], out["loudness"], out["spectral"]
    assert r["bpm"] == pytest.approx(120.0, rel=0.02)
    assert len(r["beat_times"]) > 80 and 0.0 <= r["bpm_confidence"] <= 1.0
    assert t["key"] == "C" and t["camelot"] in ("8B", "5A")
    assert 0.0 <= t["key_confidence"] <= 1.0 and int(np.argmax(t["hpcp"])) == 0
    assert 430 < t["tuning_hz"] < 450
    assert -40 < lo["lufs"] < 0 and lo["true_peak"] >= lo["sample_peak_db"] - 0.01
    assert lo["stereo_width"] == pytest.approx(0.0, abs=1e-6)     # dual mono
    assert sum(sp["band_energy"]) == pytest.approx(1.0, abs=1e-3)
    assert len(sp["mfcc"]) == 13 and len(sp["waveform_rms"]) == 360


@needs_essentia
def test_shadow_keeps_librosa_in_the_core_and_fills_the_extras(env, monkeypatch):
    models, stages, _client, tmp = env
    song = _song(models, tmp)
    monkeypatch.setenv("MASHUP_ANALYZER", "shadow")
    stages.do_analyze(song)
    row = _full_row(models, song)
    assert row["analyzer"] == "librosa"
    assert row["lufs"] is not None and row["tuning_hz"] is not None
    # The core is librosa's, byte for byte: same as a pure librosa run.
    monkeypatch.setenv("MASHUP_ANALYZER", "librosa")
    stages.do_analyze(song)
    pure = _full_row(models, song)
    assert pure["bpm"] == row["bpm"] and pure["mfcc"] == row["mfcc"]
    assert pure["lufs"] is None            # extras reflect the latest run


@needs_essentia
def test_essentia_owns_the_core_and_the_section_grid(env, monkeypatch):
    models, stages, _client, tmp = env
    song = _song(models, tmp)
    monkeypatch.setenv("MASHUP_ANALYZER", "essentia")
    stages.do_analyze(song)
    stages.do_structure(song)
    row = _full_row(models, song)
    assert row["analyzer"] == "essentia"
    assert row["bpm"] == pytest.approx(120.0, rel=0.02)
    beats = set(round(b, 1) for b in row["beat_times"])
    downbeats = [d for s in models.get_sections(song) for d in (s.get("downbeats") or [])]
    assert downbeats
    # Section downbeats come from the stored (Essentia) grid, within a frame.
    assert all(any(abs(d - b) <= 0.05 for b in row["beat_times"]) for d in downbeats)
    assert beats


@needs_essentia
def test_status_reports_coverage_and_agreement(env, monkeypatch):
    models, stages, client, tmp = env
    song = _song(models, tmp)
    monkeypatch.setenv("MASHUP_ANALYZER", "shadow")
    stages.do_analyze(song)
    body = client.get("/api/analysis/status").json()
    assert body["analyzer"]["configured"] == "shadow" and body["essentia"]["available"]
    assert body["coverage"]["essentia.rhythm"]["full"] == 1
    assert body["coverage"]["librosa.tempo"]["full"] == 1
    assert body["agreement"]["tracks_compared"]["bpm"] == 1
    assert sum(body["agreement"]["bpm"].values()) == 1


@needs_essentia
def test_a_second_shadow_run_is_served_from_the_cache(env, monkeypatch):
    models, stages, _client, tmp = env
    song = _song(models, tmp)
    monkeypatch.setenv("MASHUP_ANALYZER", "shadow")
    stages.do_analyze(song)
    conn = models.get_conn()
    conn.execute("DELETE FROM analysis_runs")
    conn.commit()
    conn.close()
    stages.do_analyze(song)
    conn = models.get_conn()
    grps = [r["grp"] for r in conn.execute("SELECT grp FROM analysis_runs")]
    conn.close()
    assert "analysis.cached" in grps
    assert "essentia.rhythm.cached" in grps and "essentia.rhythm" not in grps


@needs_essentia
def test_an_incomplete_essentia_result_fails_the_stem(env, monkeypatch):
    """An Essentia library never gets a librosa-filled row: the core columns
    would mix analysers (readme §7). The stem fails and Retry re-runs it."""
    models, stages, _client, tmp = env
    song = _song(models, tmp)
    monkeypatch.setenv("MASHUP_ANALYZER", "essentia")
    monkeypatch.setattr(stages, "_run_essentia", lambda *a, **k: ({}, False))
    with pytest.raises(stages.StageError):
        stages.do_analyze(song)
    assert _full_row(models, song) is None
    assert models.get_song(song)["status"] == "error_analysis"


def test_tags_are_an_extra_column_and_only_on_the_mix():
    from analysis.project import EXTRA_COLUMNS, extras_from_essentia
    assert "tags_json" in EXTRA_COLUMNS
    tags = {"genre": [{"label": "Electronic---House", "p": 0.4}], "genre_parent": "Electronic"}
    assert json.loads(extras_from_essentia({"effnet": {"tags": tags}}, "essentia")["tags_json"]) == tags
    assert extras_from_essentia({"rhythm": {}}, "essentia")["tags_json"] is None


@needs_essentia
def test_the_analysis_stage_tags_the_mix_but_the_quick_tier_does_not(env, monkeypatch):
    models, stages, _client, tmp = env
    import analysis.essentia_groups as eg
    import analysis.ml_models as ml
    monkeypatch.setattr(ml, "ensure_models", lambda download=True: True)   # models present
    calls = []
    monkeypatch.setattr(eg, "group_effnet",
                        lambda sig: calls.append(1) or {"tags": {"genre_parent": "Electronic"},
                                                         "embedding_mean": [0.0]})
    monkeypatch.setenv("MASHUP_ANALYZER", "essentia")
    song = _song(models, tmp)
    models.update_song_status(song, "downloaded")
    stages.do_quick(song)
    assert calls == [] and _full_row(models, song).get("tags_json") is None
    models.update_song_status(song, "stemmed")
    stages.do_analyze(song)
    assert calls == [1]
    assert json.loads(_full_row(models, song)["tags_json"])["genre_parent"] == "Electronic"


@needs_essentia
def test_missing_models_skip_the_tags_without_failing(env, monkeypatch):
    models, stages, _client, tmp = env
    import analysis.essentia_groups as eg
    monkeypatch.setattr(eg, "group_effnet", lambda sig: None)
    monkeypatch.setenv("MASHUP_ANALYZER", "essentia")
    song = _song(models, tmp)
    stages.do_analyze(song)
    row = _full_row(models, song)
    assert row["analyzer"] == "essentia" and row.get("tags_json") is None
    assert models.get_song(song)["status"] == "analysed"
