"""Phase 2 of the analysis overhaul: the Essentia analyser's groups, how they
project into the features table, and analysis refusing to run without it.

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


needs_essentia = pytest.mark.real_essentia


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


def test_essentia_projects_onto_exactly_the_core_columns():
    """core_from_essentia must fill every core column the matcher reads, and
    nothing else: upsert_features writes whatever it is given."""
    from analysis.project import core_from_essentia
    assert set(core_from_essentia(_payloads())) == {
        "bpm", "bpm_confidence", "beat_times", "beat_phase", "key", "mode",
        "camelot", "key_confidence", "loudness_rms", "energy", "mfcc",
        "spectral_centroid", "spectral_rolloff", "zero_crossing_rate",
        "waveform_rms", "band_energy"}


def test_extras_are_unmeasured_when_essentia_did_not_run():
    from analysis.project import EXTRA_COLUMNS, extras_from_essentia
    ex = extras_from_essentia(_payloads(), "librosa")
    assert set(ex) == set(EXTRA_COLUMNS)
    assert ex["lufs"] == -9.5 and ex["analyzer"] == "librosa"
    none = extras_from_essentia({}, "librosa")
    assert none["analyzer"] == "librosa"
    assert all(v is None for k, v in none.items() if k != "analyzer")


# ── The analysis stage ────────────────────────────────────────────────────────────────

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


def test_config_reads_the_essentia_settings_live(env, monkeypatch):
    import config
    assert config.current_essentia_key_profile() == "edma"
    assert config.current_essentia_rhythm_method() == "degara"


def test_without_essentia_analysis_is_refused_not_degraded(env, monkeypatch):
    models, stages, client, tmp = env
    import analysis.essentia_groups as eg
    monkeypatch.setattr(eg, "available", lambda: False)
    song = _song(models, tmp)
    with pytest.raises(stages.StageError, match="Docker or WSL2"):
        stages.do_analyze(song)
    row = models.get_song(song)
    assert row["status"] == "error_analysis" and "Docker or WSL2" in row["last_error"]
    assert _full_row(models, song) is None                 # no librosa row slipped in
    status = client.get("/api/analysis/status").json()
    assert status["analyzer"]["blocked"] is True
    assert status["models"]["available"] is False       # and says why tags are missing


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
def test_essentia_owns_the_core_and_the_section_grid(env, monkeypatch):
    models, stages, _client, tmp = env
    song = _song(models, tmp)
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


def test_status_reports_coverage(env, monkeypatch):
    models, stages, client, tmp = env
    song = _song(models, tmp)
    stages.do_analyze(song)
    body = client.get("/api/analysis/status").json()
    assert body["analyzer"]["blocked"] is False
    assert body["coverage"]["essentia.rhythm"]["full"] == 1
    assert "agreement" not in body
    assert "allin1" in body["segmentation"]


def test_a_second_run_is_served_from_the_cache(env, monkeypatch):
    models, stages, _client, tmp = env
    monkeypatch.setenv("MASHUP_KEEP_ANALYSIS_FRAMES", "0")   # the stand-in writes none
    song = _song(models, tmp)
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


def test_an_incomplete_essentia_result_fails_the_stem(env, monkeypatch):
    """A library never gets a half-filled or librosa-filled row: the core
    columns are ranked against the library (readme §7). The stem fails and
    Retry re-runs it."""
    models, stages, _client, tmp = env
    song = _song(models, tmp)
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
    song = _song(models, tmp)
    stages.do_analyze(song)
    row = _full_row(models, song)
    assert row["analyzer"] == "essentia" and row.get("tags_json") is None
    assert models.get_song(song)["status"] == "analysed"


# ── Per-frame arrays (analysis/timeseries.py): pure numpy, run everywhere ─────

def test_frame_chroma_folds_like_the_mean_and_roll_to_c():
    from analysis.essentia_groups import hpcp_frames_to_chroma
    frames = np.zeros((3, 36))
    frames[:, 9] = 1.0          # A-referenced, 3 bins/semitone: bin 9 = semitone 3 = C
    frames[:, 8] = 0.5          # its lower neighbour folds into the same pitch
    frames[1, 9] = 2.0          # a louder frame must come out the same shape
    chroma = hpcp_frames_to_chroma(frames)
    assert chroma.shape == (12, 3) and chroma.dtype == np.float32
    assert (np.argmax(chroma, axis=0) == 0).all()
    np.testing.assert_allclose(np.linalg.norm(chroma, axis=0), 1.0, atol=1e-6)
    # A frame holding only the A bin lands on index 9 (A), as hpcp_to_chroma's does.
    a = np.zeros((2, 36))
    a[:, 0] = 1.0
    assert (np.argmax(hpcp_frames_to_chroma(a), axis=0) == 9).all()
    a12 = np.zeros((2, 12))
    a12[:, 0] = 1.0             # a 12-bin HPCP has no neighbours to fold
    assert (np.argmax(hpcp_frames_to_chroma(a12), axis=0) == 9).all()


def test_frame_chroma_agrees_with_the_track_mean_it_replaces():
    """Folding before the mean (this) and after it (hpcp_to_chroma) are the
    same conversion: identical frames give that function's answer per frame."""
    from analysis.essentia_groups import hpcp_frames_to_chroma, hpcp_to_chroma
    rng = np.random.default_rng(3)
    row = rng.random(36)
    frames = np.tile(row, (5, 1))
    per_frame = hpcp_frames_to_chroma(frames)
    for j in range(5):
        np.testing.assert_allclose(per_frame[:, j], hpcp_to_chroma(frames), atol=1e-6)


def test_frame_chroma_keeps_a_silent_frame_as_zeros_not_nan():
    from analysis.essentia_groups import hpcp_frames_to_chroma
    frames = np.zeros((4, 36))
    frames[0, 3] = 1.0
    frames[2, 20] = 1.0
    chroma = hpcp_frames_to_chroma(frames)
    assert np.isfinite(chroma).all()
    assert chroma[:, 1].sum() == 0 and chroma[:, 3].sum() == 0
    assert np.linalg.norm(chroma[:, 0]) == pytest.approx(1.0)


def test_frame_chroma_declines_what_it_cannot_fold():
    from analysis.essentia_groups import hpcp_frames_to_chroma
    assert hpcp_frames_to_chroma(np.zeros((0, 36))) is None
    assert hpcp_frames_to_chroma(np.zeros(36)) is None          # not (frames, bins)
    assert hpcp_frames_to_chroma(np.zeros((4, 35))) is None     # not a multiple of 12
    assert hpcp_frames_to_chroma(None) is None


def test_the_two_essentia_hops_are_the_shared_frame_grid():
    """The invariant group_spectral asserts at run time, pinned where it can be
    seen without Essentia: TonalExtractor's 2048/44100 and the spectral pass's
    2 x 512/22050 are one interval, so no matrix is ever resampled."""
    from analysis import timeseries as ts
    from analysis.essentia_groups import HOP, SR_FRAMES
    assert 2.0 * HOP / SR_FRAMES == pytest.approx(ts.STEP_SECS, abs=1e-12)


def _fake_out(n_chroma=100, n_spec=100, chroma_step=None, spec_step=None):
    from analysis import timeseries as ts
    rng = np.random.default_rng(5)
    return {
        "chroma": rng.random((12, n_chroma)).astype(np.float32),
        "chroma_step": chroma_step or ts.STEP_SECS,
        "mfcc": rng.random((13, n_spec)).astype(np.float32),
        "bands": rng.random((8, n_spec)).astype(np.float32),
        "curves": rng.random((5, n_spec)).astype(np.float32),
        "spectral_step": spec_step or ts.STEP_SECS,
    }


def test_frames_from_has_nothing_to_say_without_frames():
    from analysis.essentia_groups import frames_from
    assert frames_from({}) is None
    assert frames_from({"chroma_step": 0.05}) is None


def test_frames_from_trims_every_matrix_to_the_shortest():
    from analysis import timeseries as ts
    from analysis.essentia_groups import frames_from
    out = _fake_out(n_chroma=103, n_spec=100)
    frames = frames_from(out)
    assert isinstance(frames, ts.StemFrames)
    assert frames.step == ts.STEP_SECS
    for name in ("chroma", "mfcc", "bands", "curves"):
        assert getattr(frames, name).shape == (ts.ROWS[name], 100)
    # Trimmed, not resampled: the leading frames are the originals.
    np.testing.assert_array_equal(frames.chroma, out["chroma"][:, :100])
    # ...and the other way round.
    frames = frames_from(_fake_out(n_chroma=97, n_spec=100))
    assert frames.n_frames() == 97 and frames.mfcc.shape == (13, 97)
    # Whatever it hands out, Task 1's one-grid check accepts.
    ts._payload(frames)


def test_frames_from_keeps_one_group_alone_and_leaves_the_rest_absent():
    from analysis.essentia_groups import frames_from
    out = _fake_out()
    only_chroma = frames_from({k: out[k] for k in ("chroma", "chroma_step")})
    assert only_chroma.chroma is not None
    assert only_chroma.mfcc is None and only_chroma.bands is None
    assert only_chroma.curves is None                  # absent, not zeros
    only_spec = frames_from({k: v for k, v in out.items() if not k.startswith("chroma")})
    assert only_spec.chroma is None and only_spec.mfcc.shape[0] == 13


def test_frames_from_resamples_a_matrix_whose_hop_is_off_the_grid():
    from analysis import timeseries as ts
    from analysis.essentia_groups import frames_from
    # A chroma at twice the grid's hop: half as many frames over the same audio.
    out = _fake_out(n_chroma=50, n_spec=100, chroma_step=2 * ts.STEP_SECS)
    frames = frames_from(out)
    assert frames.n_frames() == 100 and frames.chroma.shape == (12, 100)
    np.testing.assert_array_equal(frames.chroma[:, 0], out["chroma"][:, 0])
    np.testing.assert_array_equal(frames.chroma[:, 2], out["chroma"][:, 1])
    # Within 1 % of the grid it is left alone.
    near = frames_from(_fake_out(chroma_step=ts.STEP_SECS * 1.005))
    assert near.chroma.shape == (12, 100)


def test_frames_from_never_hands_out_a_non_finite_value():
    from analysis import timeseries as ts
    from analysis.essentia_groups import frames_from
    out = _fake_out()
    out["mfcc"][3, 10] = np.nan
    out["bands"][0, 0] = np.inf
    frames = frames_from(out)
    assert frames.mfcc is None and frames.bands is None   # unmeasured, not patched
    assert frames.chroma.shape == (12, 100) and frames.curves.shape == (5, 100)
    ts._payload(frames)
    assert frames_from({"mfcc": out["mfcc"], "spectral_step": ts.STEP_SECS}) is None


# ── Per-frame arrays out of the real groups (Docker / WSL2 only) ──────────────
# Audio comes from this file's own `_track` (a C-major triad over a kick) and
# `load_signals`, the same way test_groups_measure_a_known_track gets its signals.

@needs_essentia
def test_group_tonal_emits_per_frame_chroma(tmp_path):
    """The frames TonalExtractor already produced, kept instead of dropped."""
    from analysis import timeseries as ts
    from analysis.essentia_groups import group_tonal, load_signals

    sig = load_signals(_track(tmp_path / "t.wav", secs=20.0, root_hz=261.63))
    out = {}
    payload = group_tonal(sig, frames_out=out)

    assert len(payload["hpcp"]) == 12
    assert payload == group_tonal(sig)                  # the side channel moved nothing
    # Exactly, not within 5 %: the step is the extractor's own hop, and the
    # count is its ceil(n / hop) + 1 less the centring frame group_tonal drops.
    from analysis.essentia_groups import CHROMA_LEAD_FRAMES, TONAL_HOP
    expected = -(-len(sig.mono44) // TONAL_HOP) + 1 - CHROMA_LEAD_FRAMES
    assert out["chroma"].shape == (12, expected)
    assert out["chroma_step"] == pytest.approx(ts.STEP_SECS, abs=1e-9)
    # Each frame is L2-normalised, so a section's mean compares shape not level.
    norms = np.linalg.norm(out["chroma"], axis=0)
    assert ((norms > 0.99) & (norms < 1.01)).mean() > 0.9
    # The triad is rooted on C: C, E and G dominate both the frames' mean and
    # the stored mean. Which of the three peaks is a hairline — 0.5475 against
    # 0.5490 on this tone — so the assertion is the set, never the argmax.
    assert set(np.argsort(out["chroma"].mean(axis=1))[-3:].tolist()) == {0, 4, 7}
    assert set(np.argsort(np.asarray(payload["hpcp"]))[-3:].tolist()) == {0, 4, 7}


@needs_essentia
def test_group_spectral_emits_mfcc_bands_and_curves(tmp_path):
    from analysis import timeseries as ts
    from analysis.essentia_groups import group_spectral, load_signals

    sig = load_signals(_track(tmp_path / "t.wav", secs=20.0))
    out = {}
    payload = group_spectral(sig, frames_out=out)

    assert len(payload["mfcc"]) == 13
    assert json.dumps(payload, sort_keys=True) == json.dumps(group_spectral(sig), sort_keys=True)
    assert out["mfcc"].shape[0] == 13
    assert out["bands"].shape[0] == 8
    assert out["curves"].shape[0] == len(ts.CURVE_ROWS)
    assert out["mfcc"].shape[1] == out["bands"].shape[1] == out["curves"].shape[1] > 1
    assert out["spectral_step"] == pytest.approx(ts.STEP_SECS, rel=0.05)
    for name in ("mfcc", "bands", "curves"):
        assert np.isfinite(out[name]).all()
    # Band power is absolute (an occupancy is a ratio of sums), so not normalised,
    # and its occupancy over the track is the stored band_energy.
    occupancy = out["bands"].sum(axis=1) / out["bands"].sum()
    assert out["bands"].sum() > 0
    assert occupancy == pytest.approx(payload["band_energy"], abs=0.05)   # every 2nd frame


@needs_essentia
def test_frames_from_puts_both_groups_on_one_grid(tmp_path):
    from analysis import timeseries as ts
    from analysis.essentia_groups import (frames_from, group_spectral,
                                          group_tonal, load_signals)

    secs = 20.0
    sig = load_signals(_track(tmp_path / "t.wav", secs=secs))
    out = {}
    group_tonal(sig, frames_out=out)
    group_spectral(sig, frames_out=out)
    # The two passes see the same audio on one clock, to within a frame or two.
    assert abs(out["chroma"].shape[1] - out["mfcc"].shape[1]) <= 2
    frames = frames_from(out)

    assert isinstance(frames, ts.StemFrames)
    n = frames.n_frames()
    assert n > 1 and n == pytest.approx(secs / ts.STEP_SECS, abs=3)
    for name in ("chroma", "mfcc", "bands", "curves"):
        assert getattr(frames, name).shape == (ts.ROWS[name], n)
    ts._payload(frames)                                 # timeseries.write's own checks


@needs_essentia
def test_chroma_and_the_spectral_curves_share_a_time_origin():
    """The bug this guards: TonalExtractor centres its first frame, so the raw
    chroma ran one frame (46 ms) behind the spectral pass, which cuts from
    sample 0; group_tonal drops that frame. One onset cannot show it — the two
    read different window lengths (4096 against 2048) and jitter a frame either
    way — so the measure is the MEAN offset over onsets spread across one
    frame: a systematic lag reads about +1, a shared origin about 0."""
    from analysis import timeseries as ts
    from analysis.essentia_groups import Signals, group_spectral, group_tonal

    sr, offsets = 44100, []
    for phase_ms in (0, 10, 20, 30, 46):
        n, start = int(5.0 * sr), int((2.0 + phase_ms / 1000.0) * sr)
        y = np.zeros(n, dtype=np.float32)
        t = np.arange(n - start) / sr
        y[start:] = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sig = Signals(np.stack([y, y], axis=1))
        out = {}
        group_tonal(sig, frames_out=out)
        group_spectral(sig, frames_out=out)
        chroma = out["chroma"].max(axis=0)
        rms = out["curves"][list(ts.CURVE_ROWS).index("rms")]
        offsets.append(int(np.argmax(chroma > 0.3 * chroma.max()))
                       - int(np.argmax(rms > 0.3 * rms.max())))

    assert max(abs(o) for o in offsets) <= 1, offsets           # jitter, not drift
    assert -0.6 <= sum(offsets) / len(offsets) <= 0.5, offsets  # no systematic lag


class _MemCache:
    """The StepCache protocol (get / put), in memory."""
    def __init__(self):
        self.store = {}

    def get(self, step):
        return self.store.get(step)

    def put(self, step, payload, ms):
        self.store[step] = payload


@needs_essentia
def test_force_steps_rebuilds_the_frames_of_a_cached_stem(tmp_path):
    """A stem whose payloads are cached but whose sidecar is missing: the
    caller forces tonal/spectral, which re-run and fill frames_out, while the
    other groups are still served from the cache."""
    from analysis.essentia_groups import analyze_file_essentia

    wav = _track(tmp_path / "t.wav", secs=15.0)
    cache = _MemCache()
    first = {}
    analyze_file_essentia(wav, cache=cache, frames_out=first)
    assert {"chroma", "mfcc", "bands", "curves"} <= set(first)    # a cold run fills both

    warm, timings = {}, {}
    analyze_file_essentia(wav, cache=cache, timings=timings, frames_out=warm)
    assert warm == {} and set(timings["cached_steps"]) >= {"tonal", "spectral"}

    forced, timings = {}, {}
    out = analyze_file_essentia(wav, cache=cache, timings=timings, frames_out=forced,
                                force_steps=("tonal", "spectral"))
    assert {"chroma", "mfcc", "bands", "curves"} <= set(forced)
    assert "tonal" not in timings["cached_steps"] and "rhythm" in timings["cached_steps"]
    assert out["tonal"]["hpcp"] == cache.store["tonal"]["hpcp"]  # same payload as stored


# ── The same plumbing against a stand-in Essentia (runs everywhere) ───────────
# NOT a test of Essentia's numbers: the stand-in has numpy spectra and a random
# HPCP. What it does exercise, on Windows too, is everything this change added
# around Essentia's calls: the shapes, the decimation, the silent-frame
# handling, the shared grid, and that the stored payloads do not move.

def _stand_in_essentia(monkeypatch):
    import types

    es = types.ModuleType("essentia.standard")

    class Windowing:
        def __init__(self, type="hann"):
            pass

        def __call__(self, frame):
            return np.asarray(frame) * np.hanning(len(frame))

    class Spectrum:
        def __init__(self, size):
            pass

        def __call__(self, frame):
            return np.abs(np.fft.rfft(frame)).astype(np.float32)

    class MFCC:
        def __init__(self, inputSize, sampleRate, numberCoefficients):
            self.n = numberCoefficients

        def __call__(self, spec):
            return np.zeros(40), np.log(np.asarray(spec[: self.n]) + 1e-3)

    class SpectralContrast:
        def __init__(self, **kw):
            pass

        def __call__(self, spec):
            return np.ones(6), np.ones(6)

    class SpectralComplexity:
        def __init__(self, **kw):
            pass

        def __call__(self, spec):
            return 3.0

    class CentralMoments:
        def __init__(self, **kw):
            pass

        def __call__(self, spec):
            return np.ones(5)

    class DistributionShape:
        def __call__(self, m):
            return 1.0, 0.1, 3.0

    class SpectralPeaks:
        def __init__(self, **kw):
            pass

        def __call__(self, spec):
            return np.array([100.0, 200.0]), np.array([0.5, 0.4])

    class Dissonance:
        def __call__(self, f, m):
            return 0.2

    class FrameGenerator:
        def __init__(self, x, frameSize, hopSize, startFromZero=False):
            self.x, self.size, self.hop = np.asarray(x), frameSize, hopSize

        def __iter__(self):
            for s in range(0, len(self.x) - self.size + 1, self.hop):
                yield self.x[s:s + self.size]

    class KeyExtractor:
        def __init__(self, profileType, sampleRate):
            pass

        def __call__(self, x):
            return "C", "major", 0.8

    class TuningFrequencyExtractor:
        def __call__(self, x):
            return [440.0]

    class TonalExtractor:
        _names = ["chords_changes_rate", "chords_histogram", "chords_key",
                  "chords_number_rate", "chords_scale", "hpcp"]

        def __call__(self, x):
            n = len(x) // 2048 + 1                  # centred 4096/2048 frames
            hpcp = np.random.default_rng(0).random((n, 36)).astype(np.float32)
            hpcp[7] = 0.0                           # one silent frame
            return 0.1, np.ones(24), "C", 0.2, "major", hpcp

        def outputNames(self):
            return list(self._names)

    for cls in (Windowing, Spectrum, MFCC, SpectralContrast, SpectralComplexity,
                CentralMoments, DistributionShape, SpectralPeaks, Dissonance,
                FrameGenerator, KeyExtractor, TuningFrequencyExtractor, TonalExtractor):
        setattr(es, cls.__name__, cls)
    pkg = types.ModuleType("essentia")
    pkg.standard = es
    monkeypatch.setitem(sys.modules, "essentia", pkg)
    monkeypatch.setitem(sys.modules, "essentia.standard", es)


def _stand_in_signals(secs=12.0):
    from analysis.essentia_groups import Signals
    sr = 44100
    t = np.arange(int(secs * sr)) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y[: sr] = 0.0                                   # a silent first second
    return Signals(np.stack([y, y], axis=1))


@pytest.mark.real_analyzer_code
def test_groups_hand_out_frames_on_one_grid_and_keep_their_payloads(monkeypatch):
    from analysis import timeseries as ts
    from analysis.essentia_groups import frames_from, group_spectral, group_tonal
    _stand_in_essentia(monkeypatch)
    sig = _stand_in_signals()

    out = {}
    tonal = group_tonal(sig, frames_out=out)
    spectral = group_spectral(sig, frames_out=out)
    # The side channel is a side channel: payloads equal those without it.
    assert json.dumps(tonal, sort_keys=True) == json.dumps(group_tonal(sig), sort_keys=True)
    assert json.dumps(spectral, sort_keys=True) == json.dumps(group_spectral(sig), sort_keys=True)

    # The chroma step is TonalExtractor's known hop, never seconds / frames:
    # its count is ceil(n / 2048) + 1, so a derived step understates the hop
    # and drifts with the duration. The extra frame is the centring pad and is
    # dropped, so chroma[:, k] and mfcc[:, k] describe the same 46 ms.
    raw_hpcp = len(sig.mono44) // 2048 + 1              # the stand-in's own count
    assert out["chroma"].shape == (12, raw_hpcp - 1)
    assert out["chroma_step"] == pytest.approx(ts.STEP_SECS, abs=1e-9)
    assert out["chroma_step"] != pytest.approx(sig.secs / out["chroma"].shape[1],
                                               abs=1e-6)
    assert out["spectral_step"] == pytest.approx(ts.STEP_SECS, abs=1e-9)
    assert out["mfcc"].shape[0] == 13 and out["bands"].shape[0] == 8
    assert out["curves"].shape[0] == len(ts.CURVE_ROWS)
    n = out["mfcc"].shape[1]
    assert out["bands"].shape[1] == out["curves"].shape[1] == n
    assert n == pytest.approx(sig.secs / ts.STEP_SECS, abs=3)
    for k in ("chroma", "mfcc", "bands", "curves"):
        assert np.isfinite(out[k]).all() and out[k].dtype == np.float32

    # The silent first second: no centroid/rolloff/rms, but a number, not NaN.
    cen, rol, _flux, _zcr, rms = (out["curves"][i] for i in range(5))
    assert (cen[:10] == 0).all() and (rol[:10] == 0).all() and (rms[:10] == 0).all()
    assert cen[-10:].min() > 0                      # the 220 Hz tone, once it sounds
    assert (out["bands"][:, :10] == 0).all() and out["bands"][:, -10:].sum() > 0
    # A silent HPCP frame is zeros in the chroma: frame 7 of the stand-in's,
    # frame 6 once the centring frame is dropped.
    assert out["chroma"][:, 6].sum() == 0

    frames = frames_from(out)
    assert frames.n_frames() == min(out["chroma"].shape[1], n)
    ts._payload(frames)


@pytest.mark.real_analyzer_code
def test_force_steps_runs_a_cached_step_and_keeps_the_stored_payload_on_failure(
        monkeypatch, tmp_path):
    from analysis import essentia_groups as eg
    _stand_in_essentia(monkeypatch)
    monkeypatch.setattr(eg, "available", lambda: True)
    monkeypatch.setattr(eg, "load_signals", lambda p: _stand_in_signals(8.0))
    monkeypatch.setattr(eg, "runnable_steps", lambda steps: steps)
    cache = _MemCache()
    cache.store = {"rhythm": {"bpm": 99.0}, "loudness": {"lufs": -1.0},
                   "tonal": {"key": "stored"}, "spectral": {"mfcc": [0.0]}}

    # Nothing forced: everything is served from the cache and no frames exist.
    frames, timings = {}, {}
    out = eg.analyze_file_essentia(tmp_path / "x.wav", cache=cache, frames_out=frames,
                                   timings=timings)
    assert frames == {} and out["tonal"] == {"key": "stored"}

    # Forced: tonal and spectral re-run and fill the frames; rhythm and loudness
    # stay cached (they would raise here: the stand-in has no rhythm extractor).
    out = eg.analyze_file_essentia(tmp_path / "x.wav", cache=cache, frames_out=frames,
                                   timings=timings, force_steps=("tonal", "spectral"))
    assert {"chroma", "mfcc", "bands", "curves"} <= set(frames)
    assert out["rhythm"] == {"bpm": 99.0} and out["loudness"] == {"lufs": -1.0}
    assert out["tonal"]["key"] == "C" and len(out["spectral"]["mfcc"]) == 13
    assert set(timings["cached_steps"]) == {"rhythm", "loudness"}
    assert timings["failed_steps"] == []

    # A forced step that fails leaves the stored payload in place, and says so.
    def boom(sig, primary="edma", frames_out=None):
        raise RuntimeError("tonal broke")
    monkeypatch.setattr(eg, "group_tonal", boom)
    out = eg.analyze_file_essentia(tmp_path / "x.wav", cache=cache, frames_out={},
                                   timings=timings, force_steps=("tonal",))
    assert out["tonal"]["key"] == "C"               # the payload the cache now holds
    assert timings["failed_steps"] == ["tonal"]
