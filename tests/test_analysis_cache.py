"""Phase 1 of the analysis overhaul: decode each file once, compute each
transform once, and cache each feature group by the audio's content hash.

The invariant under all of it: the numbers written are the numbers the old
code wrote. Every sharing layer is checked against the same work done with the
sharing switched off."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _wav(path: Path, secs: float, bpm: float = 120.0, sr: int = 22050,
         seed_hz: float = 220.0) -> Path:
    import soundfile as sf
    n = int(secs * sr)
    y = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(np.arange(0, secs, 50.0 / bpm)):
        s = int(t * sr)
        e = min(n, s + int(0.08 * sr))
        tt = np.arange(e - s) / sr
        y[s:e] += (1.0 if i % 4 == 0 else 0.5) * np.sin(2 * np.pi * 80 * tt) * np.exp(-tt * 30)
    for k, t in enumerate(np.arange(0, secs, 16 * 60.0 / bpm)):
        s, e = int(t * sr), min(n, int((t + 16 * 60.0 / bpm) * sr))
        tt = np.arange(e - s) / sr
        y[s:e] += 0.2 * np.sin(2 * np.pi * (seed_hz if k % 2 else seed_hz * 1.5) * tt)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y / np.abs(y).max(), sr)
    return path


@pytest.fixture
def decode():
    from analysis import decode as mod
    mod.clear()
    yield mod
    mod.clear()


# ── Decode cache ──────────────────────────────────────────────────────────────

def _mp3(tmp_path: Path, secs: float = 4.0, sr: int = 44100) -> Path:
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    wav = _wav(tmp_path / "src.wav", secs, sr=sr)
    out = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "2",
                    "-b:a", "192k", str(out)], check=True)
    return out


def test_a_compressed_file_is_decoded_by_ffmpeg_to_what_librosa_load_gives(
        decode, tmp_path, monkeypatch):
    """libsndfile's MP3 decoder took 5-6 s per 4-minute track (FFmpeg: 0.5 s),
    measured in the container. The signal must stay what librosa.load returned,
    or every cached feature would silently change meaning."""
    import librosa
    mp3 = _mp3(tmp_path)
    ref, _sr = librosa.load(str(mp3), sr=22050, mono=True)

    calls = []
    real = decode.decode_ffmpeg
    monkeypatch.setattr(decode, "decode_ffmpeg",
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    y = decode.load_mono(mp3, sr=22050)
    assert calls, "an MP3 must go through FFmpeg"
    assert len(y) == len(ref)
    assert np.max(np.abs(y - ref)) < 1e-4


def test_lossless_files_keep_the_librosa_decode(decode, tmp_path, monkeypatch):
    monkeypatch.setattr(decode, "decode_ffmpeg",
                        lambda *a, **k: pytest.fail("WAV/FLAC need no FFmpeg"))
    decode.load_mono(_wav(tmp_path / "a.wav", 2.0), sr=22050)
    assert decode.prefers_ffmpeg(Path("x.mp3")) and decode.prefers_ffmpeg(Path("x.M4A"))
    assert not decode.prefers_ffmpeg(Path("x.flac")) and not decode.prefers_ffmpeg(Path("x.wav"))


def test_a_file_ffmpeg_cannot_read_falls_back_to_librosa(decode, tmp_path, monkeypatch):
    import subprocess
    mp3 = _mp3(tmp_path)

    def boom(*_a, **_k):
        raise subprocess.CalledProcessError(1, "ffmpeg")
    monkeypatch.setattr(decode, "decode_ffmpeg", boom)
    assert len(decode.load_mono(mp3, sr=22050)) > 0


def test_a_file_is_decoded_once_and_shared_read_only(decode, tmp_path):
    p = _wav(tmp_path / "a.wav", 3.0)
    y1 = decode.load_mono(p, sr=22050)
    y2 = decode.load_mono(p, sr=22050)
    assert y1 is y2
    assert decode.stats()["misses"] == 1 and decode.stats()["hits"] == 1
    with pytest.raises(ValueError):
        y1[0] = 1.0            # shared: nobody may write into it


def test_a_duration_is_a_slice_of_the_full_decode(decode, tmp_path):
    p = _wav(tmp_path / "a.wav", 3.0)
    full = decode.load_mono(p, sr=22050)
    head = decode.load_mono(p, sr=22050, duration=1.0)
    assert len(head) == 22050 and np.shares_memory(head, full)
    # Longer than the file: the whole file, the same object.
    assert decode.load_mono(p, sr=22050, duration=60.0) is full
    assert decode.stats()["misses"] == 1


def test_the_decode_is_librosas(decode, tmp_path):
    import librosa
    p = _wav(tmp_path / "a.wav", 2.0, sr=44100)
    y_ref, _ = librosa.load(str(p), sr=22050, mono=True)
    assert np.array_equal(decode.load_mono(p, sr=22050), y_ref)


def test_a_rewritten_file_is_decoded_again(decode, tmp_path):
    p = _wav(tmp_path / "a.wav", 2.0)
    first = decode.load_mono(p, sr=22050)
    _wav(p, 3.0)
    assert len(decode.load_mono(p, sr=22050)) != len(first)


def test_memo_only_for_arrays_the_cache_handed_out(decode, tmp_path):
    calls = []
    y = decode.load_mono(_wav(tmp_path / "a.wav", 2.0), sr=22050)
    assert decode.memo(y, ("k",), lambda: calls.append(1) or "v") == "v"
    assert decode.memo(y, ("k",), lambda: calls.append(1) or "w") == "v"
    assert len(calls) == 1
    loose = np.zeros(10, dtype=np.float32)
    decode.memo(loose, ("k",), lambda: calls.append(1))
    decode.memo(loose, ("k",), lambda: calls.append(1))
    assert len(calls) == 3


def test_capacity_evicts_least_recently_used(decode, tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DECODE_CACHE_SIZE", 2)
    a, b, c = (_wav(tmp_path / f"{n}.wav", 1.0) for n in "abc")
    decode.load_mono(a)
    decode.load_mono(b)
    decode.load_mono(a)            # a is now the most recent
    decode.load_mono(c)            # evicts b
    assert decode.stats()["entries"] == 2
    misses = decode.stats()["misses"]
    decode.load_mono(a)
    assert decode.stats()["misses"] == misses
    decode.load_mono(b)
    assert decode.stats()["misses"] == misses + 1


# ── Sharing changes nothing ───────────────────────────────────────────────────

def _run_everything(tmp_path):
    """analysis on mix + stems, band occupancy, residual, stem quality and
    structure — the whole per-track pass, in pipeline order."""
    from analysis.analyze import analyze_file
    from analysis.quality import band_energy, residual_vocal_ratio, stem_quality
    from analysis.structure import detect_sections
    mix = tmp_path / "mix.wav"
    voc = tmp_path / "voc.wav"
    bed = tmp_path / "bed.wav"
    out = {}
    for name, p in (("full", mix), ("vocals", voc), ("instrumental", bed)):
        out[name] = analyze_file(p)
        out[name + ".bands"] = band_energy(p)
    out["residual"] = residual_vocal_ratio(voc, bed)
    out["quality"] = stem_quality(voc, mix, other_path=bed)
    out["sections"] = detect_sections(mix, voc, inst_path=bed)
    return out


def test_sharing_decodes_and_transforms_changes_no_number(decode, tmp_path, monkeypatch):
    import config
    _wav(tmp_path / "mix.wav", 50.0)
    _wav(tmp_path / "voc.wav", 50.0, seed_hz=330.0)
    _wav(tmp_path / "bed.wav", 50.0, seed_hz=110.0)

    monkeypatch.setattr(config, "DECODE_CACHE_SIZE", 0)     # the old behaviour
    unshared = _run_everything(tmp_path)
    old_decodes = decode.stats()["misses"]

    monkeypatch.setattr(config, "DECODE_CACHE_SIZE", 6)
    decode.clear()
    shared = _run_everything(tmp_path)

    assert shared == unshared                 # exact, not approximate
    assert shared["sections"]                 # and there was something to compare
    # Three files, three decodes — down from one per consumer.
    assert decode.stats()["misses"] == 3
    assert old_decodes > 3 * 3
    assert decode.stats()["memo_hits"] > 0


# ── Registry + persistent cache ───────────────────────────────────────────────

def test_every_analysis_step_is_a_registered_group():
    from analysis.analyze import STEPS
    from analysis.registry import GROUPS, STEP_GROUPS
    assert set(STEP_GROUPS) == set(STEPS)
    assert {"librosa.bands", "librosa.residual", "librosa.structure"} <= set(GROUPS)


def test_params_hash_follows_the_config_it_reads(monkeypatch):
    import config
    from analysis.registry import group
    g = group("librosa.timbre")
    before = g.params_hash()
    monkeypatch.setattr(config, "N_MFCC", 20)
    assert g.params_hash() != before
    assert group("librosa.key").params_hash() == group("librosa.key").params_hash()


def test_content_hash_follows_bytes_not_path(tmp_path):
    from analysis.cache import combo_hash, content_hash
    a = _wav(tmp_path / "a.wav", 1.0)
    b = tmp_path / "copy.wav"
    b.write_bytes(a.read_bytes())
    assert content_hash(a) == content_hash(b)
    _wav(b, 1.5)
    assert content_hash(a) != content_hash(b)
    assert content_hash(tmp_path / "missing.wav") is None
    assert combo_hash({"full": "x", "vocals": None}) != combo_hash({"full": "x", "vocals": "y"})
    assert combo_hash({"full": None}, required=("full",)) is None


@pytest.fixture
def env(tmp_path, monkeypatch, decode):
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    import database.models as models
    import api.jobs
    import api.workers.stages
    for mod in (config, models, api.jobs, api.workers.stages):
        importlib.reload(mod)
    models.init_db()
    import api.workers.hook_worker as hook_worker
    monkeypatch.setattr(hook_worker, "warm_hooks", lambda *a, **k: {})

    mix = _wav(tmp_path / "mix.wav", 50.0)
    voc = _wav(tmp_path / "voc.wav", 50.0, seed_hz=330.0)
    bed = _wav(tmp_path / "bed.wav", 50.0, seed_hz=110.0)
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "stemmed", raw_path=str(mix))
    models.upsert_stem(song, "full", str(mix))
    models.upsert_stem(song, "vocals", str(voc))
    models.upsert_stem(song, "instrumental", str(bed))
    monkeypatch.setattr(api.workers.stages, "_measure_stem_quality", lambda *a, **k: None)
    return models, api.workers.stages, song


def _feature_rows(models, song):
    conn = models.get_conn()
    try:
        return {r["stem_type"]: dict(r) for r in conn.execute(
            "SELECT * FROM features WHERE song_id=? ORDER BY stem_type", (song,))}
    finally:
        conn.close()


def _sections(models, song):
    return [{k: v for k, v in s.items() if k != "id"} for s in models.get_sections(song)]


def _grps(models):
    conn = models.get_conn()
    try:
        return [r["grp"] for r in conn.execute("SELECT grp FROM analysis_runs ORDER BY id")]
    finally:
        conn.close()


def test_reanalysis_reprojects_from_the_cache_without_decoding(env, decode):
    models, stages, song = env
    stages.do_analyze(song)
    stages.do_structure(song)
    rows, secs = _feature_rows(models, song), _sections(models, song)
    assert rows["vocals"]["band_energy_json"] and rows["instrumental"]["residual_vocal_ratio"] is not None
    assert secs

    decode.clear()
    conn = models.get_conn()
    conn.execute("DELETE FROM analysis_runs")
    conn.commit()
    conn.close()

    stages.do_analyze(song)
    stages.do_structure(song)
    assert decode.stats()["misses"] == 0            # nothing was decoded
    assert _feature_rows(models, song) == rows       # the same rows
    assert _sections(models, song) == secs
    grps = _grps(models)
    assert "analysis.cached" in grps and "structure.cached" in grps
    assert "analysis.tempo.cached" in grps and "analysis.tempo" not in grps


def test_a_version_bump_recomputes_that_group_only(env, decode, monkeypatch):
    import dataclasses
    import analysis.registry as registry
    import analysis.cache as cache_mod
    models, stages, song = env
    stages.do_analyze(song)

    bumped = dataclasses.replace(registry.GROUPS["librosa.key"], version=2)
    monkeypatch.setitem(registry.GROUPS, "librosa.key", bumped)
    monkeypatch.setitem(registry.STEP_GROUPS, "key", bumped)
    monkeypatch.setitem(cache_mod.STEP_GROUPS, "key", bumped)
    conn = models.get_conn()
    conn.execute("DELETE FROM analysis_runs")
    conn.commit()
    conn.close()

    stages.do_analyze(song)
    grps = _grps(models)
    assert grps.count("analysis.key") == 3           # once per stem
    assert "analysis.tempo" not in grps and grps.count("analysis.tempo.cached") == 3
    assert "analysis.cached" not in grps             # a real analysis ran


def test_the_cache_switch_forces_a_recompute(env, decode, monkeypatch):
    models, stages, song = env
    stages.do_analyze(song)
    monkeypatch.setenv("MASHUP_ANALYSIS_CACHE", "0")
    decode.clear()
    stages.do_analyze(song)
    assert decode.stats()["misses"] == 3


def test_a_re_separated_stem_recomputes_its_groups_and_structure(env, decode, tmp_path):
    models, stages, song = env
    stages.do_analyze(song)
    stages.do_structure(song)
    _wav(tmp_path / "voc.wav", 50.0, seed_hz=392.0)     # new bytes
    conn = models.get_conn()
    conn.execute("DELETE FROM analysis_runs")
    conn.commit()
    conn.close()

    stages.do_analyze(song)
    stages.do_structure(song)
    grps = _grps(models)
    assert "analysis.tempo.cached" in grps          # the mix and bed
    assert any(g == "analysis.tempo" for g in grps)  # the new vocal
    assert "structure.cached" not in grps
