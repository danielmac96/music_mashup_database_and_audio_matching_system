"""T1.6 — pre-rendered hook clips.

decodeStem fetches and decodes a whole ~40 MB track into an AudioBuffer, which
is far too slow and too memory-hungry to do while arrowing down a ranked list.
The hook worker cuts the 16 bars chosen in T1.5 into a small standalone wav so
the browser fetches ~3 MB instead.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")

SR = 22050


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    config.ensure_dirs()
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.workers import hook_worker
    importlib.reload(hook_worker)
    return config, models, hook_worker


def _write_stem(path: Path, secs=60.0, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path


def _seed(tmp_path, models, hook=(40.0, 72.0), stem="vocals"):
    sid = models.upsert_song("T", "A", f"https://sc/{stem}", 200, "Pop",
                             status="analysed")
    p = _write_stem(tmp_path / "stems" / f"{stem}.wav")
    models.upsert_stem(sid, stem, str(p))
    models.upsert_features(sid, stem, {
        "bpm": 120.0, "hook_start": hook[0], "hook_end": hook[1],
        "hook_role": "vocal" if stem == "vocals" else "bed",
    })
    return sid


# ── rendering ────────────────────────────────────────────────────────────────

def test_render_cuts_exactly_the_hook_window(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models, hook=(10.0, 42.0))   # 32s, inside the 60s stem

    out = hook_worker.render_hook(sid, "vocals")

    assert Path(out).exists()
    info = sf.info(str(out))
    assert info.duration == pytest.approx(32.0, abs=0.05)
    assert info.samplerate == SR


def test_render_lands_in_the_hooks_dir_with_a_stable_name(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models)

    out = Path(hook_worker.render_hook(sid, "vocals"))

    assert out.parent == config.HOOKS_DIR
    assert str(sid) in out.name and "vocals" in out.name
    assert out == Path(hook_worker.hook_clip_path(sid, "vocals"))


def test_second_render_reuses_the_cached_clip(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models)

    first = Path(hook_worker.render_hook(sid, "vocals"))
    stamp = first.stat().st_mtime_ns
    second = Path(hook_worker.render_hook(sid, "vocals"))

    assert second == first
    assert second.stat().st_mtime_ns == stamp, "cached clip should not be re-rendered"


def test_clip_stays_small_enough_to_fetch_between_keypresses(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models, hook=(0.0, 32.0))

    out = Path(hook_worker.render_hook(sid, "vocals"))
    assert out.stat().st_size <= 5 * 1024 * 1024


def test_render_clamps_a_hook_that_runs_past_the_end_of_the_audio(tmp_path, monkeypatch):
    """Hook windows come from section times, which can outlast a truncated stem."""
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models, hook=(50.0, 200.0))   # stem is only 60s

    out = hook_worker.render_hook(sid, "vocals")
    assert sf.info(str(out)).duration == pytest.approx(10.0, abs=0.05)


# ── failure modes must be clear, not 500s ────────────────────────────────────

def test_missing_stem_file_raises_a_clear_error(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models)
    Path(models.get_conn().execute(
        "SELECT file_path FROM stems WHERE song_id=?", (sid,)).fetchone()[0]).unlink()

    with pytest.raises(hook_worker.HookRenderError):
        hook_worker.render_hook(sid, "vocals")


def test_track_without_a_hook_raises_rather_than_rendering_silence(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/nohook", 200, "Pop",
                             status="analysed")
    models.upsert_features(sid, "vocals", {"bpm": 120.0})

    with pytest.raises(hook_worker.HookRenderError):
        hook_worker.render_hook(sid, "vocals")


def test_unknown_stem_raises(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models)
    with pytest.raises(hook_worker.HookRenderError):
        hook_worker.render_hook(sid, "banjo")
