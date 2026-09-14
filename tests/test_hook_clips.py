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


# ── T3.3: an arbitrary section window, not just the track's own hook ─────────

def test_window_render_cuts_exactly_that_span(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models, hook=(0.0, 32.0))

    out = hook_worker.render_hook(sid, "vocals", start=12.5, end=30.5)

    assert sf.info(out).duration == pytest.approx(18.0, abs=0.05)
    # A distinct file from the track's own hook — both stay cached side by side.
    assert Path(out) != Path(hook_worker.hook_clip_path(sid, "vocals"))
    assert Path(hook_worker.render_hook(sid, "vocals")) != Path(out)


def test_window_render_is_cached_by_the_window(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models)

    first = Path(hook_worker.render_hook(sid, "vocals", start=5.0, end=25.0))
    stamp = first.stat().st_mtime_ns
    again = Path(hook_worker.render_hook(sid, "vocals", start=5.0, end=25.0))
    other = Path(hook_worker.render_hook(sid, "vocals", start=6.0, end=26.0))

    assert again == first and again.stat().st_mtime_ns == stamp
    assert other != first


def test_a_windowed_track_needs_no_stored_hook(tmp_path, monkeypatch):
    """Section windows come from the candidate row, so a track whose own hook was
    never chosen can still be previewed."""
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/win", 200, "Pop",
                             status="analysed")
    models.upsert_stem(sid, "vocals", str(_write_stem(tmp_path / "stems" / "v.wav")))
    models.upsert_features(sid, "vocals", {"bpm": 120.0})

    out = hook_worker.render_hook(sid, "vocals", start=1.0, end=11.0)
    assert sf.info(out).duration == pytest.approx(10.0, abs=0.05)


def test_empty_window_raises(tmp_path, monkeypatch):
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models)
    with pytest.raises(hook_worker.HookRenderError):
        hook_worker.render_hook(sid, "vocals", start=20.0, end=20.0)


def test_warm_hooks_force_recuts_a_moved_window(tmp_path, monkeypatch):
    """Re-running structure moves the hook. The clip cache is keyed by
    (song, stem), so without force the old 16 bars would be served for ever
    while hook_start/hook_end in the DB said something else."""
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = _seed(tmp_path, models, hook=(0.0, 10.0))
    models.upsert_stem(sid, "instrumental",
                       str(_write_stem(tmp_path / "stems" / "i.wav")))
    models.upsert_features(sid, "instrumental",
                           {"bpm": 120.0, "hook_start": 0.0, "hook_end": 10.0})

    hook_worker.warm_hooks(sid)
    assert sf.info(hook_worker.hook_clip_path(sid, "vocals")).duration \
        == pytest.approx(10.0, abs=0.05)

    # Structure re-run: the window moves.
    models.update_hook(sid, "vocals", {"hook_start": 20.0, "hook_end": 50.0})
    models.update_hook(sid, "instrumental", {"hook_start": 20.0, "hook_end": 50.0})

    hook_worker.warm_hooks(sid)          # without force: stale
    assert sf.info(hook_worker.hook_clip_path(sid, "vocals")).duration \
        == pytest.approx(10.0, abs=0.05)

    hook_worker.warm_hooks(sid, force=True)
    for stem in ("vocals", "instrumental"):
        assert sf.info(hook_worker.hook_clip_path(sid, stem)).duration \
            == pytest.approx(30.0, abs=0.05)


# ── the 'full' stem: an mp3 source, and no stems row ─────────────────────────
# Every fixture above writes a .wav stem, which is why the bug below survived
# for as long as it did. Demucs writes FLAC and the downloader writes MP3, and
# only one of those round-trips through a WAV container.


def test_a_compressed_source_is_written_as_pcm_not_copied(tmp_path, monkeypatch):
    """The clip inherited the SOURCE's subtype.

    For 'full' that source is the downloaded mp3, whose subtype is
    MPEG_LAYER_III — and libsndfile refuses to write that into a WAV container
    ("Supported file format but unsupported encoding"). So every section preview
    on the Full stem 404'd, which the browser reports as "Failed to load because
    no supported source was found". The track screen defaults to Full, so that
    was every section button on it.
    """
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    if "MP3" not in sf.available_formats():
        pytest.skip("this libsndfile cannot write mp3 to build the fixture")

    sid = models.upsert_song("T", "A", "https://sc/mp3", 60, "Pop",
                             status="analysed")
    src = tmp_path / "audio" / "full.mp3"
    src.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 60.0, int(SR * 60.0), endpoint=False)
    sf.write(str(src), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR)
    assert sf.info(str(src)).subtype == "MPEG_LAYER_III"
    models.upsert_stem(sid, "full", str(src))

    out = hook_worker.render_hook(sid, "full", start=10.0, end=18.0)

    info = sf.info(out)
    assert info.format == "WAV"
    assert info.subtype.startswith("PCM_"), info.subtype
    assert info.duration == pytest.approx(8.0, abs=0.05)


def test_an_uncompressed_source_keeps_its_own_subtype(tmp_path, monkeypatch):
    """Only compressed sources are re-encoded. A 24-bit stem must not be
    silently downgraded on its way into a preview."""
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/24", 60, "Pop",
                             status="analysed")
    src = tmp_path / "stems" / "v24.wav"
    src.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 60.0, int(SR * 60.0), endpoint=False)
    sf.write(str(src), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"),
             SR, subtype="PCM_24")
    models.upsert_stem(sid, "vocals", str(src))

    out = hook_worker.render_hook(sid, "vocals", start=1.0, end=5.0)
    assert sf.info(out).subtype == "PCM_24"


def test_full_falls_back_to_raw_path_with_no_stems_row(tmp_path, monkeypatch):
    """The audio route resolved 'full' via songs.raw_path when no stems row
    existed; the hook renderer read the stems table and nothing else. A library
    imported before the pipeline started writing that row therefore played from
    the library and 404'd on the track screen. One resolver now, so they cannot
    disagree."""
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    raw = _write_stem(tmp_path / "audio" / "raw.wav")
    sid = models.upsert_song("T", "A", "https://sc/raw", 60, "Pop",
                             status="analysed")
    models.update_song_status(sid, "analysed", raw_path=str(raw))
    # Deliberately NO upsert_stem(sid, "full", ...).

    out = hook_worker.render_hook(sid, "full", start=2.0, end=9.0)
    assert sf.info(out).duration == pytest.approx(7.0, abs=0.05)


def test_a_missing_stem_is_still_a_clean_error(tmp_path, monkeypatch):
    """The fallback must not turn "no vocals separated yet" into a crash or a
    silent empty clip."""
    config, models, hook_worker = _setup(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/none", 60, "Pop",
                             status="analysed")
    with pytest.raises(hook_worker.HookRenderError):
        hook_worker.render_hook(sid, "vocals", start=1.0, end=5.0)
