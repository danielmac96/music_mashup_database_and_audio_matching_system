"""detect_sections reads the stored frames instead of decoding and recomputing.

The structure stage was 88.6 s a track, 57.1 s of it a librosa CQT chroma over
each stem for numbers Essentia had already computed and discarded. Task 3 keeps
them as .npz sidecars (analysis/timeseries.py); these pin that `detect_sections`
cuts from them and touches neither the decoder nor a librosa transform.

Two things here are load-bearing beyond "it runs":

* ``_frames`` must give a long enough track. ``4 * SECTION_MIN_LEN_SECS`` (48 s)
  is the short-track early return, and a 600-frame stem is 27.9 s — every
  assertion about per-stem measurements would then be made about a single
  stand-in section.
* A BASS sidecar holds chroma only (Task 3 analyses it with the tonal group
  alone), so ``test_bass_sidecar_holds_chroma_only`` is the shape that really
  reaches this code. Nothing may read the bass's ``bands`` or ``curves``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Long enough to clear structure's short-track early return (4 * 12 s), on the
# shared 46.44 ms grid: 3000 frames is 139 s.
N_FRAMES = 3000
BEAT_SECS = 0.46875            # 128 BPM


def _frames(n=N_FRAMES, chroma_only=False, seed=11):
    from analysis import timeseries as ts

    rng = np.random.default_rng(seed)
    chroma = rng.random((12, n)).astype(np.float32)
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True)
    if chroma_only:
        # Exactly what Task 3 writes for the bass stem: the tonal group only.
        return ts.StemFrames(step=ts.STEP_SECS, chroma=chroma)
    return ts.StemFrames(
        step=ts.STEP_SECS,
        chroma=chroma,
        mfcc=(rng.random((13, n)) * 20 - 10).astype(np.float32),
        bands=(rng.random((8, n)) * 1e5).astype(np.float32),
        curves=(rng.random((len(ts.CURVE_ROWS), n)) * 0.3).astype(np.float32),
    )


def _beat_times(n=N_FRAMES):
    from analysis import timeseries as ts
    return [i * BEAT_SECS for i in range(int(n * ts.STEP_SECS / BEAT_SECS))]


def _grid(n=N_FRAMES, **extra):
    return {"beat_times": _beat_times(n), "bpm": 128.0, "beat_phase": 0, **extra}


def _track(*, bass_chroma_only=False, stems=("full", "vocals", "instrumental", "bass")):
    from analysis import timeseries as ts
    return ts.TrackFrames(stems={
        k: _frames(chroma_only=(bass_chroma_only and k == "bass"), seed=11 + i)
        for i, k in enumerate(stems)})


def test_sections_from_frames_do_not_decode(tmp_path, monkeypatch):
    """The whole point: no load_mono, no chroma_cqt, no second STFT."""
    from analysis import decode, frames as frames_mod, structure

    def boom(*a, **k):                      # noqa: ANN002, ANN003
        raise AssertionError("structure decoded or recomputed with frames given")

    monkeypatch.setattr(decode, "load_mono", boom)
    for name in ("chroma_cqt", "mfcc", "power_stats", "beat_track", "onset_env",
                 "rms"):
        monkeypatch.setattr(frames_mod, name, boom)

    out = structure.detect_sections(
        tmp_path / "nope.wav", tmp_path / "nope_v.wav",
        inst_path=tmp_path / "nope_i.wav", bass_path=tmp_path / "nope_b.wav",
        grid=_grid(), frames=_track(), timings={})

    assert out, "no sections cut from stored frames"
    assert all(s["end_sec"] > s["start_sec"] for s in out)


def test_sections_from_frames_carry_every_per_stem_measure(tmp_path):
    from analysis import structure

    timings: dict = {}
    out = structure.detect_sections(
        tmp_path / "a.wav", tmp_path / "v.wav", inst_path=tmp_path / "i.wav",
        bass_path=tmp_path / "b.wav", grid=_grid(),
        frames=_track(), timings=timings)

    assert timings["frame_source"] == structure.FRAME_SOURCE_STORED
    first = out[0]
    for key in ("chroma", "chroma_vocal", "chroma_bed", "bass_chroma",
                "band_energy_vocal", "band_energy_bed", "vocal_activity",
                "energy", "bpm", "beat_times", "downbeats", "section_class"):
        assert key in first, key
    assert len(first["chroma_vocal"]) == 12
    assert len(first["band_energy_vocal"]) == 8
    # The gate (bulk_worker._SECTION_CURRENT_COLUMNS) reads these two columns;
    # a cut that leaves either NULL makes every track permanently stale.
    assert first["chroma"] and first["bpm_source"]


def test_bass_sidecar_holds_chroma_only(tmp_path, monkeypatch):
    """Task 3 runs the tonal group alone on the bass: bands and curves are None.

    Nothing may reach for them — the band occupancy is the vocals' and the
    instrumental's — and the bass chroma must still land on the section.
    """
    from analysis import structure
    from analysis import timeseries as ts

    track = _track(bass_chroma_only=True)
    assert track.get("bass").bands is None
    assert track.get("bass").curves is None

    # A bass StemFrames that raises if anything but its chroma is read.
    class _Strict(ts.StemFrames):
        def __getattribute__(self, name):
            if name in ("bands", "curves", "mfcc", "f0"):
                raise AssertionError(f"structure read the bass stem's {name}")
            return object.__getattribute__(self, name)

    bass = track.get("bass")
    track.stems["bass"] = _Strict(step=bass.step, chroma=bass.chroma)

    out = structure.detect_sections(
        tmp_path / "a.wav", tmp_path / "v.wav", inst_path=tmp_path / "i.wav",
        bass_path=tmp_path / "b.wav", grid=_grid(), frames=track, timings={})
    assert out
    assert len(out[0]["bass_chroma"]) == 12


def test_no_bass_frames_leaves_the_bass_chroma_absent(tmp_path, monkeypatch):
    """Without a bass sidecar the stored path must NOT band-pass the mix — it
    has no signal to band-pass, and reaching for one is a decode."""
    from analysis import decode, structure

    def boom(*a, **k):                      # noqa: ANN002, ANN003
        raise AssertionError("decoded for the bass fallback")

    monkeypatch.setattr(decode, "load_mono", boom)
    monkeypatch.setattr(structure, "_bandpass", boom)

    out = structure.detect_sections(
        tmp_path / "a.wav", tmp_path / "v.wav", inst_path=tmp_path / "i.wav",
        grid=_grid(), frames=_track(stems=("full", "vocals", "instrumental")),
        timings={})
    assert out
    assert "bass_chroma" not in out[0]


def test_fixed_segments_from_frames_use_the_given_boundaries(tmp_path):
    from analysis import structure

    bt = _beat_times()
    out = structure.detect_sections(
        tmp_path / "a.wav", tmp_path / "v.wav", inst_path=tmp_path / "i.wav",
        grid=_grid(downbeats=bt[::4]),
        fixed_segments=[[0.0, 7.5, "intro"], [7.5, 20.0, "chorus"]],
        frames=_track(stems=("full", "vocals", "instrumental")), timings={})

    assert len(out) == 2
    assert out[0]["label"] == "intro"
    assert out[1]["label"] == "chorus"


def test_the_stem_measures_read_the_stored_clock_not_the_librosa_hop(tmp_path):
    """The stored grid is 46.44 ms; the librosa hop is 23.22 ms.

    Converting a section's seconds with the wrong one reads the wrong half of
    the track and still returns a plausible number, so this pins it with a
    step function: the vocal is silent — and all its energy in band 0 — for the
    first half of the frames, and loud in band 7 for the second.
    """
    from analysis import structure
    from analysis import timeseries as ts

    n = N_FRAMES
    half = n // 2
    vocals = _frames(seed=2)
    curves = np.zeros((len(ts.CURVE_ROWS), n), dtype=np.float32)
    curves[ts.CURVE_ROWS.index("rms"), half:] = 0.5
    vocals.curves = curves
    bands = np.zeros((8, n), dtype=np.float32)
    bands[0, :half] = 1.0
    bands[7, half:] = 1.0
    vocals.bands = bands

    track = ts.TrackFrames(stems={"full": _frames(seed=1), "vocals": vocals,
                                  "instrumental": _frames(seed=3)})
    # Fixed boundaries so the spans are exactly where this test says they are.
    split = half * ts.STEP_SECS            # 69.66 s
    out = structure.detect_sections(
        tmp_path / "a.wav", tmp_path / "v.wav", inst_path=tmp_path / "i.wav",
        grid=_grid(), fixed_segments=[[0.0, 60.0, "verse"],
                                      [60.0, n * ts.STEP_SECS, "chorus"]],
        frames=track, timings={})

    assert len(out) == 2
    assert out[0]["end_sec"] < split, "the first section must end before the step"
    # On the right clock the first section holds no sung frame at all; on the
    # librosa hop it would reach past the step and read ~0.4.
    assert out[0]["vocal_activity"] == 0.0
    assert out[1]["vocal_activity"] > 0.5
    # And its band occupancy is entirely band 0, for the same reason.
    assert out[0]["band_energy_vocal"][0] == 1.0
    assert out[1]["band_energy_vocal"][7] > 0.5


def test_stored_frames_without_a_grid_cut_nothing(tmp_path, monkeypatch):
    """Stored frames and no beat grid is a caller error, not a fallback: the
    point of this path is that nothing decodes, and there is no signal to
    beat-track."""
    from analysis import decode, structure

    def boom(*a, **k):                      # noqa: ANN002, ANN003
        raise AssertionError("fell back to decoding")

    monkeypatch.setattr(decode, "load_mono", boom)
    assert structure.detect_sections(tmp_path / "a.wav", frames=_track(),
                                     timings={}) == []


def test_frame_source_says_librosa_when_no_frames_are_given(tmp_path):
    """A short real file: the librosa path still runs and says so."""
    import soundfile as sf

    from analysis import structure

    sr = 22050
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(sr * 2, dtype=np.float32), sr)
    timings: dict = {}
    structure.detect_sections(path, timings=timings)
    assert timings["frame_source"] == structure.FRAME_SOURCE_LIBROSA


def test_librosa_path_is_untouched_when_no_frames():
    """analyzer=librosa (and every existing test) still decodes as before."""
    import inspect

    from analysis import structure

    src = inspect.getsource(structure.detect_sections)
    assert "use_stored" in src
    # The module-level import must not be named `frames` any more: the new
    # parameter shadows it, and a missed call site is an AttributeError only on
    # the path the suite does not take.
    assert "from analysis import frames as frame_ops" in \
        inspect.getsource(structure)
    assert "from analysis import frames\n" not in inspect.getsource(structure)


# ── Real Essentia frames (container only) ────────────────────────────────────

def _has_essentia() -> bool:
    try:
        from analysis.essentia_groups import available
        return available()
    except Exception:  # noqa: BLE001
        return False


needs_essentia = pytest.mark.skipif(not _has_essentia(),
                                    reason="essentia not installed")


@needs_essentia
def test_a_cut_from_real_essentia_frames(tmp_path):
    """End to end on Essentia's own per-frame output.

    The synthetic matrices above pin the arithmetic; this pins that what
    `essentia.tonal` / `essentia.spectral` actually hand over (through
    ``frames_from``) is cuttable — the right shapes, the right grid, a chroma
    whose fold (``hpcp_frames_to_chroma``) still describes the music, and a
    flux curve the grid confidence can read.
    """
    import soundfile as sf

    from analysis import structure
    from analysis.essentia_groups import analyze_file_essentia, frames_from

    sr = 44100
    rng = np.random.default_rng(7)
    # C major then A minor arpeggios, 40 s each, over a 2 Hz pulse so the
    # novelty curve and the flux curve both have something in them.
    tones = {"C": (261.63, 329.63, 392.00), "A": (220.00, 261.63, 329.63)}
    y = []
    for name in ("C", "A"):
        t = np.arange(sr * 40) / sr
        sig = sum(0.2 * np.sin(2 * np.pi * f * t) for f in tones[name])
        sig = sig * (0.6 + 0.4 * (np.sin(2 * np.pi * 2.0 * t) > 0))
        y.append((sig + 0.01 * rng.standard_normal(t.size)).astype(np.float32))
    y = np.concatenate(y)
    path = tmp_path / "mix.wav"
    sf.write(str(path), y, sr)

    frames_out: dict = {}
    payloads = analyze_file_essentia(path, frames_out=frames_out)
    built = frames_from(frames_out)
    assert built is not None
    assert built.chroma is not None and built.mfcc is not None
    assert built.bands is not None and built.curves is not None
    assert built.step == pytest.approx(2048 / 44100.0)

    from analysis import timeseries as ts
    track = ts.TrackFrames(stems={"full": built, "vocals": built,
                                  "instrumental": built})
    grid = {"beat_times": payloads["rhythm"]["beat_times"],
            "bpm": payloads["rhythm"]["bpm"],
            "beat_phase": payloads["rhythm"].get("beat_phase")}
    timings: dict = {}
    out = structure.detect_sections(tmp_path / "absent.wav",
                                    tmp_path / "absent_v.wav",
                                    inst_path=tmp_path / "absent_i.wav",
                                    grid=grid, frames=track, timings=timings)

    assert timings["frame_source"] == structure.FRAME_SOURCE_STORED
    assert "load" not in timings and "frames" in timings
    assert out and len(out) >= 2
    # The fold is right if the first section's chroma peaks on a C-major note
    # and the last one's on an A-minor note. Index 0 is C.
    first, last = out[0], out[-1]
    assert int(np.argmax(first["chroma"])) in (0, 4, 7), first["chroma"]
    assert int(np.argmax(last["chroma"])) in (9, 0, 4), last["chroma"]
    for sec in out:
        assert len(sec["chroma_vocal"]) == 12
        assert len(sec["band_energy_bed"]) == 8
        assert sec["vocal_activity"] is not None
        assert sec["bpm_source"]


# ── The stage, and the gate that decides a track is done ─────────────────────
#
# `config.DB_PATH` binds at import, so setting MASHUP_DB_PATH without reloading
# would do nothing and do_structure — which opens its own connections — would
# write sections into the user's real library.

_PATH_VARS = ("MASHUP_AUDIO_ROOT", "MASHUP_DB_PATH", "MASHUP_SETTINGS_DIR")


@pytest.fixture
def env(tmp_path, monkeypatch):
    import importlib
    import os

    prev = {k: os.environ.get(k) for k in _PATH_VARS}
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    import api.workers.bulk_worker as bulk
    import api.workers.stages as stages
    import database.models as models
    for mod in (config, models, stages, bulk):
        importlib.reload(mod)
    models.init_db()
    from analysis import decode
    decode.clear()
    try:
        yield models, stages, bulk, tmp_path / "mashup.db"
    finally:
        for key, val in prev.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        for mod in (config, models, stages, bulk):
            importlib.reload(mod)
        decode.clear()


def _seed_for_structure(models, tmp_path, db, monkeypatch):
    """A 140 s song with three stems on disk and a sidecar for each, so
    do_structure can cut from the frames alone."""
    import soundfile as sf

    from analysis import timeseries as ts

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    sr = 22050
    y = np.zeros(sr * 140, dtype=np.float32)
    paths = {}
    for stem in ("full", "vocals", "instrumental", "bass"):
        p = tmp_path / f"{stem}.wav"
        sf.write(str(p), y, sr)
        paths[stem] = str(p)
    song_id = models.upsert_song(title="T", artist="A", source_url="u://1",
                                 raw_path=paths["full"], duration_secs=140,
                                 db_path=db)
    for stem, p in paths.items():
        models.upsert_stem(song_id, stem, p, separator="demucs:htdemucs:2",
                           db_path=db)
    for i, stem in enumerate(("full", "vocals", "instrumental")):
        ts.write(song_id, stem, "legacy_2stem", _frames(seed=3 + i), f"h{i}",
                 db_path=db)
    ts.write(song_id, "bass", "legacy_2stem",
             _frames(chroma_only=True, seed=9), "hb", db_path=db)
    return song_id, paths


def test_the_stage_cuts_from_the_frames_and_the_gate_says_current(
        env, tmp_path, monkeypatch):
    """Trap 4: a cut from stored frames must still satisfthe structure gate.

    `bulk_worker.sections_are_current` is what decides whether a track needs
    re-cutting. A section missing any column it checks makes every track
    permanently stale and bulk re-analysis loop forever.
    """
    models, stages, bulk, db = env
    song_id, _paths = _seed_for_structure(models, tmp_path, db, monkeypatch)

    # Essentia mode with a beat grid in the feature cache, which is what puts
    # do_structure on the stored path at all.
    import analysis.cache as cache
    from analysis.registry import GROUPS

    bt = _beat_times()
    monkeypatch.setattr(stages, "effective_analyzer",
                        lambda: ("essentia", "essentia"))
    monkeypatch.setattr(
        cache, "lookup",
        lambda grp, h: ({"beat_times": bt, "bpm": 128.0, "beat_phase": 0}
                        if grp is GROUPS["essentia.rhythm"] else None))
    monkeypatch.setattr(stages, "warm_hooks", lambda *a, **k: 0, raising=False)

    from analysis import decode
    calls: list = []
    real_load = decode.load_mono
    monkeypatch.setattr(decode, "load_mono",
                        lambda *a, **k: (calls.append(a[0]), real_load(*a, **k))[1])

    out = stages.do_structure(song_id)
    assert out["section_count"] >= 1
    assert bulk.sections_are_current(song_id, db_path=db) is True

    rows = models.get_sections(song_id, db_path=db)
    assert rows
    for r in rows:
        # The two _SECTION_CURRENT_COLUMNS, and the conditional one the vocal
        # stem's presence turns on.
        assert r["chroma"] and r["bpm_source"]
        assert r["vocal_activity"] is not None
        assert r["chroma_vocal"] and r["chroma_bed"] and r["bass_chroma"]
        assert r["band_energy_vocal"] and r["band_energy_bed"]
        assert r["provisional"] in (0, False)

    # detect_sections decoded nothing. The one decode left in the stage is the
    # vocal stem, and it belongs to store_phrases (analysis/phrases.py reads a
    # librosa-hop RMS envelope), not to the cut — the plan's Task 4 does not
    # touch it. The mix is never decoded at all, which is what the 57 s of CQT
    # chroma was.
    assert [p.name for p in calls] == ["vocals.wav"], calls

    import sqlite3
    conn = sqlite3.connect(db)
    grps = [g for (g,) in conn.execute(
        "SELECT grp FROM analysis_runs WHERE grp LIKE 'structure%'").fetchall()]
    conn.close()
    assert "structure.load" not in grps, grps
    assert "structure.frames" in grps, grps


def test_shadow_mode_keeps_the_librosa_path(env, tmp_path, monkeypatch):
    """Sidecars exist in shadow mode too, but the core — and the beat grid —
    are librosa's. Without a grid the stored path has nothing to cut on, so the
    stage must not hand it the frames."""
    models, stages, _bulk, db = env
    song_id, _paths = _seed_for_structure(models, tmp_path, db, monkeypatch)

    monkeypatch.setattr(stages, "effective_analyzer", lambda: ("shadow", "librosa"))
    seen: dict = {}
    real = stages._load_frames
    monkeypatch.setattr(stages, "_load_frames",
                        lambda sid: (seen.setdefault("called", True), real(sid))[1])
    monkeypatch.setattr(stages, "warm_hooks", lambda *a, **k: 0, raising=False)

    # These stems are silence, so librosa finds no beats and the cut is empty —
    # which is the librosa path's own answer and exactly the point: the stored
    # frames were never offered, so nothing returned [] for want of a grid.
    with pytest.raises(stages.StageError):
        stages.do_structure(song_id)
    assert "called" not in seen


def test_sections_match_the_librosa_cut_on_the_same_numbers(tmp_path):
    """The stored path is the same arithmetic on the same matrices.

    Feeding the sidecar matrices that librosa's own transforms produce for a
    signal must give the same boundaries, labels, energies and chroma as the
    librosa path over that signal — the source of the numbers changed, not the
    maths.

    The one difference is the seconds → frame-index conversion, and it is a
    clock, not arithmetic: ``librosa.time_to_frames`` FLOORS, while
    ``sync_by_time`` (and therefore this path's ``beats``) ROUNDS, so a
    section's start can sit one frame — 23 ms here — either side. Rounding both
    to 2 dp widens that to 0.03. Everything measured over ~43 beats moves by
    far less.
    """
    import soundfile as sf

    from analysis import frames as frame_ops, structure
    from analysis import timeseries as ts
    from config import HOP_LENGTH, SAMPLE_RATE

    sr = SAMPLE_RATE
    rng = np.random.default_rng(5)
    secs = 80
    # Three 20 s timbres so the novelty curve has something to find.
    parts = []
    for k, f in enumerate((220.0, 440.0, 220.0, 660.0)):
        t = np.arange(sr * 20) / sr
        parts.append((0.3 * np.sin(2 * np.pi * f * t)
                      + 0.02 * rng.standard_normal(t.size)).astype(np.float32))
    y = np.concatenate(parts)[:sr * secs]
    path = tmp_path / "mix.wav"
    sf.write(path, y, sr)

    grid = {"beat_times": [i * BEAT_SECS for i in range(int(secs / BEAT_SECS))],
            "bpm": 128.0, "beat_phase": 0}

    librosa_out = structure.detect_sections(path, timings={}, grid=grid)

    # The same matrices, on the librosa hop clock rather than the 46 ms one.
    step = HOP_LENGTH / sr
    chroma = frame_ops.chroma_cqt(y, sr, HOP_LENGTH)
    mfcc = frame_ops.mfcc(y, sr, 13, HOP_LENGTH)
    rms = frame_ops.rms(y, HOP_LENGTH)[0]
    onset = frame_ops.onset_env(y, sr, HOP_LENGTH)
    n = min(chroma.shape[1], mfcc.shape[1], rms.size, onset.size)
    curves = np.zeros((len(ts.CURVE_ROWS), n), dtype=np.float32)
    curves[ts.CURVE_ROWS.index("rms")] = rms[:n]
    curves[ts.CURVE_ROWS.index("flux")] = onset[:n]
    full = ts.StemFrames(step=step, chroma=chroma[:, :n], mfcc=mfcc[:, :n],
                         curves=curves)
    stored_out = structure.detect_sections(
        tmp_path / "absent.wav", grid=grid,
        frames=ts.TrackFrames(stems={"full": full}), timings={})

    one_frame = step + 0.01          # + the two 2-dp roundings
    assert len(stored_out) == len(librosa_out)
    assert [s["label"] for s in stored_out] == [s["label"] for s in librosa_out]
    assert [s["start_sec"] for s in stored_out] == \
        pytest.approx([s["start_sec"] for s in librosa_out], abs=one_frame)
    assert [s["beat_count"] for s in stored_out] == \
        [s["beat_count"] for s in librosa_out]
    for a, b in zip(stored_out, librosa_out):
        assert a["energy"] == pytest.approx(b["energy"], abs=0.005)
        assert a["chroma"] == pytest.approx(b["chroma"], abs=0.01)
        assert a["bpm"] == pytest.approx(b["bpm"], abs=0.1)
        # Steadiness x salience, and these sines have no onsets at all, so the
        # salience term is noise either way — both read near zero. A one-frame
        # shift in where the beats land is the whole difference.
        assert a["bpm_confidence"] == pytest.approx(b["bpm_confidence"], abs=0.05)
