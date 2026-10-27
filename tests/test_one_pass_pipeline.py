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
    import api.jobs
    import api.workers.stages as stages
    import api.workers.pipeline_worker as pipeline_worker
    # Dependency order. api.jobs and api.workers.pipeline_worker are in the list
    # because the pipeline tests below drive them: pipeline_worker binds
    # database.models.get_conn/get_song at ITS import, so without the reload
    # next_stage would read the user's live library.
    _MODS = (config, models, api.jobs, stages, pipeline_worker)
    for mod in _MODS:
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
        for mod in _MODS:
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


def test_a_bass_chroma_alone_is_not_an_analysed_track(seeded_song, env, monkeypatch):
    """`status = analysed` means the features rows are there (readme §7). The
    bass writes none, so a track whose full, vocals and instrumental all failed
    is a failed analysis even though its bass chroma was measured — otherwise
    the matcher gets an "analysed" track with nothing to rank."""
    models, stages, db = env
    song_id, paths = seeded_song

    def fake(song_id_, stem_type, path, key, on_progress, extra_steps=(),
             frames_out=None, force_steps=(), only_steps=()):
        if stem_type == "bass":                     # its one group ran
            return {"tonal": {"key": "A", "mode": "minor", "camelot": "8A",
                              "key_confidence": 0.5, "hpcp": [0.0] * 12}}, False
        return {"tonal": {}}, False                 # incomplete core: the stem fails

    monkeypatch.setattr(stages, "_run_essentia", fake)
    monkeypatch.setattr(stages, "effective_analyzer", lambda: ("essentia", "essentia"))
    monkeypatch.setattr(stages, "require_analyzer", lambda: None)

    analysed, failed = stages._analyze_stems(song_id, paths,
                                             stages._ANALYSIS_STEM_ORDER, None)
    assert analysed == ["bass"]
    assert set(failed) == {"full", "vocals", "instrumental"}

    with pytest.raises(stages.StageError, match="every stem"):
        stages.do_analyze(song_id)
    assert models.get_song(song_id, db_path=db)["status"] == "error_analysis"


# ── One cut, one score (Task 5) ───────────────────────────────────────────────

def test_structure_is_the_last_stage_so_allin1_cuts_once():
    from api.workers import pipeline_worker as pw

    assert pw.STAGES[-1] == "structure"
    assert pw.STAGES.index("allin1") < pw.STAGES.index("structure")


def test_every_stage_has_a_queue_a_pool_and_a_gate():
    """A stage with no queue raises KeyError in queue_runner._put; one with no
    pool draws as a blank segment on the Queue screen."""
    import api.queue_runner as queue_runner
    from api import queue_eta
    from api.routes import jobs as jobs_route
    from api.workers import pipeline_worker as pw
    from api.workers import stages as stage_mod

    assert set(queue_runner._QUEUES) == set(pw.STAGES)
    assert set(queue_runner._STAGE_WORKERS) == set(pw.STAGES)
    assert set(stage_mod._STAGE_GATES) == set(pw.STAGES)
    # Structure shares the analysis gate: it is the same librosa-bound work.
    assert stage_mod._STAGE_GATES["structure"] is stage_mod._STAGE_GATES["analysis"]
    # Every stage record maps to a pool, and the two mappings agree.
    for stage in pw.STAGES:
        assert jobs_route._POOL_OF[stage] in queue_runner._QUEUES
        assert queue_eta.POOL_OF[stage] == jobs_route._POOL_OF[stage]
    assert set(queue_eta.POOL_GROUPS) == set(pw.STAGES)
    assert set(queue_eta.RECORD_GROUPS) == set(pw.STAGES)
    assert queue_eta._STAGE_ORDER == pw.STAGES


def test_the_frontend_lists_the_stages_in_the_order_they_run():
    """The Queue timeline is read left to right. Its stage list and its pool
    cards are hand-written in the JSX; the API decides what they describe."""
    import re

    from api.workers import pipeline_worker as pw

    src = ROOT / "frontend" / "src"
    use_queue = (src / "hooks" / "useQueue.js").read_text(encoding="utf-8")
    block = use_queue[use_queue.index("export const STAGES = ["):]
    block = block[:block.index("];")]
    assert re.findall(r'\["(\w+)",', block) == list(pw.STAGES)

    screen = (src / "components" / "QueueScreen.jsx").read_text(encoding="utf-8")
    pools = screen[screen.index("const POOLS = ["):]
    pools = pools[:pools.index("];")]
    assert re.findall(r'\["(\w+)",', pools) == list(pw.STAGES)


def test_needs_scoring_is_false_until_an_input_changes(seeded_song, env):
    from database.models import get_conn, mark_scored, needs_scoring

    song_id, _ = seeded_song
    assert needs_scoring(song_id) is True          # never scored

    mark_scored(song_id)
    assert needs_scoring(song_id) is False

    conn = get_conn()
    try:
        conn.execute("UPDATE songs SET pairs_input_at=datetime('now','+1 second') "
                     "WHERE id=?", (song_id,))
        conn.commit()
    finally:
        conn.close()
    assert needs_scoring(song_id) is True


def test_a_track_that_was_never_told_its_inputs_changed_is_not_rescored(seeded_song,
                                                                       env):
    """A row whose features predate pairs_input_at: scored once, then left
    alone. A full Score library is what covers it, not every pipeline pass."""
    from database.models import get_conn, mark_scored, needs_scoring

    song_id, _ = seeded_song
    conn = get_conn()
    try:
        conn.execute("UPDATE songs SET pairs_input_at=NULL WHERE id=?", (song_id,))
        conn.commit()
    finally:
        conn.close()
    assert needs_scoring(song_id) is True
    mark_scored(song_id)
    assert needs_scoring(song_id) is False


def test_a_track_is_scored_once_per_set_of_sections(monkeypatch, seeded_song, env):
    """Three scoring passes per track was three library-wide scans."""
    from api.workers import pipeline_worker as pw
    from database.models import get_conn, mark_pairs_input

    song_id, _ = seeded_song
    calls = []

    def fake_score(sid):
        calls.append(sid)
        from database.models import mark_scored
        mark_scored(sid)

    monkeypatch.setattr("matcher.match.score_song_pairs", fake_score)

    conn = get_conn()
    try:
        mark_pairs_input(conn, song_id)
        conn.commit()
    finally:
        conn.close()

    pw._score_track(song_id, "analysed")
    pw._score_track(song_id, "allin1")       # nothing changed in between
    pw._score_track(song_id, "finalize")
    assert calls == [song_id]


def test_a_per_track_score_leaves_the_docks_staleness_alone(seeded_song, env):
    """readme section 7: a per-track run never stands in for Score library.
    Stamping scored_at must not touch app_prefs.last_score, or the dock would
    stop offering the full re-score the library still needs."""
    from matcher.match import score_song_pairs

    models, _stages, db = env
    song_id, _ = seeded_song

    models.update_song_status(song_id, "analysed")
    models.upsert_features(song_id, "full", {"bpm": 124.0}, db_path=db)
    models.set_pref("last_score", {"started_at": "2000-01-01 00:00:00"},
                    db_path=db)
    assert models.pairs_freshness(db_path=db)["changed_tracks"] == 1

    score_song_pairs(song_id, db_path=db)

    assert models.get_pref("last_score", db_path=db)["started_at"] \
        == "2000-01-01 00:00:00"
    assert models.pairs_freshness(db_path=db)["changed_tracks"] == 1
    conn = models.get_conn(db)
    try:
        assert conn.execute("SELECT scored_at FROM songs WHERE id=?",
                            (song_id,)).fetchone()[0] is not None
    finally:
        conn.close()


def test_the_provisional_pairs_still_land_and_are_then_replaced(monkeypatch,
                                                                seeded_song, env):
    """readme section 3: the first pairs land straight after the quick tier, the
    mix standing in for both stems. Score once must not mean score later."""
    from api import jobs
    from api.workers import pipeline_worker as pw
    from database.models import get_conn, mark_pairs_input

    song_id, _ = seeded_song
    scored = []

    def fake_score(sid):
        scored.append(sid)
        from database.models import mark_scored
        mark_scored(sid)

    monkeypatch.setattr("matcher.match.score_song_pairs", fake_score)
    monkeypatch.setattr(pw.stages, "do_quick", lambda sid, cb=None: {})

    conn = get_conn()
    try:
        conn.execute("UPDATE songs SET status='downloaded' WHERE id=?", (song_id,))
        mark_pairs_input(conn, song_id)
        conn.commit()
    finally:
        conn.close()

    job = jobs.new_job(kind="pipeline", song_id=song_id)
    assert pw.run_stage(job, song_id, "quick") == "next"
    assert scored == [song_id]                    # the provisional pairs

    # The analysis that follows changes an input, so the real pairs replace them.
    conn = get_conn()
    try:
        conn.execute("UPDATE songs SET pairs_input_at=datetime('now','+5 second') "
                     "WHERE id=?", (song_id,))
        conn.commit()
    finally:
        conn.close()
    pw._score_track(song_id, "analysed")
    assert scored == [song_id, song_id]


def test_an_analysed_track_with_provisional_rows_is_always_rescored(seeded_song,
                                                                    env):
    """songs.pairs_input_at has one-second resolution, so an analysis served
    entirely from the caches can finish inside the same second as the
    provisional score. The prov chip must not survive that."""
    from api.workers import pipeline_worker as pw
    from database.models import mark_scored
    from matcher.match import PROVISIONAL_SCORER

    models, _stages, db = env
    song_id, _ = seeded_song
    models.update_song_status(song_id, "analysed")
    mark_scored(song_id)
    assert pw._wants_scoring(song_id) is False

    other = models.upsert_song(title="U", artist="A", source_url="u://2",
                               db_path=db)
    conn = models.get_conn(db)
    try:
        conn.execute(
            "INSERT INTO mashup_candidates (combo_type, vocal_song_id, "
            "inst_song_id, score_total, scorer) VALUES "
            "('vocal_over_instrumental', ?, ?, 0.5, ?)",
            (song_id, other, PROVISIONAL_SCORER))
        conn.commit()
    finally:
        conn.close()
    assert pw._wants_scoring(song_id) is True


def test_essentia_cut_is_skipped_under_the_allin1_engine(monkeypatch, seeded_song,
                                                         env):
    from api.workers import pipeline_worker as pw

    song_id, _ = seeded_song
    monkeypatch.setattr("config.current_segmentation_engine", lambda: "allin1")
    monkeypatch.setattr("config.current_dual_cut", lambda: False)
    monkeypatch.setattr(pw, "_allin1_section_count", lambda sid: 7)
    assert pw._wants_essentia_cut(song_id) is False

    # allin1 found nothing: the Essentia cut is the fallback, not an option.
    monkeypatch.setattr(pw, "_allin1_section_count", lambda sid: 0)
    assert pw._wants_essentia_cut(song_id) is True

    # The Engines comparison needs both cuts.
    monkeypatch.setattr(pw, "_allin1_section_count", lambda sid: 7)
    monkeypatch.setattr("config.current_dual_cut", lambda: True)
    assert pw._wants_essentia_cut(song_id) is True

    # The essentia engine: the novelty cut IS the engine.
    monkeypatch.setattr("config.current_segmentation_engine", lambda: "essentia")
    monkeypatch.setattr("config.current_dual_cut", lambda: False)
    assert pw._wants_essentia_cut(song_id) is True


def test_dual_cut_defaults_off(monkeypatch):
    import config

    monkeypatch.delenv("MASHUP_DUAL_CUT", raising=False)
    assert config.current_dual_cut() is False
    monkeypatch.setenv("MASHUP_DUAL_CUT", "1")
    assert config.current_dual_cut() is True


def test_the_structure_stage_is_terminal(monkeypatch, seeded_song, env):
    """detect_sections falls back to the track BPM, so a section it could not
    measure leaves sections_are_current False for good. Asking next_stage again
    after the structure stage would re-queue the track for its own last stage
    forever."""
    from api import jobs
    from api.workers import bulk_worker
    from api.workers import pipeline_worker as pw

    models, _stages, _db = env
    song_id, _ = seeded_song
    models.update_song_status(song_id, "analysed")
    monkeypatch.setattr(bulk_worker, "sections_are_current", lambda *a, **k: False)
    monkeypatch.setattr(pw.stages, "do_structure", lambda sid, cb=None: {})
    monkeypatch.setattr(pw, "_score_track", lambda *a, **k: None)

    assert pw.next_stage(song_id) == "structure"
    job = jobs.new_job(kind="pipeline", song_id=song_id)
    assert pw.run_stage(job, song_id, "structure") == "done"
    assert jobs.get(job)["status"] == "completed"


def test_the_structure_stage_skips_the_cut_allin1_already_made(monkeypatch,
                                                               seeded_song, env):
    from api import jobs
    from api.workers import bulk_worker
    from api.workers import pipeline_worker as pw

    models, _stages, _db = env
    song_id, _ = seeded_song
    models.update_song_status(song_id, "analysed")
    cut = []
    monkeypatch.setattr(bulk_worker, "sections_are_current", lambda *a, **k: False)
    monkeypatch.setattr(pw.stages, "do_structure",
                        lambda sid, cb=None: cut.append(sid) or {})
    monkeypatch.setattr(pw, "_score_track", lambda *a, **k: None)
    monkeypatch.setattr("config.current_segmentation_engine", lambda: "allin1")
    monkeypatch.setattr(pw, "_allin1_section_count", lambda sid: 7)

    job = jobs.new_job(kind="pipeline", song_id=song_id)
    assert pw.run_stage(job, song_id, "structure") == "done"
    assert cut == []
    assert jobs.get(job)["stages"]["structure"]["state"] == "skipped"
