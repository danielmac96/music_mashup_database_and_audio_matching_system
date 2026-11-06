"""The allin1 stage (readme §5.5): summarising, storing and caching its output,
its sections measured like the Essentia cut, phrases on its grid, routing and
failure handling. allin1 itself is faked — the suite runs without torch's
model weights — at the one seam it has: analysis/allin1_runner.run."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SECS = 64.0


def _wav(path: Path, secs: float = SECS, sr: int = 22050, seed_hz: float = 220.0,
         sung=None) -> Path:
    """A 120 BPM click-and-chord mix, or — with ``sung`` spans — a vocal stem
    that sings only inside them."""
    import soundfile as sf
    n = int(secs * sr)
    t = np.arange(n) / sr
    if sung is not None:
        y = np.zeros(n, dtype=np.float32)
        for s, e in sung:
            m = (t >= s) & (t < e)
            y[m] = 0.6 * np.sin(2 * np.pi * seed_hz * t[m])
        y[:10] += 1e-4
    else:
        y = np.zeros(n, dtype=np.float32)
        for i, bt in enumerate(np.arange(0, secs, BEAT)):
            s = int(bt * sr)
            e = min(n, s + int(0.08 * sr))
            tt = np.arange(e - s) / sr
            y[s:e] += (1.0 if i % 4 == 0 else 0.5) * np.sin(2 * np.pi * 80 * tt) * np.exp(-tt * 30)
        y += 0.2 * np.sin(2 * np.pi * seed_hz * t) * (((t // (8 * BAR)) % 2) + 0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (y / max(1e-9, np.abs(y).max())).astype(np.float32), sr)
    return path


class FakeResult:
    """allin1.typings.AnalysisResult's fields, as allin1 1.1.0 fills them."""

    def __init__(self, secs=SECS):
        frames = int(secs * 100)
        self.bpm = 120
        self.beats = [round(i * BEAT, 2) for i in range(int(secs / BEAT))]
        self.downbeats = [round(i * BAR, 2) for i in range(int(secs / BAR))]
        self.beat_positions = [(i % 4) + 1 for i in range(len(self.beats))]
        seg = [(0.0, 0.3, "start"), (0.3, 16.0, "intro"), (16.0, 32.0, "chorus"),
               (32.0, 48.0, "inst"), (48.0, 63.7, "outro"), (63.7, secs, "end")]
        self.segments = [type("S", (), {"start": a, "end": b, "label": c})() for a, b, c in seg]
        db = np.zeros(frames, dtype=np.float32)
        for t in self.downbeats:
            db[int(t * 100)] = 0.9
        # The beat activation is its own curve: allin1 reports one per beat,
        # higher on a bar line. Deliberately NOT a copy of the downbeat
        # activation, so a summary that confuses the two is visible.
        bt = np.zeros(frames, dtype=np.float32)
        for t in self.beats:
            bt[int(t * 100)] = 0.8 if t in self.downbeats else 0.5
        # Label row i holds (i + 1) / 20, so the stored distribution can be
        # checked against HARMONIX_LABELS' order rather than a single constant.
        lab = np.stack([np.full(frames, (i + 1) / 20, np.float32)
                        for i in range(10)])
        self.activations = {"beat": bt, "downbeat": db,
                            "segment": np.full(frames, 0.2, np.float32),
                            "label": lab}
        # (stems, frames, dim, models), as the ensemble stacks them.
        self.embeddings = np.random.default_rng(0).normal(
            size=(4, frames, 24, 8)).astype(np.float32)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    import database.models as models
    import api.jobs
    import api.workers.stages
    import api.workers.pipeline_worker
    import api.queue_runner
    for mod in (config, models, api.jobs, api.workers.stages,
                api.workers.pipeline_worker, api.queue_runner):
        importlib.reload(mod)
    models.init_db()
    from analysis import decode
    decode.clear()
    import api.workers.hook_worker as hook_worker
    monkeypatch.setattr(hook_worker, "warm_hooks", lambda *a, **k: {})
    monkeypatch.setattr(api.workers.stages, "_measure_stem_quality", lambda *a, **k: None)
    from analysis import allin1_runner
    calls = {"run": 0}

    def fake_run(full_path, stems, device="cpu", timings=None, threads=None):
        calls["run"] += 1
        calls["stems"] = stems
        if timings is not None:
            timings["inference"] = 1.0
        return FakeResult()
    monkeypatch.setattr(allin1_runner, "missing", lambda: None)
    monkeypatch.setattr(allin1_runner, "run", fake_run)
    return models, api.workers.stages, tmp_path, calls


def _song(models, tmp, four=True):
    mix = _wav(tmp / "mix.wav")
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "analysed", raw_path=str(mix))
    models.upsert_stem(song, "full", str(mix))
    # Sung from a beat before bar 8 (16 s) and on bar 16 (32 s).
    models.upsert_stem(song, "vocals", str(_wav(
        tmp / "voc.wav", seed_hz=440.0, sung=[(16.0 - BEAT, 28.0), (32.0, 44.0)])))
    models.upsert_stem(song, "instrumental", str(_wav(tmp / "bed.wav", seed_hz=110.0)))
    if four:
        for k, hz in (("drums", 60.0), ("bass", 55.0), ("other", 330.0)):
            models.upsert_stem(song, k, str(_wav(tmp / f"{k}.wav", seed_hz=hz)))
    return song


# ── Summary ───────────────────────────────────────────────────────────────────

def test_summary_keeps_grids_and_pools_embeddings_per_bar():
    from analysis.allin1_runner import from_cache, summarise, to_cache
    s = summarise(FakeResult())
    assert s["downbeats"][:3] == [0.0, 2.0, 4.0]
    assert s["segments"][2] == [16.0, 32.0, "chorus"]
    assert len(s["downbeat_conf"]) == len(s["downbeats"])
    assert s["conf_mean"] == pytest.approx(0.9, abs=1e-3)
    assert s["embedding"].shape == (len(s["downbeats"]), 4, 24)
    assert s["embedding"].dtype == np.float16
    back = from_cache(to_cache(s))
    assert np.array_equal(back["embedding"], s["embedding"])


def test_a_four_minute_track_stores_under_100kb(env):
    models, _stages, _tmp, _calls = env
    from analysis.allin1_runner import summarise
    from database import alignment
    song = models.upsert_song(title="T", artist="A", source_url="http://x/2")
    alignment.store_allin1(song, "unified_4stem", summarise(FakeResult(secs=240.0)))
    sizes = alignment.storage_bytes(song)
    assert sum(sizes.values()) < 100_000, sizes
    got = alignment.get_allin1(song)
    assert got["downbeats"][:2] == [0.0, 2.0]
    assert alignment.get_allin1_embedding(song).shape[1:] == (4, 24)


# ── The stage ─────────────────────────────────────────────────────────────────

def test_do_allin1_hands_over_the_four_stems_and_stores_everything(env):
    models, stages, tmp, calls = env
    from database import alignment
    song = _song(models, tmp)
    out = stages.do_allin1(song)
    assert calls["run"] == 1
    assert set(calls["stems"]) == {"bass", "drums", "other", "vocals"}   # no demix
    assert out["sections"] >= 3 and out["phrases"] >= 1
    assert models.get_song(song)["allin1_state"] == "done"

    secs = alignment.get_allin1_sections(song)
    base = models.ALLIN1_SECTION_BASE
    assert [s["section_index"] for s in secs] == list(range(base, base + len(secs)))
    assert [s["engine_label"] for s in secs][:3] == ["intro", "chorus", "inst"]
    assert secs[1]["label"] == "chorus"
    assert secs[2]["label"] in ("drop", "breakdown")
    # Bar lines are allin1's, not every 4th beat.
    assert secs[1]["downbeats"][0] == pytest.approx(16.0)
    assert secs[1]["vocal_activity"] is not None
    assert secs[1]["downbeat_conf"] == pytest.approx(0.9, abs=1e-3)

    ph = alignment.get_vocal_phrases(song, grid_source="allin1")
    first = ph[0]
    assert first["downbeat_sec"] == pytest.approx(16.0, abs=0.03)
    assert first["pickup_sec"] == pytest.approx(-BEAT, abs=0.05)

    # The Essentia sections are untouched (engine essentia reads them).
    assert all(s["section_index"] < base for s in models.get_sections(song, engine="essentia"))


def test_second_run_is_a_cache_hit(env):
    models, stages, tmp, calls = env
    song = _song(models, tmp)
    stages.do_allin1(song)
    out = stages.do_allin1(song)
    assert calls["run"] == 1 and out["cached"] is True


def test_without_four_stems_allin1_demixes_itself(env):
    models, stages, tmp, calls = env
    from database import alignment
    song = _song(models, tmp, four=False)
    stages.do_allin1(song)
    assert calls["stems"] is None
    assert alignment.get_allin1(song)["demixed_by"] == "allin1"


def test_a_set_length_track_is_skipped(env, monkeypatch):
    models, stages, tmp, calls = env
    monkeypatch.setenv("MASHUP_ALLIN1_MAX_SECS", "30")
    song = _song(models, tmp)
    out = stages.do_allin1(song)
    assert "skipped" in out and calls["run"] == 0
    assert models.get_song(song)["allin1_state"] == "skipped"


def test_missing_dependency_fails_with_the_reason(env, monkeypatch):
    models, stages, tmp, _calls = env
    from analysis import allin1_runner
    monkeypatch.setattr(allin1_runner, "missing", lambda: "natten does not import")
    song = _song(models, tmp)
    with pytest.raises(stages.StageError, match="natten"):
        stages.do_allin1(song)
    row = models.get_song(song)
    assert row["allin1_state"] == "failed" and "natten" in row["allin1_error"]


def test_one_failure_is_retried(env, monkeypatch):
    models, stages, tmp, calls = env
    from analysis import allin1_runner
    n = {"k": 0}

    def flaky(*a, **k):
        n["k"] += 1
        if n["k"] == 1:
            raise RuntimeError("transient")
        return FakeResult()
    monkeypatch.setattr(allin1_runner, "run", flaky)
    song = _song(models, tmp)
    stages.do_allin1(song)
    assert n["k"] == 2 and models.get_song(song)["allin1_state"] == "done"


# ── Routing ───────────────────────────────────────────────────────────────────

def test_next_stage_routes_to_allin1_until_it_has_run(env, monkeypatch):
    models, stages, tmp, _calls = env
    import api.workers.pipeline_worker as pw
    from database import alignment
    song = _song(models, tmp)
    # The cut is a stage of its own after allin1, and this song has no sections,
    # so one is owed; that routing has its own test
    # (tests/test_one_pass_pipeline.py). This test is about allin1.
    monkeypatch.setattr(pw, "_wants_structure", lambda sid: False)
    assert pw.next_stage(song) == "allin1"              # never run: owed
    for state in ("done", "failed", "skipped"):         # each counts as having run
        alignment.set_allin1_state(song, state)
        assert pw.next_stage(song) is None
    alignment.set_allin1_state(song, "queued")          # a backfill asked
    assert pw.next_stage(song) == "allin1"


def test_pipeline_failure_is_non_fatal(env, monkeypatch):
    models, stages, tmp, _calls = env
    import api.jobs as jobs
    import api.workers.pipeline_worker as pw
    from analysis import allin1_runner
    monkeypatch.setattr(allin1_runner, "run", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom")))
    monkeypatch.setattr(pw, "_score_track", lambda *a, **k: None)
    monkeypatch.setattr(pw, "_wants_structure", lambda sid: False)
    song = _song(models, tmp)
    job = jobs.new_job(kind="pipeline", song_id=song)
    assert pw.run_stage(job, song, "allin1") == "done"
    assert models.get_song(song)["allin1_state"] == "failed"
    assert models.get_song(song)["status"] == "analysed"
    assert jobs.get(job)["status"] == "completed"


def test_bulk_allin1_queues_what_is_missing(env):
    models, _stages, tmp, _calls = env
    from api.workers import bulk_worker
    song = _song(models, tmp)
    assert bulk_worker.stale_song_ids("allin1") == [song]
    assert bulk_worker.staleness()["missing_allin1"] == 1


def test_over_the_memory_budget_refuses_instead_of_running(env, monkeypatch):
    models, stages, tmp, calls = env
    monkeypatch.setenv("MASHUP_ALLIN1_MEM_BUDGET_MB", "500")
    song = _song(models, tmp)
    with pytest.raises(stages.StageError, match="budget"):
        stages.do_allin1(song)
    assert calls["run"] == 0
    assert "allin1_mem_budget_mb" in models.get_song(song)["allin1_error"]


# ── The model as a live setting (Task 6) ──────────────────────────────────────

def test_model_comes_from_the_live_setting(monkeypatch, tmp_path):
    """Which checkpoint ran is a setting, so Task 9's benchmark can choose it —
    and an explicit name (the benchmark's) wins over the setting."""
    import config
    from analysis import allin1_runner as runner

    monkeypatch.setattr(config, "settings_path", lambda: tmp_path / "none.json")
    monkeypatch.delenv("MASHUP_ALLIN1_MODEL", raising=False)
    assert runner.model_name() == "harmonix-fold0"        # one fold by default
    monkeypatch.setenv("MASHUP_ALLIN1_MODEL", "harmonix-all")
    assert runner.model_name() == "harmonix-all"          # the ensemble, one setting away
    monkeypatch.setenv("MASHUP_ALLIN1_MODEL", "harmonix-fold3")
    assert config.current_allin1_model() == "harmonix-fold3"
    assert runner.model_name() == "harmonix-fold3"
    assert runner.model_name("harmonix-all") == "harmonix-all"
    # Validated against config.ALLIN1_MODELS, like every other enum setting:
    # a name allin1 does not ship falls back rather than failing a 10-minute
    # inference on a checkpoint that cannot load.
    monkeypatch.setenv("MASHUP_ALLIN1_MODEL", "harmonix-fold9")
    assert runner.model_name() == "harmonix-fold0"


def test_registry_params_track_the_live_model(monkeypatch, tmp_path):
    """Two models' grids must never share a cache key."""
    import config
    from analysis.registry import GROUPS

    monkeypatch.setattr(config, "settings_path", lambda: tmp_path / "none.json")
    monkeypatch.setenv("MASHUP_ALLIN1_MODEL", "harmonix-all")
    before = GROUPS["allin1.analysis"].params_hash()
    before_structure = GROUPS["allin1.structure"].params_hash()
    monkeypatch.setenv("MASHUP_ALLIN1_MODEL", "harmonix-fold0")
    assert GROUPS["allin1.analysis"].params_hash() != before
    assert GROUPS["allin1.structure"].params_hash() != before_structure


def test_the_allin1_group_version_moved_with_its_payload():
    """readme §7: change what a group returns -> bump its version. The summary
    gained beat_conf / downbeat_conf_beats / segment_label_probs, none of which
    can be derived from a v1 payload (the 100-fps arrays are long gone), so a
    v1 cache hit must not be reused. 27 such rows exist on the live library."""
    from analysis.registry import GROUPS

    assert GROUPS["allin1.analysis"].version == 2


# ── The activations, kept at beat resolution (Task 6) ─────────────────────────

def test_summary_keeps_the_beat_level_activations():
    from analysis.allin1_runner import (HARMONIX_LABELS, from_cache, summarise,
                                        to_cache)
    s = summarise(FakeResult(), content_hash="h", demixed_by="pipeline", ms=1.0)
    assert len(s["beat_conf"]) == len(s["beats"])
    assert len(s["downbeat_conf_beats"]) == len(s["beats"])
    assert len(s["segment_label_probs"]) == len(s["segments"])
    assert s["segment_labels_order"] == list(HARMONIX_LABELS)

    # Two different curves, not one copied twice: the beat activation is high
    # on every beat, the downbeat activation only on a bar line.
    assert s["beat_conf"] != s["downbeat_conf_beats"]
    assert s["beat_conf"][0] == pytest.approx(0.8, abs=1e-3)    # beat 0 = bar 1
    assert s["beat_conf"][1] == pytest.approx(0.5, abs=1e-3)    # an off-bar beat
    assert s["downbeat_conf_beats"][0] == pytest.approx(0.9, abs=1e-3)
    assert s["downbeat_conf_beats"][1] == pytest.approx(0.0, abs=1e-3)

    back = from_cache(to_cache(s))
    assert back["beat_conf"] == s["beat_conf"]
    assert back["segment_label_probs"] == s["segment_label_probs"]


def test_the_beat_curves_use_the_same_peak_window_as_the_downbeats():
    """``downbeat_conf`` is sampled with _peak_near (±2 frames, because a
    detected event sits on a peak whose time is quantised). The per-beat curve
    stored beside it must use the same window or the two cannot be compared."""
    from analysis.allin1_runner import summarise
    s = summarise(FakeResult())
    at_downbeats = [s["downbeat_conf_beats"][s["beats"].index(t)]
                    for t in s["downbeats"]]
    assert at_downbeats == s["downbeat_conf"]


def test_a_quantised_event_is_read_from_its_peak_not_its_frame():
    """Every stored activation is sampled with _peak_near (±2 frames), because
    a detected event sits on a peak whose TIME is quantised and post-processed.
    A raw nearest-frame read would report 0.0 for exactly the events that
    matter. FakeResult's spikes land on their own beat's frame, so only an
    off-grid result tells the two conventions apart — without this test the
    per-beat curves could be read either way and nothing would notice."""
    from analysis.allin1_runner import FPS, summarise
    secs, beat = 8.0, 0.5
    beats = [round(i * beat, 2) for i in range(int(secs / beat))]
    downbeats = beats[::4]
    frames = int(secs * FPS)
    db = np.zeros(frames, dtype=np.float32)
    bt = np.zeros(frames, dtype=np.float32)
    for t in beats:
        bt[int(t * FPS) + 2] = 0.7            # the peak is two frames late
    for t in downbeats:
        db[int(t * FPS) + 2] = 0.95
    s = summarise({"bpm": 120.0, "beats": beats, "downbeats": downbeats,
                   "beat_positions": [(i % 4) + 1 for i in range(len(beats))],
                   "segments": [],
                   "activations": {"beat": bt, "downbeat": db},
                   "embeddings": None})
    assert s["beat_conf"][0] == pytest.approx(0.7, abs=1e-3)
    assert s["beat_conf"][1] == pytest.approx(0.7, abs=1e-3)
    assert s["downbeat_conf_beats"][0] == pytest.approx(0.95, abs=1e-3)
    assert s["downbeat_conf"][0] == pytest.approx(0.95, abs=1e-3)
    # An off-bar beat still reads ~0 on the downbeat curve: ±2 frames is a
    # quantisation window, not a search.
    assert s["downbeat_conf_beats"][1] == pytest.approx(0.0, abs=1e-3)


def test_segment_label_probs_are_the_whole_distribution():
    """"chorus 0.41 / verse 0.39" and "chorus 0.95" are different evidence."""
    from analysis.allin1_runner import HARMONIX_LABELS, summarise
    s = summarise(FakeResult())
    row = s["segment_label_probs"][2]                 # 16-32 s, labelled chorus
    assert len(row) == len(HARMONIX_LABELS)
    # FakeResult gives label row i the constant (i + 1) / 20.
    assert row == pytest.approx([(i + 1) / 20 for i in range(10)], abs=1e-4)
    # The winner's own probability still agrees with the distribution.
    assert s["section_conf"][2]["label_prob"] == pytest.approx(
        row[HARMONIX_LABELS.index("chorus")], abs=1e-4)


def test_activations_survive_the_database(env):
    models, _stages, _tmp, _calls = env
    from analysis.allin1_runner import summarise
    from database import alignment
    song = models.upsert_song(title="T", artist="A", source_url="http://x/3")
    s = summarise(FakeResult(), content_hash="h", demixed_by="pipeline", ms=1.0)
    alignment.store_allin1(song, "unified_4stem", s)
    got = alignment.get_allin1(song, "unified_4stem")
    assert len(got["beat_conf"]) == len(s["beats"])
    assert len(got["downbeat_conf_beats"]) == len(s["beats"])
    assert got["segment_label_probs"] == s["segment_label_probs"]
    assert got["segment_labels_order"] == s["segment_labels_order"]
    assert got["model"] == "harmonix-fold0"


def test_the_beat_curves_are_counted_in_the_storage_budget(env):
    """storage_bytes is what the < 100 KB ceiling is checked against, so a
    column it does not sum is a column nobody is watching. Measured against the
    SAME summary with the new keys emptied — a sum that ignores the columns
    reports the two rows as the same size."""
    models, _stages, _tmp, _calls = env
    from analysis.allin1_runner import summarise
    from database import alignment
    song = models.upsert_song(title="T", artist="A", source_url="http://x/4")
    lean_song = models.upsert_song(title="T2", artist="A", source_url="http://x/5")
    s = summarise(FakeResult(secs=240.0))
    lean = dict(s)
    for key in ("beat_conf", "downbeat_conf_beats", "segment_label_probs",
                "segment_labels_order"):
        lean[key] = None
    alignment.store_allin1(song, "unified_4stem", s)
    alignment.store_allin1(lean_song, "unified_4stem", lean)

    sizes = alignment.storage_bytes(song)
    lean_sizes = alignment.storage_bytes(lean_song)
    curves = 2 * 2 * len(s["beats"])                  # two float16 arrays
    probs = len(json.dumps(s["segment_label_probs"]))
    assert sizes["analysis"] - lean_sizes["analysis"] >= curves + probs
    assert sum(sizes.values()) < 100_000, sizes


# ── The second Demucs run is never silent (Task 6) ────────────────────────────

def test_self_demix_is_logged_not_silent(monkeypatch, caplog):
    """125 s a track of duplicate Demucs should never be invisible."""
    import logging

    from analysis import allin1_runner as runner

    monkeypatch.setattr(runner, "SELF_DEMIX_WARNED", False)
    with caplog.at_level(logging.WARNING):
        runner.warn_self_demix("/tmp/x.mp3")
        assert any("separat" in r.message for r in caplog.records)
        caplog.clear()
        runner.warn_self_demix("/tmp/y.mp3")          # once per process
        assert not caplog.records


def test_the_demix_hand_over_decides_and_says_which(tmp_path, monkeypatch, caplog):
    """The one place that knows whether allin1 will separate again."""
    import logging

    from analysis import allin1_runner as runner

    stems = {s: _wav(tmp_path / f"{s}.wav", secs=2.0) for s in runner.SOURCES}
    out = tmp_path / "demix" / "htdemucs" / "track"
    timings: dict = {}
    assert runner.hand_over_demix(stems, out, tmp_path / "t.wav", timings) == "pipeline"
    assert sorted(p.name for p in out.glob("*.wav")) == sorted(
        f"{s}.wav" for s in runner.SOURCES)
    assert "demix_write" in timings

    monkeypatch.setattr(runner, "SELF_DEMIX_WARNED", False)
    with caplog.at_level(logging.WARNING):
        assert runner.hand_over_demix(None, out, tmp_path / "t.wav", {}) == "allin1"
        assert caplog.records
    # A partial set is not a hand-over either.
    monkeypatch.setattr(runner, "SELF_DEMIX_WARNED", False)
    assert runner.hand_over_demix({"bass": stems["bass"]}, out,
                                  tmp_path / "t.wav", {}) == "allin1"


def test_run_routes_its_demix_through_the_hand_over():
    """``run`` cannot execute here (no weights), so this pins the wiring: the
    warning buys nothing if run() still decides for itself."""
    import inspect

    from analysis import allin1_runner as runner

    src = inspect.getsource(runner.run)
    assert "hand_over_demix(" in src
    assert "_write_demix(" not in src        # the decision lives in one place
    assert "_model(device, model)" in src    # the chosen checkpoint reaches it


# ── One separation by default (Task 6) ────────────────────────────────────────

def test_there_is_one_pipeline_and_one_engine():
    """The point of the unified pass: allin1's input is the separation playback
    already uses, and there is no other to choose."""
    import config

    assert config.current_separation_pipeline() == "unified_4stem"
    assert config.current_segmentation_engine() == "allin1"


