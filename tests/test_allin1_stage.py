"""The allin1 stage (readme §5.5): summarising, storing and caching its output,
its sections measured like the Essentia cut, phrases on its grid, routing and
failure handling. allin1 itself is faked — the suite runs without torch's
model weights — at the one seam it has: analysis/allin1_runner.run."""
import importlib
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
        self.activations = {"beat": db.copy(), "downbeat": db,
                            "segment": np.full(frames, 0.2, np.float32),
                            "label": np.full((10, frames), 0.1, np.float32)}
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
    alignment.store_allin1(song, "legacy_2stem", summarise(FakeResult(secs=240.0)))
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

def test_next_stage_routes_to_allin1_only_when_owed(env, monkeypatch):
    models, stages, tmp, _calls = env
    import api.workers.pipeline_worker as pw
    from database import alignment
    song = _song(models, tmp)
    assert pw.next_stage(song) is None                  # essentia engine
    monkeypatch.setenv("MASHUP_SEGMENTATION_ENGINE", "allin1")
    assert pw.next_stage(song) == "allin1"
    alignment.set_allin1_state(song, "done")
    assert pw.next_stage(song) is None
    monkeypatch.setenv("MASHUP_SEGMENTATION_ENGINE", "essentia")
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
    monkeypatch.setenv("MASHUP_SEGMENTATION_ENGINE", "allin1")
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
