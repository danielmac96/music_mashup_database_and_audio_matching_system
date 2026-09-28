"""Phase 3 of the analysis overhaul: the quick tier and the priority queues.

A track's mix is analysed and cut into provisional sections straight after
download, before Demucs; the full analysis re-cuts the sections once stems
exist; a button pressed on one track goes ahead of an import, and an import
ahead of a bulk backfill."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _wav(path: Path, secs: float = 50.0, bpm: float = 120.0, sr: int = 22050,
         seed_hz: float = 220.0) -> Path:
    import soundfile as sf
    n = int(secs * sr)
    y = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(np.arange(0, secs, 60.0 / bpm)):
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
    return models, api.workers.stages, api.workers.pipeline_worker, api.queue_runner, tmp_path


def _downloaded(models, tmp, name="mix.wav"):
    mix = _wav(tmp / name)
    song = models.upsert_song(title="T", artist="A", source_url=f"http://x/{name}")
    models.update_song_status(song, "downloaded", raw_path=str(mix))
    return song


# ── The quick tier ────────────────────────────────────────────────────────────

def test_quick_gives_a_downloaded_track_bpm_key_and_provisional_sections(env):
    models, stages, _pw, _q, tmp = env
    song = _downloaded(models, tmp)
    stages.do_quick(song)

    row = models.get_song(song)
    assert row["status"] == "downloaded"          # not status-bearing
    assert row["quick_state"] == "done" and row["quick_at"]
    feats = models.get_features_for_song(song, "full")
    assert feats["bpm"] == pytest.approx(120.0, rel=0.05) and feats["key"]
    sections = models.get_sections(song)
    assert sections
    assert all(s["provisional"] == 1 for s in sections)
    assert all(s["section_class"] == "unknown" for s in sections)   # no vocal stem


def test_provisional_sections_are_stale_once_a_vocal_stem_exists(env):
    from api.workers.bulk_worker import sections_are_current
    models, stages, _pw, _q, tmp = env
    song = _downloaded(models, tmp)
    stages.do_quick(song)
    assert sections_are_current(song)             # nothing better to cut them with

    models.upsert_stem(song, "full", str(tmp / "mix.wav"))
    models.upsert_stem(song, "vocals", str(_wav(tmp / "voc.wav", seed_hz=330.0)))
    models.upsert_stem(song, "instrumental", str(_wav(tmp / "bed.wav", seed_hz=110.0)))
    assert not sections_are_current(song)         # the stems are here now

    stages.do_structure(song)
    sections = models.get_sections(song)
    assert sections and all(s["provisional"] == 0 for s in sections)
    assert any(s["vocal_presence"] is not None for s in sections)
    assert sections_are_current(song)


def test_the_full_analysis_reuses_the_quick_tiers_mix_analysis(env):
    models, stages, _pw, _q, tmp = env
    song = _downloaded(models, tmp)
    stages.do_quick(song)
    models.upsert_stem(song, "full", str(tmp / "mix.wav"))
    models.update_song_status(song, "stemmed")
    conn = models.get_conn()
    conn.execute("DELETE FROM analysis_runs")
    conn.commit()
    conn.close()
    stages.do_analyze(song)
    conn = models.get_conn()
    grps = [r["grp"] for r in conn.execute(
        "SELECT grp FROM analysis_runs WHERE stem_type='full'")]
    conn.close()
    assert "analysis.tempo.cached" in grps and "analysis.tempo" not in grps


def test_a_new_download_runs_the_quick_tier_again(env, monkeypatch):
    models, stages, pw, _q, tmp = env
    song = _downloaded(models, tmp)
    models.set_quick_state(song, "done")
    assert pw.next_stage(song) == "stems"

    import downloader.download as dl

    class _R:
        path = tmp / "mix.wav"
        duration_secs = 50.0
        source_url = None
        provenance = None
    monkeypatch.setattr(dl, "download_track", lambda **_kw: _R())
    stages.do_download(song)
    assert models.get_song(song)["quick_state"] is None
    assert pw.next_stage(song) == "quick"


def test_a_failed_quick_tier_never_stops_the_track(env, monkeypatch):
    models, stages, pw, _q, tmp = env
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "downloaded", raw_path=str(tmp / "missing.mp3"))
    import api.jobs as jobs
    jid = jobs.new_job(kind="pipeline", song_id=song)
    assert pw.next_stage(song) == "quick"
    assert pw.run_stage(jid, song, "quick") == "next"
    assert jobs.get(jid)["stages"]["quick"]["state"] == "failed"
    assert jobs.get(jid)["status"] != "failed"
    assert models.get_song(song)["quick_state"] == "failed"
    assert pw.next_stage(song) == "stems"             # onwards, not round again

    # Even a crash that escapes do_quick is contained.
    song2 = _downloaded(models, tmp, "b.wav")
    monkeypatch.setattr(stages, "do_quick", lambda *a, **k: 1 / 0)
    jid2 = jobs.new_job(kind="pipeline", song_id=song2)
    assert pw.run_stage(jid2, song2, "quick") == "next"
    assert pw.next_stage(song2) == "stems"


# ── Priority queues ───────────────────────────────────────────────────────────

def test_a_pressed_button_goes_ahead_of_an_import_ahead_of_a_backfill(env):
    import config
    models, _stages, _pw, q, _tmp = env
    ids = [models.upsert_song(title=f"T{n}", artist="A", source_url=f"http://x/{n}")
           for n in range(5)]
    bulk = q.enqueue_song(ids[0], priority=config.PRIORITY_BACKFILL)
    imp1 = q.enqueue_song(ids[1])
    imp2 = q.enqueue_song(ids[2], priority=config.PRIORITY_INGEST)
    user = q.enqueue_song(ids[3], priority=config.PRIORITY_USER)
    bulk2 = q.enqueue_song(ids[4], priority=config.PRIORITY_BACKFILL)

    assert q.snapshot()["download"]["waiting"] == [user, imp1, imp2, bulk, bulk2]
    assert [q.take("download", block=False)[0] for _ in range(5)] == \
        [user, imp1, imp2, bulk, bulk2]


def test_a_job_keeps_its_priority_from_stage_to_stage(env):
    import api.jobs as jobs
    import config
    models, _stages, _pw, q, tmp = env
    a = _downloaded(models, tmp, "a.wav")
    b = _downloaded(models, tmp, "b.wav")
    for s in (a, b):
        models.set_quick_state(s, "done")
    ja = q.enqueue_song(a)
    jb = q.enqueue_song(b, priority=config.PRIORITY_USER)
    assert jobs.get(jb)["priority"] == config.PRIORITY_USER
    assert q.snapshot()["stems"]["waiting"] == [jb, ja]


# ── Demucs leaves room for the quick tier ────────────────────────────────────

def test_the_separator_is_capped_below_every_core(monkeypatch):
    import config
    import stems.separate as sep
    monkeypatch.setattr(config, "DEMUCS_THREADS", 3)
    env = sep.thread_env()
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "3"


def test_subprocess_env_is_merged_not_replaced(tmp_path):
    import os
    import sys as _sys
    from api.workers._progress import stream_subprocess
    lines = []
    out = stream_subprocess(
        [_sys.executable, "-c", "import os;print(os.environ.get('X_T'), bool(os.environ.get('PATH')))"],
        lines.append, env={"X_T": "7"})
    assert out.returncode == 0 and "7 True" in out.stdout
