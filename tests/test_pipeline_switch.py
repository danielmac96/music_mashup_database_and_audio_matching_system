"""Two separation pipelines side by side (readme §2 "Rollback", §7): a unified
separation parks the legacy rows instead of overwriting them, switching back is
a row swap with no separation or analysis, judgements survive it, and the
schema migration has a working down-migration."""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    import database.models as models
    import api.jobs
    import api.workers.stages
    import api.queue_runner
    for mod in (config, models, api.jobs, api.workers.stages, api.queue_runner):
        importlib.reload(mod)
    models.init_db()
    import stems.separate as sep
    importlib.reload(sep)
    calls = {"separate": 0}

    def fake_separate(song_id, title, audio_path, artist="", on_progress=None,
                      separator=None, force=False, mode=None, unified=False):
        calls["separate"] += 1
        dirs = sep.unified_dirs() if unified else {
            "vocals": config.VOCALS_DIR, "instrumental": config.INSTRUMENTALS_DIR}
        kinds = ["vocals", "instrumental"] + (["drums", "bass", "other"] if unified else [])
        out = {}
        for k in kinds:
            d = dirs[k]
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{song_id}_{k}.flac"
            p.write_bytes(b"unified" if unified else b"legacy")
            out[k] = p
        out["separator"] = sep.unified_tag() if unified else sep.separator_tag("demucs", "two")
        return out
    monkeypatch.setattr(sep, "separate", fake_separate)
    return models, api.workers.stages, tmp_path, calls, monkeypatch


def _analysed_legacy(models, tmp):
    raw = tmp / "mix.mp3"
    raw.write_bytes(b"mix")
    song = models.upsert_song(title="T", artist="A", source_url="http://x/1")
    models.update_song_status(song, "downloaded", raw_path=str(raw))
    return song, raw


def _fake_analysis(models, song, tag):
    for st in ("full", "vocals", "instrumental"):
        models.upsert_features(song, st, {"bpm": 120.0, "key": "A", "mode": "minor",
                                          "camelot": "8A", "loudness_rms": 0.1,
                                          "energy": 0.5 if tag == "legacy" else 0.6})
    models.replace_sections(song, [
        {"start_sec": 0.0, "end_sec": 16.0, "label": "verse",
         "vocal_presence": 0.8 if tag == "legacy" else 0.7},
        {"start_sec": 16.0, "end_sec": 32.0, "label": "chorus", "vocal_presence": 0.9}])
    models.update_song_status(song, "analysed")


def _snapshot(models, song):
    conn = models.get_conn()
    try:
        return {t: sorted(json.dumps(dict(r), sort_keys=True) for r in conn.execute(
            f"SELECT * FROM {t} WHERE song_id=?", (song,)))
            for t in ("stems", "features", "sections")}
    finally:
        conn.close()


def test_unified_parks_legacy_and_switching_back_needs_no_audio_work(env):
    models, stages, tmp, calls, mp = env
    from database import alignment
    song, _raw = _analysed_legacy(models, tmp)
    stages.do_stems(song)
    _fake_analysis(models, song, "legacy")
    models.upsert_pair_feedback(song, song, rating=5, vocal_section=1, inst_section=0)
    legacy = _snapshot(models, song)
    assert alignment.live_pipeline(song) == "legacy_2stem"

    mp.setenv("MASHUP_SEPARATION_PIPELINE", "unified_4stem")
    stages.do_stems(song)
    assert calls["separate"] == 2
    assert alignment.live_pipeline(song) == "unified_4stem"
    paths = {r["stem_type"]: r["file_path"] for r in models.get_conn().execute(
        "SELECT stem_type, file_path FROM stems WHERE song_id=?", (song,))}
    assert "unified" in paths["vocals"] and "drums" in paths
    assert alignment.has_parked(song, "legacy_2stem")
    _fake_analysis(models, song, "unified")
    unified = _snapshot(models, song)

    # Back to legacy: a swap, not a separation; the legacy rows come back as
    # they were, ids included.
    assert alignment.activate_pipeline(song, "legacy_2stem") == "swapped"
    assert calls["separate"] == 2
    assert _snapshot(models, song) == legacy
    assert len(models.get_pair_feedback()) == 1
    # And forward again.
    assert alignment.activate_pipeline(song, "unified_4stem") == "swapped"
    assert _snapshot(models, song) == unified
    assert alignment.activate_pipeline(song, "unified_4stem") == "already"


def test_a_pipeline_never_produced_needs_processing(env):
    models, stages, tmp, _calls, _mp = env
    from database import alignment
    song, _raw = _analysed_legacy(models, tmp)
    stages.do_stems(song)
    assert alignment.activate_pipeline(song, "unified_4stem") == "needs_processing"
    assert alignment.live_pipeline(song) == "legacy_2stem"


def test_switch_job_swaps_or_queues(env, monkeypatch):
    models, stages, tmp, _calls, mp = env
    import api.jobs as jobs
    from api.workers import bulk_worker
    from api import queue_runner
    queued = []
    monkeypatch.setattr(queue_runner, "enqueue_song",
                        lambda sid, priority=None: queued.append(sid) or "j")
    a, _ = _analysed_legacy(models, tmp)
    stages.do_stems(a)
    _fake_analysis(models, a, "legacy")
    mp.setenv("MASHUP_SEPARATION_PIPELINE", "unified_4stem")
    assert bulk_worker.stale_song_ids("switch_pipeline") == [a]
    job = jobs.new_job(kind="bulk")
    counts = bulk_worker.switch_pipeline(job, [a])
    assert counts["queued"] == 1 and queued == [a]
    assert models.get_song(a)["status"] == "downloaded"


def test_feedback_namespace_keeps_engines_apart(env):
    """An Essentia re-cut never re-points a judgement of an allin1 section."""
    models, _stages, tmp, _calls, _mp = env
    song, _ = _analysed_legacy(models, tmp)
    _fake_analysis(models, song, "legacy")
    base = models.ALLIN1_SECTION_BASE
    models.upsert_pair_feedback(song, song, rating=4, vocal_section=base + 1,
                                inst_section=base)
    models.replace_sections(song, [{"start_sec": 0.0, "end_sec": 32.0, "label": "verse"}])
    fb = models.get_pair_feedback()[0]
    assert (fb["vocal_section"], fb["inst_section"]) == (base + 1, base)
    assert not fb["sections_stale"]


def test_downgrade_round_trip(env):
    models, stages, tmp, _calls, mp = env
    from database import alignment
    from database.downgrade import DowngradeRefused, downgrade
    song, _ = _analysed_legacy(models, tmp)
    stages.do_stems(song)
    _fake_analysis(models, song, "legacy")
    models.upsert_pair_feedback(song, song, rating=5, vocal_section=1, inst_section=0)
    mp.setenv("MASHUP_SEPARATION_PIPELINE", "unified_4stem")
    stages.do_stems(song)
    db = Path(models.DB_PATH)
    with pytest.raises(DowngradeRefused):
        downgrade("alignment", db)
    assert alignment.activate_pipeline(song, "legacy_2stem") == "swapped"

    counts_before = _counts(db)
    out = downgrade("alignment", db)
    assert out["tables"] == 5 and out["columns"] >= 10
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not tables & {"allin1_analysis", "sections_allin1", "pipeline_parked"}
    assert "separation_pipeline" not in {r[1] for r in conn.execute("PRAGMA table_info(songs)")}
    conn.close()
    assert _counts(db) == counts_before            # nothing older was touched
    # Up again: get_conn re-runs the migration.
    models.get_conn().close()
    conn = sqlite3.connect(db)
    assert "separation_pipeline" in {r[1] for r in conn.execute("PRAGMA table_info(songs)")}
    conn.close()


def _counts(db):
    conn = sqlite3.connect(db)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("songs", "stems", "features", "sections", "pair_feedback")}
    finally:
        conn.close()
