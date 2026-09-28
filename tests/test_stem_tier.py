"""Phase 4 of the analysis overhaul: the stem tier.

Re-cut sections carry the user's judgements with them (or flag them, never
drop them); sections measured on the stems carry vocal activity, per-stem band
occupancy and the sung range; selecting a track moves it and its likeliest
partners up the queue."""
import importlib
import sys
from pathlib import Path

import numpy as np
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
    import api.workers.pipeline_worker
    import api.queue_runner
    for mod in (config, models, api.jobs, api.workers.stages,
                api.workers.pipeline_worker, api.queue_runner):
        importlib.reload(mod)
    models.init_db()
    # Binds get_conn at import: without a reload it reads an earlier test's DB.
    import api.routes.tracks
    importlib.reload(api.routes.tracks)
    from analysis import decode
    decode.clear()
    return models, tmp_path


def _secs(*spans):
    return [{"start_sec": a, "end_sec": b, "label": "verse", "energy": 0.5,
             "vocal_presence": 0.5, "repetition": 1, "confidence": 0.5}
            for a, b in spans]


def _songs(models, n=2):
    return [models.upsert_song(title=f"T{i}", artist="A", source_url=f"http://x/{i}")
            for i in range(n)]


def _feedback(models):
    conn = models.get_conn()
    rows = conn.execute(
        "SELECT vocal_song_id, inst_song_id, vocal_section, inst_section, verdict, "
        "COALESCE(sections_stale,0) AS stale FROM pair_feedback ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Section-index remap (pair_feedback is irreplaceable) ─────────────────────

def test_index_map_is_by_largest_overlap_of_the_old_section():
    from database.models import section_index_map
    old = [{"section_index": 0, "start_sec": 0, "end_sec": 30},
           {"section_index": 1, "start_sec": 30, "end_sec": 60},
           {"section_index": 2, "start_sec": 60, "end_sec": 90}]
    new = [{"section_index": 0, "start_sec": 0, "end_sec": 10},
           {"section_index": 1, "start_sec": 10, "end_sec": 38},
           {"section_index": 2, "start_sec": 38, "end_sec": 90}]
    m = section_index_map(old, new)
    assert m[0] == 1          # 20 of 30 s now live in new section 1
    assert m[1] == 2          # 22 of 30 s in new section 2
    assert m[2] == 2
    # Split down the middle: no new section holds half of it → unmappable.
    assert section_index_map([{"section_index": 0, "start_sec": 0, "end_sec": 30}],
                             [{"section_index": 0, "start_sec": 0, "end_sec": 14},
                              {"section_index": 1, "start_sec": 14, "end_sec": 30}],
                             min_overlap=0.6)[0] is None


def test_an_identical_recut_touches_nothing(env):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 30), (30, 60)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=1, inst_section=None)
    models.replace_sections(v, _secs((0, 30), (30, 60)))
    assert _feedback(models) == [{"vocal_song_id": v, "inst_song_id": b,
                                  "vocal_section": 1, "inst_section": None,
                                  "verdict": "love", "stale": 0}]


def test_a_recut_carries_judgements_to_the_section_that_holds_the_music(env):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 30), (30, 60), (60, 90)))
    models.replace_sections(b, _secs((0, 40), (40, 80)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=1, inst_section=1)
    models.upsert_pair_feedback(v, b, "no", vocal_section=2, inst_section=0)
    # An intro appears: every section of the vocal track shifts up by one.
    models.replace_sections(v, _secs((0, 8), (8, 30), (30, 60), (60, 90)))
    got = {(r["vocal_section"], r["inst_section"]): r for r in _feedback(models)}
    assert set(got) == {(2, 1), (3, 0)}
    assert all(r["stale"] == 0 for r in got.values())
    # The bed side of the same rows is remapped when the BED is re-cut.
    models.replace_sections(b, _secs((0, 10), (10, 40), (40, 80)))
    got = {(r["vocal_section"], r["inst_section"]) for r in _feedback(models)}
    assert got == {(2, 2), (3, 1)}


def test_sections_that_swap_places_do_not_collide_on_the_unique_key(env):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 30), (30, 60), (60, 90)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=1)
    models.upsert_pair_feedback(v, b, "no", vocal_section=2)
    # 0-30 merges away and a boundary appears at 75: the music of old 1 is new
    # 0, old 2 is new 1 — each row moves onto the index the other just left.
    models.replace_sections(v, _secs((0, 60), (60, 75), (75, 90)))
    rows = {r["verdict"]: r for r in _feedback(models)}
    assert rows["love"]["vocal_section"] == 0 and rows["love"]["stale"] == 0
    assert rows["no"]["stale"] == 1          # 60-90 split exactly in half: unmappable


def test_an_unmappable_judgement_is_kept_and_flagged_never_dropped(env):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 60), (60, 120)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=0, rating=5)
    models.replace_sections(v, _secs((0, 30), (30, 90), (90, 120)))
    [row] = _feedback(models)
    assert row["vocal_section"] == 0 and row["stale"] == 1 and row["verdict"] == "love"
    # A later re-cut does not remap a stale row through a structure it never named.
    models.replace_sections(v, _secs((0, 60), (60, 120)))
    [row] = _feedback(models)
    assert row["vocal_section"] == 0 and row["stale"] == 1


def test_two_judgements_collapsing_into_one_section_are_flagged_not_merged(env):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 30), (30, 60), (60, 90)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=0)
    models.upsert_pair_feedback(v, b, "no", vocal_section=1)
    models.replace_sections(v, _secs((0, 60), (60, 90)))       # 0 and 1 merge
    rows = _feedback(models)
    assert len(rows) == 2
    assert all(r["stale"] == 1 for r in rows)
    assert sorted(r["vocal_section"] for r in rows) == [0, 1]    # untouched


def test_a_failed_remap_rolls_the_new_sections_back(env, monkeypatch):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 30), (30, 60)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=1)

    def _boom(*_a, **_k):
        raise RuntimeError("pair_feedback remap changed the row count 1->0")
    monkeypatch.setattr(models, "remap_feedback_sections", _boom)
    with pytest.raises(RuntimeError):
        models.replace_sections(v, _secs((0, 10), (10, 60)))
    assert [(s["start_sec"], s["end_sec"]) for s in models.get_sections(v)] == \
        [(0.0, 30.0), (30.0, 60.0)]
    assert _feedback(models)[0]["vocal_section"] == 1


def test_the_dataset_ignores_the_indexes_of_a_stale_judgement(env):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 60), (60, 120)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=0, inst_section=0)
    models.replace_sections(v, _secs((0, 30), (30, 90), (90, 120)))
    from matcher.features import _feedback_pairs
    conn = models.get_conn()
    pos, _neg = _feedback_pairs(conn)
    conn.close()
    assert pos == [(v, b, None, None)]


# ── Vocal measures ────────────────────────────────────────────────────────────

def test_vocal_activity_is_the_share_of_sung_frames():
    from analysis.vocals import active_frames, activity_curve, span_fraction
    sr, hop = 22050, 512
    fps = sr / hop
    rms = np.zeros(int(20 * fps))
    rms[int(5 * fps):int(10 * fps)] = 1.0          # sung 5-10 s
    rms[int(12 * fps):int(13 * fps)] = 0.05         # separation residue
    act = active_frames(rms)
    assert span_fraction(act, sr, hop, 5, 10) == pytest.approx(1.0, abs=0.02)
    assert span_fraction(act, sr, hop, 0, 10) == pytest.approx(0.5, abs=0.02)
    assert span_fraction(act, sr, hop, 11, 14) == 0.0
    assert span_fraction(act, sr, hop, 30, 40) is None
    curve = activity_curve(rms, sr, hop, bin_secs=1.0)
    assert len(curve) in (20, 21) and curve[7] == 1.0 and curve[2] == 0.0
    assert not active_frames(np.zeros(100)).any()


def test_f0_summary_reports_the_sung_range_and_refuses_a_handful_of_points():
    from analysis.vocals import f0_summary
    # A4 (MIDI 69) for a second, then E5 (76), 50 ms steps, silence around.
    f0 = [0.0] * 20 + [440.0] * 20 + [659.26] * 20 + [0.0] * 20
    s = f0_summary(f0, 0.05, 0.0, 4.0)
    assert s["p10_midi"] == pytest.approx(69, abs=0.1)
    assert s["p90_midi"] == pytest.approx(76, abs=0.1)
    assert s["range_st"] == pytest.approx(7, abs=0.2)
    assert s["voiced"] == pytest.approx(0.5)
    assert f0_summary(f0, 0.05, 0.0, 1.0) is None            # all unvoiced
    assert f0_summary([440.0] * 3, 0.05, 0.0, 1.0) is None   # too few points


def test_f0_curve_downsampling_keeps_held_notes_and_drops_stray_frames():
    from analysis.essentia_groups import downsample_f0
    hop = 0.005
    pitch = np.array([220.0] * 10 + [0.0] * 8 + [330.0] * 2)   # two 50 ms bins
    assert downsample_f0(pitch, hop, 0.05) == [220.0, 0.0]


# ── Sections measured on the stems ────────────────────────────────────────────

def _tone_track(path: Path, secs: float, sr: int, parts) -> Path:
    """Kick grid at 120 BPM plus, per (start, end, hz, amp), a sine."""
    import soundfile as sf
    n = int(secs * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)
    for k, bt in enumerate(np.arange(0, secs, 0.5)):
        s = int(bt * sr)
        e = min(n, s + int(0.08 * sr))
        tt = np.arange(e - s) / sr
        y[s:e] += (1.0 if k % 4 == 0 else 0.5) * np.sin(2 * np.pi * 70 * tt) * np.exp(-tt * 30)
    for a, b, hz, amp in parts:
        m = (t >= a) & (t < b)
        y[m] += amp * np.sin(2 * np.pi * hz * t[m])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y / max(1e-9, float(np.abs(y).max())), sr)
    return path


def _vocal(path: Path, secs: float, sr: int, spans) -> Path:
    import soundfile as sf
    n = int(secs * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)
    for a, b in spans:
        m = (t >= a) & (t < b)
        y[m] = 0.5 * np.sin(2 * np.pi * 440 * t[m])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)
    return path


def test_sections_carry_vocal_activity_stem_bands_and_the_sung_range(tmp_path):
    from analysis import decode
    from analysis.structure import detect_sections
    decode.clear()
    sr, secs = 22050, 96.0
    mix = _tone_track(tmp_path / "mix.wav", secs, sr,
                      [(0, 48, 220, 0.3), (48, 96, 330, 0.3)])
    voc = _vocal(tmp_path / "voc.wav", secs, sr, [(48, 96)])
    bed = _tone_track(tmp_path / "bed.wav", secs, sr, [(0, 96, 110, 0.3)])
    step = 0.05
    f0 = [440.0 if 48 <= i * step < 96 else 0.0 for i in range(int(secs / step))]

    segs = detect_sections(mix, voc, inst_path=bed,
                           melody={"step": step, "f0": f0})
    assert segs
    for s in segs:
        assert s["vocal_activity"] is not None
        assert sum(s["band_energy_bed"]) == pytest.approx(1.0, abs=1e-3)
    first, last = segs[0], segs[-1]
    assert first["vocal_activity"] < 0.2 and last["vocal_activity"] > 0.8
    assert first.get("f0") is None                  # nothing sung there
    assert last["f0"]["median_midi"] == pytest.approx(69, abs=0.2)
    # The 440 Hz vocal sits in the 400-1000 Hz band.
    assert int(np.argmax(last["band_energy_vocal"])) == 3
    # No vocal energy at all in the first section → unmeasured, not zeros.
    assert first["band_energy_vocal"] is None or \
        sum(first["band_energy_vocal"]) == pytest.approx(1.0, abs=1e-3)

    # Without stems: none of the stem measures, and nothing pretends otherwise.
    decode.clear()
    bare = detect_sections(mix)
    assert all(s.get("vocal_activity") is None and s["band_energy_vocal"] is None
               and s["band_energy_bed"] is None for s in bare)


def test_structure_reads_the_vocal_melody_from_the_cache(env, monkeypatch):
    models, tmp = env
    import api.workers.stages as stages
    import api.workers.hook_worker as hook_worker
    from analysis.cache import content_hash, store
    from analysis.registry import GROUPS
    monkeypatch.setattr(hook_worker, "warm_hooks", lambda *a, **k: {})
    sr, secs = 22050, 60.0
    mix = _tone_track(tmp / "mix.wav", secs, sr, [(0, 60, 220, 0.3)])
    voc = _vocal(tmp / "voc.wav", secs, sr, [(0, 60)])
    bed = _tone_track(tmp / "bed.wav", secs, sr, [(0, 60, 110, 0.3)])
    [song] = _songs(models, 1)
    models.update_song_status(song, "stemmed", raw_path=str(mix))
    for kind, p in (("full", mix), ("vocals", voc), ("instrumental", bed)):
        models.upsert_stem(song, kind, str(p))
    store(GROUPS["essentia.melody"], content_hash(voc),
          {"step": 0.05, "f0": [440.0] * int(secs / 0.05), "summary": None})

    monkeypatch.setattr(stages, "effective_analyzer", lambda: ("librosa", "librosa"))
    stages.do_structure(song)
    assert all(s.get("f0") is None for s in models.get_sections(song))

    monkeypatch.setattr(stages, "effective_analyzer", lambda: ("shadow", "librosa"))
    stages.do_structure(song)                 # a new cache key: re-cut, with f0
    sections = models.get_sections(song)
    assert sections and all(s["f0"]["median_midi"] == pytest.approx(69, abs=0.1)
                            for s in sections)
    assert all(s["vocal_activity"] is not None for s in sections)
    assert all(len(s["band_energy_bed"]) == 8 for s in sections)


def test_the_melody_group_measures_a_sung_note():
    from analysis.essentia_groups import available
    if not available():
        pytest.skip("essentia not installed")
    from analysis.essentia_groups import Signals, group_melody
    sr = 44100
    t = np.arange(int(12 * sr)) / sr
    y = np.where(t > 2, 0.4 * np.sin(2 * np.pi * 440 * t), 0.0).astype(np.float32)
    out = group_melody(Signals(np.stack([y, y], axis=1)))
    assert out["step"] == 0.05
    assert out["summary"]["median_midi"] == pytest.approx(69, abs=0.5)
    assert out["summary"]["voiced"] > 0.6


# ── Partner prefetch ──────────────────────────────────────────────────────────

def test_prioritise_moves_a_waiting_job_up_and_never_down(env):
    import api.queue_runner as q
    import config
    models, _ = env
    a, b, c = _songs(models, 3)
    ja, jb, jc = (q.enqueue_song(s) for s in (a, b, c))
    assert q.snapshot()["download"]["waiting"] == [ja, jb, jc]

    assert q.prioritise(c, config.PRIORITY_PREFETCH)["action"] == "raised"
    assert q.snapshot()["download"]["waiting"] == [jc, ja, jb]
    assert q.take("download", block=False)[0] == jc

    ju = q.enqueue_song(models.upsert_song(title="U", artist="A", source_url="http://x/u"),
                        priority=config.PRIORITY_USER)
    assert q.prioritise(models.get_song_by_url("http://x/u")["id"],
                        config.PRIORITY_PREFETCH)["action"] == "kept"
    assert q.snapshot()["download"]["waiting"][0] == ju


def test_prioritise_enqueues_an_idle_unfinished_track_and_leaves_the_rest(env):
    import api.queue_runner as q
    import config
    models, _ = env
    fresh, done, broken = _songs(models, 3)
    models.update_song_status(done, "analysed")
    models.update_song_error(broken, "error_stems", "boom")
    got = q.prioritise(fresh, config.PRIORITY_PREFETCH)
    assert got["action"] == "enqueued" and got["job_id"]
    assert q.prioritise(done, config.PRIORITY_PREFETCH)["action"] == "none"
    assert q.prioritise(broken, config.PRIORITY_PREFETCH)["action"] == "none"


def test_prefetch_ranks_partners_by_tempo_and_key(env):
    models, _ = env
    from api.routes.tracks import prefetch_partners
    me, close, far, off_key, done, unmeasured = _songs(models, 6)
    for sid, bpm, cam in ((me, 124.0, "8A"), (close, 125.0, "8A"), (far, 90.0, "8A"),
                          (off_key, 124.0, "2B"), (done, 124.0, "8A")):
        models.upsert_features(sid, "full", {"bpm": bpm, "camelot": cam})
    models.update_song_status(done, "analysed")
    for sid in (me, close, far, off_key, unmeasured):
        models.update_song_status(sid, "downloaded")
    assert prefetch_partners(me, 5) == [close, off_key, far, unmeasured]
    assert prefetch_partners(me, 1) == [close]


def test_prefetch_route_bumps_the_track_and_its_partners(env, monkeypatch):
    models, _ = env
    import api.queue_runner as q
    import api.routes.tracks as tracks
    import config
    monkeypatch.setattr(config, "PREFETCH_PARTNERS", 1)
    me, partner, other = _songs(models, 3)
    for sid, bpm in ((me, 124.0), (partner, 124.0), (other, 80.0)):
        models.upsert_features(sid, "full", {"bpm": bpm, "camelot": "8A"})
    jobs = [q.enqueue_song(s) for s in (other, partner, me)]
    out = tracks.prefetch(me)
    assert [t["song_id"] for t in out["tracks"]] == [me, partner]
    assert all(t["action"] == "raised" for t in out["tracks"])
    waiting = q.snapshot()["download"]["waiting"]
    assert waiting[-1] == jobs[0]                       # the unrelated track waits
    with pytest.raises(Exception):
        tracks.prefetch(9999)


def test_new_audio_for_a_track_flags_the_sections_its_judgements_named(env):
    models, _ = env
    v, b = _songs(models)
    models.replace_sections(v, _secs((0, 30), (30, 60)))
    models.upsert_pair_feedback(v, b, "love", vocal_section=1)
    models.upsert_pair_feedback(b, v, "ok")                   # no sections named
    models.update_song_url(v, "http://x/other")
    rows = {r["verdict"]: r for r in _feedback(models)}
    assert rows["love"]["stale"] == 1 and rows["love"]["vocal_section"] == 1
    assert rows["ok"]["stale"] == 0
