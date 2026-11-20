"""Phase 5 — compute efficiency and import-to-pairs speed (readme §2, §4, §5,
§7): partner-aware stem priority, the one core budget, the allin1 skip, lazy
hook pre-cutting, the storage report and Discover's value preview."""
import importlib
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DATA_DIR", str(tmp_path / "data"))
    import config
    importlib.reload(config)
    config.ensure_dirs()
    from database import models
    importlib.reload(models)
    models.init_db()
    return config, models


def _song(models, title, status="analysed", allin1_state=None):
    sid = models.upsert_song(title, "A", f"https://sc/{title}", 200, "House",
                             status=status)
    if allin1_state:
        conn = models.get_conn()
        conn.execute("UPDATE songs SET allin1_state=? WHERE id=?", (allin1_state, sid))
        conn.commit()
        conn.close()
    return sid


def _row(models, vocal, bed, *, scorer="provisional", mask=1 | 2, shift=0,
         total=0.6, timing_ok=None, slot=None, vsec=None):
    conn = models.get_conn()
    conn.execute(
        "INSERT INTO mashup_candidates (combo_type, vocal_song_id, inst_song_id, "
        "scorer, dropin_mask, camelot_shift, score_total, timing_ok, score_slot, "
        "vocal_section_idx) VALUES ('vocal_over_instrumental',?,?,?,?,?,?,?,?,?)",
        (vocal, bed, scorer, mask, shift, total, timing_ok, slot, vsec))
    conn.commit()
    conn.close()


# ── 5.1 partner-aware stem priority ───────────────────────────────────────────

def test_stem_promise_counts_likely_dropins_with_separated_partners(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.workers import stem_priority
    new = _song(models, "new", status="downloaded")
    lib = [_song(models, f"lib{i}") for i in range(4)]
    other_import = _song(models, "other", status="downloaded")
    _row(models, new, lib[0])                          # likely Drop-in
    _row(models, lib[1], new, total=0.7)               # likely Drop-in, other role
    _row(models, new, lib[2], mask=2)                  # tempo apart: not likely
    _row(models, new, lib[3], mask=1 | 8, shift=0)     # keys only mix a semitone up,
    _row(models, new, other_import)                    # partner not separated yet
    _row(models, lib[0], lib[1], scorer="heuristic")   # not this track's
    p = stem_priority.stem_promise(new)
    assert p["dropin_partners"] == 2
    assert p["partners"] == 4
    assert p["best"] == 0.7


def test_priority_bands_sit_between_prefetch_and_ingest(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.workers.stem_priority import priority_for
    assert priority_for({"dropin_partners": 0}) == config.PRIORITY_INGEST
    bands = [priority_for({"dropin_partners": n}) for n in
             (1, config.STEM_PROMISE_SOME, config.STEM_PROMISE_MANY, 999)]
    assert bands == sorted(bands, reverse=True)
    assert all(config.PRIORITY_PREFETCH < b < config.PRIORITY_INGEST for b in bands)


def test_promotion_only_touches_an_import_and_never_lowers(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api import jobs
    from api.workers import stem_priority
    new = _song(models, "new", status="downloaded")
    lib = _song(models, "lib")
    _row(models, new, lib)
    imp = jobs.new_job(kind="pipeline", song_id=new)
    jobs.update(imp, priority=config.PRIORITY_INGEST)
    assert stem_priority.promote_for_stems(imp, new) == config.PRIORITY_PROMISE_LOW
    assert jobs.get(imp)["priority"] == config.PRIORITY_PROMISE_LOW
    assert jobs.get(imp)["stem_promise"]["dropin_partners"] == 1
    for prio in (config.PRIORITY_USER, config.PRIORITY_PREFETCH, config.PRIORITY_BACKFILL):
        j = jobs.new_job(kind="pipeline", song_id=new)
        jobs.update(j, priority=prio)
        assert stem_priority.promote_for_stems(j, new) is None
        assert jobs.get(j)["priority"] == prio


def test_dispatch_to_stems_puts_the_promising_import_first(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api import jobs, queue_runner
    from api.workers import pipeline_worker
    plain = _song(models, "plain", status="downloaded")
    good = _song(models, "good", status="downloaded")
    lib = _song(models, "lib")
    _row(models, good, lib)
    monkeypatch.setattr(pipeline_worker, "next_stage", lambda sid: "stems")
    while queue_runner.waiting("stems"):
        queue_runner.take("stems", block=False)
    for sid in (plain, good):             # plain arrives first
        j = jobs.new_job(kind="pipeline", song_id=sid)
        jobs.update(j, priority=config.PRIORITY_INGEST)
        queue_runner._dispatch(j, sid)
    first = queue_runner.take("stems", block=False)[1]
    second = queue_runner.take("stems", block=False)[1]
    assert (first, second) == (good, plain)


def test_dropin_sql_is_the_songs_half_plus_slot_and_timing(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from matcher.compat import dropin_songs_sql, dropin_sql
    a, b = _song(models, "a"), _song(models, "b")
    n = 0
    for mask in range(16):
        for shift in (-2, -1, 0, 1, 2):
            for t_ok in (0, 1, None):
                for slot in (0.1, 0.9):
                    n += 1
                    _row(models, a, b, scorer="heuristic", mask=mask, shift=shift,
                         timing_ok=t_ok, slot=slot, vsec=n)
    conn = models.get_conn()
    rows = conn.execute(
        f"SELECT {dropin_sql(0.5, 'mc')}, {dropin_songs_sql('mc')}, timing_ok, "
        f"score_slot FROM mashup_candidates mc", (0.15, 0.15)).fetchall()
    conn.close()
    assert len(rows) == 16 * 5 * 3 * 2
    for full, songs, t_ok, slot in rows:
        assert full == int(bool(songs) and t_ok == 1 and slot >= 0.5)


def test_simulator_separates_the_promoted_track_first():
    sys.path.insert(0, str(ROOT / "scripts"))
    import sim_import
    order = [1, 2, 3]
    durs = {s: {i: 10.0 for i in order} for s in sim_import.STAGES}
    durs["stems"] = {i: 100.0 for i in order}
    score = {k: {i: 1.0 for i in order} for k in ("provisional", "first_cut", "refined")}
    workers = {"download": 4, "quick": 2, "stems": 1, "analysis": 2,
               "structure": 2, "allin1": 1}
    fifo = sim_import.simulate(order, durs, score, {i: 10 for i in order}, workers)[0]
    aware = sim_import.simulate(order, durs, score, {1: 10, 2: 10, 3: 7}, workers)[0]
    assert fifo[3] > fifo[2] and aware[3] < aware[2]
    pairs = {(3, 99)}
    t_fifo = sim_import.first_dropins(pairs, set(), set(order), fifo, {})[0]
    t_aware = sim_import.first_dropins(pairs, set(), set(order), aware, {})[0]
    assert t_aware < t_fifo


# ── 5.2 the core budget ──────────────────────────────────────────────────────

def test_core_budget_never_holds_more_than_its_cores():
    from analysis.threads import CoreBudget
    b = CoreBudget(4)
    seen = []

    def work(n):
        with b.hold(n, f"w{n}"):
            seen.append(b.snapshot()["in_use"])
            time.sleep(0.002)

    ts = [threading.Thread(target=work, args=(n,)) for n in (1, 3, 2, 4, 1, 1, 3) * 15]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert b.peak <= 4 and max(seen) <= 4
    assert b.snapshot()["in_use"] == 0 and b.snapshot()["held"] == {}


def test_core_budget_is_fair_to_a_large_request():
    """A separation waiting for all 4 cores is served before one-core work
    that arrived after it, though that work could have run at once."""
    from analysis.threads import CoreBudget
    b = CoreBudget(4)
    order = []
    gate = threading.Event()

    def small(tag):
        with b.hold(1, "analysis"):
            order.append(tag)
            gate.wait(1)

    def big():
        with b.hold(4, "demucs"):
            order.append("big")

    first = threading.Thread(target=small, args=("s0",))
    first.start()
    while b.snapshot()["in_use"] < 1:
        time.sleep(0.001)
    t_big = threading.Thread(target=big)
    t_big.start()
    while b.snapshot()["waiting"] < 1:
        time.sleep(0.001)
    later = [threading.Thread(target=small, args=(f"s{i}",)) for i in (1, 2)]
    for t in later:
        t.start()
        while b.snapshot()["waiting"] < 1 + later.index(t) + 1:
            time.sleep(0.001)
    gate.set()
    for t in [first, t_big, *later]:
        t.join()
    assert order.index("big") < order.index("s1")


def test_nested_holds_are_free_and_requests_clamp():
    from analysis.threads import CoreBudget
    b = CoreBudget(2)
    with b.hold(99, "demucs") as n:
        assert n == 2
        with b.hold(1, "scoring") as inner:     # would deadlock if not free
            assert inner == 0


def test_stages_take_their_threads_from_the_budget(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from analysis import threads
    from api.workers import stages
    importlib.reload(stages)
    sid = _song(models, "s", status="downloaded")
    raw = tmp_path / "raw.mp3"
    raw.write_bytes(b"x")
    conn = models.get_conn()
    conn.execute("UPDATE songs SET raw_path=? WHERE id=?", (str(raw), sid))
    conn.commit()
    conn.close()
    held = {}

    def fake_separate(**kw):
        held.update(threads.BUDGET.snapshot()["held"])
        return None

    import stems.separate as sep
    monkeypatch.setattr(sep, "separate", fake_separate)
    with pytest.raises(stages.StageError):
        stages.do_stems(sid)
    assert held == {"demucs": threads.BUDGET.clamp(config.DEMUCS_THREADS)}


def test_scoring_holds_one_core(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from analysis import threads
    from matcher import match
    seen = {}
    monkeypatch.setattr(match, "_score_all_pairs",
                        lambda *a, **k: seen.update(threads.BUDGET.snapshot()["held"]) or {})
    match.score_all_pairs()
    assert seen == {"scoring": 1}


def test_budget_report_and_startup_check(monkeypatch):
    import config
    from analysis import threads
    rep = threads.check()
    assert rep["budget"] == rep["cores"] == threads.cores() == rep["max_at_once"]
    assert set(rep["requests"]) >= {"demucs", "allin1", "analysis", "quick",
                                    "structure", "effnet", "scoring"}
    assert rep["warnings"] == [] or all(isinstance(w, str) for w in rep["warnings"])
    monkeypatch.setattr(config, "DEMUCS_THREADS", threads.cores() + 5)
    warn = threads.budget_report()["warnings"]
    assert any("demucs" in w for w in warn)


def test_tensorflow_threads_are_capped_before_its_first_session(monkeypatch):
    from analysis import threads
    monkeypatch.delenv("TF_NUM_INTRAOP_THREADS", raising=False)
    monkeypatch.delenv("TF_NUM_INTEROP_THREADS", raising=False)
    threads.apply_tf_thread_env()
    import os
    assert os.environ["TF_NUM_INTRAOP_THREADS"] == str(threads.worker_thread_cap())
    assert os.environ["TF_NUM_INTEROP_THREADS"] == "1"
    src = (ROOT / "analysis" / "ml_models.py").read_text(encoding="utf-8")
    body = src[src.index("def _predictor"):src.index("def predict(")]
    assert body.index("apply_tf_thread_env()") < body.index("import essentia.standard")


def test_queue_route_reports_threads_and_allin1(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.routes import jobs as jobs_route
    out = jobs_route.queue_snapshot()
    assert out["threads"]["budget"] == out["threads"]["cores"]
    assert "warnings" in out["threads"]
    assert {"skipped_unusable", "skipped_too_long"} <= set(out["allin1"])
    src = (ROOT / "frontend/src/components/QueueScreen.jsx").read_text(encoding="utf-8")
    assert "<ComputeReadout threads={q.threads} allin1={q.allin1} />" in src
    assert "skipped_unusable" in src and "warnings" in src


# ── 5.3 allin1 skip ──────────────────────────────────────────────────────────

def _stems(models, sid, vq, iq):
    for kind, q in (("vocals", vq), ("instrumental", iq)):
        models.upsert_stem(sid, kind, f"/nowhere/{sid}_{kind}.flac")
        if q is not None:
            conn = models.get_conn()
            conn.execute("UPDATE stems SET quality=? WHERE song_id=? AND stem_type=?",
                         (q, sid, kind))
            conn.commit()
            conn.close()


def test_allin1_is_skipped_only_when_both_stems_are_unusable(tmp_path, monkeypatch):
    """No track that could produce a usable section is skipped: a vocal stem
    at or above stem_quality_min can give a vocal section, an instrumental at
    or above it a bed section, and an unmeasured stem is never bad."""
    config, models = _setup(tmp_path, monkeypatch)
    from api.workers import stages
    importlib.reload(stages)
    qmin = config.current_float("stem_quality_min")
    grid = (None, 0.0, qmin - 0.01, qmin, qmin + 0.01, 0.9)
    for vq in grid:
        for iq in grid:
            sid = _song(models, f"t{vq}_{iq}")
            _stems(models, sid, vq, iq)
            usable = (vq is None or vq >= qmin) or (iq is None or iq >= qmin)
            skip = stages.allin1_cannot_help(sid)
            assert (skip is None) == usable, (vq, iq, skip)
    monkeypatch.setenv("MASHUP_STEM_QUALITY_MIN", "0")
    sid = _song(models, "zero")
    _stems(models, sid, 0.01, 0.01)
    assert stages.allin1_cannot_help(sid) is None


def test_do_allin1_records_and_counts_the_skip(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.workers import stages
    importlib.reload(stages)
    from database import alignment
    sid = _song(models, "bad")
    raw = tmp_path / "raw.mp3"
    raw.write_bytes(b"x")
    conn = models.get_conn()
    conn.execute("UPDATE songs SET raw_path=? WHERE id=?", (str(raw), sid))
    conn.commit()
    conn.close()
    _stems(models, sid, 0.05, 0.05)
    out = stages.do_allin1(sid)
    assert out["skipped"].startswith(alignment.ALLIN1_SKIP_UNUSABLE)
    counts = alignment.allin1_state_counts()
    assert counts["skipped"] == 1 and counts["skipped_unusable"] == 1
    assert counts["skipped_too_long"] == 0
    from api.workers import pipeline_worker
    # A skip counts as having run: the track is not routed back to allin1.
    assert pipeline_worker.next_stage(sid) != "allin1"


def test_too_long_skip_keeps_its_reason_prefix():
    src = (ROOT / "api/workers/stages.py").read_text(encoding="utf-8")
    assert "alignment.ALLIN1_SKIP_TOO_LONG" in src
    from database import alignment
    assert alignment.ALLIN1_SKIP_TOO_LONG.startswith("Skipped:")


# ── lazy hook pre-cutting ────────────────────────────────────────────────────

def test_hooks_are_cut_only_for_tracks_in_the_top_pairs(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.workers import hook_worker
    importlib.reload(hook_worker)
    a, b, c, d = (_song(models, x) for x in "abcd")
    _row(models, a, b, scorer="heuristic", total=0.9)
    _row(models, c, d, scorer="heuristic", total=0.2)
    monkeypatch.setenv("MASHUP_HOOK_WARM_TOP_N", "1")
    assert hook_worker.in_top_pairs(a, 1) and hook_worker.in_top_pairs(b, 1)
    assert not hook_worker.in_top_pairs(c, 1)
    assert hook_worker.in_top_pairs(c, 5)     # fewer rows than n: all in
    calls = []
    monkeypatch.setattr(hook_worker, "warm_hooks",
                        lambda sid, force=False: calls.append((sid, force)) or {"rendered": []})
    stale = hook_worker.hook_clip_path(c, "vocals")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old window")
    assert hook_worker.warm_or_defer_hooks(c)["deferred"] is True
    assert not stale.exists() and calls == []
    assert hook_worker.warm_or_defer_hooks(a)["deferred"] is False
    assert calls == [(a, True)]
    monkeypatch.setenv("MASHUP_HOOK_WARM_TOP_N", "0")     # 0: every track
    assert hook_worker.warm_or_defer_hooks(c)["deferred"] is False


# ── storage ──────────────────────────────────────────────────────────────────

def test_storage_report_and_prune(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.workers import stages
    importlib.reload(stages)
    from database import storage
    done = _song(models, "done", allin1_state="done")
    queued = _song(models, "queued", allin1_state="queued")
    sizes = {"vocals": 300, "instrumental": 400, "drums": 100, "bass": 50, "other": 70}
    for sid in (done, queued):
        for kind, n in sizes.items():
            p = config.STEMS_DIR / f"s{sid}" / f"{kind}.flac"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\0" * n)
            models.upsert_stem(sid, kind, str(p))
    models.upsert_stem(done, "full", str(tmp_path / "missing.mp3"))
    rep = storage.storage_report()
    assert rep["kinds"]["vocals"]["bytes"] == 600
    assert rep["kinds"]["drums"]["per_track"] == 100
    assert rep["prunable"] == {"bytes": 220, "tracks": 1}
    assert rep["keep_analysis_stems"] is True
    out = storage.prune_analysis_stems()
    assert out == {"tracks": 1, "bytes": 220}
    left = {(r["song_id"], r["stem_type"]) for r in
            models.get_conn().execute("SELECT song_id, stem_type FROM stems")}
    assert (done, "vocals") in left and (done, "drums") not in left
    assert (queued, "drums") in left
    assert (config.STEMS_DIR / f"s{done}" / "vocals.flac").exists()
    assert storage.projection({"vocals": 10.0}, 10_000, 200.0) == {"vocals": 20_000_000}


def test_settings_storage_routes_and_panel(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.routes import settings as settings_route
    importlib.reload(settings_route)
    assert "prunable" in settings_route.get_storage()
    app = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    panel = (ROOT / "frontend/src/components/StoragePanel.jsx").read_text(encoding="utf-8")
    assert "<StoragePanel />" in app
    assert "keep_analysis_stems" in panel and "pruneAnalysisStems" in panel


# ── Discover value preview ───────────────────────────────────────────────────

GROUPS = [
    {"need": "bed", "bpm_lo": 125, "bpm_hi": 130, "genre": "house",
     "tracks": [{"camelot": "8A"}, {"camelot": "2B"}]},
    {"need": "vocal", "bpm_lo": 170, "bpm_hi": 175, "genre": "drum & bass",
     "tracks": [{"camelot": "5A"}]},
]


def test_gap_fills_reads_only_what_the_upload_says():
    from ingest.fit_hints import gap_fills
    inst = {"bpm": 126.0, "camelot": None, "role": "instrumental"}
    assert gap_fills(inst, "", GROUPS)["n"] == 2
    # A printed key narrows the group to the keys that mix.
    assert gap_fills({**inst, "camelot": "9A"}, "", GROUPS)["n"] == 1
    # An acapella is no bed; an instrumental is no vocal.
    assert gap_fills({"bpm": 126.0, "camelot": None, "role": "acapella"}, "", GROUPS)["n"] == 0
    assert gap_fills({"bpm": 172.0, "camelot": None, "role": "instrumental"}, "", GROUPS)["n"] == 0
    # Half time is the same tempo.
    assert gap_fills({"bpm": 86.0, "camelot": None, "role": None}, "", GROUPS)["n"] == 1
    # No printed tempo: the genre decides, and nothing is invented without it.
    bare = {"bpm": None, "camelot": None, "role": None}
    assert gap_fills(bare, "Deep House", GROUPS)["n"] == 2
    assert gap_fills(bare, "", GROUPS)["n"] == 0
    assert gap_fills(bare, "Techno", GROUPS)["n"] == 0


def test_discover_rows_carry_the_preview_and_never_a_measured_tempo(tmp_path, monkeypatch):
    config, models = _setup(tmp_path, monkeypatch)
    from api.routes import discovery
    importlib.reload(discovery)
    monkeypatch.setattr(discovery, "_cached_gap_groups", lambda: GROUPS)
    rows = [{"title": "Song (Instrumental) 126 BPM", "genre": "House",
             "source_url": "https://soundcloud.com/x/y"}]
    discovery._with_fit_hints(rows)
    assert rows[0]["gap_fill"]["n"] == 2
    assert set(rows[0]["fit_hint"]) == {"bpm", "camelot", "role", "fits"}
    src = (ROOT / "api/workers/discovery_worker.py").read_text(encoding="utf-8")
    assert "_with_fit_hints" in src
    sc = (ROOT / "frontend/src/components/ScRows.jsx").read_text(encoding="utf-8")
    sug = (ROOT / "frontend/src/components/Suggestions.jsx").read_text(encoding="utf-8")
    assert "row.gap_fill?.n > 0" in sc
    assert "shortlist.addMany(gapFillers)" in sug
