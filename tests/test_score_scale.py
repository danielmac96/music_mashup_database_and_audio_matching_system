"""Scoring and listing at scale (readme §5.7 "Scale", §8).

Every speed-up here is pinned to the slow path it replaced, on the same
inputs: the cached placement to `place`, the keeper to the in-order heap, the
pruned run to pricing every pair, the cached percentile ranks to SQL's
PERCENT_RANK. None of them may change an answer."""
import heapq
import importlib
import importlib.util
import random
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _synth():
    path = ROOT / "scripts" / "synth_library.py"
    spec = importlib.util.spec_from_file_location("synth_library", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    out = tmp_path_factory.mktemp("scale")
    _synth().generate(out, 90, seed=5, planted=4)
    return out


def _bind(db: Path, tmp_path, monkeypatch, cap=None):
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    if cap is not None:
        monkeypatch.setenv("MASHUP_MAX_CANDIDATE_ROWS", str(cap))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    import matcher.match as match
    importlib.reload(match)
    return m, match


def _table(db: Path):
    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")
                if r[1] not in ("id", "created_at", "updated_at")]
        return conn.execute(
            f"SELECT {', '.join(cols)} FROM mashup_candidates ORDER BY id").fetchall()
    finally:
        conn.close()


# ── the keeper ───────────────────────────────────────────────────────────────

def _in_order_heap(rows, cap):
    """The bounded heap a run used before _TopRows, fed in emission order."""
    heap = []
    for total, seq in rows:
        item = (total, seq)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif total > heap[0][0]:
            heapq.heapreplace(heap, item)
    return sorted(heap, key=lambda t: (-t[0], t[1]))


@pytest.mark.parametrize("seed", range(12))
def test_the_keeper_keeps_what_the_in_order_heap_kept(seed):
    from matcher.match import _TopRows
    rng = random.Random(seed)
    n, cap = rng.randint(1, 400), rng.randint(1, 60)
    # Few distinct totals, so the cut always falls inside a tie.
    rows = [(round(rng.choice([0.2, 0.4, 0.5, 0.5, 0.6, 0.8]) + rng.randint(0, 2) / 100, 4),
             (0, k)) for k in range(n)]
    expected = _in_order_heap(rows, cap)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    keeper = _TopRows(cap)
    for total, seq in shuffled:
        if keeper.wants(total):
            keeper.push(total, seq, None, None)
    assert [(t[0], t[1]) for t in keeper.ranked()] == expected
    # The cut never overstates: nothing the heap kept is under it.
    if keeper.floor() is not None:
        assert all(t[0] >= keeper.floor() for t in expected)


# ── the cached placement ─────────────────────────────────────────────────────

def _grid(rng, start, bars, bpm, wobble=0.0):
    bar = 240.0 / bpm
    dbs = [start + k * bar + rng.uniform(-wobble, wobble) for k in range(bars)]
    return {"start_sec": start, "end_sec": start + bars * bar, "downbeats": dbs}


def test_place_cached_is_place_to_the_last_bit():
    from matcher.phrase_align import place, place_cached
    rng = random.Random(3)
    cache = {}
    judged = 0
    for _ in range(600):
        v = _grid(rng, rng.uniform(0, 90), rng.choice([2, 4, 8, 16, 32]),
                  rng.uniform(80, 170), rng.choice([0.0, 0.004, 0.03]))
        b = _grid(rng, rng.uniform(0, 90), rng.choice([1, 4, 8, 16, 24, 40]),
                  rng.uniform(80, 170), rng.choice([0.0, 0.004, 0.03]))
        if rng.random() < 0.1:
            b["downbeats"] = []
        if rng.random() < 0.1:              # an unsorted grid takes place's path
            rng.shuffle(v["downbeats"])
        phrases = None
        if v["downbeats"] and rng.random() < 0.5:
            d = sorted(v["downbeats"])[min(1, len(v["downbeats"]) - 1)]
            pick = rng.uniform(-0.3, 0.1)
            phrases = [{"downbeat_sec": d, "pickup_sec": pick, "onset_sec": d + pick,
                        "bars": rng.choice([None, 4, 8, 12])}]
        stretch = rng.choice([1.0, rng.uniform(0.9, 1.12)])
        tol = rng.choice([25.0, 60.0])
        want = place(v, b, stretch, phrases, tol)
        # Twice: the second call answers from the cache.
        assert place_cached(v, b, stretch, phrases, tol, cache=cache) == want
        assert place_cached(v, b, stretch, phrases, tol, cache=cache) == want
        judged += want["timing_ok"] is not None
    assert judged > 200


def test_place_cached_agrees_on_a_synthetic_library(library, tmp_path, monkeypatch):
    db = tmp_path / "p.db"
    shutil.copy(library / "mashup.db", db)
    m, match = _bind(db, tmp_path, monkeypatch)
    from matcher.phrase_align import place, place_cached
    from matcher.sections import usable_sections
    from database.alignment import get_vocal_phrases
    ids = [r[0] for r in m.get_conn().execute("SELECT id FROM songs ORDER BY id LIMIT 40")]
    secs = m.sections_for_songs(ids, db_path=m.DB_PATH, engine="allin1")
    cache, verdicts = {}, set()
    for a in ids[:12]:
        phrases = get_vocal_phrases(a, db_path=m.DB_PATH)
        for v in usable_sections(secs.get(a, []), True):
            for b_id in ids:
                for b in usable_sections(secs.get(b_id, []), False):
                    for stretch in (1.0, 1.031):
                        want = place(v, b, stretch, phrases)
                        assert place_cached(v, b, stretch, phrases, cache=cache) == want
                        verdicts.add(want["timing_ok"])
    assert {0, 1} <= verdicts               # the gate passed some and failed some


# ── the pruned run ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("cap", [400, 3000])
def test_a_pruned_run_stores_the_rows_pricing_every_pair_stores(
        library, tmp_path, monkeypatch, cap):
    tables, stats = {}, {}
    for prune in (False, True):
        db = tmp_path / f"prune_{prune}.db"
        shutil.copy(library / "mashup.db", db)
        # The cap alone: each track's reserve is a search in a pruned run
        # (its own test below), so only the cap is exact against the full one.
        monkeypatch.setenv("MASHUP_RESERVE_PARTNERS", "0")
        m, match = _bind(db, tmp_path, monkeypatch, cap=cap)
        monkeypatch.setattr(match, "PRUNE_RUNS", prune)
        res = match.score_all_pairs(db_path=m.DB_PATH)
        tables[prune], stats[prune] = _table(db), res["_pruned"]
    assert stats[False] is None
    assert stats[True]["priced"] < stats[True]["pairs"]      # it did skip pairs
    # The cap bit: the cut was a real one.
    assert sum(1 for r in tables[True] if r[0] == "vocal_over_instrumental") == cap
    assert tables[True] == tables[False]


def test_a_ceiling_is_never_under_a_row_it_bounds(library, tmp_path, monkeypatch):
    """Each tier is priced only down to its own ceiling edge, so a ceiling
    that understated one pair would drop a row the full run keeps. Checked
    directly: with the cap out of reach nothing is cut, and every stored
    total sits at or under its pair's ceiling."""
    db = tmp_path / "ceil.db"
    shutil.copy(library / "mashup.db", db)
    m, match = _bind(db, tmp_path, monkeypatch, cap=3000)
    seen = {}
    real = match._score_pruned

    def spy(passes, gate, keeper, usable, price, **kw):
        def priced(top, bed, scores, seq):
            before = {id(t) for t in keeper.heap}
            price(top, bed, scores, seq)
            for t in keeper.heap:
                if id(t) not in before:
                    seen.setdefault((top["song_id"], bed["song_id"]), []).append(t[0])
        keeper.cap = 10 ** 9                # nothing is cut: every pair is priced
        kw["reserve"] = None
        out = real(passes, gate, keeper, usable, priced, **kw)
        spy.passes = passes
        return out

    monkeypatch.setattr(match, "_score_pruned", spy)
    ceilings = {}
    real_iter = match._iter_scored_pairs

    def watch(top, bed, **kw):
        pre = kw.get("pre_select")
        if pre is not None:
            ub = pre.__defaults__[0]
            for i, t in enumerate(top.feats):
                for j, b in enumerate(bed.feats):
                    ceilings[(t["song_id"], b["song_id"])] = float(ub[i, j])
        return real_iter(top, bed, **kw)

    monkeypatch.setattr(match, "_iter_scored_pairs", watch)
    match.score_all_pairs(db_path=m.DB_PATH)
    assert len(seen) > 500
    for pair, totals in seen.items():
        assert max(totals) <= ceilings[pair], pair


# ── the listing's ranks ──────────────────────────────────────────────────────

_REFERENCE = """
    WITH pct AS (SELECT id, PERCENT_RANK() OVER (PARTITION BY combo_type
                     ORDER BY score_total) AS p FROM mashup_candidates),
         nrg AS (SELECT id, PERCENT_RANK() OVER (PARTITION BY combo_type
                     ORDER BY inst_energy) AS e FROM mashup_candidates)
    SELECT mc.id, pct.p, nrg.e FROM mashup_candidates mc
    JOIN pct ON pct.id = mc.id JOIN nrg ON nrg.id = mc.id"""


@pytest.fixture()
def scored(library, tmp_path, monkeypatch):
    db = tmp_path / "listed.db"
    shutil.copy(library / "mashup.db", db)
    m, match = _bind(db, tmp_path, monkeypatch, cap=1500)
    match.score_all_pairs(db_path=m.DB_PATH)
    conn = m.get_conn()
    # Ties and NULLs, which is where a hand-rolled rank goes wrong.
    conn.execute("UPDATE mashup_candidates SET inst_energy = NULL WHERE id % 7 = 0")
    conn.execute("UPDATE mashup_candidates SET inst_energy = 0.5 WHERE id % 7 = 1")
    conn.commit()
    ref = {r[0]: (r[1], r[2]) for r in conn.execute(_REFERENCE)}
    conn.close()
    return m, ref


def test_listed_percentiles_are_sqls_percent_rank(scored):
    m, ref = scored
    for kw in ({}, {"combo_type": "instrumental_over_instrumental"},
               {"order": "tier"}):
        rows = m.get_candidates_enriched(limit=400, **kw)
        assert len(rows) > 100
        for r in rows:
            assert r["score_percentile"] == ref[r["id"]][0]
            assert r["energy_pct"] == ref[r["id"]][1]
    for r in m.best_bed_per_vocal(limit=50):
        assert r["score_percentile"] == ref[r["id"]][0]


def test_percentile_filters_select_what_the_windows_selected(scored):
    m, ref = scored
    combo = "vocal_over_instrumental"
    conn = m.get_conn()
    of_combo = {r[0] for r in conn.execute(
        "SELECT id FROM mashup_candidates WHERE combo_type=?", (combo,))}
    conn.close()

    def listed(**kw):
        return {r["id"] for r in m.get_candidates_enriched(
            combo_type=combo, limit=5000, max_per_song_pair=0, **kw)}

    for p in (0.0, 0.31, 0.5, 0.75, 0.999, 1.0):
        assert listed(min_score=p) == {i for i in of_combo if ref[i][0] >= p}, p
    for band, (lo, hi) in m.ENERGY_BANDS.items():
        assert listed(energy=band) == {
            i for i in of_combo if lo <= ref[i][1] < hi}, band
    from matcher.compat import TIER_MIN_PCT, TIERS
    strong = {r["id"] for r in m.get_candidates_enriched(
        combo_type=combo, limit=5000, max_per_song_pair=0, tier="strong")}
    assert strong and all(ref[i][0] * 100.0 >= TIER_MIN_PCT[TIERS[0]] for i in strong)


def test_the_ranks_follow_the_table(scored):
    m, _ref = scored
    before = m.candidate_ranks(m.DB_PATH)
    assert m.candidate_ranks(m.DB_PATH) is before           # unchanged: reused
    conn = m.get_conn()
    conn.execute("UPDATE mashup_candidates SET score_total = score_total / 2 "
                 "WHERE id = (SELECT MAX(id) FROM mashup_candidates)")
    conn.commit()
    conn.close()
    assert m.candidate_ranks(m.DB_PATH) is not before       # an in-place rewrite


def test_the_listing_runs_no_window_over_the_candidates():
    src = (ROOT / "database" / "models.py").read_text(encoding="utf-8")
    body = src.split("def get_candidates_enriched")[1].split("\ndef ")[0]
    assert "FROM mashup_candidates\n            )" not in body
    assert body.count("PERCENT_RANK") == 1                  # the songs' popularity


# ── Segments ─────────────────────────────────────────────────────────────────

def test_a_hidden_pair_is_not_a_live_match(scored):
    m, _ref = scored
    from matcher import segment_match
    seg, out = next(
        (s, o) for s in m.list_segments("vocal")
        for o in [segment_match.rank_partners(s["segment_id"], "vocal",
                                              db_path=m.DB_PATH)] if o["rows"])
    assert out["hidden"] == 0
    gone = out["rows"][0]["inst_song_id"]
    m.hide_pair(seg["song_id"], gone)
    again = segment_match.rank_partners(seg["segment_id"], "vocal", db_path=m.DB_PATH)
    assert again["hidden"] == 1 and again["gated"] == out["gated"]
    assert gone not in {r["inst_song_id"] for r in again["rows"]}
    assert {r["inst_song_id"] for r in again["rows"]} >= \
        {r["inst_song_id"] for r in out["rows"] if r["inst_song_id"] != gone} - {gone}
    m.unhide_pair(seg["song_id"], gone)
    back = segment_match.rank_partners(seg["segment_id"], "vocal", db_path=m.DB_PATH)
    assert [r["inst_song_id"] for r in back["rows"]] == \
        [r["inst_song_id"] for r in out["rows"]]


# ── each track's own best partners ───────────────────────────────────────────

def test_every_track_keeps_its_best_partners_under_the_cap(library, tmp_path, monkeypatch):
    """The cap keeps the library's best rows; each track also keeps its best
    partners found by a bounded search (config.RESERVE_PARTNERS from
    match.RESERVE_SEARCH times as many by ceiling), on both sides. Pinned:
    every track with that many gated partners has them, every reserved row is
    the row the full run computes for that pair, and the search finds most
    of each track's true best partners."""
    full = tmp_path / "full.db"
    shutil.copy(library / "mashup.db", full)
    m, match = _bind(full, tmp_path, monkeypatch, cap=10 ** 6)
    match.score_all_pairs(db_path=m.DB_PATH)
    conn = sqlite3.connect(full)
    best = conn.execute(
        "SELECT vocal_song_id, inst_song_id, MAX(score_total) FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental' GROUP BY 1, 2").fetchall()
    conn.close()

    capped = tmp_path / "capped.db"
    shutil.copy(library / "mashup.db", capped)
    monkeypatch.setenv("MASHUP_RESERVE_PARTNERS", "3")
    m, match = _bind(capped, tmp_path, monkeypatch, cap=300)
    res = match.score_all_pairs(db_path=m.DB_PATH)
    assert res["_pruned"]["for_reserve"] > 0
    conn = sqlite3.connect(capped)
    kept = {(r[0], r[1]): r[2] for r in conn.execute(
        "SELECT vocal_song_id, inst_song_id, MAX(score_total) FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental' GROUP BY 1, 2")}
    conn.close()
    whole = {(v, b): t for v, b, t in best}
    for pair, total in kept.items():
        assert whole[pair] == total, pair          # a true row, never an estimate
    found = wanted = 0
    for side in (0, 1):
        by_track = {}
        for row in best:
            by_track.setdefault(row[side], []).append(row)
        assert len(by_track) > 40
        for track, rows in by_track.items():
            mine = [p for p in kept if p[side] == track]
            assert len(mine) >= min(3, len(rows)), (side, track)
            rows.sort(key=lambda r: -r[2])
            top = {(v, b) for v, b, _t in rows[:3]}
            found += len(top & set(mine))
            wanted += len(top)
    recall = found / wanted
    print("\nreserve recall of the true best 3:", round(recall, 3))
    assert recall >= 0.9, recall
