"""T3.1 — the vectorised scorer must agree with the scalar one, pair for pair.

This is the golden-output gate the roadmap asks for: it computes every score the
slow, obviously-correct way (composite_score on one pair at a time, the same
functions the Plan expander and the dataset builder call) and asserts the bulk
path wrote exactly those numbers, for exactly those pairs, in exactly that order.

It runs against the current implementation as much as the new one — which is the
point. A refactor that quietly re-ranks the library is the failure mode worth
catching, and a 30-song library is small enough to brute-force here.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

N_SONGS = 24
BPM_MAX = 10.0
KEY_MIN = 0.55

# Deliberately awkward inputs, one per index below: an unknown key, a missing
# tempo, a short MFCC vector, a silent stem. Every one of them takes a different
# branch in the scalar scorer, so the vectorised path has to reproduce four
# fallbacks and not just the happy case.
UNKNOWN_KEY_IDX = 3
NO_BPM_IDX = 7
SHORT_MFCC_IDX = 11
ZERO_LOUDNESS_IDX = 15


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "vec.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


@pytest.fixture()
def library(db_path):
    """A synthetic library spread across the BPM and Camelot space, so the
    pre-filter admits a non-trivial mix of pairs rather than all or nothing."""
    from database.models import init_db, upsert_features, upsert_song

    init_db(db_path)
    codes = [f"{h}{l}" for h in range(1, 13) for l in "AB"]
    song_ids = []
    for n in range(N_SONGS):
        sid = upsert_song(
            f"Track {n}", f"Artist {n % 5}", f"https://sc/{n}", 200 + n,
            "Pop", status="analysed", db_path=db_path,
        )
        song_ids.append(sid)

        full_bpm = 96.0 + (n * 7) % 60          # 96–155, several near-misses
        camelot = codes[(n * 5) % len(codes)]
        if n == UNKNOWN_KEY_IDX:
            camelot = "?"
        if n == NO_BPM_IDX:
            full_bpm = None

        # MFCC: coefficient 0 large and same-sign (as in real data), 1–12 varied.
        mfcc = [180.0 + n] + [((n * (k + 3)) % 17) - 8.0 for k in range(12)]
        if n == SHORT_MFCC_IDX:
            mfcc = mfcc[:6]

        for stem in ("full", "vocals", "instrumental"):
            rms = 0.02 + 0.004 * ((n + len(stem)) % 9)
            if n == ZERO_LOUDNESS_IDX:
                rms = 0.0
            upsert_features(sid, stem, {
                # Stem tempo deliberately differs from the full mix's, so a
                # regression in _with_full_bpm shows up as a score mismatch.
                "bpm": (full_bpm + 3.0) if full_bpm else None,
                "key": "A", "mode": "minor", "camelot": camelot,
                "loudness_rms": rms, "energy": rms * 4,
                "bpm_confidence": 0.5, "key_confidence": 0.05,
                "mfcc": mfcc,
            }, db_path=db_path)
    return db_path, song_ids


def _reference(db_path):
    """Score every pair the slow way. Returns
    (voi_rows, ioi_rows) as lists of (vocal_song_id, inst_song_id, scores),
    in the nested-loop order the ranked list is built from."""
    from database.models import get_all_features
    from matcher.match import (
        _passes_filter, _with_full_bpm, composite_score, get_library_stats,
    )

    vocals = get_all_features(stem_type="vocals", db_path=db_path)
    inst = get_all_features(stem_type="instrumental", db_path=db_path)
    full = get_all_features(stem_type="full", db_path=db_path)
    full_by_song = {f["song_id"]: f for f in full}
    vocals = [_with_full_bpm(v, full_by_song) for v in vocals]
    inst = [_with_full_bpm(i, full_by_song) for i in inst]
    stats = get_library_stats(db_path=db_path, refresh=True)

    voi = []
    for v in vocals:
        for i in inst:
            if v["song_id"] == i["song_id"]:
                continue
            if not _passes_filter(v, i, BPM_MAX, KEY_MIN):
                continue
            voi.append((v["song_id"], i["song_id"],
                        composite_score(v, i, stats=stats,
                                        combo_type="vocal_over_instrumental")))

    ioi = []
    for a in inst:
        for b in inst:
            if a["song_id"] >= b["song_id"]:
                continue
            if not _passes_filter(a, b, BPM_MAX, KEY_MIN):
                continue
            ioi.append((a["song_id"], b["song_id"],
                        composite_score(a, b, stats=stats)))
    return voi, ioi


SUBS = ("bpm_score", "key_score", "energy_score", "timbre_score")


def test_scored_pairs_match_the_scalar_scorer(library):
    """Same pairs, same four sub-scores, same total — for the whole library."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import score_all_pairs

    score_all_pairs(db_path=db_path, bpm_max_diff=BPM_MAX,
                    key_min_score=KEY_MIN, scorer="heuristic")

    voi, ioi = _reference(db_path)
    assert voi, "fixture admits no vocal/instrumental pairs — test is vacuous"
    assert ioi, "fixture admits no instrumental/instrumental pairs"

    conn = get_conn(db_path)
    rows = {
        (r["combo_type"], r["vocal_song_id"], r["inst_song_id"]): dict(r)
        for r in conn.execute("SELECT * FROM mashup_candidates").fetchall()
    }
    conn.close()

    expected_keys = (
        {("vocal_over_instrumental", v, i) for v, i, _ in voi}
        | {("instrumental_over_instrumental", a, b) for a, b, _ in ioi}
    )
    assert set(rows) == expected_keys

    for combo, ref in (("vocal_over_instrumental", voi),
                       ("instrumental_over_instrumental", ioi)):
        for a, b, scores in ref:
            row = rows[(combo, a, b)]
            for name in SUBS:
                assert row[f"score_{name.split('_')[0]}"] == \
                    pytest.approx(scores[name], abs=1e-12), \
                    f"{combo} {a}->{b} {name}"
            assert row["score_total"] == pytest.approx(scores["total"], abs=1e-9)


def test_returned_lists_keep_the_scalar_ranking(library):
    """The returned lists drive the CLI report and the job's counts, so their
    contents and their descending-score order both have to survive."""
    db_path, _ = library
    from matcher.match import score_all_pairs

    results = score_all_pairs(db_path=db_path, bpm_max_diff=BPM_MAX,
                              key_min_score=KEY_MIN, scorer="heuristic")
    voi, ioi = _reference(db_path)

    for key, ref in (("vocal_over_instrumental", voi),
                     ("instrumental_over_instrumental", ioi)):
        got = results[key]
        assert len(got) == len(ref)
        # Stable sort on the nested-loop order: ties keep the order they were
        # generated in, so this pins ranking exactly, not just as a multiset.
        want = sorted(ref, key=lambda t: t[2]["total"], reverse=True)
        assert [(r["vocal_song_id"], r["inst_song_id"]) for r in got] == \
               [(a, b) for a, b, _ in want]
        assert [r["total"] for r in got] == [s["total"] for _, _, s in want]

    assert results["_scorer"] == "heuristic"


def test_rows_carry_the_display_metadata(library):
    """Bulk writes must still populate every column the ranked list reads."""
    db_path, _ = library
    from database.models import get_all_features, get_conn
    from matcher.match import score_all_pairs

    score_all_pairs(db_path=db_path, bpm_max_diff=BPM_MAX,
                    key_min_score=KEY_MIN, scorer="heuristic")
    by_song = {f["song_id"]: f
               for f in get_all_features(stem_type="vocals", db_path=db_path)}
    full = {f["song_id"]: f
            for f in get_all_features(stem_type="full", db_path=db_path)}

    conn = get_conn(db_path)
    row = dict(conn.execute(
        "SELECT * FROM mashup_candidates WHERE combo_type='vocal_over_instrumental' "
        "ORDER BY score_total DESC LIMIT 1").fetchone())
    conn.close()

    v = by_song[row["vocal_song_id"]]
    assert row["vocal_title"] == v["title"]
    assert row["vocal_camelot"] == v["camelot"]
    assert row["vocal_loudness_rms"] == v["loudness_rms"]
    # BPM is the full-mix reading, not the stem's (see _with_full_bpm).
    assert row["vocal_bpm"] == full[row["vocal_song_id"]]["bpm"]
    assert row["scorer"] == "heuristic" and row["model_version"] is None


def test_rescoring_is_idempotent(library):
    """Two runs, same thresholds → the same table. The bulk path must not
    accumulate duplicates or leave stale rows behind."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import score_all_pairs

    def snapshot():
        conn = get_conn(db_path)
        out = [tuple(r) for r in conn.execute(
            "SELECT combo_type, vocal_song_id, inst_song_id, score_total "
            "FROM mashup_candidates ORDER BY combo_type, vocal_song_id, "
            "inst_song_id").fetchall()]
        conn.close()
        return out

    score_all_pairs(db_path=db_path, bpm_max_diff=BPM_MAX,
                    key_min_score=KEY_MIN, scorer="heuristic")
    first = snapshot()
    score_all_pairs(db_path=db_path, bpm_max_diff=BPM_MAX,
                    key_min_score=KEY_MIN, scorer="heuristic")
    assert snapshot() == first
    assert first, "no candidates scored"


def test_empty_library_scores_nothing(db_path):
    from database.models import init_db
    from matcher.match import score_all_pairs

    init_db(db_path)
    results = score_all_pairs(db_path=db_path, scorer="heuristic")
    assert results["vocal_over_instrumental"] == []
    assert results["instrumental_over_instrumental"] == []


def test_library_too_small_to_normalise_still_matches_the_scalar_path(db_path):
    """Under 4 analysed tracks LibraryStats is unusable and timbre_score falls
    back to a raw MFCC cosine. That is a separate branch from the normalised
    one, and it is what a brand-new user's first two imports hit."""
    from database.models import init_db, upsert_features, upsert_song
    from matcher.match import score_all_pairs

    init_db(db_path)
    for n, (bpm, cam) in enumerate([(124.0, "8A"), (126.0, "8A")]):
        sid = upsert_song(f"S{n}", "A", f"https://sc/s{n}", 200, "Pop",
                          status="analysed", db_path=db_path)
        for stem in ("full", "vocals", "instrumental"):
            # Only three rows carry an MFCC vector, which is what keeps
            # LibraryStats under its 4-row minimum. The fourth (song 1's vocal)
            # has none, so the same run also covers the "unknown timbre" branch.
            has_mfcc = stem == "instrumental" or (n == 0 and stem == "vocals")
            upsert_features(sid, stem, {
                "bpm": bpm, "key": "A", "mode": "minor", "camelot": cam,
                "loudness_rms": 0.03 + 0.01 * n, "energy": 0.4,
                "mfcc": [190.0, 4.0 * n - 2, 1.0, -3.0, 2.0, 0.5, -1.0,
                         2.5, 0.0, -4.0, 1.5, 3.0, -2.0] if has_mfcc else None,
            }, db_path=db_path)

    from matcher.match import get_library_stats
    assert not get_library_stats(db_path=db_path, refresh=True).usable

    score_all_pairs(db_path=db_path, bpm_max_diff=BPM_MAX,
                    key_min_score=KEY_MIN, scorer="heuristic")
    voi, ioi = _reference(db_path)
    assert len(voi) == 2 and len(ioi) == 1

    from database.models import get_conn
    conn = get_conn(db_path)
    rows = {(r["combo_type"], r["vocal_song_id"], r["inst_song_id"]): dict(r)
            for r in conn.execute("SELECT * FROM mashup_candidates").fetchall()}
    conn.close()
    for combo, ref in (("vocal_over_instrumental", voi),
                       ("instrumental_over_instrumental", ioi)):
        for a, b, scores in ref:
            row = rows[(combo, a, b)]
            assert row["score_timbre"] == pytest.approx(scores["timbre_score"],
                                                        abs=1e-12)
            assert row["score_total"] == pytest.approx(scores["total"], abs=1e-9)


def test_progress_is_reported_monotonically(library):
    db_path, _ = library
    from matcher.match import score_all_pairs

    seen = []
    score_all_pairs(db_path=db_path, bpm_max_diff=BPM_MAX, key_min_score=KEY_MIN,
                    scorer="heuristic", progress=lambda p, m: seen.append((p, m)))
    assert seen
    pcts = [p for p, _ in seen]
    assert pcts == sorted(pcts)
    assert 0 <= pcts[0] and pcts[-1] <= 100
    assert all(m for _, m in seen)


def test_narrowing_the_filter_drops_stale_pairs(library):
    """Already covered for the scalar path in test_pipeline_queue; repeated here
    because bulk writes are a new chance to leave a previous run's rows."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import score_all_pairs

    def count():
        conn = get_conn(db_path)
        n = conn.execute("SELECT COUNT(*) c FROM mashup_candidates").fetchone()["c"]
        conn.close()
        return n

    score_all_pairs(db_path=db_path, bpm_max_diff=20.0, key_min_score=0.2,
                    scorer="heuristic")
    wide = count()
    score_all_pairs(db_path=db_path, bpm_max_diff=1.0, key_min_score=0.99,
                   scorer="heuristic")
    narrow = count()
    assert 0 <= narrow < wide
