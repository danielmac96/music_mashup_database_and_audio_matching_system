"""One track joining the library does not re-cluster the library (Task 7).

`rebuild_variant_clusters` ran after every track's analysis. Measured on the
206-track library (`analysis_runs` where `grp='dedup'`, 627 rows): median
982 ms, mean 5,343 ms, p90 18,425 ms, max 34,921 ms, 55.8 min in total — the
mean is five times the median and the worst run took 35 seconds, because each
run re-clusters a library that keeps growing while the track that triggered it
does not.

`assign_variant_cluster` reads only the candidates that could possibly be
variants of one track, and these tests pin that its answer is the SAME as the
full rebuild's on the cases that can tell the two apart — the title index above
all, which is what makes a re-upload under a different uploader name a variant.

Everything runs on `tmp_path` with an explicit `db_path`. Do NOT reach for
`monkeypatch.setenv("MASHUP_DB_PATH", …)` here: `config.DB_PATH` binds at
import, so setting the env var afterwards does nothing and the test would
silently run against the real library.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Two orthogonal timbres: cosine 0.0 over c1-12, well under AUDIO_CONFIRM_MIN
# (0.80). Random music clears 0.80 93% of the time (readme §5.6), so a
# disagreement has to be constructed, not sampled.
VA = [0.0] + [1.0] + [0.0] * 11
VB = [0.0] + [0.0, 1.0] + [0.0] * 10

# (title, artist, mfcc or None) — one fixture holding every branch that
# discriminates the incremental path from the full rebuild.
FIXTURE = [
    # Same artist, non-generic title: variants whatever the audio says (a
    # remix shares no timbre with its original, which is why case 2 of the
    # clustering rule does not ask).
    ("Fireproof", "Kade Monroe", VA),
    ("Fireproof (Extended Mix)", "Kade Monroe", VB),
    # (a) Same normalised title, DIFFERENT artists, agreeing audio → variants.
    # This is the re-upload the title index exists for, and the case a
    # candidate filter on the full `variant_key` would silently drop.
    ("Night Drive", "Uploader One", VA),
    ("Night Drive (Radio Edit)", "Uploader Two", VA),
    # (b) Same normalised title, different artists, disagreeing audio → two
    # different songs that happen to share a title.
    ("Paper Moon", "Someone A", VA),
    ("Paper Moon (Extended Mix)", "Someone B", VB),
    # (c) A generic title with the SAME artist → still needs audio, so the
    # artist rule must not fire.
    ("Intro", "Same Artist", VA),
    ("Intro (Original Mix)", "Same Artist", VB),
    # (d) A track with no features row at all → must not raise, and must
    # cluster exactly as the rebuild clusters it (the rebuild pulls
    # never-analysed songs in by title and artist alone).
    ("Ghost Lights", "Dee Jay", VA),
    ("Ghost Lights (Radio Edit)", "Dee Jay", None),
    # A record with no variant in the library at all → NULL.
    ("Solo Track", "Nobody", VA),
    # A title nothing survives of → un-keyable, never clustered, and must not
    # be grouped with other un-keyable rows.
    ("(Extended Mix)", "Ghost Uploader", VA),
    ("(Radio Edit)", "Ghost Uploader", VA),
]


def _seed(db_path, rows):
    from database.models import upsert_features, upsert_song

    ids = []
    for i, (title, artist, mfcc) in enumerate(rows):
        sid = upsert_song(title=title, artist=artist,
                          source_url=f"u://{i}", raw_path=f"/tmp/{i}.mp3",
                          duration_secs=200, db_path=db_path)
        if mfcc is not None:
            upsert_features(sid, "full", {"mfcc": list(mfcc), "bpm": 124.0},
                            db_path=db_path)
        ids.append(sid)
    return ids


def _clusters(db_path):
    from database.models import get_conn

    conn = get_conn(db_path)
    try:
        return {r["id"]: r["variant_cluster"] for r in
                conn.execute("SELECT id, variant_cluster FROM songs")}
    finally:
        conn.close()


def _clear(db_path):
    from database.models import get_conn

    conn = get_conn(db_path)
    try:
        conn.execute("UPDATE songs SET variant_cluster=NULL")
        conn.commit()
    finally:
        conn.close()


def test_incremental_assignment_matches_a_full_rebuild(tmp_path):
    """The whole basis of the change: because `cluster_variants` only ever
    unions INSIDE a normalised-title group, a cluster cannot span two groups —
    so clustering one group is exactly the rebuild's answer for that group."""
    from matcher.dedup import assign_variant_cluster, rebuild_variant_clusters

    db = tmp_path / "t.db"
    ids = _seed(db, FIXTURE)

    rebuild_variant_clusters(db_path=db)
    want = _clusters(db)
    _clear(db)

    for sid in ids:
        assign_variant_cluster(sid, db_path=db)

    assert _clusters(db) == want

    # And the fixture really does exercise the branches: it must not be a map
    # of all-NULL or all-clustered, or it would pass against a broken filter.
    f = dict(zip(ids, FIXTURE))
    by_title = {}
    for sid, (title, _a, _m) in f.items():
        by_title.setdefault(title, sid)
    assert want[by_title["Fireproof"]] is not None          # same artist
    assert want[by_title["Night Drive"]] is not None        # (a) title index
    assert want[by_title["Paper Moon"]] is None             # (b) audio says no
    assert want[by_title["Intro"]] is None                  # (c) generic
    assert want[by_title["Ghost Lights (Radio Edit)"]] is not None   # (d)
    assert want[by_title["Solo Track"]] is None
    assert want[by_title["(Extended Mix)"]] is None         # un-keyable


def test_one_assignment_writes_only_its_own_title_group(tmp_path):
    """Assigning one track leaves every other title group exactly as it was."""
    from matcher.dedup import assign_variant_cluster, rebuild_variant_clusters

    db = tmp_path / "t.db"
    ids = _seed(db, FIXTURE)
    rebuild_variant_clusters(db_path=db)
    want = _clusters(db)
    _clear(db)

    night_drive = ids[2]
    assign_variant_cluster(night_drive, db_path=db)

    got = _clusters(db)
    assert got[ids[2]] == want[ids[2]]
    assert got[ids[3]] == want[ids[3]]      # its partner, written too
    for sid in ids:
        if sid not in (ids[2], ids[3]):
            assert got[sid] is None, "another group was touched"


def test_assignment_reads_only_the_candidate_rows(tmp_path, monkeypatch):
    """The point of the task: one track's worth of features, not the library's,
    and one `cluster_variants` call over the title group alone."""
    from database import models
    from matcher import dedup

    db = tmp_path / "t.db"
    ids = _seed(db, FIXTURE)

    seen = {}
    real_feats = models.get_all_features
    real_cluster = dedup.cluster_variants

    def spy_feats(*a, **kw):
        seen.setdefault("feats", []).append(kw)
        return real_feats(*a, **kw)

    def spy_cluster(rows):
        seen.setdefault("rows", []).append(len(rows))
        return real_cluster(rows)

    monkeypatch.setattr(models, "get_all_features", spy_feats)
    monkeypatch.setattr(dedup, "cluster_variants", spy_cluster)

    dedup.assign_variant_cluster(ids[3], db_path=db)   # Night Drive (Radio Edit)

    assert seen["rows"] == [2], "clustered more than its own title group"
    assert len(seen["feats"]) == 1
    kw = seen["feats"][0]
    assert sorted(kw["song_ids"]) == sorted(ids[2:4])
    assert kw["columns"] == models.FEATURES_SCORING_COLUMNS
    assert kw["db_path"] == db


def test_an_unkeyable_title_needs_no_candidate_scan(tmp_path, monkeypatch):
    """A title nothing survives of can have no variant, so it must not read
    the library to find that out."""
    from database import models
    from matcher import dedup

    db = tmp_path / "t.db"
    ids = _seed(db, FIXTURE)

    calls = []
    monkeypatch.setattr(models, "get_all_features",
                        lambda *a, **kw: calls.append(kw) or [])

    assert dedup.assign_variant_cluster(ids[11], db_path=db) is None
    assert calls == [], "scanned the library for an un-keyable title"


def test_a_track_leaving_a_group_repairs_the_group_it_left(tmp_path):
    """A title change moves a track out of its cluster. If it was the smallest
    id, its former mates keep a `variant_cluster` naming a song that is no
    longer in their cluster — so the assignment recomputes the group it left
    as well as the one it joined."""
    from database.models import get_conn
    from matcher.dedup import assign_variant_cluster, rebuild_variant_clusters

    db = tmp_path / "t.db"
    ids = _seed(db, [("Ghost Lights", "Dee Jay", VA),
                     ("Ghost Lights (Extended Mix)", "Dee Jay", VA),
                     ("Ghost Lights (Radio Edit)", "Dee Jay", VA)])
    rebuild_variant_clusters(db_path=db)
    assert _clusters(db) == {ids[0]: ids[0], ids[1]: ids[0], ids[2]: ids[0]}

    # update_song_metadata fills a blank title; a re-upsert can replace one.
    conn = get_conn(db)
    try:
        conn.execute("UPDATE songs SET title=? WHERE id=?",
                     ("Different Work Entirely", ids[0]))
        conn.commit()
    finally:
        conn.close()

    assign_variant_cluster(ids[0], db_path=db)
    got = _clusters(db)

    rebuild_variant_clusters(db_path=db)
    assert got == _clusters(db)
    assert got == {ids[0]: None, ids[1]: ids[1], ids[2]: ids[1]}


def test_assignment_survives_a_song_id_that_is_gone(tmp_path):
    from matcher.dedup import assign_variant_cluster

    db = tmp_path / "t.db"
    _seed(db, FIXTURE[:2])
    assert assign_variant_cluster(9999, db_path=db) is None


def test_analysis_uses_the_incremental_path():
    src = (REPO_ROOT / "api" / "workers" / "stages.py").read_text(
        encoding="utf-8")
    assert "assign_variant_cluster" in src
    assert "rebuild_variant_clusters" not in src


def test_rebuild_is_kept_as_the_manual_repair():
    """It has no production caller after this change — it is the repair for the
    cases the incremental path cannot see (a deleted song's former mates), and
    what tests/test_dedup.py pins. Keep it importable."""
    from matcher.dedup import rebuild_variant_clusters

    assert callable(rebuild_variant_clusters)
