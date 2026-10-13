"""The similarity web: GET /api/tracks/{id}/neighbors in pair and sound mode,
the embeddings table filled from feature_cache without re-analysis, and
embeddings_coverage in /api/analysis/status."""
import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

NODE_KEYS = {"song_id", "title", "artist", "bpm", "camelot", "genre", "style",
             "popularity", "score", "role", "best_pair"}
BEST_PAIR_KEYS = {"vocal_song_id", "vocal_section_idx", "inst_song_id",
                  "inst_section_idx", "score_percentile"}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.jobs
    importlib.reload(api.jobs)
    import api.routes.tracks as tracks
    importlib.reload(tracks)
    import api.routes.analysis as analysis
    importlib.reload(analysis)
    import api.server as server
    importlib.reload(server)

    songs = [models.upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200,
                                "House", plays=n * 10, likes=n, status="analysed")
             for n in range(8)]
    conn = models.get_conn()
    for n, sid in enumerate(songs):
        conn.execute(
            "INSERT INTO features (song_id, stem_type, bpm, camelot, tags_json) "
            "VALUES (?,?,?,?,?)",
            (sid, "full", 120.0 + n, f"{n + 1}A", json.dumps(
                {"genre": [{"label": "Electronic---Tech House", "p": 0.5}]})))
    conn.commit()
    conn.close()
    return TestClient(server.app), models, songs


def _pairs(models, rows):
    """rows: (vocal, bed, total, vocal_section, bed_section)."""
    models.bulk_upsert_candidates([models.candidate_row(
        {"song_id": v, "title": "v", "artist": "A", "bpm": 128.0, "camelot": "8A",
         "loudness_rms": 0.1, "energy": 0.5},
        {"song_id": i, "title": "b", "artist": "A", "bpm": 128.0, "camelot": "8A",
         "loudness_rms": 0.1, "energy": 0.5},
        {"total": total, "bpm_score": 1.0, "key_score": 1.0, "energy_score": 0.5,
         "timbre_score": 0.5},
        section_pair={"vocal_section_idx": vs, "inst_section_idx": bs})
        for v, i, total, vs, bs in rows])


def _library(models, s):
    _pairs(models, [
        (s[0], s[1], 0.90, 0, 0), (s[0], s[1], 0.95, 2, 3),    # two section pairings
        (s[0], s[2], 0.80, 0, 0),
        (s[3], s[0], 0.85, 1, 1),                              # centre as the bed
        (s[1], s[0], 0.70, 0, 0),                              # s1 on both sides
        (s[0], s[4], 0.60, 0, 0), (s[0], s[5], 0.50, 0, 0),
        (s[1], s[2], 0.75, 0, 0), (s[2], s[3], 0.65, 0, 0),    # among the neighbours
        (s[3], s[1], 0.55, 0, 0),
        (s[6], s[7], 0.99, 0, 0),                              # nothing to do with s0
    ])


# ── pair mode ────────────────────────────────────────────────────────────────

def test_pair_mode_contract(app):
    client, models, s = app
    _library(models, s)
    r = client.get(f"/api/tracks/{s[0]}/neighbors")
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"center", "mode", "nodes", "edges"}
    assert d["mode"] == "pair"
    for node in [d["center"], *d["nodes"]]:
        assert set(node) == NODE_KEYS
        assert isinstance(node["song_id"], int) and isinstance(node["title"], str)
        assert node["style"] == "Tech House" and node["genre"] == "House"
        assert 0.0 <= node["popularity"] <= 1.0
        assert isinstance(node["bpm"], float) and isinstance(node["camelot"], str)
    c = d["center"]
    assert c["song_id"] == s[0] and c["score"] is None and c["best_pair"] is None
    assert c["role"] == "both"
    for node in d["nodes"]:
        assert 0.0 <= node["score"] <= 1.0
        assert node["role"] in ("vocal", "bed", "both")
        assert set(node["best_pair"]) == BEST_PAIR_KEYS
        assert s[0] in (node["best_pair"]["vocal_song_id"], node["best_pair"]["inst_song_id"])
        assert node["score"] == node["best_pair"]["score_percentile"]
    for e in d["edges"]:
        assert set(e) == {"a", "b", "weight"} and 0.0 <= e["weight"] <= 1.0


def test_pair_mode_is_the_best_pairing_per_partner_across_both_roles(app):
    client, models, s = app
    _library(models, s)
    d = client.get(f"/api/tracks/{s[0]}/neighbors").json()
    by_id = {n["song_id"]: n for n in d["nodes"]}
    assert [n["song_id"] for n in d["nodes"]] == [s[1], s[3], s[2], s[4], s[5]]
    assert s[6] not in by_id and s[7] not in by_id and s[0] not in by_id
    # s1 is a bed under the centre (0.95, sections 2/3) and a vocal over it.
    assert by_id[s[1]]["role"] == "both"
    assert by_id[s[1]]["best_pair"] == {
        "vocal_song_id": s[0], "vocal_section_idx": 2, "inst_song_id": s[1],
        "inst_section_idx": 3, "score_percentile": by_id[s[1]]["score"]}
    assert by_id[s[3]]["role"] == "vocal"
    assert by_id[s[3]]["best_pair"]["vocal_song_id"] == s[3]
    assert by_id[s[2]]["role"] == "bed"
    scores = [n["score"] for n in d["nodes"]]
    assert scores == sorted(scores, reverse=True)


def test_edges_link_neighbours_only_deduplicated_and_capped(app):
    client, models, s = app
    _library(models, s)
    d = client.get(f"/api/tracks/{s[0]}/neighbors").json()
    pairs = [(e["a"], e["b"]) for e in d["edges"]]
    assert set(pairs) == {(s[1], s[2]), (s[2], s[3]), (s[1], s[3])}
    assert len(pairs) == len(set(pairs)) and all(a < b for a, b in pairs)
    assert all(s[0] not in p for p in pairs)
    weights = [e["weight"] for e in d["edges"]]
    assert weights == sorted(weights, reverse=True)

    one = client.get(f"/api/tracks/{s[0]}/neighbors", params={"edges": 1}).json()
    # Each node keeps its single strongest link: s1–s2 (for both) and s2–s3.
    assert {(e["a"], e["b"]) for e in one["edges"]} == {(s[1], s[2]), (s[2], s[3])}
    assert client.get(f"/api/tracks/{s[0]}/neighbors",
                      params={"edges": 0}).json()["edges"] == []


def test_k_is_clamped_and_hidden_or_excluded_are_left_out(app):
    client, models, s = app
    _library(models, s)
    url = f"/api/tracks/{s[0]}/neighbors"
    assert len(client.get(url, params={"k": 2}).json()["nodes"]) == 2
    assert len(client.get(url, params={"k": 0}).json()["nodes"]) == 1
    assert len(client.get(url, params={"k": 999}).json()["nodes"]) == 5

    models.hide_pair(s[0], s[2])
    models.exclude_track(s[3])
    ids = [n["song_id"] for n in client.get(url).json()["nodes"]]
    assert ids == [s[1], s[4], s[5]]
    ranks, total = models.partner_ranks(s[0], "vocal")
    assert set(ranks) == set(ids) and total == 3        # the same visibility rule


def test_no_pairs_is_an_empty_web_and_unknown_is_404(app):
    client, _, s = app
    d = client.get(f"/api/tracks/{s[0]}/neighbors").json()
    assert d["nodes"] == [] and d["edges"] == []
    assert d["center"]["song_id"] == s[0] and d["center"]["role"] is None
    assert client.get("/api/tracks/99999/neighbors").status_code == 404
    assert client.get(f"/api/tracks/{s[0]}/neighbors",
                      params={"mode": "vibes"}).status_code == 400


def test_pair_mode_is_two_queries(app, monkeypatch):
    client, models, s = app
    _library(models, s)
    import matcher.neighbors as nb
    seen = []
    real = models.get_conn

    def traced(*a, **kw):
        conn = real(*a, **kw)
        conn.set_trace_callback(seen.append)
        return conn
    monkeypatch.setattr(models, "get_conn", traced)
    nb.neighbors(s[0], k=12, mode="pair", edges=3)
    queries = [q for q in seen if q.lstrip().upper().startswith(("SELECT", "WITH"))
               and "FROM songs WHERE id=" not in q]
    assert len(queries) == 2, queries


# ── embeddings + sound mode ──────────────────────────────────────────────────

def _cache(models, sid, vec, h=None):
    h = h or f"hash{sid}"
    conn = models.get_conn()
    conn.execute(
        "INSERT INTO stems (song_id, stem_type, file_path, content_hash) VALUES (?,?,?,?) "
        "ON CONFLICT(song_id, stem_type) DO UPDATE SET content_hash=excluded.content_hash",
        (sid, "full", f"/x/{sid}.mp3", h))
    conn.commit()
    conn.close()
    models.put_feature_cache(h, "essentia.effnet", 1, "p", json.dumps(
        {"tags": {}, "embedding_mean": vec}))


def test_sound_mode_is_409_with_coverage_until_embeddings_exist(app):
    client, models, s = app
    r = client.get(f"/api/tracks/{s[0]}/neighbors", params={"mode": "sound"})
    assert r.status_code == 409
    body = r.json()
    assert set(body) == {"detail", "coverage", "with", "total"}
    assert isinstance(body["detail"], str) and body["detail"]
    assert body["coverage"] == 0 and body["with"] == 0 and body["total"] == 8

    _cache(models, s[0], [1.0, 0.0, 0.0])
    models.sync_embeddings()
    r = client.get(f"/api/tracks/{s[0]}/neighbors", params={"mode": "sound"})
    assert r.status_code == 409 and r.json()["with"] == 1     # nothing to compare to
    assert r.json()["coverage"] == pytest.approx(0.125)


def test_backfill_copies_from_the_cache_without_reanalysis(app):
    client, models, s = app
    _cache(models, s[0], [1.0, 0.0, 0.0])
    _cache(models, s[1], [0.9, 0.1, 0.0])
    r = client.post("/api/analysis/embeddings/backfill")
    assert r.status_code == 200
    import api.jobs as jobs
    job = jobs.get(r.json()["job_id"])
    assert job["status"] == "completed" and job["kind"] == "embeddings"
    assert job["result"]["written"] == 2
    assert job["result"]["coverage"] == {"with": 2, "total": 8, "coverage": 0.25}

    conn = models.get_conn()
    row = conn.execute("SELECT * FROM embeddings WHERE song_id=?", (s[1],)).fetchone()
    conn.close()
    assert row["kind"] == "effnet" and row["dim"] == 3
    assert row["content_hash"] == f"hash{s[1]}" and row["updated_at"]
    assert len(row["vector"]) == 12                         # float32
    assert models.unpack_vector(row["vector"]) == pytest.approx([0.9, 0.1, 0.0])

    assert models.sync_embeddings() == {"written": 0, "removed": 0}   # idempotent
    status = client.get("/api/analysis/status").json()
    assert status["embeddings_coverage"] == {"with": 2, "total": 8, "coverage": 0.25}


def test_new_audio_drops_the_old_vector(app):
    _client, models, s = app
    _cache(models, s[0], [1.0, 0.0, 0.0])
    models.sync_embeddings()
    # Re-downloaded: new bytes, not analysed yet.
    conn = models.get_conn()
    conn.execute("UPDATE stems SET content_hash='new' WHERE song_id=?", (s[0],))
    conn.commit()
    conn.close()
    assert models.sync_embeddings([s[0]]) == {"written": 0, "removed": 1}
    _cache(models, s[0], [0.0, 1.0, 0.0], h="new")
    assert models.sync_embeddings([s[0]])["written"] == 1
    assert models.embeddings_coverage()["with"] == 1
    models.delete_song(s[0])
    conn = models.get_conn()
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    conn.close()


def test_sound_mode_ranks_by_cosine_in_the_same_shape(app):
    client, models, s = app
    vecs = {s[0]: [1.0, 0.0, 0.0], s[1]: [0.9, 0.1, 0.0], s[2]: [0.0, 1.0, 0.0],
            s[3]: [0.5, 0.5, 0.0], s[4]: [-1.0, 0.0, 0.0]}
    for sid, v in vecs.items():
        _cache(models, sid, v)
    models.sync_embeddings()
    r = client.get(f"/api/tracks/{s[0]}/neighbors", params={"mode": "sound", "edges": 1})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"center", "mode", "nodes", "edges"} and d["mode"] == "sound"
    assert [n["song_id"] for n in d["nodes"]] == [s[1], s[3], s[2], s[4]]
    for node in [d["center"], *d["nodes"]]:
        assert set(node) == NODE_KEYS
        assert node["role"] is None and node["best_pair"] is None
    assert d["center"]["score"] is None
    assert d["nodes"][0]["score"] == pytest.approx(0.9939, abs=1e-3)
    assert d["nodes"][1]["score"] == pytest.approx(0.7071, abs=1e-3)
    assert d["nodes"][2]["score"] == 0 and d["nodes"][3]["score"] == 0   # clamped to 0..1
    assert all(set(e) == {"a", "b", "weight"} and 0 <= e["weight"] <= 1
               for e in d["edges"])
    assert (min(s[1], s[3]), max(s[1], s[3])) in {(e["a"], e["b"]) for e in d["edges"]}


def test_the_analysis_stage_syncs_its_track(app):
    """do_analyze copies the vector for the track it just analysed."""
    src = (ROOT / "api/workers/stages.py").read_text(encoding="utf-8")
    body = src[src.index("def do_analyze"):src.index("def do_quick")]
    assert "sync_embeddings([song_id])" in body
    assert body.index('update_song_status(song_id, "analysed")') \
        < body.index("sync_embeddings([song_id])")


def test_dedup_is_not_on_embeddings_yet():
    src = (ROOT / "matcher/dedup.py").read_text(encoding="utf-8")
    assert "sync_embeddings" not in src and "FROM embeddings" not in src
