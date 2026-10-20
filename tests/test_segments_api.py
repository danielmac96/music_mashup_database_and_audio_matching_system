"""/api/segments: the section-level playground's two contracts.

The list — one row per offered section, on each side — and the live partner
ranking, which must agree with Score library wherever both answer: a section
pairing the table stores gets the same total, percentile, tier, transpose and
recipe here as on its dock card (readme §7, one implementation per question).
"""
import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


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
    import matcher.match as match
    importlib.reload(match)
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.routes.segments as segments
    importlib.reload(segments)
    import api.server as server
    importlib.reload(server)
    return TestClient(server.app), models


def _rows(client, role):
    """GET /api/segments unpacked into one dict per row, as the screen does."""
    body = client.get(f"/api/segments?role={role}").json()
    cols = body["columns"]
    out = []
    for packed in body["rows"]:
        row = dict(zip(cols, packed))
        out.append({**body["songs"][str(row["song_id"])], **row})
    return body, out


def _chroma(root, third=4):
    v = [0.05] * 12
    for k, w in ((0, 1.0), (third, 0.7), (7, 0.8)):
        v[(root + k) % 12] = w
    return v


MFCC = [-200.0, 80.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]


def _song(models, title, bpm, camelot, root, *, vocal=True, bed=True,
          quality=None, provisional=False):
    sid = models.upsert_song(title=title, artist=f"{title} Artist",
                             source_url=f"u://{title}")
    feats = {"bpm": bpm, "key": "C", "mode": "major", "camelot": camelot,
             "bpm_confidence": 0.8, "key_confidence": 0.05, "energy": 0.5,
             "loudness_rms": 0.1, "mfcc": [m + root for m in MFCC],
             "band_energy": [0.125] * 8}
    models.upsert_features(sid, "full", feats)
    if vocal:
        models.upsert_features(sid, "vocals", feats)
        models.upsert_stem(sid, "vocals", f"/x/{sid}_v.wav")
    if bed:
        models.upsert_features(sid, "instrumental", feats)
        models.upsert_stem(sid, "instrumental", f"/x/{sid}_i.wav")
    if quality is not None:
        models.update_stem_quality(sid, "vocals", {"quality": quality})

    def sec(i, label, a, b, vp, energy_abs=None, **extra):
        return {"section_index": i, "start_sec": float(a), "end_sec": float(b),
                "label": label, "energy": 0.8, "repetition": 2, "confidence": 0.9,
                "vocal_presence": vp, "downbeats": [float(a)], "bpm": bpm,
                "bar_count": (b - a) / (240.0 / bpm), "energy_absolute": energy_abs,
                "provisional": 1 if provisional else 0,
                "chroma_vocal": _chroma(root), "chroma_bed": _chroma(root), **extra}

    models.replace_sections(sid, [
        sec(0, "intro", 0, 16, 0.1, 0.2),
        sec(1, "chorus", 16, 48, 0.9, 0.9, f0={"p10_midi": 60.0, "p90_midi": 72.0,
                                                 "median_midi": 66.0}),
        sec(2, "verse", 48, 80, 0.6, None),
        sec(3, "drop", 80, 112, 0.05, 0.7),
    ])
    return sid


def _library(models):
    ids = {
        "A": _song(models, "A", 124.0, "8B", 0),
        "B": _song(models, "B", 124.0, "8B", 0),
        "C": _song(models, "C", 126.0, "9B", 7),
        "D": _song(models, "D", 122.0, "8A", 9),
        "E": _song(models, "E", 170.0, "3A", 3),       # outside the tempo gate
        "P": _song(models, "P", 124.0, "8B", 0, provisional=True),
        "N": _song(models, "N", 124.0, "8B", 0, vocal=False),   # bed only
    }
    return ids


# ── the list ─────────────────────────────────────────────────────────────────

def test_the_two_sides_and_what_each_leaves_out(app):
    client, models = app
    ids = _library(models)
    vocal, v_rows = _rows(client, "vocal")
    _, b_rows = _rows(client, "bed")
    assert vocal["role"] == "vocal" and vocal["total"] == len(v_rows)
    # Song fields travel once per song, never per row.
    assert "title" not in vocal["columns"] and "bpm" not in vocal["columns"]
    assert set(vocal["songs"]) == {str(r["song_id"]) for r in v_rows}

    v_keys = {(r["song_id"], r["section_index"]) for r in v_rows}
    b_keys = {(r["song_id"], r["section_index"]) for r in b_rows}
    # No vocal stem → not a vocal; provisional → neither side.
    assert not any(k[0] == ids["N"] for k in v_keys)
    assert any(k[0] == ids["N"] for k in b_keys)
    assert not any(k[0] == ids["P"] for k in v_keys | b_keys)
    # The scorer's vocal floor: the intro (0.1) and drop (0.05) are not vocals.
    assert (ids["A"], 1) in v_keys and (ids["A"], 2) in v_keys
    assert (ids["A"], 0) not in v_keys and (ids["A"], 3) not in v_keys
    # Every section of a track with an instrumental is a bed.
    assert {(ids["A"], i) for i in range(4)} <= b_keys


def test_a_row_carries_the_full_mix_tempo_and_leaves_unmeasured_blank(app):
    client, models = app
    ids = _library(models)
    rows = {(r["song_id"], r["section_index"]): r for r in _rows(client, "vocal")[1]}
    chorus = rows[(ids["A"], 1)]
    assert chorus["bpm"] == 124.0 and chorus["camelot"] == "8B"
    assert chorus["label"] == "chorus" and chorus["duration"] == 32.0
    assert (chorus["sung_low"], chorus["sung_high"]) == (60.0, 72.0)
    assert chorus["title"] == "A" and chorus["role"] == "vocal"
    assert "chroma_vocal_json" not in chorus and "beat_times_json" not in chorus
    verse = rows[(ids["A"], 2)]
    # Unmeasured is None, never zero.
    assert verse["energy_rank"] is None and verse["sung_low"] is None
    assert verse["uses"] == 0 and chorus["segment_id"] != verse["segment_id"]


def test_excluded_tracks_and_a_bad_role(app):
    client, models = app
    ids = _library(models)
    models.exclude_track(ids["B"])
    rows = _rows(client, "bed")[1]
    assert not any(r["song_id"] == ids["B"] for r in rows)
    assert client.get("/api/segments?role=drums").status_code == 400


def test_list_opens_and_closes_one_connection(app, monkeypatch):
    _, models = app
    _library(models)
    opened, closed = [], []
    real = models.get_conn

    class Tracked:
        def __init__(self, conn):
            self._c = conn
            opened.append(self)

        def __getattr__(self, name):
            return getattr(self._c, name)

        def close(self):
            closed.append(self)
            self._c.close()

    monkeypatch.setattr(models, "get_conn", lambda *a, **k: Tracked(real(*a, **k)))
    models.list_segments("vocal")
    assert len(opened) == 1 and len(closed) == 1


# ── the partner ranking ──────────────────────────────────────────────────────

def _seg_id(client, role, song_id, idx):
    rows = _rows(client, role)[1]
    return next(r["segment_id"] for r in rows
                if r["song_id"] == song_id and r["section_index"] == idx)


def test_matches_rank_the_other_side_through_the_gate(app):
    client, models = app
    ids = _library(models)
    anchor = _seg_id(client, "vocal", ids["A"], 1)
    out = client.get(f"/api/segments/{anchor}/matches?role=vocal").json()
    assert out["anchor"]["segment_id"] == anchor and out["role"] == "vocal"
    songs = {r["inst_song_id"] for r in out["rows"]}
    assert ids["A"] not in songs                       # never itself
    assert ids["E"] not in songs                       # 170 BPM: outside the gate
    assert ids["P"] not in songs                       # provisional
    assert {ids["B"], ids["C"], ids["D"], ids["N"]} <= songs
    assert all(r["vocal_song_id"] == ids["A"] and r["vocal_section_idx"] == 1
               for r in out["rows"])
    totals = [r["score_total"] for r in out["rows"]]
    assert totals == sorted(totals, reverse=True)
    r = out["rows"][0]
    for key in ("segment_id", "semitone_shift", "recipe", "confidence_tier",
                "compat", "score_percentile", "stored", "score_section"):
        assert key in r
    # The partner's segment id names a bed row.
    bed_ids = {b["segment_id"] for b in _rows(client, "bed")[1]}
    assert all(row["segment_id"] in bed_ids for row in out["rows"])


def test_a_bed_anchor_ranks_vocals(app):
    client, models = app
    ids = _library(models)
    anchor = _seg_id(client, "bed", ids["B"], 3)
    out = client.get(f"/api/segments/{anchor}/matches?role=bed").json()
    assert out["rows"] and all(r["inst_song_id"] == ids["B"]
                               and r["inst_section_idx"] == 3 for r in out["rows"])
    vocal_ids = {v["segment_id"] for v in _rows(client, "vocal")[1]}
    assert all(r["segment_id"] in vocal_ids for r in out["rows"])
    assert ids["N"] not in {r["vocal_song_id"] for r in out["rows"]}


def test_a_stored_pairing_reads_the_same_live_and_listed(app):
    """The live arithmetic is the scorer's: with the table cleared, every
    pairing Score library would store gets the same total here; and with the
    table full, the row returned IS the stored row, dressed like the dock."""
    client, models = app
    ids = _library(models)
    from matcher.match import score_all_pairs
    score_all_pairs(scorer="heuristic")
    stored = client.get("/api/mashups?combo_type=vocal_over_instrumental"
                        f"&vocal_song_id={ids['A']}&max_per_song=0&limit=200").json()
    stored_rows = stored["candidates"] if isinstance(stored, dict) else stored
    assert stored_rows
    anchor = _seg_id(client, "vocal", ids["A"], 1)
    live = client.get(f"/api/segments/{anchor}/matches?role=vocal").json()["rows"]
    by_key = {(r["inst_song_id"], r["inst_section_idx"]): r for r in live}
    compared = 0
    for s in stored_rows:
        # Provisional sections are not offered here (they are re-cut as soon
        # as the stems land), though the bed side of a re-score can still
        # reach them.
        if s["vocal_section_idx"] != 1 or s["inst_song_id"] == ids["P"]:
            continue
        r = by_key[(s["inst_song_id"], s["inst_section_idx"])]
        assert r["stored"] is True
        for key in ("score_total", "score_percentile", "confidence_tier",
                    "semitone_shift", "recipe", "compat"):
            assert r[key] == s[key], key
        compared += 1
    assert compared

    # Now prove the live arithmetic itself: clear the table and recompute.
    totals = {(r["inst_song_id"], r["inst_section_idx"]): r["score_total"]
              for r in live if r["stored"]}
    models.clear_candidates()
    fresh = client.get(f"/api/segments/{anchor}/matches?role=vocal").json()["rows"]
    for r in fresh:
        assert r["stored"] is False
        k = (r["inst_song_id"], r["inst_section_idx"])
        if k in totals:
            assert r["score_total"] == totals[k]


def test_percentile_is_the_listing_percent_rank():
    from matcher.segment_match import _percent_rank
    totals = [0.5, 0.6, 0.6, 0.8]
    assert _percent_rank(totals, 0.5) == 0.0
    assert _percent_rank(totals, 0.6) == pytest.approx(1 / 3)
    assert _percent_rank(totals, 0.8) == 1.0
    assert _percent_rank(totals, 0.9) == 1.0
    assert _percent_rank([], 0.7) is None


def test_match_filters_and_errors(app):
    client, models = app
    ids = _library(models)
    anchor = _seg_id(client, "vocal", ids["A"], 1)
    rows = client.get(f"/api/segments/{anchor}/matches?role=vocal&max_shift=0"
                      ).json()["rows"]
    assert all(r["semitone_shift"] in (0, None) for r in rows)
    rows = client.get(f"/api/segments/{anchor}/matches?role=vocal&max_stretch=0.5"
                      ).json()["rows"]
    assert all(abs(r["stretch_factor"] - 1.0) <= 0.005 + 1e-9 for r in rows)
    assert {r["inst_song_id"] for r in rows} == {ids["B"], ids["N"]}
    assert client.get(f"/api/segments/{anchor}/matches?role=vocal&tier=bogus"
                      ).status_code == 400
    assert client.get("/api/segments/999999/matches?role=vocal").status_code == 404
    # A drop is not a vocal: asked as one, nothing to rank.
    drop = _seg_id(client, "bed", ids["A"], 3)
    assert client.get(f"/api/segments/{drop}/matches?role=vocal").json()["rows"] == []


def test_the_cached_list_follows_the_library_and_gzips(app):
    """The packed list is cached per state of the library (segment_match.
    _fingerprint): a new section line, a re-cut or an exclusion shows on the
    next load, never a stale body."""
    client, models = app
    ids = _library(models)
    _, before = _rows(client, "vocal")
    chorus = next(r for r in before if r["song_id"] == ids["A"] and r["section_index"] == 1)
    assert chorus["line"] is None
    models.set_section_line(ids["A"], chorus["start_sec"], chorus["end_sec"], "Shout it out")
    _, after = _rows(client, "vocal")
    chorus = next(r for r in after if r["song_id"] == ids["A"] and r["section_index"] == 1)
    assert chorus["line"] == "Shout it out"
    models.exclude_track(ids["C"])
    assert not any(r["song_id"] == ids["C"] for r in _rows(client, "vocal")[1])

    r = client.get("/api/segments?role=bed", headers={"Accept-Encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip" and r.json()["role"] == "bed"
    plain = client.get("/api/segments?role=bed", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.json() == r.json()
