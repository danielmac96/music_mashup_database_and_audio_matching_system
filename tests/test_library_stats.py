"""GET /api/analysis/library-stats — the Analysis dashboard's contract, on a
synthetic 1 000-track library: exact key sets, unmeasured left out (never
binned as zero), filters, and the one-pass-per-table time budget."""
import importlib
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

N = 1000
URL = "/api/analysis/library-stats"
GENRES = ["House", "Pop", "Techno", "Drum & Bass", ""]
STYLES = ["House", "Deep House", "Techno", "Europop"]
MOODS = ["energetic", "happy", "dark", "party", "dream"]
CAMELOT = [f"{n}{letter}" for n in range(1, 13) for letter in "AB"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stats")
    mp = pytest.MonkeyPatch()
    mp.setenv("MASHUP_DB_PATH", str(tmp / "test.db"))
    mp.setenv("MASHUP_AUDIO_ROOT", str(tmp / "audio"))
    mp.setenv("MASHUP_SETTINGS_DIR", str(tmp / "settings"))
    mp.setenv("MASHUP_ANALYZER", "librosa")
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.routes.analysis as analysis
    importlib.reload(analysis)
    import api.routes.tracks as tracks
    importlib.reload(tracks)
    import api.server as server
    importlib.reload(server)

    conn = models.get_conn()
    songs, feats, secs, stems = [], [], [], []
    for n in range(1, N + 1):
        measured = n % 10 != 0            # every tenth track is unanalysed
        songs.append((n, f"T{n}", f"A{n % 50}", f"https://sc/{n}",
                      GENRES[n % len(GENRES)], 1990 + n % 35 if n % 7 else 0,
                      "analysed" if measured else "downloaded"))
        if not measured:
            continue
        tags = None
        if n % 4:
            tags = json.dumps({
                "genre": [{"label": f"Electronic---{STYLES[n % 4]}", "p": 0.5},
                          {"label": "Electronic---Other", "p": 0.1}],
                "moodtheme": [{"label": MOODS[(n + k) % 5], "p": 0.3} for k in range(5)]})
        feats.append((n, "full", 60.0 + (n % 80) * 2 + 0.5, CAMELOT[n % 24],
                      0.01 + (n % 50) / 100.0, -20.0 + (n % 30) / 2.0 if n % 3 else None,
                      tags))
        stems.append((n, "vocals", f"/x/{n}.wav"))
        cls = ["vocal", "instrumental", "mixed", "instrumental"]
        for k in range(4):
            secs.append((n, k, k * 30.0, (k + 1) * 30.0, "verse",
                         cls[(n + k) % 4] if n % 5 else "unknown",
                         0.2 + 0.2 * ((n + k) % 4)))
    conn.executemany(
        "INSERT INTO songs (id, title, artist, source_url, genre, release_year, status) "
        "VALUES (?,?,?,?,?,?,?)", songs)
    conn.executemany(
        "INSERT INTO features (song_id, stem_type, bpm, camelot, energy, lufs, tags_json) "
        "VALUES (?,?,?,?,?,?,?)", feats)
    conn.executemany(
        "INSERT INTO stems (song_id, stem_type, file_path) VALUES (?,?,?)", stems)
    conn.executemany(
        "INSERT INTO sections (song_id, section_index, start_sec, end_sec, label, "
        "section_class, vocal_presence) VALUES (?,?,?,?,?,?,?)", secs)
    conn.executemany(
        "INSERT INTO mashup_candidates (vocal_song_id, inst_song_id, combo_type, "
        "score_total, vocal_section_idx, inst_section_idx) VALUES (?,?,?,?,?,?)",
        [(v, v + 1, "vocal_over_instrumental", 0.5 + (v % 40) / 100.0, k, 0)
         for v in range(1, 400) for k in range(3)])
    conn.executemany(
        "INSERT INTO pair_feedback (vocal_song_id, inst_song_id, verdict, rating) "
        "VALUES (?,?,?,?)",
        [(1, 2, "love", 5), (3, 4, "love", 4), (5, 6, "ok", 3), (7, 8, "no", 1),
         (9, 11, "no", None), (11, 12, "love", None)])
    conn.execute("UPDATE songs SET pairs_input_at='2026-10-02 00:00:00' WHERE id IN (1,2,3)")
    conn.commit()
    conn.close()
    models.set_pref("last_score", {"started_at": "2026-10-01 00:00:00"})
    gid = models.create_crate("warm up")["id"]
    models.add_songs_to_crate(gid, [1, 2, 3, 4, 5])
    yield TestClient(server.app), models, gid
    mp.undo()


def _get(client, **params):
    r = client.get(URL, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _measured():
    return [n for n in range(1, N + 1) if n % 10]


# ── contract ─────────────────────────────────────────────────────────────────

def test_contract_key_sets_and_types(app):
    client, _, _ = app
    d = _get(client)
    assert set(d) == {"filters", "counts", "bpm_hist", "camelot_wheel", "genre_top",
                      "style_top", "mood_top", "year_hist", "energy_hist",
                      "lufs_hist", "vocal_share_hist", "scatter", "pairs"}
    assert set(d["filters"]) == {"group", "genre", "style", "bpm_min", "bpm_max",
                                 "year_min", "year_max", "camelot", "vocal_forward"}
    assert d["filters"] == {"group": None, "genre": None, "style": None,
                            "bpm_min": None, "bpm_max": None, "year_min": None,
                            "year_max": None, "camelot": [], "vocal_forward": None}
    assert set(d["counts"]) == {"tracks", "analysed", "with_stems", "with_tags"}
    assert all(isinstance(v, int) for v in d["counts"].values())
    assert set(d["pairs"]) == {"scored", "stale_tracks", "rated", "loved", "rejected"}
    assert all(isinstance(v, int) for v in d["pairs"].values())

    assert len(d["bpm_hist"]) == 70
    assert [(b["lo"], b["hi"]) for b in d["bpm_hist"]] == \
        [(60 + 2 * i, 62 + 2 * i) for i in range(70)]
    assert [w["code"] for w in d["camelot_wheel"]] == CAMELOT       # always 24
    for key in ("energy_hist", "lufs_hist", "vocal_share_hist"):
        assert len(d[key]) == 20
        for b in d[key]:
            assert set(b) == {"lo", "hi", "n"} and isinstance(b["n"], int)
            assert b["lo"] < b["hi"]
    for key in ("genre_top", "style_top", "mood_top"):
        assert d[key] and len(d[key]) <= 15
        assert all(set(x) == {"value", "n"} and isinstance(x["value"], str)
                   for x in d[key])
        ns = [x["n"] for x in d[key]]
        assert ns == sorted(ns, reverse=True)
    assert all(set(y) == {"year", "n"} for y in d["year_hist"])
    assert [y["year"] for y in d["year_hist"]] == sorted(y["year"] for y in d["year_hist"])
    for p in d["scatter"]:
        assert set(p) == {"song_id", "energy", "lufs", "camelot", "bpm"}


def test_counts_and_totals(app):
    client, _, _ = app
    d = _get(client)
    measured = _measured()
    assert d["counts"] == {"tracks": N, "analysed": len(measured),
                           "with_stems": len(measured),
                           "with_tags": sum(1 for n in measured if n % 4)}
    assert sum(w["n"] for w in d["camelot_wheel"]) == len(measured)
    assert sum(b["n"] for b in d["energy_hist"]) == len(measured)
    assert sum(x["n"] for x in d["genre_top"]) == sum(1 for n in range(1, N + 1) if n % 5)
    # Tagged tracks have n % 4 in 1..3: the top style, never the runner-up.
    assert {x["value"] for x in d["style_top"]} == set(STYLES[1:])
    assert sum(y["n"] for y in d["year_hist"]) == sum(1 for n in range(1, N + 1) if n % 7)
    assert all(y["year"] > 0 for y in d["year_hist"])
    assert d["pairs"] == {"scored": 399 * 3, "stale_tracks": 3, "rated": 6,
                          "loved": 3, "rejected": 2}


def test_unmeasured_is_left_out_not_binned_as_zero(app):
    client, _, _ = app
    d = _get(client)
    measured = _measured()
    with_lufs = [n for n in measured if n % 3]
    assert sum(b["n"] for b in d["lufs_hist"]) == len(with_lufs)
    assert d["lufs_hist"][0]["lo"] == pytest.approx(-19.5)      # not dragged to 0
    assert d["lufs_hist"][-1]["hi"] < 0
    assert len(d["scatter"]) == len(with_lufs)
    assert all(p["energy"] is not None and p["lufs"] is not None for p in d["scatter"])
    # Sections whose class is unknown are no sung share at all.
    classed = [n for n in measured if n % 5]
    assert sum(b["n"] for b in d["vocal_share_hist"]) == len(classed)
    # BPMs run 60.5 … 218.5; the histogram stops at 200 and drops the rest.
    in_range = [n for n in measured if 60.0 + (n % 80) * 2 + 0.5 < 200]
    assert sum(b["n"] for b in d["bpm_hist"]) == len(in_range)


def test_sung_share_is_the_librarys_vox_percent(app):
    """One definition: the histogram's values are /api/tracks' vocal_coverage."""
    client, models, _ = app
    from analysis.library_stats import _hist
    tracks = client.get("/api/tracks").json()
    rows = tracks["tracks"] if isinstance(tracks, dict) else tracks
    cov = [t["mash"]["vocal_coverage"] for t in rows
           if (t.get("mash") or {}).get("vocal_coverage") is not None]
    assert cov
    assert _get(client)["vocal_share_hist"] == _hist(cov)


# ── filters ──────────────────────────────────────────────────────────────────

def test_filters_narrow_and_echo(app):
    client, _, gid = app
    d = _get(client, genre="house", bpm_min=120, bpm_max=130, camelot="8a, 9B",
             year_min=2000, year_max=2010, vocal_forward="true", style="Deep House")
    assert d["filters"] == {"group": None, "genre": "house", "style": "Deep House",
                            "bpm_min": 120.0, "bpm_max": 130.0, "year_min": 2000,
                            "year_max": 2010, "camelot": ["8A", "9B"],
                            "vocal_forward": True}

    d = _get(client, bpm_min=120, bpm_max=130)
    want = [n for n in _measured() if 120 <= 60.0 + (n % 80) * 2 + 0.5 <= 130]
    assert d["counts"]["tracks"] == len(want) > 0       # unmeasured BPM fails the filter
    assert sum(b["n"] for b in d["bpm_hist"] if 120 <= b["lo"] <= 130) == len(want)

    d = _get(client, camelot="8A,8B")
    assert {w["code"] for w in d["camelot_wheel"] if w["n"]} == {"8A", "8B"}

    d = _get(client, style="deep house")
    assert [x["value"] for x in d["style_top"]] == ["Deep House"]

    d = _get(client, genre="Pop")
    assert [x["value"] for x in d["genre_top"]] == ["Pop"]

    d = _get(client, year_min=2000, year_max=2004)
    assert {y["year"] for y in d["year_hist"]} == {2000, 2001, 2002, 2003}   # 2004 never occurs

    fwd = _get(client, vocal_forward="true")["counts"]["tracks"]
    back = _get(client, vocal_forward="false")["counts"]["tracks"]
    assert fwd > 0 and back >= 0 and fwd + back == len(_measured())

    d = _get(client, group=gid)
    assert d["counts"]["tracks"] == 5
    assert d["pairs"]["scored"] == 4 * 3        # pairs with BOTH sides in the group
    assert d["pairs"]["rated"] == 2 and d["pairs"]["loved"] == 2
    assert d["pairs"]["stale_tracks"] == 3


def test_bad_filters(app):
    client, _, _ = app
    assert client.get(URL, params={"camelot": "13A"}).status_code == 400
    assert client.get(URL, params={"group": 99999}).status_code == 404


def test_one_pass_per_table_under_300ms(app):
    client, models, _ = app
    from analysis.library_stats import clean_filters, library_stats
    library_stats(clean_filters())                      # warm the page cache
    best = min(_timed(lambda: library_stats(clean_filters())) for _ in range(3))
    assert best < 0.3, f"library-stats took {best * 1000:.0f} ms on {N} tracks"
    best = min(_timed(lambda: library_stats(clean_filters(genre="House")))
               for _ in range(3))
    assert best < 0.3


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_an_empty_library_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "empty.db"))
    import database.models as models
    from analysis.library_stats import clean_filters, library_stats
    models.init_db(tmp_path / "empty.db")
    d = library_stats(clean_filters(), db_path=tmp_path / "empty.db")
    assert d["counts"] == {"tracks": 0, "analysed": 0, "with_stems": 0, "with_tags": 0}
    assert len(d["camelot_wheel"]) == 24 and len(d["bpm_hist"]) == 70
    assert d["energy_hist"] == [] and d["scatter"] == []
    assert d["pairs"] == {"scored": 0, "stale_tracks": 0, "rated": 0,
                          "loved": 0, "rejected": 0}
