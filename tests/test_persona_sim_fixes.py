"""Regressions for what the 100-persona browser simulation found (readme §9).

P0: a scoped pair list stopping at three rows, Studio conforming pairs to a
stem's mis-tracked tempo, snapshots that could be saved but never loaded, the
status pill drawn over buttons, and the judged counter stuck at zero.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "sim.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


def _app(*mods):
    """A FastAPI app over freshly reloaded route modules — paths bind at
    import, so a route reads the test DB only after config and models reload."""
    import importlib
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    for name, prefix in mods:
        mod = importlib.reload(importlib.import_module(f"api.routes.{name}"))
        app.include_router(mod.router, prefix=prefix)
    models.init_db(models.DB_PATH)
    return TestClient(app), models


def _side(sid, bpm=126.0, camelot="8A"):
    return {"song_id": sid, "title": f"S{sid}", "artist": "A", "bpm": bpm,
            "camelot": camelot, "loudness_rms": 0.1, "energy": 0.5}


@pytest.fixture()
def one_vocal_many_beds(db_path):
    from database.models import init_db, upsert_candidate, upsert_song
    init_db(db_path)
    ids = [upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200, "House",
                       status="analysed", db_path=db_path) for n in range(9)]
    vocal = ids[0]
    for n, bed in enumerate(ids[1:]):
        upsert_candidate(_side(vocal), _side(bed),
                         {"total": 0.9 - n * 0.01, "bpm_score": 1.0, "key_score": 1.0,
                          "energy_score": 0.5, "timbre_score": 0.5},
                         db_path=db_path)
    return db_path, vocal, ids[1:]


# ── 1. the scoped list is not capped by its own track ───────────────────────

def test_beds_for_one_vocal_are_not_capped_by_that_vocal(one_vocal_many_beds):
    """The per-song cap counted the scoped track, which is on every row, so
    "beds for this vocal" stopped at three. The beds are still capped."""
    from database.models import get_candidates_enriched
    db, vocal, beds = one_vocal_many_beds
    rows = get_candidates_enriched(vocal_song_id=vocal, max_per_song=3,
                                   limit=40, db_path=db)
    assert len(rows) == len(beds)
    unscoped = get_candidates_enriched(max_per_song=3, limit=40, db_path=db)
    assert len(unscoped) == 3, "the library-wide list still caps every song"


def test_cap_exempt_is_explicit():
    from database.models import _cap_per_song
    rows = [{"vocal_song_id": 1, "inst_song_id": b} for b in range(2, 9)]
    assert len(_cap_per_song(rows, 3, 40)) == 3
    assert len(_cap_per_song(rows, 3, 40, exempt={1})) == 7


# ── 2. Studio takes tempo from the full mix ─────────────────────────────────

def test_studio_lane_tempo_is_the_full_mix_tempo():
    """laneBpmFor read the vocal stem's own BPM whenever its confidence cleared
    0.35 and the instrumental stem's unconditionally; a vocal tracked at a
    quarter of the tempo set the project to 31 BPM against a 124 BPM plan."""
    studio = _read("components/MixStudio.jsx")
    fn = studio[studio.index("function laneBpmFor"):]
    fn = fn[:fn.index("\n}\n")]
    assert "feats.full?.bpm ??" in fn
    assert "bpm_confidence" not in fn
    assert "VOCAL_BPM_CONFIDENCE_MIN" not in studio


def test_a_stem_beat_grid_must_agree_with_the_full_mix_tempo():
    from api.routes.tracks import _tempo_agrees
    assert _tempo_agrees(124.0, 124.0)
    assert _tempo_agrees(123.0, 124.0)
    assert not _tempo_agrees(31.0, 124.0)
    assert not _tempo_agrees(62.0, 124.0)
    assert _tempo_agrees(62.0, None), "no full-mix tempo to disagree with"


def test_waveform_route_falls_back_to_the_full_grid_on_a_wrong_stem_tempo(db_path):
    c, m = _app(("tracks", "/api/tracks"))
    sid = m.upsert_song("T", "A", "https://sc/t", 200, "House", db_path=m.DB_PATH)
    full_beats = [i * 60 / 124 for i in range(40)]
    m.upsert_features(sid, "full", {"bpm": 124.0, "bpm_confidence": 0.9,
                                    "beat_times": full_beats}, db_path=m.DB_PATH)
    m.upsert_features(sid, "vocals", {"bpm": 31.0, "bpm_confidence": 0.95,
                                      "beat_times": full_beats[::4]}, db_path=m.DB_PATH)
    m.upsert_features(sid, "instrumental", {"bpm": 124.0, "bpm_confidence": 0.9,
                                            "beat_times": full_beats}, db_path=m.DB_PATH)
    v = c.get(f"/api/tracks/{sid}/waveform", params={"stem": "vocals"}).json()
    assert v["beat_source"] == "full" and len(v["beat_times"]) == 40
    i = c.get(f"/api/tracks/{sid}/waveform", params={"stem": "instrumental"}).json()
    assert i["beat_source"] == "instrumental"


# ── 3. snapshots can be loaded ──────────────────────────────────────────────

def test_snapshots_are_listed_and_loadable():
    studio = _read("components/MixStudio.jsx")
    rail = _read("components/StudioRail.jsx")
    assert "readSnapshots" in studio and "loadSnapshot" in studio
    assert "deleteSnapshot" in studio
    assert "onLoadSnapshot(sn)" in rail and "snap-list" in rail
    # One serialisation for the saved project, snapshots and undo.
    assert studio.count("lanes.map(laneState)") + studio.count(".map(laneState)") >= 2


# ── 4. the status readout cannot cover anything ─────────────────────────────

def test_status_readout_lives_in_the_rail():
    app = _read("App.jsx")
    main = app[app.index('<div className="app-main">'):]
    assert "float-status" not in main, "nothing may float over the screens"
    assert "status={" in app and "rail-status" in _read("shell/Sidebar.jsx")


# ── 5. the judged counter counts ────────────────────────────────────────────

def test_scorer_status_counts_judgements_without_a_model(db_path):
    c, m = _app(("mashups", "/api/mashups"))
    m.upsert_pair_feedback(1, 2, "love", rating=5, db_path=m.DB_PATH)
    m.upsert_pair_feedback(1, 3, "no", rating=1, db_path=m.DB_PATH)
    s = c.get("/api/mashups/scorer-status").json()
    assert s["scorer"] == "heuristic" and s["n_judgments"] == 2


def test_rail_counter_follows_the_live_ratings():
    assert "judged={ratings.count}" in _read("App.jsx")
    assert "judged ?? scorer?.n_judgments" in _read("shell/Sidebar.jsx")


# ── 6. sets: chosen mashups in running order ────────────────────────────────

def _item(iid, bpm, cam, start=0.0, end=16.0, **kw):
    return {"item_id": iid, "position": iid, "target_bpm": bpm, "vocal_camelot": cam,
            "inst_camelot": cam, "vocal_section_start": start, "vocal_section_end": end,
            "vocal_song_id": iid, "inst_song_id": 100 + iid, **kw}


def test_transitions_are_graded_on_tempo_and_key():
    from matcher.setflow import transition
    assert transition(_item(1, 128, "8A"), _item(2, 128, "9A"))["grade"] == "smooth"
    assert transition(_item(1, 128, "8A"), _item(2, 126, "8B"))["grade"] == "smooth"
    assert transition(_item(1, 124, "8A"), _item(2, 128, "10A"))["grade"] == "workable"
    assert transition(_item(1, 124, "8A"), _item(2, 128, "2A"))["grade"] == "jump"
    # Half time is not a tempo change.
    assert transition(_item(1, 87, "8A"), _item(2, 174, "8A"))["tempo_pct"] == 0.0
    assert transition(_item(1, None, "8A"), _item(2, 128, "8A"))["grade"] == "unknown"


def test_flow_adds_up_running_time_and_start_times():
    from matcher.setflow import flow
    f = flow([_item(1, 128, "8A", 0, 15), _item(2, 128, "8A", 30, 60)])
    assert f["total_secs"] == 45 and f["starts"] == [0, 15]
    assert f["grades"]["smooth"] == 1


def test_suggested_order_keeps_moves_small():
    from matcher.setflow import flow, suggest_order
    items = [_item(1, 128, "8A"), _item(2, 128, "3A"), _item(3, 128, "9A"),
             _item(4, 128, "4A"), _item(5, 128, "10A")]
    ids = suggest_order(items, start_item_id=1)
    assert ids[:3] == [1, 3, 5]
    by = {i["item_id"]: i for i in items}
    assert flow([by[i] for i in ids])["grades"]["jump"] <= flow(items)["grades"]["jump"]


def test_set_items_are_keyed_by_the_four_ids_and_survive_a_rescore(one_vocal_many_beds):
    from database import models
    db, vocal, beds = one_vocal_many_beds
    s = models.create_set("Vol 1", db_path=db)
    assert models.create_set("Vol 1", db_path=db)["name"] == "Vol 1 2"
    r = models.add_set_item(s["id"], vocal, beds[0], db_path=db)
    assert r["duplicate"] is False
    assert models.add_set_item(s["id"], vocal, beds[0], db_path=db)["duplicate"] is True
    models.add_set_item(s["id"], vocal, beds[1], db_path=db)
    got = models.get_set(s["id"], db_path=db)
    assert [i["inst_song_id"] for i in got["items"]] == beds[:2]
    assert got["items"][0]["inst_title"] == f"S{beds[0]}" and not got["items"][0]["stale"]
    # A re-score truncates mashup_candidates: the items keep their frozen rows.
    models.clear_candidates(db_path=db)
    got = models.get_set(s["id"], db_path=db)
    assert all(i["stale"] for i in got["items"])
    assert got["items"][1]["inst_title"] == f"S{beds[1]}"
    ids = [i["item_id"] for i in got["items"]]
    assert models.reorder_set(s["id"], ids[::-1], db_path=db)
    assert not models.reorder_set(s["id"], ids[:1], db_path=db)
    assert [i["item_id"] for i in models.get_set(s["id"], db_path=db)["items"]] == ids[::-1]
    assert models.remove_set_item(s["id"], ids[0], db_path=db)
    assert len(models.get_set(s["id"], db_path=db)["items"]) == 1


def test_pair_notes_round_trip_and_clear(db_path):
    from database import models
    models.init_db(db_path)
    models.set_pair_note(1, 2, 0, 3, "opener", db_path=db_path)
    models.set_pair_note(1, 2, 0, 3, "opener, needs a riser", db_path=db_path)
    notes = models.get_pair_notes(db_path=db_path)
    assert len(notes) == 1 and notes[0]["note"] == "opener, needs a riser"
    models.set_pair_note(1, 2, 0, 3, "", db_path=db_path)
    assert models.get_pair_notes(db_path=db_path) == []


def test_exports_csv_cue_and_rekordbox():
    import xml.etree.ElementTree as ET
    from render.exports import CSV_COLUMNS, cue_sheet, pairs_csv, rekordbox_xml
    items = [
        _item(1, 128, "8A", 30, 45, vocal_title="Vox", vocal_artist="Singer",
              inst_title="Bed", inst_artist="Producer", inst_section_start=60,
              inst_section_end=75, vocal_section_label="chorus",
              inst_section_label="drop", harmonic_shift=2, score_key=0.9,
              note="opener"),
        _item(2, 128, "9A", 10, 25, vocal_title="Vox2", inst_title="Bed2",
              inst_section_start=5, inst_section_end=20),
    ]
    csv_text = pairs_csv(items)
    head, first = csv_text.splitlines()[:2]
    assert head.split(",") == list(CSV_COLUMNS)
    assert "Singer,Vox,chorus,0:30,0:45" in first and ",+2," in first and "opener" in first
    cue = cue_sheet(items, "Vol 1")
    assert "[ 0:00]  1. Singer - Vox" in cue and "↓ smooth" in cue and "[ 0:15]" in cue

    def resolve(sid, stem):
        return {"path": f"/data/audio/{stem}/{sid}.wav", "title": f"T{sid}",
                "artist": "A", "bpm": 128.0, "key": "A", "mode": "minor",
                "duration": 200, "beat_times": [0.1, 0.57, 1.04, 1.51],
                "beat_phase": 1, "sections": [{"label": "drop", "start_sec": 60}]}
    out = rekordbox_xml(items, "Vol 1", resolve, base="/Users/me/data/audio",
                        root="/data/audio")
    root = ET.fromstring(out["xml"])
    tracks = root.findall("./COLLECTION/TRACK")
    assert len(tracks) == 4 and out["skipped"] == 0
    t0 = tracks[0]
    assert t0.get("Name") == "T1 (Acapella)" and t0.get("Tonality") == "Am"
    assert t0.get("Location") == "file://localhost/Users/me/data/audio/vocals/1.wav"
    assert t0.find("TEMPO").get("Inizio") == "0.570"
    cues = [m for m in t0.findall("POSITION_MARK") if m.get("Num") == "0"]
    assert cues and cues[0].get("Start") == "30.000"
    playlist = root.find("./PLAYLISTS/NODE/NODE")
    assert playlist.get("Name") == "Vol 1" and playlist.get("Entries") == "4"


def test_sets_routes_add_reorder_and_export(db_path):
    c, m = _app(("sets", "/api/sets"), ("mashups", "/api/mashups"))
    ids = [m.upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200, "House",
                         status="analysed", db_path=m.DB_PATH) for n in range(3)]
    for bed in ids[1:]:
        m.upsert_candidate(_side(ids[0]), _side(bed),
                           {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0,
                            "energy_score": 0.5, "timbre_score": 0.5}, db_path=m.DB_PATH)
    s = c.post("/api/sets", json={"name": "Vol 1"}).json()
    for bed in ids[1:]:
        r = c.post(f"/api/sets/{s['id']}/items",
                   json={"vocal_song_id": ids[0], "inst_song_id": bed}).json()
    assert len(r["items"]) == 2 and "flow" in r
    assert r["items"][0]["semitone_shift"] == 0, "items carry the dock's playback terms"
    order = [i["item_id"] for i in r["items"]][::-1]
    assert c.post(f"/api/sets/{s['id']}/reorder", json={"item_ids": order}).status_code == 200
    assert c.post(f"/api/sets/{s['id']}/reorder", json={"item_ids": order[:1]}).status_code == 400
    csv_r = c.get(f"/api/sets/{s['id']}/export", params={"format": "csv"})
    assert csv_r.status_code == 200 and "attachment" in csv_r.headers["content-disposition"]
    assert c.get(f"/api/sets/{s['id']}/export", params={"format": "cue"}).status_code == 200
    rb = c.get(f"/api/sets/{s['id']}/export", params={"format": "rekordbox"})
    assert rb.status_code == 200 and rb.headers["x-skipped-tracks"] == "4", "no audio on disk"
    # The dock's export: the same files, from the pairs on screen.
    ex = c.post("/api/mashups/export", json={"format": "csv", "pairs": [
        {"vocal_song_id": ids[0], "inst_song_id": ids[1]}]})
    assert ex.status_code == 200 and ex.text.count("\n") == 2
    assert c.delete(f"/api/sets/{s['id']}").json()["ok"]


def test_sets_screen_is_wired():
    app, nav = _read("App.jsx"), _read("shell/Sidebar.jsx")
    assert '["sets", "Sets"' in nav and 'route === "sets"' in app
    assert "onAddToSet: sets.addPair" in app
    dock = _read("hooks/usePairDock.js")
    assert 'e.key === "a"' in dock and "onAddToSet(row)" in dock
    assert "pc-addset" in _read("components/PairCard.jsx")
    studio = _read("components/MixStudio.jsx")
    assert "const layPairs" in studio and "seed?.chain" in studio
    assert "onAppendNext" in _read("components/StudioRail.jsx")


# ── 7 / 10. dock: keepers, landing key, section type ────────────────────────

@pytest.fixture()
def keyed_pairs(db_path):
    from database import models
    models.init_db(db_path)
    ids = [models.upsert_song(f"S{n}", "A", f"https://sc/{n}", 200, "House",
                              status="analysed", db_path=db_path) for n in range(6)]
    keys = ["8A", "9A", "3A", "8B", "12A"]
    for n, bed in enumerate(ids[1:]):
        models.upsert_candidate(_side(bed, camelot=keys[n]), _side(ids[0]),
                                {"total": 0.9 - n * 0.01, "bpm_score": 1.0, "key_score": 1.0,
                                 "energy_score": 0.5, "timbre_score": 0.5}, db_path=db_path)
    conn = models.get_conn(db_path)
    conn.execute("UPDATE mashup_candidates SET vocal_section_idx=1, inst_section_idx=2, "
                 "vocal_section_label='chorus', inst_section_label="
                 "CASE WHEN vocal_song_id % 2 = 0 THEN 'drop' ELSE 'breakdown' END")
    conn.commit(); conn.close()
    models.upsert_pair_feedback(ids[1], ids[0], "love", 1, 2, rating=5, db_path=db_path)
    models.upsert_pair_feedback(ids[2], ids[0], "no", rating=1, db_path=db_path)  # no sections
    return db_path, ids


def test_keepers_and_unrated(keyed_pairs):
    from database.models import get_candidates_enriched
    db, ids = keyed_pairs
    v = lambda **kw: sorted(r["vocal_song_id"] for r in get_candidates_enriched(limit=50, db_path=db, **kw))
    assert v(rated="loved") == [ids[1]]
    assert v(rated="rated") == [ids[1], ids[2]], "a section-less verdict counts for the pair"
    assert v(rated="unrated") == ids[3:]
    with pytest.raises(ValueError):
        get_candidates_enriched(rated="nope", db_path=db)


def test_landing_key_with_tolerance(keyed_pairs):
    from database.models import get_candidates_enriched
    db, ids = keyed_pairs
    cams = lambda **kw: sorted(r["vocal_camelot"] for r in get_candidates_enriched(limit=50, db_path=db, **kw))
    assert cams(key="8A") == ["8A", "8B"], "the relative minor/major is the same place"
    assert cams(key="8A", key_tolerance=1) == ["8A", "8B", "9A"]
    assert cams(key="9A", key_tolerance=6) == ["12A", "3A", "8A", "8B", "9A"]
    with pytest.raises(ValueError):
        get_candidates_enriched(key="Z9", db_path=db)


def test_section_type_filter_and_options(keyed_pairs):
    from database.models import candidate_filter_options, get_candidates_enriched
    db, ids = keyed_pairs
    rows = get_candidates_enriched(limit=50, vocal_label="chorus", inst_label="drop", db_path=db)
    assert rows and all(r["inst_section_label"] == "drop" for r in rows)
    opts = candidate_filter_options(db_path=db)
    assert opts["vocal_labels"] == ["chorus"] and set(opts["inst_labels"]) == {"drop", "breakdown"}


def test_dock_offers_keepers_key_and_section_filters():
    dock = _read("components/PairDock.jsx")
    assert "pd-keepers" in dock and 'rated: filters.rated === "loved"' in dock
    assert "Lands in" in dock and "Bed part" in dock and "Vocal part" in dock
    assert "api.exportPairs(rows.slice(0, exportN)" in dock
    api_js = _read("api.js")
    assert 'params.set("rated", rated)' in api_js and 'params.set("key_tolerance"' in api_js


# ── 9. mixes: the documented pairs as the engine sees them ──────────────────

def test_mix_detail_scores_and_ranks_documented_pairs(db_path):
    c, m = _app(("mixes", "/api/mixes"))
    ids = [m.upsert_song(f"S{n}", "A", f"https://sc/{n}", 200, "House",
                         status="analysed", db_path=m.DB_PATH) for n in range(4)]
    vocal = ids[0]
    for n, bed in enumerate(ids[1:3]):
        m.upsert_candidate(_side(vocal), _side(bed),
                           {"total": 0.9 - n * 0.1, "bpm_score": 1.0, "key_score": 1.0,
                            "energy_score": 0.5, "timbre_score": 0.5}, db_path=m.DB_PATH)
    conn = m.get_conn(m.DB_PATH)
    conn.execute("INSERT INTO mixes(title, source_url) VALUES('Vol', 'https://x/1')")
    rows = [(0, 1, 0, ids[2]), (None, 2, 1, vocal), (1, 3, 0, ids[3]), (None, 4, 1, None)]
    for entry, pos, ov, sid in rows:
        conn.execute("INSERT INTO mix_tracks(mix_id, entry_index, position, is_overlay, title, song_id)"
                     " VALUES(1, ?, ?, ?, ?, ?)", (entry, pos, ov, f"T{pos}", sid))
    conn.execute("INSERT INTO mashup_pairs(mix_id, inst_mix_track_id, vocal_mix_track_id) VALUES(1, 1, 2)")
    conn.execute("INSERT INTO mashup_pairs(mix_id, inst_mix_track_id, vocal_mix_track_id) VALUES(1, 3, 4)")
    conn.commit(); conn.close()
    pairs = c.get("/api/mixes/1").json()["pairs"]
    assert pairs[0]["engine_state"] == "scored"
    assert pairs[0]["engine_rank"] == 2 and pairs[0]["engine_field"] == 2
    assert pairs[0]["engine"]["semitone_shift"] == 0, "playable in the shared player"
    assert pairs[1]["engine"] is None and pairs[1]["engine_state"] == "not in library"


def test_mixes_screen_can_play_open_and_find_similar():
    mixes, app = _read("components/MixImporter.jsx"), _read("App.jsx")
    assert "function DocumentedPairs" in mixes and "onFindSimilar(p.vocal_song_id)" in mixes
    assert "<MixImporter player={player} onOpenStudio={pairToStudio}" in app


# ── 8. Studio picker: ranked by fit, matcher-scored layers ─────────────────

def test_studio_picker_ranks_by_fit_and_offers_scored_layers():
    studio = _read("components/MixStudio.jsx")
    assert "const fitOf" in studio and ".sort((x, y) => (x.fit?.cost" in studio
    assert "Second vocal over this bed" in studio and "const addLayer" in studio
    assert "Import tab" not in studio


# ── 12 / 13. pair card and library: what the scores are made of ────────────

def test_pair_card_shows_artists_song_terms_and_loop_note():
    card, model = _read("components/PairCard.jsx"), _read("components/pairs/pairModel.js")
    assert "artist={c.vocal_artist}" in card and "pc-artist" in card
    assert "export const SONG_TERMS" in model and "songTermsOf(candidate)" in card
    for key in ("score_bpm", "score_key", "score_energy", "score_collision"):
        assert key in model
    assert "loop bed ×{c.section_loop_repeats}" in card
    # A measured harmony makes the Camelot lookup context, not a verdict.
    assert 'h.known ? " muted"' in card


def test_library_summarises_each_track_as_mashup_material(one_vocal_many_beds):
    c, m = _app(("tracks", "/api/tracks"))
    vocal = m.upsert_song("V", "A", "https://sc/v", 64, "House", status="analysed",
                          db_path=m.DB_PATH)
    beds = [m.upsert_song(f"B{n}", "A", f"https://sc/b{n}", 64, "House",
                          status="analysed", db_path=m.DB_PATH) for n in range(3)]
    for n, bed in enumerate(beds):
        m.upsert_candidate(_side(vocal), _side(bed),
                           {"total": 0.9 - n * 0.2, "bpm_score": 1.0, "key_score": 1.0,
                            "energy_score": 0.5, "timbre_score": 0.5}, db_path=m.DB_PATH)
    m.replace_sections(vocal, [
        {"start_sec": 0, "end_sec": 16, "label": "intro", "energy": 0.2,
         "vocal_presence": 0.0, "section_class": "instrumental"},
        {"start_sec": 16, "end_sec": 64, "label": "chorus", "energy": 0.9,
         "vocal_presence": 0.8, "section_class": "vocal"},
    ], db_path=m.DB_PATH)
    rows = {t["id"]: t for t in c.get("/api/tracks").json()["tracks"]}
    v = rows[vocal]
    assert v["mash"]["as_vocal"] == 3 and v["mash"]["as_bed"] == 0
    assert v["mash"]["best_pct"] == 1.0 and v["mash"]["vocal_coverage"] == 0.75
    assert v["shape"][1][:3] == [16.0, 64.0, "chorus"]
    assert rows[beds[2]]["mash"]["as_bed"] == 1 and rows[beds[2]]["mash"]["best_pct"] < 1.0
    assert rows[beds[2]]["mash"]["vocal_coverage"] is None, "unmeasured, not zero"
    table = _read("components/TrackTable.jsx")
    assert 'id: "mash"' in table and "function ShapeThumb" in table
    assert "best: (t) => t.mash?.best_pct" in _read("hooks/useLibraryFilters.js")
