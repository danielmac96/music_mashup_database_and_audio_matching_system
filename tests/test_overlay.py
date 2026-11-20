"""Track detail's ghost-slot overlay (master plan Phase 4; readme §5.7, §5.12):
GET /api/tracks/{bed}/overlay prices every landing of a vocal's phrase on a bed
section with the scorer's own functions. The library run's landing is
score_pair's row to the digit (criterion 4.4); every landing's fit band is
`place`'s own per-bar error; the payload is cached by pair key + fingerprint;
and the play spans every consumer reads are one definition."""
import importlib
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _synth():
    spec = importlib.util.spec_from_file_location(
        "synth_library", ROOT / "scripts" / "synth_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlay")
    manifest = _synth().generate(out, 60, seed=11, planted=6)
    mp = pytest.MonkeyPatch()
    mp.setenv("MASHUP_DB_PATH", str(out / "mashup.db"))
    mp.setenv("MASHUP_AUDIO_ROOT", str(out / "audio"))
    mp.setenv("MASHUP_DATA_DIR", str(out / "data"))
    mp.setenv("MASHUP_SETTINGS_DIR", str(out / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    import matcher.match as match
    importlib.reload(match)
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.routes.tracks as tracks
    importlib.reload(tracks)
    import api.server as server
    importlib.reload(server)
    yield {"m": m, "match": match, "tracks": tracks, "manifest": manifest,
           "client": TestClient(server.app), "out": out}
    mp.undo()


def _planted(lib):
    return lib["manifest"]["planted"]


# ── the one definition of where a placement starts sounding ─────────────────

def test_play_spans_start_the_vocal_at_its_pickup_and_the_bed_under_it():
    from matcher.phrase_align import play_spans
    pl = {"alignment_downbeat": 40.0, "bed_landing_sec": 30.0,
          "vocal_phrase_start": 39.5, "timing_ok": 1}
    got = play_spans(pl, 1.02)
    assert got == {"vocal_play_start": 39.5, "bed_play_start": round(30.0 - 0.5 * 1.02, 4),
                   "play_lead": 0.5}
    # A late entry (onset after the downbeat) starts on the bar, not after it.
    assert play_spans({**pl, "vocal_phrase_start": 40.2}, 1.0)["play_lead"] == 0.0
    # The bed cannot start before its own file: the lead shrinks to fit.
    early = play_spans({**pl, "bed_landing_sec": 0.2}, 1.0)
    assert early["bed_play_start"] == 0.0 and early["play_lead"] == 0.2
    assert early["vocal_play_start"] == 39.8
    # No placement: the sections play. A landing that fails the timing gate
    # is never stored; in the overlay it was chosen by ear, and plays there.
    assert play_spans({**pl, "bed_landing_sec": None}, 1.0) is None
    assert play_spans({**pl, "timing_ok": 0}, 1.02) == got


def test_a_forced_landing_is_a_landing_and_the_default_is_untouched(lib):
    from matcher import phrase_align as pa
    p = _planted(lib)[0]
    ctx = lib["match"].pair_context(p["vocal_song_id"], p["inst_song_id"], lib["m"].DB_PATH)
    so = ctx["sections_of"]
    v = next(s for s in so(p["vocal_song_id"]) if s["section_index"] == p["vocal_section"])
    b = next(s for s in so(p["inst_song_id"]) if s["section_index"] == p["inst_section"])
    t = ctx["timing_ctx"]
    kw = dict(phrases=t["vocal_phrases"], tol_ms=t["tol_ms"], ratios=t["ratios"])
    default = pa.place(v, b, ctx["stretch"], **kw)
    assert default == pa.place_cached(v, b, ctx["stretch"], cache={}, **kw)
    dbs = pa._downbeats(b)
    for k in (0, 1, 2, 4):
        forced = pa.place(v, b, ctx["stretch"], landings=(k,), **kw)
        assert forced["bed_landing_sec"] == round(dbs[k], 4)
    opts = pa.vocal_phrase_options(v, t["vocal_phrases"])
    assert opts[0] == pa.pick_vocal_phrase(v, t["vocal_phrases"])
    if len(opts) > 1:
        second = pa.place(v, b, ctx["stretch"], phrase=opts[1], **kw)
        assert second["alignment_downbeat"] == opts[1]["downbeat"]


# ── criterion 4.4: the overlay's score is score_pair's ───────────────────────

def test_the_library_landing_is_score_pairs_row_to_the_digit(lib):
    from matcher import overlay
    checked = 0
    for p in _planted(lib):
        rows = lib["match"].score_pair(p["vocal_song_id"], p["inst_song_id"],
                                       db_path=lib["m"].DB_PATH)
        for r in rows:
            if r.get("vocal_section_idx") is None:
                continue
            got = overlay.build(p["inst_song_id"], p["vocal_song_id"],
                                r["vocal_section_idx"], r["inst_section_idx"])
            row = next(x for x in got["placements"] if x["offset_bars"] == 0)
            assert row["score_total"] == r["total"]
            # ...and every column of the row score_pair stored for it.
            key = (p["vocal_song_id"], p["inst_song_id"], r["vocal_section_idx"],
                   r["inst_section_idx"])
            stored = lib["m"].candidates_for_pairs([key], db_path=lib["m"].DB_PATH)[key]
            for k in lib["m"].CANDIDATE_COLUMNS:
                if k in ("id", "scorer", "created_at", "model_version"):
                    continue
                assert row.get(k) == stored.get(k), (k, row.get(k), stored.get(k))
            checked += 1
    assert checked >= 6


def test_every_landing_is_priced_and_its_band_is_places_own_error(lib):
    from matcher import overlay
    p = _planted(lib)[0]
    got = overlay.build(p["inst_song_id"], p["vocal_song_id"], p["vocal_section"],
                        p["inst_section"])
    offs = [x["offset_bars"] for x in got["placements"]]
    assert 0 in offs and len(offs) >= 4 and offs == sorted(offs)
    for x in got["placements"]:
        assert x["score_total"] is not None and isinstance(x["gated"], bool)
        assert x["gated"] == (x["timing_ok"] == 0)
        errs = [b["err_ms"] for b in x["band"] if b["err_ms"] is not None]
        if errs and x["timing_err_max_ms"] is not None:
            assert max(errs) == pytest.approx(x["timing_err_max_ms"], abs=0.06)
        for b in x["band"]:
            assert b["level"] in (None, "good", "ok", "bad")
        # The incoming phrase starts on the bed's timeline (on_bed), from the
        # placement's own downbeat on.
        d_v, land = x["alignment_downbeat"], x["bed_landing_sec"]
        assert x["incoming_phrases"][0]["at"] == pytest.approx(land, abs=1e-3) \
            or x["incoming_phrases"][0]["at"] > land
        assert overlay.on_bed(d_v, d_v, land, got["stretch"]) == pytest.approx(land)
    zero = next(x for x in got["placements"] if x["offset_bars"] == 0)
    assert zero["spans"] is not None and not zero["gated"]
    assert all(x["spans"] is not None for x in got["placements"])


def test_a_planted_drop_ins_phrases_land_on_the_originals(lib):
    """Criterion 4.5's measure, on every planted drop-in: the incoming phrase
    starts at the library's landing, drawn on the bed's timeline, against the
    bed's original phrase starts — within one 46 ms analysis frame."""
    from matcher import overlay
    within = total = 0
    for p in _planted(lib):
        got = overlay.build(p["inst_song_id"], p["vocal_song_id"], p["vocal_section"],
                            p["inst_section"])
        zero = next(x for x in got["placements"] if x["offset_bars"] == 0)
        orig = [q["at"] for q in got["bed_phrases"]]
        lo, hi = got["bed_section"]["start"] - 1e-3, got["bed_section"]["end"] + 1e-3
        for q in zero["incoming_phrases"]:
            if lo <= q["at"] <= hi:
                total += 1
                within += min(abs(u - q["at"]) for u in orig) <= got["frame_secs"]
    assert total >= 6 and within / total >= 0.9, (within, total)


def test_another_phrase_is_its_own_set_of_landings(lib):
    from matcher import overlay
    for p in _planted(lib):
        got = overlay.build(p["inst_song_id"], p["vocal_song_id"], p["vocal_section"],
                            p["inst_section"], phrase=1)
        if len(got["phrases"]) < 2:
            continue
        assert got["phrase"] == 1
        d = got["phrases"][1]["downbeat"]
        assert all(x["alignment_downbeat"] == d for x in got["placements"])
        return
    pytest.skip("no planted vocal section with two phrases")


# ── the route ────────────────────────────────────────────────────────────────

def test_the_route_serves_the_overlay_with_waveforms_and_playback_terms(lib):
    p = _planted(lib)[1]
    url = (f"/api/tracks/{p['inst_song_id']}/overlay?vocal={p['vocal_song_id']}"
           f"&vs={p['vocal_section']}&bs={p['inst_section']}")
    r = lib["client"].get(url)
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("placements", "waves", "bed_sections", "bed_phrases", "bed_section",
              "vocal_section", "view", "stretch", "tol_ms", "phrases", "current",
              "bed", "vocal", "bed_total", "vocal_total", "frame_secs"):
        assert k in d, k
    assert set(d["waves"]) == {"bed", "ghost", "vocal"}
    assert len(d["waves"]["bed"]) == 360
    assert d["current"] == 0
    row = next(x for x in d["placements"] if x["offset_bars"] == 0)
    for k in ("semitone_shift", "stretch_factor", "recipe", "confidence_tier",
              "dropin", "dropin_reason", "slot_reason", "score_percentile",
              "vocal_play_start", "bed_play_start"):
        assert k in row, k
    assert row["vocal_play_start"] == row["spans"]["vocal_play_start"]
    # offset_bars names the placement asked for; one that does not exist
    # falls back to the library's landing.
    other = next(x["offset_bars"] for x in d["placements"] if x["offset_bars"] != 0)
    assert lib["client"].get(url + f"&offset_bars={other}").json()["current"] == other
    assert lib["client"].get(url + "&offset_bars=999").json()["current"] == 0
    assert lib["client"].get(url.replace(f"vs={p['vocal_section']}", "vs=4242")).status_code == 404
    # Gzipped when the browser asks (it always does), the same JSON either way.
    import gzip
    import json
    raw = lib["client"].get(url, headers={"Accept-Encoding": "gzip"})
    assert raw.headers.get("content-encoding") == "gzip"
    plain = lib["client"].get(url, headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    assert plain.json() == d and raw.json() == d
    assert len(gzip.compress(plain.content)) < len(plain.content) / 3
    assert json.loads(plain.content) == d


def test_the_route_is_cached_by_pair_key_and_fingerprint(lib, monkeypatch):
    from database import alignment
    from matcher import overlay
    p = _planted(lib)[2]
    url = (f"/api/tracks/{p['inst_song_id']}/overlay?vocal={p['vocal_song_id']}"
           f"&vs={p['vocal_section']}&bs={p['inst_section']}")
    calls = []
    real = overlay.build
    monkeypatch.setattr(overlay, "build", lambda *a, **k: calls.append(a) or real(*a, **k))
    c = lib["client"]
    first = c.get(url).json()
    t = time.perf_counter()
    again = c.get(url).json()
    assert (time.perf_counter() - t) < 0.15
    assert again["placements"] == first["placements"] and len(calls) == 1
    # A write to anything it reads re-prices it: the vocal's phrases...
    phrases = alignment.get_vocal_phrases(p["vocal_song_id"], db_path=lib["m"].DB_PATH)
    alignment.replace_vocal_phrases(p["vocal_song_id"], "unified_4stem", "allin1",
                                    phrases, db_path=lib["m"].DB_PATH)
    c.get(url)
    assert len(calls) == 2
    # ...and a live scoring setting, which is not in the database.
    monkeypatch.setenv("MASHUP_SLOT_WEIGHT", "0.31")
    c.get(url)
    assert len(calls) == 3
    c.get(url)
    assert len(calls) == 3


def test_listing_rows_carry_the_play_spans(lib):
    from matcher.phrase_align import play_spans
    p = _planted(lib)[3]
    lib["match"].score_pair(p["vocal_song_id"], p["inst_song_id"], db_path=lib["m"].DB_PATH)
    got = lib["client"].get("/api/mashups", params={
        "vocal_song_id": p["vocal_song_id"], "inst_song_id": p["inst_song_id"],
        "limit": 20, "max_per_song": 0}).json()["candidates"]
    assert got
    for r in got:
        want = play_spans(r, r["stretch_factor"] or 1.0)
        assert r["vocal_play_start"] == (want or {}).get("vocal_play_start")
        assert r["bed_play_start"] == (want or {}).get("bed_play_start")
    assert any(r["vocal_play_start"] is not None for r in got)
