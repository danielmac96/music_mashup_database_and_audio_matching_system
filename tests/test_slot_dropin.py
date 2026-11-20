"""The vacated slot, Drop-in and phrase windows (readme §5.7).

`score_slot` says how well an incoming vocal replaces the one a bed section
used to carry; Drop-in names the pairs that go together with next to nothing
done to either record; the window is the 8 or 16 bars of a pairing to build.
Each has one definition, and the SQL filter, the card and Segments must all
give that definition's answer."""
import importlib
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from matcher import compat  # noqa: E402
from matcher.phrase_align import best_window, place, slot_phrase_fit  # noqa: E402
from matcher.section_score import (  # noqa: E402
    SLOT_BLEND, slot_ceiling, slot_fit, slot_occupancy, slot_register, slot_score,
)


# ── the three parts ──────────────────────────────────────────────────────────

def _section(start, bars, bpm=120.0, song_id=1, **extra):
    bar = 240.0 / bpm
    return {"song_id": song_id, "start_sec": start, "end_sec": start + bars * bar,
            "downbeats": [start + k * bar for k in range(bars)], **extra}


def _phrases(section, every=4, pickup=0.0, shift_bars=0):
    dbs = section["downbeats"]
    bar = dbs[1] - dbs[0]
    return [{"downbeat_sec": d, "pickup_sec": pickup, "onset_sec": d + pickup,
             "end_sec": d + (every - 0.2) * bar, "bars": every}
            for d in dbs[shift_bars::every]]


def test_occupancy_and_register_read_stored_values_and_unknown_is_none():
    assert slot_occupancy({"vocal_activity": 0.8}) == 0.8
    assert slot_occupancy({"vocal_activity": 1.7}) == 1.0
    assert slot_occupancy({}) is None                       # never read: not zero
    v = {"f0": {"median_midi": 64.0}}
    assert slot_register(v, {"f0": {"median_midi": 65.0}}) == 1.0
    assert slot_register(v, {"f0": {"median_midi": 76.0}}) == 0.0     # an octave
    # The bed moves, and the range its own vocal was written in moves with it.
    assert slot_register(v, {"f0": {"median_midi": 66.0}}, shift=-2) == 1.0
    assert slot_register(v, {"f0": {"median_midi": 60.0}}, shift=-2) \
        < slot_register(v, {"f0": {"median_midi": 60.0}}, shift=0)
    assert slot_register(v, {}) is None and slot_register({}, v) is None


def test_the_blend_takes_an_unmeasured_part_as_neutral_and_the_ceiling_bounds_it():
    assert sum(SLOT_BLEND.values()) == pytest.approx(1.0)
    assert slot_score(None, None, None) == pytest.approx(0.5)
    assert slot_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
    bed = {"vocal_activity": 0.6}
    for phrase in (None, 0.0, 0.4, 1.0):
        for register in (None, 0.0, 1.0):
            assert slot_score(slot_occupancy(bed), phrase, register) <= slot_ceiling(bed)
    assert slot_fit(0.8, 0.4, 0.0) == 0.8                   # weight 0: as before


def test_phrase_match_is_high_on_the_same_bars_and_zero_a_bar_out():
    v, b = _section(10.0, 16), _section(40.0, 16, song_id=2)
    vp = _phrases(v)
    placement = place(v, b, 1.0, vp)
    same, err = slot_phrase_fit(placement, v, b, 1.0, vp, _phrases(b))
    assert same == pytest.approx(1.0) and err == pytest.approx(0.0, abs=0.01)
    # The original sang from one bar later: every start is a bar out.
    out, err_out = slot_phrase_fit(placement, v, b, 1.0, vp, _phrases(b, shift_bars=1))
    assert out == 0.0 and err_out > 1000
    # The same bars, a different pickup: still the slot, but not a perfect one.
    picked, _ = slot_phrase_fit(placement, v, b, 1.0, vp, _phrases(b, pickup=-0.5))
    assert 0.7 < picked < same


def test_phrase_match_knows_unsung_from_unmeasured():
    v, b = _section(10.0, 16), _section(40.0, 16, song_id=2)
    vp = _phrases(v)
    placement = place(v, b, 1.0, vp)
    elsewhere = _phrases(_section(200.0, 16))               # the bed sang, not here
    assert slot_phrase_fit(placement, v, b, 1.0, vp, elsewhere) == (0.0, None)
    assert slot_phrase_fit(placement, v, b, 1.0, vp, []) == (None, None)
    assert slot_phrase_fit(placement, v, b, 1.0, [], _phrases(b)) == (None, None)


def test_the_window_is_a_sung_stretch_over_the_beds_own():
    v, b = _section(10.0, 16), _section(40.0, 32, song_id=2)
    vp = _phrases(v, every=8)
    # The bed's own vocal only sang its second half.
    bp = [p for p in _phrases(b, every=8) if p["downbeat_sec"] >= b["downbeats"][16]]
    # A window is a stretch of its placement, never a placement of its own
    # (§9 item 7: the dock, Studio and the FL session play it): landed on the
    # bed's first bar, its bed bar is the one under the vocal — not where the
    # bed sang, which only a landing there can reach.
    placement = place(v, b, 1.0, vp)
    w = best_window(placement, v, b, 1.0, vp, bp)
    assert w["window_bars"] == 16 and w["vocal_window_bar"] == 0
    assert w["bed_window_bar"] == 0                         # under the vocal
    assert w["score_window"] == pytest.approx(0.7)          # tight, sung, unslotted
    there = place(v, b, 1.0, vp, landings=[16])
    w = best_window(there, v, b, 1.0, vp, bp)
    assert w["bed_window_bar"] == 16                        # where it sang
    assert 0.9 < w["score_window"] <= 1.0
    # Its seconds, for playback: the first window downbeat, the onset (the
    # phrase pickup, never after the bar) and the end, all the vocal's.
    assert w["window_vocal_start"] == pytest.approx(v["downbeats"][0])
    assert w["window_vocal_onset"] <= w["window_vocal_start"]
    assert w["window_vocal_end"] == pytest.approx(v["end_sec"], abs=1e-3)
    # No window where the grids do not hold.
    drift = _section(40.0, 32, bpm=121.5, song_id=3)
    assert best_window(place(v, drift, 1.0, vp), v, drift, 1.0, vp, []) is None
    # Nor in a section too short for one.
    short = _section(40.0, 4, song_id=4)
    assert best_window(place(v, short, 1.0, vp), v, short, 1.0, vp, []) is None


# ── key compatibility: one rule, and the Library's 8A~ is it ─────────────────

CODES = [f"{n}{letter}" for n in range(1, 13) for letter in "AB"]


def test_camelot_compatible_is_same_adjacent_or_relative():
    assert compat.camelot_compatible("8A", "8A")
    assert compat.camelot_compatible("8A", "7A") and compat.camelot_compatible("8A", "9A")
    assert compat.camelot_compatible("8A", "8B")
    assert compat.camelot_compatible("12B", "1B") and compat.camelot_compatible("1A", "12A")
    assert not compat.camelot_compatible("8A", "10A")
    assert not compat.camelot_compatible("8A", "9B")
    assert compat.camelot_compatible("8A", None) is None
    assert compat.camelot_transposed("8A", 1) == "3A"
    assert compat.camelot_transposed("8A", -1) == "1A"
    assert compat.camelot_transposed("8A", 12) == "8A"


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed here")
def test_the_librarys_key_filter_is_the_same_rule(tmp_path):
    """`8A~` in the Library search (theme.compatibleKeys) and the Drop-in rule
    (compat.camelot_compatible) may never disagree: all 24 x 24 keys, run for
    real."""
    src = (ROOT / "frontend" / "src" / "theme.js").read_text(encoding="utf-8")
    (tmp_path / "theme.mjs").write_text(src, encoding="utf-8")
    (tmp_path / "run.mjs").write_text(
        'import { compatibleKeys } from "./theme.mjs";\n'
        f"const codes = {json.dumps(CODES)};\n"
        "console.log(JSON.stringify(Object.fromEntries("
        "codes.map((c) => [c, compatibleKeys(c)]))));\n", encoding="utf-8")
    out = subprocess.run([NODE, "run.mjs"], cwd=tmp_path, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    theirs = json.loads(out.stdout)
    for a in CODES:
        ours = {b for b in CODES if compat.camelot_compatible(a, b)}
        assert set(theirs[a]) == ours, a


# ── the Drop-in rule ─────────────────────────────────────────────────────────

def _row(**over):
    row = {"dropin_mask": compat.dropin_mask("8A", "8A", 124.0, 124.5),
           "semitone_shift": 0, "timing_ok": 1, "score_slot": 0.9,
           "slot_occupancy": 0.8, "slot_phrase_match": 1.0, "slot_phrase_err_ms": 3.0,
           "inst_title": "Bed"}
    row.update(over)
    return row


def test_a_dropin_needs_every_rule_and_says_which_one_it_missed():
    ok, why = compat.dropin(_row(), 0.7)
    assert ok and "keys mix" in why and "own vocal was" in why
    cases = {
        "the bed needs": _row(semitone_shift=3),
        "keys do not mix": _row(dropin_mask=compat.dropin_mask("8A", "11A", 124, 124)),
        "tempo more than": _row(dropin_mask=compat.dropin_mask("8A", "8A", 124, 130)),
        "grid not judged": _row(timing_ok=None),
        "no vacated vocal slot": _row(score_slot=0.5),
    }
    for expect, row in cases.items():
        got, reason = compat.dropin(row, 0.7)
        assert got is False and expect in reason, (expect, reason)
    assert compat.dropin(_row(dropin_mask=None), 0.7) == (False, None)   # not scored for it
    assert compat.dropin(_row(), None)[0] is False                      # no floor yet


def test_one_semitone_of_the_bed_is_allowed_and_the_tempo_folds():
    # 8A over 1A: a semitone up on the bed lands it on 8A.
    mask = compat.dropin_mask("8A", "1A", 124.0, 62.5)       # half time, 0.8 % out
    assert mask & compat.DROPIN_TEMPO and mask & compat.DROPIN_KEY_UP
    assert not mask & compat.DROPIN_KEY_AS_IS
    assert compat.dropin(_row(dropin_mask=mask, semitone_shift=1), 0.7)[0]
    assert not compat.dropin(_row(dropin_mask=mask, semitone_shift=0), 0.7)[0]
    assert not compat.dropin(_row(dropin_mask=mask, semitone_shift=-1), 0.7)[0]
    assert not compat.dropin_mask("8A", "8A", 124.0, 128.5) & compat.DROPIN_TEMPO  # 3.5 %


def _synth():
    spec = importlib.util.spec_from_file_location(
        "synth_library", ROOT / "scripts" / "synth_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    out = tmp_path_factory.mktemp("slot")
    manifest = _synth().generate(out, 90, seed=5, planted=6)
    return out, manifest


@pytest.fixture()
def scored(library, tmp_path, monkeypatch):
    out, manifest = library
    db = tmp_path / "scored.db"
    shutil.copy(out / "mashup.db", db)
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    import matcher.match as match
    importlib.reload(match)
    mashups = importlib.reload(importlib.import_module("api.routes.mashups"))
    match.score_all_pairs(db_path=m.DB_PATH)
    return m, mashups, manifest


def _listed(mashups, **kw):
    kw.setdefault("combo_type", "vocal_over_instrumental")
    return mashups.list_candidates(limit=500, max_per_song=0, **kw)["candidates"]


def test_the_sql_filter_and_the_card_are_one_rule(scored):
    m, mashups, _manifest = scored
    conn = m.get_conn()
    n = conn.execute("SELECT COUNT(*) FROM mashup_candidates "
                     "WHERE combo_type='vocal_over_instrumental'").fetchone()[0]
    conf = sorted(r[0] for r in conn.execute(
        "SELECT harmonic_confidence FROM mashup_candidates "
        "WHERE harmonic_confidence IS NOT NULL"))
    conn.close()
    # Sure and unsure measured shifts both occur under the median confidence.
    m.set_pref("harmony_unsure_below", {"value": conf[len(conf) // 2]})
    rows = mashups._with_playback_terms(m.get_candidates_enriched(
        combo_type="vocal_over_instrumental", limit=n, max_per_song_pair=0))
    assert len(rows) == n
    flagged = {r["id"] for r in rows if r["dropin"]}
    assert 0 < len(flagged) < n
    by_sql = {r["id"] for r in m.get_candidates_enriched(
        combo_type="vocal_over_instrumental", limit=n, max_per_song_pair=0, dropin=True)}
    not_sql = {r["id"] for r in m.get_candidates_enriched(
        combo_type="vocal_over_instrumental", limit=n, max_per_song_pair=0, dropin=False)}
    assert by_sql == flagged
    assert not_sql == {r["id"] for r in rows} - flagged
    # Both ways the transpose can be decided are in the comparison.
    assert {r["harmonic_sure"] for r in rows} >= {True, False}
    for r in rows:
        if r["dropin"]:
            assert r["timing_ok"] == 1 and abs(r["semitone_shift"]) <= 1
            assert r["dropin_reason"] and r["slot_reason"]


def test_the_floor_is_the_librarys_not_the_tables(scored):
    """The stored table keeps the best rows, which are the high-slot ones; the
    Drop-in floor is the 70th percentile of a random sample of the library."""
    m, _mashups, _manifest = scored
    pref = m.get_pref("slot_dropin_floor")
    assert pref["pairings"] > 100 and 0.0 < pref["value"] < 1.0
    assert compat.slot_floor(m.DB_PATH) == pref["value"]


def test_easy_first_is_tier_then_dropin_then_score(scored):
    _m, mashups, _manifest = scored
    rows = _listed(mashups)                                  # the route's default
    assert rows == _listed(mashups, order="easy")
    tiers = list(compat.TIERS)
    keys = [(tiers.index(r["confidence_tier"]), not r["dropin"]) for r in rows]
    assert keys == sorted(keys)
    assert {r["dropin"] for r in rows} == {True, False}
    first = _listed(mashups, order="dropin")
    flags = [r["dropin"] for r in first]
    assert flags == sorted(flags, reverse=True)


def test_the_bonus_lifts_only_dropins_and_never_past_the_gate(scored, monkeypatch):
    """The bonus is an order key, never the stored total; it is added for a
    Drop-in alone, and a pair that failed the timing gate is not stored at
    all — so there is nothing for it to lift."""
    m, mashups, _manifest = scored
    monkeypatch.setenv("MASHUP_DROPIN_BONUS", "0")
    plain = _listed(mashups, order="score")
    assert [r["score_total"] for r in plain] == sorted(
        (r["score_total"] for r in plain), reverse=True)
    monkeypatch.setenv("MASHUP_DROPIN_BONUS", "0.1")
    lifted = _listed(mashups, order="score")
    was = {r["id"]: r["score_total"] for r in plain}
    shared = [r for r in lifted if r["id"] in was]
    assert len(shared) > 100
    assert all(r["score_total"] == was[r["id"]] for r in shared)   # the total is untouched
    key = [r["score_total"] + (0.1 if r["dropin"] else 0.0) for r in lifted]
    assert key == sorted(key, reverse=True)
    assert [r["id"] for r in lifted] != [r["id"] for r in plain]
    conn = m.get_conn()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM mashup_candidates WHERE timing_ok = 0").fetchone()[0] == 0
    finally:
        conn.close()
    assert not any(r["dropin"] for r in lifted if r["timing_ok"] != 1)


def test_planted_dropins_are_stored_and_tagged(scored):
    m, mashups, manifest = scored
    assert manifest["planted"]
    for p in manifest["planted"]:
        rows = _listed(mashups, vocal_song_id=p["vocal_song_id"],
                       inst_song_id=p["inst_song_id"])
        assert rows, p
        assert any(r["dropin"] for r in mashups._with_playback_terms(
            m.get_candidates_enriched(
                vocal_song_id=p["vocal_song_id"], inst_song_id=p["inst_song_id"],
                combo_type="vocal_over_instrumental", limit=10,
                max_per_song_pair=0))), p


# ── stored, explained, and the same in Segments ──────────────────────────────

def test_every_placed_row_stores_the_slot_and_its_facts(scored):
    m, _mashups, _manifest = scored
    conn = m.get_conn()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM mashup_candidates WHERE combo_type='vocal_over_instrumental'")]
        ioi = conn.execute(
            "SELECT COUNT(*) FROM mashup_candidates WHERE combo_type != "
            "'vocal_over_instrumental' AND (score_slot IS NOT NULL "
            "OR dropin_mask IS NOT NULL)").fetchone()[0]
    finally:
        conn.close()
    assert ioi == 0
    placed = [r for r in rows if r["vocal_section_idx"] is not None]
    assert len(placed) > 300
    for r in placed:
        assert 0.0 <= r["score_slot"] <= 1.0
        assert r["dropin_mask"] is not None
        if r["window_bars"] is not None:
            assert r["window_bars"] in (8, 16)
            assert r["vocal_window_bar"] >= 0 and r["bed_window_bar"] >= 0
            assert 0.0 <= r["score_window"] <= 1.0
    assert any(r["window_bars"] for r in placed)
    assert any(r["slot_phrase_match"] is None for r in placed)      # unmeasured stays NULL
    from database.models import SECTION_PAIR_COLUMNS, SLOT_COLUMNS
    assert set(SLOT_COLUMNS) <= set(SECTION_PAIR_COLUMNS)


def test_the_slot_moves_the_fit_only_through_its_weight(library, tmp_path, monkeypatch):
    """slot_weight 0 scores exactly as before the term existed: the same
    pairings win with the same section fit and total."""
    out, _manifest = library
    tables = {}
    for weight in ("0", "0.5"):
        db = tmp_path / f"w{weight}.db"
        shutil.copy(out / "mashup.db", db)
        monkeypatch.setenv("MASHUP_DB_PATH", str(db))
        monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
        monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
        monkeypatch.setenv("MASHUP_SLOT_WEIGHT", weight)
        import config
        importlib.reload(config)
        import database.models as m
        importlib.reload(m)
        import matcher.match as match
        importlib.reload(match)
        match.score_all_pairs(db_path=m.DB_PATH)
        conn = sqlite3.connect(db)
        tables[weight] = {r[:4]: r[4:] for r in conn.execute(
            "SELECT vocal_song_id, inst_song_id, vocal_section_idx, inst_section_idx, "
            "score_section, score_timing, score_slot, score_total FROM mashup_candidates "
            "WHERE combo_type='vocal_over_instrumental' AND vocal_section_idx IS NOT NULL")}
        conn.close()
    from matcher.sections import timed_fit
    off = tables["0"]
    assert len(off) > 300
    for (section, timing, slot, _total) in off.values():
        # With no weight the fit is the timed fit alone; the slot is still measured.
        assert slot is not None
        assert 0.0 <= section <= 1.0
    moved = [k for k in off if k in tables["0.5"] and tables["0.5"][k][0] != off[k][0]]
    assert moved, "a slot weight of 0.5 changed no section fit"
    for k in moved[:50]:
        section0, _t, slot, _ = off[k]
        section5 = tables["0.5"][k][0]
        assert section5 == pytest.approx(0.5 * section0 + 0.5 * slot, abs=2e-4)
    assert timed_fit(0.6, {"score_timing": 1.0}, 0.3) == pytest.approx(0.72)


def test_segments_live_rows_carry_the_stored_slot(scored):
    """Criterion 1.3: Segments' live total — and the slot inside it — equals
    the stored one for a stored pairing. The stored rows are deleted so the
    ranking has to compute them."""
    m, _mashups, _manifest = scored
    from matcher import segment_match
    seg, out = next(
        (s, o) for s in m.list_segments("vocal")
        for o in [segment_match.rank_partners(s["segment_id"], "vocal", limit=50,
                                              db_path=m.DB_PATH)]
        if sum(1 for r in o["rows"] if r["stored"]) >= 3)
    stored = [r for r in out["rows"] if r["stored"]]
    conn = m.get_conn()
    conn.execute("DELETE FROM mashup_candidates WHERE vocal_song_id=?", (seg["song_id"],))
    conn.commit()
    conn.close()
    live = {(r["inst_song_id"], r["inst_section_idx"]): r
            for r in segment_match.rank_partners(seg["segment_id"], "vocal", limit=2000,
                                                 db_path=m.DB_PATH)["rows"]}
    for r in stored:
        again = live[(r["inst_song_id"], r["inst_section_idx"])]
        assert again["stored"] is False
        for col in ("score_total", "score_section", "score_slot", "slot_occupancy",
                    "slot_phrase_match", "slot_register", "slot_phrase_err_ms",
                    "window_bars", "vocal_window_bar", "bed_window_bar",
                    "score_window", "dropin_mask", "camelot_shift"):
            assert again[col] == r[col], (col, again[col], r[col])


def test_the_term_report_and_the_card_name_the_slot():
    from matcher.term_report import TERMS, term_report
    keys = {t["key"]: t for t in TERMS}
    assert keys["score_slot"]["kind"] == "slot"
    for part in ("slot_occupancy", "slot_phrase_match", "slot_register", "score_window"):
        assert keys[part]["kind"] == "measured"
    rep = term_report([], [], section_weights={}, match_weights={}, slot_weight=0.2)
    assert next(t for t in rep["terms"] if t["key"] == "score_slot")["weight"] == 0.2
    card = (ROOT / "frontend" / "src" / "components" / "PairCard.jsx").read_text(encoding="utf-8")
    dock = (ROOT / "frontend" / "src" / "hooks" / "usePairDock.js").read_text(encoding="utf-8")
    model = (ROOT / "frontend" / "src" / "components" / "pairs" / "pairModel.js").read_text(
        encoding="utf-8")
    css = (ROOT / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")
    # The card prints the server's verdict and words; it computes neither.
    for name in ("c.dropin", "c.dropin_reason", "c.slot_reason", "c.window_bars",
                 "c.score_slot"):
        assert name in card, name
    for src in (card, dock, model):
        assert "dropin_mask" not in src and "camelot_shift" not in src
    assert ".pc-dropin" in css
    assert 'dropin: false' in dock and 'dropin: f.dropin' in dock
    api = (ROOT / "frontend" / "src" / "api.js").read_text(encoding="utf-8")
    assert 'params.set("dropin", "true")' in api


def test_migration_nine_adds_only_nullable_candidate_columns(tmp_path, monkeypatch):
    from database import schema
    assert schema.SCHEMA_VERSION >= 9
    src = (ROOT / "database" / "schema.py").read_text(encoding="utf-8")
    body = src.split("def _slot_and_dropin")[1].split("\n_WINDOW_SPAN_COLUMNS")[0]
    assert "executescript" not in body and "pair_feedback" not in body
    assert body.count("ALTER TABLE") == 1 and "mashup_candidates ADD COLUMN" in body
    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    schema._baseline(conn)
    before = conn.execute("SELECT sql FROM sqlite_master WHERE name='pair_feedback'").fetchone()
    schema._slot_and_dropin(conn)
    schema._slot_and_dropin(conn)                            # idempotent
    cols = {r[1]: r for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    for name, _kind in schema._SLOT_COLUMNS:
        assert name in cols and cols[name][3] == 0           # nullable
    assert conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='pair_feedback'").fetchone() == before
    conn.close()


# ── a computed order over the whole library ──────────────────────────────────

@pytest.mark.parametrize("order", ["easy", "dropin", "tier", "score"])
def test_a_bounded_listing_is_the_whole_tables(scored, monkeypatch, order):
    """The easy, Drop-in, tier and bonus orders run over the top slice of
    scores first (models._bounded_listing); whatever they return must be the
    list the whole table returns — for small and large pages, with filters
    that thin the slice out, and with both combo types mixed."""
    m, _mashups, _manifest = scored
    monkeypatch.setenv("MASHUP_DROPIN_BONUS", "0.05")
    calls = {"bounded": 0, "answered": 0}
    real = m._bounded_listing

    def spy(*a, **k):
        calls["bounded"] += 1
        out = real(*a, **k)
        calls["answered"] += out is not None
        return out

    cases = [dict(combo_type="vocal_over_instrumental", limit=5, max_per_song=0),
             dict(combo_type="vocal_over_instrumental", limit=12, max_per_song=0),
             dict(combo_type="", limit=8, max_per_song=0),
             dict(combo_type="vocal_over_instrumental", limit=6, max_per_song=0,
                  dropin=False),
             dict(combo_type="vocal_over_instrumental", limit=4, max_per_song=0,
                  energy="high"),
             dict(combo_type="vocal_over_instrumental", limit=3, max_per_song=2)]
    for kw in cases:
        monkeypatch.setattr(m, "_bounded_listing", spy)
        got = [r["id"] for r in m.get_candidates_enriched(order=order, **kw)]
        monkeypatch.setattr(m, "_bounded_listing", lambda *a, **k: None)
        want = [r["id"] for r in m.get_candidates_enriched(order=order, **kw)]
        assert got == want, (order, kw)
        assert got
    assert calls["bounded"] == len(cases)
    assert calls["answered"] >= 2          # the slice really answered some
