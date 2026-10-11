"""A scannable dock (2026-10-10 persona simulation, phase C): compact rows with
the focused pair as the full card, plain term names, a headline that separates
the top pairs (rank among a track's partners when scoped, one-decimal
percentile otherwise), and the transpose warning ranked against the library.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mashups
    importlib.reload(mashups)
    return models, mashups


def _side(sid, bpm=124.0):
    return {"song_id": sid, "title": f"S{sid}", "artist": "A", "bpm": bpm,
            "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5}


def _score(models, v, i, total, combo="vocal_over_instrumental"):
    models.upsert_candidate(_side(v), _side(i),
                            {"total": total, "bpm_score": 1.0, "key_score": 1.0,
                             "energy_score": 0.5, "timbre_score": 0.5},
                            combo_type=combo)


def _songs(models, n):
    return [models.upsert_song(f"S{k}", "A", f"https://sc/{k}", 200, status="analysed")
            for k in range(n)]


# ── the headline ─────────────────────────────────────────────────────────────

def test_a_scoped_list_ranks_each_partner(tmp_path, monkeypatch):
    models, mashups = _setup(tmp_path, monkeypatch)
    v, a, b, c = _songs(models, 4)
    _score(models, v, a, 0.70)
    _score(models, v, b, 0.90)
    _score(models, v, c, 0.80)
    out = mashups.list_candidates(vocal_song_id=v, limit=10, max_per_song=0)
    got = {r["inst_song_id"]: (r["scope_rank"], r["scope_total"]) for r in out["candidates"]}
    assert got == {b: (1, 3), c: (2, 3), a: (3, 3)}


def test_hidden_pairs_and_excluded_tracks_leave_the_ranking(tmp_path, monkeypatch):
    models, _ = _setup(tmp_path, monkeypatch)
    v, a, b, c = _songs(models, 4)
    for i, s in ((a, 0.9), (b, 0.8), (c, 0.7)):
        _score(models, v, i, s)
    models.hide_pair(v, a)
    models.exclude_track(b)
    ranks, total = models.partner_ranks(v, "vocal")
    assert (ranks, total) == ({c: 1}, 1)


def test_a_bed_ranks_its_vocals(tmp_path, monkeypatch):
    models, mashups = _setup(tmp_path, monkeypatch)
    bed, v1, v2 = _songs(models, 3)
    _score(models, v1, bed, 0.6)
    _score(models, v2, bed, 0.8)
    out = mashups.list_candidates(inst_song_id=bed, limit=10, max_per_song=0)
    got = {r["vocal_song_id"]: r["scope_rank"] for r in out["candidates"]}
    assert got == {v2: 1, v1: 2}


def test_an_unscoped_list_carries_no_rank(tmp_path, monkeypatch):
    models, mashups = _setup(tmp_path, monkeypatch)
    v, a = _songs(models, 2)
    _score(models, v, a, 0.7)
    out = mashups.list_candidates(limit=10)
    assert all("scope_rank" not in r for r in out["candidates"])


def test_the_card_headline_is_rank_when_scoped_and_one_decimal_otherwise():
    model = _read("components/pairs/pairModel.js")
    assert "Math.round((c.score_percentile ?? 0) * 1000) / 10" in model
    fn = model[model.index("export function headlineOf"):]
    fn = fn[:fn.index("\n}")]
    assert "c?.scope_rank != null" in fn and "`#${c.scope_rank}`" in fn
    card = _read("components/PairCard.jsx")
    assert "headlineOf(c, scopeNoun)" in card and "{head.big}" in card


# ── harmony, ranked against the library ──────────────────────────────────────

def test_the_transpose_warning_uses_the_librarys_threshold():
    from matcher.recipe import pair_recipe
    row = {"vocal_bpm": 124.0, "inst_bpm": 124.0, "harmonic_shift": 2,
           "harmonic_confidence": 0.05, "score_key": 0.8, "alignment_offset": 0.0,
           "stretch_factor": 1.0, "semitone_shift": 2, "section_loop_repeats": 1}
    fixed = pair_recipe(row)                         # no library threshold yet
    assert fixed["harmonic_sure"] is False
    assert any(w["key"] == "key_unsure" for w in fixed["warnings"])
    ranked = pair_recipe(row, unsure_below=0.02)     # this library's 20th percentile
    assert ranked["harmonic_sure"] is True
    assert not any(w["key"] == "key_unsure" for w in ranked["warnings"])
    unknown = pair_recipe({**row, "harmonic_shift": None, "harmonic_confidence": None})
    assert unknown["harmonic_sure"] is None


def test_a_score_run_records_the_librarys_harmony_threshold(tmp_path, monkeypatch):
    models, _ = _setup(tmp_path, monkeypatch)
    songs = _songs(models, 11)
    for s in songs[1:]:
        _score(models, songs[0], s, 0.5)
    conn = models.get_conn()
    for k, s in enumerate(songs[1:]):
        conn.execute("UPDATE mashup_candidates SET harmonic_confidence=? "
                     "WHERE inst_song_id=?", (0.01 * (k + 1), s))
    conn.commit()
    conn.close()
    from matcher import match
    importlib.reload(match)
    match._record_harmony_threshold(models.DB_PATH)
    pref = models.get_pref("harmony_unsure_below")
    assert pref["value"] == 0.03 and pref["pairs"] == 10   # the 20th percentile of 0.01..0.10
    from matcher.recipe import harmony_unsure_below
    assert harmony_unsure_below(models.DB_PATH) == 0.03


def test_listings_and_plan_options_carry_harmonic_sure():
    src = (ROOT / "api" / "routes" / "mashups.py").read_text(encoding="utf-8")
    assert 'r["harmonic_sure"] = r["recipe"]["harmonic_sure"]' in src
    from matcher.plan import OPTION_HARMONY_KEYS
    assert "harmonic_sure" in OPTION_HARMONY_KEYS
    model = _read("components/pairs/pairModel.js")
    assert "c.harmonic_sure != null ? !!c.harmonic_sure" in model


# ── the rows ─────────────────────────────────────────────────────────────────

def test_unfocused_pairs_are_rows_and_the_focused_one_the_card():
    dock = _read("components/PairDock.jsx")
    assert "row={i !== cursor}" in dock and "focused={i === cursor}" in dock
    card = _read("components/PairCard.jsx")
    assert "function PairRow(" in card and "if (row) {" in card
    row = card[card.index("function PairRow("):]
    for bit in ("onPlay()", "StarRating", "onDoubleClick={onStudio}", "pc-effort"):
        assert bit in row, bit
    css = _read("styles.css")
    assert ".pc-row {" in css and ".pcr-line {" in css


def test_a_scoped_row_names_only_the_partner():
    card = _read("components/PairCard.jsx")
    assert 'scopedSide !== "vocal" && <b>{c.vocal_title}</b>' in card
    assert "scopedSide={scopeTitle ? role : null}" in _read("components/PairDock.jsx")


def test_term_sorts_are_a_named_menu():
    hook = _read("hooks/usePairDock.js")
    assert "[t.order, t.name," in hook
    dock = _read("components/PairDock.jsx")
    assert "pd-termsort" in dock and "By…" in dock


def test_filters_overlay_the_list_and_show_as_pills():
    dock = _read("components/PairDock.jsx")
    wrap = dock[dock.index('<div className="pd-listwrap">'):]
    assert wrap.index('<div className="pd-filters">') < wrap.index('<div className="pd-list"')
    assert "function filterPills(" in dock and "pd-pill" in dock
    assert "pd-filterdone" in dock
    css = _read("styles.css")
    assert ".pd-listwrap > .pd-filters {" in css and "position: absolute" in css
