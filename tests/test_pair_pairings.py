"""The pair dock's pairing stepper: one card per song pair, ‹ ● ○ ○ › through
every stored section pairing of those two songs.

Backend: every listing row carries `pairing_count` / `pairing_rank`, and
GET /api/mashups/pairings returns the song pair's pairings best first, each
enriched exactly like a listed row. Frontend: the contracts the stepper rests
on, read from the JSX and CSS (there is no JS test runner)."""
import importlib
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FRONT = ROOT / "frontend" / "src"


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
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.server as server
    importlib.reload(server)

    a, b, c = (models.upsert_song(f"S{n}", "A", f"https://sc/{n}", 200, "House",
                                  status="analysed") for n in range(3))

    def side(song_id, title):
        return {"song_id": song_id, "title": title, "artist": "A", "bpm": 128.0,
                "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5}

    def row(v, i, total, vs, ins):
        return models.candidate_row(
            side(v, "V"), side(i, "B"),
            {"total": total, "bpm_score": 1.0, "key_score": 0.8,
             "energy_score": 0.5, "timbre_score": 0.5},
            section_pair={"vocal_section_idx": vs, "inst_section_idx": ins})

    models.bulk_upsert_candidates([
        # A over B: three section pairings, stored out of score order.
        row(a, b, 0.70, 1, 2), row(a, b, 0.90, 0, 0), row(a, b, 0.80, 2, 1),
        # A over C: one.
        row(a, c, 0.60, 0, 0),
    ])
    return TestClient(server.app), models, (a, b, c)


def test_listing_shows_one_card_per_song_pair_with_its_count(app):
    client, _, (a, b, c) = app
    rows = client.get("/api/mashups", params={"limit": 50}).json()["candidates"]
    ab = [r for r in rows if (r["vocal_song_id"], r["inst_song_id"]) == (a, b)]
    assert len(ab) == 1, "still one card per song pair"
    assert ab[0]["pairing_count"] == 3
    assert ab[0]["pairing_rank"] == 1 and ab[0]["vocal_section_idx"] == 0
    ac = next(r for r in rows if r["inst_song_id"] == c)
    assert ac["pairing_count"] == 1 and ac["pairing_rank"] == 1


def test_pairing_rank_follows_score_under_a_section_order(app):
    client, models, (a, b, _) = app
    rows = models.get_candidates_enriched(pairs=[(a, b)], max_per_song_pair=0)
    assert sorted(r["pairing_rank"] for r in rows) == [1, 2, 3]
    by_score = sorted(rows, key=lambda r: -r["score_total"])
    assert [r["pairing_rank"] for r in by_score] == [1, 2, 3]


def test_pairings_route_returns_every_pairing_best_first_enriched(app):
    client, _, (a, b, _) = app
    r = client.get("/api/mashups/pairings",
                   params={"vocal_song_id": a, "inst_song_id": b})
    assert r.status_code == 200, r.text
    got = r.json()["candidates"]
    assert r.json()["count"] == 3
    assert [x["vocal_section_idx"] for x in got] == [0, 2, 1]
    scores = [x["score_total"] for x in got]
    assert scores == sorted(scores, reverse=True)
    for x in got:
        # The same playback terms as a listed row (§7: one transpose, one tier).
        for k in ("semitone_shift", "recipe", "confidence_tier", "compat",
                  "harmonic_sure", "pairing_count"):
            assert k in x, k


def test_pairings_answer_for_a_hidden_pair_and_reject_a_bad_combo(app):
    client, models, (a, b, _) = app
    models.hide_pair(a, b)
    r = client.get("/api/mashups/pairings",
                   params={"vocal_song_id": a, "inst_song_id": b})
    assert r.json()["count"] == 3
    r = client.get("/api/mashups/pairings",
                   params={"vocal_song_id": a, "inst_song_id": b, "combo_type": "x"})
    assert r.status_code == 400


def test_an_unscored_pair_has_no_pairings(app):
    client, _, (_, b, c) = app
    r = client.get("/api/mashups/pairings",
                   params={"vocal_song_id": b, "inst_song_id": c})
    assert r.status_code == 200 and r.json()["candidates"] == []


# ── frontend contracts ───────────────────────────────────────────────────────

def _read(rel):
    return (FRONT / rel).read_text(encoding="utf-8")


def test_card_draws_the_stepper_only_with_more_than_one_pairing():
    src = _read("components/PairCard.jsx")
    assert "optionCount > 1 && onStep" in src
    assert 'className="pc-steps"' in src
    assert "pc-dot" in src and "stopPropagation" in src


def test_dock_keys_cards_by_song_pair_and_acts_on_the_shown_pairing():
    dock = _read("components/PairDock.jsx")
    assert "key={pairKey(c)}" in dock
    assert "expanded={expanded === pairKey(c)}" in dock
    assert "dock.groupTierOf(c)" in dock, "tier headers follow the listed row"
    hook = _read("hooks/usePairDock.js")
    assert "rows: display" in hook
    assert '"ArrowLeft"' in hook and '"ArrowRight"' in hook
    # The loop follows the dots.
    assert re.search(r"if \(wasLooping\) play\(opts\[next\]\)", hook)
    # Hiding is per song pair, whichever pairing is on screen.
    assert "hiddenKeys.has(pairKey(c))" in hook
    assert "getPairings" in _read("api.js")


def test_stepper_css_exists_and_uses_tokens():
    css = _read("styles.css")
    block = css[css.index(".pc-steps"):css.index(".pc-dot.on")]
    block += css[css.index(".pc-dot.on"):].split("\n", 1)[0]
    assert ".pc-dot" in block
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
