"""confidence_tier and compat on every pair row (matcher/compat.py), the tier=
filter and order=tier — and the pins that keep them one definition:
the Python tier ≡ its SQL twin ≡ theme.js tierFor's cut-offs, and compat's
transpose ≡ the row's semitone_shift (recipe.bed_shift)."""
import importlib
import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TIERS = {"strong", "good", "maybe", "long_shot"}
RELATIONS = {"same", "adjacent", "relative", "transpose", "clash", None}
N_SONGS = 42


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
    import api.routes.sets as sets
    importlib.reload(sets)
    import api.server as server
    importlib.reload(server)

    songs = [models.upsert_song(f"S{n}", f"A{n}", f"https://sc/{n}", 200,
                                "House" if n % 2 else "Pop", status="analysed")
             for n in range(N_SONGS)]
    conn = models.get_conn()
    conn.execute(
        "INSERT INTO features (song_id, stem_type, bpm, tags_json) VALUES (?,?,?,?)",
        (songs[0], "full", 128.0, json.dumps(
            {"genre": [{"label": "Electronic---Deep House", "p": 0.6},
                       {"label": "Electronic---House", "p": 0.2}]})))
    conn.commit()
    conn.close()

    cams = ["8A", "8A", "8B", "9A", "10A", "2A", None]
    rows = []
    for n in range(N_SONGS - 1):
        v, i = songs[n], songs[n + 1]
        scores = {"total": 0.95 - n * 0.01, "bpm_score": 1.0, "key_score": 0.8,
                  "energy_score": 0.5, "timbre_score": 0.5}
        # Every third pair has a measured transpose; half of those a close call.
        if n % 3 == 0:
            scores.update(harmonic_shift=(n % 5) - 2,
                          harmonic_confidence=0.01 if n % 2 else 0.6)
        rows.append(models.candidate_row(
            {"song_id": v, "title": f"S{n}", "artist": "A", "bpm": 128.0,
             "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
            {"song_id": i, "title": f"S{n + 1}", "artist": "A",
             "bpm": [128.0, 64.0, 250.0, 120.0, None][n % 5],
             "camelot": cams[n % len(cams)], "loudness_rms": 0.1, "energy": 0.5},
            scores, section_pair={"vocal_section_idx": 0, "inst_section_idx": 0}))
    models.bulk_upsert_candidates(rows)
    return TestClient(server.app), models, songs


def _all(client, **params):
    params = {"limit": 500, "max_per_song": 0, **params}
    r = client.get("/api/mashups", params=params)
    assert r.status_code == 200, r.text
    return r.json()["candidates"]


# ── contract ─────────────────────────────────────────────────────────────────

def _check_row(c):
    assert c["confidence_tier"] in TIERS
    compat = c["compat"]
    assert set(compat) == {"key", "bpm", "genre"}
    key, bpm, genre = compat["key"], compat["bpm"], compat["genre"]
    assert set(key) == {"vocal", "bed", "landing", "relation", "steps", "shift_st"}
    assert set(bpm) == {"vocal", "bed", "delta_pct", "fold"}
    assert set(genre) == {"vocal", "bed", "vocal_style", "bed_style", "same"}
    for k in ("vocal", "bed", "landing"):
        assert key[k] is None or isinstance(key[k], str)
    assert key["relation"] in RELATIONS
    assert key["steps"] is None or isinstance(key["steps"], (int, float))
    assert key["shift_st"] is None or isinstance(key["shift_st"], int)
    for k in ("vocal", "bed", "delta_pct"):
        assert bpm[k] is None or isinstance(bpm[k], (int, float))
    assert bpm["fold"] in ("half", "none", "double", None)
    for k in ("vocal", "bed", "vocal_style", "bed_style"):
        assert genre[k] is None or isinstance(genre[k], str)
    assert genre["same"] in (True, False, None)
    # One transpose per pair: compat never recomputes it.
    assert key["shift_st"] == c["semitone_shift"]
    assert key["landing"] == key["vocal"]


def test_every_listing_row_carries_tier_and_compat(app):
    client, _, _ = app
    rows = _all(client)
    assert len(rows) == N_SONGS - 1
    for c in rows:
        _check_row(c)


def test_by_vocal_and_set_rows_carry_them_too(app):
    client, _, songs = app
    for c in client.get("/api/mashups/by-vocal").json()["candidates"]:
        _check_row(c)

    sid = client.post("/api/sets", json={"name": "s"}).json()["id"]
    r = client.post(f"/api/sets/{sid}/items", json={
        "vocal_song_id": songs[0], "inst_song_id": songs[1],
        "vocal_section_idx": 0, "inst_section_idx": 0})
    assert r.status_code == 200, r.text
    items = client.get(f"/api/sets/{sid}").json()["items"]
    assert items
    for it in items:
        _check_row(it)
    for c in client.get(f"/api/sets/{sid}/next").json()["candidates"]:
        _check_row(c)


def test_unmeasured_is_null_not_zero(app):
    client, _, _ = app
    rows = _all(client)
    no_key = [c for c in rows if not c.get("inst_camelot")]
    assert no_key
    for c in no_key:
        key = c["compat"]["key"]
        assert key["bed"] is None and key["relation"] is None and key["steps"] is None
    no_bpm = [c for c in rows if not c.get("inst_bpm")]
    assert no_bpm
    for c in no_bpm:
        bpm = c["compat"]["bpm"]
        assert bpm["bed"] is None and bpm["delta_pct"] is None and bpm["fold"] is None


def test_fold_delta_genre_and_style(app):
    client, _, songs = app
    rows = {(c["vocal_song_id"], c["inst_song_id"]): c for c in _all(client)}
    first = rows[(songs[0], songs[1])]
    assert first["compat"]["bpm"] == {"vocal": 128.0, "bed": 128.0,
                                      "delta_pct": 0.0, "fold": "none"}
    assert first["compat"]["genre"] == {"vocal": "Pop", "bed": "House",
                                        "vocal_style": "Deep House",
                                        "bed_style": None, "same": False}
    half = rows[(songs[1], songs[2])]["compat"]["bpm"]     # bed at 64
    assert half["fold"] == "double" and half["delta_pct"] == 0.0
    fast = rows[(songs[2], songs[3])]["compat"]["bpm"]     # bed at 250
    assert fast["fold"] == "half" and fast["delta_pct"] == pytest.approx(2.4)


# ── one definition ───────────────────────────────────────────────────────────

def test_tier_cutoffs_are_theme_js_tierfor(app):
    from matcher.compat import TIER_MIN_PCT, TIERS as ORDER
    src = (ROOT / "frontend/src/theme.js").read_text(encoding="utf-8")
    body = src[src.index("export function tierFor"):]
    body = body[:body.index("\n}")]
    js = {name: int(n) for n, name in
          re.findall(r'totalPct >= (\d+)\) return \{ tier: "(\w+)"', body)}
    assert js == {"EXCELLENT": 85, "STRONG": 75, "GOOD": 65, "FAIR": 50}
    assert TIER_MIN_PCT == {"strong": js["STRONG"], "good": js["GOOD"],
                            "maybe": js["FAIR"]}
    assert ORDER == ("strong", "good", "maybe", "long_shot")


@pytest.mark.parametrize("pct,sure,want", [
    (1.0, None, "strong"), (0.75, True, "strong"), (0.7499, None, "good"),
    (0.65, None, "good"), (0.6499, None, "maybe"), (0.50, None, "maybe"),
    (0.4999, None, "long_shot"), (0.0, None, "long_shot"),
    (0.9, False, "good"), (0.7, False, "maybe"), (0.55, False, "long_shot"),
    (0.1, False, "long_shot"), (None, None, None),
])
def test_tier_thresholds_and_downgrade(pct, sure, want):
    from matcher.compat import confidence_tier
    assert confidence_tier(pct, sure) == want


def test_the_sql_tier_is_the_python_tier(app):
    """tier= filters with compat.tier_sql; the row prints confidence_tier."""
    client, models, _ = app
    models.set_pref("harmony_unsure_below", {"value": 0.2})
    everything = _all(client)
    downgraded = [c for c in everything if c["harmonic_sure"] is False]
    assert downgraded, "the fixture must exercise the close-call downgrade"
    seen = set()
    for tier in sorted(TIERS):
        got = _all(client, tier=tier)
        want = [c for c in everything if c["confidence_tier"] == tier]
        assert [c["id"] for c in got] == [c["id"] for c in want], tier
        assert all(c["confidence_tier"] == tier for c in got)
        seen |= {c["id"] for c in got}
    assert seen == {c["id"] for c in everything}
    both = _all(client, tier="strong, maybe")
    assert {c["confidence_tier"] for c in both} == {"strong", "maybe"}


def test_order_tier_is_tier_then_score(app):
    client, models, _ = app
    models.set_pref("harmony_unsure_below", {"value": 0.2})
    from matcher.compat import TIERS as ORDER
    rows = _all(client, order="tier")
    keys = [(ORDER.index(c["confidence_tier"]), -c["score_total"]) for c in rows]
    assert keys == sorted(keys)
    assert rows != _all(client)      # the downgrade really reorders something


def test_an_unknown_tier_is_400_and_existing_orders_still_work(app):
    client, _, _ = app
    assert client.get("/api/mashups", params={"tier": "great"}).status_code == 400
    assert client.get("/api/mashups", params={"order": "nope"}).status_code == 400
    for order in ("score", "uncertain", "label", "duration", "voice", "phrase", "tier"):
        assert client.get("/api/mashups", params={"order": order}).status_code == 200


# ── key relation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("v,b,want", [
    ("8A", "8A", "same"), ("8A", "8B", "relative"), ("8A", "9A", "adjacent"),
    ("8A", "7B", "adjacent"), ("8A", "10A", "transpose"), ("8A", "2A", "transpose"),
    ("12B", "1A", "adjacent"), ("8A", None, None), ("", "8A", None),
    ("8A", "13A", None),
])
def test_key_relation(v, b, want):
    from matcher.compat import key_relation
    assert key_relation(v, b) == want


def test_clash_is_the_recipes_measured_clash(app):
    """`clash` is not a wheel distance: it is the recipe's own warning (measured
    fit under HARMONY_CLASH), so the card's WATCH row and compat agree."""
    from matcher.compat import pair_compat
    from matcher.recipe import HARMONY_CLASH, pair_recipe
    row = {"vocal_song_id": 1, "inst_song_id": 2, "vocal_camelot": "8A",
           "inst_camelot": "8A", "vocal_bpm": 128.0, "inst_bpm": 128.0,
           "harmonic_shift": 0, "harmonic_confidence": 0.5,
           "score_key": HARMONY_CLASH - 0.05, "semitone_shift": 0,
           "stretch_factor": 1.0}
    row["recipe"] = pair_recipe(row)
    assert any(w["key"] == "clash" for w in row["recipe"]["warnings"])
    assert pair_compat(row)["key"]["relation"] == "clash"
    row["score_key"] = HARMONY_CLASH + 0.05
    row["recipe"] = pair_recipe(row)
    assert pair_compat(row)["key"]["relation"] == "same"


def test_camelot_steps_is_the_sets_distance():
    from matcher.compat import camelot_steps
    from matcher.features import _camelot_distance
    codes = [f"{n}{l}" for n in range(1, 13) for l in "AB"]
    for a in codes:
        for b in codes:
            assert camelot_steps(a, b) == _camelot_distance(a, b)
    assert camelot_steps("8A", None) is None       # unknown is not "far"
