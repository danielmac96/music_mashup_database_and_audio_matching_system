"""P1 — casting a wider net, and pointing the weights at the right question.

P1.1  The key gate deleted roughly three quarters of the library before scoring,
      for a cost (`pitch_cost`) that matcher/effort.py was already charging. It
      now defaults off, tempo widens, and a hard row cap keeps the extra volume
      bounded — evicting the WORST rows, so widening can only add ideas at the
      top of the list.

P1.3  `timbre_score` rewards a vocal and a bed sounding like the same record.
      For an instrumental blend that is the right question; for a vocal over a
      bed the question is whether the bed leaves room, which is
      `collision_score`. On the vocal path timbre's weight moves there.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _sec(idx, start, end, label, *, energy=0.7, vp=0.7):
    return {"section_index": idx, "start_sec": start, "end_sec": end,
            "label": label, "energy": energy, "vocal_presence": vp,
            "repetition": 2, "confidence": 0.8}


@pytest.fixture()
def library(tmp_path, monkeypatch):
    """Four tracks at a common tempo, spread right around the Camelot wheel.

    8A / 9A are neighbours; 2B and 3B are as far from 8A as the wheel goes. The
    old gate (0.55) admitted only the first group.
    """
    p = tmp_path / "gate.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    from database.models import (
        init_db, replace_sections, upsert_features, upsert_song,
    )
    init_db(p)
    ids = []
    for n, cam in enumerate(("8A", "9A", "2B", "3B")):
        sid = upsert_song(f"S{n}", f"A{n}", f"https://sc/g{n}", 200, "Pop",
                          status="analysed", db_path=p)
        ids.append(sid)
        for stem in ("full", "vocals", "instrumental"):
            upsert_features(sid, stem, {
                "bpm": 126.0, "bpm_confidence": 0.8,
                "key": "A", "mode": "minor", "camelot": cam,
                "key_confidence": 0.2,
                "loudness_rms": 0.04 + 0.005 * n, "energy": 0.5,
                "mfcc": [190.0] + [float((n * k) % 7) for k in range(12)],
                "band_energy": [0.1 + 0.02 * ((n + b) % 4) for b in range(8)],
            }, db_path=p)
        replace_sections(sid, [
            _sec(0, 0, 20, "intro", vp=0.05),
            _sec(1, 20, 50, "verse", vp=0.6),
            _sec(2, 50, 80, "chorus", vp=0.9, energy=0.9),
            _sec(3, 80, 110, "drop", vp=0.1, energy=0.95),
        ], db_path=p)
    return p, ids


# ── P1.1  the gate ───────────────────────────────────────────────────────────

def test_key_gate_defaults_off(library):
    """A pair the old 0.55 gate deleted must now reach the ranked list, priced
    rather than discarded."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import camelot_score, score_all_pairs

    assert camelot_score("8A", "3B") < 0.55, "fixture must contain a distant pair"
    score_all_pairs(db_path=db_path, scorer="heuristic")

    conn = get_conn(db_path)
    distant = conn.execute(
        """SELECT COUNT(*) FROM mashup_candidates
           WHERE combo_type='vocal_over_instrumental'
             AND vocal_camelot='8A' AND inst_camelot='3B'""").fetchone()[0]
    conn.close()
    assert distant > 0


def test_the_old_gate_was_rejecting_the_cheapest_pairs():
    """Why the key gate had to go, in one assertion.

    Camelot distance measures FIFTHS, so it does not order pairs by how much
    transposition they need — and transposition is the thing that actually costs
    something. One step around the wheel is a perfect fifth: 8A -> 9A needs 5
    semitones. Six steps is a tritone: 8A -> 2B needs 6. But 8A -> 3B, which the
    0.55 gate scored 0.25 and deleted outright, needs ONE semitone, because Db
    major is a semitone above A minor's relative major.

    So the gate was admitting a 5-semitone transpose and rejecting a free one.
    """
    from matcher.match import camelot_score, compute_semitone_shift

    assert compute_semitone_shift("8A", "9A") == 5      # admitted by the gate
    assert compute_semitone_shift("8A", "3B") == -1     # rejected by the gate
    assert camelot_score("8A", "9A") >= 0.55
    assert camelot_score("8A", "3B") < 0.55


def test_a_distant_pair_is_charged_for_the_transpose_it_needs(library):
    """The gate is gone, but the cost is not: pitch_cost still prices the move,
    and it prices it by semitones rather than by hours on the wheel."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import compute_semitone_shift, score_all_pairs

    # A tritone apart: the widest transpose there is.
    assert abs(compute_semitone_shift("8A", "2B")) == 6
    assert abs(compute_semitone_shift("8A", "3B")) == 1

    score_all_pairs(db_path=db_path, scorer="heuristic")
    conn = get_conn(db_path)
    rows = {(r["vocal_camelot"], r["inst_camelot"]): dict(r) for r in conn.execute(
        """SELECT * FROM mashup_candidates
           WHERE combo_type='vocal_over_instrumental'""").fetchall()}
    conn.close()

    cheap = rows[("8A", "3B")]
    expensive = rows[("8A", "2B")]
    assert cheap["effort_pitch"] == 0.0          # within PITCH_FREE
    assert expensive["effort_pitch"] == 1.0      # at PITCH_MAX
    assert expensive["score_effort"] > cheap["score_effort"]


def test_the_key_gate_still_works_when_asked_for(library):
    """Tight still exists — the default moved, the knob did not disappear."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import score_all_pairs

    score_all_pairs(db_path=db_path, scorer="heuristic", key_min_score=0.75)
    conn = get_conn(db_path)
    cams = {(r["vocal_camelot"], r["inst_camelot"]) for r in conn.execute(
        "SELECT vocal_camelot, inst_camelot FROM mashup_candidates").fetchall()}
    conn.close()
    assert ("8A", "3B") not in cams


def test_row_cap_keeps_the_best_rows(library, monkeypatch):
    """The cap is what makes widening the gate safe. It must evict the WORST
    rows, so raising the gate can only ever add ideas at the top."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import score_all_pairs

    uncapped = score_all_pairs(db_path=db_path, scorer="heuristic")
    want = [r["total"] for r in uncapped["vocal_over_instrumental"]][:3]
    assert len(want) == 3, "fixture must produce more than 3 rows"

    monkeypatch.setenv("MASHUP_MAX_CANDIDATE_ROWS", "3")
    capped = score_all_pairs(db_path=db_path, scorer="heuristic")

    got = capped["vocal_over_instrumental"]
    assert len(got) == 3
    assert [r["total"] for r in got] == want

    conn = get_conn(db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchone()[0]
    conn.close()
    assert n == 3


def test_capped_rows_stay_in_descending_score_order(library, monkeypatch):
    db_path, _ = library
    from matcher.match import score_all_pairs

    monkeypatch.setenv("MASHUP_MAX_CANDIDATE_ROWS", "5")
    got = score_all_pairs(db_path=db_path,
                          scorer="heuristic")["vocal_over_instrumental"]
    assert got == sorted(got, key=lambda r: r["total"], reverse=True)


# ── P1.3  per-combo weights ──────────────────────────────────────────────────

def test_vocal_path_moves_timbre_onto_collision():
    from config import current_match_weights

    base = current_match_weights()
    vocal = current_match_weights("vocal_over_instrumental")

    assert vocal["timbre_score"] == 0.0
    assert vocal["collision_score"] == pytest.approx(
        base["collision_score"] + base["timbre_score"])
    # Still a convex combination — otherwise every score in the library
    # rescales and the Min-match slider stops meaning anything.
    assert sum(vocal.values()) == pytest.approx(1.0)
    assert sum(base.values()) == pytest.approx(1.0)


def test_instrumental_blends_keep_timbre():
    """Two beds that do not cohere sound like a crossfade, not a mashup."""
    from config import current_match_weights

    ioi = current_match_weights("instrumental_over_instrumental")
    assert ioi == current_match_weights()
    assert ioi["timbre_score"] > 0


def test_timbre_is_still_measured_and_stored_on_vocal_rows(library):
    """Zero weight is not the same as not knowing. The sub-score still shows on
    the row, and the model still gets the column."""
    db_path, _ = library
    from database.models import get_conn
    from matcher.match import score_all_pairs

    score_all_pairs(db_path=db_path, scorer="heuristic")
    conn = get_conn(db_path)
    row = conn.execute(
        "SELECT score_timbre FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental' LIMIT 1").fetchone()
    conn.close()
    assert row["score_timbre"] is not None


def test_timbre_no_longer_moves_a_vocal_row_total():
    """The behavioural claim: on the vocal path, changing timbre alone must not
    change the ranking."""
    from matcher.match import LibraryStats, composite_score

    stats = LibraryStats()
    a = {"bpm": 128.0, "camelot": "8A", "loudness_rms": 0.05,
         "mfcc": [190.0] + [1.0] * 12, "band_energy": [0.125] * 8}
    similar = {**a, "mfcc": [190.0] + [1.0] * 12}
    different = {**a, "mfcc": [190.0] + [-1.0] * 12}

    vocal_same = composite_score(a, similar, stats=stats,
                                 combo_type="vocal_over_instrumental")
    vocal_diff = composite_score(a, different, stats=stats,
                                 combo_type="vocal_over_instrumental")
    assert vocal_same["timbre_score"] != vocal_diff["timbre_score"]
    assert vocal_same["total"] == pytest.approx(vocal_diff["total"])

    # …and on the instrumental path it still does.
    ioi_same = composite_score(a, similar, stats=stats,
                               combo_type="instrumental_over_instrumental")
    ioi_diff = composite_score(a, different, stats=stats,
                               combo_type="instrumental_over_instrumental")
    assert ioi_same["total"] != pytest.approx(ioi_diff["total"])


# ── B.1: the legend has to be able to state the weights in force ─────────────

def test_settings_expose_the_vocal_path_weights():
    """Discover defaults to vocal-over-instrumental, so the generic weight set
    describes a ranking the user is almost never looking at. The legend used to
    hardcode 'Key 30 · BPM 25 · Timbre 25 · Energy 20' — a set that matches
    neither the defaults, nor the user's saved values, nor the vocal path (where
    timbre is zero), and that omits collision entirely."""
    import config

    prov = config.settings_provenance()
    assert "match_weights_vocal" in prov
    vocal = prov["match_weights_vocal"]["value"]
    generic = prov["match_weights"]["value"]

    assert vocal["timbre_score"] == 0.0
    assert vocal["collision_score"] == pytest.approx(
        generic["collision_score"] + generic["timbre_score"])
    assert vocal == config.current_match_weights("vocal_over_instrumental")


def test_every_weighted_subscore_reaches_the_row():
    """A term that carries weight but is never sent to the client cannot be
    drawn, which is how collision stayed invisible while being the heaviest
    term on the vocal path."""
    from database.models import _CANDIDATE_INSERT_SQL
    import config

    for name in config.current_match_weights():
        column = "score_" + name.removesuffix("_score")
        assert column in _CANDIDATE_INSERT_SQL, \
            f"{name} is weighted but {column} is never persisted"
