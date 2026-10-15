"""Golden pairs: does the engine agree with how a producer would judge them?

A designed library (tests/fixtures/golden_pairs.json) of vocals and beds whose
tempo, key and harmony are chosen so a working DJ would call each pair good,
workable or bad, plus section shapes whose label is not in doubt. The library
goes through the real scorer and the real listing route, so the numbers
measured are the ones the dock prints.

Three measures, each with a floor in the fixture's `floors` that a change may
not drop below (raise a floor when a phase improves it — readme §8):

- `pair_auc`: of every (good, bad) pair, the share where good outranks bad.
- `rule_pass`: the share of listing rows that keep two producer rules —
  a transpose of 4+ semitones is never Free or Light effort, and a measured
  transpose the library calls unsure never overrides the Camelot estimate.
- `label_accuracy`: label_segments on the designed section shapes.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "golden_pairs.json")
                     .read_text(encoding="utf-8"))

NOTE = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
CAMELOT_MINOR = {9: "8A", 4: "9A", 11: "10A", 6: "11A", 1: "12A", 8: "1A",
                 3: "2A", 10: "3A", 5: "4A", 0: "5A", 7: "6A", 2: "7A"}
CAMELOT_MAJOR = {0: "8B", 7: "9B", 2: "10B", 9: "11B", 4: "12B", 11: "1B",
                 6: "2B", 1: "3B", 8: "4B", 3: "5B", 10: "6B", 5: "7B"}


def _camelot(key: str, mode: str) -> str:
    return (CAMELOT_MINOR if mode == "minor" else CAMELOT_MAJOR)[NOTE[key]]


def _chroma(root: int, mode: str, flat: bool = False, kind: str = "triad"):
    """A triad's pitch-class profile; `flat` is a near-featureless one whose
    best rotation is a coin flip (what a sparse or noisy section measures);
    `cluster` three adjacent semitones, which no transpose makes agree."""
    if kind == "cluster":
        v = [0.05] * 12
        for k, w in ((0, 1.0), (1, 0.9), (2, 0.85), (6, 0.6)):
            v[(root + k) % 12] = w
        return v
    if flat:
        v = [0.50 + 0.01 * ((k * 7) % 5) for k in range(12)]
        v[(root + 5) % 12] = 0.53
        return v
    # A section's mean chroma over many bars looks like its scale with the
    # triad on top, not a bare triad: that is what the analyser stores, and it
    # is why a vocal one wheel step from its bed (six shared notes) works.
    scale = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    third = 3 if mode == "minor" else 4
    v = [0.05] * 12
    for k in scale:
        v[(root + k) % 12] = 0.4
    for k, w in ((0, 1.0), (third, 0.7), (7, 0.8)):
        v[(root + k) % 12] = w
    return v


def _secs(role: str, bpm: float, root: int, mode: str, flat: bool, kind: str = "triad"):
    bar = 4 * 60.0 / bpm
    plan = ([("intro", 8, False), ("verse", 16, True), ("chorus", 16, True),
             ("verse", 16, True), ("chorus", 16, True), ("outro", 8, False)]
            if role == "vocal" else
            [("intro", 8, False), ("build", 8, False), ("drop", 16, False),
             ("breakdown", 8, False), ("drop", 16, False), ("outro", 8, False)])
    out, t = [], 0.0
    for i, (label, bars, sung) in enumerate(plan):
        a, b = t, t + bars * bar
        sec = {"section_index": i, "start_sec": round(a, 3), "end_sec": round(b, 3),
               "label": label, "energy": 0.9 if label in ("chorus", "drop") else 0.5,
               "repetition": 2, "confidence": 0.9, "bpm": bpm,
               "vocal_presence": 0.85 if sung else 0.05, "bar_count": bars,
               "downbeats": [round(a + k * bar, 3) for k in range(bars)]}
        chroma = _chroma(root, mode, flat, kind)
        if role == "vocal" and sung:
            sec["chroma_vocal"] = chroma
        if role == "bed" and label in ("drop", "breakdown", "build"):
            sec["chroma_bed"] = chroma
        out.append(sec)
        t = b
    return out


@pytest.fixture()
def listing(tmp_path, monkeypatch):
    """Score the designed library and return the dock's rows for it."""
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "golden.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()

    ids = {}
    for n, song in enumerate(FIXTURE["songs"]):
        sid = m.upsert_song(song["name"], f"Golden {n}", f"u://golden/{n}", 200,
                            "House", status="analysed")
        ids[song["name"]] = sid
        cam = _camelot(song["key"], song["mode"])
        mfcc = [150.0] + [((n * (k + 3)) % 11) - 5.0 for k in range(12)]
        for stem in ("full", "vocals", "instrumental"):
            m.upsert_features(sid, stem, {
                "bpm": song["bpm"], "key": song["key"], "mode": song["mode"],
                "camelot": cam, "loudness_rms": 0.1, "energy": 0.5,
                "bpm_confidence": 0.8, "key_confidence": 0.05, "mfcc": mfcc,
                "band_energy": ([0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
                                if stem == "vocals" else
                                [0.3, 0.25, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02])})
        m.replace_sections(sid, _secs(song["role"], song["bpm"], NOTE[song["key"]],
                                      song["mode"], song.get("flat_chroma", False),
                                      song.get("chroma", "triad")))

    from matcher.match import score_all_pairs
    score_all_pairs(db_path=m.DB_PATH)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    mashups = importlib.reload(importlib.import_module("api.routes.mashups"))
    app = FastAPI()
    app.include_router(mashups.router, prefix="/api/mashups")
    rows = TestClient(app).get(
        "/api/mashups", params={"limit": 1000, "max_per_song": 0,
                                "combo_type": "vocal_over_instrumental"}).json()["candidates"]
    best = {}
    for r in rows:   # best section pairing per song pair, as the dock ranks it
        key = (r["vocal_song_id"], r["inst_song_id"])
        if key not in best or (r.get("score_total") or 0) > (best[key].get("score_total") or 0):
            best[key] = r
    return ids, best


def measure(ids, best):
    by_name = {v: k for k, v in ids.items()}
    labelled = []
    for case in FIXTURE["pairs"]:
        row = best.get((ids[case["vocal"]], ids[case["bed"]]))
        labelled.append((case, row))

    good = [(r.get("score_total") or 0) if r else 0.0 for c, r in labelled if c["expect"] == "good"]
    bad = [(r.get("score_total") or 0) if r else 0.0 for c, r in labelled if c["expect"] == "bad"]
    wins = sum((g > b) + 0.5 * (g == b) for g in good for b in bad)
    auc = wins / (len(good) * len(bad)) if good and bad else 1.0

    from matcher.match import compute_semitone_shift
    checks, failures = 0, []
    for r in best.values():
        shift = r.get("semitone_shift")
        if shift is not None and abs(shift) >= 4:
            checks += 1
            if r.get("effort_label") in ("Free", "Light"):
                failures.append(f"{by_name[r['vocal_song_id']]} over {by_name[r['inst_song_id']]}: "
                                f"{shift:+d} st labelled {r['effort_label']}")
        if r.get("harmonic_sure") is False and r.get("harmonic_shift") is not None:
            checks += 1
            camelot = compute_semitone_shift(r.get("vocal_camelot") or "",
                                             r.get("inst_camelot") or "")
            if shift != camelot:
                failures.append(f"{by_name[r['vocal_song_id']]} over {by_name[r['inst_song_id']]}: "
                                f"unsure measured shift {shift:+d} beat Camelot {camelot:+d}")
    rule_pass = 1.0 - len(failures) / checks if checks else 1.0
    cases = {c["name"]: [c["expect"], r and round(r.get("score_total") or 0, 4),
                         r and r.get("semitone_shift"), r and r.get("effort_label")]
             for c, r in labelled}
    return {"pair_auc": round(auc, 3), "rule_pass": round(rule_pass, 3), "cases": cases,
            "failures": failures, "missing": [c["name"] for c, r in labelled if r is None]}


def label_accuracy():
    from analysis.structure import label_segments
    right, wrong = 0, []
    for case in FIXTURE["labels"]:
        segs = [dict(s) for s in case["segments"]]
        got = [s["label"] for s in label_segments(segs, case.get("has_vocals", True))]
        for i, want in case["expect"].items():
            if got[int(i)] in want.split("|"):
                right += 1
            else:
                wrong.append(f"{case['name']}[{i}]: want {want}, got {got[int(i)]}")
    total = right + len(wrong)
    return round(right / total, 3), wrong


def test_every_designed_pair_is_scored(listing):
    ids, best = listing
    assert not measure(ids, best)["missing"], "a designed pair fell outside the gates"


def test_pairs_rank_and_rules_hold_their_floors(listing):
    ids, best = listing
    got = measure(ids, best)
    print("\ngolden pairs:", json.dumps(got, indent=1))
    floors = FIXTURE["floors"]
    assert got["pair_auc"] >= floors["pair_auc"], got
    assert got["rule_pass"] >= floors["rule_pass"], got


def test_section_labels_hold_their_floor():
    acc, wrong = label_accuracy()
    print("\ngolden labels:", acc, wrong)
    assert acc >= FIXTURE["floors"]["label_accuracy"], wrong
