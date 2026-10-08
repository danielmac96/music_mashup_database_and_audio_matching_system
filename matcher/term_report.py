"""
matcher/term_report.py — does each scored term agree with your ears?

A term earns a weight by separating the pairs you loved from the ones you
rejected, on THIS library. This module lines every stored term of the judged
pairs up against the verdicts and against the reasons given for them:

  * AUC — the chance a pair you rated 4–5 scores higher on the term than one
    you rated 1–2 (0.5 = no signal, 1 = perfect; below 0.5 the term points
    the wrong way);
  * Spearman ρ against the 1–5 stars, over every rated pair;
  * for each reason chip, the term's mean on the pairs tagged with it against
    the rest — "key clash" pairs should score low on KEY, "vocal buried" ones
    low on ROOM, or the term is not measuring what the reason names.

It only reads. Nothing here changes a weight: it is the evidence a person
looks at before moving one (readme §9, phase 3). Pure python, no audio.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# Every stored term worth checking, with where it sits in the ranking today.
# `kind`: section (score_section, by config.SECTION_WEIGHTS), track
# (score_total, by config.MATCH_WEIGHTS), measured (stored at weight 0), or
# total (the composite). `invert`: lower is better (effort).
TERMS: List[Dict] = [
    {"key": "score_total", "label": "Total", "kind": "total"},
    {"key": "score_label", "label": "LBL label", "kind": "section", "weight": "label"},
    {"key": "score_duration", "label": "DUR duration", "kind": "section", "weight": "duration"},
    {"key": "score_voice", "label": "VOI voice", "kind": "section", "weight": "voice"},
    {"key": "score_phrase", "label": "PHR phrase", "kind": "section", "weight": "phrase"},
    {"key": "score_rhythm", "label": "rhythm", "kind": "section", "weight": "rhythm"},
    {"key": "score_structure", "label": "structure", "kind": "section", "weight": "structure"},
    {"key": "score_room_section", "label": "section room", "kind": "measured"},
    {"key": "score_coverage", "label": "vocal coverage", "kind": "measured"},
    {"key": "score_energy_match", "label": "section energy match", "kind": "measured"},
    {"key": "score_bpm", "label": "BPM tempo", "kind": "track", "weight": "bpm_score"},
    {"key": "score_key", "label": "KEY harmony", "kind": "track", "weight": "key_score"},
    {"key": "score_energy", "label": "NRG loudness", "kind": "track", "weight": "energy_score"},
    {"key": "score_collision", "label": "ROOM track", "kind": "track", "weight": "collision_score"},
    {"key": "score_timbre", "label": "TIM timbre", "kind": "track", "weight": "timbre_score"},
    {"key": "score_effort", "label": "effort", "kind": "effort", "invert": True},
]

# Which terms a reason chip should move, if the term measures what it claims.
# `low`: a pair tagged with the reason should score LOW on these.
REASON_SUSPECTS: Dict[str, List[str]] = {
    "key_clash": ["score_key"],
    "harmony": ["score_key"],
    "timing_off": ["score_phrase", "score_duration"],
    "groove": ["score_phrase", "score_rhythm"],
    "vocal_buried": ["score_room_section", "score_collision", "score_coverage"],
    "vocal_sits": ["score_room_section", "score_collision"],
    "bass_mud": ["score_room_section", "score_collision"],
    "energy_mismatch": ["score_energy_match", "score_energy"],
    "energy_lift": ["score_energy_match", "score_energy"],
    "boring": ["score_total"],
    "contrast": ["score_total"],
    "bad_separation": ["score_voice", "score_coverage"],
}

# Below this many good AND bad pairs an AUC is noise, and the report says so.
MIN_PER_CLASS = 10
GOOD, BAD = 4, 2      # stars at or above / at or below


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based), ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean
        i = j + 1
    return ranks


def auc(good: Sequence[float], bad: Sequence[float]) -> Optional[float]:
    """P(a good pair outscores a bad one), ties counting half (Mann–Whitney)."""
    if not good or not bad:
        return None
    r = _ranks(list(good) + list(bad))
    rg = sum(r[:len(good)])
    u = rg - len(good) * (len(good) + 1) / 2.0
    return round(u / (len(good) * len(bad)), 4)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman's ρ (Pearson on average ranks); None when either is constant."""
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return round(cov / math.sqrt(vx * vy), 4)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def judged_rows(feedback: Sequence[Dict], candidates: Sequence[Dict]) -> List[Tuple[Dict, Dict]]:
    """(feedback, candidate) for every judgement whose pair is still scored.

    Matched on the four ids, like ux_pair_feedback_section; a judgement made
    before section pairs existed (NULL sections) is matched to its song pair's
    best row. A flagged `sections_stale` judgement names an older structure, so
    only its song pair is used."""
    by_key: Dict[tuple, Dict] = {}
    best: Dict[tuple, Dict] = {}
    for c in candidates:
        if c.get("combo_type", "vocal_over_instrumental") != "vocal_over_instrumental":
            continue
        pair = (c["vocal_song_id"], c["inst_song_id"])
        by_key[pair + (c.get("vocal_section_idx"), c.get("inst_section_idx"))] = c
        if pair not in best or (_num(c.get("score_total")) or 0) > (_num(best[pair].get("score_total")) or 0):
            best[pair] = c
    out = []
    for f in feedback:
        pair = (f["vocal_song_id"], f["inst_song_id"])
        stale = bool(f.get("sections_stale"))
        sections = (f.get("vocal_section"), f.get("inst_section"))
        if not stale and sections != (None, None):
            c = by_key.get(pair + sections)
        else:
            c = best.get(pair)
        if c is not None:
            out.append((f, c))
    return out


def term_report(feedback: Sequence[Dict], candidates: Sequence[Dict],
                section_weights: Optional[Dict] = None,
                match_weights: Optional[Dict] = None) -> Dict:
    """The evidence for and against each term, from the judged pairs."""
    rows = judged_rows(feedback, candidates)
    rated = [(f, c) for f, c in rows if _num(f.get("rating")) is not None]
    n_good = sum(1 for f, _ in rated if f["rating"] >= GOOD)
    n_bad = sum(1 for f, _ in rated if f["rating"] <= BAD)

    terms = []
    for t in TERMS:
        pts = [(float(f["rating"]), _num(c.get(t["key"]))) for f, c in rated]
        pts = [(r, v) for r, v in pts if v is not None]
        if t.get("invert"):
            pts = [(r, 1.0 - v) for r, v in pts]
        good = [v for r, v in pts if r >= GOOD]
        bad = [v for r, v in pts if r <= BAD]
        weight = None
        if t["kind"] == "section" and section_weights is not None:
            weight = section_weights.get(t["weight"])
        elif t["kind"] == "track" and match_weights is not None:
            weight = match_weights.get(t["weight"])
        elif t["kind"] == "measured":
            weight = 0.0
        a = auc(good, bad)
        enough = len(good) >= MIN_PER_CLASS and len(bad) >= MIN_PER_CLASS
        terms.append({
            "key": t["key"], "label": t["label"], "kind": t["kind"],
            "weight": None if weight is None else round(float(weight), 4),
            "n": len(pts), "n_good": len(good), "n_bad": len(bad),
            "mean_good": _mean(good), "mean_bad": _mean(bad),
            "auc": a,
            "spearman": spearman([r for r, _ in pts], [v for _, v in pts]),
            "enough": enough,
            "verdict": _verdict(a, weight, enough),
        })

    reasons = []
    tagged_any = [(f, c) for f, c in rated if f.get("reasons")]
    all_reasons = sorted({r for f, _ in tagged_any for r in f["reasons"]})
    for reason in all_reasons:
        with_r = [(f, c) for f, c in rated if reason in (f.get("reasons") or [])]
        without = [(f, c) for f, c in rated if reason not in (f.get("reasons") or [])]
        suspects = []
        for key in REASON_SUSPECTS.get(reason, []):
            a = [v for v in (_num(c.get(key)) for _, c in with_r) if v is not None]
            b = [v for v in (_num(c.get(key)) for _, c in without) if v is not None]
            suspects.append({"key": key, "mean_tagged": _mean(a), "mean_rest": _mean(b),
                             "n_tagged": len(a)})
        reasons.append({"reason": reason, "n": len(with_r), "suspects": suspects})

    return {"judged": len(rows), "rated": len(rated), "n_good": n_good, "n_bad": n_bad,
            "min_per_class": MIN_PER_CLASS, "terms": terms, "reasons": reasons}


def _verdict(a: Optional[float], weight: Optional[float], enough: bool) -> str:
    """One line a person can act on."""
    if a is None or not enough:
        return "not enough ratings yet"
    weighted = bool(weight)
    if a >= 0.65:
        return "agrees with your ears" if weighted else "agrees with your ears — worth a weight"
    if a <= 0.40:
        return "points the wrong way" + (" — reconsider its weight" if weighted else "")
    return "no clear signal" + (" — weighted anyway" if weighted else "")
