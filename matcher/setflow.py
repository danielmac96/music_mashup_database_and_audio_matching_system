"""
matcher/setflow.py — how a run of mashups flows, one into the next.

A set item is a pair (vocal over bed). It plays at the vocal's tempo with the
bed conformed and transposed to it, so the item LANDS at the vocal's tempo and
key — unless the set has a TEMPO CURVE (`tempo_plan`: start → end BPM), in
which case each item lands at its point on the curve, both sides stretched to
it (matcher/recipe.pair_recipe prices that), and the key is still the vocal's.
Between two consecutive items the DJ has to move from one landing to the next:
this module grades that move, suggests HOW to make it (a bed swap, a blend, a
cut…), adds up the running time with the overlaps, and suggests an order. Pure
functions over the dicts `database.models.get_set` returns, so the Set screen,
the exports and the tests read one definition.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence  # noqa: F401

from matcher.features import _camelot_distance
from matcher.match import _parse_camelot
from matcher.recipe import bed_shift

# Transition grades: (max Camelot wheel distance, max tempo change %).
SMOOTH = (1.0, 3.0)
WORKABLE = (2.0, 6.0)


# How each move is made, and how many bars the two mashups overlap for it.
TRANSITIONS = {
    "bed_swap": {"bars": 8, "label": "bed swap",
                 "text": "the vocal keeps going — bring the next bed in on the next phrase"},
    "vocal_swap": {"bars": 8, "label": "vocal swap",
                   "text": "the bed keeps going — drop the next vocal in on the one"},
    "blend": {"bars": 8, "label": "blend",
              "text": "bring the next bed in under the last bars, high-passed, then swap the lows"},
    "echo_out": {"bars": 0, "label": "echo out",
                 "text": "throw a delay on the last vocal line and drop the next in on the one"},
    "cut": {"bars": 0, "label": "cut",
            "text": "cut on the downbeat — the next mashup drops in on the one"},
}
MAX_TRANSITION_BARS = 32


def tempo_targets(n: int, plan: Optional[Dict]) -> List[Optional[float]]:
    """Each item's tempo on the curve: start → end BPM, linear by position.
    No plan (or no start) → None for all, i.e. each lands at its vocal's tempo.
    A start with no end holds one tempo."""
    start = (plan or {}).get("start_bpm")
    if not start or n <= 0:
        return [None] * n
    end = (plan or {}).get("end_bpm") or start
    if n == 1:
        return [round(float(start), 2)]
    return [round(float(start) + (float(end) - float(start)) * k / (n - 1), 2)
            for k in range(n)]


def apply_tempo_plan(items: Sequence[Dict], plan: Optional[Dict]) -> None:
    """Write each item's `set_bpm` (its tempo on the curve), or clear it."""
    for item, bpm in zip(items, tempo_targets(len(items), plan)):
        if bpm is None:
            item.pop("set_bpm", None)
        else:
            item["set_bpm"] = bpm


def landing(item: Dict) -> Dict:
    """Where one mashup sits: tempo, key, and how long it plays."""
    bpm = item.get("set_bpm") or item.get("target_bpm") or item.get("vocal_bpm")
    start, end = item.get("vocal_section_start"), item.get("vocal_section_end")
    dur = (end - start) if start is not None and end is not None and end > start else None
    # The vocal section plays at the landing tempo: on a curve, the vocal is
    # stretched, so its section lasts vocal_bpm/landing_bpm as long (fold-aware).
    if dur and item.get("set_bpm") and item.get("vocal_bpm"):
        from matcher.match import effective_bpm
        dur = dur * effective_bpm(float(bpm), float(item["vocal_bpm"])) / float(bpm)
    return {"bpm": float(bpm) if bpm else None,
            "camelot": item.get("vocal_camelot") or None,
            "secs": dur, "bed_shift": bed_shift(item)}


def _tempo_change(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Percent change from a to b, read at half/double time when that is
    closer — a DJ moving 87 → 174 is not changing tempo at all."""
    if not a or not b:
        return None
    best = None
    for m in (0.5, 1.0, 2.0):
        pct = (b * m / a - 1.0) * 100.0
        if best is None or abs(pct) < abs(best):
            best = pct
    return round(best, 1)


def transition(a: Dict, b: Dict) -> Dict:
    """The move from mashup `a` to mashup `b`."""
    la, lb = landing(a), landing(b)
    tempo = _tempo_change(la["bpm"], lb["bpm"])
    key = (_camelot_distance(la["camelot"], lb["camelot"])
           if la["camelot"] and lb["camelot"] else None)
    if tempo is None or key is None:
        grade = "unknown"
    elif key <= SMOOTH[0] and abs(tempo) <= SMOOTH[1]:
        grade = "smooth"
    elif key <= WORKABLE[0] and abs(tempo) <= WORKABLE[1]:
        grade = "workable"
    else:
        grade = "jump"
    return {"tempo_pct": tempo, "key_steps": key, "grade": grade,
            "from_key": la["camelot"], "to_key": lb["camelot"],
            "from_bpm": la["bpm"], "to_bpm": lb["bpm"]}


def suggest_move(a: Dict, b: Dict, t: Dict) -> Dict:
    """How to get from mashup `a` into mashup `b`, given their transition `t`.

    A shared record is the classic move: the same vocal over a new bed is a bed
    swap, a new vocal over the same bed a vocal swap. Otherwise the grade
    decides — a smooth move blends for 8 bars, a workable one for 4, a key or
    tempo jump is hidden behind an echo-out or simply cut on the one."""
    if a.get("vocal_song_id") == b.get("vocal_song_id"):
        kind, bars = "bed_swap", 8
    elif a.get("inst_song_id") == b.get("inst_song_id"):
        kind, bars = "vocal_swap", 8
    elif t["grade"] == "smooth":
        kind, bars = "blend", 8
    elif t["grade"] == "workable":
        kind, bars = "blend", 4
    elif t["grade"] == "jump" and (t["key_steps"] or 0) > WORKABLE[0]:
        kind, bars = "echo_out", 0
    else:
        kind, bars = "cut", 0
    return {"type": kind, "bars": bars}


def _move(stored: Optional[Dict], suggested: Dict) -> Dict:
    """The move into an item: the user's choice when there is one."""
    chosen = stored if stored and stored.get("type") in TRANSITIONS else None
    m = dict(chosen or suggested)
    bars = m.get("bars")
    m["bars"] = int(max(0, min(MAX_TRANSITION_BARS,
                               bars if bars is not None else TRANSITIONS[m["type"]]["bars"])))
    spec = TRANSITIONS[m["type"]]
    m.update({"label": spec["label"], "text": spec["text"], "suggested": chosen is None,
              "suggestion": suggested})
    return m


def flow(items: Sequence[Dict], tempo_plan: Optional[Dict] = None) -> Dict:
    """Landings, the transitions between consecutive items — graded and with the
    move that makes each — and running time, overlaps included.

    `tempo_plan` (a set's curve) moves every landing onto it first; without
    one, an item lands at its vocal's tempo as it always has."""
    if tempo_plan is not None:
        apply_tempo_plan(items, tempo_plan)
    lands = [landing(i) for i in items]
    trans = []
    for k in range(len(items) - 1):
        a, b = items[k], items[k + 1]
        t = transition(a, b)
        t["move"] = _move(b.get("transition"), suggest_move(a, b, t))
        bpm = lands[k + 1]["bpm"]
        overlap = (t["move"]["bars"] * 4 * 60.0 / bpm
                   if bpm and t["move"]["bars"] else 0.0)
        # Never more than half of either mashup: an 8-bar blend out of an
        # 8-bar section would otherwise swallow it whole.
        shorter = min([x for x in (lands[k]["secs"], lands[k + 1]["secs"]) if x] or [0.0])
        t["overlap_secs"] = round(min(overlap, shorter / 2.0), 2)
        trans.append(t)
    t, starts = 0.0, []
    for k, land in enumerate(lands):
        if k:
            # The next mashup comes in `overlap` before this one ends — never
            # before the previous one started.
            t = max(starts[-1], t - trans[k - 1]["overlap_secs"])
        starts.append(round(t, 2))
        t += land["secs"] or 0.0
    return {"landings": lands, "transitions": trans, "starts": starts,
            "total_secs": round(t, 2),
            "tempo_plan": tempo_plan or None,
            "grades": {g: sum(1 for x in trans if x["grade"] == g)
                       for g in ("smooth", "workable", "jump", "unknown")}}


def _cost(a: Dict, b: Dict) -> float:
    t = transition(a, b)
    key = t["key_steps"] if t["key_steps"] is not None else 6.0
    tempo = abs(t["tempo_pct"]) if t["tempo_pct"] is not None else 20.0
    # One wheel step weighs about as much as a 3% tempo move — both are what
    # "smooth" allows on its own.
    return key + tempo / 3.0


def suggest_order(items: Sequence[Dict], start_item_id: Optional[int] = None) -> List[int]:
    """A running order that keeps each move small: start from the given item
    (else the slowest), then repeatedly take the cheapest next move. Greedy,
    which is what a DJ does by hand, and returns item ids."""
    if not items:
        return []
    rest = list(items)
    if start_item_id is not None and any(i["item_id"] == start_item_id for i in rest):
        cur = next(i for i in rest if i["item_id"] == start_item_id)
    else:
        cur = min(rest, key=lambda i: (landing(i)["bpm"] or 1e9, i.get("position", 0)))
    order = [cur]
    rest.remove(cur)
    while rest:
        nxt = min(rest, key=lambda i: (_cost(cur, i), i.get("position", 0)))
        order.append(nxt)
        rest.remove(nxt)
        cur = nxt
    return [i["item_id"] for i in order]


def camelot_ok(code: Optional[str]) -> bool:
    return _parse_camelot(code or "") is not None


def next_candidates(items: Sequence[Dict], pool: Sequence[Dict],
                    tempo_plan: Optional[Dict] = None, limit: int = 8) -> List[Dict]:
    """What could come next: pairs from `pool` (scored rows) ranked by how good
    they are AND how easily the set moves into them from its last landing.

    rank = score percentile − 0.06 × key steps − 0.02 × |tempo %| (+0.05 for a
    shared record, which makes a bed or vocal swap). On a tempo curve the
    candidate is landed at the curve's next point — its end tempo, held — so
    the tempo cost is the stretch it would need there. Pairs already in the set
    are skipped. Each row carries `next_why`, `next_move` and `next_rank`."""
    if not items:
        return []
    have = {(i["vocal_song_id"], i["inst_song_id"], i.get("vocal_section_idx"),
             i.get("inst_section_idx")) for i in items}
    last = items[-1]
    nxt_bpm = None
    if tempo_plan and tempo_plan.get("start_bpm"):
        nxt_bpm = float(tempo_plan.get("end_bpm") or tempo_plan["start_bpm"])
    out = []
    for c in pool:
        key = (c["vocal_song_id"], c["inst_song_id"], c.get("vocal_section_idx"),
               c.get("inst_section_idx"))
        if key in have:
            continue
        cand = dict(c)
        if nxt_bpm:
            cand["set_bpm"] = nxt_bpm
        t = transition(last, cand)
        steps = t["key_steps"] if t["key_steps"] is not None else 6.0
        tempo = abs(t["tempo_pct"]) if t["tempo_pct"] is not None else 20.0
        if nxt_bpm and cand.get("vocal_bpm"):
            from matcher.match import effective_bpm
            tempo = max(tempo, abs(nxt_bpm / effective_bpm(nxt_bpm, float(cand["vocal_bpm"])) - 1) * 100)
        move = suggest_move(last, cand, t)
        shared = move["type"] in ("bed_swap", "vocal_swap")
        rank = (float(c.get("score_percentile") or 0.0) - 0.06 * steps - 0.02 * tempo
                + (0.05 if shared else 0.0))
        why = [f"{t['from_key'] or '?'} → {t['to_key'] or '?'}"
               + (f" ({steps:g} step{'s' if steps != 1 else ''})" if t["key_steps"] is not None else ""),
               f"{tempo:.1f}% tempo"]
        if shared:
            why.insert(0, TRANSITIONS[move["type"]]["label"])
        cand.update({"next_rank": round(rank, 4), "next_grade": t["grade"],
                     "next_move": move["type"], "next_why": " · ".join(why)})
        out.append(cand)
    out.sort(key=lambda r: -r["next_rank"])
    return out[:max(1, limit)]
