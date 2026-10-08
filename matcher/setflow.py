"""
matcher/setflow.py — how a run of mashups flows, one into the next.

A set item is a pair (vocal over bed). It plays at the vocal's tempo with the
bed conformed and transposed to it, so the item LANDS at the vocal's tempo and
key. Between two consecutive items the DJ has to move from one landing to the
next: this module grades that move, adds up the running time and suggests an
order. Pure functions over the dicts `database.models.get_set` returns, so the
Set screen, the exports and the tests read one definition.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from matcher.features import _camelot_distance
from matcher.match import _parse_camelot
from matcher.recipe import bed_shift

# Transition grades: (max Camelot wheel distance, max tempo change %).
SMOOTH = (1.0, 3.0)
WORKABLE = (2.0, 6.0)


def landing(item: Dict) -> Dict:
    """Where one mashup sits: tempo, key, and how long it plays."""
    bpm = item.get("target_bpm") or item.get("vocal_bpm")
    start, end = item.get("vocal_section_start"), item.get("vocal_section_end")
    dur = (end - start) if start is not None and end is not None and end > start else None
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


def flow(items: Sequence[Dict]) -> Dict:
    """Landings, the transitions between consecutive items, and running time."""
    lands = [landing(i) for i in items]
    trans = [transition(items[k], items[k + 1]) for k in range(len(items) - 1)]
    t, starts = 0.0, []
    for land in lands:
        starts.append(round(t, 2))
        t += land["secs"] or 0.0
    return {"landings": lands, "transitions": trans, "starts": starts,
            "total_secs": round(t, 2),
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
