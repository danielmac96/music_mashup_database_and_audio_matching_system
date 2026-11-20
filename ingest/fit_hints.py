"""
ingest/fit_hints.py — what a SoundCloud row says about itself, for mashing.

Nothing outside the library is analysed, so Discover rows have no measured BPM
or key. Many uploads PRINT them, though — "128 BPM", "8A", "F#m" in the title
or tags, especially the acapellas and instrumentals a mashup is built from.
This reads those, says whether the upload is an acapella or an instrumental,
and is careful: a bare "A" or "E" is an article or a typo, so a key is only
taken as a Camelot code or a note with an explicit minor/major marker.
Pure text parsing; never touches the network or the database.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional

_BPM = re.compile(r"\b(\d{2,3}(?:\.\d)?)\s*-?\s*bpm\b", re.I)
_CAMELOT = re.compile(r"(?<![\w#])(1[0-2]|[1-9])\s?([AB])(?![\w#])")
_NOTE_KEY = re.compile(
    r"(?<![\w#])([A-G])([#♯b♭]?)\s?(m(?:in(?:or)?)?|maj(?:or)?|minor|major)(?![a-z])")
_ACAPELLA = re.compile(r"\b(a\s?cap+el+a|acc?ap+el+a|vocals? only|vocal stem|studio acapella)\b", re.I)
_INSTRUMENTAL = re.compile(r"\b(instrumental|inst\.?|beat only|no vocals|karaoke)\b", re.I)

_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
# Camelot number for each note, minor (A) and major (B).
_MINOR = {"G#": 1, "D#": 2, "A#": 3, "F": 4, "C": 5, "G": 6, "D": 7, "A": 8,
          "E": 9, "B": 10, "F#": 11, "C#": 12}
_MAJOR = {"B": 1, "F#": 2, "C#": 3, "G#": 4, "D#": 5, "A#": 6, "F": 7, "C": 8,
          "G": 9, "D": 10, "A": 11, "E": 12}


def _texts(row: dict) -> str:
    tags = row.get("tags") or ""
    if isinstance(tags, str):
        try:
            tags = json.loads(tags) if tags.startswith("[") else [tags]
        except ValueError:
            tags = [tags]
    return " · ".join([row.get("title") or "", *[str(t) for t in tags or []]])


def parse(row: dict) -> dict:
    """{"bpm", "camelot", "role"} from a row's title and tags (any may be None)."""
    text = _texts(row)
    bpm = None
    m = _BPM.search(text)
    if m:
        v = float(m.group(1))
        if 50 <= v <= 220:
            bpm = v
    camelot = None
    m = _CAMELOT.search(text)
    if m:
        camelot = f"{int(m.group(1))}{m.group(2).upper()}"
    else:
        m = _NOTE_KEY.search(text)
        if m:
            note = m.group(1) + ("#" if m.group(2) in ("#", "♯") else "b" if m.group(2) in ("b", "♭") else "")
            note = _FLAT.get(note, note)
            minor = m.group(3).lower().startswith("m") and not m.group(3).lower().startswith("maj")
            table = _MINOR if minor else _MAJOR
            if note in table:
                camelot = f"{table[note]}{'A' if minor else 'B'}"
    role = ("acapella" if _ACAPELLA.search(text)
            else "instrumental" if _INSTRUMENTAL.search(text) else None)
    return {"bpm": bpm, "camelot": camelot, "role": role}


def _num(code: Optional[str]) -> Optional[int]:
    m = re.fullmatch(r"(1[0-2]|[1-9])[AB]", code or "")
    return int(m.group(1)) if m else None


def fits(hint: dict, library: Iterable[dict]) -> Optional[int]:
    """How many library tracks this upload would sit with: tempo within 6%
    (half/double time allowed) and, when both keys are known, within one
    Camelot step. None when the row printed no tempo — nothing to compare."""
    bpm = hint.get("bpm")
    if not bpm:
        return None
    n = _num(hint.get("camelot"))
    count = 0
    for t in library:
        tb = t.get("bpm")
        if not tb:
            continue
        if min(abs(bpm * m / tb - 1) for m in (0.5, 1, 2)) > 0.06:
            continue
        tn = _num(t.get("camelot"))
        if n is not None and tn is not None and min(abs(n - tn), 12 - abs(n - tn)) > 1:
            continue
        count += 1
    return count


def _genre_words(text: Optional[str]) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 2}


def gap_fills(hint: dict, genre: Optional[str], groups: Iterable[dict]) -> dict:
    """Which library gaps (api/routes/discovery.library_gaps) an upload could
    fill, from what it says about itself only — never an invented tempo or
    key: {"n": lonely library tracks it could partner, "groups": [labels]}.

    A gap group needs a bed (its vocals found few) or a vocal. The upload
    qualifies when its printed role does not rule the need out (an acapella
    is no bed, an instrumental no vocal; an unmarked song can be either once
    separated) and it matches the group on the one measure it has: its
    printed tempo against the group's band (half/double time allowed, 6 %),
    or — with no printed tempo — its genre against the group's genre. A
    printed key then narrows the group's tracks to those within one Camelot
    step."""
    role = hint.get("role")
    bpm = hint.get("bpm")
    n_key = _num(hint.get("camelot"))
    words = _genre_words(genre)
    total, labels = 0, []
    for g in groups or []:
        need = g.get("need")
        if (need == "bed" and role == "acapella") or (need == "vocal" and role == "instrumental"):
            continue
        if bpm:
            lo, hi = float(g.get("bpm_lo") or 0), float(g.get("bpm_hi") or 0)
            if not any(lo * 0.94 <= bpm * m < hi * 1.06 for m in (0.5, 1, 2)):
                continue
        else:
            gw = _genre_words(g.get("genre"))
            if not words or not gw or not (words & gw):
                continue
        tracks = g.get("tracks") or []
        if n_key is not None:
            tracks = [t for t in tracks if _num(t.get("camelot")) is None
                      or min(abs(n_key - _num(t["camelot"])),
                             12 - abs(n_key - _num(t["camelot"]))) <= 1]
        if not tracks:
            continue
        total += len(tracks)
        labels.append(f"{'beds' if need == 'bed' else 'vocals'} at "
                      f"{int(g.get('bpm_lo') or 0)}–{int(g.get('bpm_hi') or 0)} BPM")
    return {"n": total, "groups": labels}
