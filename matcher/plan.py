"""
matcher/plan.py — Turn a scored mashup candidate into an actionable plan.

Given a vocal song + instrumental song, produce:
  * tempo work:   target BPM, stretch factor (halftime/doubletime aware)
  * key work:     semitone shift for the instrumental, key relation
  * section work: which vocal sections (chorus/verse timestamps) to lay over
                  which instrumental sections (drop/chorus), duration-matched
  * a numbered human-readable recipe for the DAW

Everything is plain python + sqlite reads, so it is unit-testable without
librosa/demucs installed.
"""
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from matcher.match import (
    camelot_score, compute_semitone_shift, compute_stretch_factor, effective_bpm,
)

# Band edges for the occupancy overlay (E.1). Imported from the module that
# MEASURES them so the drawn axis cannot drift from the vector it labels.
try:
    from analysis.quality import BAND_EDGES
except ImportError:                      # numpy/librosa absent — plan still works
    BAND_EDGES = (0.0, 60.0, 150.0, 400.0, 1000.0, 2500.0, 5000.0, 8000.0, 11025.0)

# Section label priority when choosing what to mash.
_VOCAL_LABEL_PRIORITY = {"chorus": 0, "verse": 1, "bridge": 2, "drop": 3,
                         "breakdown": 4, "intro": 5, "outro": 6}
_INST_LABEL_PRIORITY = {"drop": 0, "chorus": 1, "verse": 2, "breakdown": 3,
                        "bridge": 4, "intro": 5, "outro": 6}


def _fmt_ts(secs: float) -> str:
    s = int(round(secs or 0))
    m, sec = divmod(s, 60)
    return f"{m}:{sec:02d}"


def _key_relation(camelot_a: str, camelot_b: str) -> str:
    s = camelot_score(camelot_a, camelot_b)
    if s >= 1.0:
        return "same key"
    if s >= 0.85:
        return "adjacent on Camelot wheel (energy shift)"
    if s >= 0.75:
        return "relative major/minor"
    if s >= 0.55:
        return "two steps on Camelot wheel"
    return "distant — pitch the instrumental to match"


def _pick_sections(sections: List[Dict], priority: Dict[str, int],
                   vocal_side: bool) -> List[Dict]:
    """Order a song's sections by usefulness for mashing."""
    usable = []
    for s in sections:
        label = s.get("label") or "verse"
        if label in ("intro", "outro"):
            continue
        if vocal_side and s.get("vocal_presence") is not None \
                and s["vocal_presence"] < 0.25:
            continue  # nothing to sing over there
        usable.append(s)
    usable.sort(key=lambda s: (priority.get(s.get("label") or "verse", 9),
                               -(s.get("energy") or 0)))
    return usable


def _pairing_row(v_start: float, v_end: float, v_label: Optional[str],
                 i_start: float, i_end: float, i_label: Optional[str],
                 stretch_factor: float, extra: Optional[Dict] = None) -> Dict:
    """One pairing in the shape the recipe, the README and PlanDetails read."""
    v_dur = float(v_end) - float(v_start)
    i_dur_stretched = (float(i_end) - float(i_start)) / max(stretch_factor, 1e-6)
    return {
        "vocal_label": v_label,
        "vocal_start": v_start,
        "vocal_end": v_end,
        "vocal_duration": round(v_dur, 1),
        "inst_label": i_label,
        "inst_start": i_start,
        "inst_end": i_end,
        "inst_duration_stretched": round(i_dur_stretched, 1),
        "note": (
            f"Lay vocal {v_label} ({_fmt_ts(v_start)}–{_fmt_ts(v_end)}) "
            f"over instrumental {i_label} ({_fmt_ts(i_start)}–{_fmt_ts(i_end)})"
        ),
        **(extra or {}),
    }


def build_pairings(vocal_sections: List[Dict], inst_sections: List[Dict],
                   stretch_factor: float, max_pairings: int = 4,
                   bpm: Optional[float] = None) -> List[Dict]:
    """Which vocal sections to lay over which instrumental sections.

    Delegates to matcher.sections.top_section_pairs — the SAME chooser that
    produced the candidate row — rather than ranking again here.

    It used to rank independently: label priority plus a *seconds*-based
    duration fit, while scoring used label, vocal presence and a *bars*-based
    phrase fit. The two disagree, and both answers were reaching the user at
    once — the ranked row showed the scorer's section pair while the Plan
    expander directly beneath it showed this function's, and the FL export
    silently rendered this one. Two definitions of "the good part" is exactly
    how the preview and the recipe end up describing different mashups.

    `bpm` is the target tempo (the vocal's), so the fit is measured in bars.
    Without it the phrase fit degrades to the seconds-based one, which is the
    old behaviour and is why the parameter is threaded through from the plan.
    """
    from matcher.sections import top_section_pairs

    rows = top_section_pairs(vocal_sections or [], inst_sections or [],
                             stretch=stretch_factor or 1.0, bpm=bpm,
                             limit=max_pairings)
    return [
        _pairing_row(
            r["vocal_section_start"], r["vocal_section_end"],
            r.get("vocal_section_label"),
            r["inst_section_start"], r["inst_section_end"],
            r.get("inst_section_label"),
            stretch_factor or 1.0,
            # Carried through so a caller can resolve the pairing back to the
            # rows it came from without re-matching on start_sec.
            extra={"vocal_section_idx": r.get("vocal_section_idx"),
                   "inst_section_idx": r.get("inst_section_idx"),
                   "score_section": r.get("score_section"),
                   "section_note": r.get("section_note")},
        )
        for r in rows
    ]


def _pinned_pairing(vocal_sections: List[Dict], inst_sections: List[Dict],
                    vocal_section_idx: int, inst_section_idx: int,
                    stretch_factor: float) -> Optional[Dict]:
    """The pairing for two EXPLICIT section indices, or None if either is gone.

    This is what makes an export reproducible: the candidate row already decided
    which chorus goes over which drop, and re-deriving that choice at render
    time is how the exported folder stopped matching the row the user judged.
    """
    v = next((s for s in vocal_sections or []
              if s.get("section_index") == vocal_section_idx), None)
    i = next((s for s in inst_sections or []
              if s.get("section_index") == inst_section_idx), None)
    if v is None or i is None:
        return None
    return _pairing_row(
        float(v.get("start_sec") or 0.0), float(v.get("end_sec") or 0.0),
        v.get("label"),
        float(i.get("start_sec") or 0.0), float(i.get("end_sec") or 0.0),
        i.get("label"),
        stretch_factor or 1.0,
        extra={"vocal_section_idx": vocal_section_idx,
               "inst_section_idx": inst_section_idx, "pinned": True},
    )


def _section_for(sections: List[Dict], pairing: Dict, side: str) -> Optional[Dict]:
    """Resolve one side of a pairing back to its `sections` row.

    By index when the pairing carries one (every pairing does now), falling back
    to the start time for a pairing built before the index travelled with it.
    Matching on start_sec alone is fragile: it is a rounded float, and two
    sections of a re-analysed track can round to the same value.
    """
    idx = pairing.get(f"{side}_section_idx")
    if idx is not None:
        hit = next((s for s in sections or []
                    if s.get("section_index") == idx), None)
        if hit is not None:
            return hit
    return next((s for s in sections or []
                 if s.get("start_sec") == pairing[f"{side}_start"]), None)


def build_mashup_plan(vocal_song_id: int, inst_song_id: int,
                      db_path=None, *,
                      vocal_section_idx: Optional[int] = None,
                      inst_section_idx: Optional[int] = None,
                      harmonic_shift: Optional[int] = None,
                      combo_type: str = "vocal_over_instrumental",
                      ) -> Optional[Dict]:
    """Full actionable plan for one vocal-over-instrumental pair.
    Returns None when either song is missing.

    `vocal_section_idx` / `inst_section_idx` pin the pairing to the exact
    sections a candidate row was scored for, instead of re-choosing them here.
    Pass them whenever the plan is being built FOR a row the user has seen —
    the ranked list, the Plan expander, an FL export — or the plan will describe
    a different moment than the one that was auditioned. `harmonic_shift` does
    the same for the measured transpose: the row already cross-correlated those
    two sections' chroma, and re-deriving a Camelot shift here throws that away.

    Unresolvable indices (the track was re-analysed and the section is gone)
    fall through to the default pick rather than failing — a stale pin should
    cost you the exact moment, not the export.

    `combo_type` decides which stem the top layer contributes: on the
    instrumental-over-instrumental path the "vocal" side is an instrumental.
    """
    from database.models import (
        DB_PATH, get_conn, get_features_for_song, get_sections, get_song,
    )

    db = db_path or DB_PATH
    v_song = get_song(vocal_song_id, db_path=db)
    i_song = get_song(inst_song_id, db_path=db)
    if not v_song or not i_song:
        return None

    # Tempo and key come from the full mix where it exists (P0.3) — the same
    # swap the scorer applies via matcher.match._with_full_bpm. Without it the
    # recipe would print a target BPM and a semitone shift derived from the
    # acapella's own key estimate, which is the least reliable number in the
    # database, while the ranked row above it used the full-mix one. Two
    # different answers to "what key is this" on the same screen.
    from matcher.match import _with_full_bpm

    # Which stem the TOP layer contributes. On the instrumental-over-
    # instrumental path the "vocal" side is an instrumental (matcher.match
    # reuses the columns — see _emit), so reading its acapella would describe a
    # stem that is not in the mashup: wrong band occupancy under a collision
    # score measured between two beds, and a key off a discarded vocal.
    top_stem = ("instrumental"
                if combo_type == "instrumental_over_instrumental" else "vocals")

    v_full = get_features_for_song(vocal_song_id, "full", db_path=db) or {}
    i_full = get_features_for_song(inst_song_id, "full", db_path=db) or {}
    v_feat = get_features_for_song(vocal_song_id, top_stem, db_path=db) or v_full or {}
    i_feat = get_features_for_song(inst_song_id, "instrumental", db_path=db) \
        or i_full or {}
    if v_full:
        v_feat = _with_full_bpm({**v_feat, "song_id": vocal_song_id},
                                {vocal_song_id: v_full})
    if i_full:
        i_feat = _with_full_bpm({**i_feat, "song_id": inst_song_id},
                                {inst_song_id: i_full})

    v_bpm = v_feat.get("bpm") or 0.0
    i_bpm = i_feat.get("bpm") or 0.0
    i_bpm_eff = effective_bpm(v_bpm, i_bpm)
    stretch = compute_stretch_factor(v_bpm, i_bpm)
    shift = compute_semitone_shift(v_feat.get("camelot") or "",
                                   i_feat.get("camelot") or "")

    v_sections = get_sections(vocal_song_id, db_path=db)
    i_sections = get_sections(inst_song_id, db_path=db)

    # A pinned pair is the ONLY pairing: the caller is describing one specific
    # row, and offering three alternatives underneath it would put the same
    # ambiguity back that the pin exists to remove.
    pinned = None
    if vocal_section_idx is not None and inst_section_idx is not None:
        pinned = _pinned_pairing(v_sections, i_sections, vocal_section_idx,
                                 inst_section_idx, stretch or 1.0)
    if pinned is not None:
        pairings = [pinned]
    else:
        pairings = build_pairings(v_sections, i_sections, stretch or 1.0,
                                  bpm=v_bpm or None)

    # Phase E: prefer the MEASURED transpose over the Camelot-derived one.
    # Camelot says whether two scales are compatible; cross-correlating the two
    # sections' chroma says what actually makes the notes line up, and hands
    # back a bass-clash warning as a by-product. Falls back to the Camelot
    # estimate when either section has no stored chroma.
    # The two sections the plan is ABOUT, resolved once — the harmony below and
    # the per-section keys further down are both asking about this same moment.
    v_sec = _section_for(v_sections, pairings[0], "vocal") if pairings else None
    i_sec = _section_for(i_sections, pairings[0], "inst") if pairings else None

    harmony = None
    if pairings:
        from matcher.harmony import section_harmony
        h = section_harmony(v_sec, i_sec)
        if h["known"]:
            harmony = h
            shift = h["shift"]

    # An explicitly supplied shift wins over both. It came from the candidate
    # row, which measured it on these same two sections during scoring; letting
    # the re-measurement above override it is how the exported folder ends up
    # transposed differently from the pair that was auditioned.
    #
    # `harmony` is corrected alongside it, not just `shift`: the two are rendered
    # on the same screen — the recipe reads `semitone_shift` while the harmony
    # line reads `harmony.shift` — so leaving them to disagree would print two
    # different transposes for one row, with the "measured from the two
    # sections' chroma" wording attached to whichever one won.
    if harmonic_shift is not None:
        shift = int(harmonic_shift)
        if harmony is not None and harmony.get("shift") != shift:
            harmony = {**harmony, "shift": shift}

    # E.2 — the chosen sections' OWN key, not the track's.
    #
    # A track has one key only in the sense that an average has one value: real
    # records modulate, and the chorus is frequently not the key the whole-track
    # mean reports. detect_sections has stored a per-section Krumhansl estimate
    # since Phase E and nothing ever showed it, so "the track is 8A but this
    # chorus is 3B" — the reason a pair that looks compatible is not — was
    # invisible on the one screen where it decides what you do next.
    section_keys = None
    if pairings:
        section_keys = {
            side: (None if sec is None else {
                "key": sec.get("key"), "mode": sec.get("mode"),
                "camelot": sec.get("camelot"),
                "key_confidence": sec.get("key_confidence"),
                "label": sec.get("label"),
                # Whether it disagrees with the whole-track estimate, which is
                # the part worth reading: agreement is the boring case.
                "differs_from_track": bool(
                    sec.get("camelot") and track_camelot
                    and sec.get("camelot") != track_camelot),
            })
            for side, sec, track_camelot in (
                ("vocal", v_sec, v_feat.get("camelot")),
                ("inst", i_sec, i_feat.get("camelot")),
            )
        }

    # Stem file paths for drag-and-drop into the DAW.
    conn = get_conn(db)
    stem_rows = conn.execute(
        """SELECT song_id, stem_type, file_path FROM stems
           WHERE (song_id=? AND stem_type='vocals')
              OR (song_id=? AND stem_type='instrumental')""",
        (vocal_song_id, inst_song_id),
    ).fetchall()
    conn.close()
    paths = {(r["song_id"], r["stem_type"]): r["file_path"] for r in stem_rows}

    def _side(song: dict, feat: dict) -> dict:
        return {
            "song_id": song["id"],
            "title": song.get("title"),
            "artist": song.get("artist"),
            "genre": song.get("genre"),
            "release_year": song.get("release_year"),
            "plays": song.get("plays"),
            "likes": song.get("likes"),
            "bpm": feat.get("bpm"),
            "key": feat.get("key"),
            "mode": feat.get("mode"),
            "camelot": feat.get("camelot"),
            # E.1 — where this stem sits in the spectrum, 8 log-spaced bands
            # each a fraction of total energy. Measured since Phase D, feeding
            # collision_score, and never drawn: the number said "these two
            # fight" without ever saying WHERE, which is the only part you can
            # act on. None when the stem predates the measurement.
            "band_energy": feat.get("band_energy"),
            # On a bed, how much of it is still voice — an instrumental cut from
            # a full record carries the original topline, and two competing
            # melodies is a different problem from a frequency clash.
            "residual_vocal_ratio": feat.get("residual_vocal_ratio"),
        }

    steps = []
    steps.append(
        f"1. Import vocal stem of \"{v_song.get('title')}\" and instrumental "
        f"stem of \"{i_song.get('title')}\" into your DAW."
    )
    if v_bpm:
        steps.append(
            f"2. Set project tempo to {v_bpm:.1f} BPM. "
            + (f"Stretch the instrumental from {i_bpm_eff:.1f} BPM "
               f"(factor {stretch:.4f}x)." if stretch else
               "Instrumental BPM unknown — beat-match by ear.")
        )
    measured = " (measured from the two sections' chroma, not inferred " \
               "from the Camelot wheel)" if harmony else ""
    if shift is not None and shift != 0:
        steps.append(
            f"3. Pitch the instrumental {shift:+d} semitones to match the "
            f"vocal key ({v_feat.get('key')} {v_feat.get('mode')}){measured}."
        )
    else:
        steps.append(f"3. Keys already align — no pitch shift needed{measured}.")
    if harmony and harmony.get("advice"):
        steps.append(f"3b. {harmony['advice'].capitalize()}.")
    if pairings:
        for n, p in enumerate(pairings, start=4):
            steps.append(f"{n}. {p['note']} "
                         f"(vocal {p['vocal_duration']}s vs "
                         f"inst {p['inst_duration_stretched']}s after stretch).")
    else:
        steps.append("4. No section data yet — run analysis on both tracks to "
                     "get chorus/verse timestamps for cut suggestions.")

    return {
        "vocal": _side(v_song, v_feat),
        "inst": _side(i_song, i_feat),
        "target_bpm": v_bpm or None,
        "inst_effective_bpm": i_bpm_eff or None,
        "stretch_factor": stretch,
        "semitone_shift": shift,
        "key_relation": _key_relation(v_feat.get("camelot") or "",
                                      i_feat.get("camelot") or ""),
        "harmony": harmony,
        "section_keys": section_keys,
        # The band edges the two occupancy vectors are measured against, so the
        # UI can label them in Hz rather than as "band 4".
        "band_edges": list(BAND_EDGES),
        "vocal_sections": v_sections,
        "inst_sections": i_sections,
        "pairings": pairings,
        "steps": steps,
        "files": {
            "vocals": paths.get((vocal_song_id, "vocals")),
            "instrumental": paths.get((inst_song_id, "instrumental")),
        },
    }
