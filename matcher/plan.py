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

from matcher.patterns import priority_for
from matcher.recipe import (
    GAIN_MIN_DB, VOCAL_LANE_GAIN, bed_gain_db, bed_lane_gain,
)

# Section label priority when choosing what to mash — now DERIVED from the
# configured mashup patterns (matcher/patterns.py) rather than hard-coded here.
# Same shape as before, so _pick_sections and matcher.sections.usable_sections
# are unchanged; the patterns are simply where the ordering now comes from.
#
# These module-level snapshots exist because both are imported by name
# elsewhere. Prefer current_priority() below in new code: patterns are read live
# from settings.json, and a module constant freezes whatever they were at import.
_VOCAL_LABEL_PRIORITY = priority_for(vocal_side=True)
_INST_LABEL_PRIORITY = priority_for(vocal_side=False)


# How many timing options the plan offers. The Studio renders one pill per
# option, so this is a UI budget as much as a compute one; six fits the toolbar
# and is more moments than anyone auditions in one sitting.
SECTION_OPTION_LIMIT = 6


def current_priority(vocal_side: bool) -> Dict[str, int]:
    """Live label priority for one side, honouring edited patterns."""
    return priority_for(vocal_side=vocal_side)


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


def build_pairings(vocal_sections: List[Dict], inst_sections: List[Dict],
                   stretch_factor: float, max_pairings: int = 4) -> List[Dict]:
    """Match vocal sections to instrumental sections by label priority and
    duration fit (after the instrumental is stretched to the vocal tempo)."""
    v_use = _pick_sections(vocal_sections, _VOCAL_LABEL_PRIORITY, vocal_side=True)
    i_use = _pick_sections(inst_sections, _INST_LABEL_PRIORITY, vocal_side=False)
    if not v_use or not i_use:
        return []

    pairings = []
    for v in v_use[:max_pairings]:
        v_dur = (v["end_sec"] - v["start_sec"])
        best = min(
            i_use,
            key=lambda i: (
                _INST_LABEL_PRIORITY.get(i.get("label") or "verse", 9),
                abs((i["end_sec"] - i["start_sec"]) / max(stretch_factor, 1e-6) - v_dur),
            ),
        )
        i_dur_stretched = (best["end_sec"] - best["start_sec"]) / max(stretch_factor, 1e-6)
        pairings.append({
            "vocal_label": v.get("label"),
            "vocal_start": v["start_sec"],
            "vocal_end": v["end_sec"],
            "vocal_duration": round(v_dur, 1),
            "inst_label": best.get("label"),
            "inst_start": best["start_sec"],
            "inst_end": best["end_sec"],
            "inst_duration_stretched": round(i_dur_stretched, 1),
            "note": (
                f"Lay vocal {v.get('label')} ({_fmt_ts(v['start_sec'])}–{_fmt_ts(v['end_sec'])}) "
                f"over instrumental {best.get('label')} "
                f"({_fmt_ts(best['start_sec'])}–{_fmt_ts(best['end_sec'])})"
            ),
        })
    return pairings


def _section_at(sections: List[Dict], idx: Optional[int]) -> Optional[Dict]:
    if idx is None:
        return None
    return next((s for s in sections if s.get("section_index") == idx), None)


def _chosen_pairing(vocal_sections: List[Dict], inst_sections: List[Dict],
                    vocal_idx: Optional[int], inst_idx: Optional[int],
                    stretch_factor: float) -> Optional[Dict]:
    """One pairing, in build_pairings' shape, for an explicit section pair."""
    v = _section_at(vocal_sections, vocal_idx)
    i = _section_at(inst_sections, inst_idx)
    if not v or not i or v.get("start_sec") is None or i.get("start_sec") is None:
        return None
    v_dur = v["end_sec"] - v["start_sec"]
    i_dur_stretched = (i["end_sec"] - i["start_sec"]) / max(stretch_factor, 1e-6)
    return {
        "vocal_label": v.get("label"),
        "vocal_start": v["start_sec"],
        "vocal_end": v["end_sec"],
        "vocal_duration": round(v_dur, 1),
        "inst_label": i.get("label"),
        "inst_start": i["start_sec"],
        "inst_end": i["end_sec"],
        "inst_duration_stretched": round(i_dur_stretched, 1),
        "vocal_section_idx": vocal_idx,
        "inst_section_idx": inst_idx,
        "note": (
            f"Lay vocal {v.get('label')} ({_fmt_ts(v['start_sec'])}–{_fmt_ts(v['end_sec'])}) "
            f"over instrumental {i.get('label')} "
            f"({_fmt_ts(i['start_sec'])}–{_fmt_ts(i['end_sec'])})"
        ),
    }


# What _with_option_harmony adds to a top_section_pairs row: the option's own
# measured harmony, in the candidate row's column names, and the shift it plays.
OPTION_HARMONY_KEYS = ("harmonic_shift", "score_key", "harmonic_confidence",
                       "bass_clash", "harmony_advice", "semitone_shift")


def _with_option_harmony(opt: Dict, vocal_sections: List[Dict],
                         inst_sections: List[Dict],
                         camelot_shift: Optional[int]) -> None:
    """Annotate one timing option with its own measured harmony, in the column
    names a scored candidate row uses, plus the shift it plays."""
    from matcher.harmony import section_harmony

    h = section_harmony(_section_at(vocal_sections, opt.get("vocal_section_idx")),
                        _section_at(inst_sections, opt.get("inst_section_idx")))
    if h["known"]:
        opt["harmonic_shift"] = h["shift"]
        opt["score_key"] = round(h["harmonic_fit"], 4)
        opt["harmonic_confidence"] = round(h["confidence"], 4)
        opt["bass_clash"] = 1 if h["bass_clash"] else 0
        opt["harmony_advice"] = h["advice"]
    else:
        opt["harmonic_shift"] = None
    # recipe.bed_shift's rule, applied to an option (which carries no keys).
    opt["semitone_shift"] = (opt["harmonic_shift"]
                             if opt["harmonic_shift"] is not None else camelot_shift)


def build_mashup_plan(vocal_song_id: int, inst_song_id: int,
                      db_path=None,
                      vocal_section_idx: Optional[int] = None,
                      inst_section_idx: Optional[int] = None) -> Optional[Dict]:
    """Full actionable plan for one vocal-over-instrumental pair.
    Returns None when either song is missing.

    With both section indexes — the section pair a dock card, a set item or
    Studio's armed timing is about — that pairing comes first, so the harmony,
    the recipe and the FL export describe the moment the user chose rather than
    the label-priority pick below. Without them (or when either index no longer
    names a section), the plan is exactly what it was."""
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

    v_full = get_features_for_song(vocal_song_id, "full", db_path=db) or {}
    i_full = get_features_for_song(inst_song_id, "full", db_path=db) or {}
    v_feat = get_features_for_song(vocal_song_id, "vocals", db_path=db) or v_full or {}
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
    pairings = build_pairings(v_sections, i_sections, stretch or 1.0)
    chosen = _chosen_pairing(v_sections, i_sections, vocal_section_idx,
                             inst_section_idx, stretch or 1.0)
    if chosen:
        pairings = [chosen] + [
            p for p in pairings
            if (p["vocal_start"], p["inst_start"])
            != (chosen["vocal_start"], chosen["inst_start"])][:3]

    # The SCORED timing options — the same ranked section pairs the candidate
    # row itself is built from, so Discover's plan table, the Studio's timing
    # pills and the ranked list all describe the same moments. Imported here
    # rather than at module scope because matcher.sections imports THIS module.
    #
    # Reuse only: top_section_pairs feeds matcher.match's scoring loop, so
    # changing its selection would change the ranked list. Note it emits at most
    # one row per VOCAL section, so the options never offer the same chorus over
    # two different drops — that cap is what stops scoring multiplying the
    # candidates table, and relaxing it here would mean re-ranking by hand.
    from matcher.sections import top_section_pairs

    section_options = top_section_pairs(
        v_sections, i_sections, stretch or 1.0, bpm=v_bpm or None,
        limit=SECTION_OPTION_LIMIT,
    )

    # Each timing option carries its own measured harmony: the best transpose
    # belongs to a SECTION pair, so Studio's pills must not all borrow the
    # first pairing's. semitone_shift is what the option plays (recipe.bed_shift).
    camelot_shift = shift
    for opt in section_options:
        _with_option_harmony(opt, v_sections, i_sections, camelot_shift)

    # Phase E: prefer the MEASURED transpose over the Camelot-derived one.
    # Camelot says whether two scales are compatible; cross-correlating the two
    # sections' chroma says what actually makes the notes line up, and hands
    # back a bass-clash warning as a by-product. Falls back to the Camelot
    # estimate when either section has no stored chroma.
    harmony = None
    if pairings:
        from matcher.harmony import section_harmony
        v_sec = next((s for s in v_sections
                      if s.get("start_sec") == pairings[0]["vocal_start"]), None)
        i_sec = next((s for s in i_sections
                      if s.get("start_sec") == pairings[0]["inst_start"]), None)
        h = section_harmony(v_sec, i_sec)
        if h["known"]:
            harmony = h
            shift = h["shift"]

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
        }

    # The level the recipe arms the bed at: the loudness of the vocal record's
    # own instrumental (matcher/recipe.bed_gain_db).
    v_own_bed = get_features_for_song(vocal_song_id, "instrumental", db_path=db) or {}
    gain_db = bed_gain_db(v_own_bed.get("lufs"), i_feat.get("lufs"))

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
    if gain_db is not None and abs(gain_db) >= GAIN_MIN_DB:
        steps.append(
            f"3c. Set the instrumental {gain_db:+.1f} dB — the loudness of the "
            f"vocal's own instrumental ({v_own_bed.get('lufs'):.1f} LUFS, this one "
            f"{i_feat.get('lufs'):.1f}), so the vocal sits as it was mixed.")
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
        "bed_gain_db": gain_db,
        "vocal_lane_gain": VOCAL_LANE_GAIN,
        "bed_lane_gain": bed_lane_gain(gain_db),
        "vocal_sections": v_sections,
        "inst_sections": i_sections,
        "pairings": pairings,
        "section_options": section_options,
        "steps": steps,
        "files": {
            "vocals": paths.get((vocal_song_id, "vocals")),
            "instrumental": paths.get((inst_song_id, "instrumental")),
        },
    }
