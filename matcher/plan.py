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


def build_mashup_plan(vocal_song_id: int, inst_song_id: int,
                      db_path=None) -> Optional[Dict]:
    """Full actionable plan for one vocal-over-instrumental pair.
    Returns None when either song is missing."""
    from database.models import (
        DB_PATH, get_conn, get_features_for_song, get_sections, get_song,
    )

    db = db_path or DB_PATH
    v_song = get_song(vocal_song_id, db_path=db)
    i_song = get_song(inst_song_id, db_path=db)
    if not v_song or not i_song:
        return None

    v_feat = get_features_for_song(vocal_song_id, "vocals", db_path=db) \
        or get_features_for_song(vocal_song_id, "full", db_path=db) or {}
    i_feat = get_features_for_song(inst_song_id, "instrumental", db_path=db) \
        or get_features_for_song(inst_song_id, "full", db_path=db) or {}

    v_bpm = v_feat.get("bpm") or 0.0
    i_bpm = i_feat.get("bpm") or 0.0
    i_bpm_eff = effective_bpm(v_bpm, i_bpm)
    stretch = compute_stretch_factor(v_bpm, i_bpm)
    shift = compute_semitone_shift(v_feat.get("camelot") or "",
                                   i_feat.get("camelot") or "")

    v_sections = get_sections(vocal_song_id, db_path=db)
    i_sections = get_sections(inst_song_id, db_path=db)
    pairings = build_pairings(v_sections, i_sections, stretch or 1.0)

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
    if shift is not None and shift != 0:
        steps.append(
            f"3. Pitch the instrumental {shift:+d} semitones to match the "
            f"vocal key ({v_feat.get('key')} {v_feat.get('mode')})."
        )
    else:
        steps.append("3. Keys already align — no pitch shift needed.")
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
        "vocal_sections": v_sections,
        "inst_sections": i_sections,
        "pairings": pairings,
        "steps": steps,
        "files": {
            "vocals": paths.get((vocal_song_id, "vocals")),
            "instrumental": paths.get((inst_song_id, "instrumental")),
        },
    }
