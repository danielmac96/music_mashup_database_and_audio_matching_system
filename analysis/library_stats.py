"""
analysis/library_stats.py — the Analysis dashboard's numbers, in one pass per
table.

Reads stored rows only (songs, full-mix features, sections, stems,
mashup_candidates, pair_feedback); nothing is recomputed and there is no
per-track query, so a 1 000-track library answers in well under 300 ms.
Unmeasured values are left OUT of a histogram, never binned as 0, and a track
without the measurement a filter asks about does not pass that filter — as in
the Library's own filters.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

BPM_LO, BPM_HI, BPM_STEP = 60, 200, 2
HIST_BINS = 20
TOP_N = 15
# A top-N tag list counts for a track when the value is among its top 3 — the
# Library's Attribute filter reads moods/themes the same way.
MOOD_TOP_PER_TRACK = 3
CAMELOT_CODES = [f"{n}{letter}" for n in range(1, 13) for letter in "AB"]
_CAMELOT_RE = re.compile(r"(1[0-2]|[1-9])([AB])")


def norm_camelot(code) -> Optional[str]:
    m = _CAMELOT_RE.fullmatch(str(code or "").strip().upper())
    return f"{m.group(1)}{m.group(2)}" if m else None


def _tags(raw) -> dict:
    if not raw:
        return {}
    try:
        tags = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    return tags if isinstance(tags, dict) else {}


def _labels(node, n: int) -> List[str]:
    out = []
    for x in (node if isinstance(node, list) else [])[:n]:
        label = x.get("label") if isinstance(x, dict) else None
        if label:
            out.append(str(label).split("---", 1)[-1])
    return out


def _hist(values: Sequence[float], bins: int = HIST_BINS) -> List[Dict]:
    """`bins` equal bins over the observed range; [] when nothing is measured."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        counts[min(bins - 1, int((v - lo) / width))] += 1
    return [{"lo": round(lo + i * width, 4), "hi": round(lo + (i + 1) * width, 4),
             "n": counts[i]} for i in range(bins)]


def _top(counts: Dict[str, int], n: Optional[int] = None) -> List[Dict]:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].casefold()))
    return [{"value": k, "n": v} for k, v in (ranked[:n] if n else ranked)]


def clean_filters(group=None, genre="", style="", bpm_min=None, bpm_max=None,
                  year_min=None, year_max=None, camelot="", vocal_forward=None) -> Dict:
    """The filters as they will be applied and echoed. ValueError on a Camelot
    code that is not one."""
    codes = []
    for part in str(camelot or "").split(","):
        if not part.strip():
            continue
        code = norm_camelot(part)
        if code is None:
            raise ValueError("camelot must be a comma list of codes like 8A")
        if code not in codes:
            codes.append(code)
    return {
        "group": None if group is None else int(group),
        "genre": (genre or "").strip() or None,
        "style": (style or "").strip() or None,
        "bpm_min": None if bpm_min is None else float(bpm_min),
        "bpm_max": None if bpm_max is None else float(bpm_max),
        "year_min": None if year_min is None else int(year_min),
        "year_max": None if year_max is None else int(year_max),
        "camelot": codes,
        "vocal_forward": None if vocal_forward is None else bool(vocal_forward),
    }


def library_stats(filters: Optional[Dict] = None, db_path=None) -> Optional[Dict]:
    """The dashboard payload for the tracks passing `filters` (clean_filters).
    None when `group` names a crate that does not exist."""
    from database import models
    f = filters or clean_filters()
    conn = models.get_conn(db_path if db_path is not None else models.DB_PATH)
    try:
        group_ids = None
        if f["group"] is not None:
            if not conn.execute("SELECT 1 FROM crates WHERE id=?",
                                (f["group"],)).fetchone():
                return None
            group_ids = {r[0] for r in conn.execute(
                "SELECT song_id FROM crate_items "
                "WHERE crate_id=? AND song_id IS NOT NULL", (f["group"],))}
        songs = conn.execute(
            """SELECT s.id, s.genre, s.release_year, s.status, s.pairs_input_at,
                      ft.bpm, ft.camelot, ft.energy, ft.lufs, ft.tags_json
               FROM songs s
               LEFT JOIN features ft ON ft.song_id = s.id AND ft.stem_type = 'full'
            """).fetchall()
        # The sung share is the Library's VOX% (api/routes/tracks.py
        # vocal_coverage): seconds in vocal/mixed sections over all seconds,
        # unmeasured until a section has a class.
        sections = {r["song_id"]: r for r in conn.execute(
            """SELECT song_id,
                      SUM(MAX(0, COALESCE(end_sec, 0) - COALESCE(start_sec, 0))) AS total,
                      SUM(CASE WHEN section_class IN ('vocal', 'mixed')
                               THEN MAX(0, end_sec - start_sec) ELSE 0 END)     AS sung,
                      SUM(section_class IS NOT NULL AND section_class != 'unknown') AS known,
                      MAX(vocal_presence) AS max_presence
               FROM sections GROUP BY song_id""")}
        stemmed = {r[0] for r in conn.execute(
            "SELECT DISTINCT song_id FROM stems WHERE stem_type = 'vocals'")}

        chosen: List[dict] = []
        for s in songs:
            sid = s["id"]
            if group_ids is not None and sid not in group_ids:
                continue
            tags = _tags(s["tags_json"])
            styles = _labels(tags.get("genre"), 1)
            sec = sections.get(sid)
            share = (sec["sung"] / sec["total"]
                     if sec and sec["known"] and sec["total"] and sec["total"] > 0 else None)
            row = {"id": sid, "genre": (s["genre"] or "").strip() or None,
                   "year": s["release_year"] if (s["release_year"] or 0) > 0 else None,
                   "status": s["status"], "pairs_input_at": s["pairs_input_at"],
                   "bpm": s["bpm"] if s["bpm"] else None,
                   "camelot": norm_camelot(s["camelot"]),
                   "energy": s["energy"], "lufs": s["lufs"],
                   "tagged": bool(tags), "style": styles[0] if styles else None,
                   "moods": _labels(tags.get("moodtheme"), MOOD_TOP_PER_TRACK),
                   "vocal_share": share,
                   "max_presence": sec["max_presence"] if sec else None}
            if _passes(row, f):
                chosen.append(row)
        ids = {r["id"] for r in chosen}
        pairs = _pair_counts(conn, ids, filtered=len(ids) != len(songs), models=models,
                             chosen=chosen, db_path=db_path)
    finally:
        conn.close()

    bpm_bins = [0] * ((BPM_HI - BPM_LO) // BPM_STEP)
    wheel = {c: 0 for c in CAMELOT_CODES}
    genres: Dict[str, int] = {}
    styles_n: Dict[str, int] = {}
    moods: Dict[str, int] = {}
    years: Dict[int, int] = {}
    scatter = []
    for r in chosen:
        if r["bpm"] is not None and BPM_LO <= r["bpm"] < BPM_HI:
            bpm_bins[int((r["bpm"] - BPM_LO) // BPM_STEP)] += 1
        if r["camelot"]:
            wheel[r["camelot"]] += 1
        if r["genre"]:
            genres[r["genre"]] = genres.get(r["genre"], 0) + 1
        if r["style"]:
            styles_n[r["style"]] = styles_n.get(r["style"], 0) + 1
        for m in r["moods"]:
            moods[m] = moods.get(m, 0) + 1
        if r["year"]:
            years[r["year"]] = years.get(r["year"], 0) + 1
        if r["energy"] is not None and r["lufs"] is not None:
            scatter.append({"song_id": r["id"], "energy": round(r["energy"], 5),
                            "lufs": round(r["lufs"], 2), "camelot": r["camelot"],
                            "bpm": None if r["bpm"] is None else round(r["bpm"], 2)})
    return {
        "filters": f,
        "counts": {"tracks": len(chosen),
                   "analysed": sum(1 for r in chosen if r["status"] == "analysed"),
                   "with_stems": sum(1 for r in chosen if r["id"] in stemmed),
                   "with_tags": sum(1 for r in chosen if r["tagged"])},
        "bpm_hist": [{"lo": BPM_LO + i * BPM_STEP, "hi": BPM_LO + (i + 1) * BPM_STEP,
                      "n": n} for i, n in enumerate(bpm_bins)],
        "camelot_wheel": [{"code": c, "n": wheel[c]} for c in CAMELOT_CODES],
        "genre_top": _top(genres, TOP_N),
        "style_top": _top(styles_n, TOP_N),
        "mood_top": _top(moods, TOP_N),
        "year_hist": [{"year": y, "n": years[y]} for y in sorted(years)],
        "energy_hist": _hist([r["energy"] for r in chosen]),
        "lufs_hist": _hist([r["lufs"] for r in chosen]),
        "vocal_share_hist": _hist([r["vocal_share"] for r in chosen]),
        "scatter": scatter,
        "pairs": pairs,
    }


def _passes(r: dict, f: Dict) -> bool:
    if f["genre"] and f["genre"].casefold() not in (r["genre"] or "").casefold():
        return False
    if f["style"] and (r["style"] or "").casefold() != f["style"].casefold():
        return False
    if f["bpm_min"] is not None and (r["bpm"] is None or r["bpm"] < f["bpm_min"]):
        return False
    if f["bpm_max"] is not None and (r["bpm"] is None or r["bpm"] > f["bpm_max"]):
        return False
    if f["year_min"] is not None and (r["year"] is None or r["year"] < f["year_min"]):
        return False
    if f["year_max"] is not None and (r["year"] is None or r["year"] > f["year_max"]):
        return False
    if f["camelot"] and r["camelot"] not in f["camelot"]:
        return False
    if f["vocal_forward"] is not None:
        # The listing's vocal-forward test, per track: its most vocal section.
        from database.models import VOCAL_FORWARD_MIN
        if r["max_presence"] is None:
            return False
        if (r["max_presence"] >= VOCAL_FORWARD_MIN) != f["vocal_forward"]:
            return False
    return True


def _pair_counts(conn, ids: set, filtered: bool, models, chosen, db_path) -> Dict:
    """Scored rows and your verdicts among the chosen tracks (both sides in the
    set), and how many of them changed since the last Score library run."""
    if not filtered:
        scored = conn.execute("SELECT COUNT(*) FROM mashup_candidates").fetchone()[0]
    else:
        scored = sum(r["n"] for r in conn.execute(
            "SELECT vocal_song_id AS v, inst_song_id AS i, COUNT(*) AS n "
            "FROM mashup_candidates GROUP BY v, i")
            if r["v"] in ids and r["i"] in ids)
    rated = loved = rejected = 0
    for r in conn.execute(
            "SELECT vocal_song_id AS v, inst_song_id AS i, verdict, rating FROM pair_feedback"):
        if r["v"] not in ids or r["i"] not in ids:
            continue
        rated += 1
        rating = r["rating"]
        if (rating is not None and rating >= 4) or (rating is None and r["verdict"] == "love"):
            loved += 1
        elif (rating is not None and rating <= 2) or (rating is None and r["verdict"] == "no"):
            rejected += 1
    row = conn.execute("SELECT value FROM app_prefs WHERE key='last_score'").fetchone()
    try:
        started = (json.loads(row["value"]) or {}).get("started_at") if row else None
    except (TypeError, ValueError):
        started = None
    stale = sum(1 for r in chosen if started and r["status"] == "analysed"
                and r["pairs_input_at"] and r["pairs_input_at"] > started)
    return {"scored": int(scored), "stale_tracks": stale, "rated": rated,
            "loved": loved, "rejected": rejected}
