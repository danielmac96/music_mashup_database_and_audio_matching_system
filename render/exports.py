"""
render/exports.py — chosen pairs as files other tools read.

Three formats over the same list of pair rows (a set's items, or the dock's
top N), so a set and a filtered dock export cannot disagree about what a pair
is:

  * CSV — one row per pair: both sides, sections, landing tempo and key, the
    bed's transpose, measured harmony, note. Opens in a spreadsheet.
  * cue sheet — plain text, one line per mashup with its start time in the
    running order. For planning a mix on paper or pasting into a tracklist.
  * rekordbox XML — the vocal's acapella and the bed's full track as a
    playlist, each with a beat grid from the analysis and a hot cue at the
    chosen section (memory cues at every section). rekordbox imports it via
    Preferences → Advanced → rekordbox xml. The files are referenced where they
    are, not copied. Serato and Traktor are not written.

A pair row is the dict `get_candidates_enriched` / `get_set` returns.
"""
from __future__ import annotations

import csv
import io
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import quote
from xml.sax.saxutils import quoteattr

from matcher.setflow import flow

CSV_COLUMNS = (
    "position", "vocal_artist", "vocal_title", "vocal_section", "vocal_start",
    "vocal_end", "bed_artist", "bed_title", "bed_section", "bed_start", "bed_end",
    "landing_bpm", "landing_key", "bed_shift_st", "bed_bpm", "bed_key",
    "harmony_pct", "bass_clash", "nudge_ms", "loop_bed_x", "score_pct",
    "set_start", "note",
)


def mmss(secs: Optional[float]) -> str:
    if secs is None:
        return ""
    n = max(0, int(round(secs)))
    return f"{n // 60}:{n % 60:02d}"


def _rows(items: Sequence[Dict]) -> List[Dict]:
    f = flow(items)
    out = []
    for k, (it, land) in enumerate(zip(items, f["landings"])):
        fit = it.get("score_key") if it.get("harmonic_shift") is not None else None
        out.append({
            "position": k + 1,
            "vocal_artist": it.get("vocal_artist") or "",
            "vocal_title": it.get("vocal_title") or "",
            "vocal_section": it.get("vocal_section_label") or "",
            "vocal_start": mmss(it.get("vocal_section_start")),
            "vocal_end": mmss(it.get("vocal_section_end")),
            "bed_artist": it.get("inst_artist") or "",
            "bed_title": it.get("inst_title") or "",
            "bed_section": it.get("inst_section_label") or "",
            "bed_start": mmss(it.get("inst_section_start")),
            "bed_end": mmss(it.get("inst_section_end")),
            "landing_bpm": f"{land['bpm']:.1f}" if land["bpm"] else "",
            "landing_key": land["camelot"] or "",
            "bed_shift_st": "" if land["bed_shift"] is None else f"{land['bed_shift']:+d}",
            "bed_bpm": f"{it['inst_bpm']:.1f}" if it.get("inst_bpm") else "",
            "bed_key": it.get("inst_camelot") or "",
            "harmony_pct": "" if fit is None else str(round(float(fit) * 100)),
            "bass_clash": "yes" if it.get("bass_clash") else "",
            "nudge_ms": "" if it.get("alignment_offset") is None
                        else str(round(float(it["alignment_offset"]) * 1000)),
            "loop_bed_x": str(it.get("section_loop_repeats") or "")
                          if (it.get("section_loop_repeats") or 1) > 1 else "",
            "score_pct": "" if it.get("score_percentile") is None
                         else str(round(float(it["score_percentile"]) * 100)),
            "set_start": mmss(f["starts"][k]),
            "note": it.get("note") or "",
        })
    return out


def pairs_csv(items: Sequence[Dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in _rows(items):
        w.writerow(r)
    return buf.getvalue()


def cue_sheet(items: Sequence[Dict], title: str = "") -> str:
    """One line per mashup, timed from the top of the set."""
    rows = _rows(items)
    f = flow(items)
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(f"{len(rows)} mashups · {mmss(f['total_secs'])} of vocal sections")
    lines.append("")
    for r, t in zip(rows, [None] + f["transitions"]):
        if t is not None:
            tempo = "" if t["tempo_pct"] is None else f"{t['tempo_pct']:+.1f}% tempo"
            key = "" if t["key_steps"] is None else f"{t['key_steps']:g} key steps"
            lines.append(f"      ↓ {t['grade']}" + (f" ({', '.join(x for x in (tempo, key) if x)})"
                                                    if tempo or key else ""))
        head = (f"[{r['set_start']:>5}] {r['position']:>2}. "
                f"{r['vocal_artist']} - {r['vocal_title']} ({r['vocal_section']} {r['vocal_start']}–{r['vocal_end']})")
        lines.append(head)
        bed = (f"            w/ {r['bed_artist']} - {r['bed_title']} ({r['bed_section']} "
               f"{r['bed_start']}–{r['bed_end']})")
        lines.append(bed)
        detail = [x for x in (
            r["landing_bpm"] and f"{r['landing_bpm']} BPM",
            r["landing_key"],
            r["bed_shift_st"] and r["bed_shift_st"] != "+0" and f"bed {r['bed_shift_st']} st",
            r["harmony_pct"] and f"harmony {r['harmony_pct']}%",
            r["loop_bed_x"] and f"loop the bed ×{r['loop_bed_x']}",
            r["bass_clash"] and "bass clash: high-pass the bed",
            r["note"] and f"note: {r['note']}",
        ) if x]
        if detail:
            lines.append("            " + " · ".join(detail))
    return "\n".join(lines) + "\n"


# ── rekordbox ────────────────────────────────────────────────────────────────

def _tonality(key: Optional[str], mode: Optional[str]) -> str:
    if not key:
        return ""
    return f"{key}m" if (mode or "").lower().startswith("min") else key


def _file_url(path: str, base: Optional[str] = None, root: Optional[str] = None) -> str:
    """rekordbox wants file://localhost/<absolute path>. With `base`, the
    library root (`root`) at the front of the path is swapped for it, so a
    library inside Docker (/data/audio/...) can be addressed as the folder it
    is mounted from on the host (/Users/me/mashup/data/audio/...)."""
    full = str(path).replace("\\", "/")
    if base and root:
        r = str(root).replace("\\", "/").rstrip("/")
        if full.startswith(r + "/"):
            full = base.replace("\\", "/").rstrip("/") + full[len(r):]
    if not full.startswith("/"):
        full = "/" + full          # C:/Music -> /C:/Music
    return "file://localhost" + quote(full, safe="/:")


def rekordbox_xml(items: Sequence[Dict], playlist: str,
                  resolve: Callable[[int, str], Optional[Dict]],
                  base: Optional[str] = None, root: Optional[str] = None) -> Dict:
    """The XML and the audio files it references.

    `resolve(song_id, stem)` returns {"path", "title", "artist", "bpm", "key",
    "mode", "duration", "beat_times", "beat_phase", "sections"} or None when the
    audio is not on disk. Returns {"xml": str, "files": [paths], "skipped": n}.
    """
    tracks: List[Dict] = []
    order: List[int] = []
    index: Dict[tuple, int] = {}
    skipped = 0

    def add(song_id: int, stem: str, label: str, cue_at: Optional[float],
            cue_name: str) -> None:
        nonlocal skipped
        key = (song_id, stem)
        if key not in index:
            info = resolve(song_id, stem)
            if not info:
                skipped += 1
                return
            index[key] = len(tracks) + 1
            tracks.append({**info, "label": label, "cues": [], "tid": index[key]})
        t = tracks[index[key] - 1]
        if cue_at is not None and len(t["cues"]) < 8:
            t["cues"].append((cue_name, float(cue_at)))
        order.append(t["tid"])

    for k, it in enumerate(items):
        n = k + 1
        add(it["vocal_song_id"], "vocals", "Acapella", it.get("vocal_section_start"),
            f"#{n} VOX {it.get('vocal_section_label') or ''}".strip())
        add(it["inst_song_id"], "full", "", it.get("inst_section_start"),
            f"#{n} BED {it.get('inst_section_label') or ''}".strip())

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<DJ_PLAYLISTS Version="1.0.0">',
           '  <PRODUCT Name="Mashup Engine" Version="1.0" Company=""/>',
           f'  <COLLECTION Entries="{len(tracks)}">']
    for t in tracks:
        name = t["title"] + (f" ({t['label']})" if t["label"] else "")
        attrs = {
            "TrackID": str(t["tid"]), "Name": name, "Artist": t.get("artist") or "",
            "Kind": "Audio File", "TotalTime": str(int(round(t.get("duration") or 0))),
            "AverageBpm": f"{t['bpm']:.2f}" if t.get("bpm") else "0.00",
            "Tonality": _tonality(t.get("key"), t.get("mode")),
            "Location": _file_url(str(t["path"]), base, root),
            "Comments": "Mashup Engine export",
        }
        out.append("    <TRACK " + " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items()) + ">")
        beats = t.get("beat_times") or []
        if t.get("bpm") and beats:
            phase = int(t.get("beat_phase") or 0) % 4
            first = beats[phase] if phase < len(beats) else beats[0]
            out.append(f'      <TEMPO Inizio="{first:.3f}" Bpm="{t["bpm"]:.2f}" '
                       f'Metro="4/4" Battito="1"/>')
        for num, (cname, at) in enumerate(t["cues"]):
            out.append(f'      <POSITION_MARK Name={quoteattr(cname)} Type="0" '
                       f'Start="{at:.3f}" Num="{num}" Red="40" Green="226" Blue="20"/>')
        for sec in t.get("sections") or []:
            out.append(f'      <POSITION_MARK Name={quoteattr(sec.get("label") or "")} '
                       f'Type="0" Start="{float(sec["start_sec"]):.3f}" Num="-1"/>')
        out.append("    </TRACK>")
    out.append("  </COLLECTION>")
    out.append('  <PLAYLISTS>')
    out.append('    <NODE Type="0" Name="ROOT" Count="1">')
    out.append(f'      <NODE Name={quoteattr(playlist)} Type="1" KeyType="0" Entries="{len(order)}">')
    for tid in order:
        out.append(f'        <TRACK Key="{tid}"/>')
    out.append("      </NODE>")
    out.append("    </NODE>")
    out.append("  </PLAYLISTS>")
    out.append("</DJ_PLAYLISTS>")
    return {"xml": "\n".join(out) + "\n", "files": [t["path"] for t in tracks],
            "skipped": skipped}


def library_resolver(db_path=None) -> Callable[[int, str], Optional[Dict]]:
    """resolve() for rekordbox_xml over the live library."""
    from database import models

    db = db_path if db_path is not None else models.DB_PATH

    def resolve(song_id: int, stem: str) -> Optional[Dict]:
        path = models.resolve_audio_path(song_id, stem, db_path=db)
        if path is None:
            return None
        song = models.get_song(song_id, db_path=db) or {}
        full = models.get_features_for_song(song_id, "full", db_path=db) or {}
        return {"path": str(path), "title": song.get("title") or f"song {song_id}",
                "artist": song.get("artist") or "", "bpm": full.get("bpm"),
                "key": full.get("key"), "mode": full.get("mode"),
                "duration": song.get("duration_secs"),
                "beat_times": full.get("beat_times") or [],
                "beat_phase": full.get("beat_phase") or 0,
                "sections": models.get_sections(song_id, db_path=db)}
    return resolve
