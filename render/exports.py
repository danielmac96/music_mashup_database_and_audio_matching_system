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

A pair row is the dict `get_candidates_enriched` / `get_set` returns. A set
can also hold tracks, breaks and gaps (matcher/setflow): a track is a row with
only the bed side filled; a break is a row of its own (a cue with no track);
a gap — a slot not filled yet — is left out of every file, and rekordbox
leaves out breaks as well.
"""
from __future__ import annotations

import csv
import re
import io
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import quote
from xml.sax.saxutils import quoteattr

from matcher.setflow import flow, kind_of

CSV_COLUMNS = (
    "position", "vocal_artist", "vocal_title", "vocal_section", "vocal_start",
    "vocal_end", "bed_artist", "bed_title", "bed_section", "bed_start", "bed_end",
    "landing_bpm", "landing_key", "bed_shift_st", "bed_bpm", "bed_key",
    "harmony_pct", "bass_clash", "nudge_ms", "loop_bed_x", "score_pct",
    "set_start", "note", "move_in", "overlap_bars", "vocal_tempo_pct", "kind",
)


def mmss(secs: Optional[float]) -> str:
    if secs is None:
        return ""
    n = max(0, int(round(secs)))
    return f"{n // 60}:{n % 60:02d}"


def _blank() -> Dict:
    return {c: "" for c in CSV_COLUMNS}


def _rows(items: Sequence[Dict]) -> List[Dict]:
    """One row per slot that plays (pairs, tracks) or pauses (breaks), in
    running order; gaps are not written. `position` counts what is written."""
    f = flow(items)
    out = []
    moves = [None] + [t["move"] for t in f["transitions"]]
    n = 0
    for k, (it, land) in enumerate(zip(items, f["landings"])):
        kind = kind_of(it)
        if kind == "gap":
            continue
        n += 1
        if kind == "break":
            out.append({**_blank(), "position": n, "kind": "break",
                        "set_start": mmss(f["starts"][k]),
                        "note": it.get("label") or "break",
                        "vocal_end": mmss(land["secs"])})
            continue
        if kind == "track":
            out.append({**_blank(), "position": n, "kind": "track",
                        "bed_artist": it.get("artist") or "", "bed_title": it.get("title") or "",
                        "landing_bpm": f"{land['bpm']:.1f}" if land["bpm"] else "",
                        "landing_key": land["camelot"] or "",
                        "bed_bpm": f"{it['bpm']:.1f}" if it.get("bpm") else "",
                        "bed_key": it.get("camelot") or "",
                        "set_start": mmss(f["starts"][k])})
            continue
        move = moves[k]
        v_pct = None
        if it.get("set_bpm") and it.get("vocal_bpm"):
            from matcher.match import effective_bpm
            v_pct = (float(it["set_bpm"]) / effective_bpm(float(it["set_bpm"]),
                                                          float(it["vocal_bpm"])) - 1) * 100
        fit = it.get("score_key") if it.get("harmonic_shift") is not None else None
        out.append({
            "position": n,
            "kind": "pair",
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
            # How this mashup comes in (matcher/setflow: suggested or chosen),
            # and — on a tempo curve — how far the vocal is stretched to land.
            "move_in": move["label"] if move else "",
            "overlap_bars": str(move["bars"]) if move and move["bars"] else "",
            "vocal_tempo_pct": "" if v_pct is None or abs(v_pct) < 0.05 else f"{v_pct:+.1f}",
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
    """One line per mashup, timed from the top of the set. A track is one
    line; a break is a cue with no track; gaps are not printed."""
    f = flow(items)
    written = [k for k, it in enumerate(items) if kind_of(it) != "gap"]
    rows = _rows(items)
    trans = [None] + f["transitions"]
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    pairs = sum(1 for r in rows if r["kind"] == "pair")
    lines.append(f"{pairs} mashups · {mmss(f['total_secs'])} running time")
    lines.append("")
    for r, k in zip(rows, written):
        t = trans[k]
        if r["kind"] == "break":
            lines.append(f"[{r['set_start']:>5}] {r['position']:>2}. — {r['note']} "
                         f"({r['vocal_end']}) —")
            continue
        if r["kind"] == "track":
            lines.append(f"[{r['set_start']:>5}] {r['position']:>2}. {r['bed_artist']} - "
                         f"{r['bed_title']}" + (f" · {r['landing_bpm']} BPM" if r["landing_bpm"] else "")
                         + (f" · {r['landing_key']}" if r["landing_key"] else ""))
            continue
        if t is not None and t.get("grade") != "none":
            tempo = "" if t["tempo_pct"] is None else f"{t['tempo_pct']:+.1f}% tempo"
            key = "" if t["key_steps"] is None else f"{t['key_steps']:g} key steps"
            mv = t.get("move")
            how = ""
            if mv:
                how = f" — {mv['label']}" + (f" {mv['bars']} bars" if mv["bars"] else "")
            lines.append(f"      ↓ {t['grade']}" + (f" ({', '.join(x for x in (tempo, key) if x)})"
                                                    if tempo or key else "") + how)
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


def _collect(items: Sequence[Dict], resolve: Callable[[int, str], Optional[Dict]]):
    """The files a set or pair list plays, once each, in first-play order, with
    the cues each carries (≤ 8), and the playlist order as track ids — one
    collection for rekordbox, Serato and Traktor, so the three exports name
    the same files and cues. Returns (tracks, order, skipped)."""
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

    n = 0
    for it in items:
        kind = kind_of(it)
        if kind in ("break", "gap"):
            continue
        n += 1
        if kind == "track":
            add(it["song_id"], "full", "", None, "")
            continue
        add(it["vocal_song_id"], "vocals", "Acapella", it.get("vocal_section_start"),
            f"#{n} VOX {it.get('vocal_section_label') or ''}".strip())
        add(it["inst_song_id"], "full", "", it.get("inst_section_start"),
            f"#{n} BED {it.get('inst_section_label') or ''}".strip())

    return tracks, order, skipped


def rekordbox_xml(items: Sequence[Dict], playlist: str,
                  resolve: Callable[[int, str], Optional[Dict]],
                  base: Optional[str] = None, root: Optional[str] = None) -> Dict:
    """The XML and the audio files it references.

    `resolve(song_id, stem)` returns {"path", "title", "artist", "bpm", "key",
    "mode", "duration", "beat_times", "beat_phase", "sections"} or None when the
    audio is not on disk. Returns {"xml": str, "files": [paths], "skipped": n}.
    """
    tracks, order, skipped = _collect(items, resolve)
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


# ── Serato and Traktor ────────────────────────────────────────────────────────

def _local_path(path: str, base: Optional[str] = None, root: Optional[str] = None) -> str:
    """The file's path as the DJ software's machine sees it: the library root
    swapped for `base` (Docker's /data/audio → the host's folder), forward
    slashes."""
    full = str(path).replace("\\", "/")
    if base and root:
        r = str(root).replace("\\", "/").rstrip("/")
        if full.startswith(r + "/"):
            full = base.replace("\\", "/").rstrip("/") + full[len(r):]
    return full


def _serato_tag(name: bytes, payload: bytes) -> bytes:
    import struct
    return name + struct.pack(">I", len(payload)) + payload


def serato_crate(items: Sequence[Dict], resolve: Callable[[int, str], Optional[Dict]],
                 base: Optional[str] = None, root: Optional[str] = None) -> Dict:
    """A Serato DJ crate (.crate): a `vrsn` header, then one `otrk` per file
    holding its `ptrk` path, UTF-16BE, relative to the drive root (Serato
    stores "Users/me/Music/x.mp3", no leading slash or drive letter). Drop it
    in _Serato_/Subcrates on the drive that holds the audio. Cues are not
    part of the crate format; the rekordbox XML carries them.
    Returns {"data": bytes, "files": [...], "skipped": n}."""
    tracks, order, skipped = _collect(items, resolve)
    out = _serato_tag(b"vrsn", "1.0/Serato ScratchLive Crate".encode("utf-16-be"))
    by_tid = {t["tid"]: t for t in tracks}
    seen = set()
    for tid in order:
        if tid in seen:
            continue
        seen.add(tid)
        p = _local_path(by_tid[tid]["path"], base, root)
        p = re.sub(r"^[A-Za-z]:", "", p).lstrip("/")
        out += _serato_tag(b"otrk", _serato_tag(b"ptrk", p.encode("utf-16-be")))
    return {"data": out, "files": [t["path"] for t in tracks], "skipped": skipped}


# Traktor's MUSICAL_KEY: 0–11 the major keys from C, 12–23 the minor keys from Cm.
_PITCH = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
_PITCH.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})


def _traktor_location(path: str) -> Dict[str, str]:
    """VOLUME / DIR / FILE as Traktor writes them: DIR is every folder
    prefixed "/:" ("/:Users/:me/:Music/:"), VOLUME the drive ("C:") or empty
    on a Unix path, where Traktor resolves from the root."""
    p = path.replace("\\", "/")
    m = re.match(r"^([A-Za-z]:)(/.*)$", p)
    volume, rest = (m.group(1), m.group(2)) if m else ("", p)
    folders = [x for x in rest.split("/")[:-1] if x]
    return {"VOLUME": volume, "DIR": "".join(f"/:{f}" for f in folders) + "/:",
            "FILE": rest.split("/")[-1]}


def traktor_nml(items: Sequence[Dict], playlist: str,
                resolve: Callable[[int, str], Optional[Dict]],
                base: Optional[str] = None, root: Optional[str] = None) -> Dict:
    """A Traktor collection (.nml, version 19) holding each file with its
    tempo, key and the paired sections as hot cues, and one playlist in set
    order. Import it from Traktor's browser (Import Collection).
    Returns {"xml": str, "files": [...], "skipped": n}."""
    tracks, order, skipped = _collect(items, resolve)
    loc = {t["tid"]: _traktor_location(_local_path(t["path"], base, root)) for t in tracks}
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="no" ?>',
           '<NML VERSION="19"><HEAD COMPANY="www.native-instruments.com" PROGRAM="Traktor"></HEAD>',
           f'<COLLECTION ENTRIES="{len(tracks)}">']
    for t in tracks:
        name = t["title"] + (f" ({t['label']})" if t["label"] else "")
        L = loc[t["tid"]]
        out.append(f'<ENTRY TITLE={quoteattr(name)} ARTIST={quoteattr(t.get("artist") or "")}>')
        out.append(f'<LOCATION DIR={quoteattr(L["DIR"])} FILE={quoteattr(L["FILE"])} '
                   f'VOLUME={quoteattr(L["VOLUME"])} VOLUMEID={quoteattr(L["VOLUME"])}></LOCATION>')
        out.append(f'<INFO PLAYTIME="{int(round(t.get("duration") or 0))}" '
                   f'COMMENT="Mashup Engine export"></INFO>')
        if t.get("bpm"):
            out.append(f'<TEMPO BPM="{float(t["bpm"]):.6f}" BPM_QUALITY="100.000000"></TEMPO>')
        if t.get("key") in _PITCH:
            minor = (t.get("mode") or "").lower().startswith("min")
            out.append(f'<MUSICAL_KEY VALUE="{_PITCH[t["key"]] + (12 if minor else 0)}"></MUSICAL_KEY>')
        for num, (cname, at) in enumerate(t["cues"]):
            out.append(f'<CUE_V2 NAME={quoteattr(cname)} DISPL_ORDER="0" TYPE="0" '
                       f'START="{at * 1000.0:.6f}" LEN="0.000000" REPEATS="-1" HOTCUE="{num}"></CUE_V2>')
        out.append("</ENTRY>")
    out.append("</COLLECTION>")
    out.append('<PLAYLISTS><NODE TYPE="FOLDER" NAME="$ROOT"><SUBNODES COUNT="1">')
    out.append(f'<NODE TYPE="PLAYLIST" NAME={quoteattr(playlist)}>'
               f'<PLAYLIST ENTRIES="{len(order)}" TYPE="LIST" UUID="">')
    for tid in order:
        L = loc[tid]
        out.append(f'<ENTRY><PRIMARYKEY TYPE="TRACK" '
                   f'KEY={quoteattr(L["VOLUME"] + L["DIR"] + L["FILE"])}></PRIMARYKEY></ENTRY>')
    out.append("</PLAYLIST></NODE></SUBNODES></NODE></PLAYLISTS></NML>")
    return {"xml": "\n".join(out) + "\n", "files": [t["path"] for t in tracks],
            "skipped": skipped}
