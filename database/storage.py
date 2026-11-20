"""What the library keeps on disk, per kind and per track (readme §4, §6;
Phase 5): the downloaded mixes, the five stems, the per-frame `.npz`
sidecars, the hook clips and the database — and how much
`keep_analysis_stems` = off would free (the drums / bass / other of every
track allin1 has already refined: nothing reads them after that, §5.3).

Measured by stat-ing the files the database points at (a missing file is
counted as missing, never guessed), so it is exact for this library. The
per-audio-second rates it reports are what `projection` scales to another
library size (bench: the 10,000-track line in the readme).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

STEM_KINDS = ("vocals", "instrumental", "drums", "bass", "other")
ANALYSIS_ONLY = ("drums", "bass", "other")


def _size(path) -> Optional[int]:
    try:
        return os.stat(path).st_size if path else None
    except OSError:
        return None


def _dir_size(folder: Path) -> int:
    total = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def storage_report(db_path=None) -> Dict:
    """{"kinds": {kind: {bytes, files, missing, tracks, per_track,
    per_audio_sec}}, "tracks", "audio_secs", "total_bytes",
    "per_track_bytes", "prunable": {bytes, tracks}, "keep_analysis_stems"}.
    Never raises on a missing file."""
    from config import HOOKS_DIR, current_keep_analysis_stems
    from database.models import DB_PATH, get_conn

    db = db_path if db_path is not None else DB_PATH
    conn = get_conn(db)
    try:
        songs = {int(r[0]): (r[1], float(r[2] or 0.0), r[3]) for r in conn.execute(
            "SELECT id, raw_path, duration_secs, allin1_state FROM songs")}
        stems = conn.execute(
            "SELECT song_id, stem_type, file_path FROM stems").fetchall()
        frames = conn.execute(
            "SELECT song_id, stem_type, path FROM analysis_frames").fetchall()
    finally:
        conn.close()

    kinds: Dict[str, Dict] = {}

    def add(kind: str, song_id: int, path) -> Optional[int]:
        k = kinds.setdefault(kind, {"bytes": 0, "files": 0, "missing": 0,
                                    "_tracks": set(), "_secs": 0.0})
        n = _size(path)
        if n is None:
            k["missing"] += 1
            return None
        k["bytes"] += n
        k["files"] += 1
        if song_id not in k["_tracks"]:
            k["_tracks"].add(song_id)
            k["_secs"] += songs.get(song_id, (None, 0.0, None))[1]
        return n

    for sid, (raw, _secs, _state) in songs.items():
        if raw:
            add("mix", sid, raw)
    prunable, prunable_tracks = 0, set()
    for sid, kind, path in stems:
        n = add(kind if kind in STEM_KINDS else "stems_other", int(sid), path)
        if n and kind in ANALYSIS_ONLY and songs.get(int(sid), (0, 0, None))[2] == "done":
            prunable += n
            prunable_tracks.add(int(sid))
    for sid, _kind, path in frames:
        add("sidecars", int(sid), path)

    out_kinds = {}
    for kind, k in kinds.items():
        tracks = len(k.pop("_tracks"))
        secs = k.pop("_secs")
        out_kinds[kind] = {**k, "tracks": tracks,
                           "per_track": round(k["bytes"] / tracks) if tracks else None,
                           "per_audio_sec": round(k["bytes"] / secs, 1) if secs else None}
    hooks = _dir_size(Path(HOOKS_DIR))
    db_bytes = sum(_size(str(db) + s) or 0 for s in ("", "-wal"))
    out_kinds["hooks"] = {"bytes": hooks}
    out_kinds["database"] = {"bytes": db_bytes}
    total = sum(v.get("bytes", 0) for v in out_kinds.values())
    n_tracks = len(songs)
    return {"tracks": n_tracks,
            "audio_secs": round(sum(v[1] for v in songs.values()), 1),
            "kinds": out_kinds, "total_bytes": total,
            "per_track_bytes": round(total / n_tracks) if n_tracks else None,
            "prunable": {"bytes": prunable, "tracks": len(prunable_tracks)},
            "keep_analysis_stems": current_keep_analysis_stems()}


def projection(rates: Dict[str, float], tracks: int, mean_secs: float) -> Dict[str, int]:
    """Bytes per kind for ``tracks`` tracks of ``mean_secs`` audio, from
    per-audio-second ``rates`` ({kind: bytes per audio second})."""
    return {kind: int(round(r * mean_secs * tracks)) for kind, r in rates.items()}


def prune_analysis_stems(db_path=None, progress=None) -> Dict:
    """keep_analysis_stems = off, applied to the tracks allin1 already
    refined: their drums / bass / other are deleted exactly as the allin1
    stage would have (stages._drop_analysis_sources — under the stems folder
    only; the vocals and instrumental are never touched). A later allin1 run
    on such a track re-separates (§5.5). Returns {"tracks", "bytes"}."""
    from api.workers.stages import _drop_analysis_sources, _stem_paths
    from database.models import DB_PATH, get_conn

    db = db_path if db_path is not None else DB_PATH
    conn = get_conn(db)
    try:
        ids = [int(r[0]) for r in conn.execute(
            "SELECT DISTINCT s.id FROM songs s JOIN stems t ON t.song_id = s.id "
            "WHERE s.allin1_state='done' AND t.stem_type IN ('drums','bass','other')")]
    finally:
        conn.close()
    freed = 0
    for n, sid in enumerate(ids, 1):
        paths = _stem_paths(sid)
        before = {k: _size(paths.get(k)) or 0 for k in ANALYSIS_ONLY}
        _drop_analysis_sources(sid, paths)
        # Only what really went (a file outside the stems folder is kept).
        freed += sum(n for k, n in before.items() if _size(paths.get(k)) is None)
        if progress and n % 50 == 0:
            progress(int(100 * n / len(ids)), f"{n}/{len(ids)} tracks")
    return {"tracks": len(ids), "bytes": freed}
