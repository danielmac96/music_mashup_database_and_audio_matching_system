"""database/alignment.py — storage for the allin1 engine, vocal phrases and the
per-frame analysis sidecars (readme §5.5, §6).

What lives here
  allin1_analysis     one row per (song, pipeline): beats, downbeats, beat-in-bar,
                      bpm, labelled segments, activation summaries. Arrays are
                      float32 / float16 BLOBs — a few KB a track.
  allin1_embeddings   the per-bar, per-stem embedding (bars × 4 × 24, float16),
                      apart from the core row so no ordinary query loads it.
  sections_allin1     allin1's sections, measured by the same per-section code
                      as `sections` (analysis/structure.measure_sections), with
                      indexes from models.ALLIN1_SECTION_BASE up.
  vocal_phrases       sung phrase starts on the vocal stem, snapped to a
                      downbeat, with the signed pickup kept.
  analysis_frames     where each stem's per-frame .npz sidecar lives.

The tables themselves are in database/schema.sql. The ``pipeline`` key they all
carry is always ``PIPELINE`` (config.PIPELINE): one separation per track.

Every function opens its own connection through ``models.get_conn`` looked up
at call time — tests reload database.models, and a module-level binding would
write to whichever database was configured first.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def _models():
    from database import models
    return models


def _conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    m = _models()
    return m.get_conn(db_path if db_path is not None else m.DB_PATH)


PIPELINE = "unified_4stem"


# ── Which pipeline a song's rows belong to ────────────────────────────────────

def live_pipeline(song_id: int, db_path: Optional[Path] = None,
                  conn: Optional[sqlite3.Connection] = None) -> str:
    own = conn is None
    conn = conn or _conn(db_path)
    try:
        r = conn.execute("SELECT separation_pipeline FROM songs WHERE id=?",
                         (song_id,)).fetchone()
    finally:
        if own:
            conn.close()
    return (r and r[0]) or PIPELINE


# ── allin1 analysis ───────────────────────────────────────────────────────────

def _f32(a) -> Optional[bytes]:
    return None if a is None else np.asarray(a, dtype=np.float32).tobytes()


def _f16(a) -> Optional[bytes]:
    return None if a is None else np.asarray(a, dtype=np.float16).tobytes()


def _arr(b, dtype) -> list:
    return [] if not b else np.frombuffer(b, dtype=dtype).astype(float).tolist()


def store_allin1(song_id: int, pipeline: str, summary: Dict,
                 db_path: Optional[Path] = None) -> None:
    """Write one allin1 summary (analysis/allin1_runner.summarise) and, when it
    carries one, its pooled embedding."""
    conn = _conn(db_path)
    try:
        conn.execute(
            """INSERT INTO allin1_analysis
                   (song_id, pipeline, content_hash, model, version, pooling,
                    demixed_by, bpm, beats, downbeats, beat_positions,
                    segments_json, downbeat_conf, section_conf_json,
                    conf_mean, conf_min, ms, beat_conf, downbeat_conf_beats,
                    segment_label_probs, segment_labels_order)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(song_id, pipeline) DO UPDATE SET
                   content_hash=excluded.content_hash, model=excluded.model,
                   version=excluded.version, pooling=excluded.pooling,
                   demixed_by=excluded.demixed_by, bpm=excluded.bpm,
                   beats=excluded.beats, downbeats=excluded.downbeats,
                   beat_positions=excluded.beat_positions,
                   segments_json=excluded.segments_json,
                   downbeat_conf=excluded.downbeat_conf,
                   section_conf_json=excluded.section_conf_json,
                   conf_mean=excluded.conf_mean, conf_min=excluded.conf_min,
                   ms=excluded.ms, beat_conf=excluded.beat_conf,
                   downbeat_conf_beats=excluded.downbeat_conf_beats,
                   segment_label_probs=excluded.segment_label_probs,
                   segment_labels_order=excluded.segment_labels_order,
                   created_at=datetime('now')""",
            (song_id, pipeline, summary.get("content_hash"), summary.get("model"),
             summary.get("version"), summary.get("pooling"),
             summary.get("demixed_by"), summary.get("bpm"),
             _f32(summary.get("beats")), _f32(summary.get("downbeats")),
             None if summary.get("beat_positions") is None else
             np.asarray(summary["beat_positions"], dtype=np.int8).tobytes(),
             json.dumps(summary.get("segments") or []),
             _f16(summary.get("downbeat_conf")),
             json.dumps(summary.get("section_conf") or []),
             summary.get("conf_mean"), summary.get("conf_min"), summary.get("ms"),
             _f16(summary.get("beat_conf")),
             _f16(summary.get("downbeat_conf_beats")),
             None if summary.get("segment_label_probs") is None else
             json.dumps(summary["segment_label_probs"]),
             None if summary.get("segment_labels_order") is None else
             json.dumps(summary["segment_labels_order"])))
        emb = summary.get("embedding")
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float16)
            conn.execute(
                """INSERT INTO allin1_embeddings
                       (song_id, pipeline, content_hash, model, pooling, n_bars,
                        n_stems, dim, vector)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(song_id, pipeline) DO UPDATE SET
                       content_hash=excluded.content_hash, model=excluded.model,
                       pooling=excluded.pooling, n_bars=excluded.n_bars,
                       n_stems=excluded.n_stems, dim=excluded.dim,
                       vector=excluded.vector, created_at=datetime('now')""",
                (song_id, pipeline, summary.get("content_hash"), summary.get("model"),
                 summary.get("pooling"), int(emb.shape[0]), int(emb.shape[1]),
                 int(emb.shape[2]), emb.tobytes()))
        _models().mark_pairs_input(conn, song_id)
        conn.commit()
    finally:
        conn.close()


def get_allin1(song_id: int, pipeline: Optional[str] = None,
               db_path: Optional[Path] = None) -> Optional[Dict]:
    """The stored analysis for the song's live pipeline (or the one named),
    arrays decoded to lists. None when allin1 has not run for it."""
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        r = conn.execute("SELECT * FROM allin1_analysis WHERE song_id=? AND pipeline=?",
                         (song_id, pipeline)).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    d = dict(r)
    d["beats"] = _arr(d["beats"], np.float32)
    d["downbeats"] = _arr(d["downbeats"], np.float32)
    d["beat_positions"] = [int(x) for x in _arr(d["beat_positions"], np.int8)]
    d["downbeat_conf"] = _arr(d["downbeat_conf"], np.float16)
    d["segments"] = json.loads(d.pop("segments_json") or "[]")
    d["section_conf"] = json.loads(d.pop("section_conf_json") or "[]")
    # NULL reads back as [] (unmeasured, never zero).
    d["beat_conf"] = _arr(d.get("beat_conf"), np.float16)
    d["downbeat_conf_beats"] = _arr(d.get("downbeat_conf_beats"), np.float16)
    d["segment_label_probs"] = json.loads(d.get("segment_label_probs") or "[]")
    d["segment_labels_order"] = json.loads(d.get("segment_labels_order") or "[]")
    return d


def allin1_for_songs(song_ids: Iterable[int], db_path: Optional[Path] = None
                     ) -> Dict[int, Dict]:
    """get_allin1 for many songs at their live pipelines (no embeddings)."""
    out: Dict[int, Dict] = {}
    for sid in sorted({int(s) for s in song_ids}):
        a = get_allin1(sid, db_path=db_path)
        if a:
            out[sid] = a
    return out


def get_allin1_embedding(song_id: int, pipeline: Optional[str] = None,
                         db_path: Optional[Path] = None) -> Optional[np.ndarray]:
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        r = conn.execute("SELECT n_bars, n_stems, dim, vector FROM allin1_embeddings "
                         "WHERE song_id=? AND pipeline=?", (song_id, pipeline)).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    return np.frombuffer(r["vector"], dtype=np.float16).reshape(
        r["n_bars"], r["n_stems"], r["dim"])


def set_allin1_state(song_id: int, state: Optional[str], error: Optional[str] = None,
                     db_path: Optional[Path] = None) -> None:
    conn = _conn(db_path)
    try:
        conn.execute("UPDATE songs SET allin1_state=?, allin1_error=?, "
                     "allin1_at=datetime('now') WHERE id=?",
                     (state, (error or None) and str(error)[:500], song_id))
        conn.commit()
    finally:
        conn.close()


def storage_bytes(song_id: int, db_path: Optional[Path] = None) -> Dict[str, int]:
    """Bytes this feature stores for one song (core row / embedding / sections /
    phrases) — the brief's < 100 KB per track budget is checked against it."""
    conn = _conn(db_path)
    try:
        def one(sql):
            r = conn.execute(sql, (song_id,)).fetchone()
            return int(r[0] or 0)
        return {
            "analysis": one("""SELECT SUM(LENGTH(beats)+LENGTH(downbeats)
                                   +COALESCE(LENGTH(beat_positions),0)
                                   +LENGTH(segments_json)
                                   +COALESCE(LENGTH(downbeat_conf),0)
                                   +LENGTH(section_conf_json)
                                   +COALESCE(LENGTH(beat_conf),0)
                                   +COALESCE(LENGTH(downbeat_conf_beats),0)
                                   +COALESCE(LENGTH(segment_label_probs),0)
                                   +COALESCE(LENGTH(segment_labels_order),0))
                                FROM allin1_analysis WHERE song_id=?"""),
            "embedding": one("SELECT SUM(LENGTH(vector)) FROM allin1_embeddings "
                             "WHERE song_id=?"),
            "sections": one("""SELECT SUM(COALESCE(LENGTH(chroma_json),0)
                                   +COALESCE(LENGTH(chroma_vocal_json),0)
                                   +COALESCE(LENGTH(chroma_bed_json),0)
                                   +COALESCE(LENGTH(beat_times_json),0)
                                   +COALESCE(LENGTH(downbeats_json),0) + 200)
                                FROM sections_allin1 WHERE song_id=?"""),
            "phrases": one("SELECT COUNT(*) * 64 FROM vocal_phrases WHERE song_id=?"),
        }
    finally:
        conn.close()


# ── allin1 sections ───────────────────────────────────────────────────────────

def replace_allin1_sections(song_id: int, pipeline: str, sections: List[Dict],
                            db_path: Optional[Path] = None) -> None:
    """Replace one pipeline's allin1 sections, carrying judgements of the old
    ones onto the new (remap_feedback_sections, allin1 namespace) in the same
    transaction — the same contract as models.replace_sections."""
    m = _models()
    base = m.ALLIN1_SECTION_BASE
    conn = _conn(db_path)
    try:
        old = [dict(r) for r in conn.execute(
            "SELECT section_index, start_sec, end_sec FROM sections_allin1 "
            "WHERE song_id=? AND pipeline=?", (song_id, pipeline)).fetchall()]
        conn.execute("DELETE FROM sections_allin1 WHERE song_id=? AND pipeline=?",
                     (song_id, pipeline))
        cols = m.SECTION_INSERT_COLUMNS
        conn.executemany(
            f"INSERT INTO sections_allin1 (song_id, pipeline, section_index, "
            f"engine_label, downbeat_conf, {', '.join(cols)}) "
            f"VALUES (?,?,?,?,?,{','.join('?' * len(cols))})",
            [(song_id, pipeline, base + i, s.get("engine_label"),
              s.get("downbeat_conf"), *m.section_values(s))
             for i, s in enumerate(sections)])
        new = [{"section_index": base + i, "start_sec": float(s["start_sec"]),
                "end_sec": float(s["end_sec"])} for i, s in enumerate(sections)]
        if pipeline == live_pipeline(song_id, conn=conn):
            m.remap_feedback_sections(conn, song_id, old, new, namespace="allin1")
            m.mark_pairs_input(conn, song_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_allin1_sections(song_id: int, pipeline: Optional[str] = None,
                        db_path: Optional[Path] = None) -> List[Dict]:
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        rows = conn.execute(
            "SELECT * FROM sections_allin1 WHERE song_id=? AND pipeline=? "
            "ORDER BY section_index", (song_id, pipeline)).fetchall()
    finally:
        conn.close()
    return [_decode(r) for r in rows]


def _decode(row) -> Dict:
    """A sections_allin1 row decoded like a sections row, its id moved into
    the allin1 segment-id namespace (models.ALLIN1_SEGMENT_ID_BASE)."""
    m = _models()
    d = m._decode_section(row)
    d["row_id"] = d.get("id")
    d["id"] = m.ALLIN1_SEGMENT_ID_BASE + int(d["id"])
    return d


def allin1_sections_for_songs(song_ids, db_path: Optional[Path] = None
                              ) -> Dict[int, List[Dict]]:
    """{song_id: allin1 sections at the song's live pipeline}, one query."""
    ids = sorted({int(i) for i in song_ids if i is not None})
    out: Dict[int, List[Dict]] = {}
    if not ids:
        return out
    conn = _conn(db_path)
    try:
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            rows = conn.execute(
                f"""SELECT x.* FROM sections_allin1 x JOIN songs s ON s.id = x.song_id
                     WHERE x.song_id IN ({','.join('?' * len(chunk))})
                       AND x.pipeline = COALESCE(s.separation_pipeline, '{PIPELINE}')
                     ORDER BY x.song_id, x.section_index""", chunk).fetchall()
            for r in rows:
                out.setdefault(r["song_id"], []).append(_decode(r))
    finally:
        conn.close()
    return out


# ── Vocal phrases ─────────────────────────────────────────────────────────────

def replace_vocal_phrases(song_id: int, pipeline: str, grid_source: str,
                          phrases: List[Dict], content_hash: Optional[str] = None,
                          version: Optional[int] = None,
                          db_path: Optional[Path] = None) -> None:
    conn = _conn(db_path)
    try:
        conn.execute("DELETE FROM vocal_phrases WHERE song_id=? AND pipeline=? "
                     "AND grid_source=?", (song_id, pipeline, grid_source))
        conn.executemany(
            """INSERT INTO vocal_phrases
                   (song_id, pipeline, grid_source, phrase_index, onset_sec,
                    downbeat_sec, pickup_sec, end_sec, bars, confidence,
                    content_hash, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(song_id, pipeline, grid_source, i, float(p["onset_sec"]),
              float(p["downbeat_sec"]), float(p["pickup_sec"]), p.get("end_sec"),
              p.get("bars"), p.get("confidence"), content_hash, version)
             for i, p in enumerate(phrases)])
        _models().mark_pairs_input(conn, song_id)
        conn.commit()
    finally:
        conn.close()


def get_vocal_phrases(song_id: int, grid_source: Optional[str] = None,
                      pipeline: Optional[str] = None,
                      db_path: Optional[Path] = None) -> List[Dict]:
    """Phrases at the song's live pipeline. With no grid_source: the allin1
    grid's when there are any, else the Essentia grid's."""
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        rows = conn.execute(
            "SELECT * FROM vocal_phrases WHERE song_id=? AND pipeline=? "
            "ORDER BY grid_source, phrase_index", (song_id, pipeline)).fetchall()
    finally:
        conn.close()
    by_src: Dict[str, List[Dict]] = {}
    for r in rows:
        by_src.setdefault(r["grid_source"], []).append(dict(r))
    if grid_source:
        return by_src.get(grid_source, [])
    return by_src.get("allin1") or by_src.get("essentia") or []


def phrases_for_songs(song_ids, db_path: Optional[Path] = None) -> Dict[int, List[Dict]]:
    return {int(s): get_vocal_phrases(int(s), db_path=db_path)
            for s in {int(i) for i in song_ids if i is not None}}


# ── Which pipeline made a song's stems (songs.separation_pipeline) ─────────────

def set_live_pipeline(song_id: int, pipeline: str, db_path: Optional[Path] = None) -> None:
    conn = _conn(db_path)
    try:
        conn.execute("UPDATE songs SET separation_pipeline=? WHERE id=?",
                     (pipeline, song_id))
        conn.commit()
    finally:
        conn.close()


# ── Per-frame analysis matrices (analysis/timeseries.py) ──────────────────────

def put_frames(song_id: int, stem_type: str, pipeline: str, meta: Dict,
               db_path: Optional[Path] = None) -> None:
    """Register one stem's .npz sidecar. `meta` is what timeseries.write built:
    path, content_hash, version, step_secs, n_frames, matrices, bytes."""
    conn = _conn(db_path)
    try:
        conn.execute(
            """INSERT INTO analysis_frames
                 (song_id, stem_type, pipeline, path, content_hash, version,
                  step_secs, n_frames, matrices, bytes, written_at)
               VALUES (?,?,?,?,?,?,?,?,?,?, datetime('now'))
               ON CONFLICT(song_id, stem_type, pipeline) DO UPDATE SET
                 path=excluded.path, content_hash=excluded.content_hash,
                 version=excluded.version, step_secs=excluded.step_secs,
                 n_frames=excluded.n_frames, matrices=excluded.matrices,
                 bytes=excluded.bytes, written_at=excluded.written_at""",
            (song_id, stem_type, pipeline, str(meta["path"]),
             meta.get("content_hash"), int(meta["version"]),
             float(meta["step_secs"]), int(meta["n_frames"]),
             json.dumps(meta["matrices"]), meta.get("bytes")))
        conn.commit()
    finally:
        conn.close()


def get_frames(song_id: int, stem_type: Optional[str] = None,
               pipeline: Optional[str] = None,
               db_path: Optional[Path] = None) -> List[Dict]:
    """The registered sidecars for a song, `matrices` parsed."""
    conn = _conn(db_path)
    try:
        sql = "SELECT * FROM analysis_frames WHERE song_id=?"
        args: list = [song_id]
        if stem_type:
            sql += " AND stem_type=?"
            args.append(stem_type)
        if pipeline:
            sql += " AND pipeline=?"
            args.append(pipeline)
        rows = conn.execute(sql + " ORDER BY stem_type", args).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["matrices"] = json.loads(d["matrices"] or "{}")
        except (TypeError, ValueError):
            d["matrices"] = {}
        out.append(d)
    return out


def delete_frames(song_id: int, pipeline: Optional[str] = None,
                  db_path: Optional[Path] = None) -> int:
    """Remove a song's registry rows (one pipeline's, or all). The files are
    `analysis.timeseries.drop`'s business."""
    conn = _conn(db_path)
    try:
        if pipeline:
            cur = conn.execute(
                "DELETE FROM analysis_frames WHERE song_id=? AND pipeline=?",
                (song_id, pipeline))
        else:
            cur = conn.execute("DELETE FROM analysis_frames WHERE song_id=?",
                               (song_id,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()
