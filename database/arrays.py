"""
database/arrays.py — the float arrays that have no business in a hot row.

``features`` was 5.63 MB over 612 rows, and 4.5 MB of it was two columns:
``beat_times_json`` (2.55 MB) and ``waveform_rms_json`` (1.95 MB). Neither is
read by the pair scorer, the library listing or the dedup pass, and all three
go through ``SELECT f.*`` — which is why asking for 204 rows of eight numbers
was a 52.8 ms full table scan.

They live here instead, as float32 BLOBs keyed by (song, stem, name): a third
of the bytes of the same values as JSON text, and out of the way of every query
that does not ask for them. A reader that wants one asks for it by name.

Absence means unmeasured. Writing ``None`` deletes the row rather than storing a
zero-length blob, so "no beat grid" and "a beat grid of nothing" cannot be
confused — the same rule the rest of the schema follows.

Every function that takes ``conn`` works on the caller's connection and never
commits: ``upsert_features`` calls ``put`` inside its own transaction, and a
second connection there is the deadlock readme §7 describes. Rows are read by
name (``row["data"]``), so the connection must have ``sqlite3.Row`` set, as
``get_conn`` does.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)

# The arrays moved out of `features`. A reader checks this tuple before
# reaching for a legacy JSON column.
NAMES: Tuple[str, ...] = ("beat_times", "waveform_rms")

# One statement per entry, never an executescript: a migration runs inside
# schema.apply's explicit BEGIN/COMMIT, and executescript COMMITs early, which
# would silently make the migration non-atomic (database/alignment.py's migrate
# carries the same note).
_DDL_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS feature_arrays (
        song_id    INTEGER NOT NULL,
        stem_type  TEXT NOT NULL,
        name       TEXT NOT NULL,      -- database.arrays.NAMES
        dtype      TEXT NOT NULL,      -- 'float32' | 'float16'
        dim        INTEGER NOT NULL,
        data       BLOB NOT NULL,
        PRIMARY KEY (song_id, stem_type, name)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_feature_arrays_song
        ON feature_arrays(song_id, stem_type)
    """,
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the table and its index on the caller's connection, statement by
    statement. Safe to re-run: a rollback (database/downgrade.py) replays every
    migration from 1."""
    for statement in _DDL_STATEMENTS:
        conn.execute(statement)


def pack(values, dtype: str = "float32") -> Optional[bytes]:
    """A sequence as a BLOB, or None for nothing to store."""
    if values is None:
        return None
    a = np.asarray(values, dtype=dtype).ravel()
    return a.tobytes() if a.size else None


def unpack(blob, dtype: str = "float32") -> List[float]:
    if not blob:
        return []
    return np.frombuffer(blob, dtype=dtype).astype(float).tolist()


def put(conn: sqlite3.Connection, song_id: int, stem_type: str, name: str,
        values, dtype: str = "float32") -> None:
    """Store one array on the caller's connection, inside the caller's
    transaction — this is called from `upsert_features`, which already holds
    one, and opening a second connection there is the deadlock readme §7
    describes. ``None`` (or an empty sequence) removes the row."""
    blob = pack(values, dtype)
    if blob is None:
        conn.execute("DELETE FROM feature_arrays WHERE song_id=? AND stem_type=? "
                     "AND name=?", (song_id, stem_type, name))
        return
    conn.execute(
        """INSERT INTO feature_arrays (song_id, stem_type, name, dtype, dim, data)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(song_id, stem_type, name) DO UPDATE SET
             dtype=excluded.dtype, dim=excluded.dim, data=excluded.data""",
        (song_id, stem_type, name, dtype, len(blob) // np.dtype(dtype).itemsize,
         blob))


def get(conn: sqlite3.Connection, song_id: int, stem_type: str,
        name: str) -> List[float]:
    row = conn.execute(
        "SELECT dtype, data FROM feature_arrays WHERE song_id=? AND stem_type=? "
        "AND name=?", (song_id, stem_type, name)).fetchone()
    return unpack(row["data"], row["dtype"]) if row else []


def get_many(conn: sqlite3.Connection, song_ids: Sequence[int], stem_type: str,
             names: Iterable[str]) -> Dict[Tuple[int, str], List[float]]:
    """{(song_id, name): values} for several songs in one query — what a route
    that draws many waveforms needs instead of one query a track."""
    ids = [int(i) for i in song_ids]
    names = list(names)
    if not ids or not names:
        return {}
    sql = (f"SELECT song_id, name, dtype, data FROM feature_arrays "
           f"WHERE stem_type=? AND song_id IN ({','.join('?' * len(ids))}) "
           f"AND name IN ({','.join('?' * len(names))})")
    out: Dict[Tuple[int, str], List[float]] = {}
    for r in conn.execute(sql, [stem_type, *ids, *names]):
        out[(int(r["song_id"]), r["name"])] = unpack(r["data"], r["dtype"])
    return out


def song_ids_with(conn: sqlite3.Connection, stem_type: str, name: str) -> Set[int]:
    """Which songs have this array measured for a stem, without reading a blob
    — the track list's "waveform" flag, for every row in one query."""
    return {int(r[0]) for r in conn.execute(
        "SELECT song_id FROM feature_arrays WHERE stem_type=? AND name=?",
        (stem_type, name))}


def delete_for(conn: sqlite3.Connection, song_id: int,
               stem_type: Optional[str] = None) -> int:
    if stem_type:
        cur = conn.execute("DELETE FROM feature_arrays WHERE song_id=? AND "
                           "stem_type=?", (song_id, stem_type))
    else:
        cur = conn.execute("DELETE FROM feature_arrays WHERE song_id=?", (song_id,))
    return cur.rowcount or 0
