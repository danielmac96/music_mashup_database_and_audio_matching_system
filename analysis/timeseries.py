"""
analysis/timeseries.py — the per-frame matrices every stage shares.

Essentia already computes, for every stem, exactly what the structure pass used
to recompute with librosa:

  * ``TonalExtractor`` produces a per-frame HPCP (4096/2048 at 44.1 kHz), and
    ``group_tonal`` averaged it to twelve numbers and dropped the frames.
  * ``group_spectral``'s loop produces a per-frame magnitude spectrum, a
    per-frame MFCC and per-frame band power (2048/512 at 22.05 kHz), and kept
    only the track means.

So ``detect_sections`` decoded the mix and every stem a second time and asked
librosa for a CQT chroma (~19 s a stem), an MFCC and a second 2048/512 STFT —
88.6 s a track on the 206-track library (57.1 s of it the stems' CQT chroma),
for numbers that already existed.

This module keeps them. One grid serves everything, because the two Essentia
hops are the same interval: 2048/44100 == 2*512/22050 == 46.44 ms. Nothing is
resampled anywhere downstream.

Storage is a compressed ``.npz`` beside the audio, not a column: a four-minute
stem is ~5,200 frames, and chroma + MFCC + bands + curves is ~400 KB a stem —
2 % of the stem's own WAV, and far too much for SQLite. ``analysis_frames``
(database/alignment.py, created by schema migration 4) records where each
sidecar is, the content hash of the stem it describes and its shape, so a
re-separated stem invalidates it exactly as the feature cache invalidates a
group.

``chroma`` and ``mfcc`` are stored float16 (relative error ~1e-3; both are
normalised or z-scored before use), ``bands`` float16 after scaling by its own
maximum — the scale is kept as a float64, so the ratios an occupancy reads
hold to float16 precision (~5e-4 relative; a band below ~6e-8 of the loudest
flushes to zero), and raw band power overflows float16. ``curves`` and ``f0`` stay
float32: those values are absolute Hz and nothing normalises them.

A matrix that was not measured is an absent key in the file and ``None`` on
``StemFrames``, never a matrix of zeros (readme §5: unknown is not bad), and a
zero-length matrix is never written.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

VERSION = 1

# The one frame grid. Pinned against both Essentia hops in the tests.
STEP_SECS = 2048 / 44100.0

# Rows of the `curves` matrix, in order.
CURVE_ROWS = ("centroid", "rolloff", "flux", "zcr", "rms")

# How many rows each matrix must have. `f0` is the vocal stem's sung pitch in Hz
# (filled by a later task, vocal stem only) — one row, zero where a frame is
# unvoiced, and absent entirely where the melody was never measured.
ROWS: Dict[str, int] = {"chroma": 12, "mfcc": 13, "bands": 8,
                        "curves": len(CURVE_ROWS), "f0": 1}
_F16 = ("chroma", "mfcc")          # float16 as they are
_SCALED = ("bands",)               # float16 after /max, with the scale kept
_F32 = ("curves", "f0")            # float32

# Set by tests to redirect the sidecars; None means config.AUDIO_DIR/features,
# resolved lazily so a settings change after import still lands.
FRAMES_DIR: Optional[Path] = None


def _root() -> Path:
    if FRAMES_DIR is not None:
        return Path(FRAMES_DIR)
    import config
    return Path(config.AUDIO_DIR) / "features"


@dataclass
class StemFrames:
    """One stem's matrices on the shared grid. Any of them may be None — a
    missing matrix is unmeasured, never zeros."""
    step: float
    chroma: Optional[np.ndarray] = None      # (12, n) float32
    mfcc: Optional[np.ndarray] = None        # (13, n) float32
    bands: Optional[np.ndarray] = None       # (8, n) float32, absolute power
    curves: Optional[np.ndarray] = None      # (5, n) float32, CURVE_ROWS order
    f0: Optional[np.ndarray] = None          # (1, n) float32 Hz, vocal stem only

    def n_frames(self) -> int:
        for name in ROWS:
            m = getattr(self, name)
            if m is not None:
                return int(np.asarray(m).shape[1])
        return 0

    def matrices(self) -> Dict[str, int]:
        return {name: ROWS[name] for name in ROWS
                if getattr(self, name) is not None}


@dataclass
class TrackFrames:
    """Every stem's frames for one song and pipeline."""
    stems: Dict[str, StemFrames]

    def get(self, stem_type: str) -> Optional[StemFrames]:
        return self.stems.get(stem_type)


def path_for(song_id: int, stem_type: str, pipeline: str) -> Path:
    return _root() / str(int(song_id)) / f"{pipeline}.{stem_type}.npz"


def _payload(frames: StemFrames) -> Dict[str, np.ndarray]:
    """What goes into the .npz, validated. Raises before anything is written."""
    payload: Dict[str, np.ndarray] = {
        "step": np.asarray([frames.step], dtype=np.float64)}
    stored = 0
    lengths: Dict[str, int] = {}
    for name in ROWS:
        m = getattr(frames, name)
        if m is None:
            continue
        a = np.ascontiguousarray(m, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] != ROWS[name]:
            raise ValueError(f"{name} must be ({ROWS[name]}, n), got {a.shape}")
        if a.shape[1] == 0:
            raise ValueError(f"{name} has no frames; leave it None instead")
        # NaN/inf would be "unmeasured" stored as a value (readme §5: unknown is
        # absent, never a number), and one inf makes the bands' scale inf so the
        # whole matrix reads back NaN or zero.
        if not np.isfinite(a).all():
            raise ValueError(f"{name} contains NaN or inf; leave it None "
                             "(unmeasured) instead of storing a stand-in")
        lengths[name] = int(a.shape[1])
        if name in _SCALED:
            scale = float(np.max(np.abs(a))) or 1.0
            payload[name] = (a / scale).astype(np.float16)
            payload[f"{name}_scale"] = np.asarray([scale], dtype=np.float64)
        elif name in _F16:
            payload[name] = a.astype(np.float16)
        else:
            payload[name] = a
        stored += 1
    if not stored:
        raise ValueError("no matrices to store")
    # One grid is the whole design: nothing downstream resamples. A chroma of 500
    # frames beside an MFCC of 498 would otherwise be stored, and n_frames()
    # would report whichever came first; the misalignment would surface much
    # later as a broadcast error or, worse, a silently shifted section.
    if len(set(lengths.values())) > 1:
        raise ValueError("matrices must share one frame count, got "
                         + ", ".join(f"{k}={v}" for k, v in lengths.items()))
    return payload


def write(song_id: int, stem_type: str, pipeline: str, frames: StemFrames,
          content_hash: Optional[str], db_path=None) -> Dict:
    """Write one stem's sidecar (atomically) and register it. Returns the meta
    dict that database.alignment.put_frames stored."""
    from database import alignment

    payload = _payload(frames)
    out = path_for(song_id, stem_type, pipeline)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A unique temp name: a fixed one is shared by two writers of the same
    # (song, stem, pipeline), and one could `replace` the other's partial bytes
    # into place. It is unlinked on any failure, since an orphan `.part` would
    # also keep `drop` from removing the song's folder.
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".",
                                    suffix=".part")
    tmp = Path(tmp_name)
    try:
        # A file object, not the path: np.savez_compressed appends ".npz" to a
        # path that does not end in it, which would leave the rename below
        # nothing to move.
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **payload)
        tmp.replace(out)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

    meta = {"path": out, "content_hash": content_hash, "version": VERSION,
            "step_secs": float(frames.step), "n_frames": frames.n_frames(),
            "matrices": frames.matrices(), "bytes": out.stat().st_size}
    alignment.put_frames(song_id, stem_type, pipeline, meta, db_path=db_path)
    return meta


def read(song_id: int, stem_type: str, pipeline: str,
         db_path=None) -> Optional[StemFrames]:
    """The stem's frames, or None when there is no readable sidecar — a missing
    or corrupt one is "not measured here", never a failure. (`db_path` is
    accepted for symmetry with `write`; the file alone is the source of truth
    and ``have`` is the check that it matches the audio.)"""
    path = path_for(song_id, stem_type, pipeline)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            step = float(z["step"][0]) if "step" in z else STEP_SECS
            kw = {}
            for name in ROWS:
                if name not in z:
                    continue
                a = np.asarray(z[name], dtype=np.float32)
                scale_key = f"{name}_scale"
                if scale_key in z:
                    a = a * np.float32(z[scale_key][0])
                kw[name] = a
        return StemFrames(step=step, **kw)
    except Exception:  # noqa: BLE001
        log.warning("could not read frames sidecar %s", path, exc_info=True)
        return None


def load_track(song_id: int, pipeline: str,
               stem_types: Iterable[str] = ("full", "vocals", "instrumental", "bass"),
               db_path=None) -> TrackFrames:
    stems = {}
    for stem_type in stem_types:
        got = read(song_id, stem_type, pipeline, db_path=db_path)
        if got is not None:
            stems[stem_type] = got
    return TrackFrames(stems=stems)


def have(song_id: int, stem_type: str, pipeline: str,
         content_hash: Optional[str], db_path=None) -> bool:
    """True when a sidecar for exactly this audio and format version is both on
    disk AND registered. Both halves matter: a row without a file is what a
    partial restore of ./data leaves behind, a file without a row is what an
    interrupted write leaves behind."""
    from database import alignment

    if not content_hash:
        return False
    rows = alignment.get_frames(song_id, stem_type, pipeline, db_path=db_path)
    if not rows:
        return False
    row = rows[0]
    if row.get("content_hash") != content_hash:
        return False
    if int(row.get("version") or 0) != VERSION:
        return False
    return path_for(song_id, stem_type, pipeline).exists()


def drop(song_id: int, pipeline: Optional[str] = None, db_path=None) -> int:
    """Delete a song's sidecars (one pipeline's, or all) and their rows.
    Returns how many files were removed."""
    from database import alignment

    removed = 0
    for row in alignment.get_frames(song_id, pipeline=pipeline, db_path=db_path):
        # Resolved against the current configuration, not the stored path: the
        # app runs both in Docker (/data/...) and on the host against one data
        # folder, and a path registered by the other side would not be found,
        # leaving the file orphaned while its row is deleted.
        p = path_for(song_id, row["stem_type"], row["pipeline"])
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except OSError:
            log.warning("could not delete %s", p, exc_info=True)
    alignment.delete_frames(song_id, pipeline, db_path=db_path)
    folder = _root() / str(int(song_id))
    try:
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass
    return removed


def sync_by_time(matrix: np.ndarray, step: float, beat_times,
                 aggregate: Callable = np.median) -> np.ndarray:
    """Beat-synchronous aggregation of a frame matrix — the same numbers
    ``librosa.util.sync(matrix, frame_indices, aggregate=aggregate)`` returns
    for the frame indices of ``beat_times`` at this step.

    The boundary list is librosa's own ``fix_frames(pad=True)``: the beat
    frames inside the matrix, sorted and de-duplicated, with 0 and the frame
    count added at the ends. De-duplicating matters — a beat at t=0, or two
    beats landing on one frame, would otherwise leave an empty interval and a
    column of zeros that librosa never produces.

    Pinned against librosa itself in tests/test_timeseries.py: a section's
    per-stem means must not change because the matrix now comes from disk.
    """
    M = np.asarray(matrix)
    if M.ndim == 1:
        M = M[np.newaxis, :]
    n = M.shape[1]
    idx = np.asarray(np.round(np.asarray(beat_times, dtype=float) / float(step)),
                     dtype=int)
    idx = idx[(idx >= 0) & (idx < n)]
    bounds = np.unique(np.concatenate([[0], idx, [n]])).tolist()
    cols = [np.asarray(aggregate(M[:, lo:hi], axis=1), dtype=np.float64)
            for lo, hi in zip(bounds[:-1], bounds[1:])]
    return np.stack(cols, axis=1) if cols else np.zeros((M.shape[0], 0))
