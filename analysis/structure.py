"""
analysis/structure.py — Detect song structure (intro/verse/chorus/drop/…)
with start/end timestamps for each section.

Method (librosa + scipy only, no extra deps):
  1. Beat-track the full mix, beat-synchronise chroma + MFCC features.
  2. Build a self-similarity matrix and a checkerboard-kernel novelty curve.
  3. Pick novelty peaks as section boundaries (snapped to beats).
  4. Score each section: relative energy (full mix RMS) and vocal presence
     (RMS of the Demucs vocal stem inside the section).
  5. Count repetitions (sections whose mean chroma is near-identical) —
     the repeated, high-energy, vocal-heavy cluster is the chorus.
  6. Label sections with explainable heuristics (label_segments below).

The output feeds the `sections` table and powers section-level mashup
suggestions ("lay vocal chorus of A over the drop of B").
"""
from typing import Callable, List, Optional
import logging

import numpy as np
from pathlib import Path

ProgressCb = Optional[Callable[[Optional[int], str], None]]

log = logging.getLogger(__name__)

try:
    from config import (
        SAMPLE_RATE, HOP_LENGTH,
        SECTION_MIN_LEN_SECS, SECTION_MAX_COUNT, SECTION_SIM_THRESHOLD,
    )
except ImportError:
    SAMPLE_RATE, HOP_LENGTH = 22050, 512
    SECTION_MIN_LEN_SECS, SECTION_MAX_COUNT, SECTION_SIM_THRESHOLD = 12.0, 14, 0.92


# ── Labelling heuristics (pure python — unit-testable without librosa) ────────

def label_segments(segs: List[dict], has_vocals: bool) -> List[dict]:
    """Assign a structural label to each segment dict in place and return them.

    Expects per-segment keys: energy (0-1), vocal_presence (0-1 or None),
    repetition (int). Adds: label, confidence.
    """
    n = len(segs)
    for idx, s in enumerate(segs):
        e = s.get("energy") or 0.0
        v = s.get("vocal_presence")
        v = v if v is not None else 0.0
        rep = s.get("repetition") or 1

        label, conf = "verse", 0.4

        if has_vocals:
            if v >= 0.5 and e >= 0.6 and rep >= 2:
                label, conf = "chorus", min(1.0, 0.6 + 0.4 * min(v, e))
            elif e >= 0.7 and v < 0.35:
                label, conf = "drop", min(1.0, 0.5 + 0.5 * e)
            elif e < 0.45 and v < 0.4:
                label, conf = "breakdown", 0.5
            elif v >= 0.4:
                # vocal section that doesn't repeat and isn't peak energy
                label = "verse" if rep >= 2 or e < 0.6 else "bridge"
                conf = 0.5
        else:
            # No vocal stem: separate by energy + repetition only.
            if e >= 0.7:
                label, conf = ("chorus", 0.5) if rep >= 2 else ("drop", 0.5)
            elif e < 0.45:
                label, conf = "breakdown", 0.5

        # Positional overrides — quiet first/last segments are intro/outro.
        if idx == 0 and (e < 0.55 or v < 0.3):
            label, conf = "intro", 0.7
        elif idx == n - 1 and e < 0.55:
            label, conf = "outro", 0.7

        s["label"] = label
        s["confidence"] = round(conf, 3)

    # Guarantee at least one chorus when vocals exist: promote the repeated
    # segment with the best energy+vocal blend if nothing qualified above.
    if has_vocals and not any(s["label"] == "chorus" for s in segs):
        candidates = [s for s in segs if (s.get("repetition") or 1) >= 2
                      and s["label"] not in ("intro", "outro")]
        pool = (candidates
                or [s for s in segs if s["label"] not in ("intro", "outro")]
                or segs)
        if pool:
            best = max(pool, key=lambda s: (s.get("energy") or 0)
                       + (s.get("vocal_presence") or 0))
            best["label"] = "chorus"
            best["confidence"] = 0.4
    return segs


# ── Segmentation ──────────────────────────────────────────────────────────────

def _novelty_boundaries(X: np.ndarray, min_beats: int, max_sections: int) -> List[int]:
    """Boundary beat indices from a checkerboard-kernel novelty curve over the
    self-similarity matrix of beat-synchronous features X (features x beats)."""
    from scipy.signal import find_peaks

    n = X.shape[1]
    if n < 2 * min_beats:
        return []

    # Cosine self-similarity
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = Xn.T @ Xn

    # Gaussian checkerboard kernel
    L = min(max(min_beats, 8), n // 2)
    t = np.arange(-L, L)
    g = np.exp(-(t ** 2) / (2 * (L / 2.0) ** 2))
    kernel = np.outer(g, g) * np.outer(np.sign(t + 0.5), np.sign(t + 0.5))

    novelty = np.zeros(n)
    Spad = np.pad(S, L, mode="edge")
    for i in range(n):
        patch = Spad[i:i + 2 * L, i:i + 2 * L]
        novelty[i] = float((patch * kernel).sum())
    novelty -= novelty.min()
    if novelty.max() > 0:
        novelty /= novelty.max()

    peaks, props = find_peaks(novelty, distance=min_beats, prominence=0.05)
    if len(peaks) == 0:
        return []
    # Keep the most prominent peaks if we found too many sections.
    if len(peaks) > max_sections - 1:
        order = np.argsort(props["prominences"])[::-1][: max_sections - 1]
        peaks = np.sort(peaks[order])
    return [int(p) for p in peaks]


def _frame_rms(y: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]


def detect_sections(full_path: Path, vocals_path: Optional[Path] = None,
                    on_progress: ProgressCb = None) -> List[dict]:
    """Analyse the full mix (and the vocal stem when available) and return an
    ordered list of section dicts: start_sec, end_sec, label, energy,
    vocal_presence, repetition, confidence. Returns [] on failure."""
    def _tick(msg: str) -> None:
        if on_progress:
            on_progress(None, msg)

    try:
        import librosa
    except ImportError:
        log.error("librosa not installed. Run: pip install librosa")
        return []

    log.info(f"Detecting structure: {full_path.name}")
    _tick("Loading full mix…")
    y, sr = librosa.load(str(full_path), sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    if duration < 4 * SECTION_MIN_LEN_SECS:
        log.info("  Track too short for structural segmentation, single section.")
        return [{
            "start_sec": 0.0, "end_sec": round(duration, 2), "label": "verse",
            "energy": 1.0, "vocal_presence": None, "repetition": 1,
            "confidence": 0.3,
        }]

    _tick("Beat tracking…")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    if len(beats) < 16:
        log.warning("  Too few beats detected — skipping structure analysis.")
        return []
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)

    _tick("Computing beat-synchronous features…")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    rms = _frame_rms(y)

    chroma_b = librosa.util.sync(chroma, beats, aggregate=np.median)
    mfcc_b = librosa.util.sync(mfcc, beats, aggregate=np.mean)
    rms_b = librosa.util.sync(rms[np.newaxis, :], beats, aggregate=np.mean)[0]

    # Standardise MFCC rows so chroma and timbre carry comparable weight.
    mfcc_z = (mfcc_b - mfcc_b.mean(axis=1, keepdims=True)) / \
             (mfcc_b.std(axis=1, keepdims=True) + 1e-9)
    X = np.vstack([chroma_b, mfcc_z])

    beat_dur = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    min_beats = max(8, int(round(SECTION_MIN_LEN_SECS / max(beat_dur, 1e-3))))

    _tick("Finding section boundaries…")
    bounds = _novelty_boundaries(X, min_beats=min_beats,
                                 max_sections=SECTION_MAX_COUNT)

    # Beat-index boundaries → [start, end) segments in seconds.
    edges = [0] + bounds + [len(beat_times) - 1]
    seg_ranges = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg_ranges.append((a, b))
    if not seg_ranges:
        return []

    # Vocal stem RMS on the same clock (frame level, mapped by time).
    vocal_rms = None
    if vocals_path and Path(vocals_path).exists():
        _tick("Measuring vocal activity…")
        try:
            yv, _ = librosa.load(str(vocals_path), sr=SAMPLE_RATE, mono=True)
            vocal_rms = _frame_rms(yv)
        except Exception:
            log.warning("  Could not load vocal stem for vocal-presence scoring.")
            vocal_rms = None

    frames_per_sec = sr / HOP_LENGTH
    rms_max = float(np.percentile(rms_b, 95)) or 1.0
    v_scale = float(np.percentile(vocal_rms, 95)) if vocal_rms is not None else 1.0

    _tick("Scoring + labelling sections…")
    segs = []
    chroma_means = []
    for a, b in seg_ranges:
        start_t = float(beat_times[a])
        end_t = float(beat_times[b]) if b < len(beat_times) else duration
        energy = float(np.clip(rms_b[a:b].mean() / (rms_max + 1e-9), 0, 1))

        vp = None
        if vocal_rms is not None:
            f0 = int(start_t * frames_per_sec)
            f1 = max(f0 + 1, int(end_t * frames_per_sec))
            seg_v = vocal_rms[f0:min(f1, len(vocal_rms))]
            if len(seg_v):
                vp = float(np.clip(seg_v.mean() / (v_scale + 1e-9), 0, 1))

        chroma_means.append(chroma_b[:, a:b].mean(axis=1))
        segs.append({
            "start_sec": round(start_t, 2),
            "end_sec": round(end_t, 2),
            "energy": round(energy, 4),
            "vocal_presence": round(vp, 4) if vp is not None else None,
        })
    # Extend the last section to the true end of the audio.
    segs[-1]["end_sec"] = round(duration, 2)

    # Repetition: count near-identical chroma profiles (chorus repeats).
    C = np.array(chroma_means)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sim = Cn @ Cn.T
    for i, s in enumerate(segs):
        s["repetition"] = int((sim[i] >= SECTION_SIM_THRESHOLD).sum())

    label_segments(segs, has_vocals=vocal_rms is not None)

    log.info("  → " + ", ".join(
        f"{s['label']} {s['start_sec']:.0f}-{s['end_sec']:.0f}s" for s in segs))
    return segs
