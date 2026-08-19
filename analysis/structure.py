"""
analysis/structure.py — Detect song structure (intro/verse/chorus/drop/…)
with start/end timestamps for each section.

Method (librosa + scipy only, no extra deps):
  1. Beat-track the full mix, beat-synchronise chroma + MFCC features.
  2. Build a self-similarity matrix and a checkerboard-kernel novelty curve.
  3. Pick novelty peaks as section boundaries, then snap them to the 8-bar
     phrase grid measured from the detected downbeat (snap_boundaries_to_phrases).
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


# ── Phrase snapping (pure python — unit-testable without librosa) ────────────
#
# Pop and EDM are written in 8- and 16-bar phrases: the drop lands on a phrase
# boundary, never three beats into one. The novelty curve, though, peaks
# wherever the timbre changes fastest, which is typically a beat or two after
# the real edit. Left alone that error compounds — a hook cut from the section
# starts late, a section-level match compares two windows that are misaligned by
# a beat, and every downstream length in bars is fractional.

BEATS_PER_BAR = 4
PHRASE_BARS = 8
PHRASE_BEATS = PHRASE_BARS * BEATS_PER_BAR

# How far a detected boundary may be pulled. Beyond 2 bars the novelty peak is
# more likely to be a real event the phrase grid does not describe (a track in
# 3/4, a half-bar edit, a tempo the beat tracker read at double time) than a
# late detection, so the detection is kept and the section is marked less
# trustworthy instead of being moved somewhere it was never heard.
SNAP_TOLERANCE_BARS = 2
SNAP_TOLERANCE_BEATS = SNAP_TOLERANCE_BARS * BEATS_PER_BAR

# Applied to the confidence of a section that starts off the phrase grid.
PHRASE_CONFIDENCE_FACTOR = 0.75


def snap_boundaries_to_phrases(bounds: List[int], phase: int, n_beats: int,
                               min_beats: int,
                               phrase_beats: int = PHRASE_BEATS,
                               tolerance_beats: int = SNAP_TOLERANCE_BEATS):
    """Pull boundary beat indices onto the 8-bar grid. Returns (bounds, snapped).

    `phase` is the beat index the bar grid starts on (T1.4's beat_phase), so the
    phrase grid is every `phrase_beats`th beat counted from there — snapping to
    beat 32 when the bar actually starts on beat 2 would land mid-bar.

    A boundary is only moved when the nearest phrase line is within
    `tolerance_beats` AND the move leaves every section at least `min_beats`
    long; otherwise the detected boundary is kept and flagged unsnapped. A
    boundary that cannot satisfy the minimum either way is dropped — merging it
    into its neighbour is better than emitting a two-bar "section".

    The walk looks one boundary ahead before committing. Peak picking already
    spaces detections at least `min_beats` apart, so pulling one forward onto a
    phrase line can leave the NEXT one under the floor and get it merged away —
    losing a whole section to buy alignment on the one before it. Measured on a
    real library that cascade accounted for a third of all dropped boundaries.
    When it would happen, this keeps the earlier boundary where it was detected:
    an unsnapped boundary is a confidence discount, a dropped one is a section
    the user no longer has.
    """
    out: List[int] = []
    snapped: List[bool] = []
    prev = 0
    last = max(n_beats - 1, 0)
    for i, b in enumerate(bounds):
        target = phase + int(round((b - phase) / phrase_beats)) * phrase_beats
        nxt = bounds[i + 1] if i + 1 < len(bounds) else None

        def _fits(idx: int) -> bool:
            return prev + min_beats <= idx <= last - min_beats

        # Only a concern when the next boundary is viable where it was found:
        # if it is already too close to the end to keep, or too close to this
        # one, then yielding gives up an alignment and saves nothing.
        crowds_next = (nxt is not None
                       and nxt <= last - min_beats
                       and nxt - b >= min_beats
                       and nxt - target < min_beats)

        if abs(target - b) <= tolerance_beats and _fits(target) and not crowds_next:
            out.append(target)
            snapped.append(True)
            prev = target
        elif _fits(b):
            out.append(b)
            snapped.append(False)
            prev = b
    return out, snapped


def apply_phrase_alignment(segs: List[dict],
                           factor: float = PHRASE_CONFIDENCE_FACTOR) -> List[dict]:
    """Discount the confidence of sections that start off the phrase grid.

    Consumers already rank by confidence — the hook picker
    (analysis/hooks.py _best_section) and the plan builder both do — so this is
    how a boundary the grid could not explain gets quietly deprioritised
    without discarding it. Consumes the private phrase_aligned marker."""
    for s in segs:
        if s.pop("phrase_aligned", True) is False:
            s["confidence"] = round((s.get("confidence") or 0.0) * factor, 3)
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


# Bass region for the second chroma. Below ~40 Hz is mostly rumble; above
# ~250 Hz the bassline stops being separable from the rest of the arrangement.
BASS_LOW_HZ = 40.0
BASS_HIGH_HZ = 250.0


def _bandpass(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Fourth-order Butterworth band-pass. scipy only, so this stays testable
    without touching the audio stack twice."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    sos = butter(4, [max(low / nyq, 1e-4), min(high / nyq, 0.99)],
                 btype="band", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _norm_chroma(vec: np.ndarray) -> list:
    """L2-normalised 12-bin chroma, rounded for compact JSON. Normalising here
    means every consumer compares shape rather than loudness."""
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return [0.0] * 12
    return [round(float(x), 6) for x in (v / n)]


# A section's own tempo is only worth believing when its grid is steady enough
# to be worth believing. Below this we keep the track's BPM and say so, rather
# than letting a four-beat breakdown claim 74 BPM and drag a match with it.
SECTION_BPM_MIN_CONFIDENCE = 0.25
SECTION_BPM_MIN_BEATS = 8
# How far a section's own estimate may sit from the track's before we treat it
# as a half/double-time reading of the same grid rather than a real change.
SECTION_BPM_FOLD_TOLERANCE = 0.08

# vocal_presence thresholds for the categorical class. Between them is "mixed":
# a section with singing over a full arrangement, which is usable on either side
# but ideal on neither.
SECTION_VOCAL_MIN = 0.35
SECTION_INSTRUMENTAL_MAX = 0.12

# Energy slope below this (per second, on the 0-1 normalised curve) reads as
# flat. A build is not subtle, and calling every gentle wobble a "build" would
# make the label useless for picking one.
ENERGY_TREND_EPSILON = 0.004


def _section_bpm(section_beats: np.ndarray, track_bpm: Optional[float],
                 confidence: float) -> tuple:
    """(bpm, source) for one section.

    Falls back to the track's tempo when the section's own grid is too short or
    too unsteady to trust — and records WHICH, because a caller weighing a match
    should know whether it is looking at a measurement or an inheritance.

    A section estimate that lands within tolerance of half or double the track
    tempo is a fold of the same grid, not a tempo change, so it is snapped back:
    an 8-bar half-time bridge is still the same record.
    """
    if track_bpm and (len(section_beats) < SECTION_BPM_MIN_BEATS
                      or confidence < SECTION_BPM_MIN_CONFIDENCE):
        return float(track_bpm), "track_fallback"

    intervals = np.diff(section_beats)
    intervals = intervals[intervals > 1e-6]
    if len(intervals) < 2:
        return (float(track_bpm), "track_fallback") if track_bpm else (None, None)

    bpm = 60.0 / float(np.median(intervals))
    if not np.isfinite(bpm) or bpm <= 0:
        return (float(track_bpm), "track_fallback") if track_bpm else (None, None)

    if track_bpm:
        for factor in (0.5, 2.0):
            if abs(bpm - track_bpm * factor) / max(track_bpm * factor, 1e-6) \
                    < SECTION_BPM_FOLD_TOLERANCE:
                return float(track_bpm), "section_estimate"
    return round(float(bpm), 2), "section_estimate"


def _energy_shape(curve: np.ndarray, duration: float) -> tuple:
    """(slope per second, trend label) over a section's energy curve.

    A least-squares fit rather than end-minus-start: a drop's last beat can be a
    tail and its first a pickup, and either would flip the sign of a difference
    while saying nothing about the shape between them.
    """
    if curve is None or len(curve) < 3 or duration <= 0:
        return None, "stable"
    x = np.linspace(0.0, float(duration), len(curve))
    try:
        slope = float(np.polyfit(x, curve.astype(float), 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return None, "stable"
    if not np.isfinite(slope):
        return None, "stable"
    if slope > ENERGY_TREND_EPSILON:
        trend = "increasing"
    elif slope < -ENERGY_TREND_EPSILON:
        trend = "decreasing"
    else:
        trend = "stable"
    return round(slope, 6), trend


def _section_class(vocal_presence: Optional[float]) -> str:
    """vocal | instrumental | mixed | unknown.

    unknown means the vocal stem was missing, NOT that the section is quiet —
    the spec's §5 says not to match unknown sections unless explicitly enabled,
    and conflating "no stem" with "no vocal" would silently drop half a library
    that has not been separated yet.
    """
    if vocal_presence is None:
        return "unknown"
    if vocal_presence >= SECTION_VOCAL_MIN:
        return "vocal"
    if vocal_presence <= SECTION_INSTRUMENTAL_MAX:
        return "instrumental"
    return "mixed"


def _phrase_length(bar_count: float) -> Optional[float]:
    """The power-of-two phrase this section's length is nearest, in bars.

    Sections are snapped to an 8-bar grid upstream, so this is usually exact;
    it exists so a 15.8-bar section reports the 16 a producer would call it.
    """
    if not bar_count or bar_count <= 0:
        return None
    candidates = (1, 2, 4, 8, 16, 32, 64)
    best = min(candidates, key=lambda c: abs(bar_count - c))
    # Beyond the top candidate, report the rounded bar count rather than
    # claiming a 200-bar section is a 64-bar phrase.
    return float(best) if bar_count <= candidates[-1] * 1.5 else round(bar_count, 2)


def detect_sections(full_path: Path, vocals_path: Optional[Path] = None,
                    inst_path: Optional[Path] = None,
                    bass_path: Optional[Path] = None,
                    on_progress: ProgressCb = None) -> List[dict]:
    """Analyse the full mix (and the stems when available) and return an
    ordered list of section dicts: start_sec, end_sec, label, energy,
    vocal_presence, repetition, confidence. Returns [] on failure.

    Boundaries, energy and repetition are measured on the full mix — that is the
    arrangement, and it is what a listener hears as structure.

    Harmony is measured on the STEMS. A mashup lays this track's *vocal* over
    that track's *bed*, so "does this vocal fit that bed" has to be asked of the
    vocal stem's notes and the instrumental stem's chords. Reading both sides off
    the full mix — which is what this did before — means the vocal side's chroma
    is dominated by an arrangement that is about to be thrown away, so the
    measured transposition and the bass-clash tonic describe a record that will
    never be heard. `chroma` (full mix) is still stored so a library analysed
    before this change keeps working.
    """
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
    # librosa returns tempo as a 0-d array in some versions and a float in
    # others; a section's fallback has to be a plain number either way.
    track_bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else None
    if track_bpm is not None and not np.isfinite(track_bpm):
        track_bpm = None
    if len(beats) < 16:
        log.warning("  Too few beats detected — skipping structure analysis.")
        return []
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)

    _tick("Computing beat-synchronous features…")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    rms = _frame_rms(y)

    # Per-stem chroma on the SAME beat grid. Demucs writes stems sample-aligned
    # with the mix, so the frame indices in `beats` index all of them.
    def _stem_chroma(path: Optional[Path], what: str):
        if not path or not Path(path).exists():
            return None
        try:
            ys, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
            # Pad/trim to the mix so sync() cannot run off the end on a stem the
            # separator emitted a few samples short.
            if len(ys) < len(y):
                ys = np.pad(ys, (0, len(y) - len(ys)))
            c = librosa.feature.chroma_cqt(y=ys[:len(y)], sr=sr,
                                           hop_length=HOP_LENGTH)
            return librosa.util.sync(c, beats, aggregate=np.median)
        except Exception:  # noqa: BLE001
            log.warning("  %s chroma failed; falling back to the full mix", what,
                        exc_info=True)
            return None

    _tick("Measuring per-stem harmony…")
    vocal_chroma_b = _stem_chroma(vocals_path, "vocal")
    bed_chroma_b = _stem_chroma(inst_path, "bed")

    # Phase E: a second chroma over the bass region only. Root clash in the low
    # end is the most common reason a "key-compatible" mashup sounds wrong, and
    # it is invisible to a full-spectrum chroma dominated by pads and hi-hats.
    # The dedicated bass stem is the better source when four-stem separation ran;
    # band-passing the mix is the fallback.
    bass_chroma_b = _stem_chroma(bass_path, "bass stem")
    if bass_chroma_b is None:
        try:
            y_bass = _bandpass(y, sr, BASS_LOW_HZ, BASS_HIGH_HZ)
            bass_chroma = librosa.feature.chroma_cqt(y=y_bass, sr=sr,
                                                     hop_length=HOP_LENGTH)
            bass_chroma_b = librosa.util.sync(bass_chroma, beats, aggregate=np.median)
        except Exception:  # noqa: BLE001
            log.warning("  bass chroma failed; sections keep the full-mix chroma only",
                        exc_info=True)

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

    # Snap to the 8-bar phrase grid, measured from this track's own downbeat.
    # The phase is recomputed here rather than read from `features`: analysis
    # trims the head of the file before beat-tracking (BEAT_TRIM_SECS), so the
    # stored beat_phase indexes a different grid from the one above.
    from analysis.analyze import _pick_beat_phase, beat_grid_confidence
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    phase = _pick_beat_phase(onset_env, beats)
    bounds, snapped = snap_boundaries_to_phrases(
        bounds, phase, len(beat_times), min_beats)

    # Beat-index boundaries → [start, end) segments in seconds. Each section
    # inherits the alignment of the boundary that starts it; the first starts at
    # the top of the track, which is not a detected boundary to doubt.
    edges = [0] + bounds + [len(beat_times) - 1]
    aligned = [True] + snapped
    seg_ranges = []
    for idx, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if b <= a:
            continue
        seg_ranges.append((a, b, aligned[idx]))
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
    for a, b, phrase_aligned in seg_ranges:
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

        seg_chroma = chroma_b[:, a:b].mean(axis=1)
        chroma_means.append(seg_chroma)

        # Persist the section's own harmony. The whole-track key is an average,
        # and an average over a record that modulates describes a moment that
        # never occurs — least of all the chorus you actually want to layer.
        from analysis.analyze import key_from_chroma
        section_key = key_from_chroma(seg_chroma)
        # ── the section's own tempo and grid (P2.1) ──────────────────────────
        # Everything here comes off arrays already in memory for the boundary
        # work above, so this costs no extra decode and no extra DSP pass.
        seg_beats = beat_times[a:min(b + 1, len(beat_times))]
        seg_frames = beats[a:min(b + 1, len(beats))]
        grid_conf = beat_grid_confidence(seg_beats, onset_env, seg_frames)
        seg_bpm, bpm_source = _section_bpm(seg_beats, track_bpm, grid_conf)

        beat_count = max(0, len(seg_beats) - 1)
        bar_count = round(beat_count / BEATS_PER_BAR, 3) if beat_count else 0.0
        # Bar lines, on the track's phase so a section's downbeats agree with
        # every other consumer of beat_phase (render/session.py, hooks.py).
        downbeats = [round(float(t), 4) for i, t in enumerate(seg_beats)
                     if (a + i) % BEATS_PER_BAR == phase]

        seg_curve = rms_b[a:b]
        energy_abs = float(seg_curve.mean()) if len(seg_curve) else None
        slope, trend = _energy_shape(
            seg_curve / (rms_max + 1e-9) if len(seg_curve) else None,
            end_t - start_t)

        seg = {
            "start_sec": round(start_t, 2),
            "end_sec": round(end_t, 2),
            "energy": round(energy, 4),
            "vocal_presence": round(vp, 4) if vp is not None else None,
            "phrase_aligned": phrase_aligned,
            "chroma": _norm_chroma(seg_chroma),
            "bpm": seg_bpm,
            "bpm_source": bpm_source,
            "bpm_confidence": round(float(grid_conf), 4),
            "energy_absolute": round(energy_abs, 6) if energy_abs is not None else None,
            "energy_slope": slope,
            "energy_trend": trend,
            "beat_times": [round(float(t), 4) for t in seg_beats],
            "downbeats": downbeats,
            "beat_count": beat_count,
            "bar_count": bar_count,
            "beats_per_bar": BEATS_PER_BAR,
            "phrase_length_bars": _phrase_length(bar_count),
            "section_class": _section_class(vp),
            **section_key,
        }
        if bass_chroma_b is not None:
            seg["bass_chroma"] = _norm_chroma(bass_chroma_b[:, a:b].mean(axis=1))
        # The two the matcher actually wants: what this track SINGS, and what it
        # PLAYS. Absent when the stem was missing, in which case harmony.py
        # falls back to the full-mix chroma above.
        if vocal_chroma_b is not None:
            seg["chroma_vocal"] = _norm_chroma(vocal_chroma_b[:, a:b].mean(axis=1))
        if bed_chroma_b is not None:
            seg["chroma_bed"] = _norm_chroma(bed_chroma_b[:, a:b].mean(axis=1))
        segs.append(seg)
    # Extend the last section to the true end of the audio.
    segs[-1]["end_sec"] = round(duration, 2)

    # Repetition: count near-identical chroma profiles (chorus repeats).
    C = np.array(chroma_means)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sim = Cn @ Cn.T
    for i, s in enumerate(segs):
        s["repetition"] = int((sim[i] >= SECTION_SIM_THRESHOLD).sum())

    label_segments(segs, has_vocals=vocal_rms is not None)
    apply_phrase_alignment(segs)

    log.info("  → " + ", ".join(
        f"{s['label']} {s['start_sec']:.0f}-{s['end_sec']:.0f}s" for s in segs))
    return segs
