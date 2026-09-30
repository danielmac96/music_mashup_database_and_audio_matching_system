"""
analysis/essentia_groups.py — the Essentia analyser's tier-1 feature groups.

Four groups per audio file, each cached under the file's content hash like the
librosa groups (analysis/registry.py, analysis/cache.py):

  essentia.rhythm    BPM (RhythmExtractor2013), beat grid, grid confidence,
                     beat phase from the kick band, Percival and BPM-histogram
                     votes, onset rate, danceability
  essentia.tonal     key per profile (KeyExtractor) with the configured profile
                     as the answer and cross-profile agreement as confidence,
                     tuning frequency, chords (TonalExtractor), HPCP, dissonance
  essentia.loudness  EBU R128 integrated loudness + range, true peak, ReplayGain,
                     dynamic complexity, crest, stereo width, frame RMS
  essentia.spectral  one windowed-spectrum pass: MFCC, centroid, rolloff, ZCR,
                     flux, flatness, complexity, HFC, contrast, spread/skew/
                     kurtosis, 8-band and 3-band energy, the waveform envelope

and one more, run on the vocal stem only (an ``extra_steps`` of
analyze_file_essentia):

  essentia.melody    the sung pitch (PitchMelodia on the isolated vocal),
                     a 50 ms f0 curve and the sung range — what a section's
                     ``f0`` is cut from (analysis/structure.py)

Every payload uses the key names the ``features`` table already stores (bpm,
key, mfcc, band_energy, waveform_rms, …) for the values that replace a librosa
one, so analysis/project.py can project either analyser into the same row, plus
new keys for what librosa never measured.

One decode per file: FFmpeg to 44.1 kHz stereo float (analysis/decode.py),
mono by averaging, 22.05 kHz by polyphase resampling (scipy — ~5× faster than
essentia.Resample, Phase 0). The decode happens only when some group is not
cached.

Essentia is optional (Linux/Docker only; readme §1). ``available()`` says
whether it imports; nothing here is imported by code that must run without it.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

STEPS = ("rhythm", "tonal", "loudness", "spectral")
# Steps a caller asks for by name, on the stems they make sense for.
EXTRA_STEPS = ("melody", "effnet")

# PitchMelodia on the 22.05 kHz signal: hop 128 is 5.8 ms, and the stored curve
# is the median of each 50 ms of voiced frames — fine enough for a sung note,
# ~2 k points for 100 s instead of 17 k. ~1.8 s per 100 s of audio (Phase 4).
MELODY_HOP = 128
MELODY_FRAME = 1024
MELODY_STEP_SECS = 0.05

SR_FULL = 44100
SR_FRAMES = 22050
FRAME = 2048
HOP = 512
N_WAVEFORM = 360

# Profiles that vote on key confidence. Each KeyExtractor recomputes its own
# HPCP (~0.1 s per 100 s of audio), so the panel is kept to four
# differently-derived profiles: two trained on electronic music (edma, bgate),
# two classic probe-tone profiles. The configured primary
# (config.current_essentia_key_profile) is always added.
KEY_PROFILES = ("edma", "bgate", "krumhansl", "temperley")

_AVAILABLE: Optional[bool] = None


def available() -> bool:
    """Whether essentia imports here (it has no Windows wheels)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import essentia  # noqa: F401
            import essentia.standard  # noqa: F401
            essentia.log.infoActive = False
            essentia.log.warningActive = False
            _AVAILABLE = True
        except Exception:  # noqa: BLE001 — ImportError, or a broken native lib
            _AVAILABLE = False
    return _AVAILABLE


def version() -> Optional[str]:
    if not available():
        return None
    import essentia
    return getattr(essentia, "__version__", None)


# ── Signals ───────────────────────────────────────────────────────────────────

class Signals:
    """The one decode a file's groups share: stereo 44.1k, mono 44.1k, mono 22.05k."""

    def __init__(self, stereo: np.ndarray):
        from scipy.signal import resample_poly
        self.stereo = np.ascontiguousarray(stereo, dtype=np.float32)
        self.mono44 = np.ascontiguousarray(self.stereo.mean(axis=1), dtype=np.float32)
        self.mono22 = np.ascontiguousarray(resample_poly(self.mono44, 1, 2), dtype=np.float32)
        self.secs = len(self.mono44) / float(SR_FULL)


def load_signals(path: Path) -> Signals:
    """soundfile for a lossless file at 44.1 kHz (~2× faster than piping
    FFmpeg's output, measured on WAV), FFmpeg for everything else — MP3 above
    all, where libsndfile took 5-6 s per 4-minute track against FFmpeg's 0.5 s."""
    from analysis.decode import decode_ffmpeg, prefers_ffmpeg
    if prefers_ffmpeg(path):
        try:
            return Signals(decode_ffmpeg(path, sr=SR_FULL, channels=2))
        except Exception:  # noqa: BLE001 — soundfile below is the fallback
            pass
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr == SR_FULL and data.shape[1] in (1, 2):
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            return Signals(data)
    except Exception:  # noqa: BLE001 — anything soundfile cannot open
        pass
    return Signals(decode_ffmpeg(path, sr=SR_FULL, channels=2))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r(v, nd=4) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if np.isfinite(f) else None


def normalise_key(name: str) -> Optional[str]:
    """Essentia spells keys with flats ('Bb', 'Eb'); the library uses sharps."""
    from analysis.analyze import KEY_NAMES
    from analysis.compare import pitch_class
    pc = pitch_class(name)
    return KEY_NAMES[pc] if pc is not None else None


def _onset_envelope(mono22: np.ndarray) -> np.ndarray:
    """Half-wave-rectified log-energy rise per 512-sample frame — the cheap
    onset curve beat-grid salience is measured against."""
    n = len(mono22) // HOP
    if n < 2:
        return np.zeros(max(n, 0))
    frames = mono22[: n * HOP].reshape(n, HOP).astype(np.float64)
    loge = np.log1p(1000.0 * (frames ** 2).mean(axis=1))
    return np.maximum(np.diff(loge, prepend=loge[0]), 0.0)


def onset_rate_from_envelope(env: np.ndarray, secs: float) -> Optional[float]:
    """Onsets per second: peaks of the onset envelope that stand above its
    median by two MADs, at least ~50 ms apart. (essentia.OnsetRate ran its own
    onset detection from scratch for the same number.)"""
    from scipy.signal import find_peaks
    if secs <= 0 or env.size < 3:
        return None
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) or 1e-9
    peaks, _ = find_peaks(env, height=med + 2.0 * mad,
                          distance=max(1, int(0.05 * SR_FRAMES / HOP)))
    return len(peaks) / secs


def true_peak_dbtp(stereo: np.ndarray, windows: int = 16, half: int = 64) -> Optional[float]:
    """True peak by 4× oversampling only around the loudest samples.

    essentia.TruePeakDetector oversamples the whole file (~5 s per 100 s of
    audio, measured); an inter-sample peak can only sit next to a large sample,
    so upsampling ±``half`` samples around the ``windows`` largest per channel
    finds the same maximum for a fraction of the cost."""
    from scipy.signal import resample_poly
    if stereo.size == 0:
        return None
    # All windows go through ONE resample_poly call (it designs its filter per
    # call, which dominated the cost), separated by zero gaps longer than the
    # filter so no window's ringing reaches its neighbour.
    gap = np.zeros(4 * half, dtype=np.float64)
    parts = []
    for ch in range(stereo.shape[1]):
        x = stereo[:, ch]
        k = min(windows, len(x))
        for i in np.argpartition(np.abs(x), -k)[-k:]:
            a, b = max(0, int(i) - half), min(len(x), int(i) + half)
            parts.extend((x[a:b].astype(np.float64), gap))
    up = resample_poly(np.concatenate(parts), 4, 1) if parts else np.zeros(0)
    best = float(np.max(np.abs(up))) if up.size else 0.0
    return round(20.0 * np.log10(best), 3) if best > 0 else None


# ── The groups ────────────────────────────────────────────────────────────────

def hpcp_to_chroma(hpcp_frames: np.ndarray) -> np.ndarray:
    """Mean HPCP folded to the library's chroma convention: 12 bins, index 0 =
    C, L2-normalised (analysis/structure._norm_chroma).

    Essentia's HPCP is referenced to A (bin 0 = A440) and TonalExtractor's has
    three bins per semitone, the centre one on the pitch: semitone i is bins
    3i-1, 3i, 3i+1. C is three semitones above A."""
    h = np.asarray(hpcp_frames, dtype=float)
    if h.ndim != 2 or not len(h):
        return np.zeros(12)
    mean = h.mean(axis=0)
    size = mean.shape[0]
    if size % 12:
        return np.zeros(12)
    per = size // 12
    if per == 1:
        from_a = mean
    else:
        half = per // 2
        from_a = np.array([mean[[(per * i + d) % size for d in range(-half, per - half)]].sum()
                           for i in range(12)])
    chroma = np.roll(from_a, -3)            # index 0: A → C
    n = float(np.linalg.norm(chroma))
    return chroma / n if n > 0 else chroma


def group_rhythm(sig: Signals, method: str = "degara") -> dict:
    import essentia.standard as es
    from analysis.analyze import _pick_beat_phase, beat_grid_confidence

    bpm, ticks, conf, _estimates, intervals = es.RhythmExtractor2013(method=method)(sig.mono44)
    ticks = np.asarray(ticks, dtype=float)

    # Grid confidence on the same scale librosa's is (steadiness × salience):
    # RhythmExtractor's own confidence is 0 by construction for degara, and on
    # a 0-5.32 scale for multifeature — stored, but not comparable.
    env = _onset_envelope(sig.mono22)
    beat_frames = np.round(ticks * SR_FRAMES / HOP).astype(int)
    grid_conf = beat_grid_confidence(ticks, env, beat_frames)

    # Bar phase from the kick band: the beat that starts a bar is loudest below
    # ~150 Hz, which the onset curve blurs with the hats.
    phase = 0
    if len(ticks) >= 4:
        loud, ratios = es.BeatsLoudness(sampleRate=SR_FULL, beats=ticks.tolist())(sig.mono44)
        loud, ratios = np.asarray(loud, dtype=float), np.asarray(ratios, dtype=float)
        low = loud * (ratios[:, 0] if ratios.ndim == 2 and len(ratios) else 1.0)
        phase = _pick_beat_phase(low, np.arange(len(low)))

    hist = {}
    if len(intervals):
        h = es.BpmHistogramDescriptors()(np.asarray(intervals, dtype=np.float32))
        hist = {"first_peak_bpm": _r(h[0], 2), "first_peak_weight": _r(h[1]),
                "second_peak_bpm": _r(h[3], 2), "second_peak_weight": _r(h[4])}
    percival = _r(es.PercivalBpmEstimator(sampleRate=SR_FULL)(sig.mono44), 2)
    onset_rate = _r(onset_rate_from_envelope(env, sig.secs), 3)
    dance = _r(es.Danceability(sampleRate=SR_FULL)(sig.mono44)[0], 4)

    return {
        "bpm": float(round(float(bpm), 2)),
        "bpm_confidence": float(grid_conf),
        "beat_times": [round(float(t), 4) for t in ticks],
        "beat_phase": int(phase),
        "rhythm_confidence": _r(conf),
        "bpm_candidates": {"rhythm_" + method: _r(bpm, 2), "percival": percival, **hist},
        "onset_rate": onset_rate,
        "danceability": dance,
    }


def group_tonal(sig: Signals, primary: str = "edma") -> dict:
    import essentia.standard as es
    from analysis.analyze import CAMELOT, KEY_NAMES

    profiles = list(dict.fromkeys([primary, *KEY_PROFILES]))
    candidates: Dict[str, list] = {}
    for p in profiles:
        try:
            k, scale, strength = es.KeyExtractor(profileType=p, sampleRate=SR_FULL)(sig.mono44)
        except RuntimeError:
            continue                       # a profile this build does not know
        name = normalise_key(k)
        if name:
            candidates[p] = [name, str(scale), round(float(strength), 4)]
    if primary not in candidates:
        raise RuntimeError(f"key profile {primary!r} produced no key")
    key, mode, strength = candidates[primary]
    # Confidence: the primary profile's strength × the share of profiles that
    # name the same key and mode. One profile's strength alone is high for
    # most records; agreement across differently-trained profiles is what
    # separates a clear key from a coin toss.
    agree = sum(1 for c in candidates.values() if c[0] == key and c[1] == mode)
    consensus = agree / len(candidates)

    tuning = es.TuningFrequencyExtractor()(sig.mono44)
    tonal = es.TonalExtractor()(sig.mono44)
    names = es.TonalExtractor().outputNames()
    t = dict(zip(names, tonal))
    hpcp_mean = hpcp_to_chroma(np.asarray(t.get("hpcp"), dtype=float))

    return {
        "key": key, "mode": mode,
        "camelot": CAMELOT.get((KEY_NAMES.index(key), mode), "?"),
        "key_confidence": round(float(strength) * consensus, 4),
        "key_strength": strength,
        "key_consensus": round(consensus, 4),
        "key_candidates": candidates,
        "tuning_hz": _r(np.median(tuning), 3) if len(tuning) else None,
        "chords": {
            "key": normalise_key(str(t.get("chords_key", ""))),
            "scale": str(t.get("chords_scale", "")) or None,
            "changes_rate": _r(t.get("chords_changes_rate")),
            "number_rate": _r(t.get("chords_number_rate")),
            "histogram": [round(float(v), 3) for v in np.asarray(t.get("chords_histogram", []))],
        },
        "hpcp": [round(float(v), 5) for v in hpcp_mean],
    }


def group_loudness(sig: Signals) -> dict:
    import essentia.standard as es

    _m, short, integrated, lra = es.LoudnessEBUR128(sampleRate=SR_FULL)(sig.stereo)
    short = np.asarray(short, dtype=float)
    dyn, _loud = es.DynamicComplexity(sampleRate=SR_FULL)(sig.mono44)
    replay = es.ReplayGain(sampleRate=SR_FULL)(sig.mono44)

    y = sig.mono22
    n = len(y) // HOP
    frame_rms = np.sqrt((y[: n * HOP].reshape(n, HOP).astype(np.float64) ** 2).mean(axis=1)) \
        if n else np.zeros(0)
    peak = float(np.max(np.abs(sig.stereo))) if sig.stereo.size else 0.0
    rms_all = float(np.sqrt(np.mean(sig.mono44.astype(np.float64) ** 2))) if sig.mono44.size else 0.0
    mid = sig.stereo.mean(axis=1).astype(np.float64)
    side = (sig.stereo[:, 0] - sig.stereo[:, 1]).astype(np.float64) / 2.0
    e_mid, e_side = float(np.sum(mid ** 2)), float(np.sum(side ** 2))

    return {
        "loudness_rms": float(round(float(frame_rms.mean()), 6)) if n else None,
        "lufs": _r(integrated, 3),
        "lra": _r(lra, 3),
        "short_term_lufs_max": _r(short.max(), 3) if short.size else None,
        "true_peak": true_peak_dbtp(sig.stereo),
        "sample_peak_db": _r(20 * np.log10(peak), 3) if peak > 0 else None,
        "crest_db": _r(20 * np.log10(peak / rms_all), 3) if peak > 0 and rms_all > 0 else None,
        "replay_gain": _r(replay, 3),
        "dynamic_complexity": _r(dyn, 4),
        "stereo_width": _r(e_side / (e_mid + e_side), 4) if e_mid + e_side > 0 else None,
    }


def group_spectral(sig: Signals, n_mfcc: int = 13, detail_every: int = 4) -> dict:
    """One spectrum per frame (Essentia window + FFT + MFCC, 2048/512 at
    22.05 kHz), every other descriptor computed from those spectra in numpy.

    The descriptors with no cheap vectorised form — spectral contrast,
    complexity, dissonance and the spectral moments — are averaged over every
    ``detail_every``-th frame: a track-level mean over ~1,000 frames instead of
    ~4,000 is the same number to three figures, at a quarter of the cost.
    """
    import essentia.standard as es
    from analysis.analyze import _step_waveform
    from analysis.quality import BAND_EDGES

    sr = SR_FRAMES
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum(size=FRAME)
    mfcc_alg = es.MFCC(inputSize=FRAME // 2 + 1, sampleRate=sr, numberCoefficients=n_mfcc)
    contrast = es.SpectralContrast(frameSize=FRAME, sampleRate=sr)
    complexity = es.SpectralComplexity(sampleRate=sr)
    moments = es.CentralMoments(range=sr / 2)
    shape = es.DistributionShape()
    peaks = es.SpectralPeaks(sampleRate=sr, orderBy="frequency", minFrequency=20,
                             maxFrequency=5000, maxPeaks=60, magnitudeThreshold=1e-5)
    dissonance = es.Dissonance()

    mags, mfccs, zcrs = [], [], []
    detail = {k: [] for k in ("contrast", "valley", "complexity", "spread",
                              "skewness", "kurtosis", "dissonance")}
    for i, frame in enumerate(es.FrameGenerator(sig.mono22, frameSize=FRAME,
                                                hopSize=HOP, startFromZero=True)):
        spec = spectrum(window(frame))
        mags.append(spec)
        mfccs.append(mfcc_alg(spec)[1])
        # Zero crossings per sample, as librosa's zero_crossing_rate reports.
        zcrs.append(float(np.count_nonzero(np.diff(np.signbit(frame)))) / len(frame))
        if i % detail_every == 0:
            c, v = contrast(spec)
            detail["contrast"].append(c)
            detail["valley"].append(v)
            detail["complexity"].append(complexity(spec))
            sp, sk, ku = shape(moments(spec))
            detail["spread"].append(sp)
            detail["skewness"].append(sk)
            detail["kurtosis"].append(ku)
            pf, pm = peaks(spec)
            detail["dissonance"].append(dissonance(pf, pm) if len(pf) > 1 else 0.0)

    if not mags:
        raise RuntimeError("audio too short for one analysis frame")
    mag = np.asarray(mags, dtype=np.float64)            # (frames, bins)
    power = mag ** 2
    freqs = np.linspace(0.0, sr / 2.0, mag.shape[1])
    mag_sum = mag.sum(axis=1)
    pow_sum = power.sum(axis=1)
    ok = mag_sum > 1e-12

    centroid = (mag[ok] * freqs).sum(axis=1) / mag_sum[ok]
    cum = np.cumsum(power[ok], axis=1)
    roll_idx = (cum < 0.85 * cum[:, -1:]).sum(axis=1).clip(0, len(freqs) - 1)
    rolloff = freqs[roll_idx]
    flux = np.linalg.norm(np.diff(mag, axis=0), axis=1) if len(mag) > 1 else np.zeros(1)
    geo = np.exp(np.mean(np.log(mag[ok] + 1e-20), axis=1))
    flat_db = 10.0 * np.log10(geo / (mag[ok].mean(axis=1) + 1e-20) + 1e-20)
    hfc = (power * np.arange(mag.shape[1])).sum(axis=1)

    total = float(power.sum())
    col = power.sum(axis=0)
    bands = [float(col[(freqs >= lo) & (freqs < hi)].sum())
             for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    bands = [round(b / sum(bands), 6) for b in bands] if sum(bands) > 0 else [0.0] * len(bands)
    b3 = [float(col[(freqs >= 20) & (freqs < 250)].sum()),
          float(col[(freqs >= 250) & (freqs < 4000)].sum()),
          float(col[freqs >= 4000].sum())]
    b3 = [round(b / sum(b3), 6) for b in b3] if sum(b3) > 0 else [0.0, 0.0, 0.0]

    def _m(values):
        a = np.asarray(values, dtype=float)
        if a.size == 0:
            return None
        return a.mean(axis=0)

    mf, con, val = _m(mfccs), _m(detail["contrast"]), _m(detail["valley"])
    return {
        "mfcc": [round(float(v), 4) for v in mf] if mf is not None else None,
        "spectral_centroid": _r(centroid.mean() if centroid.size else None, 2),
        "spectral_rolloff": _r(rolloff.mean() if rolloff.size else None, 2),
        "zero_crossing_rate": _r(np.mean(zcrs), 6),
        # Mean |X|² per bin and frame — the same definition as librosa's
        # "energy", on Essentia's (differently scaled) spectrum.
        "energy": float(round(total / (mag.shape[0] * mag.shape[1]), 6)),
        "band_energy": bands,
        "bands3": b3,
        "waveform_rms": _step_waveform(sig.mono22, N_WAVEFORM)["waveform_rms"],
        "spectral": {
            "flux": _r(flux.mean(), 5),
            "flatness_db": _r(flat_db.mean() if flat_db.size else None, 5),
            "hfc": _r(hfc.mean(), 3),
            "complexity": _r(_m(detail["complexity"]), 3),
            "spread": _r(_m(detail["spread"]), 3),
            "skewness": _r(_m(detail["skewness"]), 4),
            "kurtosis": _r(_m(detail["kurtosis"]), 4),
            "dissonance": _r(_m(detail["dissonance"]), 4),
            "contrast": [round(float(v), 4) for v in con] if con is not None else None,
            "valley": [round(float(v), 4) for v in val] if val is not None else None,
        },
    }


def downsample_f0(pitch: np.ndarray, hop_secs: float,
                  step_secs: float = MELODY_STEP_SECS) -> List[float]:
    """Median voiced pitch per ``step_secs`` bin (0 when under half the bin is
    voiced) — a note held through a bin survives, a stray frame does not."""
    pitch = np.asarray(pitch, dtype=float)
    per = max(1, int(round(step_secs / hop_secs)))
    out: List[float] = []
    for i in range(0, len(pitch), per):
        chunk = pitch[i:i + per]
        voiced = chunk[chunk > 0]
        out.append(round(float(np.median(voiced)), 1)
                   if voiced.size * 2 >= chunk.size and voiced.size else 0.0)
    return out


def group_melody(sig: Signals) -> dict:
    """The vocal stem's sung pitch. An isolated vocal is monophonic enough for
    PitchMelodia (the single-source variant); on a full mix the predominant-
    melody version would be needed, at several times the cost, which is why
    this runs on the stem only."""
    import essentia.standard as es
    from analysis.vocals import f0_summary
    # No EqualLoudness pre-filter: it is only defined for 8/16/32/44.1/48 kHz,
    # and on an isolated vocal there is no accompaniment for it to tilt away.
    pitch, _conf = es.PitchMelodia(sampleRate=SR_FRAMES, hopSize=MELODY_HOP,
                                   frameSize=MELODY_FRAME)(sig.mono22)
    curve = downsample_f0(np.asarray(pitch), MELODY_HOP / SR_FRAMES)
    return {"step": MELODY_STEP_SECS, "f0": curve,
            "summary": f0_summary(curve, MELODY_STEP_SECS, 0.0, sig.secs)}


SR_EFFNET = 16000
TOP_N = 5
# Gender is only meaningful for a track that sings.
VOICE_FOR_GENDER = 0.5


def _top(labels: list, probs: np.ndarray, n: int = TOP_N) -> list:
    order = np.argsort(probs)[::-1][:n]
    return [{"label": labels[i], "p": round(float(probs[i]), 4)} for i in order]


def summarise_heads(preds: dict, labels: dict) -> dict:
    """Per-patch head outputs -> the track's tags: each head averaged over the
    track; binary heads keep their positive class, multi-label heads and the
    Discogs styles their top labels. The parent genre sums its styles'
    probability (a track split between House and Tropical House is Electronic
    even when a single Hip Hop style scores highest)."""
    from analysis.ml_models import HEADS
    tags: dict = {"mood": {}}
    for h in HEADS:
        p = np.asarray(preds[h.model], dtype=float)
        mean = p.mean(axis=0) if p.ndim == 2 else p
        lab = labels[h.model]
        if h.kind == "genre":
            tags["genre"] = _top(lab, mean)
            parents: dict = {}
            for name, prob in zip(lab, mean):
                parent = name.split("---", 1)[0]
                parents[parent] = parents.get(parent, 0.0) + float(prob)
            tags["genre_parent"] = max(parents, key=parents.get) if parents else None
        elif h.kind == "multi":
            tags[h.key] = _top(lab, mean)
        else:
            val = round(float(mean[lab.index(h.positive)]), 4)
            if h.model.startswith("mood_"):
                tags["mood"][h.key] = val
            else:
                tags[h.key] = val
    if (tags.get("voice") or 0.0) < VOICE_FOR_GENDER:
        tags["female"] = None
    return tags


def group_effnet(sig: Signals) -> Optional[dict]:
    """Discogs-EffNet genre and tags for the full mix; None (skipped, not
    failed) when the models are not on disk and cannot be fetched."""
    from scipy.signal import resample_poly
    from analysis import ml_models
    if not ml_models.ensure_models():
        return None
    audio16 = resample_poly(sig.mono44, SR_EFFNET // 100, SR_FULL // 100).astype(np.float32)
    try:
        preds = ml_models.predict(audio16)
    except ml_models.ModelsUnavailable:
        return None
    labels = {h.model: ml_models.classes(h.model) for h in ml_models.HEADS}
    emb = np.asarray(preds["embeddings"], dtype=float)
    return {"tags": summarise_heads(preds, labels),
            "embedding_mean": [round(float(v), 5) for v in emb.mean(axis=0)]}


# ── Run all groups for one file ───────────────────────────────────────────────

def _runners() -> Dict[str, Callable[[Signals], dict]]:
    from config import (N_MFCC, current_essentia_key_profile,
                        current_essentia_rhythm_method)
    return {
        "rhythm": lambda s: group_rhythm(s, current_essentia_rhythm_method()),
        "tonal": lambda s: group_tonal(s, current_essentia_key_profile()),
        "loudness": group_loudness,
        "spectral": lambda s: group_spectral(s, N_MFCC),
        "melody": group_melody,
        "effnet": group_effnet,
    }


def runnable_steps(steps: tuple) -> tuple:
    """``steps`` without the tag step when its models are not on disk (and
    cannot be fetched now): dropped before the cache is consulted, so a
    re-analysis of unchanged audio stays a projection instead of decoding the
    file to learn the models are still missing."""
    if "effnet" not in steps:
        return steps
    from analysis import ml_models
    return steps if ml_models.ensure_models() else tuple(s for s in steps if s != "effnet")


def analyze_file_essentia(path: Path, cache=None, timings: Optional[dict] = None,
                          on_progress: Optional[Callable] = None,
                          extra_steps: tuple = ()) -> Dict[str, dict]:
    """Every Essentia group for one file: {step: payload}, plus any of
    EXTRA_STEPS named in ``extra_steps`` (the melody, for a vocal stem).

    ``cache`` is the same protocol analyze_file takes (analysis/cache.py
    StepCache, here bound to the essentia groups). A step that raises is logged,
    listed in ``timings['failed_steps']`` and absent from the result; the file
    is decoded only if some step is not cached.
    """
    if not available():
        raise RuntimeError("essentia is not installed")
    out: Dict[str, dict] = {}
    cached_steps: List[str] = []
    steps = runnable_steps(STEPS + tuple(s for s in extra_steps if s in EXTRA_STEPS))
    if cache is not None:
        for step in steps:
            hit = cache.get(step)
            if hit is not None:
                out[step] = hit
                cached_steps.append(step)
    todo = [s for s in steps if s not in out]
    if timings is not None:
        timings["cached_steps"] = list(cached_steps)
        timings["failed_steps"] = []
    if not todo:
        return out

    if on_progress:
        on_progress(None, "Essentia: decoding…")
    t0 = time.perf_counter()
    sig = load_signals(Path(path))
    if timings is not None:
        timings["load"] = (time.perf_counter() - t0) * 1000.0
        timings["audio_secs"] = sig.secs

    runners = _runners()
    failed: List[str] = []
    for step in todo:
        if on_progress:
            on_progress(None, f"Essentia: {step}…")
        t0 = time.perf_counter()
        try:
            payload = runners[step](sig)
        except Exception:  # noqa: BLE001 — one group must not take the rest down
            log.exception("essentia %s failed for %s", step, Path(path).name)
            failed.append(step)
            payload = None
        ms = (time.perf_counter() - t0) * 1000.0
        if payload is not None:
            out[step] = payload
            if cache is not None:
                cache.put(step, payload, ms)
        if timings is not None:
            timings[step] = ms
    if timings is not None:
        timings["failed_steps"] = failed
    return out
