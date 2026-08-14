"""analysis/quality.py — is this stem actually usable, and where does it sit in
the spectrum? (Phase D.2 / D.3)

Two questions the ranked list could not previously ask.

**Stem quality.** Separation quality varies enormously with the source material:
dense masters, heavy reverb, doubled vocals and anything already loud-mastered
come back with bleed, spectral holes and warbling. The `stems` table records
which separator produced a file but nothing about how well it did, so a pristine
studio acapella and an artefact-riddled mush rank identically. One unusable
vocal at rank 3 is all it takes to stop trusting the list.

**Band occupancy.** `spectral_centroid` / `rolloff` / `zcr` are single scalars
over a whole track. They cannot express "these two both live in 400 Hz–2 kHz",
which is why a mid-heavy vocal over a mid-dense bed can score well on all four
sub-scores and still be inaudible. An 8-band occupancy vector can.

Everything here is librosa + numpy over audio already on disk. Each metric is
independently fail-safe: a stem that defeats one measurement still gets the rest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Analysis rate for the quality pass. Matches config.SAMPLE_RATE; the Nyquist at
# 22050 is 11 kHz, which is above the band where separation artefacts live.
QUALITY_SR = 22050

# Log-spaced band edges in Hz, from sub-bass to air. Eight bands is enough to
# say "the vocal lives here, the bed is busy there" without becoming a spectrum
# the model has to learn to read.
BAND_EDGES = (0.0, 60.0, 150.0, 400.0, 1000.0, 2500.0, 5000.0, 8000.0, 11025.0)
N_BANDS = len(BAND_EDGES) - 1

# Above this fraction of the full mix's high-band energy, a stem has not been
# smeared. MDX in particular tends to lose the top octave.
HF_BAND_HZ = 6000.0


def _load(path: Path, sr: int = QUALITY_SR, max_secs: float = 240.0):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_secs)
    return y


def _band_energy(y: np.ndarray, sr: int = QUALITY_SR) -> List[float]:
    """Fraction of total energy in each of the N_BANDS bands. Sums to 1.

    A fraction rather than an absolute level: the question is *where* a stem
    sits, not how loud it was mastered, and two records at different loudness
    can occupy exactly the same space.
    """
    import librosa
    if y is None or len(y) < 2048:
        return [0.0] * N_BANDS
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    per_band = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        per_band.append(float(S[mask].sum()) if mask.any() else 0.0)
    total = sum(per_band)
    if total <= 0:
        return [0.0] * N_BANDS
    return [round(v / total, 6) for v in per_band]


def band_energy(path: Path) -> List[float]:
    """Band occupancy for one audio file, or zeros when it cannot be read."""
    try:
        return _band_energy(_load(Path(path)))
    except Exception:  # noqa: BLE001
        log.warning("band_energy failed for %s", path, exc_info=True)
        return [0.0] * N_BANDS


def collision_score(bands_a: Optional[Sequence[float]],
                    bands_b: Optional[Sequence[float]]) -> float:
    """How well two band profiles stay out of each other's way, 0-1.

    1.0 means the top's energy sits exactly where the bed's is quiet; 0.0 means
    they occupy the same bands. Computed as 1 - the overlap coefficient
    (sum of per-band minima), which is the fraction of one profile that lands on
    top of the other.

    Unknown on either side returns the neutral 0.5 rather than a flattering 1.0 —
    "we did not measure it" is not "they are complementary".
    """
    if not bands_a or not bands_b:
        return 0.5
    a = np.asarray(bands_a, dtype=float)
    b = np.asarray(bands_b, dtype=float)
    if a.shape != b.shape or a.sum() <= 0 or b.sum() <= 0:
        return 0.5
    a = a / a.sum()
    b = b / b.sum()
    overlap = float(np.minimum(a, b).sum())
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def collision_block(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """collision_score over every pair of two (n, N_BANDS) matrices.

    Rows that are all zero (unmeasured) yield the same neutral 0.5 the scalar
    form returns.
    """
    a_ok = A.sum(axis=1) > 0
    b_ok = B.sum(axis=1) > 0
    An = A / np.where(A.sum(axis=1, keepdims=True) > 0,
                      A.sum(axis=1, keepdims=True), 1.0)
    Bn = B / np.where(B.sum(axis=1, keepdims=True) > 0,
                      B.sum(axis=1, keepdims=True), 1.0)
    # Per-band minima summed over bands, for every (i, j).
    overlap = np.minimum(An[:, None, :], Bn[None, :, :]).sum(axis=2)
    out = np.clip(1.0 - overlap, 0.0, 1.0)
    return np.where(a_ok[:, None] & b_ok[None, :], out, 0.5)


def residual_vocal_ratio(vocals_path: Optional[Path],
                         full_path: Optional[Path]) -> Optional[float]:
    """How much of a track is voice, 0-1.

    Read on the BED side this is the "is this actually a usable instrumental"
    signal: an instrumental cut from a full record still carries the original
    topline, and laying a new vocal over it produces two competing melodies.
    A high ratio on a bed candidate is a reason to demote it.

    None when either file is missing — unmeasured, not clean.
    """
    if not vocals_path or not full_path:
        return None
    try:
        v = _load(Path(vocals_path))
        f = _load(Path(full_path))
    except Exception:  # noqa: BLE001
        log.warning("residual_vocal_ratio failed", exc_info=True)
        return None
    n = min(len(v), len(f))
    if n < 2048:
        return None
    ev = float(np.sum(v[:n] ** 2))
    ef = float(np.sum(f[:n] ** 2))
    if ef <= 0:
        return None
    return float(np.clip(ev / ef, 0.0, 1.0))


def _bleed(stem: np.ndarray, other: np.ndarray) -> Optional[float]:
    """Correlation between a stem and its complement, 0-1.

    A clean separation leaves the two nearly uncorrelated. Real bleed shows up
    as the same musical events appearing in both, which correlates.
    """
    n = min(len(stem), len(other))
    if n < 2048:
        return None
    a = stem[:n] - stem[:n].mean()
    b = other[:n] - other[:n].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return None
    return float(np.clip(abs(float(np.dot(a, b)) / denom), 0.0, 1.0))


def _hf_loss(stem: np.ndarray, full: np.ndarray, sr: int = QUALITY_SR) -> Optional[float]:
    """How much of the mix's top end the stem lost, 0 (none) to 1 (all).

    The classic MDX artefact is a smeared or missing top octave. Measured
    against the full mix so a genuinely dark record is not mistaken for a
    damaged stem.
    """
    import librosa
    if len(stem) < 2048 or len(full) < 2048:
        return None
    def hf(y):
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask = freqs >= HF_BAND_HZ
        return float(S[mask].sum()), float(S.sum())
    s_hf, s_tot = hf(stem)
    f_hf, f_tot = hf(full)
    if s_tot <= 0 or f_tot <= 0 or f_hf <= 0:
        return None
    # Relative HF share, stem vs full mix.
    share_stem = s_hf / s_tot
    share_full = f_hf / f_tot
    return float(np.clip(1.0 - (share_stem / share_full), 0.0, 1.0))


def _noise_floor(stem: np.ndarray, quiet_windows: Sequence[tuple],
                 sr: int = QUALITY_SR) -> Optional[float]:
    """RMS in regions the stem should be silent, relative to its overall RMS.

    For a vocal stem, "should be silent" means the sections structure detection
    found no voice in. Anything audible there is separation residue.
    """
    if not quiet_windows or len(stem) < 2048:
        return None
    overall = float(np.sqrt(np.mean(stem ** 2)))
    if overall <= 1e-9:
        return None
    vals = []
    for start, end in quiet_windows:
        a, b = int(start * sr), int(end * sr)
        a, b = max(0, a), min(len(stem), b)
        if b - a > sr // 2:
            vals.append(float(np.sqrt(np.mean(stem[a:b] ** 2))))
    if not vals:
        return None
    return float(np.clip(float(np.mean(vals)) / overall, 0.0, 1.0))


def _rollup(bleed: Optional[float], hf_loss: Optional[float],
            noise: Optional[float]) -> float:
    """One 0-1 number, where 1 is a clean stem.

    Components that could not be measured are dropped rather than guessed, and
    a stem where NOTHING could be measured scores the neutral 0.5 — the same
    convention the sub-scores use for unknown inputs.
    """
    parts = [(1.0 - v) for v in (bleed, hf_loss, noise) if v is not None]
    if not parts:
        return 0.5
    return float(np.clip(sum(parts) / len(parts), 0.0, 1.0))


def stem_quality(stem_path: Path, full_path: Path,
                 other_path: Optional[Path] = None,
                 quiet_windows: Optional[Sequence[tuple]] = None) -> Dict:
    """Quality metrics for one separated stem.

    Returns {bleed, hf_loss, noise_floor, quality}; any metric that could not be
    computed is None and is dropped from the roll-up rather than guessed.
    """
    out: Dict[str, Optional[float]] = {
        "bleed": None, "hf_loss": None, "noise_floor": None, "quality": 0.5,
    }
    try:
        stem = _load(Path(stem_path))
        full = _load(Path(full_path))
    except Exception:  # noqa: BLE001
        log.warning("stem_quality could not load %s", stem_path, exc_info=True)
        return out

    if other_path and Path(other_path).exists():
        try:
            out["bleed"] = _bleed(stem, _load(Path(other_path)))
        except Exception:  # noqa: BLE001
            log.warning("bleed failed for %s", stem_path, exc_info=True)
    try:
        out["hf_loss"] = _hf_loss(stem, full)
    except Exception:  # noqa: BLE001
        log.warning("hf_loss failed for %s", stem_path, exc_info=True)
    if quiet_windows:
        try:
            out["noise_floor"] = _noise_floor(stem, quiet_windows)
        except Exception:  # noqa: BLE001
            log.warning("noise_floor failed for %s", stem_path, exc_info=True)

    out["quality"] = _rollup(out["bleed"], out["hf_loss"], out["noise_floor"])
    return out


def quiet_windows_for(sections: Sequence[dict], threshold: float = 0.2) -> List[tuple]:
    """(start, end) of sections the separator found no voice in — where a vocal
    stem should read as silence."""
    out = []
    for s in sections or []:
        vp = s.get("vocal_presence")
        if vp is not None and vp < threshold:
            start, end = s.get("start_sec"), s.get("end_sec")
            if start is not None and end is not None and end > start:
                out.append((float(start), float(end)))
    return out
