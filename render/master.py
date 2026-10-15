"""render/master.py — the last stage of every offline render: integrated
loudness (ITU-R BS.1770-4), true peak, and a lookahead limiter.

Mixdowns used to be peak-normalised only when they clipped, so exports varied
in loudness pair to pair and could sit at 0 dBFS true peak — a DJ's
next-track gain and a streaming upload both pay for that (2026-10-15
simulation). `master` brings a mix to an optional LUFS target and never lets
it past the ceiling (−1 dBTP by default), measured at 4× oversampling.

numpy + scipy only, mono (the mixdown is mono).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

CEILING_DBTP = -1.0
OVERSAMPLE = 4          # BS.1770-4 Annex 2 (_TP_PHASES)
LOOKAHEAD_MS = 5.0
RELEASE_MS = 80.0
# 4× oversampling can read a peak up to ~0.1 dB under a finer meter, so the
# limiter aims that far under the ceiling: −1 dBTP holds by any meter.
TP_MARGIN_DB = 0.1
BLOCK = 64          # gain is decided per block of samples, then interpolated


def _k_weighting(sr: int):
    """The BS.1770 pre-filter (high shelf + RLB high-pass) as SOS for `sr`,
    by the bilinear-transform design every implementation uses."""
    from scipy.signal import tf2sos
    # Stage 1: high shelf, +4 dB above ~1.5 kHz.
    f0, g_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10 ** (g_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b1 = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a1 = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    # Stage 2: high-pass at ~38 Hz.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return np.vstack([tf2sos(b1, a1), tf2sos(b2, a2)])


def integrated_lufs(y, sr: int) -> Optional[float]:
    """Integrated loudness of a mono signal in LUFS (BS.1770-4: K-weighting,
    400 ms blocks at 75% overlap, −70 LUFS absolute gate, −10 LU relative
    gate). None for silence or anything shorter than one block."""
    from scipy.signal import sosfilt
    y = np.asarray(y, dtype="float32")
    n_block = int(round(0.4 * sr))
    step = int(round(0.1 * sr))
    if len(y) < n_block:
        return None
    # float32 filtering (10× faster here), float64 sums (8M-sample cumsums).
    z = sosfilt(_k_weighting(sr).astype("float32"), y).astype("float64") ** 2
    csum = np.concatenate([[0.0], np.cumsum(z)])
    starts = np.arange(0, len(z) - n_block + 1, step)
    ms = (csum[starts + n_block] - csum[starts]) / n_block
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10.0 * np.log10(ms)
    ms = ms[loud > -70.0]
    if not len(ms):
        return None
    rel = -0.691 + 10.0 * math.log10(float(ms.mean())) - 10.0
    with np.errstate(divide="ignore"):
        keep = ms[-0.691 + 10.0 * np.log10(ms) > rel]
    if not len(keep):
        return None
    return round(-0.691 + 10.0 * math.log10(float(keep.mean())), 2)


# The true-peak interpolator BS.1770-4 Annex 2 specifies: 4× oversampling by
# a 48-tap FIR in four 12-tap phases. Short enough to run as four np.convolve
# passes (~0.2 s on a 200 s mix, where scipy's resample_poly with its default
# long filter took ~2 s a call and made a mastered export 2–3× slower).
_TP_PHASES = np.array([
    [0.0017089843750, 0.0109863281250, -0.0196533203125, 0.0332031250000,
     -0.0594482421875, 0.1373291015625, 0.9721679687500, -0.1022949218750,
     0.0476074218750, -0.0266113281250, 0.0148925781250, -0.0083007812500],
    [-0.0291748046875, 0.0292968750000, -0.0517578125000, 0.0891113281250,
     -0.1665039062500, 0.4650878906250, 0.7797851562500, -0.2003173828125,
     0.1015625000000, -0.0582275390625, 0.0330810546875, -0.0189208984375],
    [-0.0189208984375, 0.0330810546875, -0.0582275390625, 0.1015625000000,
     -0.2003173828125, 0.7797851562500, 0.4650878906250, -0.1665039062500,
     0.0891113281250, -0.0517578125000, 0.0292968750000, -0.0291748046875],
    [-0.0083007812500, 0.0148925781250, -0.0266113281250, 0.0476074218750,
     -0.1022949218750, 0.9721679687500, 0.1373291015625, -0.0594482421875,
     0.0332031250000, -0.0196533203125, 0.0109863281250, 0.0017089843750],
], dtype="float64")
# Each phase scaled to unity DC gain: as printed, phases 1–2 sum to 0.973 and
# read a plateau between samples 0.24 dB low — enough to let a −1 dBTP export
# measure −0.87 against a 16× reference.
_TP_PHASES = (_TP_PHASES / _TP_PHASES.sum(axis=1, keepdims=True)).astype("float32")


def _true_abs(y) -> np.ndarray:
    """Per input sample, the largest |value| among it and the three
    interpolated points after it (BS.1770-4 Annex 2)."""
    y = np.asarray(y, dtype="float32")
    out = np.abs(y)
    if not len(y):
        return out
    for h in _TP_PHASES:
        np.maximum(out, np.abs(np.convolve(y, h, mode="same")), out=out)
    return out


def true_peak_db(y) -> Optional[float]:
    """True peak in dBTP (BS.1770-4 4× oversampling). None for silence."""
    if not len(y):
        return None
    peak = float(_true_abs(y).max())
    return round(20.0 * math.log10(peak), 2) if peak > 0 else None


def _limit(y, sr: int, ceiling_dbtp: float, true_abs=None) -> Tuple[np.ndarray, float]:
    """`limit`, also returning the result's true peak (linear). `true_abs` is
    y's own `_true_abs` when the caller has it — the true peak scales with a
    constant gain, so master() measures its input once, not once a pass."""
    y = np.asarray(y, dtype="float32")
    if not len(y):
        return y, 0.0
    ceil = 10.0 ** ((ceiling_dbtp - TP_MARGIN_DB) / 20.0)
    per_sample = _true_abs(y) if true_abs is None else true_abs
    top = float(per_sample.max())
    if top <= ceil:
        return y, top
    nb = int(math.ceil(len(y) / BLOCK))
    pad = np.zeros(nb * BLOCK, dtype="float32")
    pad[: len(per_sample)] = per_sample
    need = np.minimum(1.0, ceil / np.maximum(pad.reshape(nb, BLOCK).max(axis=1), 1e-12))
    # Lookahead: a block already turns down for a peak in the next few.
    ahead = max(1, int(round(LOOKAHEAD_MS / 1000.0 * sr / BLOCK)))
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, size=2 * ahead + 1, origin=0, mode="nearest")
    rel = math.exp(-BLOCK / (RELEASE_MS / 1000.0 * sr))
    g = np.empty(nb)
    cur = 1.0
    for i in range(nb):
        cur = need[i] if need[i] < cur else min(need[i], 1.0 - (1.0 - cur) * rel)
        g[i] = cur
    centres = np.arange(nb) * BLOCK + BLOCK / 2.0
    gain = np.interp(np.arange(len(y), dtype="float32"), centres, g).astype("float32")
    out = y * gain
    peak = float(_true_abs(out).max())
    if peak > ceil:
        out *= ceil / peak
        peak = ceil
    return out.astype("float32"), peak


def limit(y, sr: int, ceiling_dbtp: float = CEILING_DBTP) -> np.ndarray:
    """A lookahead brickwall: the gain each block needs to keep the
    oversampled peak under the ceiling, held back by the lookahead and
    released slowly, interpolated per sample. A final trim catches what the
    interpolation lets through, so the result never exceeds the ceiling."""
    return _limit(y, sr, ceiling_dbtp)[0]


def master(y, sr: int, target_lufs: Optional[float] = None,
           ceiling_dbtp: float = CEILING_DBTP) -> Tuple[np.ndarray, Dict]:
    """The mix brought to `target_lufs` (when given) and limited to the
    ceiling. Two passes: limiting a loud target costs a little loudness, so
    the second pass makes it back where the ceiling allows.
    Returns (samples, {"lufs", "true_peak", "gain_db"})."""
    y = np.asarray(y, dtype="float32")
    gain_db = 0.0
    ta = _true_abs(y)

    def at(db):
        k = 10 ** (db / 20.0)
        return _limit(y * k, sr, ceiling_dbtp, true_abs=ta * k)

    now = integrated_lufs(y, sr) if target_lufs is not None else None
    if now is None:
        out, peak = at(0.0)
    else:
        gain_db = float(target_lufs) - now
        out, peak = at(gain_db)
        after = integrated_lufs(out, sr)
        if after is not None and abs(after - target_lufs) > 0.3:
            gain_db += float(target_lufs) - after
            out, peak = at(gain_db)
    return out, {"lufs": integrated_lufs(out, sr),
                 "true_peak": round(20.0 * math.log10(peak), 2) if peak > 0 else None,
                 "gain_db": round(gain_db, 2)}
