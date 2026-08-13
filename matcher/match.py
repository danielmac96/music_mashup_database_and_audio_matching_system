"""
matcher/match.py — Score qualifying vocal+instrumental and
instrumental+instrumental pairs and persist to mashup_candidates.

Pre-filter rules (both must pass before scoring):
  1. BPM compatible — within BPM_MAX_DIFF after accounting for halftime/doubletime
  2. Key compatible  — Camelot score >= KEY_MIN_SCORE

Combo types scored:
  vocal_over_instrumental        — song A vocals over song B instrumental
  instrumental_over_instrumental — song A instrumental over song B instrumental
"""
import math
import logging
import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# database.models never imports matcher, so this is safe at module scope — and
# it has to be, so the library-stats cache can read features without threading a
# handle through every scoring call.
from database.models import DB_PATH, get_all_features

log = logging.getLogger(__name__)


# ── Semitone / key helpers ────────────────────────────────────────────────────

def _sanitize_folder_name(s: str, max_len: int = 40) -> str:
    from config import sanitize_filename_chars
    s = sanitize_filename_chars(s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] or "unknown"


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src → dst; fall back to copy if symlinks are unsupported."""
    import os, shutil
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(str(src), str(dst))


# ── Camelot wheel ─────────────────────────────────────────────────────────────

def _parse_camelot(c: str) -> Optional[tuple]:
    if not c or c == "?":
        return None
    try:
        return int(c[:-1]), c[-1]
    except (ValueError, IndexError):
        return None


def camelot_score(c1: str, c2: str) -> float:
    if not c1 or not c2:
        return 0.5
    p1 = _parse_camelot(c1)
    p2 = _parse_camelot(c2)
    if p1 is None or p2 is None:
        return 0.5
    n1, s1 = p1
    n2, s2 = p2
    if n1 == n2 and s1 == s2:          return 1.00   # perfect
    if s1 == s2 and abs(n1-n2) in (1, 11): return 0.85  # adjacent on wheel
    if n1 == n2 and s1 != s2:          return 0.75   # relative major/minor
    if s1 == s2 and abs(n1-n2) in (2, 10): return 0.55  # two steps
    return 0.25


def compute_semitone_shift(vocal_camelot: str, inst_camelot: str) -> Optional[int]:
    """
    Minimum semitones to shift the INSTRUMENTAL so it sits in the vocal's key.
    Positive = shift up, negative = shift down. Range: -6 to +6.
    Returns None if either Camelot code is unknown.

    Derived from the Camelot pair, not from raw root notes, so the shift agrees
    with camelot_score by construction — a pair the wheel calls compatible is
    never handed a destructive transposition. One step around the wheel is a
    perfect fifth, so the shift is 7 × the hour difference, folded into [-6, +6].

    The letter is deliberately ignored: 8A and 8B (A minor / C major) are the
    same pitch collection, so a relative major/minor pair needs no transposition
    at all. The old root-note formula returned +3 there, dragging the bed to C
    minor against the vocal's major third.
    """
    v = _parse_camelot(vocal_camelot or "")
    i = _parse_camelot(inst_camelot or "")
    if v is None or i is None:
        return None
    diff = ((v[0] - i[0]) * 7) % 12
    return diff if diff <= 6 else diff - 12


# ── BPM compatibility ─────────────────────────────────────────────────────────

def _bpm(value) -> float:
    """Coerce a possibly-NULL/garbage tempo to a float. features.bpm is
    nullable — a track whose tempo step failed stores None, and .get("bpm", 0)
    returns that None rather than the default, so every arithmetic path below
    has to survive it."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def effective_bpm(target_bpm: float, other_bpm: float) -> float:
    """other_bpm interpreted at half/normal/double time, whichever lands
    closest to target_bpm."""
    target_bpm, other_bpm = _bpm(target_bpm), _bpm(other_bpm)
    if not target_bpm or not other_bpm:
        return other_bpm or 0.0
    options = (other_bpm, other_bpm / 2, other_bpm * 2)
    return min(options, key=lambda b: abs(target_bpm - b))


def compute_stretch_factor(vocal_bpm: float, inst_bpm: float) -> Optional[float]:
    """Ratio to stretch the instrumental (at whichever of half/normal/double
    time is closest to the vocal) to reach the vocal's tempo."""
    vocal_bpm, inst_bpm = _bpm(vocal_bpm), _bpm(inst_bpm)
    if not vocal_bpm or not inst_bpm:
        return None
    inst_eff = effective_bpm(vocal_bpm, inst_bpm)
    return round(vocal_bpm / inst_eff, 4) if inst_eff else None


def _bpm_min_diff(bpm1: float, bpm2: float) -> float:
    """Smallest BPM difference accounting for halftime and doubletime."""
    bpm1, bpm2 = _bpm(bpm1), _bpm(bpm2)
    if bpm1 <= 0 or bpm2 <= 0:
        return 999.0
    return abs(bpm1 - effective_bpm(bpm1, bpm2))


def bpm_score(bpm1: float, bpm2: float) -> float:
    diff = _bpm_min_diff(bpm1, bpm2)
    if diff < 0.5:  return 1.00
    if diff < 3:    return 0.95
    if diff < 6:    return 0.85
    if diff < 10:   return 0.65
    if diff < 15:   return 0.40
    if diff < 25:   return 0.20
    return max(0.0, 0.20 - (diff - 25) / 100)


# ── Library statistics (T2.2) ─────────────────────────────────────────────────
#
# Timbre and energy only mean anything RELATIVE to the rest of the library.
# Both normalisations need library-wide mean/std, which is read once per scoring
# run and passed down — recomputing it per pair would re-read every features row
# 810k times on a 900-song library.

class LibraryStats:
    """Normalisation constants for timbre and energy, over one library.

    mfcc_mean/mfcc_std cover coefficients 1..12 only (0 is dropped — see
    timbre_score). `loudness` is keyed by stem_type because vocal stems are
    systematically quieter than instrumental ones; comparing them on a raw scale
    measures which stem it is rather than whether the two tracks fit.
    """

    __slots__ = ("mfcc_mean", "mfcc_std", "loudness", "n")

    def __init__(self, mfcc_mean=None, mfcc_std=None, loudness=None, n=0):
        self.mfcc_mean = mfcc_mean
        self.mfcc_std = mfcc_std
        self.loudness = loudness or {}
        self.n = n

    @property
    def usable(self) -> bool:
        """False for a library too small to normalise against, in which case
        callers fall back to the un-normalised comparison rather than dividing
        by a meaningless std."""
        return self.n >= 4 and self.mfcc_mean is not None


_STATS_CACHE: Dict[str, "LibraryStats"] = {}

# Stems that get matched. 'full' is included so its stats exist for fallbacks.
_STEM_KINDS = ("vocals", "instrumental", "full")


def compute_library_stats(db_path=None) -> LibraryStats:
    rows = []
    for stem in _STEM_KINDS:
        rows.extend(get_all_features(stem_type=stem,
                                     db_path=db_path or DB_PATH) or [])
    mats = [r["mfcc"] for r in rows if r.get("mfcc") and len(r["mfcc"]) >= 13]
    mfcc_mean = mfcc_std = None
    if len(mats) >= 4:
        M = np.array(mats, dtype=float)[:, 1:13]
        mfcc_mean = M.mean(axis=0)
        # Guard a degenerate coefficient (constant across the library) so it
        # contributes 0 rather than exploding the z-score.
        mfcc_std = np.where(M.std(axis=0) > 1e-9, M.std(axis=0), 1.0)

    loudness = {}
    for stem in _STEM_KINDS:
        vals = [r.get("loudness_rms") for r in rows
                if r.get("stem_type") == stem and (r.get("loudness_rms") or 0) > 0]
        if len(vals) >= 2:
            lg = np.log(np.array(vals, dtype=float))
            loudness[stem] = (float(lg.mean()), float(max(lg.std(), 1e-6)))
    return LibraryStats(mfcc_mean, mfcc_std, loudness, len(mats))


def get_library_stats(db_path=None, refresh: bool = False) -> LibraryStats:
    """Cached per DB path. score_all_pairs refreshes once at the start so a
    re-analysis is picked up; everything else reuses the cached constants."""
    key = str(db_path or DB_PATH)
    if refresh or key not in _STATS_CACHE:
        _STATS_CACHE[key] = compute_library_stats(db_path)
    return _STATS_CACHE[key]


# ── Energy compatibility ──────────────────────────────────────────────────────

def energy_score(e1: float, e2: float) -> float:
    """Raw level-ratio similarity. Kept as a model input feature and for
    callers without library context; sub_scores uses energy_match instead."""
    if e1 <= 0 or e2 <= 0:
        return 0.5
    ratio = min(e1, e2) / max(e1, e2)
    return float(math.exp(-((1 - ratio) ** 2) / (2 * 0.25 ** 2)))


def _loudness_z(feat: dict, stats: LibraryStats) -> Optional[float]:
    """How loud this stem is *for its kind*, in standard deviations."""
    rms = feat.get("loudness_rms") or 0.0
    if rms <= 0:
        return None
    ref = stats.loudness.get(feat.get("stem_type") or "full") \
        or stats.loudness.get("full")
    if not ref:
        return None
    mu, sd = ref
    return (math.log(rms) - mu) / sd


def energy_match(feat_a: dict, feat_b: dict, stats: LibraryStats) -> float:
    """Do these two sit at a comparable level *within their own stem types*?

    A raw min/max ratio over commercially mastered, loudness-normalised releases
    carries almost no information, and across stem types it mostly reports that
    vocal stems are quieter than instrumentals — a constant offset dressed up as
    a score. Comparing z-scores removes that offset and leaves the real signal:
    a belted chorus over a sparse bed genuinely does not sit right.
    """
    za, zb = _loudness_z(feat_a, stats), _loudness_z(feat_b, stats)
    if za is None or zb is None:
        return energy_score(feat_a.get("loudness_rms") or feat_a.get("energy") or 0,
                            feat_b.get("loudness_rms") or feat_b.get("energy") or 0)
    # 1.2 sd of separation is where a pairing starts to feel unbalanced.
    return float(math.exp(-((za - zb) ** 2) / (2 * 1.2 ** 2)))


# ── Timbre similarity ─────────────────────────────────────────────────────────

def mfcc_cosine(mfcc1: list, mfcc2: list) -> float:
    """Raw cosine over the full MFCC vector. Retained as a model input feature
    and for backwards compatibility; sub_scores uses timbre_score instead."""
    if not mfcc1 or not mfcc2:
        return 0.5
    v1 = np.array(mfcc1, dtype=float)
    v2 = np.array(mfcc2, dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / norm, 0, 1))


def timbre_score(feat_a: dict, feat_b: dict, stats: LibraryStats) -> float:
    """Timbral similarity that can actually tell two records apart.

    mfcc_cosine compares raw mean-MFCC vectors, where coefficient 0 is a large
    same-sign loudness term — measured on a real library it is ~12x the mean
    magnitude of coefficients 1-12 — so the dot product is dominated by "both of
    these are music" and every pair lands near 1.0. Clipping to [0, 1] then
    discarded what spread survived.

    So: drop c0, z-score c1-12 against the library (otherwise the
    high-variance low coefficients still swamp the rest), and map the cosine
    from [-1, 1] onto [0, 1] instead of clipping — an anti-correlated timbre is
    the most different thing available, not the same as an orthogonal one.
    """
    m1, m2 = feat_a.get("mfcc") or [], feat_b.get("mfcc") or []
    if len(m1) < 13 or len(m2) < 13:
        return 0.5                      # unknown, not "perfectly similar"
    if not stats.usable:
        return mfcc_cosine(m1, m2)      # too small a library to normalise against

    v1 = (np.array(m1[1:13], dtype=float) - stats.mfcc_mean) / stats.mfcc_std
    v2 = (np.array(m2[1:13], dtype=float) - stats.mfcc_mean) / stats.mfcc_std
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-12:
        return 0.5
    cos = float(np.dot(v1, v2) / norm)
    return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))


# ── Pre-filter ────────────────────────────────────────────────────────────────

def _passes_filter(feat_a: dict, feat_b: dict,
                   bpm_max_diff: float, key_min_score: float) -> bool:
    """
    Returns True only if the pair meets both BPM and key thresholds.
    Both conditions must pass — failing either skips scoring entirely.
    """
    bpm_a = feat_a.get("bpm") or 0
    bpm_b = feat_b.get("bpm") or 0
    if _bpm_min_diff(bpm_a, bpm_b) > bpm_max_diff:
        return False

    key_s = camelot_score(feat_a.get("camelot", ""),
                           feat_b.get("camelot", ""))
    if key_s < key_min_score:
        return False

    return True


# ── Composite score ───────────────────────────────────────────────────────────

def sub_scores(feat_a: dict, feat_b: dict,
               stats: Optional[LibraryStats] = None) -> dict:
    """The four heuristic sub-scores for a (top, bed) pair. Extracted so the
    learned pair-feature builder (matcher/features.py) and the heuristic
    composite_score compute them identically — keeping train and serve aligned.

    `stats` supplies the library normalisation for timbre and energy. It is
    resolved from the cache when omitted so every caller — scoring, dataset
    build, inference — normalises against the same constants; passing it
    explicitly just avoids the dict lookup in a hot loop.
    """
    stats = stats if stats is not None else get_library_stats()
    return {
        "bpm_score":    bpm_score(feat_a.get("bpm", 0),
                                   feat_b.get("bpm", 0)),
        "key_score":    camelot_score(feat_a.get("camelot", ""),
                                       feat_b.get("camelot", "")),
        "energy_score": energy_match(feat_a, feat_b, stats),
        "timbre_score": timbre_score(feat_a, feat_b, stats),
    }


def composite_score(feat_a: dict, feat_b: dict,
                    weights: Optional[Dict] = None,
                    stats: Optional[LibraryStats] = None) -> dict:
    try:
        from config import MATCH_WEIGHTS
        weights = weights or MATCH_WEIGHTS
    except ImportError:
        weights = {"bpm_score": 0.25, "key_score": 0.30,
                   "energy_score": 0.20, "timbre_score": 0.25}

    scores = sub_scores(feat_a, feat_b, stats)
    scores["total"] = round(
        sum(scores[k] * weights.get(k, 0) for k in scores), 4
    )
    return scores


# ── Vectorised bulk scoring (T3.1) ────────────────────────────────────────────
#
# At ~900 songs the space is 810k vocal×instrumental evaluations plus ~405k
# instrumental×instrumental. Walking that in Python — a dict lookup per feature,
# four function calls per pair, and one sqlite connection + commit per survivor —
# is a multi-hour job dominated by interpreter and fsync overhead, not by maths.
#
# The block functions below compute exactly the same four sub-scores over numpy
# blocks: same formulas, same fallbacks, same constants. They are the ONLY copy
# of that arithmetic that may exist alongside the scalar one, and
# tests/test_score_vectorised.py asserts pair-for-pair agreement with
# composite_score over a whole synthetic library so the two cannot drift.
#
# Pairs are generated by bucketing candidates on tempo (and, on the heuristic
# path, key) before any score is computed — the gate is evaluated as a boolean
# block and only survivors are scored, so the expensive work scales with the
# number of plausible pairs rather than the number of possible ones.

_ENERGY_Z_SIGMA = 1.2       # sd of separation where a pairing starts to feel off
_ENERGY_RATIO_SIGMA = 0.25  # spread of the raw-ratio fallback
_TIMBRE_NORM_FLOOR = 1e-12
_MISSING_BPM_DIFF = 999.0

# Rows of the vocal side handled per block. Bounds peak memory at
# _BLOCK_ROWS × n_beds float64 matrices (a handful of MB at any library size)
# and gives the progress callback something to report against.
_BLOCK_ROWS = 256


class _StemBlock:
    """Column arrays for one stem's feature rows, prepared once per run.

    Everything here is per-track, not per-pair: the loudness z-score, the
    z-scored MFCC vector and its norm, the Camelot code's index into the lookup
    table. Doing it once turns the per-pair work into pure array arithmetic."""

    __slots__ = ("feats", "song_id", "bpm", "code", "loud_z", "loud_raw",
                 "mfcc_v", "mfcc_norm", "mfcc_ok", "variant")


def _camelot_index(feat_lists) -> tuple:
    """Map every Camelot string in the library to a small integer, and build the
    exact score table over those strings with the scalar camelot_score.

    A real library holds at most 24 codes plus unknowns, so the table costs a
    few hundred scalar calls and then every key comparison is an array index.
    Deriving it from camelot_score rather than reimplementing the wheel means
    odd codes (blank, '?', out-of-range hours) score identically to before."""
    codes: List[str] = []
    index: Dict[str, int] = {}
    for feats in feat_lists:
        for f in feats:
            c = str(f.get("camelot") or "")
            if c not in index:
                index[c] = len(codes)
                codes.append(c)
    table = np.array([[camelot_score(a, b) for b in codes] for a in codes],
                     dtype=np.float64).reshape(len(codes), len(codes))
    return index, table


def _prepare_block(feats: List[dict], code_index: Dict[str, int],
                   stats: LibraryStats) -> _StemBlock:
    b = _StemBlock()
    b.feats = feats
    n = len(feats)
    b.song_id = np.array([f["song_id"] for f in feats], dtype=np.int64)
    b.bpm = np.array([_bpm(f.get("bpm")) for f in feats], dtype=np.float64)
    b.code = np.array([code_index[str(f.get("camelot") or "")] for f in feats],
                      dtype=np.intp)

    # Near-duplicate group (A.2). 0 means "no known variant" — sentinel rather
    # than NaN so the comparison stays an integer equality, and chosen as 0
    # because cluster ids are real song_ids and sqlite AUTOINCREMENT starts at 1.
    b.variant = np.array([int(f.get("variant_cluster") or 0) for f in feats],
                         dtype=np.int64)

    # Energy: the z-score when the library has a reference for this stem kind,
    # otherwise NaN — which selects the raw-ratio fallback, exactly as
    # energy_match does when _loudness_z returns None.
    loud_z = np.empty(n, dtype=np.float64)
    loud_raw = np.empty(n, dtype=np.float64)
    for idx, f in enumerate(feats):
        z = _loudness_z(f, stats)
        loud_z[idx] = np.nan if z is None else z
        loud_raw[idx] = float(f.get("loudness_rms") or f.get("energy") or 0.0)
    b.loud_z, b.loud_raw = loud_z, loud_raw

    # Timbre: coefficients 1-12, z-scored against the library (see timbre_score).
    # mfcc_ok is False for a track whose analysis produced a short vector; those
    # pairs score the neutral 0.5 rather than a bogus similarity.
    b.mfcc_ok = np.zeros(n, dtype=bool)
    if stats.usable:
        V = np.zeros((n, 12), dtype=np.float64)
        for idx, f in enumerate(feats):
            m = f.get("mfcc") or []
            if len(m) >= 13:
                V[idx] = (np.array(m[1:13], dtype=float)
                          - stats.mfcc_mean) / stats.mfcc_std
                b.mfcc_ok[idx] = True
        b.mfcc_v = V
        b.mfcc_norm = np.linalg.norm(V, axis=1)
    else:
        b.mfcc_v = None
        b.mfcc_norm = None
    return b


def _bpm_min_diff_block(a_bpm: np.ndarray, b_bpm: np.ndarray) -> np.ndarray:
    """_bpm_min_diff over every pair: the smallest distance to the bed read at
    half, normal or double time. A missing tempo on either side is 999."""
    A = a_bpm[:, None]
    B = b_bpm[None, :]
    diff = np.minimum(np.minimum(np.abs(A - B), np.abs(A - B / 2.0)),
                      np.abs(A - B * 2.0))
    return np.where((A <= 0) | (B <= 0), _MISSING_BPM_DIFF, diff)


def _bpm_score_block(diff: np.ndarray) -> np.ndarray:
    """The bpm_score step function, applied to a matrix of BPM distances."""
    return np.select(
        [diff < 0.5, diff < 3, diff < 6, diff < 10, diff < 15, diff < 25],
        [1.00, 0.95, 0.85, 0.65, 0.40, 0.20],
        default=np.maximum(0.0, 0.20 - (diff - 25) / 100),
    )


def _energy_block(a: _StemBlock, b: _StemBlock, rows: slice) -> np.ndarray:
    """energy_match over every pair in the block."""
    za = a.loud_z[rows][:, None]
    zb = b.loud_z[None, :]
    zscored = np.exp(-((za - zb) ** 2) / (2 * _ENERGY_Z_SIGMA ** 2))

    # energy_score fallback for pairs where either side has no usable reference.
    ra = a.loud_raw[rows][:, None]
    rb = b.loud_raw[None, :]
    lo = np.minimum(ra, rb)
    hi = np.maximum(ra, rb)
    ratio = lo / np.where(hi > 0, hi, 1.0)      # denominator guarded, not clipped
    fallback = np.where(
        (ra <= 0) | (rb <= 0), 0.5,
        np.exp(-((1 - ratio) ** 2) / (2 * _ENERGY_RATIO_SIGMA ** 2)))

    return np.where(np.isnan(za) | np.isnan(zb), fallback, zscored)


def _timbre_block(a: _StemBlock, b: _StemBlock, rows: slice,
                  stats: LibraryStats) -> np.ndarray:
    """timbre_score over every pair in the block.

    Falls back to the scalar path when the library is too small to normalise
    against — that branch compares raw MFCC vectors of whatever length analysis
    produced, and by definition it only happens for a handful of tracks."""
    n_a = len(a.feats[rows])
    n_b = len(b.feats)
    if not stats.usable:
        out = np.empty((n_a, n_b), dtype=np.float64)
        for i, fa in enumerate(a.feats[rows]):
            for j, fb in enumerate(b.feats):
                out[i, j] = timbre_score(fa, fb, stats)
        return out

    V = a.mfcc_v[rows]
    na = a.mfcc_norm[rows][:, None]
    nb = b.mfcc_norm[None, :]
    prod = na * nb
    safe = np.where(prod < _TIMBRE_NORM_FLOOR, 1.0, prod)
    cos = (V @ b.mfcc_v.T) / safe
    out = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)
    out = np.where(prod < _TIMBRE_NORM_FLOOR, 0.5, out)
    # A short MFCC vector is "unknown", not "identical" — 0.5, checked before
    # the library normalisation just as timbre_score does.
    usable = a.mfcc_ok[rows][:, None] & b.mfcc_ok[None, :]
    return np.where(usable, out, 0.5)


def _iter_scored_pairs(top: _StemBlock, bed: _StemBlock, *,
                       stats: LibraryStats, key_table: np.ndarray,
                       bpm_max: float, key_min: Optional[float],
                       upper_triangle: bool, weights: Dict,
                       on_block=None):
    """Yield (top_idx, bed_idx, scores) for every pair that passes the gate.

    Emission order is row-major over the top side, i.e. exactly the order the
    nested `for top: for bed:` loop produced — so ties in the final ranking
    resolve the same way they always have.

    key_min None applies the BPM window only (the model path: documented
    mashups sometimes break the key gate). upper_triangle keeps only
    top.song_id < bed.song_id, for the instrumental×instrumental pass.
    """
    n_top = len(top.feats)
    if not n_top or not len(bed.feats):
        return
    w_bpm = weights.get("bpm_score", 0)
    w_key = weights.get("key_score", 0)
    w_energy = weights.get("energy_score", 0)
    w_timbre = weights.get("timbre_score", 0)

    for start in range(0, n_top, _BLOCK_ROWS):
        rows = slice(start, min(start + _BLOCK_ROWS, n_top))

        # ── Gate first: only survivors are worth scoring ──────────────────────
        diff = _bpm_min_diff_block(top.bpm[rows], bed.bpm)
        keep = diff <= bpm_max
        key_s = key_table[np.ix_(top.code[rows], bed.code)]
        if key_min is not None:
            keep &= key_s >= key_min
        ids_a = top.song_id[rows][:, None]
        ids_b = bed.song_id[None, :]
        keep &= (ids_a < ids_b) if upper_triangle else (ids_a != ids_b)

        # Same work, different upload (Original/Extended/Radio/remix/re-upload).
        # These agree on every sub-score by construction, so without this they
        # take the top of the list with pairings that are not mashups. 0 is the
        # "no known variant" sentinel and must never match itself.
        va = top.variant[rows][:, None]
        vb = bed.variant[None, :]
        keep &= ~((va == vb) & (va > 0))

        if keep.any():
            bpm_s = _bpm_score_block(diff)
            energy_s = _energy_block(top, bed, rows)
            timbre_s = _timbre_block(top, bed, rows, stats)
            total = (bpm_s * w_bpm + key_s * w_key
                     + energy_s * w_energy + timbre_s * w_timbre)

            # np.nonzero returns row-major index pairs, preserving the original
            # iteration order. round() stays Python's so the persisted total is
            # bit-identical to composite_score's.
            for i, j in zip(*np.nonzero(keep)):
                yield start + int(i), int(j), {
                    "bpm_score": float(bpm_s[i, j]),
                    "key_score": float(key_s[i, j]),
                    "energy_score": float(energy_s[i, j]),
                    "timbre_score": float(timbre_s[i, j]),
                    "total": round(float(total[i, j]), 4),
                }
        if on_block is not None:
            on_block(rows.stop, n_top)


def _apply_model_scores(pairs, bundle, sections_of, stats: LibraryStats,
                        on_block=None, batch: int = 4096) -> None:
    """Replace each pair's heuristic total with the learned model's probability.

    Mutates the scores dicts in place. Batched because scikit-learn's per-call
    overhead dwarfs the arithmetic for a single row — the model path scores
    every pair inside the BPM window, which is a lot of single rows."""
    from matcher.features import pair_features
    from matcher.model_scorer import model_score_batch

    n = len(pairs)
    for start in range(0, n, batch):
        chunk = pairs[start:start + batch]
        feats = [pair_features(top, bed,
                               sections_of(top.get("song_id")),
                               sections_of(bed.get("song_id")), stats)
                 for top, bed, _ in chunk]
        for (_, _, scores), prob in zip(chunk, model_score_batch(feats, bundle)):
            scores["total"] = round(float(prob), 4)
        if on_block is not None:
            on_block(min(start + batch, n), n)


# ── Score all qualifying pairs ────────────────────────────────────────────────

def _with_full_bpm(feat: dict, full_by_song: Dict[int, dict]) -> dict:
    """Return a copy of `feat` with bpm/bpm_confidence swapped to the song's
    full-mix values when available, so matching uses the more reliable
    whole-track tempo while key/camelot/timbre stay stem-derived."""
    full_feat = full_by_song.get(feat.get("song_id"))
    if not full_feat or not full_feat.get("bpm"):
        return feat
    out = dict(feat)
    out["stem_bpm"] = feat.get("bpm")
    out["bpm"] = full_feat["bpm"]
    out["bpm_confidence"] = full_feat.get("bpm_confidence")
    return out


def score_all_pairs(db_path=None, bpm_max_diff: Optional[float] = None,
                    key_min_score: Optional[float] = None,
                    scorer: str = "auto", progress=None) -> dict:
    """
    Score every unique cross-song pair that passes the pre-filter.
    Handles two combo types:
      - vocal_over_instrumental
      - instrumental_over_instrumental

    bpm_max_diff / key_min_score override the config defaults (BPM_MAX_DIFF /
    KEY_MIN_SCORE) so the Mashups UI can widen or narrow the candidate set.

    scorer:
      'auto'      — use the active learned model if one loads, else heuristic
      'heuristic' — the hand-weighted composite score (byte-for-byte legacy path)
      'model'     — force the learned model; silently falls back to heuristic if
                    no active model can be loaded

    Heuristic path pre-filters on BOTH the BPM and key gates. The model path uses
    the BPM window ONLY (documented mashups sometimes break the key gate), still
    computes the four heuristic sub-scores for display, and sets score_total to
    the model's probability.

    progress: optional callable (pct: int, message: str) — the standard job
    progress callback, so a library-wide re-score reports movement instead of
    looking hung.

    The candidates table is cleared first so the result reflects exactly the
    current features and thresholds — no stale pairs from a looser prior run.

    Returns { 'vocal_over_instrumental': [...], 'instrumental_over_instrumental': [...] }
    Each list is sorted by total score descending.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import (
        bulk_upsert_candidates, candidate_row, clear_candidates,
        get_all_features, get_sections, DB_PATH,
    )
    from config import BPM_MAX_DIFF, KEY_MIN_SCORE, MATCH_WEIGHTS

    db = db_path or DB_PATH
    bpm_max = float(bpm_max_diff) if bpm_max_diff is not None else BPM_MAX_DIFF
    key_min = float(key_min_score) if key_min_score is not None else KEY_MIN_SCORE

    # Resolve which scorer to use. 'auto'/'model' try to load the active model;
    # both fall back to the heuristic when none is available.
    bundle = None
    if scorer in ("auto", "model"):
        try:
            from matcher.model_scorer import load_active_model
            bundle = load_active_model(db_path=db)
        except Exception:  # noqa: BLE001 — never let model loading break scoring
            bundle = None
    use_model = bundle is not None
    active_scorer = "model" if use_model else "heuristic"
    model_version = bundle.get("version") if use_model else None

    clear_candidates(db_path=db)

    vocals      = get_all_features(stem_type="vocals",        db_path=db)
    inst        = get_all_features(stem_type="instrumental",  db_path=db)
    full        = get_all_features(stem_type="full",          db_path=db)
    full_by_song = {f["song_id"]: f for f in full}

    # Sections feed two things: the model's pair features, and the winning
    # section pair stored on every vocal-over-instrumental row (T3.3). Read once
    # per song — the alternative is one query per candidate.
    _sections_cache: Dict[int, list] = {}

    def _sections(song_id):
        if song_id not in _sections_cache:
            _sections_cache[song_id] = get_sections(song_id, db_path=db)
        return _sections_cache[song_id]

    # Read the library's normalisation constants ONCE for this run — timbre and
    # energy are relative measures, and recomputing them per pair would re-read
    # every features row for every candidate. Refreshed here so a re-analysis
    # since the last run is picked up.
    lib_stats = get_library_stats(db_path=db, refresh=True)

    # Tempo compatibility is scored off the full-mix BPM (vocal-stem and even
    # instrumental-stem tempo tracking can drift octave/onset-detection errors
    # introduced by stem separation artifacts) while key/camelot/timbre stay
    # stem-derived, since those are what the listener actually hears layered.
    vocals = [_with_full_bpm(v, full_by_song) for v in vocals]
    inst   = [_with_full_bpm(i, full_by_song) for i in inst]

    code_index, key_table = _camelot_index([vocals, inst])
    v_block = _prepare_block(vocals, code_index, lib_stats)
    i_block = _prepare_block(inst, code_index, lib_stats)

    results = {
        "vocal_over_instrumental":        [],
        "instrumental_over_instrumental": [],
    }
    rows: List[tuple] = []
    scored = 0

    def _report(base: int, span: int, label: str):
        """Map a pass's block progress onto a slice of the 0-100 job bar."""
        def _on_block(done: int, total: int) -> None:
            if progress:
                pct = base + int(span * done / max(total, 1))
                progress(pct, f"{label}: {done}/{total} tracks")
        return _on_block

    def _run(top_block, bed_block, *, key_gate, upper_triangle, on_block):
        """Every surviving pair of one pass, as (top_feat, bed_feat, scores)."""
        return [
            (top_block.feats[ti], bed_block.feats[bi], scores)
            for ti, bi, scores in _iter_scored_pairs(
                top_block, bed_block, stats=lib_stats, key_table=key_table,
                bpm_max=bpm_max, key_min=key_gate,
                upper_triangle=upper_triangle, weights=MATCH_WEIGHTS,
                on_block=on_block)
        ]

    # The sections each side would actually contribute, filtered and ordered
    # once per song. best_section_pair then only walks the survivors.
    _usable_cache: Dict[tuple, list] = {}

    def _usable(song_id: int, vocal_side: bool) -> list:
        key = (song_id, vocal_side)
        if key not in _usable_cache:
            from matcher.sections import usable_sections
            _usable_cache[key] = usable_sections(_sections(song_id), vocal_side)
        return _usable_cache[key]

    def _section_pair(top: dict, bed: dict) -> Optional[dict]:
        """Which sections this pair is really about — None until both sides have
        structure, in which case readers fall back to each track's hook."""
        from matcher.sections import best_section_pair
        v_use = _usable(top["song_id"], True)
        i_use = _usable(bed["song_id"], False)
        if not v_use or not i_use:
            return None
        stretch = compute_stretch_factor(top.get("bpm"), bed.get("bpm")) or 1.0
        return best_section_pair(v_use, i_use, stretch, prefiltered=True)

    def _emit(pairs, combo_type, row_scorer, row_version, with_sections=False):
        nonlocal scored
        out = results[combo_type]
        for top, bed, scores in pairs:
            section_pair = _section_pair(top, bed) if with_sections else None
            rows.append(candidate_row(top, bed, scores, combo_type,
                                      row_scorer, row_version, section_pair))
            out.append(_build_row(top, bed, scores, section_pair))
        scored += len(pairs)

    # ── vocal over instrumental ───────────────────────────────────────────────
    # This is the only combo the learned model scores (it was trained on
    # documented vocal-over-bed mashups). The model path widens the gate to the
    # BPM window alone, then replaces total with the model's probability — the
    # four heuristic sub-scores stay for display either way.
    voi = _run(v_block, i_block,
               key_gate=None if use_model else key_min, upper_triangle=False,
               on_block=_report(0, 55, "Scoring vocal over instrumental"))
    if use_model:
        _apply_model_scores(voi, bundle, _sections, lib_stats,
                            _report(55, 10, "Applying the learned model"))
    _emit(voi, "vocal_over_instrumental", active_scorer, model_version,
          with_sections=True)
    voi = None                      # release before the second pass allocates

    # ── instrumental over instrumental ────────────────────────────────────────
    # Always heuristic (BPM + key gate), even when a model is active — the model
    # has no training signal for instrumental↔instrumental transitions. Only
    # lower id over higher, so A/B and B/A are not both scored.
    # Reuse vocal/inst columns: vocal_* = the "top" layer, inst_* = the "bed".
    # No section pair: the top layer here is an instrumental, and the vocal-side
    # filter (vocal_presence ≥ 0.25) would be asking the wrong question of it.
    _emit(_run(i_block, i_block, key_gate=key_min, upper_triangle=True,
               on_block=_report(65, 25, "Scoring instrumental over instrumental")),
          "instrumental_over_instrumental", "heuristic", None)

    if progress:
        progress(90, f"Writing {len(rows)} candidates…")
    bulk_upsert_candidates(rows, db_path=db)

    for key in results:
        results[key].sort(key=lambda x: x["total"], reverse=True)

    evaluated = len(vocals) * len(inst) + len(inst) * (len(inst) - 1) // 2
    log.info(f"  Pairs scored: {scored} of {evaluated} possible  "
             f"|  scorer={active_scorer}"
             + (f" ({model_version})" if model_version else "")
             + f"  filter: bpm_max_diff={bpm_max} key_min_score={key_min}")
    return {**results, "_scorer": active_scorer, "_model_version": model_version}


def _build_row(feat_a: dict, feat_b: dict, scores: dict,
               section_pair: Optional[dict] = None) -> dict:
    return {
        **(section_pair or {}),
        "vocal_song_id":  feat_a["song_id"],
        "vocal_title":    feat_a.get("title", "?"),
        "vocal_artist":   feat_a.get("artist", "?"),
        "vocal_camelot":  feat_a.get("camelot", "?"),
        "vocal_bpm":      feat_a.get("bpm", 0),
        "inst_song_id":   feat_b["song_id"],
        "inst_title":     feat_b.get("title", "?"),
        "inst_artist":    feat_b.get("artist", "?"),
        "inst_camelot":   feat_b.get("camelot", "?"),
        "inst_bpm":       feat_b.get("bpm", 0),
        **scores,
    }


# ── Lookup from DB ────────────────────────────────────────────────────────────

def find_matches(seed_song_id: int, top_k: int = 10,
                 seed_role: str = "vocal",
                 combo_type: str = "vocal_over_instrumental",
                 db_path=None) -> List[dict]:
    """
    Pull pre-scored candidates from the database for a given seed song.

    seed_role:  'vocal' | 'instrumental'
    combo_type: 'vocal_over_instrumental' | 'instrumental_over_instrumental'
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import get_candidates_for_song, DB_PATH

    db = db_path or DB_PATH
    return get_candidates_for_song(
        seed_song_id, role=seed_role,
        combo_type=combo_type, db_path=db
    )[:top_k]


# ── Formatting ────────────────────────────────────────────────────────────────

def format_results(results: List[dict], seed_title: str = "",
                   combo_type: str = "") -> str:
    lines = []
    label = {
        "vocal_over_instrumental":        "Vocals → Instrumental",
        "instrumental_over_instrumental": "Instrumental → Instrumental",
    }.get(combo_type, "Matches")

    if seed_title:
        lines.append(f"\n{'='*60}")
        lines.append(f"  Seed: {seed_title}  [{label}]")
        lines.append(f"{'='*60}")

    if not results:
        lines.append("  No qualifying matches found for this seed.")
        lines.append(f"  (Check BPM_MAX_DIFF and KEY_MIN_SCORE in config.py)")
        return "\n".join(lines)

    for i, r in enumerate(results, 1):
        total    = r.get("score_total") or r.get("total", 0)
        bpm_s    = r.get("score_bpm")   or r.get("bpm_score", 0)
        key_s    = r.get("score_key")   or r.get("key_score", 0)
        energy_s = r.get("score_energy") or r.get("energy_score", 0)
        timbre_s = r.get("score_timbre") or r.get("timbre_score", 0)

        lines.append(
            f"\n  #{i:>2}  TOP:   {r.get('vocal_title','?')} — {r.get('vocal_artist','?')}"
            f"  [{r.get('vocal_bpm','?')} BPM  {r.get('vocal_camelot','?')}]"
            f"\n       BED:   {r.get('inst_title','?')} — {r.get('inst_artist','?')}"
            f"  [{r.get('inst_bpm','?')} BPM  {r.get('inst_camelot','?')}]"
            f"\n       Score: {total:.3f}  |  "
            f"BPM: {bpm_s:.2f}  Key: {key_s:.2f}  "
            f"Energy: {energy_s:.2f}  Timbre: {timbre_s:.2f}"
        )

    return "\n".join(lines)


# ── FL Studio export helpers ──────────────────────────────────────────────────

def export_mashup_report(db_path=None, output_path: str = "mashup_report",
                         top_n: int = 20) -> None:
    """
    Write a ranked mashup report as {output_path}.csv and {output_path}.txt.

    Includes FL Studio workflow data: BPM stretch ratio and semitone shift
    for each vocal+instrumental pair.
    """
    import sys, csv
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import get_conn, DB_PATH

    db = db_path or DB_PATH
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    conn = get_conn(db)
    rows = conn.execute(
        """SELECT mc.*,
                  sv.file_path AS vocal_wav_path,
                  si.file_path AS inst_wav_path
           FROM mashup_candidates mc
           LEFT JOIN stems sv ON sv.song_id = mc.vocal_song_id
                              AND sv.stem_type = 'vocals'
           LEFT JOIN stems si ON si.song_id  = mc.inst_song_id
                              AND si.stem_type = 'instrumental'
           ORDER BY mc.score_total DESC
           LIMIT ?""",
        (top_n,)
    ).fetchall()
    conn.close()

    if not rows:
        log.warning("export_mashup_report: no candidates in DB — run the match stage first.")
        return

    def _path_str(p: Optional[str]) -> str:
        if not p:
            return "FILE_MISSING"
        return str(p) if Path(p).exists() else f"FILE_MISSING:{p}"

    CSV_FIELDS = [
        "rank", "combo_type",
        "vocal_title", "vocal_artist", "inst_title", "inst_artist",
        "vocal_bpm", "inst_bpm", "bpm_stretch_ratio",
        "vocal_key", "vocal_mode", "inst_key", "inst_mode",
        "vocal_camelot", "inst_camelot", "semitone_shift",
        "score_total", "score_bpm", "score_key", "score_energy", "score_timbre",
        "vocal_wav_path", "inst_wav_path",
    ]

    csv_path = out.with_suffix(".csv")
    txt_path = out.with_suffix(".txt")
    SEP = "=" * 80

    with open(csv_path, "w", newline="", encoding="utf-8") as cf, \
         open(txt_path, "w", encoding="utf-8") as tf:

        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()

        tf.write(f"Mashup Report — Top {len(rows)} Pairs\n")
        tf.write(f"Generated for FL Studio session setup\n\n")

        for rank, row in enumerate(rows, 1):
            r = dict(row)
            v_bpm = r.get("vocal_bpm") or 0.0
            i_bpm = r.get("inst_bpm") or 0.0
            stretch = compute_stretch_factor(v_bpm, i_bpm)
            shift   = compute_semitone_shift(
                r.get("vocal_camelot") or "", r.get("inst_camelot") or ""
            )
            v_path = _path_str(r.get("vocal_wav_path"))
            i_path = _path_str(r.get("inst_wav_path"))

            writer.writerow({
                "rank":             rank,
                "combo_type":       r.get("combo_type", ""),
                "vocal_title":      r.get("vocal_title", ""),
                "vocal_artist":     r.get("vocal_artist", ""),
                "inst_title":       r.get("inst_title", ""),
                "inst_artist":      r.get("inst_artist", ""),
                "vocal_bpm":        v_bpm,
                "inst_bpm":         i_bpm,
                "bpm_stretch_ratio": stretch,
                "vocal_key":        r.get("vocal_key", ""),
                "vocal_mode":       r.get("vocal_mode", ""),
                "inst_key":         r.get("inst_key", ""),
                "inst_mode":        r.get("inst_mode", ""),
                "vocal_camelot":    r.get("vocal_camelot", ""),
                "inst_camelot":     r.get("inst_camelot", ""),
                "semitone_shift":   shift if shift is not None else "",
                "score_total":      r.get("score_total", ""),
                "score_bpm":        r.get("score_bpm", ""),
                "score_key":        r.get("score_key", ""),
                "score_energy":     r.get("score_energy", ""),
                "score_timbre":     r.get("score_timbre", ""),
                "vocal_wav_path":   v_path,
                "inst_wav_path":    i_path,
            })

            stretch_str = f"{stretch:.4f}x" if stretch else "?"
            shift_str   = (f"{shift:+d}" if shift is not None else "?") + " semitones"
            tf.write(f"{SEP}\n")
            tf.write(
                f"#{rank:02d}  [{r.get('combo_type', '')}]"
                f"  Score: {r.get('score_total', 0):.3f}\n"
            )
            tf.write(
                f"  VOCAL:  {r.get('vocal_title', '?')} — {r.get('vocal_artist', '?')}\n"
                f"          {v_bpm:.2f} BPM  |  Key: {r.get('vocal_key','?')}"
                f" {r.get('vocal_mode','?')}  |  Camelot: {r.get('vocal_camelot','?')}\n"
            )
            tf.write(
                f"  INST:   {r.get('inst_title', '?')} — {r.get('inst_artist', '?')}\n"
                f"          {i_bpm:.2f} BPM  |  Key: {r.get('inst_key','?')}"
                f" {r.get('inst_mode','?')}  |  Camelot: {r.get('inst_camelot','?')}\n"
            )
            tf.write(
                f"  BPM:    Stretch instrumental by {stretch_str}"
                f"  (set inst tempo to {v_bpm:.2f} BPM)\n"
                f"  KEY:    Pitch instrumental {shift_str}"
                f"  (in channel rack / Newtone / Pitcher)\n"
            )
            tf.write(
                f"  SCORE:  {r.get('score_total',0):.3f}"
                f"  (BPM:{r.get('score_bpm',0):.2f}"
                f"  Key:{r.get('score_key',0):.2f}"
                f"  Energy:{r.get('score_energy',0):.2f}"
                f"  Timbre:{r.get('score_timbre',0):.2f})\n"
            )
            tf.write(f"  FILES:\n")
            tf.write(f"    Vocals: {v_path}\n")
            tf.write(f"    Inst:   {i_path}\n")

        tf.write(f"{SEP}\n")

    log.info(f"  Mashup report written: {csv_path}  |  {txt_path}")


def prep_fl_session(db_path=None, output_dir: str = "fl_session",
                    top_n: int = 10) -> None:
    """
    Create an FL Studio session folder with one sub-folder per top mashup pair.

    Each sub-folder contains:
      vocals.wav         — symlink (or copy) to the vocal stem WAV
      instrumental.wav   — symlink (or copy) to the instrumental stem WAV
      session_info.txt   — BPM stretch ratio, semitone shift, scores, source paths
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import get_conn, DB_PATH

    db  = db_path or DB_PATH
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conn = get_conn(db)
    rows = conn.execute(
        """SELECT mc.*,
                  sv.file_path AS vocal_wav_path,
                  si.file_path AS inst_wav_path
           FROM mashup_candidates mc
           LEFT JOIN stems sv ON sv.song_id = mc.vocal_song_id
                              AND sv.stem_type = 'vocals'
           LEFT JOIN stems si ON si.song_id  = mc.inst_song_id
                              AND si.stem_type = 'instrumental'
           WHERE mc.combo_type = 'vocal_over_instrumental'
           ORDER BY mc.score_total DESC
           LIMIT ?""",
        (top_n,)
    ).fetchall()
    conn.close()

    if not rows:
        log.warning("prep_fl_session: no vocal_over_instrumental candidates found.")
        (out / "README.txt").write_text(
            "No mashup candidates found.\n"
            "Run the match stage first: python test_flow.py --stages match\n"
        )
        return

    seen_names: set = set()
    created = 0

    for rank, row in enumerate(rows, 1):
        r = dict(row)
        safe_v = _sanitize_folder_name(r.get("vocal_title") or "vocal")
        safe_i = _sanitize_folder_name(r.get("inst_title") or "inst")
        folder_name = f"{rank:02d}_{safe_v}_over_{safe_i}"

        # Resolve name collision from truncation
        suffix, attempt = "", 1
        while (folder_name + suffix) in seen_names:
            attempt += 1
            suffix = f"_{chr(96 + attempt)}"   # _b, _c, …
        folder_name += suffix
        seen_names.add(folder_name)

        folder = out / folder_name
        folder.mkdir(exist_ok=True)

        v_bpm   = r.get("vocal_bpm") or 0.0
        i_bpm   = r.get("inst_bpm") or 0.0
        stretch = compute_stretch_factor(v_bpm, i_bpm)
        shift   = compute_semitone_shift(
            r.get("vocal_camelot") or "", r.get("inst_camelot") or ""
        )
        stretch_str = f"{stretch:.4f}x" if stretch else "?"
        shift_str   = (f"{shift:+d}" if shift is not None else "?")

        v_src = Path(r["vocal_wav_path"]) if r.get("vocal_wav_path") else None
        i_src = Path(r["inst_wav_path"])  if r.get("inst_wav_path")  else None

        missing_warnings = []
        if v_src and v_src.exists():
            _link_or_copy(v_src, folder / "vocals.wav")
        else:
            missing_warnings.append(f"vocals.wav source missing: {v_src}")

        if i_src and i_src.exists():
            _link_or_copy(i_src, folder / "instrumental.wav")
        else:
            missing_warnings.append(f"instrumental.wav source missing: {i_src}")

        info_lines = [
            "FL Studio Session Info",
            "======================",
            f"Rank:   #{rank:02d} / {r.get('combo_type', '')}",
            f"Score:  {r.get('score_total', 0):.3f}"
            f"  (BPM:{r.get('score_bpm',0):.2f}"
            f"  Key:{r.get('score_key',0):.2f}"
            f"  Energy:{r.get('score_energy',0):.2f}"
            f"  Timbre:{r.get('score_timbre',0):.2f})",
            "",
            f"TOP (vocals)   {r.get('vocal_title','?')} — {r.get('vocal_artist','?')}",
            f"  BPM: {v_bpm:.2f}  |  Key: {r.get('vocal_key','?')}"
            f" {r.get('vocal_mode','?')}  |  Camelot: {r.get('vocal_camelot','?')}",
            "",
            f"BED (instrumental)  {r.get('inst_title','?')} — {r.get('inst_artist','?')}",
            f"  BPM: {i_bpm:.2f}  |  Key: {r.get('inst_key','?')}"
            f" {r.get('inst_mode','?')}  |  Camelot: {r.get('inst_camelot','?')}",
            "",
            "FL Studio Parameters",
            f"  BPM Stretch Ratio:  {stretch_str}",
            f"    → Stretch instrumental to {v_bpm:.2f} BPM to match vocals",
            f"  Semitone Shift:     {shift_str} semitones",
            f"    → Pitch instrumental {shift_str} semitones"
            f" ({'up' if (shift or 0) >= 0 else 'down'}) in channel settings",
            "",
            "Files (absolute paths)",
            f"  vocals.wav        → {v_src or 'MISSING'}",
            f"  instrumental.wav  → {i_src or 'MISSING'}",
        ]
        if missing_warnings:
            info_lines += ["", "WARNINGS:"] + [f"  {w}" for w in missing_warnings]

        (folder / "session_info.txt").write_text("\n".join(info_lines) + "\n",
                                                  encoding="utf-8")
        created += 1

    log.info(f"  FL session folders created: {created}  →  {out.resolve()}")