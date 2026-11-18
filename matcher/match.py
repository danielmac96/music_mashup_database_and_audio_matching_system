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
import heapq
import itertools
import threading
import math
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Iterable

# database.models never imports matcher, so this is safe at module scope — and
# it has to be, so the library-stats cache can read features without threading a
# handle through every scoring call.
from database.models import DB_PATH, FEATURES_SCORING_COLUMNS, get_all_features
from analysis.quality import collision_block, collision_score
from matcher.effort import effort_block

log = logging.getLogger(__name__)


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


def _hours_to_semitones(hours: int) -> int:
    """Moving a key `hours` steps round the Camelot wheel (one step = a perfect
    fifth = 7 semitones), folded into [-6, +6]."""
    diff = (hours * 7) % 12
    return diff if diff <= 6 else diff - 12


def compute_semitone_shift(vocal_camelot: str, inst_camelot: str) -> Optional[int]:
    """
    Smallest semitone shift for the INSTRUMENTAL that lands it within one step
    of the vocal on the Camelot wheel — the same key, the relative major/minor,
    or an adjacent hour, all of which mix as they are. Positive = shift up,
    negative = down, range [-6, +6]. None if either Camelot code is unknown.

    So same, relative and adjacent keys need NO transposition (0): a DJ mixes
    8A into 9A untouched, and dragging the bed a fourth to land exactly on the
    vocal's key was a destructive move for no harmonic gain. Further apart, the
    bed moves to whichever of the three wheel positions round the vocal costs
    the fewest semitones (an exact landing wins a tie).

    The letter is deliberately ignored: 8A and 8B (A minor / C major) are the
    same pitch collection, so a relative major/minor pair needs no
    transposition at all.

    This is the Camelot ESTIMATE. A pair whose two sections had chroma to
    compare plays its measured shift instead (matcher/recipe.bed_shift).
    """
    v = _parse_camelot(vocal_camelot or "")
    i = _parse_camelot(inst_camelot or "")
    if v is None or i is None:
        return None
    steps = (v[0] - i[0]) % 12          # hours the bed sits from the vocal
    if steps > 6:
        steps -= 12
    best = None
    for d in (steps, steps - 1, steps + 1):
        semis = _hours_to_semitones(d)
        key = (abs(semis), d != steps, semis)
        if best is None or key < best[0]:
            best = (key, semis)
    return best[1]


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

    __slots__ = ("mfcc_mean", "mfcc_std", "loudness", "n", "conf")

    def __init__(self, mfcc_mean=None, mfcc_std=None, loudness=None, n=0,
                 conf=None):
        self.mfcc_mean = mfcc_mean
        self.mfcc_std = mfcc_std
        self.loudness = loudness or {}
        self.n = n
        # Sorted library values for each confidence kind, for conf_pct below.
        self.conf = conf or {}

    @property
    def usable(self) -> bool:
        """False for a library too small to normalise against, in which case
        callers fall back to the un-normalised comparison rather than dividing
        by a meaningless std."""
        return self.n >= 4 and self.mfcc_mean is not None

    def conf_pct(self, kind: str, value):
        """Where `value` sits in the library's distribution of that confidence,
        0 (worst in the library) to 1 (best). Accepts a scalar or an array.

        The effort penalty asks "will this pair fight me in the DAW", and that
        is a question about THIS library: a beat-grid confidence of 0.6 is a
        problem in a library of programmed house and unremarkable in a library
        of live recordings. Ranking on the absolute number also makes every
        threshold hostage to the estimator's calibration — which is exactly how
        the old beats-per-frame `bpm_confidence` put a constant 0.24 floor under
        every effort score without anyone noticing.

        Falls through to the raw value when the library is too small to have a
        distribution (fewer than MIN_CONF_SAMPLES), so a handful of tracks still
        rank sensibly rather than all landing on the same percentile.
        """
        ref = self.conf.get(kind)
        arr = np.asarray(value, dtype=np.float64)
        if ref is None or len(ref) < MIN_CONF_SAMPLES:
            return np.clip(np.nan_to_num(arr, nan=1.0), 0.0, 1.0)
        # Fraction of the library at or below this value. `right` so the best
        # track in the library scores 1.0 rather than 1 - 1/n.
        pct = np.searchsorted(ref, arr, side="right") / float(len(ref))
        return np.clip(pct, 0.0, 1.0)


_STATS_CACHE: Dict[str, "LibraryStats"] = {}

# Stems that get matched. 'full' is included so its stats exist for fallbacks.
_STEM_KINDS = ("vocals", "instrumental", "full")

# Below this many measured tracks there is no distribution to rank against, and
# conf_pct hands back the raw value instead.
MIN_CONF_SAMPLES = 8

# The confidence columns conf_pct normalises, and the feature key each reads.
_CONF_KINDS = {"bpm": "bpm_confidence", "key": "key_confidence"}


def compute_library_stats(db_path=None) -> LibraryStats:
    rows = []
    for stem in _STEM_KINDS:
        rows.extend(get_all_features(stem_type=stem,
                                     columns=FEATURES_SCORING_COLUMNS,
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

    # Confidence distributions, sorted so conf_pct is a searchsorted away.
    # Measured over every stem kind together: the question "is this grid worse
    # than most of what I own" does not change depending on which stem it is.
    conf: Dict[str, np.ndarray] = {}
    for kind, key in _CONF_KINDS.items():
        vals = []
        for r in rows:
            try:
                v = float(r.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append(v)
        if len(vals) >= MIN_CONF_SAMPLES:
            conf[kind] = np.sort(np.asarray(vals, dtype=np.float64))
    return LibraryStats(mfcc_mean, mfcc_std, loudness, len(mats), conf)


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
        "collision_score": collision_score(feat_a.get("band_energy"),
                                           feat_b.get("band_energy")),
    }


def composite_score(feat_a: dict, feat_b: dict,
                    weights: Optional[Dict] = None,
                    stats: Optional[LibraryStats] = None,
                    effort_weight: Optional[float] = None,
                    combo_type: Optional[str] = None) -> dict:
    """The four sub-scores, the effort penalty, and the discounted total.

    `total` is the weighted fit discounted by EFFORT_WEIGHT × effort, so a pair
    that is free to build can outrank one that fits marginally better but needs
    a destructive stretch or transpose. The effort components come back
    alongside so callers can show which cost dominates.

    Pass effort_weight=0.0 to rank on similarity alone.
    """
    try:
        from config import current_float, current_match_weights
        weights = weights or current_match_weights(combo_type)
        if effort_weight is None:
            effort_weight = current_float("effort_weight")
    except ImportError:
        weights = weights or {"bpm_score": 0.22, "key_score": 0.26,
                              "energy_score": 0.17, "timbre_score": 0.20,
                              "collision_score": 0.15}
        if effort_weight is None:
            effort_weight = 0.0

    scores = sub_scores(feat_a, feat_b, stats)
    fit = sum(scores[k] * weights.get(k, 0) for k in scores)

    from matcher.effort import effort_penalty
    stretch = compute_stretch_factor(feat_a.get("bpm"), feat_b.get("bpm"))
    shift = compute_semitone_shift(feat_a.get("camelot") or "",
                                   feat_b.get("camelot") or "")
    conf_stats = stats if stats is not None else get_library_stats()
    effort, parts = effort_penalty(
        feat_a, feat_b, stretch, shift,
        conf_norm=lambda kind, v: float(conf_stats.conf_pct(kind, v)))

    scores["total"] = round(fit * (1.0 - effort_weight * effort), 4)
    scores["score_effort"] = round(effort, 4)
    scores["effort_stretch"] = round(parts["stretch_cost"], 4)
    scores["effort_pitch"] = round(parts["pitch_cost"], 4)
    scores["effort_tempo_fold"] = round(parts["tempo_fold_cost"], 4)
    scores["effort_grid"] = round(parts["grid_cost"], 4)
    scores["effort_key_certainty"] = round(parts["key_certainty_cost"], 4)
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
                 "mfcc_v", "mfcc_norm", "mfcc_ok", "variant",
                 "bpm_conf", "key_conf", "bands", "quality")


def _conf(value) -> float:
    """A 0-1 confidence, defaulting to 1.0 (certain) when absent or unusable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return float(min(1.0, max(0.0, v)))


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

    # The recommended transpose over the same code pairs, derived from the same
    # scalar function so the effort penalty charges exactly the shift the plan
    # will tell the user to dial in. None (unknown key) reads as "not known"
    # rather than as a free 0.
    shifts = [[compute_semitone_shift(a, b) for b in codes] for a in codes]
    shift_table = np.array(
        [[0 if s is None else s for s in row] for row in shifts],
        dtype=np.float64).reshape(len(codes), len(codes))
    known_table = np.array(
        [[s is not None for s in row] for row in shifts],
        dtype=bool).reshape(len(codes), len(codes))
    return index, table, shift_table, known_table


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

    # Effort inputs (Phase C). A missing confidence means the track was analysed
    # before the column existed; treat it as certain rather than retroactively
    # penalising every old track — matching matcher.effort's scalar default.
    #
    # Both are then mapped onto their place in the library's own distribution
    # (LibraryStats.conf_pct). The raw numbers live on incomparable, estimator-
    # defined scales, so using them directly turns grid_cost and
    # key_certainty_cost into near-constants — see conf_pct's docstring.
    b.bpm_conf = stats.conf_pct("bpm", np.array(
        [_conf(f.get("bpm_confidence")) for f in feats], dtype=np.float64))
    b.key_conf = stats.conf_pct("key", np.array(
        [_conf(f.get("key_confidence")) for f in feats], dtype=np.float64))

    # Phase D: band occupancy (all-zero row = unmeasured, which collision_block
    # reads as the neutral 0.5) and this stem's separation quality.
    from analysis.quality import N_BANDS
    bands = np.zeros((n, N_BANDS), dtype=np.float64)
    for idx, f in enumerate(feats):
        be = f.get("band_energy") or []
        if len(be) == N_BANDS:
            bands[idx] = np.asarray(be, dtype=np.float64)
    b.bands = bands
    # NULL quality means "not measured", which must not be read as "bad" — a
    # library analysed before Phase D would otherwise vanish from the list.
    b.quality = np.array(
        [1.0 if f.get("stem_quality") is None else float(f["stem_quality"])
         for f in feats], dtype=np.float64)

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


def _effective_bpm_block(a_bpm: np.ndarray, b_bpm: np.ndarray):
    """(effective bed tempo, folded?) over every pair.

    Mirrors effective_bpm's `min((b, b/2, b*2), key=…)`, which resolves ties in
    that order: the bed as written wins unless halving is strictly closer, and
    doubling only wins when it beats both."""
    A = a_bpm[:, None]
    B = b_bpm[None, :]
    d0, dh, dd = np.abs(A - B), np.abs(A - B / 2.0), np.abs(A - B * 2.0)
    eff = np.where(dh < d0, B / 2.0, np.broadcast_to(B, d0.shape).astype(np.float64))
    eff = np.where(dd < np.minimum(d0, dh), B * 2.0, eff)
    usable = (A > 0) & (B > 0)
    eff = np.where(usable, eff, 0.0)
    folded = usable & (np.abs(eff - B) > 1e-9)
    return eff, folded


def _stretch_block(a_bpm: np.ndarray, eff: np.ndarray) -> np.ndarray:
    """compute_stretch_factor over every pair, including its rounding — the
    persisted effort must charge the same stretch the plan prints. 0 marks
    'unknown', which the effort block charges at full cost."""
    A = a_bpm[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where((A > 0) & (eff > 0), A / np.where(eff > 0, eff, 1.0), 0.0)
    return np.round(s, 4)


def _iter_scored_pairs(top: _StemBlock, bed: _StemBlock, *,
                       stats: LibraryStats, key_table: np.ndarray,
                       shift_table: np.ndarray, shift_known: np.ndarray,
                       bpm_max: float, key_min: Optional[float],
                       upper_triangle: bool, weights: Dict,
                       effort_weight: float = 0.0,
                       quality_min: float = 0.0,
                       on_block=None, pre_select=None, post_select=None,
                       terms_only: bool = False):
    """Yield (top_idx, bed_idx, scores) for every pair that passes the gate.

    ``pre_select(rows, keep)`` and ``post_select(rows, keep, total)`` narrow a
    block's gated pairs further — before its sub-scores are computed, and once
    its totals are known. They only ever skip pairs the caller has proved
    cannot be kept (readme §5.7, "Scale"); the gate itself is the lines below.
    ``terms_only`` yields one ``(rows, keep, terms)`` per block instead — the
    matrices themselves, for the caller that bounds a block rather than
    walking it.

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
    w_collision = weights.get("collision_score", 0)

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

        # An unusable top stem is not a candidate however well it matches: a
        # bleeding, smeared acapella near the top of the list is what stops the
        # list being trusted. Unmeasured quality is 1.0, so a pre-Phase-D
        # library is unaffected.
        if quality_min > 0:
            keep &= (top.quality[rows][:, None] >= quality_min)
        if pre_select is not None:
            keep = pre_select(rows, keep)

        if keep.any():
            bpm_s = _bpm_score_block(diff)
            energy_s = _energy_block(top, bed, rows)
            timbre_s = _timbre_block(top, bed, rows, stats)
            # Do the two sides stay out of each other's way in the spectrum?
            # Four sub-scores of similarity cannot answer that, and a mid-heavy
            # vocal over a mid-dense bed is inaudible however well they match.
            collision_s = collision_block(top.bands[rows], bed.bands)
            fit = (bpm_s * w_bpm + key_s * w_key
                   + energy_s * w_energy + timbre_s * w_timbre
                   + collision_s * w_collision)

            # ── Effort (Phase C) ──────────────────────────────────────────────
            # Similarity says whether the pair fits; this says what building it
            # costs. score_total discounts the fit by it, so a free-to-build 78%
            # can outrank an 84% that needs a 12% stretch and +5 semitones.
            eff_bpm, folded = _effective_bpm_block(top.bpm[rows], bed.bpm)
            stretch = _stretch_block(top.bpm[rows], eff_bpm)
            semis = shift_table[np.ix_(top.code[rows], bed.code)]
            known = shift_known[np.ix_(top.code[rows], bed.code)]
            effort, parts = effort_block(
                top.bpm[rows][:, None], bed.bpm[None, :], stretch, semis, known,
                top.bpm_conf[rows][:, None], bed.bpm_conf[None, :],
                top.key_conf[rows][:, None], bed.key_conf[None, :], folded)
            total = fit * (1.0 - effort_weight * effort)
            if terms_only:
                yield rows, keep, {"fit": fit, "key": key_s, "total": total,
                                   "effort": effort, "parts": parts}
                if on_block is not None:
                    on_block(rows.stop, n_top)
                continue
            if post_select is not None:
                keep = post_select(rows, keep, total)

            # np.nonzero returns row-major index pairs, preserving the original
            # iteration order. round() stays Python's so the persisted total is
            # bit-identical to composite_score's.
            for i, j in zip(*np.nonzero(keep)):
                yield start + int(i), int(j), {
                    "bpm_score": float(bpm_s[i, j]),
                    "key_score": float(key_s[i, j]),
                    "energy_score": float(energy_s[i, j]),
                    "timbre_score": float(timbre_s[i, j]),
                    "collision_score": float(collision_s[i, j]),
                    "total": round(float(total[i, j]), 4),
                    "score_effort": round(float(effort[i, j]), 4),
                    "effort_stretch": round(float(parts["stretch_cost"][i, j]), 4),
                    "effort_pitch": round(float(parts["pitch_cost"][i, j]), 4),
                    "effort_tempo_fold": round(float(parts["tempo_fold_cost"][i, j]), 4),
                    "effort_grid": round(float(parts["grid_cost"][i, j]), 4),
                    "effort_key_certainty": round(
                        float(parts["key_certainty_cost"][i, j]), 4),
                }
        if on_block is not None:
            on_block(rows.stop, n_top)


class _TopRows:
    """The best ``cap`` rows of a run, whatever order they are priced in.

    A run used to push every row through one bounded heap in emission order
    and let the heap decide; which rows survive a tie at the cut then depends
    on that order. This keeps every row at or above the running cut (so
    nothing that could still be kept is lost) and applies the same rule once,
    at the end, from each row's emission sequence — so a run may price its
    most promising pairs first, or skip ones that cannot reach the cut, and
    still store exactly the rows the in-order heap would have.
    tests/test_score_scale.py pins it to that heap."""

    def __init__(self, cap: int):
        self.cap = max(1, int(cap))
        self.heap: list = []
        self._count: Dict[float, int] = {}

    def floor(self) -> Optional[float]:
        """The cut: a total below it can no longer be kept. None until ``cap``
        rows are held."""
        return self.heap[0][0] if len(self.heap) >= self.cap else None

    def wants(self, total: float) -> bool:
        return len(self.heap) < self.cap or total >= self.heap[0][0]

    def push(self, total: float, seq: tuple, row, built) -> None:
        heapq.heappush(self.heap, (total, seq, row, built))
        self._count[total] = self._count.get(total, 0) + 1
        while len(self.heap) > self.cap:
            low = self.heap[0][0]
            n_low = self._count[low]
            if len(self.heap) - n_low < self.cap:
                break                   # the lowest total is still inside the cut
            for _ in range(n_low):
                heapq.heappop(self.heap)
            del self._count[low]

    def ranked(self) -> list:
        """The kept rows, best first, ties in emission order.

        The in-order heap admits a row when it is not full or the row beats
        its lowest total, and evicts the earliest row at that lowest total.
        With T the final cut: every row at or above T is admitted until the
        heap holds ``cap`` of them; after that a row at T is refused and each
        row above T evicts the earliest T-row still held."""
        items = sorted(self.heap, key=lambda t: t[1])
        if len(items) > self.cap:
            cut = self.heap[0][0]
            first, later = items[:self.cap], items[self.cap:]
            evict = sum(1 for t in later if t[0] > cut)
            kept = []
            for t in first:
                if evict and t[0] == cut:
                    evict -= 1
                    continue
                kept.append(t)
            kept.extend(t for t in later if t[0] > cut)
            items = kept
        return sorted(items, key=lambda t: (-t[0], t[1]))


class _Reserve:
    """Each track's best partners, whatever the row cap keeps (readme §5.7).

    Per vocal song and per bed song: the best row of each of its ``k`` best
    partner songs, best total first, the earlier-emitted pair on a tie. A
    pair is offered once, with its best row."""

    def __init__(self, k: int):
        self.k = max(0, int(k))
        self.heaps: Dict[tuple, list] = {}

    def _key(self, total: float, seq: tuple) -> tuple:
        return (total, tuple(-x for x in seq))      # earlier seq = larger key

    def floor(self, side: str, song_id: int) -> Optional[float]:
        """The total a new partner must reach to be kept for this track; None
        while it has fewer than ``k``."""
        heap = self.heaps.get((side, song_id))
        return heap[0][0][0] if heap and len(heap) >= self.k else None

    def wants(self, side: str, song_id: int, total: float, seq: tuple) -> bool:
        if not self.k:
            return False
        heap = self.heaps.get((side, song_id))
        return heap is None or len(heap) < self.k or self._key(total, seq) > heap[0][0]

    def push(self, side: str, song_id: int, total: float, seq: tuple, item) -> None:
        heap = self.heaps.setdefault((side, song_id), [])
        entry = (self._key(total, seq), item)
        if len(heap) < self.k:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def items(self) -> list:
        """Every reserved row once, as _TopRows items."""
        seen, out = set(), []
        for heap in self.heaps.values():
            for _key, item in heap:
                if item[1] not in seen:
                    seen.add(item[1])
                    out.append(item)
        return out


# Added to every ceiling: totals are rounded to four places on the way to the
# table (5e-5 each for the section fit, the effort and the total) and the
# ceilings are held as float32.
_BOUND_SLACK = 3e-4
# How many song pairs the first, most promising tier of a pruned run prices,
# as a share of the row cap; each later tier is twice the one before.
_FIRST_TIER_SHARE = 0.5
# Off: every gated pair is priced, as before pruning existed. The switch the
# tests use to run both ways on one library; nothing else turns it off.
PRUNE_RUNS = True
# How many of its most promising unpriced partners (by ceiling), per reserved
# partner, a track has searched for its reserve after the cap is settled.
RESERVE_SEARCH = 3
# ...and how many in all a track still short of its partners may have searched.
RESERVE_SEARCH_MAX = 12
# Gated song pairs searched, and not stored, for the library's slot floor.
SLOT_SAMPLE_PAIRS = 3000
_BOUND_BINS = 4096
_BOUND_TOP = 1.5


def _section_ceilings(feats: List[dict], usable, vocal_side: bool,
                      section_weights: Dict[str, float], cache: dict):
    """Per song on one side: the most its sections can add to a section fit,
    whether it has any usable section at all, and — a bed — the most the
    vacated slot can be for any vocal over it."""
    from matcher.section_score import slot_ceiling
    from matcher.sections import _statics
    w_label = section_weights["label"]
    w_voice = section_weights["voice"]
    best = np.zeros(len(feats), dtype=np.float64)
    has = np.zeros(len(feats), dtype=bool)
    slot = np.ones(len(feats), dtype=np.float64)
    for k, f in enumerate(feats):
        secs = usable(f["song_id"], vocal_side)
        if not secs:
            continue
        has[k] = True
        if not vocal_side:
            slot[k] = max(slot_ceiling(x) for x in secs)
        if vocal_side:
            best[k] = max(w_label * st[1] + w_voice * st[2]
                          for st in (_statics(x, True, cache) for x in secs))
        else:
            best[k] = max(w_label * _statics(x, False, cache)[1] for x in secs)
    return best, has, slot


def _chroma_rows(feats: List[dict], usable, vocal_side: bool, cache: dict):
    """The unit chroma of every usable section on one side, stacked, with each
    song's slice — and whether any of its usable sections has none (that
    pairing then keeps the Camelot key score, _apply_measured_harmony)."""
    from matcher.harmony import _side_chroma, _vec_of
    key = "chroma_vocal" if vocal_side else "chroma_bed"
    vecs, spans = [], []
    unknown = np.zeros(len(feats), dtype=bool)
    for k, f in enumerate(feats):
        start = len(vecs)
        for sec in usable(f["song_id"], vocal_side):
            v = _vec_of(_side_chroma(sec, key), cache)
            if v is None:
                unknown[k] = True
            else:
                vecs.append(v)
        spans.append((start, len(vecs)))
    mat = (np.asarray(vecs, dtype=np.float32) if vecs
           else np.zeros((0, 12), dtype=np.float32))
    return mat, spans, unknown


def _harmony_ceiling(v_mat, v_spans, b_mat, b_spans, rows: slice) -> np.ndarray:
    """For the top songs in ``rows`` against every bed song: the best measured
    harmony any of their section pairs could reach — the correlation at its
    best rotation, before the transpose is priced or a bass clash taken off,
    so never below matcher.harmony.section_harmony's fit. -inf where either
    song has no section with chroma."""
    from matcher.harmony import _ROTATIONS
    n_bed = len(b_spans)
    out = np.full((rows.stop - rows.start, n_bed), -np.inf, dtype=np.float32)
    lo, hi = v_spans[rows.start][0], v_spans[rows.stop - 1][1]
    if hi <= lo or not len(b_mat):
        return out
    V = v_mat[lo:hi]
    dots = None
    for k in range(_ROTATIONS.shape[0]):
        d = V @ b_mat[:, _ROTATIONS[k]].T
        dots = d if dots is None else np.maximum(dots, d, out=dots)
    # Section pairs -> song pairs: the best over each song's slice, on both
    # axes. reduceat needs non-empty groups, so songs without chroma sit out.
    b_ok = [j for j, (a, z) in enumerate(b_spans) if z > a]
    if not b_ok:
        return out
    by_bed = np.maximum.reduceat(dots, [b_spans[j][0] for j in b_ok], axis=1)
    t_ok = [i for i in range(rows.start, rows.stop) if v_spans[i][1] > v_spans[i][0]]
    if not t_ok:
        return out
    by_pair = np.maximum.reduceat(by_bed, [v_spans[i][0] - lo for i in t_ok], axis=0)
    fit = np.clip((by_pair + 1.0) / 2.0, 0.0, 1.0) + 1e-5       # float32 dots
    out[np.ix_([i - rows.start for i in t_ok], b_ok)] = fit
    return out


# ── Score all qualifying pairs ────────────────────────────────────────────────

# How far allin1's BPM may sit from Essentia's before the two are treated as
# disagreeing about the tempo octave rather than refining one estimate. Same
# bound analyze.fitted_bpm uses for the same reason, and it is what keeps a
# manually corrected half/double-time read (update_features_manual) from being
# overwritten by allin1's own read of the original.
ALLIN1_BPM_MAX_DRIFT = 0.06


def _full_by_song(full_rows, db_path=None) -> Dict[int, dict]:
    """{song_id: the song's full-mix feature row}, with the tempo refined to
    allin1's where it has one. The ONE builder of the map _with_full_bpm reads,
    so the library run, build_dataset, score_pair and Segments' live ranking
    cannot disagree about a track's tempo (readme §7).

    Why the refinement belongs here. A pair is placed and judged on SECTION bar
    lines, which for an allin1-cut track are allin1's downbeats
    (matcher.phrase_align), while the ratio those grids are conformed by comes
    from the song-level BPM (compute_stretch_factor). Reading the grid from one
    analyser and the ratio from another puts the two estimates' disagreement
    straight into the drift the 25 ms timing gate measures. allin1 tracked the
    downbeats, so its BPM is the one that matches the bar lines.

    Only the BPM moves. `bpm_confidence` stays Essentia's: effort_grid ranks it
    across the library (matcher.effort) and allin1's conf_mean is a downbeat
    activation on an unrelated scale, so mixing the two would corrupt that
    ranking for every track. The Essentia beat grid in `feature_arrays` is left
    alone too — nothing here rewrites a stored row.
    """
    out = {f["song_id"]: f for f in full_rows}
    if not out:
        return out
    try:
        from database.alignment import allin1_bpm_by_song
        refined = allin1_bpm_by_song(out.keys(), db_path=db_path)
    except Exception:  # noqa: BLE001 — a refinement, never a reason not to score
        log.warning("could not read allin1 tempos; scoring on the Essentia grid",
                    exc_info=True)
        return out
    for song_id, a1_bpm in refined.items():
        row = out.get(song_id)
        ess = _bpm(row.get("bpm")) if row else 0.0
        if not ess:
            continue
        if abs(a1_bpm - ess) / ess > ALLIN1_BPM_MAX_DRIFT:
            # The two read different tempo octaves (or the user fixed one by
            # hand). Keep the stored value rather than silently re-folding it.
            continue
        row = dict(row)
        row["bpm"] = round(float(a1_bpm), 2)
        out[song_id] = row
    return out


def _with_full_bpm(feat: dict, full_by_song: Dict[int, dict]) -> dict:
    """Return a copy of `feat` with tempo AND key swapped to the song's full-mix
    values when available, leaving timbre, loudness and band occupancy
    stem-derived.

    Tempo was the original reason: vocal-stem and even instrumental-stem beat
    tracking picks up octave and onset errors introduced by separation.

    Key belongs here for a stronger version of the same argument (P0.3). The
    stored key comes from correlating a whole-track mean chroma against
    Krumhansl profiles — on an isolated acapella that is close to noise, because
    a sung melody rarely outlines a scale and a mean chroma over one voice is
    mostly whichever notes were held longest. That estimate then drove the
    `key_min_score` PRE-FILTER, which runs before Phase E ever measures the real
    harmony, so pairs were being discarded on the least reliable number in the
    database and never got a second opinion. The full mix has the chords in it.

    Timbre, loudness and band energy stay stem-derived on purpose: those are
    what the listener actually hears layered, and they are measured directly
    rather than estimated.
    """
    full_feat = full_by_song.get(feat.get("song_id"))
    if not full_feat:
        return feat
    out = dict(feat)
    if full_feat.get("bpm"):
        out["stem_bpm"] = feat.get("bpm")
        out["bpm"] = full_feat["bpm"]
        out["bpm_confidence"] = full_feat.get("bpm_confidence")
    if full_feat.get("camelot"):
        out["stem_camelot"] = feat.get("camelot")
        out["camelot"] = full_feat["camelot"]
        out["key"] = full_feat.get("key")
        out["mode"] = full_feat.get("mode")
        out["key_confidence"] = full_feat.get("key_confidence")
    return out


def _record_harmony_threshold(db) -> None:
    """Store the confidence below which a measured transpose is a close call in
    this library: the HARMONY_UNSURE_QUANTILE of the stored vocal-over-bed
    confidences (matcher.recipe.harmony_unsure_below reads it). Measure, then
    rank against the library — the confidence's absolute scale says little."""
    from database.models import get_conn, set_pref
    from matcher.recipe import HARMONY_UNSURE_QUANTILE
    conn = get_conn(db)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM mashup_candidates WHERE harmonic_confidence IS NOT NULL "
            "AND combo_type='vocal_over_instrumental'").fetchone()[0]
        if not n:
            return
        row = conn.execute(
            "SELECT harmonic_confidence FROM mashup_candidates "
            "WHERE harmonic_confidence IS NOT NULL AND combo_type='vocal_over_instrumental' "
            "ORDER BY harmonic_confidence LIMIT 1 OFFSET ?",
            (int(n * HARMONY_UNSURE_QUANTILE),)).fetchone()
    finally:
        conn.close()
    set_pref("harmony_unsure_below", {"value": float(row[0]), "pairs": int(n),
                                      "quantile": HARMONY_UNSURE_QUANTILE}, db_path=db)


PROVISIONAL_SCORER = "provisional"
# One scoring run at a time: a track's incremental run (after its quick tier
# and again after its stems) must not interleave with a library re-score.
_SCORE_LOCK = threading.RLock()


def _provisional_stand_ins(full: list, vocals: list, inst: list):
    """A track with a full-mix analysis and no stem rows yet (the quick tier
    ran, Demucs has not) stands its mix in for both stems, flagged, so it
    pairs minutes after download instead of after separation. Its rows are
    written with the provisional scorer and replaced when the stems land."""
    stemmed = {f["song_id"] for f in vocals} | {f["song_id"] for f in inst}
    v_add, i_add = [], []
    for f in full:
        if f["song_id"] in stemmed:
            continue
        base = {**f, "provisional": True, "stem_quality": None}
        v_add.append({**base, "stem_type": "vocals"})
        i_add.append({**base, "stem_type": "instrumental"})
    return v_add, i_add


def score_song_pairs(song_id: int, db_path=None, progress=None) -> dict:
    """Score one track against the library, both roles, and replace its
    vocal-over-instrumental rows — what the pipeline calls after the quick
    tier (provisional pairs) and after the final analysis (real ones)."""
    return score_all_pairs(db_path=db_path, song_ids=[song_id], progress=progress)


def _mark_scored(db, only: Optional[set]) -> None:
    """Stamp songs.scored_at for what this run covered: the named tracks in
    incremental mode, every song in a full run — which really is the whole
    library, provisional stand-ins included (_provisional_stand_ins), and a row
    with nothing to score is stamped harmlessly because its first features
    write bumps pairs_input_at past it again. One UPDATE, never one per track,
    and never fatal — the stamp decides whether the pipeline scores a track
    again, not whether its pairs exist."""
    from database.models import get_conn
    try:
        conn = get_conn(db)
        try:
            if only:
                ids = sorted(only)
                conn.execute(
                    "UPDATE songs SET scored_at=datetime('now') WHERE id IN "
                    f"({','.join('?' * len(ids))})", ids)
            else:
                conn.execute("UPDATE songs SET scored_at=datetime('now')")
            conn.commit()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        log.warning("could not stamp songs.scored_at for this run", exc_info=True)


def score_all_pairs(db_path=None, bpm_max_diff: Optional[float] = None,
                    key_min_score: Optional[float] = None,
                    scorer: str = "auto", progress=None,
                    song_ids: Optional[Iterable[int]] = None) -> dict:
    with _SCORE_LOCK:
        return _score_all_pairs(db_path, bpm_max_diff, key_min_score, scorer,
                                progress, song_ids)


def _score_all_pairs(db_path=None, bpm_max_diff: Optional[float] = None,
                     key_min_score: Optional[float] = None,
                     scorer: str = "auto", progress=None,
                     song_ids: Optional[Iterable[int]] = None) -> dict:
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

    song_ids: incremental mode. Only pairs with one of these tracks on either
    side are scored, and only their vocal-over-instrumental rows (forced rows
    aside) are replaced; the rest of the table, the instrumental pass and
    `last_score` are left alone, so the dock still says when a full re-score
    is due.

    Returns { 'vocal_over_instrumental': [...], 'instrumental_over_instrumental': [...] }
    Each list is sorted by total score descending.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import (
        bulk_upsert_candidates, candidate_row, clear_candidates,
        get_all_features, get_sections, DB_PATH, FEATURES_SCORING_COLUMNS,
    )
    # Read once per run, so a scoring pass is internally consistent even if the
    # user saves settings while it is going, and so a knob turned in the UI
    # applies to the very next re-score rather than the next server restart.
    from config import current_scoring_settings
    cfg = current_scoring_settings()
    MATCH_WEIGHTS = cfg["match_weights"]
    # P1.3 — the vocal path moves timbre's weight onto collision. See
    # config._for_combo: "does the bed leave room for the vocal" is the question
    # that decides a vocal mashup; "do these sound like the same record" is the
    # one that decides an instrumental blend.
    VOCAL_WEIGHTS = cfg["match_weights_vocal"]
    EFFORT_WEIGHT = cfg["effort_weight"]
    STEM_QUALITY_MIN = cfg["stem_quality_min"]
    BPM_MAX_DIFF = cfg["bpm_max_diff"]
    KEY_MIN_SCORE = cfg["key_min_score"]
    MAX_SECTION_PAIRS_PER_SONG_PAIR = cfg["max_section_pairs"]
    # The section weights and the mashup patterns, read ONCE for the run and
    # handed to every section search. Read per section pair (the old default)
    # they cost a stat + read + JSON parse of settings.json each — 624k of them
    # at 204 tracks, the bulk of an hour-long re-score.
    from config import current_section_weights
    from matcher.patterns import current_patterns
    SECTION_WEIGHTS_RUN = current_section_weights()
    PATTERNS_RUN = current_patterns()

    db = db_path or DB_PATH
    # The segmentation engine, once per run (readme §5.7): which sections pairs
    # are cut from and — for allin1 — phrase placement and the timing gate.
    ENGINE, TIMING_BASE = run_engine(db, cfg)
    RUN_CACHE: dict = TIMING_BASE["_cache"] if TIMING_BASE is not None else {}
    pipelines = song_pipelines(db)
    # When this run started reading inputs, in SQLite's datetime('now') form so
    # songs.pairs_input_at compares as text. Stored only on success: a failed
    # run must not make stale pairs look fresh.
    from datetime import datetime, timezone
    run_started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    bpm_max = float(bpm_max_diff) if bpm_max_diff is not None else BPM_MAX_DIFF
    key_min = float(key_min_score) if key_min_score is not None else KEY_MIN_SCORE

    # The rule prices and selects every row. A learned model no longer
    # replaces the total: once the rows are chosen, the active model's
    # shrinkage blend is applied to them (matcher/learning.blend_table, below),
    # so the gate, the pruning and the cap are the rule's whatever is active.
    # scorer='heuristic' skips the blend; 'auto' and 'model' apply it.
    active_scorer = "heuristic"
    model_version = None

    only = {int(x) for x in song_ids} if song_ids is not None else None
    if only is None:
        clear_candidates(db_path=db)
    else:
        from database.models import delete_song_candidates
        delete_song_candidates(only, keep_scorers=(FORCED_SCORER,), db_path=db)

    cols        = FEATURES_SCORING_COLUMNS
    vocals      = get_all_features(stem_type="vocals",        columns=cols, db_path=db)
    inst        = get_all_features(stem_type="instrumental",  columns=cols, db_path=db)
    full        = get_all_features(stem_type="full",          columns=cols, db_path=db)
    full_by_song = _full_by_song(full, db_path=db)
    v_add, i_add = _provisional_stand_ins(full, vocals, inst)
    vocals += v_add
    inst += i_add

    # Sections feed two things: the model's pair features, and the winning
    # section pair stored on every vocal-over-instrumental row (T3.3). Read once
    # per song — the alternative is one query per candidate.
    _sections_cache: Dict[int, list] = {}

    def _sections(song_id):
        if song_id not in _sections_cache:
            _sections_cache[song_id] = get_sections(song_id, db_path=db, engine=ENGINE)
        return _sections_cache[song_id]

    # Read the library's normalisation constants ONCE for this run — timbre and
    # energy are relative measures, and recomputing them per pair would re-read
    # every features row for every candidate. Refreshed here so a re-analysis
    # since the last run is picked up.
    lib_stats = get_library_stats(db_path=db, refresh=only is None)

    # Tempo compatibility is scored off the full-mix BPM (vocal-stem and even
    # instrumental-stem tempo tracking can drift octave/onset-detection errors
    # introduced by stem separation artifacts) while key/camelot/timbre stay
    # stem-derived, since those are what the listener actually hears layered.
    vocals = [_with_full_bpm(v, full_by_song) for v in vocals]
    inst   = [_with_full_bpm(i, full_by_song) for i in inst]

    code_index, key_table, shift_table, shift_known = _camelot_index([vocals, inst])
    # The whole-library blocks serve only a full run: an incremental run builds
    # its own (this track's rows against the rest), so building these too was
    # two library-sized preparations thrown away on every per-track score.
    if only is None:
        v_block = _prepare_block(vocals, code_index, lib_stats)
        i_block = _prepare_block(inst, code_index, lib_stats)
    else:
        def _part(feats, inside):
            return [f for f in feats if (f["song_id"] in only) == inside]
        v_in, v_out, i_in = _part(vocals, True), _part(vocals, False), _part(inst, True)

    results = {
        "vocal_over_instrumental":        [],
        "instrumental_over_instrumental": [],
    }
    # P1.1 — with the key gate off, the surviving pair count is roughly 4x what
    # it was, and each survivor contributes up to MAX_SECTION_PAIRS rows. Only
    # the best MAX_CANDIDATE_ROWS per combo type are kept (_TopRows), so
    # widening the gate costs a bounded amount of memory instead of an
    # unbounded one. Nothing good is lost: what is dropped is always the worst.
    max_rows = cfg["max_candidate_rows"]
    keepers: Dict[str, _TopRows] = {k: _TopRows(max_rows) for k in results}
    # Every track's best partners, kept whatever the cap does (_Reserve).
    reserve = _Reserve(cfg["reserve_partners"])
    scored = 0

    def _report(base: int, span: int, label: str):
        """Map a pass's block progress onto a slice of the 0-100 job bar."""
        def _on_block(done: int, total: int) -> None:
            if progress:
                pct = base + int(span * done / max(total, 1))
                progress(pct, f"{label}: {done}/{total} tracks")
        return _on_block

    def _gate(*, key_gate, upper_triangle, weights, bpm_gate=None) -> dict:
        """The arguments every block pass of one combo shares."""
        return dict(stats=lib_stats, key_table=key_table,
                    shift_table=shift_table, shift_known=shift_known,
                    bpm_max=bpm_gate if bpm_gate is not None else bpm_max,
                    key_min=key_gate, upper_triangle=upper_triangle,
                    weights=weights, effort_weight=EFFORT_WEIGHT,
                    quality_min=STEM_QUALITY_MIN)

    # The sections each side would actually contribute, filtered and ordered
    # once per song. best_section_pair then only walks the survivors.
    _usable_cache: Dict[tuple, list] = {}

    def _usable(song_id: int, vocal_side: bool) -> list:
        key = (song_id, vocal_side)
        if key not in _usable_cache:
            from matcher.sections import usable_sections
            _usable_cache[key] = usable_sections(_sections(song_id), vocal_side)
        return _usable_cache[key]

    def _section_pairs(top: dict, bed: dict) -> list:
        """Every section pairing worth a row for these two songs (E.3).

        Empty until both sides have structure, in which case the caller emits a
        single section-less row and readers fall back to each track's hook.
        """
        from matcher.sections import top_section_pairs
        v_use = _usable(top["song_id"], True)
        i_use = _usable(bed["song_id"], False)
        if not v_use or not i_use:
            out = []
        else:
            stretch = compute_stretch_factor(top.get("bpm"), bed.get("bpm")) or 1.0
            # The vocal sets the target tempo, so bars are counted at its BPM.
            out = top_section_pairs(v_use, i_use, stretch, prefiltered=True,
                                    bpm=top.get("bpm"),
                                    limit=MAX_SECTION_PAIRS_PER_SONG_PAIR,
                                    weights=SECTION_WEIGHTS_RUN,
                                    patterns=PATTERNS_RUN,
                                    timing=timing_for(
                                        TIMING_BASE, top["song_id"], db,
                                        compute_semitone_shift(
                                            str(top.get("camelot") or ""),
                                            str(bed.get("camelot") or ""))),
                                    lazy=TIMING_BASE is not None)
        return out

    def _price(top, bed, scores, seq: tuple, combo_type, row_scorer,
               row_version, weights, with_sections=False) -> None:
        """Price one song pair's rows and offer them to the keeper. ``seq`` is
        the pair's place in emission order (row-major over the top side) — the
        tiebreak, so equal-scoring rows rank as the in-order loop ranked them
        whatever order the pairs are priced in."""
        nonlocal scored
        keeper = keepers[combo_type]
        # One row per section pair (E.3): "chorus over drop" and "verse over
        # breakdown" are different ideas about the same two records.
        section_pairs = _section_pairs(top, bed) if with_sections else []
        pair_scorer = (PROVISIONAL_SCORER
                       if top.get("provisional") or bed.get("provisional")
                       else row_scorer)
        if with_sections and TIMING_BASE is not None and not section_pairs \
                and _usable(top["song_id"], True) and _usable(bed["song_id"], False):
            # Every section pairing failed the timing gate: the song pair is
            # dropped, not stored section-less (that would hide the verdict).
            return
        facts = dropin_facts(top, bed) if with_sections else {}
        best_item = None
        for j, sp in enumerate(section_pairs or [None]):
            s = dict(scores)
            s.update(facts)
            s["engine"] = ENGINE if TIMING_BASE is None else "essentia_fallback"
            s["pipeline"] = pair_pipeline(pipelines, top["song_id"], bed["song_id"])
            if sp is not None:
                _apply_measured_harmony(top, bed, s, sp, _sections,
                                        weights, EFFORT_WEIGHT,
                                        cache=RUN_CACHE)
                _apply_section_fit(s, sp, EFFORT_WEIGHT,
                                   weights, cfg["section_weight"])
            scored += 1
            total = float(s.get("total") or 0.0)
            row_seq = seq + (j,)
            # The pair's best row is also offered to each side's reserve.
            leads = with_sections and (best_item is None or total > best_item[0])
            if leads:
                best_item = (total, row_seq, None, None)
            kept = keeper.wants(total)
            if not kept and not (leads and (
                    reserve.wants("v", top["song_id"], total, row_seq)
                    or reserve.wants("b", bed["song_id"], total, row_seq))):
                continue
            # Priced; only a row that may be kept is put in its stored shape.
            if sp is not None and "_row" in sp:
                sp = sp["_row"]()
            item = (total, row_seq,
                    candidate_row(top, bed, s, combo_type,
                                  pair_scorer, row_version, sp),
                    _build_row(top, bed, s, sp))
            if leads:
                best_item = item
            if kept:
                keeper.push(*item)
        if best_item is not None and best_item[2] is not None:
            for side, song_id in (("v", top["song_id"]), ("b", bed["song_id"])):
                if reserve.wants(side, song_id, best_item[0], best_item[1]):
                    reserve.push(side, song_id, best_item[0], best_item[1], best_item)

    # ── vocal over instrumental ───────────────────────────────────────────────
    voi_gate = _gate(key_gate=key_min, upper_triangle=False, weights=VOCAL_WEIGHTS)
    if only is None:
        passes = [(v_block, i_block)]
    else:
        # This track on top of everything, then everything else on top of it.
        passes = [(_prepare_block(tops, code_index, lib_stats),
                   _prepare_block(beds, code_index, lib_stats))
                  for tops, beds in ((v_in, inst), (v_out, i_in)) if tops and beds]
    from database.models import sections_for_songs
    timing_weight = float(TIMING_BASE["weight"]) if TIMING_BASE is not None else 0.0
    # A run that could store more rows than the cap keeps only the best of
    # them, so it need not price a song pair whose ceiling is under the cut
    # (_score_pruned). A smaller run — one track, a small library — prices
    # every pair. A learned model does not stop the pruning: it only blends
    # the rows the rule kept (matcher/learning.blend_table).
    can_overflow = sum(len(t.feats) * len(b.feats) for t, b in passes) \
        * max(1, MAX_SECTION_PAIRS_PER_SONG_PAIR) > max_rows
    slot_weight = float(TIMING_BASE["slot_weight"]) if TIMING_BASE is not None else 0.0
    pruned = PRUNE_RUNS and can_overflow \
        and 0.0 <= timing_weight <= 1.0 and 0.0 <= slot_weight <= 1.0 \
        and all(w >= 0 for w in SECTION_WEIGHTS_RUN.values())

    def _load_phrases(vocal_ids) -> None:
        """Every song's phrases in one read, into the run's timing context:
        the vocal's own (timing_for) and the bed's — the vocal it used to
        carry, which the slot reads (sections.bed_phrases)."""
        if TIMING_BASE is None:
            return
        missing = set(vocal_ids) - set(TIMING_BASE["_phrases"])
        if missing:
            from database.alignment import phrases_for_songs
            TIMING_BASE["_phrases"].update(phrases_for_songs(missing, db_path=db))

    slot_sample: List[float] = []

    def _sample_slots(pairs) -> None:
        """The slot over a random sample of the gated pairs — what the Drop-in
        rule ranks a slot against (compat.slot_floor). The table cannot say:
        it keeps the best rows, which are the high-slot ones. The pairs are
        searched as any pair is and nothing of them is stored; the run's
        timing counts are put back as they were."""
        if TIMING_BASE is None or only is not None:
            return
        counts = dict(TIMING_BASE["stats"])
        for top, bed in pairs:
            for sp in _section_pairs(top, bed):
                row = sp["_row"]() if "_row" in sp else sp
                if row.get("score_slot") is not None:
                    slot_sample.append(float(row["score_slot"]))
        TIMING_BASE["stats"].clear()
        TIMING_BASE["stats"].update(counts)

    def _on_section_pair(done: int, total: int) -> None:
        if progress:
            progress(40 + int(45 * done / max(total, 1)),
                     f"Choosing sections: {done}/{total} pairs")

    pruned_stats = None
    if not pruned:
        voi = []
        for top_block, bed_block in passes:
            voi += [(top_block.feats[ti], bed_block.feats[bi], scores)
                    for ti, bi, scores in _iter_scored_pairs(
                        top_block, bed_block, **voi_gate,
                        on_block=_report(0, 30, "Scoring vocal over instrumental"
                                         if only is None else "Scoring this track"))]
        # Every song whose sections the section search will read, in ONE query
        # rather than one (or, under allin1, two) per partner song.
        _wanted = {t["song_id"] for t, _b, _sc in voi} | {b["song_id"] for _t, b, _sc in voi}
        _wanted -= set(_sections_cache)
        if _wanted:
            _sections_cache.update(sections_for_songs(_wanted, db_path=db, engine=ENGINE))
        _load_phrases({x["song_id"] for t, b, _sc in voi for x in (t, b)})
        if voi:
            picks = np.random.default_rng(len(voi)).choice(
                len(voi), size=min(SLOT_SAMPLE_PAIRS, len(voi)), replace=False)
            _sample_slots((voi[int(k)][0], voi[int(k)][1]) for k in sorted(picks))
        # The section search is per song pair and was the silent half of a
        # run: the bar sat at 55% for the whole of it. Report every ~1%.
        tick = max(1, len(voi) // 100)
        for k, (top, bed, scores) in enumerate(voi):
            if k % tick == 0:
                _on_section_pair(k, len(voi))
            _price(top, bed, scores, (0, k), "vocal_over_instrumental",
                   active_scorer, model_version, VOCAL_WEIGHTS, with_sections=True)
        voi = None                  # release before the second pass allocates
    else:
        _wanted = {f["song_id"] for t, b in passes for f in t.feats + b.feats}
        _wanted -= set(_sections_cache)
        if _wanted:
            _sections_cache.update(sections_for_songs(_wanted, db_path=db, engine=ENGINE))
        _load_phrases({f["song_id"] for t, b in passes for f in t.feats + b.feats})
        pruned_stats = _score_pruned(
            passes, voi_gate, keepers["vocal_over_instrumental"], _usable,
            lambda top, bed, scores, seq: _price(
                top, bed, scores, seq, "vocal_over_instrumental", active_scorer,
                model_version, VOCAL_WEIGHTS, with_sections=True),
            weights=VOCAL_WEIGHTS, section_weights=SECTION_WEIGHTS_RUN,
            section_weight=cfg["section_weight"], effort_weight=EFFORT_WEIGHT,
            timing_weight=timing_weight if TIMING_BASE is not None else None,
            slot_weight=slot_weight, cache=RUN_CACHE, reserve=reserve,
            sample=_sample_slots, progress=progress)

    # ── instrumental over instrumental ────────────────────────────────────────
    # Always heuristic (BPM + key gate), even when a model is active — the model
    # has no training signal for instrumental↔instrumental transitions. Only
    # lower id over higher, so A/B and B/A are not both scored.
    # Reuse vocal/inst columns: vocal_* = the "top" layer, inst_* = the "bed".
    # No section pair: the top layer here is an instrumental, and the vocal-side
    # filter (vocal_presence ≥ 0.25) would be asking the wrong question of it.
    if only is None:
        ioi_keeper = keepers["instrumental_over_instrumental"]

        def _ioi_worth(rows, keep, total):
            """A row here is its block total, so what cannot be kept is known
            before a single row is built: anything under the cut so far, or
            under this block's own ``max_rows``-th best. (1e-4: the stored
            total is this one rounded to four places.)"""
            cut = ioi_keeper.floor()
            n = int(keep.sum())
            if n > max_rows:
                kth = float(np.partition(total[keep], n - max_rows)[n - max_rows])
                cut = kth if cut is None else max(cut, kth)
            return keep if cut is None else keep & (total >= cut - 1e-4)

        for ti, bi, scores in _iter_scored_pairs(
                i_block, i_block,
                **_gate(key_gate=key_min, upper_triangle=True, weights=MATCH_WEIGHTS),
                post_select=_ioi_worth,
                on_block=_report(85, 5, "Scoring instrumental over instrumental")):
            _price(i_block.feats[ti], i_block.feats[bi], scores, (0, ti, bi),
                   "instrumental_over_instrumental", "heuristic", None, MATCH_WEIGHTS)

    # Drain the keepers in ranked order: (-total, seq), so ties keep emission
    # order, which is how this function has always resolved them.
    rows: List[tuple] = []
    kept = 0
    for key, keeper in keepers.items():
        ordered = keeper.ranked()
        if key == "vocal_over_instrumental":
            have = {t[1] for t in ordered}
            ordered += sorted((t for t in reserve.items() if t[1] not in have),
                              key=lambda t: (-t[0], t[1]))
        rows.extend(t[2] for t in ordered)
        results[key] = [t[3] for t in ordered]
        kept += len(ordered)
    keepers = None

    if progress:
        progress(90, f"Writing {len(rows)} candidates…")
    bulk_upsert_candidates(rows, db_path=db)
    rows = None
    # The learned model's shrinkage blend over the rows just written — the
    # one place a stored total moves off the rule's (matcher/learning.py).
    # No active model, or no verdicts behind it: alpha is 0 and nothing is
    # touched, so every total stays the rule's to the bit.
    if scorer != "heuristic":
        try:
            from matcher.learning import blend_table
            blended = blend_table(db, song_ids=only,
                                  progress=_report(91, 8, "Applying the learned model"))
        except Exception:  # noqa: BLE001 — a model must never break scoring
            log.exception("could not apply the learned model's blend; "
                          "rows keep the rule's totals")
            blended = None
        if blended and blended.get("alpha"):
            active_scorer = "blend"
            model_version = blended.get("version")
    # The tracks this run covered now have pairs for their current inputs
    # (songs.scored_at). One statement, on its own connection, after the rows
    # are committed — never a per-track call, and never a write of
    # app_prefs.last_score, which only a full run owns.
    _mark_scored(db, only)

    evaluated = len(vocals) * len(inst) + len(inst) * (len(inst) - 1) // 2
    dropped = scored - kept
    log.info(f"  Pairs scored: {scored} of {evaluated} possible  "
             f"|  kept: {kept}"
             + (f" (capped, {dropped} lowest-scoring dropped)" if dropped else "")
             + f"  |  scorer={active_scorer}"
             + (f" ({model_version})" if model_version else "")
             + f"  filter: bpm_max_diff={bpm_max} key_min_score={key_min}")
    if TIMING_BASE is not None:
        st = TIMING_BASE["stats"]
        log.info("  timing gate (%s ms): %d vocal sections placed, %d dropped as drifting",
                 TIMING_BASE["tol_ms"], st.get("passed", 0), st.get("gated", 0))
    if only is None:
        from database.models import set_pref
        set_pref("last_score", {"started_at": run_started, "scorer": active_scorer,
                                "engine": ENGINE,
                                "timing": (TIMING_BASE or {}).get("stats")},
                 db_path=db)
        _record_harmony_threshold(db)
        if slot_sample:
            from config import DROPIN_SLOT_PERCENTILE
            set_pref("slot_dropin_floor", {
                "value": round(float(np.quantile(slot_sample, DROPIN_SLOT_PERCENTILE)), 4),
                "pairings": len(slot_sample)}, db_path=db)
    return {**results, "_scorer": active_scorer, "_model_version": model_version,
            "_engine": ENGINE,
            # {"pairs", "priced"} when the run skipped pairs that could not be
            # kept (_score_pruned), else None. The timing stats then count the
            # vocal sections of the pairs that were priced.
            "_pruned": pruned_stats,
            "_timing": (TIMING_BASE or {}).get("stats")}


def _score_pruned(passes, gate: dict, keeper: _TopRows, usable, price, *,
                  weights: Dict, section_weights: Dict[str, float],
                  section_weight: float, effort_weight: float,
                  timing_weight: Optional[float], cache: dict,
                  slot_weight: float = 0.0, reserve: Optional[_Reserve] = None,
                  sample=None, progress=None) -> dict:
    """Price the vocal-over-instrumental pairs of a run that can overflow the
    row cap, most promising first, and stop when nothing left can be kept.

    Every gated song pair gets a CEILING — the most any of its rows could
    total: its own song terms, with the key term at the best harmony its
    sections' chroma could measure (_harmony_ceiling), a free transpose, the
    best labels and voice its sections offer, a perfect duration fit, a
    perfect grid and the fullest slot the bed's sections offer. Pairs are then priced exactly, in tiers of falling ceiling;
    once ``keeper`` is full its cut only rises, a pair whose ceiling is under
    the cut is skipped, and the run ends when the cut reaches the next tier's
    ceiling. The gate and every row's arithmetic are untouched, so the rows
    kept are the ones pricing every pair would keep (tests/test_score_scale.py
    runs both on one library) — what changes is that the rest are never built.

    Returns {"pairs", "priced"}: gated song pairs, and how many were priced.
    """
    from matcher.effort import EFFORT_WEIGHTS
    w_key = weights.get("key_score", 0)
    w_pitch = EFFORT_WEIGHTS["pitch_cost"]
    # The terms a section's own statics cannot bound are bounded by 1.
    # (Every weight but the two the statics answer — so a term added later is
    # covered without being named here.)
    open_terms = sum(w for k, w in section_weights.items()
                     if k not in ("label", "voice"))

    def _tick(pct: int, message: str) -> None:
        if progress:
            progress(pct, message)

    ceilings = []
    hist = np.zeros(_BOUND_BINS, dtype=np.int64)
    for top, bed in passes:
        a_v, has_v, _ = _section_ceilings(top.feats, usable, True,
                                          section_weights, cache)
        b_b, has_b, slot_ub = _section_ceilings(bed.feats, usable, False,
                                                section_weights, cache)
        v_mat, v_spans, v_unknown = _chroma_rows(top.feats, usable, True, cache)
        b_mat, b_spans, b_unknown = _chroma_rows(bed.feats, usable, False, cache)
        ub = np.full((len(top.feats), len(bed.feats)), -np.inf, dtype=np.float32)
        for rows, keep, t in _iter_scored_pairs(
                top, bed, **gate, terms_only=True,
                on_block=lambda done, n: _tick(
                    int(15 * done / max(n, 1)), f"Bounding pairs: {done}/{n} tracks")):
            measured = _harmony_ceiling(v_mat, v_spans, b_mat, b_spans, rows)
            # A section without chroma keeps the Camelot key score.
            unknown = (v_unknown[rows][:, None] | b_unknown[None, :]) \
                | ~np.isfinite(measured)
            key_ub = np.where(unknown, np.maximum(measured, t["key"]), measured)
            whole = t["fit"] + w_key * (key_ub - t["key"])
            fit_ub = a_v[rows][:, None] + b_b[None, :] + open_terms
            if timing_weight is not None:
                fit_ub = (1.0 - timing_weight) * fit_ub + timing_weight
                # ...and the slot at the most this bed's sections allow.
                fit_ub = (1.0 - slot_weight) * fit_ub + slot_weight * slot_ub[None, :]
            effort_lb = np.maximum(
                t["effort"] - w_pitch * t["parts"]["pitch_cost"] - 2e-4, 0.0)
            bound = (((1.0 - section_weight) * whole + section_weight * fit_ub)
                     * (1.0 - effort_weight * effort_lb))
            # Without usable sections on both sides the pair is one
            # section-less row: its block total.
            both = has_v[rows][:, None] & has_b[None, :]
            bound = np.where(both, bound, t["total"]) + _BOUND_SLACK
            ub[rows] = np.where(keep, bound, -np.inf)
            hist += np.histogram(np.clip(bound[keep], 0.0, _BOUND_TOP - 1e-9),
                                 bins=_BOUND_BINS, range=(0.0, _BOUND_TOP))[0]
        ceilings.append(ub)

    # Before anything is priced (pricing blanks a pair's ceiling): a uniform
    # sample of the gated pairs, for the caller's library-wide statistics.
    if sample is not None and int(hist.sum()):
        # Drawn by rejection — a random cell, kept when it is a gated pair —
        # so nothing the size of the pair matrix is built to choose from.
        rng = np.random.default_rng(int(hist.sum()))
        counts = [int(np.count_nonzero(np.isfinite(ub))) for ub in ceilings]
        chosen = []
        for (top, bed), ub, n in zip(passes, ceilings, counts):
            quota = min(n, int(round(SLOT_SAMPLE_PAIRS * n / max(1, sum(counts)))))
            seen = set()
            tries = 0
            while len(seen) < quota and tries < 200:
                tries += 1
                ti = rng.integers(0, ub.shape[0], size=4 * quota)
                bi = rng.integers(0, ub.shape[1], size=4 * quota)
                for i, j in zip(ti[np.isfinite(ub[ti, bi])], bi[np.isfinite(ub[ti, bi])]):
                    seen.add((int(i), int(j)))
                    if len(seen) >= quota:
                        break
            chosen += [(top.feats[i], bed.feats[j]) for i, j in sorted(seen)]
        sample(chosen)

    # Tier edges, from the top of the ceiling histogram: the first tier is
    # enough pairs to fill the keeper if they are as good as their ceilings,
    # each later one twice the last.
    n_pairs = int(hist.sum())
    edges, cum, want = [], 0, max(1, int(keeper.cap * _FIRST_TIER_SHARE))
    step = want
    for k in range(_BOUND_BINS - 1, 0, -1):
        cum += int(hist[k])
        if cum >= want and cum < n_pairs:
            edges.append(k * _BOUND_TOP / _BOUND_BINS)
            step *= 2
            want = cum + step
    edges.append(-np.inf)

    priced, hi = 0, np.inf
    for lo in edges:
        for p, (top, bed) in enumerate(passes):
            ub = ceilings[p]

            def _in_tier(rows, keep, ub=ub, lo=lo, hi=hi):
                cut = keeper.floor()
                u = ub[rows]
                return keep & (u >= (lo if cut is None else max(lo, cut))) & (u < hi)

            for ti, bi, scores in _iter_scored_pairs(top, bed, **gate,
                                                     pre_select=_in_tier):
                cut = keeper.floor()
                if cut is not None and ub[ti, bi] < cut:
                    continue                # the cut rose since this block began
                price(top.feats[ti], bed.feats[bi], scores, (p, ti, bi))
                ub[ti, bi] = -np.inf        # priced: never selected again
                priced += 1
                if priced % 2000 == 0:
                    _tick(15 + min(69, int(70 * priced / max(n_pairs, 1))),
                          f"Choosing sections: {priced} pairs priced of {n_pairs}"
                          + (f", cut {cut:.3f}" if cut is not None else ""))
        hi = lo
        cut = keeper.floor()
        if cut is not None and cut >= lo:
            break                           # every pair left has a ceiling under the cut
    capped = priced

    # ── each track's own best partners ────────────────────────────────────────
    # The cap is settled; what is left unpriced has a ceiling under its cut.
    # A track whose best partners are down there still gets them by a bounded
    # search: every track has its RESERVE_SEARCH × k most promising unpriced
    # partners (by ceiling, on each side) priced, then a track still short of
    # k partners — many pairs fail the timing gate and give no row — has its
    # next most promising priced, round by round, up to RESERVE_SEARCH_MAX × k
    # in all. A search, not a certificate: certifying the exact best took
    # rounds that did not end in hours at 10,000 tracks, because a ceiling is
    # loose by the duration, grid and slot it assumes.
    if reserve is not None and reserve.k:
        first_k = max(4, RESERVE_SEARCH * reserve.k)
        more_k = max(2, 2 * reserve.k)
        limit = RESERVE_SEARCH_MAX * reserve.k
        for p, (top, bed) in enumerate(passes):
            ub = ceilings[p]
            if not ub.size:
                continue
            sides = ((1, [f["song_id"] for f in top.feats], "v"),
                     (0, [f["song_id"] for f in bed.feats], "b"))
            used = {axis: np.zeros(len(ids), dtype=np.int64) for axis, ids, _s in sides}
            first = True
            while True:
                want = np.zeros(ub.shape, dtype=bool)
                for axis, ids, side in sides:
                    if first:
                        active = np.ones(len(ids), dtype=bool)
                        k = first_k
                    else:
                        active = np.array(
                            [len(reserve.heaps.get((side, s), ())) < reserve.k
                             for s in ids], dtype=bool) & (used[axis] < limit)
                        k = more_k
                    if not active.any():
                        continue
                    k = min(k, ub.shape[axis])
                    # In strips of tracks, so no index array the size of the
                    # pair matrix is ever built.
                    for lo_ in range(0, len(ids), 512):
                        strip = slice(lo_, min(lo_ + 512, len(ids)))
                        on = active[strip]
                        if not on.any():
                            continue
                        cut = ub[strip] if axis == 1 else ub[:, strip]
                        cut = np.where(on[:, None] if axis == 1 else on[None, :],
                                       cut, -np.inf)
                        part = np.argpartition(-cut, k - 1, axis=axis)
                        idx = np.take(part, np.arange(k), axis=axis)
                        pick = np.zeros(cut.shape, dtype=bool)
                        np.put_along_axis(pick, idx, True, axis=axis)
                        pick &= np.isfinite(cut)
                        used[axis][strip] += pick.sum(axis=axis)
                        if axis == 1:
                            want[strip] |= pick
                        else:
                            want[:, strip] |= pick
                first = False
                if not want.any():
                    break

                def _wanted(rows, keep, want=want):
                    return keep & want[rows]

                for ti, bi, scores in _iter_scored_pairs(top, bed, **gate,
                                                         pre_select=_wanted):
                    price(top.feats[ti], bed.feats[bi], scores, (p, ti, bi))
                    ub[ti, bi] = -np.inf
                    priced += 1
                    if priced % 2000 == 0:
                        _tick(88, f"Each track's best partners: {priced - capped} "
                                  "more pairs priced")
                # A picked pair the gate keeps out is never priced: spent.
                ub[want] = -np.inf
    log.info("  pruned run: %d of %d gated song pairs priced (%d for the cap, "
             "cut %s; %d for each track's own best partners)",
             priced, n_pairs, capped, keeper.floor(), priced - capped)
    return {"pairs": n_pairs, "priced": priced, "for_reserve": priced - capped}


# ── The segmentation engine (readme §5.7) ─────────────────────────────────────

def run_engine(db, cfg: Optional[dict] = None, engine: Optional[str] = None):
    """(engine, timing context or None) for one scoring run. In essentia mode
    the context is None and the section search is exactly what it was."""
    from config import TIMING_PHRASE_RATIOS, current_float, current_segmentation_engine
    engine = engine or current_segmentation_engine()
    if engine != "allin1":
        return engine, None
    try:
        from analysis.allin1_runner import version
        ver = version()
    except Exception:  # noqa: BLE001
        ver = None
    return engine, {
        "tol_ms": current_float("timing_tol_ms"),
        "weight": current_float("timing_weight"),
        # The vacated slot (readme §5.7): its share of a placed pairing's fit.
        "slot_weight": current_float("slot_weight"),
        "_db": db,
        "ratios": TIMING_PHRASE_RATIOS,
        "engine_version": ver,
        "stats": {},
        "_phrases": {},
        # Per-section work the placement and the section fit would otherwise
        # redo for every partner (matcher/phrase_align.place_cached). Lives as
        # long as this context, i.e. one run.
        "_cache": {},
    }


def timing_for(base: Optional[dict], vocal_song_id: int, db,
               shift: Optional[int] = None) -> Optional[dict]:
    """The timing context for one vocal song: the run's, plus that song's
    phrases (memoised in the run's context) and ``shift`` — the Camelot
    transpose of the bed it is about to be searched against, which the slot's
    register reads."""
    if base is None:
        return None
    cache = base["_phrases"]
    if vocal_song_id not in cache:
        from database.alignment import get_vocal_phrases
        cache[vocal_song_id] = get_vocal_phrases(vocal_song_id, db_path=db)
    return {**base, "vocal_phrases": cache[vocal_song_id], "shift": shift}


def song_pipelines(db) -> Dict[int, str]:
    from database.models import get_conn
    conn = get_conn(db)
    try:
        return {r[0]: r[1] or "unified_4stem" for r in conn.execute(
            "SELECT id, separation_pipeline FROM songs")}
    finally:
        conn.close()


def pair_pipeline(pipelines: Dict[int, str], a: int, b: int) -> str:
    pa, pb = pipelines.get(a, "unified_4stem"), pipelines.get(b, "unified_4stem")
    return pa if pa == pb else f"{pa}+{pb}"


FORCED_SCORER = "forced"


def dropin_facts(top: dict, bed: dict) -> dict:
    """What the Drop-in rule reads of the two songs, as stored on every
    vocal-over-instrumental row (models.DROPIN_COLUMNS)."""
    from matcher.compat import dropin_mask
    v_cam, b_cam = str(top.get("camelot") or ""), str(bed.get("camelot") or "")
    return {"dropin_mask": dropin_mask(v_cam, b_cam, top.get("bpm"), bed.get("bpm")),
            "camelot_shift": compute_semitone_shift(v_cam, b_cam)}


def score_pair(vocal_song_id: int, inst_song_id: int, db_path=None) -> List[dict]:
    """Score one vocal-over-bed pair the library run left out — outside the
    tempo or key gate, two cuts of one record — on demand ("Score this pair
    anyway"). The same functions as score_all_pairs: composite_score, the
    section search, measured harmony, the section fit. Rows are stored with
    scorer='forced' so the card can say the gates would have dropped it, and
    the next Score library run clears them like any other row.

    Returns the stored rows best-first; [] when either side has no analysed
    stem (nothing to score)."""
    from config import current_scoring_settings, current_section_weights
    from matcher.patterns import current_patterns
    from database.models import bulk_upsert_candidates, candidate_row, get_sections
    from matcher.sections import top_section_pairs, usable_sections

    # Resolved per call, never the DB_PATH bound when this module was imported
    # (readme §7: workers call without a path).
    import database.models as _models
    db = db_path if db_path is not None else _models.DB_PATH
    cfg = current_scoring_settings()
    weights = cfg["match_weights_vocal"]
    effort_weight = cfg["effort_weight"]
    pair_ids = (vocal_song_id, inst_song_id)
    full_by_song = _full_by_song(get_all_features(
        stem_type="full", columns=FEATURES_SCORING_COLUMNS, song_ids=pair_ids,
        db_path=db), db_path=db)
    top = next(iter(get_all_features(
        stem_type="vocals", columns=FEATURES_SCORING_COLUMNS,
        song_ids=[vocal_song_id], db_path=db)), None)
    bed = next(iter(get_all_features(
        stem_type="instrumental", columns=FEATURES_SCORING_COLUMNS,
        song_ids=[inst_song_id], db_path=db)), None)
    if top is None or bed is None:
        return []
    top, bed = _with_full_bpm(top, full_by_song), _with_full_bpm(bed, full_by_song)
    stats = get_library_stats(db_path=db)
    base = composite_score(top, bed, weights=weights, stats=stats,
                           effort_weight=effort_weight,
                           combo_type="vocal_over_instrumental")
    cache: Dict[int, list] = {}

    engine, timing = run_engine(db, cfg)

    def sections_of(song_id):
        if song_id not in cache:
            cache[song_id] = get_sections(song_id, db_path=db, engine=engine)
        return cache[song_id]

    v_use = usable_sections(sections_of(vocal_song_id), True)
    i_use = usable_sections(sections_of(inst_song_id), False)
    stretch = compute_stretch_factor(top.get("bpm"), bed.get("bpm")) or 1.0
    pairs = (top_section_pairs(v_use, i_use, stretch, prefiltered=True, bpm=top.get("bpm"),
                               limit=cfg["max_section_pairs"],
                               weights=current_section_weights(), patterns=current_patterns(),
                               timing=timing_for(
                                   timing, vocal_song_id, db,
                                   compute_semitone_shift(
                                       str(top.get("camelot") or ""),
                                       str(bed.get("camelot") or ""))))
             if v_use and i_use else [])
    rows, built = [], []
    pipelines = song_pipelines(db)
    for sp in (pairs or [None]):
        sc = dict(base)
        sc.update(dropin_facts(top, bed))
        sc["engine"] = engine if timing is None else "essentia_fallback"
        sc["pipeline"] = pair_pipeline(pipelines, vocal_song_id, inst_song_id)
        if sp is not None:
            _apply_measured_harmony(top, bed, sc, sp, sections_of, weights, effort_weight)
            _apply_section_fit(sc, sp, effort_weight, weights, cfg["section_weight"])
        rows.append(candidate_row(top, bed, sc, "vocal_over_instrumental",
                                  FORCED_SCORER, None, sp))
        built.append(_build_row(top, bed, sc, sp))
    from database.models import get_conn
    conn = get_conn(db)
    try:
        # A re-force replaces the pair's forced rows rather than piling them up.
        conn.execute("DELETE FROM mashup_candidates WHERE scorer=? AND vocal_song_id=? "
                     "AND inst_song_id=?", (FORCED_SCORER, vocal_song_id, inst_song_id))
        conn.commit()
    finally:
        conn.close()
    bulk_upsert_candidates(rows, db_path=db)
    try:
        from matcher.learning import blend_table
        blend_table(db, pairs=[(vocal_song_id, inst_song_id)])
    except Exception:  # noqa: BLE001
        log.exception("could not blend the forced pair")
    return sorted(built, key=lambda r: -(r.get("total") or 0))


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


def _apply_measured_harmony(top: dict, bed: dict, scores: dict,
                            section_pair: dict, sections_of,
                            weights: Dict, effort_weight: float,
                            cache: Optional[dict] = None) -> None:
    """Replace the Camelot key_score with the MEASURED harmonic fit (Phase E).

    The block scorer works on whole-track Camelot codes because that is what
    vectorises. Once the winning section pair is known, the two sections'
    stored chroma give a better answer to the same question: not "are these
    scales compatible" but "do these notes agree, and at what transposition".

    Mutates `scores` in place — key_score, the total, and the harmony fields the
    row carries — and does nothing at all when either section has no chroma, so
    a library analysed before Phase E ranks exactly as before.
    """
    from matcher.harmony import section_harmony

    v_idx = section_pair.get("vocal_section_idx")
    b_idx = section_pair.get("inst_section_idx")
    if v_idx is None or b_idx is None:
        return
    # A lazy row (matcher.sections._lazy_row) carries the two sections; a
    # stored-shape row is resolved by index.
    v_sec = section_pair.get("_v") or next(
        (s for s in sections_of(top["song_id"])
         if s.get("section_index") == v_idx), None)
    b_sec = section_pair.get("_i") or next(
        (s for s in sections_of(bed["song_id"])
         if s.get("section_index") == b_idx), None)
    h = section_harmony(v_sec, b_sec, cache=cache)
    if not h["known"]:
        return

    scores["key_score"] = h["harmonic_fit"]
    scores["harmonic_shift"] = h["shift"]
    scores["harmonic_confidence"] = round(h["confidence"], 4)
    scores["bass_clash"] = 1 if h["bass_clash"] else 0

    # Effort priced the Camelot transpose; the pair now plays the measured one
    # (matcher/recipe.bed_shift), so the cost is re-priced on it — a "Free"
    # chip beside a −5 st transpose was a contradiction on one card.
    if "effort_pitch" in scores:
        from matcher.effort import effort_total_from_columns, transpose_cost
        scores["effort_pitch"] = round(transpose_cost(h["shift"]), 4)
        scores["score_effort"] = round(effort_total_from_columns(scores), 4)

    fit = sum(scores[k] * weights.get(k, 0) for k in weights)
    effort = scores.get("score_effort") or 0.0
    scores["total"] = round(fit * (1.0 - effort_weight * effort), 4)


def _apply_section_fit(scores: dict, section_pair: dict,
                       effort_weight: float,
                       weights: Optional[Dict] = None,
                       section_weight: Optional[float] = None) -> None:
    """Blend the section fit into the total (E.3).

    While the row was a SONG pair and the section was chosen afterwards, folding
    the section fit in would have been double-counting a post-hoc annotation —
    which is what T3.3's "selects and describes; it must not re-rank" was
    protecting against. Now the row IS the section pair, so how well those two
    sections cover each other is part of what is being ranked.

    Mutates `scores` in place, and does nothing when the pair carries no fit.
    """
    from config import current_match_weights, current_float

    weights = weights if weights is not None else current_match_weights()
    section_weight = (section_weight if section_weight is not None
                      else current_float("section_weight"))

    fit_section = section_pair.get("score_section")
    if fit_section is None:
        return
    whole = sum(scores[k] * weights.get(k, 0) for k in weights)
    blended = (1.0 - section_weight) * whole + section_weight * float(fit_section)
    effort = scores.get("score_effort") or 0.0
    scores["total"] = round(blended * (1.0 - effort_weight * effort), 4)
