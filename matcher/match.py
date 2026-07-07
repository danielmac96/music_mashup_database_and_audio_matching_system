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

log = logging.getLogger(__name__)


# ── Semitone / key helpers ────────────────────────────────────────────────────

_KEY_SEMITONE: Dict[str, int] = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,  "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11,
}


def compute_semitone_shift(vocal_key: str, inst_key: str) -> Optional[int]:
    """
    Minimum semitones to shift the INSTRUMENTAL to match the vocal's root note.
    Positive = shift up, negative = shift down. Range: -6 to +6.
    Returns None if either key is unknown.

    Formula: (vocal_semitone - inst_semitone) % 12 gives the "shift up" value.
    If > 6, shifting down by (value - 12) is cheaper.
    Relative major/minor is covered naturally: C(0) over A(9) → (0-9)%12=3 → +3.
    """
    v = _KEY_SEMITONE.get(vocal_key or "")
    i = _KEY_SEMITONE.get(inst_key or "")
    if v is None or i is None:
        return None
    diff = (v - i) % 12
    return diff if diff <= 6 else diff - 12


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


# ── BPM compatibility ─────────────────────────────────────────────────────────

def effective_bpm(target_bpm: float, other_bpm: float) -> float:
    """other_bpm interpreted at half/normal/double time, whichever lands
    closest to target_bpm."""
    if not target_bpm or not other_bpm:
        return other_bpm or 0.0
    options = (other_bpm, other_bpm / 2, other_bpm * 2)
    return min(options, key=lambda b: abs(target_bpm - b))


def compute_stretch_factor(vocal_bpm: float, inst_bpm: float) -> Optional[float]:
    """Ratio to stretch the instrumental (at whichever of half/normal/double
    time is closest to the vocal) to reach the vocal's tempo."""
    if not vocal_bpm or not inst_bpm:
        return None
    inst_eff = effective_bpm(vocal_bpm, inst_bpm)
    return round(vocal_bpm / inst_eff, 4) if inst_eff else None


def _bpm_min_diff(bpm1: float, bpm2: float) -> float:
    """Smallest BPM difference accounting for halftime and doubletime."""
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


# ── Energy compatibility ──────────────────────────────────────────────────────

def energy_score(e1: float, e2: float) -> float:
    if e1 <= 0 or e2 <= 0:
        return 0.5
    ratio = min(e1, e2) / max(e1, e2)
    return float(math.exp(-((1 - ratio) ** 2) / (2 * 0.25 ** 2)))


# ── Timbre similarity ─────────────────────────────────────────────────────────

def mfcc_cosine(mfcc1: list, mfcc2: list) -> float:
    if not mfcc1 or not mfcc2:
        return 0.5
    v1 = np.array(mfcc1, dtype=float)
    v2 = np.array(mfcc2, dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / norm, 0, 1))


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

def sub_scores(feat_a: dict, feat_b: dict) -> dict:
    """The four heuristic sub-scores for a (top, bed) pair. Extracted so the
    learned pair-feature builder (matcher/features.py) and the heuristic
    composite_score compute them identically — keeping train and serve aligned."""
    return {
        "bpm_score":    bpm_score(feat_a.get("bpm", 0),
                                   feat_b.get("bpm", 0)),
        "key_score":    camelot_score(feat_a.get("camelot", ""),
                                       feat_b.get("camelot", "")),
        "energy_score": energy_score(
                            feat_a.get("loudness_rms") or feat_a.get("energy", 0),
                            feat_b.get("loudness_rms") or feat_b.get("energy", 0)),
        "timbre_score": mfcc_cosine(feat_a.get("mfcc", []),
                                     feat_b.get("mfcc", [])),
    }


def composite_score(feat_a: dict, feat_b: dict,
                    weights: Optional[Dict] = None) -> dict:
    try:
        from config import MATCH_WEIGHTS
        weights = weights or MATCH_WEIGHTS
    except ImportError:
        weights = {"bpm_score": 0.25, "key_score": 0.30,
                   "energy_score": 0.20, "timbre_score": 0.25}

    scores = sub_scores(feat_a, feat_b)
    scores["total"] = round(
        sum(scores[k] * weights.get(k, 0) for k in scores), 4
    )
    return scores


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
                    scorer: str = "auto") -> dict:
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

    The candidates table is cleared first so the result reflects exactly the
    current features and thresholds — no stale pairs from a looser prior run.

    Returns { 'vocal_over_instrumental': [...], 'instrumental_over_instrumental': [...] }
    Each list is sorted by total score descending.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import (
        clear_candidates, get_all_features, get_sections, upsert_candidate, DB_PATH,
    )
    from config import BPM_MAX_DIFF, KEY_MIN_SCORE

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

    # Section lookups are only needed for the model's pair features.
    _sections_cache: Dict[int, list] = {}

    def _sections(song_id):
        if song_id not in _sections_cache:
            _sections_cache[song_id] = get_sections(song_id, db_path=db) if use_model else []
        return _sections_cache[song_id]

    def _passes(feat_a, feat_b):
        # Model path: BPM window only. Heuristic path: BPM + key (unchanged).
        if use_model:
            return _bpm_min_diff(feat_a.get("bpm") or 0, feat_b.get("bpm") or 0) <= bpm_max
        return _passes_filter(feat_a, feat_b, bpm_max, key_min)

    def _score(feat_a, feat_b):
        """Return a scores dict {bpm_score,key_score,energy_score,timbre_score,total}.
        Heuristic: total = weighted composite. Model: total = model probability,
        with the four heuristic sub-scores kept for display."""
        if not use_model:
            return composite_score(feat_a, feat_b)
        from matcher.features import pair_features
        from matcher.model_scorer import model_score
        scores = sub_scores(feat_a, feat_b)
        feats = pair_features(feat_a, feat_b,
                              _sections(feat_a.get("song_id")),
                              _sections(feat_b.get("song_id")))
        scores["total"] = round(model_score(feats, bundle), 4)
        return scores

    # Tempo compatibility is scored off the full-mix BPM (vocal-stem and even
    # instrumental-stem tempo tracking can drift octave/onset-detection errors
    # introduced by stem separation artifacts) while key/camelot/timbre stay
    # stem-derived, since those are what the listener actually hears layered.
    vocals = [_with_full_bpm(v, full_by_song) for v in vocals]
    inst   = [_with_full_bpm(i, full_by_song) for i in inst]

    results = {
        "vocal_over_instrumental":        [],
        "instrumental_over_instrumental": [],
    }

    skipped = 0
    scored  = 0

    # ── vocal over instrumental ───────────────────────────────────────────────
    for v in vocals:
        for i in inst:
            if v["song_id"] == i["song_id"]:
                continue
            if not _passes(v, i):
                skipped += 1
                continue

            scores = _score(v, i)
            upsert_candidate(v, i, scores, combo_type="vocal_over_instrumental",
                             scorer=active_scorer, model_version=model_version, db_path=db)
            results["vocal_over_instrumental"].append(_build_row(v, i, scores))
            scored += 1

    # ── instrumental over instrumental ────────────────────────────────────────
    for i_a in inst:
        for i_b in inst:
            if i_a["song_id"] == i_b["song_id"]:
                continue
            # Avoid duplicate A/B + B/A pairs — only score lower id over higher
            if i_a["song_id"] >= i_b["song_id"]:
                continue
            if not _passes(i_a, i_b):
                skipped += 1
                continue

            scores = _score(i_a, i_b)
            # Reuse vocal/inst columns: vocal_* = the "top" layer, inst_* = the "bed"
            upsert_candidate(i_a, i_b, scores, combo_type="instrumental_over_instrumental",
                             scorer=active_scorer, model_version=model_version, db_path=db)
            results["instrumental_over_instrumental"].append(
                _build_row(i_a, i_b, scores))
            scored += 1

    for key in results:
        results[key].sort(key=lambda x: x["total"], reverse=True)

    log.info(f"  Pairs scored: {scored}  |  Skipped (pre-filter): {skipped}  "
             f"|  scorer={active_scorer}"
             + (f" ({model_version})" if model_version else "")
             + f"  filter: bpm_max_diff={bpm_max} key_min_score={key_min}")
    return {**results, "_scorer": active_scorer, "_model_version": model_version}


def _build_row(feat_a: dict, feat_b: dict, scores: dict) -> dict:
    return {
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
                r.get("vocal_key") or "", r.get("inst_key") or ""
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
            r.get("vocal_key") or "", r.get("inst_key") or ""
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