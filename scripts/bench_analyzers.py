#!/usr/bin/env python
"""
scripts/bench_analyzers.py — librosa vs. Essentia on real tracks from the library.

Answers the Phase 0 questions before any analyser is swapped:

  * how long each unit of work takes (decode, tempo, key, spectral frame pass,
    loudness, segmentation, each Essentia estimator), median of --repeats runs,
    and the real-time factor;
  * whether the two agree on BPM (same / ×2 / ×½ / ×3/2 / ×2/3 / other), on key
    (same / relative / fifth / parallel / other, MIREX-weighted) for every
    Essentia key profile, on section boundaries (F-measure at ±0.5 s and ±3 s)
    and on downbeats;
  * how each compares with ground truth where there is some: BPM/key tags
    embedded in the file (TBPM, TKEY, initialkey) and an optional --truth CSV.

Essentia is optional. Without it (native Windows, or before the Docker image
carries it) the librosa half still runs and still reports timings.

Run from the repo root:

    python scripts/bench_analyzers.py --auto 10
    python scripts/bench_analyzers.py --songs 12,40,41 --repeats 3
    python scripts/bench_analyzers.py --files a.mp3 b.mp3 --truth truth.csv
    python scripts/bench_analyzers.py --auto 10 --models-dir data/essentia_models

truth.csv columns (any may be blank): song_id or file, bpm, key, mode,
boundaries (seconds, separated by ';').

Outputs go to <data_dir>/bench/<timestamp>/ — never into the repo: results.csv
(one row per measurement), timings.csv, summary.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.compare import (  # noqa: E402  (path set above)
    boundaries_from_sections, boundary_prf, bpm_relation, downbeat_agreement,
    key_relation, mirex_key_score, parse_key_text, tally,
)

log = logging.getLogger("bench")

# The profiles KeyExtractor accepts in essentia 2.1b6.dev1389 that are worth
# comparing (faraldo, in older docs, is rejected by this build).
KEY_PROFILES = ("edma", "edmm", "bgate", "braw", "krumhansl", "temperley",
                "shaath")
SR_FULL, SR_ANALYSIS, SR_TEMPOCNN = 44100, 22050, 11025
FRAME, HOP = 2048, 512
TEMPOCNN_MODEL = "deeptemp-k16-3.pb"


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    """Median-of-N wall time per unit, and the last result of each."""

    def __init__(self, repeats: int):
        self.repeats = max(1, repeats)
        self.rows: list[dict] = []

    def run(self, track: str, analyzer: str, unit: str, fn: Callable,
            audio_secs: Optional[float] = None, repeats: Optional[int] = None):
        times, result, error = [], None, None
        for _ in range(repeats or self.repeats):
            t0 = time.perf_counter()
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 — one unit must not stop the run
                error = f"{type(exc).__name__}: {exc}"
                log.warning("%s %s/%s failed: %s", track, analyzer, unit, error)
                break
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times) if times else None
        self.rows.append({
            "track": track, "analyzer": analyzer, "unit": unit,
            "median_ms": round(ms, 1) if ms is not None else "",
            "rtf": round(ms / 1000.0 / audio_secs, 4) if ms and audio_secs else "",
            "runs": len(times), "error": error or "",
        })
        return result if error is None else None


# ── Inputs ────────────────────────────────────────────────────────────────────

def _bpm_band(bpm: Optional[float]) -> str:
    if not bpm:
        return "unknown"
    return "<100" if bpm < 100 else ("100-130" if bpm <= 130 else ">130")


def pick_songs(ids: Optional[list[int]], auto: int) -> list[dict]:
    """Library tracks with audio on disk. --auto spreads the pick round-robin
    over BPM bands, taking an unseen genre first within a band, so ten tracks
    are not ten 128 BPM house records."""
    from database.models import get_conn
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT s.id, s.title, s.artist, s.genre, s.raw_path, s.duration_secs,
                      f.bpm, f.key, f.mode
                 FROM songs s LEFT JOIN features f
                   ON f.song_id = s.id AND f.stem_type = 'full'
                WHERE s.raw_path IS NOT NULL AND s.raw_path != ''
                ORDER BY s.id""").fetchall()
    finally:
        conn.close()
    songs = [dict(r) for r in rows if r["raw_path"] and Path(r["raw_path"]).exists()]
    if ids:
        wanted = set(ids)
        return [s for s in songs if s["id"] in wanted]

    bands: dict[str, list[dict]] = {}
    for s in songs:
        bands.setdefault(_bpm_band(s["bpm"]), []).append(s)
    for band in bands.values():
        seen: set = set()
        first, rest = [], []
        for s in band:
            g = (s["genre"] or "").strip().lower()
            (rest if g in seen else first).append(s)
            seen.add(g)
        band[:] = first + rest
    picked: list[dict] = []
    while len(picked) < auto and any(bands.values()):
        for name in sorted(bands):
            if bands[name] and len(picked) < auto:
                picked.append(bands[name].pop(0))
    return picked


def stored_sections(song_id: int) -> list[dict]:
    try:
        from database.models import get_sections
        return get_sections(song_id)
    except Exception:  # noqa: BLE001
        return []


def load_truth(path: Optional[Path]) -> dict:
    """{song_id or file name: {bpm, key, mode, boundaries}}"""
    if not path:
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ref = (row.get("song_id") or "").strip() or Path(row.get("file") or "").name
            if not ref:
                continue
            bounds = [float(x) for x in (row.get("boundaries") or "").split(";") if x.strip()]
            out[ref] = {
                "bpm": float(row["bpm"]) if (row.get("bpm") or "").strip() else None,
                "key": (row.get("key") or "").strip() or None,
                "mode": (row.get("mode") or "").strip() or None,
                "boundaries": bounds or None,
            }
    return out


def probe_tags(path: Path) -> dict:
    """BPM and key tags already in the file, via ffprobe. Empty on any failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format_tags:stream_tags", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out or "{}")
    except Exception:  # noqa: BLE001
        return {}
    tags: dict = {}
    for block in [data.get("format", {})] + data.get("streams", []):
        for k, v in (block.get("tags") or {}).items():
            tags[k.lower()] = v
    bpm = None
    for k in ("tbpm", "bpm"):
        try:
            bpm = float(str(tags.get(k, "")).strip()) or None
        except ValueError:
            pass
        if bpm:
            break
    key, mode = None, None
    for k in ("tkey", "initialkey", "key"):
        key, mode = parse_key_text(tags.get(k))
        if key:
            break
    return {"bpm": bpm, "key": key, "mode": mode}


# ── librosa (the analyser in production) ──────────────────────────────────────

def bench_librosa(timer: Timer, name: str, path: Path) -> dict:
    import librosa
    from analysis import decode
    from analysis.analyze import analyze_file
    from analysis.structure import detect_sections

    audio_secs = float(librosa.get_duration(path=str(path)))
    timer.run(name, "librosa", "decode", lambda: librosa.load(
        str(path), sr=SR_ANALYSIS, mono=True), audio_secs)

    # analyze_file reports its own per-step split; the repeat loop here gives the
    # median of the whole call and the last run's split.
    steps: dict = {}

    def _analyze():
        steps.clear()
        # Cold every time: analyze_file shares decodes and transforms through
        # analysis/decode.py, which would otherwise make repeats 2..n (and the
        # structure pass below) measure a cache instead of the analysis.
        decode.clear()
        return analyze_file(path, timings=steps)

    feats = timer.run(name, "librosa", "analyze_file(total)", _analyze, audio_secs) or {}
    for step, ms in steps.items():
        if isinstance(ms, (int, float)) and step != "audio_secs":
            timer.rows.append({"track": name, "analyzer": "librosa",
                               "unit": f"analyze.{step}", "median_ms": round(ms, 1),
                               "rtf": round(ms / 1000.0 / audio_secs, 4),
                               "runs": 1, "error": ""})

    # Structure on the mix alone: the Essentia side has no stems either, so this
    # is the like-for-like cost of finding boundaries.
    sphases: dict = {}

    def _structure():
        sphases.clear()
        decode.clear()
        return detect_sections(path, timings=sphases)

    sections = timer.run(name, "librosa", "detect_sections(mix)", _structure,
                         audio_secs, repeats=1) or []
    for phase, ms in sphases.items():
        if isinstance(ms, (int, float)) and phase != "audio_secs":
            timer.rows.append({"track": name, "analyzer": "librosa",
                               "unit": f"structure.{phase}", "median_ms": round(ms, 1),
                               "rtf": round(ms / 1000.0 / audio_secs, 4),
                               "runs": 1, "error": ""})

    beats = feats.get("beat_times") or []
    phase = feats.get("beat_phase") or 0
    return {
        "audio_secs": audio_secs,
        "bpm": feats.get("bpm"), "key": feats.get("key"), "mode": feats.get("mode"),
        "key_confidence": feats.get("key_confidence"),
        "boundaries": boundaries_from_sections(sections),
        "downbeats": beats[phase::4],
    }


# ── Essentia (the candidate) ──────────────────────────────────────────────────

def ffmpeg_decode(path: Path):
    """One decode, 44.1 kHz stereo float32 — the Tier 1 design's only decode."""
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2",
         "-ar", str(SR_FULL), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)


def essentia_frames(es, mono22):
    """The one spectral frame pass Tier 1 would run: every per-frame descriptor
    from a single windowed spectrum."""
    import numpy as np
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC(inputSize=FRAME // 2 + 1, sampleRate=SR_ANALYSIS)
    peaks = es.SpectralPeaks(sampleRate=SR_ANALYSIS, orderBy="magnitude",
                             magnitudeThreshold=1e-5, minFrequency=40,
                             maxFrequency=5000, maxPeaks=60)
    hpcp = es.HPCP(size=12, sampleRate=SR_ANALYSIS, minFrequency=40,
                   maxFrequency=5000)
    centroid = es.Centroid(range=SR_ANALYSIS / 2)
    rolloff = es.RollOff(sampleRate=SR_ANALYSIS)
    flux = es.Flux()
    flatness = es.FlatnessDB()
    hfc = es.HFC(sampleRate=SR_ANALYSIS)
    zcr = es.ZeroCrossingRate()
    rms = es.RMS()
    bands = [es.EnergyBand(sampleRate=SR_ANALYSIS, startCutoffFrequency=lo,
                           stopCutoffFrequency=hi)
             for lo, hi in ((20, 250), (250, 4000), (4000, SR_ANALYSIS / 2 - 1))]
    cols: dict = {k: [] for k in ("mfcc", "hpcp", "centroid", "rolloff", "flux",
                                  "flatness", "hfc", "zcr", "rms", "bands3")}
    for frame in es.FrameGenerator(mono22, frameSize=FRAME, hopSize=HOP,
                                   startFromZero=True):
        spec = spectrum(window(frame))
        cols["mfcc"].append(mfcc(spec)[1])
        freqs, mags = peaks(spec)
        cols["hpcp"].append(hpcp(freqs, mags))
        cols["centroid"].append(centroid(spec))
        cols["rolloff"].append(rolloff(spec))
        cols["flux"].append(flux(spec))
        cols["flatness"].append(flatness(spec))
        cols["hfc"].append(hfc(spec))
        cols["zcr"].append(zcr(frame))
        cols["rms"].append(rms(frame))
        cols["bands3"].append([b(spec) for b in bands])
    return {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}


def beat_sync(values, frame_times, ticks, agg):
    """Aggregate frame rows between consecutive beats. Returns (dims, beats)."""
    import numpy as np
    idx = np.searchsorted(ticks, frame_times, side="right") - 1
    out = []
    for b in range(len(ticks) - 1):
        rows = values[idx == b]
        out.append(agg(rows, axis=0) if len(rows) else np.zeros(values.shape[1]))
    return np.asarray(out, dtype=np.float32).T


def essentia_novelty(frames, ticks, low_band) -> list[float]:
    """The production boundary method (checkerboard novelty + 8-bar phrase snap)
    fed Essentia features on Essentia's beat grid, so a difference in the
    result is the features' and the grid's, not the algorithm's."""
    import numpy as np
    from analysis.analyze import _pick_beat_phase
    from analysis.structure import (
        SECTION_MAX_COUNT, SECTION_MIN_LEN_SECS, _novelty_boundaries,
        snap_boundaries_to_phrases,
    )
    ticks = np.asarray(ticks, dtype=float)
    if len(ticks) < 16:
        return []
    frame_times = np.arange(len(frames["mfcc"])) * HOP / SR_ANALYSIS
    chroma_b = beat_sync(frames["hpcp"], frame_times, ticks, np.median)
    mfcc_b = beat_sync(frames["mfcc"], frame_times, ticks, np.mean)
    mfcc_z = (mfcc_b - mfcc_b.mean(axis=1, keepdims=True)) / \
             (mfcc_b.std(axis=1, keepdims=True) + 1e-9)
    X = np.vstack([chroma_b, mfcc_z])
    beat_dur = float(np.median(np.diff(ticks)))
    min_beats = max(8, int(round(SECTION_MIN_LEN_SECS / max(beat_dur, 1e-3))))
    bounds = _novelty_boundaries(X, min_beats=min_beats, max_sections=SECTION_MAX_COUNT)
    phase = _pick_beat_phase(np.asarray(low_band, dtype=float), np.arange(len(low_band)))
    bounds, _snapped = snap_boundaries_to_phrases(bounds, phase, len(ticks), min_beats)
    return [float(ticks[b]) for b in bounds if b < len(ticks)]


def bench_essentia(timer: Timer, name: str, path: Path,
                   models_dir: Optional[Path]) -> dict:
    import numpy as np
    import essentia
    import essentia.standard as es
    essentia.log.infoActive = False
    essentia.log.warningActive = False

    stereo = timer.run(name, "essentia", "decode(ffmpeg 44.1k stereo)",
                       lambda: ffmpeg_decode(path))
    if stereo is None:
        return {}
    audio_secs = len(stereo) / SR_FULL
    timer.rows[-1]["rtf"] = round(timer.rows[-1]["median_ms"] / 1000.0 / audio_secs, 4)
    mono44 = np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32)
    # Polyphase FIR (scipy) rather than essentia.Resample: measured ~5x faster
    # than Resample's default quality on the same signal, for Tier 1's purposes
    # (beat tracking, spectra) the same result.
    from scipy.signal import resample_poly
    mono22 = timer.run(name, "essentia", "resample→22.05k(scipy poly)",
                       lambda: resample_poly(mono44, 1, 2).astype(np.float32), audio_secs)
    mono11 = timer.run(name, "essentia", "resample→11.025k(scipy poly)",
                       lambda: resample_poly(mono44, 1, 4).astype(np.float32), audio_secs)

    out: dict = {"audio_secs": audio_secs, "bpm": {}, "keys": {}}

    # What the pipeline actually runs (analysis/essentia_groups.py), group by
    # group, on the same decode — the numbers the Phase 2 analyser costs.
    from analysis import essentia_groups as eg
    sig = eg.Signals(stereo)
    for step, run in eg._runners().items():
        timer.run(name, "essentia", f"group.{step} (pipeline)",
                  lambda r=run: r(sig), audio_secs, repeats=1)

    for method in ("degara", "multifeature"):
        r = timer.run(name, "essentia", f"RhythmExtractor2013({method})",
                      lambda m=method: es.RhythmExtractor2013(method=m)(mono44),
                      audio_secs)
        if r is not None:
            out["bpm"][f"rhythm_{method}"] = float(r[0])
            out[f"ticks_{method}"] = [float(t) for t in r[1]]
            out[f"rhythm_{method}_confidence"] = float(r[2])

    p = timer.run(name, "essentia", "PercivalBpmEstimator",
                  lambda: es.PercivalBpmEstimator(sampleRate=SR_FULL)(mono44), audio_secs)
    if p is not None:
        out["bpm"]["percival"] = float(p)

    model = (models_dir / TEMPOCNN_MODEL) if models_dir else None
    if model and model.exists() and mono11 is not None:
        t = timer.run(name, "essentia", "TempoCNN",
                      lambda: es.TempoCNN(graphFilename=str(model))(mono11), audio_secs)
        if t is not None:
            out["bpm"]["tempocnn"] = float(t[0])

    for profile in KEY_PROFILES:
        k = timer.run(name, "essentia", f"KeyExtractor({profile})",
                      lambda pr=profile: es.KeyExtractor(profileType=pr,
                                                         sampleRate=SR_FULL)(mono44),
                      audio_secs)
        if k is not None:
            out["keys"][profile] = {"key": k[0], "mode": k[1], "strength": float(k[2])}

    tuning = timer.run(name, "essentia", "TuningFrequencyExtractor",
                       lambda: es.TuningFrequencyExtractor()(mono44), audio_secs)
    if tuning is not None and len(tuning):
        out["tuning_hz"] = float(np.median(tuning))

    loud = timer.run(name, "essentia", "LoudnessEBUR128(stereo)",
                     lambda: es.LoudnessEBUR128(sampleRate=SR_FULL)(stereo), audio_secs)
    if loud is not None:
        out["lufs"], out["lra"] = float(loud[2]), float(loud[3])

    timer.run(name, "essentia", "Danceability",
              lambda: es.Danceability(sampleRate=SR_FULL)(mono44), audio_secs)
    timer.run(name, "essentia", "DynamicComplexity",
              lambda: es.DynamicComplexity(sampleRate=SR_FULL)(mono44), audio_secs)
    timer.run(name, "essentia", "OnsetRate", lambda: es.OnsetRate()(mono44), audio_secs)

    frames = timer.run(name, "essentia", "frame pass(22.05k, all spectral)",
                       lambda: essentia_frames(es, mono22), audio_secs, repeats=1)

    ticks = out.get("ticks_degara") or out.get("ticks_multifeature") or []
    low_band = None
    if len(ticks) >= 4:
        bl = timer.run(name, "essentia", "BeatsLoudness",
                       lambda: es.BeatsLoudness(sampleRate=SR_FULL, beats=ticks)(mono44),
                       audio_secs)
        if bl is not None:
            loudness, ratios = np.asarray(bl[0]), np.asarray(bl[1])
            low_band = loudness * (ratios[:, 0] if ratios.ndim == 2 and len(ratios) else 1.0)
            from analysis.analyze import _pick_beat_phase
            phase = _pick_beat_phase(low_band, np.arange(len(low_band)))
            out["downbeats"] = ticks[phase::4]

    if frames is not None and len(ticks) >= 16 and low_band is not None:
        b = timer.run(name, "essentia", "segments(novelty on HPCP+MFCC)",
                      lambda: essentia_novelty(frames, ticks, low_band), audio_secs,
                      repeats=1)
        out["boundaries_novelty"] = b or []
    if frames is not None:
        def _sbic():
            feats = essentia.array(frames["mfcc"].T)
            seg = es.SBic(cpw=1.5, inc1=60, inc2=20, minLength=10, size1=300,
                          size2=200)(feats)
            return [float(i) * HOP / SR_ANALYSIS for i in seg]
        b = timer.run(name, "essentia", "segments(SBic on MFCC)", _sbic, audio_secs,
                      repeats=1)
        out["boundaries_sbic"] = b or []
    return out


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(name: str, lib: dict, ess: dict, stored: dict, tags: dict,
            truth: dict, results: list) -> None:
    def add(measure: str, method: str, value) -> None:
        results.append({"track": name, "measure": measure, "method": method,
                        "value": value if value is not None else ""})

    refs = {"librosa": lib.get("bpm"), "stored": stored.get("bpm"),
            "tag": tags.get("bpm"), "truth": truth.get("bpm")}
    add("bpm", "librosa", lib.get("bpm"))
    for method, bpm in (ess.get("bpm") or {}).items():
        add("bpm", method, round(bpm, 2))
        for ref_name, ref in refs.items():
            add(f"bpm_vs_{ref_name}", method, bpm_relation(bpm, ref))
    add("bpm_vs_tag", "librosa", bpm_relation(lib.get("bpm"), tags.get("bpm")))
    add("bpm_vs_truth", "librosa", bpm_relation(lib.get("bpm"), truth.get("bpm")))

    key_refs = {"librosa": (lib.get("key"), lib.get("mode")),
                "tag": (tags.get("key"), tags.get("mode")),
                "truth": (truth.get("key"), truth.get("mode"))}
    add("key", "librosa", f"{lib.get('key')} {lib.get('mode')}" if lib.get("key") else None)
    for ref_name in ("tag", "truth"):
        rel = key_relation(lib.get("key"), lib.get("mode"), *key_refs[ref_name])
        add(f"key_vs_{ref_name}", "librosa", rel)
        add(f"key_mirex_vs_{ref_name}", "librosa", mirex_key_score(rel))
    for profile, k in (ess.get("keys") or {}).items():
        add("key", profile, f"{k['key']} {k['mode']}")
        add("key_strength", profile, round(k["strength"], 4))
        for ref_name, (rk, rm) in key_refs.items():
            rel = key_relation(k["key"], k["mode"], rk, rm)
            add(f"key_vs_{ref_name}", profile, rel)
            add(f"key_mirex_vs_{ref_name}", profile, mirex_key_score(rel))

    secs = lib.get("audio_secs") or ess.get("audio_secs")
    trim = (0.0, secs) if secs else None
    bound_sets = {"librosa": lib.get("boundaries"), "stored": stored.get("boundaries"),
                  "truth": truth.get("boundaries"),
                  "ess_novelty": ess.get("boundaries_novelty"),
                  "ess_sbic": ess.get("boundaries_sbic")}
    for est in ("librosa", "ess_novelty", "ess_sbic"):
        for ref in ("librosa", "stored", "truth"):
            if est == ref or bound_sets.get(est) is None or not bound_sets.get(ref):
                continue
            for window in (0.5, 3.0):
                prf = boundary_prf(bound_sets[est], bound_sets[ref], window, trim)
                add(f"bounds_f@{window}_vs_{ref}", est, prf["f"])
        if bound_sets.get(est) is not None:
            add("section_count", est, len(bound_sets[est]) + 1)

    add("downbeat_agreement_vs_librosa", "essentia_beatsloudness",
        downbeat_agreement(ess.get("downbeats") or [], lib.get("downbeats") or []))
    for extra in ("tuning_hz", "lufs", "lra", "rhythm_degara_confidence",
                  "rhythm_multifeature_confidence"):
        if ess.get(extra) is not None:
            add(extra, "essentia", round(ess[extra], 3))


def summarise(results: list, timings: list, n_tracks: int, essentia_ok: bool) -> str:
    lines = [f"# Analyser benchmark — {datetime.now():%Y-%m-%d %H:%M}", "",
             f"{n_tracks} tracks. Essentia: {'yes' if essentia_ok else 'NOT INSTALLED'}.",
             ""]

    # Timings: median over tracks of each unit's median.
    by_unit: dict = {}
    for r in timings:
        if r["median_ms"] != "":
            by_unit.setdefault((r["analyzer"], r["unit"]), []).append(
                (r["median_ms"], r["rtf"] or None))
    lines += ["## Time per unit (median across tracks)", "",
              "| analyzer | unit | median ms | median RTF |", "|---|---|---|---|"]
    for (analyzer, unit), vals in sorted(by_unit.items()):
        ms = statistics.median(v[0] for v in vals)
        rtfs = [v[1] for v in vals if v[1]]
        rtf = f"{statistics.median(rtfs):.4f}" if rtfs else ""
        lines.append(f"| {analyzer} | {unit} | {ms:.0f} | {rtf} |")

    def relation_table(title: str, prefix: str) -> None:
        groups: dict = {}
        for r in results:
            if r["measure"].startswith(prefix) and r["value"] != "":
                groups.setdefault((r["measure"], r["method"]), []).append(r["value"])
        if not groups:
            return
        lines.extend(["", f"## {title}", "", "| vs | method | counts |", "|---|---|---|"])
        for (measure, method), vals in sorted(groups.items()):
            counts = ", ".join(f"{k} {v}" for k, v in sorted(tally(vals).items(),
                                                             key=lambda kv: -kv[1]))
            lines.append(f"| {measure[len(prefix):]} | {method} | {counts} |")

    def mean_table(title: str, prefix: str) -> None:
        groups: dict = {}
        for r in results:
            if r["measure"].startswith(prefix) and r["value"] != "":
                groups.setdefault((r["measure"], r["method"]), []).append(float(r["value"]))
        if not groups:
            return
        lines.extend(["", f"## {title}", "", "| measure | method | mean | n |",
                      "|---|---|---|---|"])
        for (measure, method), vals in sorted(groups.items()):
            lines.append(f"| {measure} | {method} | {statistics.mean(vals):.3f} | {len(vals)} |")

    relation_table("BPM agreement", "bpm_vs_")
    relation_table("Key agreement", "key_vs_")
    mean_table("Key — MIREX weighted score", "key_mirex_vs_")
    mean_table("Section boundaries — F-measure", "bounds_f@")
    mean_table("Downbeats", "downbeat_agreement")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--songs", help="comma-separated song ids from the library")
    src.add_argument("--auto", type=int, default=0,
                     help="pick N library tracks spread across BPM band and genre")
    src.add_argument("--files", nargs="+", type=Path, help="audio files outside the DB")
    ap.add_argument("--truth", type=Path, help="ground-truth CSV (see module docstring)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--models-dir", type=Path,
                    help=f"folder holding Essentia models (e.g. {TEMPOCNN_MODEL})")
    ap.add_argument("--no-librosa", action="store_true")
    ap.add_argument("--no-essentia", action="store_true")
    ap.add_argument("--out", type=Path, help="output folder (default <data_dir>/bench/<ts>)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.files:
        tracks = [{"id": None, "title": p.stem, "raw_path": str(p)} for p in args.files]
    else:
        ids = [int(x) for x in args.songs.split(",")] if args.songs else None
        tracks = pick_songs(ids, args.auto or 10)
    if not tracks:
        log.error("no tracks with audio on disk matched")
        return 1

    essentia_ok = False
    # TensorFlow (inside essentia-tensorflow) prints CUDA probing noise on import.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if not args.no_essentia:
        try:
            import essentia.standard  # noqa: F401
            essentia_ok = True
        except ImportError:
            log.warning("essentia is not installed — benchmarking librosa only")

    if args.out:
        out_dir = args.out
    else:
        from config import DATA_DIR
        out_dir = Path(DATA_DIR) / "bench" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_all = load_truth(args.truth)
    timer = Timer(args.repeats)
    results: list = []
    for n, t in enumerate(tracks, start=1):
        path = Path(t["raw_path"])
        name = f"{t['id']}:{t['title']}" if t.get("id") is not None else path.name
        log.info("[%d/%d] %s", n, len(tracks), name)
        truth = truth_all.get(str(t.get("id"))) or truth_all.get(path.name) or {}
        tags = probe_tags(path)
        stored = {}
        if t.get("id") is not None:
            stored = {"bpm": t.get("bpm"),
                      "boundaries": boundaries_from_sections(stored_sections(t["id"]))}
        lib = {} if args.no_librosa else bench_librosa(timer, name, path)
        ess = bench_essentia(timer, name, path, args.models_dir) if essentia_ok else {}
        compare(name, lib, ess, stored, tags, truth, results)

    for fname, rows in (("results.csv", results), ("timings.csv", timer.rows)):
        if rows:
            with open(out_dir / fname, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    summary = summarise(results, timer.rows, len(tracks), essentia_ok)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        log.info("process peak RSS ≈ %.0f MB", peak_mb)
    except ImportError:
        pass
    print(summary)
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
