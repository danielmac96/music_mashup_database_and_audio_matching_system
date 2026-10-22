"""analysis/allin1_runner.py — run allin1 (All-In-One Music Structure Analyzer)
on a track and keep only what the app uses (readme §5.5).

allin1 is beats, downbeats, beat-in-bar and labelled sections from one neural
pass over the four Demucs sources (bass, drums, other, vocals). It needs four
stems — it cannot read a two-stem split — and by default it runs its own
htdemucs separation first. Here it never does when the unified pipeline has
already separated the track: the four kept sources are written as WAV into the
exact folder allin1's ``demix()`` checks
(``<demix_dir>/htdemucs/<track stem>/{bass,drums,other,vocals}.wav``, verified
against allin1 1.1.0's demix.py), so its separation step finds them and skips.

Only this module imports torch / allin1 / madmom / natten, and lazily: the API
and the test suite start without them (``available()`` says why not).

``summarise()`` is pure numpy over allin1's AnalysisResult shape, so it is
tested without the model. What it keeps (the brief's "what to keep" table):
  beats / downbeats / beat positions / bpm     in full, float32
  segments (start, end, label)                 in full
  activations                                  one value per downbeat and per
                                               segment + track mean / min
  embeddings                                   mean of the 8 ensemble models,
                                               then mean per bar, per stem:
                                               (bars, 4, 24) float16
The 100-fps arrays are dropped as soon as the summary exists.
"""
from __future__ import annotations

import base64
import contextlib
import gc
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

FPS = 100                     # allin1 frames: 44100 / 441
SAMPLE_RATE = 44100           # what its spectrogram front end assumes
SOURCES = ("bass", "drums", "other", "vocals")   # allin1's instrument order
HARMONIX_LABELS = ("start", "end", "intro", "outro", "break", "bridge", "inst",
                   "solo", "verse", "chorus")    # allin1.config.HARMONIX_LABELS
POOLING = "mean8models+bar"

_MODEL_LOCK = threading.Lock()
_MODELS: Dict[Tuple[str, str], object] = {}


# ── Availability ──────────────────────────────────────────────────────────────

def missing() -> Optional[str]:
    """Why allin1 cannot run here, or None when it can."""
    for mod, hint in (("torch", "torch"), ("allin1", "allin1"),
                      ("madmom", "madmom (git)"), ("natten", "natten")):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001 — natten raises more than ImportError
            return (f"{hint} does not import ({type(exc).__name__}: {exc}). "
                    "Install requirements-allin1.txt (the Docker image has it).")
    return None


def available() -> bool:
    return missing() is None


def version() -> Optional[str]:
    try:
        from importlib.metadata import version as _v
        return _v("allin1")
    except Exception:  # noqa: BLE001
        return None


# ── Pure summarising (no torch) ───────────────────────────────────────────────

def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _frame(t: float, n: int) -> int:
    return int(min(max(round(float(t) * FPS), 0), max(n - 1, 0)))


def _peak_near(act: np.ndarray, t: float, radius: int = 2) -> float:
    """Highest activation within ±radius frames of t (a detected event sits on
    a peak, but its time is quantised and post-processed)."""
    if act is None or not len(act):
        return float("nan")
    i = _frame(t, len(act))
    return float(np.max(act[max(0, i - radius): i + radius + 1]))


def pool_embeddings(emb: np.ndarray, downbeats: List[float],
                    n_frames: Optional[int] = None) -> Optional[np.ndarray]:
    """allin1's embeddings → (bars, stems, dim) float16.

    allin1 1.1.0's ensemble stacks each model's embedding on the LAST axis:
    (stems=4, frames, dim=24, models=8). A single model has no model axis.
    Averaged over models, then over each bar's frames (downbeat to downbeat; the
    last bar runs to the end). No downbeats → one "bar" for the whole track."""
    if emb is None:
        return None
    e = np.asarray(emb, dtype=np.float32)
    if e.ndim == 4:
        e = e.mean(axis=-1)
    if e.ndim != 3:
        raise ValueError(f"unexpected embedding shape {e.shape}")
    stems, frames, dim = e.shape
    n = frames if n_frames is None else min(frames, n_frames)
    edges = sorted({_frame(t, n) for t in (downbeats or [])})
    if not edges or edges[0] > 0:
        edges = [0] + edges
    edges = [x for x in edges if x < n] + [n]
    bars = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            bars.append(e[:, a:b, :].mean(axis=1))      # (stems, dim)
    if not bars:
        return None
    return np.stack(bars).astype(np.float16)              # (bars, stems, dim)


def summarise(result, content_hash: Optional[str] = None,
              demixed_by: str = "pipeline", ms: Optional[float] = None) -> Dict:
    """allin1 AnalysisResult (or the same fields as a dict) → the stored summary."""
    from config import ALLIN1_MODEL
    beats = [round(float(t), 4) for t in (_get(result, "beats") or [])]
    downbeats = [round(float(t), 4) for t in (_get(result, "downbeats") or [])]
    positions = [int(p) for p in (_get(result, "beat_positions") or [])]
    raw_segments = _get(result, "segments") or []
    segments = [[round(float(_get(s, "start")), 3), round(float(_get(s, "end")), 3),
                 str(_get(s, "label"))] for s in raw_segments]

    acts = _get(result, "activations") or {}
    db_act = np.asarray(acts["downbeat"], dtype=np.float32) if "downbeat" in acts else None
    seg_act = np.asarray(acts["segment"], dtype=np.float32) if "segment" in acts else None
    lab_act = np.asarray(acts["label"], dtype=np.float32) if "label" in acts else None

    downbeat_conf = None
    conf_mean = conf_min = None
    if db_act is not None and len(db_act) and downbeats:
        downbeat_conf = [round(_peak_near(db_act, t), 4) for t in downbeats]
        conf_mean = round(float(np.mean(downbeat_conf)), 4)
        conf_min = round(float(np.min(downbeat_conf)), 4)

    section_conf = []
    for start, end, label in segments:
        entry = {"boundary": None, "label_prob": None}
        if seg_act is not None and len(seg_act):
            entry["boundary"] = round(_peak_near(seg_act, start), 4)
        if lab_act is not None and lab_act.ndim == 2 and label in HARMONIX_LABELS:
            row = lab_act[HARMONIX_LABELS.index(label)]
            a, b = _frame(start, len(row)), _frame(end, len(row))
            if b > a:
                entry["label_prob"] = round(float(row[a:b].mean()), 4)
        section_conf.append(entry)

    n_frames = len(db_act) if db_act is not None else None
    emb = pool_embeddings(_get(result, "embeddings"), downbeats, n_frames)

    bpm = _get(result, "bpm")
    return {
        "content_hash": content_hash,
        "model": ALLIN1_MODEL,
        "version": version(),
        "pooling": POOLING,
        "demixed_by": demixed_by,
        "bpm": float(bpm) if bpm is not None else None,
        "beats": beats,
        "downbeats": downbeats,
        "beat_positions": positions,
        "segments": segments,
        "downbeat_conf": downbeat_conf,
        "section_conf": section_conf,
        "conf_mean": conf_mean,
        "conf_min": conf_min,
        "embedding": emb,
        "ms": ms,
    }


def to_cache(summary: Dict) -> Dict:
    """JSON-safe copy for feature_cache (the embedding as base64 float16)."""
    out = dict(summary)
    emb = out.pop("embedding", None)
    if emb is not None:
        emb = np.asarray(emb, dtype=np.float16)
        out["embedding_b64"] = base64.b64encode(emb.tobytes()).decode("ascii")
        out["embedding_shape"] = list(emb.shape)
    return out


def from_cache(payload: Dict) -> Dict:
    out = dict(payload)
    b64 = out.pop("embedding_b64", None)
    shape = out.pop("embedding_shape", None)
    out["embedding"] = (np.frombuffer(base64.b64decode(b64), dtype=np.float16)
                        .reshape(shape) if b64 and shape else None)
    return out


# ── Memory ────────────────────────────────────────────────────────────────────

def rss_mb() -> Optional[float]:
    """Resident memory of this process in MB (Linux /proc), else None."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


# A run's resident memory, roughly: torch + the 8-model ensemble, plus what
# grows with the track — the 4-stem spectrogram and, mostly, the per-frame
# embeddings (4 stems × 24 dims × 8 models × 4 bytes at 100 fps ≈ 0.3 MB/s) and
# activations. Deliberately generous; scripts/bench_pipelines.py measures the
# real peaks (analysis_runs.peak_mb) to replace these numbers with this
# machine's.
BASE_MB = 1500.0
MB_PER_AUDIO_SEC = 1.5


def estimate_mb(audio_secs: Optional[float]) -> float:
    return BASE_MB + MB_PER_AUDIO_SEC * float(audio_secs or 0.0)


def over_budget(audio_secs: Optional[float], budget_mb: float) -> Optional[str]:
    """Why running now would pass the memory budget, or None. A model already
    loaded in this process is part of the current RSS, so it is not counted
    twice."""
    now = rss_mb() or 0.0
    need = estimate_mb(audio_secs) - (BASE_MB if _MODELS else 0.0)
    if now + need > budget_mb:
        return (f"allin1 would need about {now + need:.0f} MB (budget "
                f"{budget_mb:.0f} MB, allin1_mem_budget_mb) — raise the budget or "
                f"lower allin1_max_secs")
    return None


class PeakRss:
    """Sample resident memory every 100 ms while the block runs; ``peak_mb``
    afterwards (None where /proc is unavailable)."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll(self):
        while not self._stop.is_set():
            r = rss_mb()
            if r is not None:
                self.peak_mb = r if self.peak_mb is None else max(self.peak_mb, r)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_mb = rss_mb()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        r = rss_mb()
        if r is not None and (self.peak_mb is None or r > self.peak_mb):
            self.peak_mb = r
        return False


# ── Running the model ─────────────────────────────────────────────────────────

def _device(requested: str) -> str:
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("allin1: cuda requested but torch sees no GPU — using cpu")
        return "cpu"
    return requested


def _model(device: str):
    """The ensemble, loaded once per process and device (a worker keeps it)."""
    from config import ALLIN1_MODEL
    key = (ALLIN1_MODEL, device)
    with _MODEL_LOCK:
        m = _MODELS.get(key)
        if m is None:
            from allin1.models import load_pretrained_model
            t0 = time.perf_counter()
            m = load_pretrained_model(model_name=ALLIN1_MODEL, device=device)
            log.info("allin1: loaded %s on %s in %.1f s", ALLIN1_MODEL, device,
                     time.perf_counter() - t0)
            _MODELS[key] = m
        return m


def _write_demix(stems: Dict[str, Path], out_dir: Path) -> Dict[str, float]:
    """The four sources as 44.1 kHz WAV in allin1's layout. Returns ms per step.
    Our stems are FLAC at the source rate (Demucs htdemucs: 44.1 kHz), so this
    is a decode + write, no resampling."""
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for src in SOURCES:
        data, sr = sf.read(str(stems[src]), dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), SAMPLE_RATE)
            data = resample_poly(data, SAMPLE_RATE // g, int(sr) // g, axis=0
                                 ).astype(np.float32)
        # 16-bit, as `demucs.separate` writes by default: the input allin1's
        # spectrogram front end (madmom's wave reader) was trained and tested on.
        sf.write(str(out_dir / f"{src}.wav"), np.clip(data, -1.0, 1.0), SAMPLE_RATE,
                 subtype="PCM_16")
    return {"demix_write": (time.perf_counter() - t0) * 1000.0}


def run(full_path: Path, stems: Optional[Dict[str, Path]], device: str = "cpu",
        timings: Optional[dict] = None, threads: Optional[int] = None):
    """Run allin1 on one track and return its AnalysisResult (activations and
    embeddings included). ``stems`` — the four sources — makes it skip its own
    separation; None lets allin1 demix (the legacy pipeline, timed as
    ``demix``). Temporary WAVs and spectrograms are deleted before returning."""
    import torch
    from allin1.demix import demix
    from allin1.helpers import run_inference
    from allin1.spectrogram import extract_spectrograms

    timings = timings if timings is not None else {}
    if threads:
        torch.set_num_threads(max(1, int(threads)))
    device = _device(device)
    work = Path(tempfile.mkdtemp(prefix="allin1_"))
    # allin1 names the demix folder after the track's file stem.
    track = work / "track" / Path(full_path).name
    track.parent.mkdir(parents=True, exist_ok=True)
    try:
        with contextlib.suppress(OSError):
            track.symlink_to(Path(full_path).resolve())
        if not track.exists():
            shutil.copyfile(full_path, track)
        demix_root = work / "demix"
        if stems and all(stems.get(s) for s in SOURCES):
            timings.update(_write_demix(stems, demix_root / "htdemucs" / track.stem))
        t0 = time.perf_counter()
        demix_paths = demix([track], demix_root, device)   # skips when present
        timings["demix"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        spec_paths = extract_spectrograms(demix_paths, work / "spec", multiprocess=False)
        timings["spectrogram"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        model = _model(device)
        timings["model_load"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        with torch.no_grad():
            result = run_inference(path=track, spec_path=spec_paths[0], model=model,
                                   device=device, include_activations=True,
                                   include_embeddings=True)
        timings["inference"] = (time.perf_counter() - t0) * 1000.0
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()
