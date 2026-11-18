"""
config.py — Central configuration for the mashup engine.

Path/setting resolution order (highest priority first):
    1. Environment variable   — set by Docker / the shell
    2. settings.json          — written by the first-run Setup Wizard
    3. Built-in default       — <repo>/audio, <repo>/mashup.db, 1 worker

Env overrides:
    MASHUP_AUDIO_ROOT        — relocate the audio library root
    MASHUP_DB_PATH           — relocate the SQLite database file
    MASHUP_PIPELINE_WORKERS  — pipeline worker thread count
    MASHUP_DATA_DIR          — where engine artifacts (snapshots/datasets/models) live
    MASHUP_SETTINGS_DIR      — override the settings.json directory (tests/Docker)

The settings layer must be read BEFORE the module-level constants bind, because
the whole codebase does `from config import RAW_DIR` (etc.) at import time. So
`_load_settings()` runs first, then constants resolve, then dirs are created.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent


# ── Settings file (env > settings.json > default) ─────────────────────────────

def _settings_dir() -> Path:
    """Platform-appropriate directory for settings.json.

    Windows: %APPDATA%\\mashup-engine
    macOS:   ~/Library/Application Support/mashup-engine
    Linux:   $XDG_CONFIG_HOME/mashup-engine (or ~/.config/mashup-engine)
    Override with MASHUP_SETTINGS_DIR (used by Docker + tests)."""
    override = os.environ.get("MASHUP_SETTINGS_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "mashup-engine"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mashup-engine"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mashup-engine"


def settings_path() -> Path:
    """Absolute path to settings.json (may not exist yet)."""
    return _settings_dir() / "settings.json"


def _load_settings() -> dict:
    """Read settings.json, tolerating a missing or corrupt file."""
    p = settings_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        pass
    return {}


_SETTINGS = _load_settings()


def _resolve(env_key: str, settings_key: str, default):
    """Return (value, source) where source is 'env' | 'settings' | 'default'."""
    env_val = os.environ.get(env_key)
    if env_val not in (None, ""):
        return env_val, "env"
    file_val = _SETTINGS.get(settings_key)
    if file_val not in (None, ""):
        return file_val, "settings"
    return default, "default"


_audio_val, AUDIO_ROOT_SOURCE = _resolve(
    "MASHUP_AUDIO_ROOT", "audio_root", str(BASE_DIR / "audio"))
AUDIO_DIR         = Path(_audio_val)
RAW_DIR           = AUDIO_DIR / "full_song"
STEMS_DIR         = AUDIO_DIR / "stems"      # stems/separate.py: vocals, instrumentals, drums, bass, other
PREVIEWS_DIR      = AUDIO_DIR / "previews"   # rendered Studio mixdowns (render/mixdown.py)
HOOKS_DIR         = AUDIO_DIR / "hooks"      # 16-bar preview clips (api/workers/hook_worker.py)

# SQLite file path — must not be BASE_DIR / "database" (that is the Python package directory).
_db_val, DB_PATH_SOURCE = _resolve(
    "MASHUP_DB_PATH", "db_path", str(BASE_DIR / "mashup.db"))
DB_PATH = Path(_db_val)

# Engine artifacts (page snapshots, training datasets, learned models) live next
# to the database by default, so a single volume mount persists everything.
_data_val, _ = _resolve("MASHUP_DATA_DIR", "data_dir", str(DB_PATH.parent))
DATA_DIR      = Path(_data_val)
DATASETS_DIR  = DATA_DIR / "datasets"    # exported training sets (Phase 4)
MODELS_DIR    = DATA_DIR / "models"      # learned pairwise models (Phase 5)

# `configured` is True once the user has explicitly chosen an audio library
# location (via env or the wizard). When False, App.jsx shows the Setup Wizard.
CONFIGURED = AUDIO_ROOT_SOURCE in ("env", "settings")


def ensure_dirs() -> None:
    """Create all working directories. Called at import (best-effort) and again
    after the wizard saves settings, so a freshly-chosen library folder exists."""
    for d in (RAW_DIR, STEMS_DIR, PREVIEWS_DIR, HOOKS_DIR,
              DATASETS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# Best-effort at import: a wizard-chosen path on an unmounted volume shouldn't
# crash `import config`; get_conn() and the workers mkdir lazily on real use.
try:
    ensure_dirs()
except OSError:
    pass


def save_settings(new: dict) -> Path:
    """Merge `new` into settings.json and write it. Empty/None values are
    ignored so a partial save (e.g. just audio_root) doesn't clobber other keys.
    Returns the settings.json path. Caller should treat changes as needing a
    process restart (constants are bound at import)."""
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    current = _load_settings()
    for k, v in new.items():
        if v not in (None, ""):
            current[k] = v
    p.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return p


def settings_provenance() -> dict:
    """Report each resolved setting's value and where it came from, so the
    Settings UI can show 'set by environment' vs 'editable'. `paths` carries
    the derived working directories (downloads/stems/previews/data) so the
    Import tab can show the user exactly where their data lives."""
    return {
        "audio_root": {"value": str(AUDIO_DIR), "source": AUDIO_ROOT_SOURCE},
        "db_path":    {"value": str(DB_PATH),   "source": DB_PATH_SOURCE},
        "pipeline_workers": {"value": PIPELINE_WORKERS, "source": PIPELINE_WORKERS_SOURCE},
        # Live-read: these apply without a restart.
        "essentia_key_profile": {
            "value": current_essentia_key_profile(),
            "source": "env" if os.environ.get("MASHUP_ESSENTIA_KEY_PROFILE") else "settings"},
        "essentia_rhythm_method": {
            "value": current_essentia_rhythm_method(),
            "source": "env" if os.environ.get("MASHUP_ESSENTIA_RHYTHM") else "settings"},
        "analysis_cache": {"value": current_analysis_cache(),
                           "source": "env" if os.environ.get("MASHUP_ANALYSIS_CACHE") else "settings"},
        "allin1_device": {"value": current_allin1_device(),
                          "source": "env" if os.environ.get("MASHUP_ALLIN1_DEVICE") else "settings"},
        "allin1_model": {"value": current_allin1_model(),
                         "source": "env" if os.environ.get("MASHUP_ALLIN1_MODEL") else "settings"},
        "keep_analysis_stems": {
            "value": current_keep_analysis_stems(),
            "source": "env" if os.environ.get("MASHUP_KEEP_ANALYSIS_STEMS") else "settings"},
        # Scoring knobs. `source` is "env" only when pinned by an environment
        # variable, in which case the UI must show the control as locked rather
        # than letting the user save a value that will be ignored.
        **{name: {"value": current_float(name),
                  "source": "env" if os.environ.get(spec[0]) else "settings"}
           for name, spec in _TUNABLE_FLOATS.items()},
        **{name: {"value": current_int(name),
                  "source": "env" if os.environ.get(spec[0]) else "settings"}
           for name, spec in _TUNABLE_INTS.items()},
        "match_weights": {"value": current_match_weights(), "source": "settings"},
        "section_weights": {"value": current_section_weights(), "source": "settings"},
        # Presence only, never the value — this response goes to the browser.
        "firecrawl_api_key": {"value": bool(current_firecrawl_api_key()),
                              "source": current_firecrawl_key_source()},
        "stem_format": {"value": STEM_FORMAT, "source": "code"},
        "configured": CONFIGURED,
        "settings_path": str(settings_path()),
        "paths": {
            "downloads":     str(RAW_DIR),
            "stems":         str(STEMS_DIR),
            "previews":      str(PREVIEWS_DIR),
            "data_dir":      str(DATA_DIR),
        },
    }


# ── Download ─────────────────────────────────────────────────────────────────
# yt-dlp format string — prefers best dedicated audio, else combined (then ffmpeg extracts mp3)
YTDLP_FORMAT = "bestaudio/best"
# Last-resort when bestaudio yields no matching stream for a given YouTube player client
YTDLP_FORMAT_FALLBACK = "ba/b"
YTDLP_POSTARGS = [
    "--extract-audio",
    "--audio-format", "mp3",
    "--audio-quality", "0",   # 0 = best VBR
]

# ── Stem separation ───────────────────────────────────────────────────────────
# htdemucs: Hybrid Transformer Demucs, four sources (drums/bass/other/vocals).
DEMUCS_MODEL = "htdemucs"
# How much consecutive Demucs windows overlap (demucs' own default). Lower is
# faster (0.1 ≈ 15 % less work) at a small separation-quality cost: listen to an
# A/B before moving it. Live (demucs_overlap).
DEMUCS_OVERLAP = 0.25

# Stems are written as FLAC: lossless, and roughly half the size of WAV. At
# ~900 tracks x 4 stems that is the difference between ~160 GB and ~80 GB.
STEM_FORMAT = "flac"

# The four Demucs sources, and the three that sum to the instrumental.
DEMUCS_SOURCES = ("drums", "bass", "other", "vocals")
INSTRUMENTAL_SOURCES = ("drums", "bass", "other")


# ── Analysis ──────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050
HOP_LENGTH       = 512
N_MFCC           = 13      # MFCC coefficients stored per track
BEAT_TRIM_SECS   = None    # None = analyse the FULL track (best match quality —
                           # BPM/key from only the intro is unreliable).
                           # Set to e.g. 30 to trade accuracy for speed.

# Decoded signals kept in memory (analysis/decode.py). One track touches 3-5
# files (mix + stems) and the analysis pool runs 2 tracks at once; 6 keeps a
# whole analysis → structure → quality pass decoding each file once. A 4-minute
# file is ~21 MB at 22.05 kHz mono.
DECODE_CACHE_SIZE = 6


def current_analysis_cache() -> bool:
    """Whether analysis reuses stored feature groups (analysis/cache.py).

    On by default: a group is recomputed only when its audio, its version or
    its parameters changed. Off (``MASHUP_ANALYSIS_CACHE=0`` or settings.json
    ``"analysis_cache": false``) recomputes everything, which is only ever
    worth it to rule the cache out while chasing a bug. Re-read live."""
    env = os.environ.get("MASHUP_ANALYSIS_CACHE")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in ("0", "false", "off", "no")
    val = _load_settings().get("analysis_cache")
    if isinstance(val, bool):
        return val
    if isinstance(val, str) and val.strip():
        # POST /api/settings stores "off": save_settings drops falsy values.
        return val.strip().lower() not in ("0", "false", "off", "no")
    return True


ESSENTIA_RHYTHM_METHODS = ("degara", "multifeature")
ESSENTIA_KEY_PROFILES = ("edma", "edmm", "bgate", "braw", "krumhansl", "temperley",
                         "shaath", "diatonic", "noland", "tonictriad",
                         "temperley2005", "thpcp", "gomez", "weichai")


def _live_choice(env_name: str, key: str, allowed: tuple, default: str) -> str:
    env = (os.environ.get(env_name) or "").strip().lower()
    if env in allowed:
        return env
    val = str(_load_settings().get(key) or "").strip().lower()
    return val if val in allowed else default


def current_essentia_key_profile() -> str:
    """The KeyExtractor profile whose answer is THE key (the others vote on
    confidence). edma — trained on electronic dance music — until the Phase 0
    benchmark on your own tracks says otherwise."""
    return _live_choice("MASHUP_ESSENTIA_KEY_PROFILE", "essentia_key_profile",
                        ESSENTIA_KEY_PROFILES, "edma")


def current_essentia_rhythm_method() -> str:
    """RhythmExtractor2013 method. degara costs ~¼ of multifeature (Phase 0)."""
    return _live_choice("MASHUP_ESSENTIA_RHYTHM", "essentia_rhythm_method",
                        ESSENTIA_RHYTHM_METHODS, "degara")


# ── The one pipeline (readme §2) ─────────────────────────────────────────────
# Every track is separated once by htdemucs into four sources; the vocal stem
# and the summed instrumental are what playback reads, the four sources are what
# allin1 reads. allin1 cuts the sections; the Essentia novelty cut runs only
# when allin1 produced none (failed, skipped or found nothing).
PIPELINE = "unified_4stem"
ENGINE = "allin1"
# One fold of allin1's 8-fold ensemble: ~1/8 of the inference (the ensemble's
# median 468 s a track was 79 % of the allin1 stage). harmonix-all — the
# ensemble — stays one setting away (allin1_model), and
# scripts/bench_allin1_models.py measures what the fold gives up against it.
ALLIN1_MODEL = "harmonix-fold0"
# What allin1 1.1.0 publishes: the ensemble and its eight individual folds.
ALLIN1_MODELS = ("harmonix-all",) + tuple(f"harmonix-fold{i}" for i in range(8))
ALLIN1_DEVICES = ("cpu", "cuda")


def current_allin1_model() -> str:
    """Which allin1 checkpoint to run, re-read live.

    The default is one fold (``harmonix-fold0``). ``harmonix-all`` is the
    8-fold ensemble — measured at a median 468 s of inference a track (27
    tracks, 2026-10-24/25), roughly 8x a fold's; set it to trade time for the
    ensemble's agreement. scripts/bench_allin1_models.py measures the
    difference (downbeat agreement, boundaries, labels), which matters because
    the 25 ms timing gate rests on these downbeats. Validated against
    ALLIN1_MODELS, so a name allin1 does not ship is refused by the settings
    API rather than failing ten minutes in."""
    return _live_choice("MASHUP_ALLIN1_MODEL", "allin1_model",
                        ALLIN1_MODELS, ALLIN1_MODEL)


def current_separation_pipeline() -> str:
    """The separation pipeline: always ``unified_4stem``. Kept as a function
    because ~20 readers ask; there is no other pipeline to choose."""
    return PIPELINE


def current_segmentation_engine() -> str:
    """The section engine: always ``allin1``, with the Essentia novelty cut as
    the per-track fallback (pipeline_worker._wants_essentia_cut)."""
    return ENGINE


def current_allin1_device() -> str:
    """cpu unless set. The Docker image ships CPU torch; cuda falls back to cpu
    in analysis/allin1_runner when torch sees no GPU."""
    return _live_choice("MASHUP_ALLIN1_DEVICE", "allin1_device", ALLIN1_DEVICES, "cpu")


def current_keep_analysis_stems() -> bool:
    """Keep the unified pipeline's drums/bass/other after analysis (default) —
    what lets allin1 re-run without separating again. Off deletes them once
    allin1 has stored its result; the playback stems are never deleted."""
    raw = os.environ.get("MASHUP_KEEP_ANALYSIS_STEMS")
    if raw is None:
        raw = _load_settings().get("keep_analysis_stems")
    if raw is None or str(raw).strip() == "":
        return True
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("0", "false", "off", "no")


def current_keep_frames() -> bool:
    """Keep the per-frame analysis matrices (analysis/timeseries.py) as .npz
    sidecars. On, because they are what makes a re-cut, the Segments screen and
    phrase matching free; off throws the arrays away after the sections are
    measured and every later re-cut decodes and recomputes them."""
    raw = os.environ.get("MASHUP_KEEP_ANALYSIS_FRAMES")
    if raw is None:
        raw = _load_settings().get("keep_analysis_frames")
    if raw is None or str(raw).strip() == "":
        return True
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("0", "false", "off", "no")

# ── Structure detection (sections: intro/verse/chorus/drop/…) ─────────────────
SECTION_MIN_LEN_SECS  = 12.0   # minimum section length
SECTION_MAX_COUNT     = 14     # cap on sections per track
SECTION_SIM_THRESHOLD = 0.92   # chroma cosine sim above which two sections
                               # count as repeats of each other (chorus finder)

# ── Matching ──────────────────────────────────────────────────────────────────
# Weights used in the composite similarity score (must sum to 1.0)
MATCH_WEIGHTS = {
    "bpm_score":       0.22,
    "key_score":       0.26,
    "energy_score":    0.17,
    "timbre_score":    0.20,
    # Phase D — do the two sides stay out of each other's way in the spectrum?
    # The other four all measure similarity; a mid-heavy vocal over a mid-dense
    # bed can score well on every one of them and still be inaudible. The other
    # weights are scaled down proportionally so the total still sums to 1.
    "collision_score": 0.15,
}

# A top stem below this quality is not offered at all, however well it matches:
# a bleeding, smeared acapella near the top of the list is what stops the list
# being trusted. NULL quality (analysed before Phase D) counts as 1.0, so an
# existing library is unaffected until it is re-analysed.
STEM_QUALITY_MIN = 0.35

# Candidate gate — pairs that don't pass are never scored.
#
# BPM: maximum difference allowed (accounts for halftime/doubletime).
BPM_MAX_DIFF   = 16.0   # e.g. 120 BPM pairs with anything 104–136 (or half/double)
#
# Key: minimum Camelot score to qualify (0.0–1.0). DEFAULTS OFF (P1.1).
#
# This used to be 0.55, which kept only same / adjacent / relative-major-minor /
# two-steps — 6 of the 24 Camelot codes, so roughly three quarters of the
# library was unreachable from any given track. But transposing a bed by a
# semitone or two is an ordinary move, and matcher/effort.py ALREADY prices it
# (`pitch_cost`, weighted 0.30, ramping to full cost at ±6). Keeping the gate as
# well charged for the same thing twice: the pair was deleted before scoring,
# and had it survived it would have been demoted anyway.
#
# Worse, the gate ran on a key estimated from an isolated acapella (see P0.3),
# and it ran BEFORE Phase E measures the harmony that actually decides the
# question. Pairs were dying on the least reliable number in the database with
# no appeal.
#
# The gate's job is to bound the matrix, not to express taste. Tempo does that;
# key is left to the scorer. Set this above 0 to get the old behaviour, or use
# the Tight preset in the UI.
KEY_MIN_SCORE  = 0.0

# ── Mix training-data quality gate ────────────────────────────────────────────
# A mix-track auto-linked by yt-dlp search (resolve_status='auto') only counts as
# a trusted ground-truth link — i.e. eligible to become a training positive — when
# its fuzzy search score clears this bar AND its resolved audio is a full track (not
# a ~30s SoundCloud Go+ preview). Manual/ingested links are always trusted. Below
# the bar the link is still usable for ingest; it's just flagged for quick review.
AUTO_LINK_MIN_SCORE    = 0.72
AUTO_LINK_MIN_DURATION = 60.0   # seconds; guards against preview-length mislinks
# Fraction of the wanted artist's words that must appear in the hit's title or
# uploader name. A title-only match with an unrelated artist is the classic
# mislink ("Take On The World" by You Me At Six for a jeonghyeon track), and it
# can score well on title alone — so artist agreement is checked separately
# rather than being averaged away.
AUTO_LINK_MIN_ARTIST   = 0.5

# ── SoundCloud ────────────────────────────────────────────────────────────────
# Reading SoundCloud needs no credentials: ingest/soundcloud_api.py scrapes a
# working client_id out of SoundCloud's own public JS bundles, and both the
# mixes auto-resolver and the Discovery browse layer use it. The app never
# logs in and never writes to SoundCloud.

# ── Firecrawl (1001tracklists structured scrape) ──────────────────────────────
# Firecrawl's hosted stealth proxy renders + bypasses the Cloudflare Turnstile
# that blocks a plain urllib GET of 1001tracklists. Needs an API key; ~9 credits
# per page, so per-track link scraping is on-demand only (not a 216-page bulk).
_fc_val, FIRECRAWL_KEY_SOURCE = _resolve("FIRECRAWL_API_KEY", "firecrawl_api_key", "")
FIRECRAWL_API_KEY   = _fc_val
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"


def current_firecrawl_api_key() -> str:
    """The key to use RIGHT NOW. The Mixes tab saves it to settings.json when an
    import 501s and retries straight away, so callers must not read the
    import-time constant above. A non-empty env var still wins (Docker .env
    pinning); compose passes "" when .env has none, which counts as unset."""
    env = (os.environ.get("FIRECRAWL_API_KEY") or "").strip()
    if env:
        return env
    return str(_load_settings().get("firecrawl_api_key") or "").strip()


def current_firecrawl_key_source() -> str:
    """'env' | 'settings' | 'default', by the same lookup as the key itself."""
    if (os.environ.get("FIRECRAWL_API_KEY") or "").strip():
        return "env"
    if str(_load_settings().get("firecrawl_api_key") or "").strip():
        return "settings"
    return "default"

# ── Background processing ─────────────────────────────────────────────────────
# Number of worker threads that drain the ingest→download→stems→analysis→structure
# pipeline queue. Default 1: stem separation (Demucs) is CPU/GPU heavy, so running
# several tracks at once thrashes the machine. Raise on a beefy box via env/settings.
_workers_val, PIPELINE_WORKERS_SOURCE = _resolve(
    "MASHUP_PIPELINE_WORKERS", "pipeline_workers", "1")
try:
    PIPELINE_WORKERS = max(1, int(_workers_val))
except (TypeError, ValueError):
    PIPELINE_WORKERS = 1


def _resolve_int(env_key: str, settings_key: str, default: int) -> int:
    val, _ = _resolve(env_key, settings_key, str(default))
    try:
        return max(1, int(val))
    except (TypeError, ValueError):
        return default


# Per-stage concurrency for the pipeline queue. Downloads are network-bound
# (cheap to parallelise); Demucs is CPU-heavy (stays at PIPELINE_WORKERS, i.e. 1
# by default); librosa analysis sits in between. ENRICH_WORKERS bounds the
# parallel per-track metadata fetches during playlist preview/ingest.
DOWNLOAD_WORKERS = _resolve_int("MASHUP_DOWNLOAD_WORKERS", "download_workers", 4)
STEM_WORKERS     = _resolve_int("MASHUP_STEM_WORKERS", "stem_workers", PIPELINE_WORKERS)
ANALYSIS_WORKERS = _resolve_int("MASHUP_ANALYSIS_WORKERS", "analysis_workers", 2)
ENRICH_WORKERS   = _resolve_int("MASHUP_ENRICH_WORKERS", "enrich_workers", 5)
# The quick tier (readme §9, phase 3): the full mix analysed and segmented right
# after download, while the track waits for Demucs. Threads, not processes:
# librosa/numpy release the GIL (3 analyses on 3 threads ran 2.2× faster than
# in sequence on 4 cores, measured), so a process pool would buy little and
# cost the shared decode cache.
QUICK_WORKERS    = _resolve_int("MASHUP_QUICK_WORKERS", "quick_workers", 2)
# allin1 (beats, downbeats, labelled sections): one torch model held per worker.
ALLIN1_WORKERS   = _resolve_int("MASHUP_ALLIN1_WORKERS", "allin1_workers", 1)
# Threads a Demucs run may use. Left alone, torch takes every core and the
# quick tier stalls behind it; one core is kept back for it by default.
DEMUCS_THREADS   = _resolve_int("MASHUP_DEMUCS_THREADS", "demucs_threads",
                                max(1, (os.cpu_count() or 2) - 1))
# By default allin1 shares the stems gate, so it never runs beside Demucs (both
# want every core on CPU). On, it takes its own gate and ALLIN1_THREADS: the
# next song's separation overlaps this song's allin1. Worth it only with the
# threads split (e.g. half the cores each) and the memory for both (~3.3 GB
# for allin1 plus Demucs's own) — off until measured on the machine.
ALLIN1_PARALLEL  = str(_resolve("MASHUP_ALLIN1_PARALLEL", "allin1_parallel", "")[0]
                       ).strip().lower() in ("1", "true", "on", "yes")
ALLIN1_THREADS   = _resolve_int("MASHUP_ALLIN1_THREADS", "allin1_threads",
                                DEMUCS_THREADS)

# Selecting a track moves it and this many of its likeliest unprocessed partners
# (tempo + key against its quick-tier analysis) ahead of the import queue.
PREFETCH_PARTNERS = _resolve_int("MASHUP_PREFETCH_PARTNERS", "prefetch_partners", 5)

# Queue priorities (api/queue_runner.py): lower runs first, FIFO within one.
PRIORITY_USER     = 0     # a button pressed on one track
PRIORITY_PREFETCH = 5     # the track you are looking at and its best partners
PRIORITY_INGEST   = 10    # tracks arriving from an import
PRIORITY_BACKFILL = 30    # bulk re-processing of the library

# ── Shared helpers ────────────────────────────────────────────────────────────

def sanitize_filename_chars(name: str) -> str:
    """Replace characters not safe for a filename/foldername with underscores.
    Callers handle their own truncation/collapsing on top of this."""
    return re.sub(r"[^\w]", "_", name or "")


def format_duration(secs) -> str:
    """Seconds → 'm:ss' (or 'h:mm:ss'). Empty string for missing/zero."""
    if not secs or secs <= 0:
        return ""
    s = int(round(secs))
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

# ── Effort penalty (Phase C) ──────────────────────────────────────────────────
# The four MATCH_WEIGHTS sub-scores all measure similarity; none of them
# measures what a mashup COSTS to build. score_total is discounted by
# EFFORT_WEIGHT × effort, where effort is 0 (same tempo, same key, confident
# grid) to 1 (maximum stretch, maximum transpose, unusable grid).
#
# 0.25 is deliberately modest: it lets a free-to-build pair overtake one that is
# a few points better on paper, without letting convenience beat a genuinely
# stronger match. Set to 0.0 to rank on similarity alone.
EFFORT_WEIGHT = 0.25

# ── Section-pair ranking (E.3) ────────────────────────────────────────────────
# The candidate row is the SECTION PAIR now, so how well those two sections fit
# each other is part of what the row IS, not a post-hoc annotation on it. Before
# E.3 this was deliberately excluded from score_total (T3.3's "selects and
# describes; it must not re-rank") — correct while the row was a song pair and
# the section was chosen afterwards, wrong once the section is the unit.
SECTION_WEIGHT = 0.25

# ── The vacated slot and Drop-in (readme §5.7) ────────────────────────────────
# How much of a placed pairing's section fit is the slot term: how well the
# incoming vocal replaces the vocal the bed section used to carry (occupancy,
# phrase starts, register). Moderate: enough that a bed's sung chorus beats
# its drop for a vocal that fits it, not enough to bury a good instrumental
# bed. 0 scores exactly as before the term existed.
SLOT_WEIGHT = 0.20
# What a Drop-in pair is lifted by in the listing's order (never in the stored
# total). Bounded, and small against the 0.05-0.10 a tier spans.
DROPIN_BONUS = 0.02
# The Drop-in rule's fixed thresholds (owner-approved): tempo within this
# share after the half/double fold, the slot at or above this library
# percentile, the bed moved by at most this many semitones.
DROPIN_TEMPO_PCT = 3.0
DROPIN_SLOT_PERCENTILE = 0.70
DROPIN_MAX_SHIFT_ST = 1

# ── The learning loop (readme §5.8) ──────────────────────────────────────────
# The shrinkage blend: score_total = (1 - a)·rule + a·model, with
# a = n_strong / (n_strong + LEARN_K), capped at LEARN_ALPHA_MAX, where
# n_strong is how many of your verdicts the active model was trained on. No
# verdicts, no model: a = 0 and every total is the rule's, to the bit.
LEARN_K = 40.0
LEARN_ALPHA_MAX = 0.8
# Weak labels from what you DO with a pair (pair_events), as sample weights
# beside a verdict's 1.0. Conservative: a hundred exports weigh about what
# thirty verdicts do. A verdict on the same pairing overrides them all.
WEAK_W_BUILD = 0.30        # open_studio / render / fl_export / export → good
WEAK_W_LOOP = 0.15         # a loop_play of LOOP_POSITIVE_SECS or more → good
LOOP_POSITIVE_SECS = 20.0
WEAK_W_SKIP = 0.10         # skip → bad
WEAK_W_HIDE = 0.30         # hide → bad
WEAK_W_SHOWN = 0.03        # shown in the top 5, nothing else done → bad
# Retrain when this many new verdicts, or this many new non-`shown` events,
# arrived since the last train — and only while the pipeline queues are idle.
RETRAIN_MIN_STRONG = 10
RETRAIN_MIN_WEAK = 200
AUTO_RETRAIN = True
# One dock slot in this many shows an uncertain pair (a high rule score the
# model is unsure about), labelled as such. 0 = never.
EXPLORE_EVERY = 8

# Weights inside score_section itself (phrase / rhythm / structure,
# plus the label/duration/voice terms that were already there).
#
# The three new ones ship at ZERO on purpose. They read per-section columns that
# P2.1 only just added, so until a library has been re-analysed they would be
# scoring on missing data — and flipping them on before the backfill would move
# every ranking in an existing library for reasons the user cannot see. Raise
# them (Settings → Tuning, or here) once "Re-analyse" reports nothing stale.
#
# These stay at zero as the SHIPPED default even after a backfill, because the
# right values are a property of a library, not of the code — settings.json is
# where a tuned set belongs. Measured once on a backfilled 30-track library
# (1197 section pairs), for whoever tunes next:
#
#   phrase     stdev 0.31 over 362 distinct values, rho +0.37 vs duration.
#              Real, independent signal — the one worth weighting.
#   rhythm     stdev 0.0033, range 0.972-1.000. NOT a missing-data problem
#              (0% at the neutral fallback): bar-profile cosine saturates
#              because 4/4 dance records all have the same bar-level onset
#              shape. Weighting it rescales the list rather than reordering it.
#   structure  rho +0.88 with `label`. Both are functions of the same two
#              section labels, so weighting both counts one signal twice.
#              Any weight it gets should come OUT of label's budget.
SECTION_WEIGHTS = {
    "label":     0.40,
    "duration":  0.35,
    "voice":     0.25,
    "phrase":    0.0,
    "rhythm":    0.0,
    "structure": 0.0,
}

# How many section pairs one song pair may contribute. Two tracks with six
# usable sections each would otherwise produce 36 rows and drown everything
# else; three is enough to show that a pair works in more than one place.
MAX_SECTION_PAIRS_PER_SONG_PAIR = 3

# Hard ceiling on persisted candidate rows, per combo type. Dropping the key
# gate (P1.1) multiplies the surviving pair count by roughly four, and each
# survivor contributes up to MAX_SECTION_PAIRS_PER_SONG_PAIR rows. Without a
# ceiling a large library turns a re-score into hundreds of megabytes of
# in-flight dicts and a candidates table nobody will ever read past row 500.
#
# The cap keeps the BEST rows by score_total, so raising the gate can only ever
# add ideas at the top of the list; it cannot push a good pair out in favour of
# a worse one.
MAX_CANDIDATE_ROWS = 200_000

# ── Phrase placement and the timing gate (allin1 engine, readme §5.7) ─────────
# After the bed is conformed to the vocal and placed so the vocal's phrase
# downbeat sits on the bed's landing downbeat, every later downbeat in the
# overlap must stay within TIMING_TOL_MS of its partner, or the pair is dropped
# (a hard gate, like BPM). TIMING_WEIGHT blends the timing score into the
# section fit in allin1 mode only; essentia mode is unchanged.
TIMING_TOL_MS = 25.0
TIMING_WEIGHT = 0.30
# Allowed vocal-bars : bed-bars ratios. 1 = 16 over 16; 2 = the bed's 16 under
# an 8-bar vocal repeated; 0.5 = an 8-bar bed looped under 16 bars of vocal.
TIMING_PHRASE_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
# Phrase detection on the vocal stem: a sung onset after at least this much
# quiet (or one beat, whichever is longer) starts a phrase; an onset more than
# PHRASE_PICKUP_MAX_BEATS from its nearest downbeat is not a phrase start.
PHRASE_GAP_SECS = 0.6
PHRASE_MIN_SUNG_SECS = 0.4
PHRASE_PICKUP_MAX_BEATS = 2.0
# allin1 resource limits: longer tracks (DJ sets) are skipped with a reason,
# and a run whose resident memory would pass the budget refuses rather than OOM.
ALLIN1_MAX_SECS = 900.0
ALLIN1_MEM_BUDGET_MB = 6000


# ── Live-read scoring knobs (Settings UI) ─────────────────────────────────────
# The constants above bind at import. score_all_pairs re-imports them per call,
# which is enough for a restart but not for a settings save — and every one of
# these is a knob you want to turn, re-score, and hear the difference, not
# restart the server over. These read settings.json each time, with the same
# precedence as the rest: env var wins (Docker/CI pinning), then settings.json,
# then the module constant.
#
# The weights are exposed because they ARE the ranking. Someone who cares more
# about tempo than timbre should be able to say so without editing Python.

# name -> (env var, module constant, low, high)
_TUNABLE_FLOATS = {
    "effort_weight":     ("MASHUP_EFFORT_WEIGHT", "EFFORT_WEIGHT", 0.0, 1.0),
    "section_weight":    ("MASHUP_SECTION_WEIGHT", "SECTION_WEIGHT", 0.0, 1.0),
    "stem_quality_min":  ("MASHUP_STEM_QUALITY_MIN", "STEM_QUALITY_MIN", 0.0, 1.0),
    "bpm_max_diff":      ("MASHUP_BPM_MAX_DIFF", "BPM_MAX_DIFF", 1.0, 60.0),
    "key_min_score":     ("MASHUP_KEY_MIN_SCORE", "KEY_MIN_SCORE", 0.0, 1.0),
    "timing_tol_ms":     ("MASHUP_TIMING_TOL_MS", "TIMING_TOL_MS", 5.0, 250.0),
    "timing_weight":     ("MASHUP_TIMING_WEIGHT", "TIMING_WEIGHT", 0.0, 1.0),
    "slot_weight":       ("MASHUP_SLOT_WEIGHT", "SLOT_WEIGHT", 0.0, 1.0),
    "dropin_bonus":      ("MASHUP_DROPIN_BONUS", "DROPIN_BONUS", 0.0, 0.1),
    "phrase_gap_secs":   ("MASHUP_PHRASE_GAP_SECS", "PHRASE_GAP_SECS", 0.1, 4.0),
    "allin1_max_secs":   ("MASHUP_ALLIN1_MAX_SECS", "ALLIN1_MAX_SECS", 30.0, 7200.0),
    "demucs_overlap":    ("MASHUP_DEMUCS_OVERLAP", "DEMUCS_OVERLAP", 0.0, 0.9),
    "learn_k":           ("MASHUP_LEARN_K", "LEARN_K", 1.0, 10_000.0),
    "learn_alpha_max":   ("MASHUP_LEARN_ALPHA_MAX", "LEARN_ALPHA_MAX", 0.0, 1.0),
    "weak_w_build":      ("MASHUP_WEAK_W_BUILD", "WEAK_W_BUILD", 0.0, 1.0),
    "weak_w_loop":       ("MASHUP_WEAK_W_LOOP", "WEAK_W_LOOP", 0.0, 1.0),
    "loop_positive_secs": ("MASHUP_LOOP_POSITIVE_SECS", "LOOP_POSITIVE_SECS", 1.0, 600.0),
    "weak_w_skip":       ("MASHUP_WEAK_W_SKIP", "WEAK_W_SKIP", 0.0, 1.0),
    "weak_w_hide":       ("MASHUP_WEAK_W_HIDE", "WEAK_W_HIDE", 0.0, 1.0),
    "weak_w_shown":      ("MASHUP_WEAK_W_SHOWN", "WEAK_W_SHOWN", 0.0, 1.0),
}
# However the row cap falls, every track keeps the best row of this many of
# its best partners on each side (as the vocal, and as the bed). Without it
# the cap keeps the library's best rows overall, and at a few thousand tracks
# most tracks' own best beds are not among them: their scoped dock is empty.
RESERVE_PARTNERS = 8

_TUNABLE_INTS = {
    "reserve_partners": ("MASHUP_RESERVE_PARTNERS", "RESERVE_PARTNERS", 0, 50),
    "allin1_mem_budget_mb": ("MASHUP_ALLIN1_MEM_BUDGET_MB", "ALLIN1_MEM_BUDGET_MB",
                             500, 256_000),
    "max_section_pairs": ("MASHUP_MAX_SECTION_PAIRS",
                          "MAX_SECTION_PAIRS_PER_SONG_PAIR", 1, 8),
    "max_candidate_rows": ("MASHUP_MAX_CANDIDATE_ROWS",
                           "MAX_CANDIDATE_ROWS", 1, 5_000_000),
    "retrain_min_strong": ("MASHUP_RETRAIN_MIN_STRONG", "RETRAIN_MIN_STRONG", 1, 10_000),
    "retrain_min_weak":  ("MASHUP_RETRAIN_MIN_WEAK", "RETRAIN_MIN_WEAK", 1, 1_000_000),
    "explore_every":     ("MASHUP_EXPLORE_EVERY", "EXPLORE_EVERY", 0, 100),
}


def current_auto_retrain() -> bool:
    """Whether the learning loop retrains on its own (readme §5.8). Live-read;
    'off' / '0' / 'false' in settings.json or MASHUP_AUTO_RETRAIN turn it off."""
    raw = os.environ.get("MASHUP_AUTO_RETRAIN")
    if raw is None:
        raw = _load_settings().get("auto_retrain")
    if raw is None:
        return AUTO_RETRAIN
    return str(raw).strip().lower() not in ("0", "off", "false", "no")

# Sub-score weights, tuned as a group. Stored as a dict in settings.json.
_WEIGHT_KEYS = ("bpm_score", "key_score", "energy_score", "timbre_score",
                "collision_score")

_SECTION_WEIGHT_KEYS = ("label", "duration", "voice",
                        "phrase", "rhythm", "structure")


def current_section_weights() -> dict:
    """The six score_section weights, normalised to sum to 1.

    Same contract as current_match_weights: normalised rather than validated, so
    dragging one slider does not silently rescale every section score in the
    library. An all-zero saved set falls back to the defaults."""
    saved = _load_settings().get("section_weights")
    out = dict(SECTION_WEIGHTS)
    if isinstance(saved, dict):
        for key in _SECTION_WEIGHT_KEYS:
            try:
                val = float(saved[key])
            except (KeyError, TypeError, ValueError):
                continue
            out[key] = max(0.0, val)
    total = sum(out.values())
    if total <= 0:
        out = dict(SECTION_WEIGHTS)
        total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def current_float(name: str) -> float:
    env_key, const_name, lo, hi = _TUNABLE_FLOATS[name]
    raw = os.environ.get(env_key)
    if raw is None:
        raw = _load_settings().get(name)
    if raw is not None:
        try:
            return _clamp(float(raw), lo, hi)
        except (TypeError, ValueError):
            pass
    return globals()[const_name]


def current_int(name: str) -> int:
    env_key, const_name, lo, hi = _TUNABLE_INTS[name]
    raw = os.environ.get(env_key)
    if raw is None:
        raw = _load_settings().get(name)
    if raw is not None:
        try:
            return int(_clamp(int(raw), lo, hi))
        except (TypeError, ValueError):
            pass
    return globals()[const_name]


def current_match_weights(combo_type: Optional[str] = None) -> dict:
    """The five sub-score weights, normalised to sum to 1.

    Normalised rather than validated: a user dragging five sliders should not
    have to make them add up, and an un-normalised set would silently rescale
    every score in the library so the Min-match slider stopped meaning anything.
    A saved set that is all zeros falls back to the defaults rather than making
    every pair score 0.

    `combo_type` selects the per-combo adjustment below. Omit it for the generic
    weights (a one-off composite_score call).
    """
    saved = _load_settings().get("match_weights")
    if not isinstance(saved, dict):
        out = dict(MATCH_WEIGHTS)
    else:
        out = {}
        for k in _WEIGHT_KEYS:
            try:
                out[k] = max(0.0, float(saved.get(k, MATCH_WEIGHTS[k])))
            except (TypeError, ValueError):
                out[k] = MATCH_WEIGHTS[k]
        total = sum(out.values())
        if total <= 0:
            out = dict(MATCH_WEIGHTS)
        else:
            out = {k: v / total for k, v in out.items()}
    return _for_combo(out, combo_type)


# Timbre similarity asks "do these two sound like the same record". For an
# instrumental-over-instrumental blend that is the right question: the two beds
# have to cohere or the result sounds like a crossfade between two songs.
#
# For a VOCAL over a bed it is close to the wrong question, and arguably
# backwards. What decides whether you hear the vocal is whether the bed leaves
# room for it — which is collision_score, measured on the band occupancy of the
# two stems. Rewarding timbral sameness on top of that pushes the head of the
# ranked list towards the safest, most homogeneous pairings in the library,
# which is the opposite of what the whole thing is for; and it fights the
# `surprise_timbre` contrast term directly, so the two mostly cancel and leave
# variance behind.
#
# So on the vocal path timbre's weight moves to collision. The sub-score is
# still computed, stored and displayed — it is informative, it just should not
# be pulling the ranking towards sameness — and the model still receives both
# `timbre_score` and `surprise_timbre` as columns, so it can learn where this
# user's taste actually sits.
def _for_combo(weights: dict, combo_type: Optional[str]) -> dict:
    if combo_type != "vocal_over_instrumental":
        return weights
    out = dict(weights)
    out["collision_score"] = out.get("collision_score", 0.0) \
        + out.get("timbre_score", 0.0)
    out["timbre_score"] = 0.0
    return out


def current_scoring_settings() -> dict:
    """Everything score_all_pairs needs, read fresh. One call, so a scoring run
    is internally consistent even if the file changes mid-run."""
    return {
        "match_weights": current_match_weights(),
        # The vocal path redistributes timbre onto collision — see _for_combo.
        "match_weights_vocal": current_match_weights("vocal_over_instrumental"),
        "effort_weight": current_float("effort_weight"),
        "section_weight": current_float("section_weight"),
        "stem_quality_min": current_float("stem_quality_min"),
        "bpm_max_diff": current_float("bpm_max_diff"),
        "key_min_score": current_float("key_min_score"),
        "max_section_pairs": current_int("max_section_pairs"),
        "max_candidate_rows": current_int("max_candidate_rows"),
        "reserve_partners": current_int("reserve_partners"),
    }
