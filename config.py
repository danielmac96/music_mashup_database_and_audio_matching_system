"""
config.py — Central configuration for the mashup engine.

Path/setting resolution order (highest priority first):
    1. Environment variable   — set by Docker / CLI wrappers
    2. settings.json          — written by the first-run Setup Wizard
    3. Built-in default       — <repo>/audio, <repo>/mashup.db, 1 worker

Env overrides (unchanged names, so existing CLI wrappers keep working):
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
VOCALS_DIR        = AUDIO_DIR / "vocals"
INSTRUMENTALS_DIR = AUDIO_DIR / "instrumentals"
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
SNAPSHOTS_DIR = DATA_DIR / "snapshots"   # saved 1001tracklists HTML (Phase 3)
DATASETS_DIR  = DATA_DIR / "datasets"    # exported training sets (Phase 4)
MODELS_DIR    = DATA_DIR / "models"      # learned pairwise models (Phase 5)

# `configured` is True once the user has explicitly chosen an audio library
# location (via env or the wizard). When False, App.jsx shows the Setup Wizard.
CONFIGURED = AUDIO_ROOT_SOURCE in ("env", "settings")


def ensure_dirs() -> None:
    """Create all working directories. Called at import (best-effort) and again
    after the wizard saves settings, so a freshly-chosen library folder exists."""
    for d in (RAW_DIR, VOCALS_DIR, INSTRUMENTALS_DIR, PREVIEWS_DIR, HOOKS_DIR,
              SNAPSHOTS_DIR, DATASETS_DIR, MODELS_DIR):
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
        # Live-read: the stem-separator toggle applies without a restart.
        "stem_separator": {"value": current_stem_separator(),
                           "source": STEM_SEPARATOR_SOURCE},
        "configured": CONFIGURED,
        "settings_path": str(settings_path()),
        "paths": {
            "downloads":     str(RAW_DIR),
            "vocals":        str(VOCALS_DIR),
            "instrumentals": str(INSTRUMENTALS_DIR),
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
# "htdemucs" = Hybrid Transformer Demucs (best quality, slower)
# "mdx_extra" = MDX-Net extra (faster, slightly lower quality)
DEMUCS_MODEL = "htdemucs"
STEMS_TO_KEEP = ["vocals", "no_vocals"]   # no_vocals = instrumental

# Which separator new stem jobs use: "demucs" (quality, slower) or "mdx"
# (audio-separator / UVR MDX-Net ONNX — ~2-4x faster on CPU, slightly lower
# quality). Every stems row is tagged with the separator that produced it.
MDX_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
_SEPARATORS = ("demucs", "mdx")
_sep_val, STEM_SEPARATOR_SOURCE = _resolve(
    "MASHUP_STEM_SEPARATOR", "stem_separator", "demucs")
STEM_SEPARATOR = str(_sep_val).lower() if str(_sep_val).lower() in _SEPARATORS else "demucs"


def current_stem_separator() -> str:
    """The separator to use RIGHT NOW. Unlike the import-time constant, this
    re-reads settings.json so the UI toggle applies to the next separation
    without a server restart. Env var still wins (Docker/CI pinning)."""
    env = os.environ.get("MASHUP_STEM_SEPARATOR")
    if env and env.lower() in _SEPARATORS:
        return env.lower()
    saved = str(_load_settings().get("stem_separator") or "").lower()
    if saved in _SEPARATORS:
        return saved
    return "demucs"

# ── Analysis ──────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050
HOP_LENGTH       = 512
N_MFCC           = 13      # MFCC coefficients stored per track
BEAT_TRIM_SECS   = None    # None = analyse the FULL track (best match quality —
                           # BPM/key from only the intro is unreliable).
                           # Set to e.g. 30 to trade accuracy for speed.

# ── Structure detection (sections: intro/verse/chorus/drop/…) ─────────────────
SECTION_MIN_LEN_SECS  = 12.0   # minimum section length
SECTION_MAX_COUNT     = 14     # cap on sections per track
SECTION_SIM_THRESHOLD = 0.92   # chroma cosine sim above which two sections
                               # count as repeats of each other (chorus finder)

# ── Matching ──────────────────────────────────────────────────────────────────
# Weights used in the composite similarity score (must sum to 1.0)
MATCH_WEIGHTS = {
    "bpm_score":      0.25,
    "key_score":      0.30,
    "energy_score":   0.20,
    "timbre_score":   0.25,
}
TOP_K_RESULTS = 10

# Minimum thresholds — pairs that don't meet BOTH are skipped entirely
# BPM: maximum difference allowed (accounts for halftime/doubletime)
BPM_MAX_DIFF   = 10.0   # e.g. 120 BPM pairs with anything 110–130 (or half/double)
# Key: minimum Camelot score to qualify (0.0–1.0)
KEY_MIN_SCORE  = 0.55   # allows perfect + adjacent + relative major/minor matches

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

# ── SoundCloud scrape ─────────────────────────────────────────────────────────
# Used when you pass a playlist URL rather than a local file list
SC_CLIENT_ID = os.getenv("SC_CLIENT_ID", "")   # optional, for higher rate limits

# ── Firecrawl (1001tracklists structured scrape) ──────────────────────────────
# Firecrawl's hosted stealth proxy renders + bypasses the Cloudflare Turnstile
# that blocks a plain urllib GET of 1001tracklists. Needs an API key; ~9 credits
# per page, so per-track link scraping is on-demand only (not a 216-page bulk).
_fc_val, FIRECRAWL_KEY_SOURCE = _resolve("FIRECRAWL_API_KEY", "firecrawl_api_key", "")
FIRECRAWL_API_KEY   = _fc_val
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"

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

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"


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
