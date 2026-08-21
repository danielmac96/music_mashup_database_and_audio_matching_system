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
VOCALS_DIR        = AUDIO_DIR / "vocals"
INSTRUMENTALS_DIR = AUDIO_DIR / "instrumentals"
# Four-stem separation (Phase D). `instrumental` is still written as the sum of
# these three, so every existing consumer keeps working unchanged.
DRUMS_DIR         = AUDIO_DIR / "drums"
BASS_DIR          = AUDIO_DIR / "bass"
OTHER_DIR         = AUDIO_DIR / "other"
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
    for d in (RAW_DIR, VOCALS_DIR, INSTRUMENTALS_DIR, DRUMS_DIR, BASS_DIR,
              OTHER_DIR, PREVIEWS_DIR, HOOKS_DIR,
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
        # Live-read: these apply without a restart.
        "stem_separator": {"value": current_stem_separator(),
                           "source": STEM_SEPARATOR_SOURCE},
        "stem_mode": {"value": current_stem_mode(), "source": STEM_MODE_SOURCE},
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
        # Presence only, never the values — this response goes to the browser.
        # The secret must not leave the server, and neither must the token.
        "soundcloud_client_id": {"value": bool(SOUNDCLOUD_CLIENT_ID),
                                 "source": SOUNDCLOUD_CLIENT_ID_SOURCE},
        "soundcloud_client_secret": {"value": bool(SOUNDCLOUD_CLIENT_SECRET),
                                     "source": SOUNDCLOUD_CLIENT_SECRET_SOURCE},
        "stem_format": {"value": STEM_FORMAT, "source": "code"},
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


# How many stems to split into (Phase D).
#   "two"  — vocals + instrumental. What the app shipped with; MDX only does this.
#   "four" — drums / bass / other / vocals, PLUS a summed instrumental so every
#            existing consumer is unchanged. Demucs only.
# Four stems are what make the two arrangement problems visible: a bed that
# still contains its own topline, and which frequency bands each side occupies.
# They also unlock the real producer move — drop the bed's bass, keep the
# vocal track's; swap the drums.
_STEM_MODES = ("two", "four")
_mode_val, STEM_MODE_SOURCE = _resolve("MASHUP_STEM_MODE", "stem_mode", "two")
STEM_MODE = str(_mode_val).lower() if str(_mode_val).lower() in _STEM_MODES else "two"

# Stems are written as FLAC: lossless, and roughly half the size of WAV. At
# ~900 tracks x 4 stems that is the difference between ~160 GB and ~80 GB.
STEM_FORMAT = "flac"

# The four Demucs sources, and the three that sum to the instrumental.
DEMUCS_SOURCES = ("drums", "bass", "other", "vocals")
INSTRUMENTAL_SOURCES = ("drums", "bass", "other")


def current_stem_mode() -> str:
    """The stem mode to use RIGHT NOW, re-reading settings.json like
    current_stem_separator. MDX cannot do four stems, so an mdx run is always
    two regardless of this setting."""
    env = os.environ.get("MASHUP_STEM_MODE")
    if env and env.lower() in _STEM_MODES:
        return env.lower()
    val = _load_settings().get("stem_mode")
    if isinstance(val, str) and val.lower() in _STEM_MODES:
        return val.lower()
    return STEM_MODE


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
TOP_K_RESULTS = 10

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

# The gate exists to bound the matrix, not to express taste. On the MODEL path
# the key half is already dropped (documented mashups sometimes break it); this
# widens the tempo half too, so the model can learn that you happily halftime a
# 150 BPM vocal over a 75 BPM bed. Candidate generation stays tractable because
# bucketing, not this threshold, is what keeps the matrix small.
BPM_MAX_DIFF_MODEL = 20.0

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
# READING SoundCloud needs no credentials at all: ingest/soundcloud_api.py scrapes
# a working client_id out of SoundCloud's own public JS bundles, and that is what
# both the mixes auto-resolver and the Discovery browse layer use.
#
# WRITING (creating a playlist on your account, liking, reposting) needs OAuth 2.1
# against a *registered* app — a client id AND secret from developers.soundcloud.com.
# That registration has been closed to new applicants since 2019, so these are
# empty for almost everyone and the write layer stays dormant: see
# ingest/soundcloud_oauth.py, which reports `configured: false` and makes every
# write endpoint answer 501 with an explanation rather than failing obscurely.
#
# Set them and the "Push to SoundCloud" action on a crate lights up. SC_CLIENT_ID
# is kept as an env alias for the older name.
_sc_id_val, SOUNDCLOUD_CLIENT_ID_SOURCE = _resolve(
    "SOUNDCLOUD_CLIENT_ID", "soundcloud_client_id", os.getenv("SC_CLIENT_ID", ""))
SOUNDCLOUD_CLIENT_ID = _sc_id_val
_sc_secret_val, SOUNDCLOUD_CLIENT_SECRET_SOURCE = _resolve(
    "SOUNDCLOUD_CLIENT_SECRET", "soundcloud_client_secret", "")
SOUNDCLOUD_CLIENT_SECRET = _sc_secret_val

# The OAuth token lives in its own file, NOT settings.json — GET /api/settings is
# read by the browser and must never carry a bearer token.
def soundcloud_token_path() -> Path:
    return _settings_dir() / "soundcloud_token.json"

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

# Weights inside score_section itself (spec §7's phrase / rhythm / structure,
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
    "bpm_max_diff_model": ("MASHUP_BPM_MAX_DIFF_MODEL", "BPM_MAX_DIFF_MODEL", 1.0, 60.0),
}
_TUNABLE_INTS = {
    "max_section_pairs": ("MASHUP_MAX_SECTION_PAIRS",
                          "MAX_SECTION_PAIRS_PER_SONG_PAIR", 1, 8),
    "max_candidate_rows": ("MASHUP_MAX_CANDIDATE_ROWS",
                           "MAX_CANDIDATE_ROWS", 1, 5_000_000),
}

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
    weights (the CLI, a one-off composite_score call).
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
        "bpm_max_diff_model": current_float("bpm_max_diff_model"),
        "max_section_pairs": current_int("max_section_pairs"),
        "max_candidate_rows": current_int("max_candidate_rows"),
    }
