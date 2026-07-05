"""
config.py — Central configuration for the mashup engine.
Edit this file to change paths, models, and thresholds.

Path overrides at runtime:
    MASHUP_AUDIO_ROOT  — relocate the audio library root (default: <repo>/audio)
    MASHUP_DB_PATH     — relocate the SQLite database file (default: <repo>/mashup.db)

CLI flags `--audio-root` / `--db-path` translate to these env vars before any
project import runs, so config.py is the single source of truth for paths.
"""
import os
import re
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

AUDIO_DIR        = Path(os.environ.get("MASHUP_AUDIO_ROOT", BASE_DIR / "audio"))
RAW_DIR          = AUDIO_DIR / "full_song"
VOCALS_DIR       = AUDIO_DIR / "vocals"
INSTRUMENTALS_DIR = AUDIO_DIR / "instrumentals"
PREVIEWS_DIR     = AUDIO_DIR / "previews"   # rendered audition mashup previews

# SQLite file path — must not be BASE_DIR / "database" (that is the Python package directory).
DB_PATH = Path(os.environ.get("MASHUP_DB_PATH", BASE_DIR / "mashup.db"))

# Create dirs if missing
for d in [RAW_DIR, VOCALS_DIR, INSTRUMENTALS_DIR, PREVIEWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

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

# ── SoundCloud scrape ─────────────────────────────────────────────────────────
# Used when you pass a playlist URL rather than a local file list
SC_CLIENT_ID = os.getenv("SC_CLIENT_ID", "")   # optional, for higher rate limits

# ── Background processing ─────────────────────────────────────────────────────
# Number of worker threads that drain the ingest→download→stems→analysis→structure
# pipeline queue. Default 1: stem separation (Demucs) is CPU/GPU heavy, so running
# several tracks at once thrashes the machine. Raise on a beefy box via env.
PIPELINE_WORKERS = max(1, int(os.getenv("MASHUP_PIPELINE_WORKERS", "1")))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"


# ── Shared helpers ────────────────────────────────────────────────────────────

def sanitize_filename_chars(name: str) -> str:
    """Replace characters not safe for a filename/foldername with underscores.
    Callers handle their own truncation/collapsing on top of this."""
    return re.sub(r"[^\w]", "_", name or "")
