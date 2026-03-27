"""
config.py — Central configuration for the mashup engine.
Edit this file to change paths, models, and thresholds.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
AUDIO_DIR        = BASE_DIR / "audio"
RAW_DIR          = AUDIO_DIR / "full_song"
VOCALS_DIR       = AUDIO_DIR / "vocals"
INSTRUMENTALS_DIR = AUDIO_DIR / "instrumentals"
DB_PATH          = BASE_DIR / "database"

# Create dirs if missing
for d in [RAW_DIR, VOCALS_DIR, INSTRUMENTALS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
BEAT_TRIM_SECS   = 30      # analyse first N seconds for speed during testing
                           # set to None for full track

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

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"