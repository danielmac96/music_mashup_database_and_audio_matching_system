# Mashup Engine — single-process image (FastAPI + built React UI).
# See readme.md "Quick start (Docker)".

# ── Stage 1: build the frontend ────────────────────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# ffmpeg/ffprobe: audio extraction (yt-dlp) + duration checks.
# libsndfile1: required by soundfile/librosa. libsamplerate0: audio-separator's
# samplerate binding. build-essential: diffq (a Demucs dep on non-Windows)
# compiles a C extension at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libsamplerate0 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    # CPU-only PyTorch wheels — the default PyPI build pulls multi-GB CUDA
    # deps that are useless (and mostly broken) in a CPU-only container.
    && pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    # Optional fast MDX-Net separator (the "Fast" toggle in Stems); no release
    # resolves against this pin set, so it's installed dep-less on top — see
    # the note above `onnxruntime` in requirements.txt.
    && pip install --no-cache-dir audio-separator==0.30.0 --no-deps

# The Essentia analyser (optional, benchmarked by scripts/bench_analyzers.py):
# its own layer so changing requirements.txt does not re-download ~290 MB.
COPY requirements-essentia.txt ./
RUN pip install --no-cache-dir -r requirements-essentia.txt \
    && python -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__"

# allin1 (beats, downbeats, labelled sections — readme §5.5): its own layer.
# natten is compiled against the torch above (no CPU wheel is published for it),
# and allin1 goes in dep-less so it cannot move the numpy / torch pins — see
# requirements-allin1.txt for why each step is the way it is.
COPY requirements-allin1.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements-allin1.txt \
    && pip install --no-cache-dir allin1==1.1.0 --no-deps \
    && pip install --no-cache-dir natten==0.17.3 --no-build-isolation \
    && python -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__" \
    && python -c "import allin1, natten, madmom; from allin1.helpers import run_inference"

COPY . .
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
# BuildKit stamps context files with the time their COPY ran, not the host
# mtime, so frontend/src (copied one layer up) always ends up NEWER than the
# dist stage 1 built minutes earlier — and api.server.frontend_build_state()
# then flags a "Stale UI" on a bundle it just built from that exact source.
# The bundle IS this source; stamp it so.
RUN find frontend/dist -exec touch {} +

# Docker always sets these, so the Setup Wizard's folder step is skipped
# (readme.md "Settings & configuration"). ./data is the compose volume mount.
# TF_CPP_MIN_LOG_LEVEL: essentia-tensorflow logs CUDA probing on import, and
# this is a CPU image. HF_HOME: allin1's model weights persist with the data.
ENV MASHUP_AUDIO_ROOT=/data/audio \
    MASHUP_DB_PATH=/data/mashup.db \
    MASHUP_DATA_DIR=/data \
    MASHUP_SETTINGS_DIR=/data/settings \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HF_HOME=/data/huggingface

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
