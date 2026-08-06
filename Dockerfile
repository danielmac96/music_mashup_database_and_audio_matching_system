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

COPY . .
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Docker always sets these, so the Setup Wizard's folder step is skipped
# (readme.md "Settings & configuration"). ./data is the compose volume mount.
ENV MASHUP_AUDIO_ROOT=/data/audio \
    MASHUP_DB_PATH=/data/mashup.db \
    MASHUP_DATA_DIR=/data \
    MASHUP_SETTINGS_DIR=/data/settings \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
