"""FastAPI app for the mashup web UI (ingest + download + stems)."""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.models import init_db  # noqa: E402

from api.routes import database as database_routes  # noqa: E402
from api.routes import jobs as jobs_routes  # noqa: E402
from api.routes import mashups as mashup_routes  # noqa: E402
from api.routes import playlists as playlist_routes  # noqa: E402
from api.routes import tracks as track_routes  # noqa: E402


from api import queue_runner  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Start the bounded pipeline worker pool, then re-enqueue any track that was
    # mid-pipeline when the server last stopped (status-derived resume).
    queue_runner.start()
    queue_runner.resume_pending()
    yield


app = FastAPI(title="Mashup Engine API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(playlist_routes.router, prefix="/api/playlists", tags=["playlists"])
app.include_router(track_routes.router, prefix="/api/tracks", tags=["tracks"])
app.include_router(jobs_routes.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(mashup_routes.router, prefix="/api/mashups", tags=["mashups"])
app.include_router(database_routes.router, prefix="/api/db", tags=["database"])


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/health/deps")
def health_deps() -> dict:
    """Report whether the external tools the pipeline needs are available, so a
    first user learns about a missing ffmpeg/demucs BEFORE kicking off a big
    import rather than watching every track fail."""
    import shutil
    import importlib.util

    def _on_path(binary: str) -> bool:
        return shutil.which(binary) is not None

    def _importable(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:  # noqa: BLE001 — a broken install shouldn't 500 the check
            return False

    deps = [
        {"name": "ffmpeg", "ok": _on_path("ffmpeg"),
         "detail": "audio extraction/decoding (required)", "required": True},
        {"name": "ffprobe", "ok": _on_path("ffprobe"),
         "detail": "duration checks (required)", "required": True},
        {"name": "yt-dlp", "ok": _importable("yt_dlp"),
         "detail": "SoundCloud/YouTube metadata + download (required)", "required": True},
        {"name": "demucs", "ok": _importable("demucs"),
         "detail": "stem separation (required to split vocals/instrumental)", "required": True},
        {"name": "librosa", "ok": _importable("librosa"),
         "detail": "audio feature analysis (required for BPM/key/structure)", "required": True},
    ]
    missing = [d["name"] for d in deps if d["required"] and not d["ok"]]
    return {"ok": not missing, "missing": missing, "deps": deps}
