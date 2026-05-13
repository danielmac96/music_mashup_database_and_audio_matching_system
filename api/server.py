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

from api.routes import jobs as jobs_routes  # noqa: E402
from api.routes import playlists as playlist_routes  # noqa: E402
from api.routes import tracks as track_routes  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
