"""FastAPI app for the mashup web UI (ingest + download + stems)."""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.models import init_db  # noqa: E402

from api.routes import database as database_routes  # noqa: E402
from api.routes import datasets as dataset_routes  # noqa: E402
from api.routes import jobs as jobs_routes  # noqa: E402
from api.routes import mashups as mashup_routes  # noqa: E402
from api.routes import mixes as mix_routes  # noqa: E402
from api.routes import models as model_routes  # noqa: E402
from api.routes import playlists as playlist_routes  # noqa: E402
from api.routes import settings as settings_routes  # noqa: E402
from api.routes import studio as studio_routes  # noqa: E402
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
app.include_router(settings_routes.router, prefix="/api/settings", tags=["settings"])
app.include_router(mix_routes.router, prefix="/api/mixes", tags=["mixes"])
app.include_router(dataset_routes.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(model_routes.router, prefix="/api/models", tags=["models"])
app.include_router(studio_routes.router, prefix="/api/studio", tags=["studio"])


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "frontend": frontend_build_state()}


def _ytdlp_version_info() -> tuple[str | None, bool]:
    """Installed yt-dlp version and whether it looks stale (>90 days old).
    yt-dlp versions are dates (YYYY.M.D) and SoundCloud/YouTube change their
    APIs often enough that a stale build is the #1 cause of failed downloads.
    Offline check only — no network call."""
    from datetime import date, datetime

    try:
        from importlib.metadata import version
        ver = version("yt-dlp")
    except Exception:  # noqa: BLE001
        return None, False
    try:
        released = datetime.strptime(ver.split(".post")[0], "%Y.%m.%d").date()
        return ver, (date.today() - released).days > 90
    except ValueError:
        return ver, False


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

    ytdlp_version, ytdlp_stale = _ytdlp_version_info()

    deps = [
        {"name": "ffmpeg", "ok": _on_path("ffmpeg"),
         "detail": "audio extraction/decoding (required)", "required": True},
        {"name": "ffprobe", "ok": _on_path("ffprobe"),
         "detail": "duration checks (required)", "required": True},
        {"name": "yt-dlp", "ok": _importable("yt_dlp"),
         "detail": "SoundCloud/YouTube metadata + download (required)", "required": True,
         "version": ytdlp_version, "stale": ytdlp_stale},
        {"name": "demucs", "ok": _importable("demucs"),
         "detail": "stem separation (required to split vocals/instrumental)", "required": True},
        {"name": "librosa", "ok": _importable("librosa"),
         "detail": "audio feature analysis (required for BPM/key/structure)", "required": True},
        {"name": "playwright", "ok": _importable("playwright"),
         "detail": "optional — 1001tracklists scraping (paste-HTML works without it)",
         "required": False},
    ]
    missing = [d["name"] for d in deps if d["required"] and not d["ok"]]
    stale = [d["name"] for d in deps if d.get("stale")]
    return {"ok": not missing, "missing": missing, "stale": stale, "deps": deps}


@app.post("/api/health/update-ytdlp")
def update_ytdlp() -> dict:
    """Upgrade yt-dlp in the server's environment (pip install -U yt-dlp).
    SoundCloud API changes are the top cause of download failures, so the
    Import tab offers this as a one-click fix when the version looks stale."""
    import subprocess
    from importlib.metadata import version

    old_version, _ = _ytdlp_version_info()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="pip install timed out after 180s")

    output_tail = (result.stdout or "")[-1000:]
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"pip install failed: {(result.stderr or output_tail)[-500:]}",
        )

    # yt-dlp runs as a subprocess (fresh interpreter), so the new version takes
    # effect immediately — no server restart needed. importlib.metadata caches
    # per-lookup, not per-process, so re-reading gives the new version.
    try:
        new_version = version("yt-dlp")
    except Exception:  # noqa: BLE001
        new_version = None
    try:
        from downloader.download import _ytdlp_version
        _ytdlp_version.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "old_version": old_version, "new_version": new_version,
            "output_tail": output_tail}


# ── Serve the built frontend (production / Docker) ────────────────────────────
# Only mounts when frontend/dist exists, so the two-terminal dev flow (Vite on
# :5173 proxying to :8000) is untouched. All /api/* routes are registered above
# and take precedence; the catch-all below returns index.html for client routes.
_DIST = ROOT / "frontend" / "dist"
_SRC = ROOT / "frontend" / "src"

# The banner injected into a stale page. Inline styles and no dependencies,
# because the bundle it is warning you about is the one that would have styled
# it. Dismissible: it is a warning, not a modal.
_STALE_BANNER = """
<div id="stale-build" style="position:fixed;left:0;right:0;top:0;z-index:99999;
     background:#7c2d12;color:#fed7aa;font:13px/1.5 system-ui,sans-serif;
     padding:9px 14px;display:flex;gap:12px;align-items:center;
     box-shadow:0 2px 8px rgba(0,0,0,.4)">
  <b style="flex-shrink:0">Stale UI</b>
  <span style="flex:1">
    This page was built before the current source. The backend is up to date but
    you are looking at an old interface &mdash; new controls will be missing.
    Run <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:3px">
    npm&nbsp;run&nbsp;build</code> in <code style="background:rgba(0,0,0,.3);
    padding:1px 5px;border-radius:3px">frontend/</code>, then reload.
  </span>
  <button onclick="document.getElementById('stale-build').remove()"
     style="background:transparent;border:1px solid rgba(254,215,170,.5);
     color:inherit;border-radius:5px;padding:2px 9px;cursor:pointer">dismiss</button>
</div>
"""


def _newest_mtime(root: Path, suffixes: tuple = ()) -> float:
    """Newest modification time under `root`, 0.0 when it does not exist."""
    if not root.exists():
        return 0.0
    newest = 0.0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if suffixes and p.suffix not in suffixes:
            continue
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    return newest


def frontend_build_state() -> dict:
    """Whether the built bundle is older than the source it was built from.

    frontend/dist is gitignored, so `git pull` updates the source and leaves the
    bundle alone: the server keeps serving an interface from before the pull and
    nothing says so. Restarting does not help, which is exactly what makes it
    confusing — the backend is visibly current while the UI is not.
    """
    if not _DIST.exists():
        return {"built": False, "stale": False,
                "hint": "No frontend/dist — run `npm run build` in frontend/, "
                        "or use the Vite dev server on :5173."}
    src = _newest_mtime(_SRC)
    built = _newest_mtime(_DIST)
    stale = bool(src and built and src > built)
    return {
        "built": True,
        "stale": stale,
        "source_mtime": src,
        "build_mtime": built,
        "hint": ("Source is newer than the build — run `npm run build` in "
                 "frontend/ and reload." if stale else None),
    }


if _DIST.exists():
    _ASSETS = _DIST / "assets"
    if _ASSETS.exists():
        app.mount("/assets", StaticFiles(directory=_ASSETS), name="assets")

    state = frontend_build_state()
    if state["stale"]:
        log.warning(
            "frontend/dist is OLDER than frontend/src — serving a stale UI. "
            "Run `npm run build` in frontend/ and reload. (dist is gitignored, "
            "so a git pull updates the source but not the bundle.)")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        # An unmatched /api path is a real 404 (a missing endpoint), not the SPA.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)

        index = _DIST / "index.html"
        # Checked per request, not once at startup: rebuilding while the server
        # runs is the normal fix, and the banner has to disappear on the reload
        # that follows rather than needing yet another restart.
        if frontend_build_state()["stale"]:
            try:
                html = index.read_text(encoding="utf-8")
                return HTMLResponse(html.replace("<body>", "<body>" + _STALE_BANNER, 1))
            except OSError:
                pass
        return FileResponse(index)
