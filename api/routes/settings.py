"""Settings endpoints: first-run wizard + settings readout.

GET  /api/settings                — resolved values, their provenance, and the
                                    `configured` flag that gates the Setup Wizard
POST /api/settings                — persist audio_root / db_path / pipeline_workers
                                    to settings.json (config.save_settings)
POST /api/settings/validate-path  — dry-run a proposed library folder before saving

Note: config.py binds its constants at import time, so a saved change takes
full effect on the next server start. The response says so explicitly rather
than pretending it was applied live.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config

router = APIRouter()


def _provenance_live() -> dict:
    """config.CONFIGURED binds at import, but the wizard saves settings.json and
    reloads the page without restarting the server — re-read the file so the
    `configured` gate lifts immediately instead of trapping the user in the
    wizard until a restart."""
    prov = config.settings_provenance()
    if not prov.get("configured"):
        try:
            import json
            p = config.settings_path()
            saved = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, ValueError):
            saved = {}
        if saved.get("audio_root"):
            prov["configured"] = True
    return prov


@router.get("")
def get_settings() -> dict:
    return _provenance_live()


class ValidatePathRequest(BaseModel):
    path: str


@router.post("/validate-path")
def validate_path(req: ValidatePathRequest) -> dict:
    """Check a proposed audio-library folder: expandable, absolute-able,
    creatable, and writable. Never creates anything permanent — a probe file
    is written and removed."""
    raw = (req.path or "").strip()
    if not raw:
        return {"ok": False, "reason": "Path is empty."}

    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        return {"ok": False, "reason": "Use an absolute path (e.g. /Users/you/Music/mashups).",
                "resolved": str(p)}

    exists = p.exists()
    if exists and not p.is_dir():
        return {"ok": False, "reason": "That path exists but is a file, not a folder.",
                "resolved": str(p)}

    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".mashup-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return {"ok": False, "reason": f"Folder is not writable: {exc}", "resolved": str(p)}

    return {"ok": True, "resolved": str(p), "existed": exists}


class NewLibraryRequest(BaseModel):
    path: str
    force: bool = False  # allow reusing a folder that already has a mashup.db


@router.post("/new-library")
def new_library(req: NewLibraryRequest) -> dict:
    """Create a fresh, empty music library at `path` and make it active.

    Materializes an empty SQLite schema at <path>/mashup.db and the audio
    subfolders under <path>/audio, then persists db_path/audio_root/data_dir to
    settings.json. Because config.py binds its path constants at import, the new
    library only becomes live on the next server start — the response says so.
    """
    raw = (req.path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="path is empty")

    root = Path(os.path.expanduser(raw))
    if not root.is_absolute():
        raise HTTPException(status_code=400,
                            detail="Use an absolute path (e.g. C:\\Music\\mashups).")
    if root.exists() and not root.is_dir():
        raise HTTPException(status_code=400,
                            detail="That path exists but is a file, not a folder.")

    db_path = root / "mashup.db"
    audio_root = root / "audio"

    if db_path.exists() and not req.force:
        raise HTTPException(
            status_code=409,
            detail=f"A database already exists at {db_path}. Pass force to reuse this folder.")

    # Create folders + probe writability before touching anything permanent.
    try:
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("full_song", "vocals", "instrumentals", "previews"):
            (audio_root / sub).mkdir(parents=True, exist_ok=True)
        probe = root / ".mashup-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Folder is not writable: {exc}")

    # Materialize an empty schema in the new DB file (idempotent if it exists).
    try:
        from database.models import init_db
        init_db(db_path=db_path)
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise HTTPException(status_code=500,
                            detail=f"Could not create the database: {exc}")

    config.save_settings({
        "db_path": str(db_path),
        "audio_root": str(audio_root),
        "data_dir": str(root),
    })

    return {
        "created": True,
        "db_path": str(db_path),
        "audio_root": str(audio_root),
        "restart_required": True,
        "settings": _provenance_live(),
    }


class SaveSettingsRequest(BaseModel):
    audio_root: Optional[str] = None
    db_path: Optional[str] = None
    pipeline_workers: Optional[int] = None
    stem_separator: Optional[str] = None  # "demucs" (quality) | "mdx" (fast)


@router.post("")
def save_settings(req: SaveSettingsRequest) -> dict:
    if (req.audio_root is None and req.db_path is None
            and req.pipeline_workers is None and req.stem_separator is None):
        raise HTTPException(status_code=400, detail="nothing to save")
    if req.pipeline_workers is not None and req.pipeline_workers < 1:
        raise HTTPException(status_code=400, detail="pipeline_workers must be >= 1")
    if req.stem_separator is not None and req.stem_separator not in ("demucs", "mdx"):
        raise HTTPException(status_code=400,
                            detail="stem_separator must be 'demucs' or 'mdx'")

    new: dict = {}
    if req.audio_root:
        new["audio_root"] = str(Path(os.path.expanduser(req.audio_root)))
    if req.db_path:
        new["db_path"] = str(Path(os.path.expanduser(req.db_path)))
    if req.pipeline_workers is not None:
        new["pipeline_workers"] = str(req.pipeline_workers)
    if req.stem_separator is not None:
        new["stem_separator"] = req.stem_separator

    path = config.save_settings(new)

    # Best-effort: create the newly chosen library folders right away so the
    # wizard's "done" state is real even before the restart.
    if "audio_root" in new:
        root = Path(new["audio_root"])
        try:
            for sub in ("full_song", "vocals", "instrumentals", "previews"):
                (root / sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    return {
        "saved": True,
        "settings_path": str(path),
        # stem_separator is re-read on every separation, so it applies live;
        # paths/worker counts bind at import and need a restart.
        "restart_required": any(
            k in new for k in ("audio_root", "db_path", "pipeline_workers")),
        "settings": _provenance_live(),
    }
