"""api/workers/hook_worker.py — cut each track's hook into a small standalone wav.

The ranked list's whole premise is that stepping to a candidate makes a sound
almost immediately. decodeStem fetches and decodes an entire ~40 MB stem into an
AudioBuffer, which is far too slow and too memory-hungry to do per keypress, so
the 16 bars chosen in T1.5 are pre-cut into a ~3 MB clip the browser can fetch
and decode between arrow presses.

Deliberately soundfile-only — no librosa, no resampling, no DSP. This is a
seek-and-copy of a byte range that already exists on disk, and it runs inside
the request path on a cache miss, so it has to stay cheap.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import HOOKS_DIR
from database.models import get_conn, get_features_for_song

log = logging.getLogger(__name__)

# Stems a hook can be cut from. 'full' is allowed so a track whose separation
# failed can still be auditioned against the mix.
HOOK_STEMS = ("vocals", "instrumental", "full")

# Read in blocks rather than one slab: a hook is small, but a pathological
# hook_end on a long file should not pull the whole track into memory.
_BLOCK_FRAMES = 1 << 16


class HookRenderError(RuntimeError):
    """Raised when a hook clip cannot be produced. Callers turn this into a 404
    or 501 with the message intact — never a bare 500."""


def hook_clip_path(song_id: int, stem: str,
                   window: Optional[tuple] = None) -> Path:
    """Stable on-disk location for a rendered clip.

    Without a window this is the track's own hook (T1.5/T1.6). With one it is an
    arbitrary section — the pair-specific clip a candidate row points at (T3.3).
    Windows are keyed in milliseconds so the same section always resolves to the
    same file and the cache actually hits; a track has at most a handful of
    usable sections, so this stays a small bounded set per stem."""
    if window is None:
        return HOOKS_DIR / f"{stem}_{song_id}_hook.wav"
    start_ms, end_ms = (int(round(v * 1000)) for v in window)
    return HOOKS_DIR / f"{stem}_{song_id}_{start_ms}_{end_ms}.wav"


def _stem_file(song_id: int, stem: str) -> Optional[str]:
    conn = get_conn()
    row = conn.execute(
        "SELECT file_path FROM stems WHERE song_id=? AND stem_type=?",
        (song_id, stem),
    ).fetchone()
    conn.close()
    return row["file_path"] if row else None


def render_hook(song_id: int, stem: str = "vocals", force: bool = False,
                start: Optional[float] = None,
                end: Optional[float] = None) -> str:
    """Render (or reuse) a clip of this track. Returns the file path.

    With no window this is the track's stored hook. With `start`/`end` it is
    that exact span — how a candidate's winning section pair (T3.3) gets
    previewed instead of each track's generic hook. Either way it is a
    seek-and-copy, so a cold render is cheap enough for the request path."""
    if stem not in HOOK_STEMS:
        raise HookRenderError(f"stem must be one of {sorted(HOOK_STEMS)}")

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise HookRenderError(
            "soundfile is not installed — hook previews are unavailable. "
            "Install it to enable instant audition."
        ) from exc

    windowed = start is not None and end is not None
    if windowed:
        start = max(0.0, float(start))
        end = float(end)
        if end <= start:
            raise HookRenderError(
                f"requested window is empty for song {song_id} stem '{stem}'")

    out = hook_clip_path(song_id, stem, (start, end) if windowed else None)
    if out.exists() and not force:
        return str(out)

    if not windowed:
        feat = get_features_for_song(song_id, stem)
        if not feat or feat.get("hook_start") is None or feat.get("hook_end") is None:
            raise HookRenderError(
                f"no hook chosen for song {song_id} stem '{stem}' — "
                "run structure detection, or GET /api/tracks/{id}/hook to backfill")
        start = max(0.0, float(feat["hook_start"]))
        end = float(feat["hook_end"])
        if end <= start:
            raise HookRenderError(
                f"hook window is empty for song {song_id} stem '{stem}'")

    src = _stem_file(song_id, stem)
    if not src or not Path(src).exists():
        raise HookRenderError(f"stem '{stem}' audio is missing for song {song_id}")

    HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    try:
        with sf.SoundFile(str(src)) as f:
            sr = f.samplerate
            begin = min(int(start * sr), len(f))
            # Section times can outlast a truncated stem; clamp rather than fail.
            want = max(0, min(int(end * sr), len(f)) - begin)
            if want == 0:
                raise HookRenderError(
                    f"clip window starts past the end of the audio for song {song_id}")
            f.seek(begin)
            # format is explicit because the temp name ends in .part, and
            # soundfile otherwise infers the container from the extension.
            with sf.SoundFile(str(tmp), mode="w", samplerate=sr,
                              channels=f.channels, subtype=f.subtype,
                              format="WAV") as o:
                left = want
                while left > 0:
                    block = f.read(min(_BLOCK_FRAMES, left), dtype="float32",
                                   always_2d=True)
                    if len(block) == 0:
                        break
                    o.write(block)
                    left -= len(block)
        tmp.replace(out)
    except HookRenderError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HookRenderError(
            f"could not render hook for song {song_id} stem '{stem}': "
            f"{type(exc).__name__}: {exc}") from exc

    return str(out)


def warm_hooks(song_id: int) -> dict:
    """Best-effort pre-render of both preview stems after analysis.

    Never raises: a cold hook is a slower first keypress, not a broken track,
    and this runs at the tail of a pipeline that has already succeeded.
    """
    done, failed = [], {}
    for stem in ("vocals", "instrumental"):
        try:
            render_hook(song_id, stem)
            done.append(stem)
        except HookRenderError as exc:
            failed[stem] = str(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("unexpected hook render failure")
            failed[stem] = f"{type(exc).__name__}: {exc}"
    return {"rendered": done, "skipped": failed}
