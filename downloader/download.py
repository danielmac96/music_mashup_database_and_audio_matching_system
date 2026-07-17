"""
downloader/download.py — Download audio in best quality via yt-dlp.

SoundCloud Go+ handling:
  If the downloaded file is under PREVIEW_MAX_SECS (30s preview),
  fall back to a YouTube search for the full song using the title + artist.

Output: MP3 in config.RAW_DIR / "{title}_{artist}.mp3"
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple
import subprocess
import sys
import logging
import re
from pathlib import Path

from config import RAW_DIR, YTDLP_FORMAT, YTDLP_FORMAT_FALLBACK, YTDLP_POSTARGS

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Download failed with a known reason. ``kind`` is a machine-readable
    category: drm / premium / geo / private / removed / network / outdated /
    unknown. The message is user-facing and lands in ``songs.last_error``."""

    def __init__(self, message: str, kind: str = "unknown"):
        super().__init__(message)
        self.kind = kind


@lru_cache(maxsize=1)
def _ytdlp_version() -> str:
    try:
        from importlib.metadata import version
        return version("yt-dlp")
    except Exception:  # noqa: BLE001
        return "unknown"


def _extract_ytdlp_errors(output: str) -> list[str]:
    """Pull yt-dlp ERROR lines out of the merged stdout/stderr stream."""
    return [ln.strip() for ln in output.splitlines() if ln.strip().startswith("ERROR")]


# (regex, kind, user-facing message) — first match on the joined lowercased
# ERROR lines wins, so order encodes priority. {ver} = installed yt-dlp version.
_ERROR_CLASSES: tuple[tuple[str, str, str], ...] = (
    (r"drm.protected", "drm",
     "SoundCloud serves this track DRM-protected — it cannot be downloaded directly."),
    (r"go\+|premium|subscribers", "premium",
     "This track is for SoundCloud Go+ subscribers only."),
    (r"not available in your country|geo.?restrict", "geo",
     "This track is geo-blocked in your region."),
    (r"private|sign in|log ?in|authentication", "private",
     "This track is private or requires a SoundCloud login."),
    (r"unable to download json metadata.{0,80}404", "outdated",
     "SoundCloud rejected the request (HTTP 404). The track may have been removed, "
     "or yt-dlp {ver} is outdated — use the Update yt-dlp button on the Import tab."),
    (r"http error 404|not found|no longer available|has been removed", "removed",
     "Track not found — it may have been removed from SoundCloud."),
    (r"getaddrinfo|timed out|connection|temporary failure|unable to download webpage",
     "network",
     "Network error while downloading — check your connection and retry."),
)

# Failure kinds where SoundCloud has the track but won't serve it — a
# title/artist YouTube search is a legitimate way to get the same song.
_GATED_KINDS = frozenset({"drm", "premium", "geo"})


def classify_download_error(error_lines: list[str]) -> tuple[str, str]:
    """Map raw yt-dlp ERROR lines to (kind, user-facing message)."""
    joined = " ".join(error_lines).lower()
    if joined:
        for pattern, kind, template in _ERROR_CLASSES:
            if re.search(pattern, joined):
                return kind, template.format(ver=_ytdlp_version())
        return "unknown", error_lines[0][:400]
    return "unknown", "Download failed — yt-dlp gave no error detail (see server logs)."

# Files shorter than this are considered previews and trigger the YT fallback
PREVIEW_MAX_SECS = 35

# How many YouTube search results to try per query (1 = top hit, …, N = Nth hit)
YOUTUBE_SEARCH_MAX_RESULTS = 5

# Optional progress callback. percent is None for status-only updates.
ProgressCb = Optional[Callable[[Optional[int], str], None]]


class DownloadResult(NamedTuple):
    """download_track return value. duration_secs is set when audio came from YouTube fallback."""
    path: Path
    duration_secs: Optional[float] = None  # if set, persist to songs.duration_secs


class _YtAttempt(NamedTuple):
    label: str
    format_str: str
    player_client: str
    use_cookies: bool


def _youtube_attempts() -> tuple[_YtAttempt, ...]:
    return (
        _YtAttempt("ios+bestaudio", YTDLP_FORMAT, "youtube:player_client=ios", False),
        _YtAttempt(
            "android_web+bestaudio",
            YTDLP_FORMAT,
            "youtube:player_client=android,web",
            False,
        ),
        _YtAttempt(
            "android_web+ba/b",
            YTDLP_FORMAT_FALLBACK,
            "youtube:player_client=android,web",
            False,
        ),
        _YtAttempt(
            "cookies+android_web+ba/b",
            YTDLP_FORMAT_FALLBACK,
            "youtube:player_client=android,web",
            True,
        ),
    )


def download_track(song_id: int, title: str, source_url: str,
                   artist: str = "",
                   on_progress: ProgressCb = None) -> DownloadResult:
    """Download a track's audio. Returns a DownloadResult on success; raises
    DownloadError with a user-facing reason on failure (never returns None)."""
    out_path = RAW_DIR / f"{_safe(title)}_{_safe(artist)}.mp3"

    if out_path.exists():
        duration = _get_duration(out_path)
        if duration and duration > PREVIEW_MAX_SECS:
            log.info(f"Already downloaded (full): {out_path.name}")
            if on_progress:
                on_progress(100, "Already downloaded")
            # Pass duration so the worker refreshes the DB row — fixes stale 30s
            # rows seeded from SoundCloud Go+ previews during ingest.
            return DownloadResult(out_path, duration)
        else:
            log.warning(f"Existing file is a preview ({duration:.0f}s) — re-downloading")
            out_path.unlink()

    # A direct YouTube source routes through the retry ladder (see _download_ytdlp)
    # and a short result is a genuine short video, not a SoundCloud Go+ preview —
    # so we skip the preview→YouTube-search fallback for it.
    is_yt_source = _is_youtube_like(source_url)

    if on_progress:
        on_progress(0, "Downloading from YouTube…" if is_yt_source
                    else "Downloading from SoundCloud…")

    dl = _download_ytdlp(source_url, out_path, on_progress=on_progress)
    path = dl.path

    if path and path.exists() and not is_yt_source:
        duration = _get_duration(path)
        if duration and duration <= PREVIEW_MAX_SECS:
            log.warning(
                f"Downloaded file is only {duration:.0f}s — SoundCloud Go+ preview detected. "
                f"Searching YouTube for full track..."
            )
            if on_progress:
                on_progress(None, "Got SoundCloud preview only — searching YouTube fallback…")
            path.unlink()
            fb = _fallback_youtube(title, artist, out_path, on_progress=on_progress)
            if fb:
                yt_path, yt_secs = fb
                return DownloadResult(yt_path, yt_secs)
            raise DownloadError(
                "SoundCloud served only a Go+ 30s preview and no full-length "
                "YouTube match was found.", kind="premium")

    if path and path.exists():
        return DownloadResult(path, _get_duration(path))

    kind, msg = classify_download_error(dl.error_lines)

    # SoundCloud has the track but won't serve it (DRM / Go+ / geo) — the same
    # YouTube title search used for Go+ previews can still get the full song.
    if not is_yt_source and kind in _GATED_KINDS:
        log.warning(f"SoundCloud blocked this track ({kind}) — trying YouTube fallback")
        if on_progress:
            on_progress(None, "SoundCloud blocked this track — searching YouTube…")
        fb = _fallback_youtube(title, artist, out_path, on_progress=on_progress)
        if fb:
            yt_path, yt_secs = fb
            return DownloadResult(yt_path, yt_secs)
        msg += " No full-length YouTube match was found either."

    raise DownloadError(msg, kind=kind)


class ReverifyResult(NamedTuple):
    """reverify_track return value.
    path/duration_secs: the current (possibly freshly re-downloaded) full file.
    replaced: True when a stale <=35s preview was swapped for a full-length file,
              so callers should re-run stems/analysis on the new audio."""
    path: Optional[Path]
    duration_secs: Optional[float]
    replaced: bool


def reverify_track(song_id: int, title: str, source_url: str,
                   artist: str = "",
                   on_progress: ProgressCb = None) -> ReverifyResult:
    """Re-check a previously-downloaded track (WORKFLOW_AUDIT ISSUE-1).

    If the file on disk is already full-length, just report its true duration so
    a stale DB `duration_secs` (e.g. a 30s value seeded from a SoundCloud Go+
    preview during ingest) can be corrected. If the file is missing or still a
    ~30s preview, re-run the normal download — which unlinks the preview and
    fires the YouTube full-track fallback — and flag the swap via `replaced`."""
    out_path = RAW_DIR / f"{_safe(title)}_{_safe(artist)}.mp3"
    disk_dur = _get_duration(out_path) if out_path.exists() else None

    if disk_dur and disk_dur > PREVIEW_MAX_SECS:
        # Full file already present — no re-download, just surface true duration.
        return ReverifyResult(out_path, disk_dur, replaced=False)

    was_preview = disk_dur is not None and disk_dur <= PREVIEW_MAX_SECS
    try:
        result = download_track(song_id, title, source_url, artist=artist,
                                on_progress=on_progress)
    except DownloadError as exc:
        log.warning(f"Reverify re-download failed: {exc}")
        return ReverifyResult(None, None, replaced=False)
    if not result.path.exists():
        return ReverifyResult(None, None, replaced=False)

    new_dur = result.duration_secs
    if new_dur is None:
        new_dur = _get_duration(result.path)
    replaced = bool(new_dur and new_dur > PREVIEW_MAX_SECS) and (was_preview or disk_dur is None)
    return ReverifyResult(result.path, new_dur, replaced=replaced)


# ── yt-dlp download ───────────────────────────────────────────────────────────


def _is_youtube_like(url: str) -> bool:
    u = url.lower().strip()
    if u.startswith("ytsearch"):
        return True
    return (
        "youtube.com/" in u
        or "youtu.be/" in u
        or "music.youtube.com" in u
    )


def _cleanup_stem_outputs(out_path: Path) -> None:
    pattern = f"{out_path.stem}.*"
    for p in out_path.parent.glob(pattern):
        try:
            p.unlink()
        except OSError:
            pass


class _RunOutcome(NamedTuple):
    ok: bool
    error_lines: list


def _run_ytdlp(
    url: str,
    out_path: Path,
    format_str: str,
    *,
    extractor_args: Optional[str] = None,
    use_cookies: bool,
    playlist_item: Optional[int] = None,
    on_progress: ProgressCb = None,
) -> _RunOutcome:
    """
    Run yt-dlp once. ok=True means it exited 0 and an output file exists;
    on failure error_lines carries the raw yt-dlp ERROR lines for classification.
    """
    from api.workers._progress import parse_ytdlp, stream_subprocess

    tmp_template = str(out_path.with_suffix("")) + ".%(ext)s"
    cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "-f", format_str,
        "--output", tmp_template,
        "--no-warnings",
        "--newline",   # progress lines on their own line so our line splitter sees them
        *YTDLP_POSTARGS,
    ]
    if playlist_item is None:
        cmd.append("--no-playlist")
    else:
        cmd.extend(["--playlist-items", str(playlist_item)])
    if use_cookies:
        cmd.extend(["--cookies-from-browser", "chrome"])
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    cmd.append(url)

    log.info(f"Downloading: {url}" + (f" (item {playlist_item})" if playlist_item else ""))

    def _on_line(line: str) -> None:
        if not on_progress:
            return
        pct = parse_ytdlp(line)
        if pct is not None:
            on_progress(pct, f"yt-dlp: {pct}%")
        elif line.strip().startswith("[") and len(line) < 200:
            on_progress(None, line.strip())

    try:
        result = stream_subprocess(cmd, _on_line, timeout=300)
    except FileNotFoundError:
        log.error("Python or yt-dlp not found. Install with: pip install yt-dlp")
        return _RunOutcome(False, ["ERROR: yt-dlp is not installed on the server "
                                   "(pip install yt-dlp)"])
    except subprocess.TimeoutExpired:
        log.error(f"Download timed out: {url}")
        return _RunOutcome(False, ["ERROR: download timed out after 300s"])

    if result.returncode != 0:
        errors = _extract_ytdlp_errors(result.stdout)
        log.warning(
            f"yt-dlp attempt failed [{result.returncode}]: "
            f"{' | '.join(errors) if errors else result.stdout[-400:]}"
        )
        return _RunOutcome(False, errors)

    if out_path.exists():
        return _RunOutcome(True, [])

    for candidate in out_path.parent.glob(f"{out_path.stem}.*"):
        candidate.rename(out_path)
        return _RunOutcome(True, [])

    log.warning("yt-dlp exited 0 but output file not found")
    return _RunOutcome(False, ["ERROR: yt-dlp exited 0 but produced no output file"])


class _DlOutcome(NamedTuple):
    path: Optional[Path]
    error_lines: list


def _download_ytdlp(
    url: str,
    out_path: Path,
    *,
    playlist_item: Optional[int] = None,
    on_progress: ProgressCb = None,
) -> _DlOutcome:
    """
    Download with yt-dlp. YouTube / ytsearch URLs use a retry ladder;
    other sites (e.g. SoundCloud) use a single plain invocation.
    On failure error_lines carries the yt-dlp ERROR lines (last attempt's).
    """
    _cleanup_stem_outputs(out_path)

    if not _is_youtube_like(url):
        run = _run_ytdlp(
            url,
            out_path,
            YTDLP_FORMAT,
            extractor_args=None,
            use_cookies=False,
            playlist_item=playlist_item,
            on_progress=on_progress,
        )
        if run.ok and out_path.exists():
            log.info(f"Downloaded: {out_path.name}")
            return _DlOutcome(out_path, [])
        log.error("yt-dlp failed for non-YouTube URL after 1 attempt")
        return _DlOutcome(None, run.error_lines)

    last_errors: list = []
    for att in _youtube_attempts():
        _cleanup_stem_outputs(out_path)
        if on_progress:
            on_progress(None, f"YouTube attempt: {att.label}")
        run = _run_ytdlp(
            url,
            out_path,
            att.format_str,
            extractor_args=att.player_client,
            use_cookies=att.use_cookies,
            playlist_item=playlist_item,
            on_progress=on_progress,
        )
        if run.ok and out_path.exists():
            log.info(f"Downloaded ({att.label}): {out_path.name}")
            return _DlOutcome(out_path, [])
        if run.error_lines:
            last_errors = run.error_lines

    log.error(f"yt-dlp failed for URL after all YouTube retries: {url[:120]}")
    return _DlOutcome(None, last_errors)


# ── YouTube fallback ──────────────────────────────────────────────────────────


def _fallback_youtube(title: str, artist: str, out_path: Path,
                       on_progress: ProgressCb = None) -> Optional[Tuple[Path, float]]:
    """
    Search YouTube for the full track using multiple query strategies.
    Strips parenthetical suffixes from title for cleaner search results.
    Uses ytsearchN and walks top results so one bad hit does not sink the track.
    """
    clean_title = re.sub(r'\s*[\(\[].*?[\)\]]', '', title).strip()
    n = YOUTUBE_SEARCH_MAX_RESULTS
    queries = [
        f"ytsearch{n}:{artist} {clean_title} official audio",
        f"ytsearch{n}:{artist} {clean_title} lyrics",
        f"ytsearch{n}:{artist} {clean_title}",
    ]

    for query in queries:
        for rank in range(1, n + 1):
            log.info(f"YouTube search: {query}  [trying result #{rank}]")
            if on_progress:
                on_progress(None, f"YT search #{rank}: {clean_title[:40]}")
            path = _download_ytdlp(query, out_path, playlist_item=rank,
                                   on_progress=on_progress).path
            if path and path.exists():
                duration = _get_duration(path)
                if duration and duration > PREVIEW_MAX_SECS:
                    log.info(f"YouTube fallback succeeded ({duration:.0f}s): {out_path.name}")
                    return path, duration
                log.warning(
                    f"YouTube result #{rank} too short ({duration or 0:.0f}s), trying next"
                )
                if path.exists():
                    path.unlink()
            _cleanup_stem_outputs(out_path)

    log.error(f"Could not find full version of '{title}' by '{artist}' on YouTube")
    return None


# ── Duration check ────────────────────────────────────────────────────────────


def _get_duration(path: Path) -> Optional[float]:
    """
    Use ffprobe to get the duration of an audio file in seconds.
    Returns None if ffprobe is unavailable or the file is unreadable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # ffprobe unavailable — fall back to file size heuristic
    # 128kbps MP3 ≈ 16KB/s, so <560KB is likely a 35s preview
    size_kb = path.stat().st_size / 1024
    if size_kb < 560:
        return 30.0   # assume preview
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe(name: str, max_len: int = 40) -> str:
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[\s-]+', '_', name)
    return name[:max_len]