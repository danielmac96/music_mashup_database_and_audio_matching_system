"""
downloader/download.py — Download audio in best quality via yt-dlp.

SoundCloud-first policy:
  We try SoundCloud before searching YouTube, because a title/artist YouTube
  search can grab a different upload of the song. In order:
    1. Anonymous SoundCloud download (no login — the request is never tied to a
       personal account).
    2. Only if SoundCloud won't serve it (auth-gated Go+/private/encrypted, or a
       <35s Go+ preview) do we fall back to a YouTube search for the full song
       via title + artist.

  We deliberately stay anonymous: authenticating with browser cookies would
  attach ToS-violating downloads to a real SoundCloud account. Gated tracks fall
  through to the YouTube fallback instead.

  The fallback is VERIFIED, not "first result longer than 35s": every hit is
  checked by ingest.match_score.assess_substitute — no remix or altered audio,
  the credited artist on the upload, and the same length as the record we were
  linked to. Nothing passing is a clear failure, never a guess.

Output: MP3 in config.RAW_DIR / "{title}_{artist}.mp3"
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple, Optional
import subprocess
import sys
import logging
import re
from pathlib import Path

from config import RAW_DIR, YTDLP_FORMAT, YTDLP_FORMAT_FALLBACK, YTDLP_POSTARGS
from ingest.match_score import duration_agrees, is_rework_title

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

# How many YouTube search hits to rank per query, and how many verified ones to
# try downloading before giving up.
YOUTUBE_SEARCH_MAX_RESULTS = 5

# Optional progress callback. percent is None for status-only updates.
ProgressCb = Optional[Callable[[Optional[int], str], None]]


def expected_length(*values) -> Optional[float]:
    """The first of ``values`` that is a real track length, else None.

    A length at or under PREVIEW_MAX_SECS is a Go+ snippet's, not the record's,
    so it says nothing about how long the full upload should be."""
    for value in values:
        try:
            secs = float(value)
        except (TypeError, ValueError):
            continue
        if secs > PREVIEW_MAX_SECS:
            return secs
    return None


class DownloadResult(NamedTuple):
    """download_track return value. duration_secs is set when audio came from YouTube fallback."""
    path: Path
    duration_secs: Optional[float] = None  # if set, persist to songs.duration_secs
    # Set only when the audio came from somewhere other than the URL we were
    # asked to download — i.e. SoundCloud refused and the YouTube fallback found
    # the track instead. Callers persist it so the row records where the file
    # really came from; None means "the recorded source_url still holds".
    source_url: Optional[str] = None
    # What the fallback substituted and why it was trusted (upload title,
    # uploader, score, lengths) — songs.audio_provenance. None for a direct
    # download of source_url.
    provenance: Optional[dict] = None


class _YtAttempt(NamedTuple):
    label: str
    format_str: str
    player_client: str
    use_cookies: bool


def _youtube_attempts() -> tuple[_YtAttempt, ...]:
    # All attempts are anonymous by design — nothing in the downloader logs in
    # with browser cookies, so downloads are never tied to a personal Google or
    # SoundCloud account. A track that only a logged-in session could fetch just
    # fails the ladder rather than authenticating as the user.
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
    )


def download_track(song_id: int, title: str, source_url: str,
                   artist: str = "",
                   on_progress: ProgressCb = None,
                   expected_duration: Optional[float] = None) -> DownloadResult:
    """Download a track's audio. Returns a DownloadResult on success; raises
    DownloadError with a user-facing reason on failure (never returns None).

    ``expected_duration`` is the length of the record the row was linked to
    (songs.origin_duration_secs). A substitute upload must agree with it."""
    out_path = RAW_DIR / f"{_safe(title)}_{_safe(artist)}.mp3"

    # A direct YouTube source routes through the retry ladder (see _download_ytdlp)
    # and a short result is a genuine short video, not a SoundCloud Go+ preview —
    # so we skip the preview→YouTube-search fallback for it.
    is_yt_source = _is_youtube_like(source_url)
    expected = expected_length(expected_duration)

    if out_path.exists():
        duration = _get_duration(out_path)
        if (duration and duration > PREVIEW_MAX_SECS
                and (is_yt_source or duration_agrees(duration, expected))):
            log.info(f"Already downloaded (full): {out_path.name}")
            if on_progress:
                on_progress(100, "Already downloaded")
            # Pass duration so the worker refreshes the DB row — fixes stale 30s
            # rows seeded from SoundCloud Go+ previews during ingest.
            return DownloadResult(out_path, duration)
        log.warning(f"Existing file is {duration or 0:.0f}s — not the "
                    f"{expected or 0:.0f}s record, or a preview — re-downloading")
        out_path.unlink()

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
            fb = _fallback_youtube(title, artist, out_path, on_progress=on_progress,
                                   expected_duration=expected)
            if fb.result:
                return _from_fallback(fb.result)
            raise DownloadError(
                "SoundCloud served only a Go+ 30s preview and no matching "
                f"full-length YouTube upload was found. {fb.note}".strip(),
                kind="premium")

    if path and path.exists():
        return DownloadResult(path, _get_duration(path))

    kind, msg = classify_download_error(dl.error_lines)

    # SoundCloud has the track but won't serve it (DRM / Go+ / geo) — the same
    # YouTube title search used for Go+ previews can still get the full song.
    if not is_yt_source and kind in _GATED_KINDS:
        log.warning(f"SoundCloud blocked this track ({kind}) — trying YouTube fallback")
        if on_progress:
            on_progress(None, "SoundCloud blocked this track — searching YouTube…")
        fb = _fallback_youtube(title, artist, out_path, on_progress=on_progress,
                               expected_duration=expected)
        if fb.result:
            return _from_fallback(fb.result)
        msg = f"{msg} No matching YouTube upload was found either. {fb.note}".strip()

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
                   on_progress: ProgressCb = None,
                   expected_duration: Optional[float] = None) -> ReverifyResult:
    """Re-check a previously-downloaded track.

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
                                on_progress=on_progress,
                                expected_duration=expected_duration)
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
    resolved_url: Optional[str] = None


# yt-dlp announces the video it actually settled on before downloading it:
#   [youtube] Extracting URL: https://www.youtube.com/watch?v=<id>
#   [youtube] <id>: Downloading webpage
# When a search picked the video, that line is the only record of *which* upload
# the audio came from — see _resolved_youtube_url.
_YT_URL_RE = re.compile(
    r"https?://(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})")
_YT_ID_RE = re.compile(r"^\[youtube\]\s+([A-Za-z0-9_-]{11}):", re.MULTILINE)


def _resolved_youtube_url(stdout: str) -> Optional[str]:
    """The canonical watch URL yt-dlp actually downloaded, from its own output.

    Returns None when the id can't be read off — callers must then leave the
    recorded source URL alone rather than guess at provenance."""
    m = _YT_URL_RE.search(stdout or "") or _YT_ID_RE.search(stdout or "")
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else None


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

    resolved = _resolved_youtube_url(result.stdout)

    if out_path.exists():
        return _RunOutcome(True, [], resolved)

    for candidate in out_path.parent.glob(f"{out_path.stem}.*"):
        candidate.rename(out_path)
        return _RunOutcome(True, [], resolved)

    log.warning("yt-dlp exited 0 but output file not found")
    return _RunOutcome(False, ["ERROR: yt-dlp exited 0 but produced no output file"])


class _DlOutcome(NamedTuple):
    path: Optional[Path]
    error_lines: list
    resolved_url: Optional[str] = None


def _download_soundcloud(
    url: str,
    out_path: Path,
    *,
    playlist_item: Optional[int] = None,
    on_progress: ProgressCb = None,
) -> _DlOutcome:
    """Get the audio straight from SoundCloud with a single anonymous request.

    We stay anonymous by design: authenticating with browser cookies would tie
    ToS-violating downloads to a real SoundCloud account. So when a track is
    auth-gated (Go+/private/encrypted → "no formats"), we let it fail here and
    the caller falls back to a YouTube title search rather than logging in."""
    run = _run_ytdlp(
        url, out_path, YTDLP_FORMAT, extractor_args=None,
        use_cookies=False, playlist_item=playlist_item, on_progress=on_progress,
    )
    if run.ok and out_path.exists():
        log.info(f"Downloaded from SoundCloud: {out_path.name}")
        return _DlOutcome(out_path, [], run.resolved_url)

    log.error(f"SoundCloud download failed for {url[:120]}")
    return _DlOutcome(None, run.error_lines)


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
        return _download_soundcloud(
            url, out_path, playlist_item=playlist_item, on_progress=on_progress)

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
            return _DlOutcome(out_path, [], run.resolved_url)
        if run.error_lines:
            last_errors = run.error_lines

    log.error(f"yt-dlp failed for URL after all YouTube retries: {url[:120]}")
    return _DlOutcome(None, last_errors)


# ── YouTube fallback ──────────────────────────────────────────────────────────


def _usable_search_terms(title: str, artist: str) -> bool:
    """True when there's enough real metadata to run a meaningful YouTube
    search. A missing/"Unknown" title with no artist would only return a
    random unrelated video, so we treat that as unusable."""
    t = (title or "").strip().lower()
    if t in ("", "unknown"):
        t = ""
    return bool(t or (artist or "").strip())


_BRACKET_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


def _search_title(title: str) -> str:
    """The title as a search query: bracketed noise ("(Official Audio)",
    "[Free DL]") dropped, but a rework credit kept — searching "Levels" for
    "Levels (Skrillex Remix)" would only find the original."""
    kept = _BRACKET_RE.sub(
        lambda m: m.group(0) if is_rework_title(m.group(0)) else "", title or "")
    return kept.strip() or (title or "").strip()


def youtube_candidates(title: str, artist: str,
                       expected_duration: Optional[float] = None, *,
                       exhaustive: bool = False,
                       limit: int = YOUTUBE_SEARCH_MAX_RESULTS) -> list[dict]:
    """Ranked YouTube uploads that might stand in for this record.

    Each hit is a search row (url, title, uploader, duration_secs, score,
    artist_score) annotated with ``passes``, ``reason`` and ``duration_delta``
    from ingest.match_score.assess_substitute — the one gate both the download
    fallback and the "Wrong audio?" picker use. Passing hits come first.

    Stops after the first query that yields a passing hit unless ``exhaustive``.
    Scoring uses the full title, so a requested remix is not mistaken for an
    unrequested one."""
    from ingest.match_score import assess_substitute
    from ingest.soundcloud import search_candidates

    if not _usable_search_terms(title, artist):
        return []
    name = (artist or "").strip()
    base = " ".join(p for p in (name, _search_title(title)) if p)
    seen: dict[str, dict] = {}
    for query in (base, f"{base} official audio"):
        for hit in search_candidates(name, title, platform="youtube",
                                     limit=limit, query=query):
            if hit["url"] in seen:
                continue
            verdict = assess_substitute(name, title, hit, expected_duration)
            seen[hit["url"]] = {**hit, "passes": verdict.passes,
                                "reason": verdict.reason,
                                "duration_delta": verdict.duration_delta}
        if not exhaustive and any(h["passes"] for h in seen.values()):
            break
    return sorted(seen.values(), key=lambda h: (h["passes"], h["score"]),
                  reverse=True)


class _FallbackResult(NamedTuple):
    """What the YouTube fallback actually fetched. ``url`` is the watch URL of
    the upload the audio came from, so the caller can record where the file
    really came from instead of leaving a stale SoundCloud URL on the row."""
    path: Path
    duration_secs: float
    url: Optional[str]
    provenance: dict


class _FallbackOutcome(NamedTuple):
    result: Optional[_FallbackResult]
    # Why nothing was accepted, for the user-facing error. '' on success.
    note: str = ""


def _from_fallback(fb: _FallbackResult) -> DownloadResult:
    return DownloadResult(fb.path, fb.duration_secs, fb.url, fb.provenance)


def _mmss(secs: Optional[float]) -> str:
    if not secs:
        return "?"
    m, s = divmod(int(round(secs)), 60)
    return f"{m}:{s:02d}"


def _no_match_note(hits: list[dict], expected: Optional[float]) -> str:
    if not hits:
        return "YouTube search returned nothing."
    rejected = [h for h in hits if not h["passes"]]
    if not rejected:
        return "The matching uploads could not be downloaded."
    best = rejected[0]
    length = _mmss(best.get("duration_secs"))
    if expected:
        length += f" vs expected {_mmss(expected)}"
    return (f"Closest: '{best['title']}' by {best.get('uploader') or '?'} "
            f"({length}) — rejected: {best['reason']}.")


def _fallback_youtube(title: str, artist: str, out_path: Path,
                      on_progress: ProgressCb = None,
                      expected_duration: Optional[float] = None) -> _FallbackOutcome:
    """Find this record on YouTube and download it — only if an upload passes
    verification (see youtube_candidates). Tries the verified hits best-first,
    and re-checks the downloaded file's real length, since a search listing's
    duration is the video's and a download can still come back as something else.
    """
    # Guard: a title-based YouTube search only makes sense with real search
    # terms. If metadata extraction failed upstream we may have "Unknown"/""
    # here — searching YouTube for that returns an arbitrary unrelated video
    # (the exact "wrong audio" users hit). Bail out so the caller surfaces a
    # clear error instead of ingesting a random track.
    if not _usable_search_terms(title, artist):
        log.error(
            "Skipping YouTube fallback — no usable title/artist to search "
            f"(title={title!r}, artist={artist!r})"
        )
        return _FallbackOutcome(None, "No usable title or artist to search for.")

    if on_progress:
        on_progress(None, f"Searching YouTube: {(title or '')[:40]}")
    hits = youtube_candidates(title, artist, expected_duration)
    accepted = [h for h in hits if h["passes"]]

    for rank, hit in enumerate(accepted[:YOUTUBE_SEARCH_MAX_RESULTS], start=1):
        log.info(f"YouTube fallback #{rank}: '{hit['title']}' by {hit.get('uploader')} "
                 f"({hit.get('duration_secs') or 0:.0f}s, score {hit['score']})")
        if on_progress:
            on_progress(None, f"YouTube: {hit['title'][:50]}")
        dl = _download_ytdlp(hit["url"], out_path, on_progress=on_progress)
        path = dl.path
        if path and path.exists():
            duration = _get_duration(path)
            if (duration and duration > PREVIEW_MAX_SECS
                    and duration_agrees(duration, expected_duration)):
                url = dl.resolved_url or hit["url"]
                log.info(f"YouTube fallback succeeded ({duration:.0f}s): "
                         f"{out_path.name} from {url}")
                return _FallbackOutcome(_FallbackResult(path, duration, url, {
                    "via": "youtube_fallback",
                    "url": url,
                    "title": hit["title"],
                    "uploader": hit.get("uploader") or "",
                    "score": hit["score"],
                    "duration_secs": round(duration, 1),
                    "expected_secs": round(expected_duration, 1)
                                     if expected_duration else None,
                }))
            log.warning(f"YouTube upload downloaded as {duration or 0:.0f}s — "
                        "not the expected record, trying next")
            path.unlink()
        _cleanup_stem_outputs(out_path)

    note = _no_match_note(hits, expected_duration)
    log.error(f"No verified YouTube upload of '{title}' by '{artist}'. {note}")
    return _FallbackOutcome(None, note)


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
