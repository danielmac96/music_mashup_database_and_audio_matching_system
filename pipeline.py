"""
pipeline.py — Orchestrates the full mashup engine pipeline.

Stages:
  1. ingest   → fetch track metadata from SoundCloud playlist URL
  2. download → download audio files via yt-dlp
  3. stems    → separate vocals / instrumental with Demucs
  4. analyse  → extract audio features with librosa
  5. match    → find best mashup candidates for a seed song

Status taxonomy (songs.status, in lifecycle order):
    queued → downloaded → stemmed → analysed
Failure terminals: error_download, error_stems, error_analysis.

Each stage filters by status (the contract), with file-existence as a sanity
guard. A failure inside one track sets that track's error_* status and the
stage continues with the next track.
"""
import sys
import logging
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import BEAT_TRIM_SECS, TOP_K_RESULTS, format_duration
from database.models import (
    init_db, upsert_song, update_song_status, update_song_duration,
    upsert_stem, upsert_features, get_all_songs, get_songs_by_status,
    get_features_for_song, replace_sections,
)
from ingest.soundcloud   import fetch_playlist
from downloader.download import download_track
from stems.separate      import separate
from analysis.analyze    import analyze_file
from analysis.structure  import detect_sections

log = logging.getLogger(__name__)


def _fmt_secs(secs: float) -> str:
    return format_duration(secs) or "0:00"


# ── Logging helpers ──────────────────────────────────────────────────────────

def _stage_header(num: int, name: str) -> None:
    log.info(f"-- Stage {num}: {name} " + "-" * (52 - len(name)))


def _track_header(idx: int, total: int, song: dict, verb: str) -> None:
    title = song.get("title") or "Untitled"
    artist = song.get("artist") or "Unknown"
    log.info(f"  [{idx}/{total}] {verb}: \"{title}\" — {artist}")
    src = song.get("source_url")
    if src:
        log.info(f"         Source: {src}")


def _track_note(msg: str) -> None:
    log.info(f"         -> {msg}")


def _stage_summary(stage_num: int, counters: dict) -> None:
    parts = [f"{v} {k}" for k, v in counters.items() if v]
    if not parts:
        parts = ["nothing to do"]
    log.info(f"  Stage {stage_num} summary: " + ", ".join(parts))


# ── Stages ───────────────────────────────────────────────────────────────────

def run_ingest(playlist_url: str) -> list:
    _stage_header(1, "Ingest")
    log.info(f"  Fetching playlist: {playlist_url}")
    tracks = fetch_playlist(playlist_url)
    if not tracks:
        log.error("  No tracks returned from yt-dlp.")
        _stage_summary(1, {"ingested": 0})
        return []

    song_ids = []
    for idx, t in enumerate(tracks, 1):
        try:
            sid = upsert_song(
                title=t["title"],
                artist=t["artist"],
                source_url=t["source_url"],
                duration_secs=t.get("duration_secs") or 0,
                genre=t.get("genre", ""),
                status="queued",
                artist_id=t.get("artist_id", ""),
                track_id=t.get("track_id", ""),
                duration_str=t.get("duration_str", ""),
                upload_date=t.get("upload_date", ""),
                likes=t.get("likes", 0),
                reposts=t.get("reposts", 0),
                comments=t.get("comments", 0),
                plays=t.get("plays", 0),
                thumbnail=t.get("thumbnail", ""),
                tags=t.get("tags", ""),
                release_year=t.get("release_year", 0),
            )
            song_ids.append(sid)
            log.info(f"  [{idx}/{len(tracks)}] Ingested [{sid}] {t['title']} — {t['artist']}")
        except Exception:
            log.warning(f"  [{idx}/{len(tracks)}] Failed to upsert track {t.get('title')!r}:\n{traceback.format_exc()}")

    _stage_summary(1, {"ingested": len(song_ids), "in DB total": len(get_all_songs())})
    return song_ids


def run_download() -> dict:
    _stage_header(2, "Download")
    pending = get_songs_by_status("queued", "error_download")
    skipped_done = [
        s for s in get_all_songs()
        if s["status"] in ("downloaded", "stemmed", "analysed")
    ]

    counters = {"downloaded": 0, "skipped (already done)": len(skipped_done), "error": 0}
    downloaded = {}

    if not pending:
        log.info("  Nothing to download (every track already past this stage).")
        _stage_summary(2, counters)
        return downloaded

    total = len(pending)
    for idx, song in enumerate(pending, 1):
        sid = song["id"]
        _track_header(idx, total, song, "Downloading")
        try:
            outcome = download_track(sid, song["title"], song["source_url"], artist=song["artist"])
        except Exception:
            log.warning(f"         Exception during download:\n{traceback.format_exc()}")
            outcome = None

        if outcome and outcome.path.exists():
            update_song_status(sid, "downloaded", raw_path=str(outcome.path))
            if outcome.duration_secs is not None:
                update_song_duration(sid, outcome.duration_secs)
                _track_note(
                    f"Got full version ({_fmt_secs(outcome.duration_secs)}) "
                    f"from YouTube fallback"
                )
            else:
                _track_note(f"Saved to {outcome.path.name}")
            downloaded[sid] = outcome.path
            counters["downloaded"] += 1
        else:
            update_song_status(sid, "error_download")
            _track_note("Download FAILED — see warnings above")
            counters["error"] += 1

    _stage_summary(2, counters)
    return downloaded


def run_stems() -> dict:
    _stage_header(3, "Stem separation")
    pending = get_songs_by_status("downloaded", "error_stems")
    skipped_done = get_songs_by_status("stemmed", "analysed")

    counters = {"stemmed": 0, "skipped (already done)": len(skipped_done), "error": 0}
    results = {}

    if not pending:
        log.info("  Nothing to separate (every downloaded track already stemmed).")
        _stage_summary(3, counters)
        return results

    total = len(pending)
    for idx, song in enumerate(pending, 1):
        sid = song["id"]
        _track_header(idx, total, song, "Separating")
        raw = song.get("raw_path") or ""
        path = Path(raw) if raw else None
        if not path or not path.exists():
            _track_note(f"Raw file missing on disk ({raw!r}), marking error_stems")
            update_song_status(sid, "error_stems")
            counters["error"] += 1
            continue

        try:
            stems = separate(sid, song["title"], path, artist=song["artist"])
        except Exception:
            log.warning(f"         Exception during separation:\n{traceback.format_exc()}")
            stems = None

        if stems:
            sep_tag = stems.get("separator")  # None when existing files reused
            upsert_stem(sid, "vocals",        str(stems["vocals"]), separator=sep_tag)
            upsert_stem(sid, "instrumental",  str(stems["instrumental"]), separator=sep_tag)
            upsert_stem(sid, "full",          str(path))
            update_song_status(sid, "stemmed")
            results[sid] = stems
            _track_note("vocals + instrumental ready")
            counters["stemmed"] += 1
        else:
            update_song_status(sid, "error_stems")
            _track_note("Demucs FAILED — see warnings above")
            counters["error"] += 1

    _stage_summary(3, counters)
    return results


def run_analysis() -> dict:
    _stage_header(4, "Analysis")
    pending = get_songs_by_status("stemmed", "error_analysis")
    skipped_done = get_songs_by_status("analysed")

    counters = {"analysed": 0, "skipped (already done)": len(skipped_done), "error": 0}
    results = {}

    if not pending:
        log.info("  Nothing to analyse (every stemmed track already analysed).")
        _stage_summary(4, counters)
        return results

    from database.models import get_conn
    conn = get_conn()
    stem_rows = conn.execute("SELECT song_id, stem_type, file_path FROM stems").fetchall()
    conn.close()

    stem_map = {}
    for row in stem_rows:
        stem_map.setdefault(row["song_id"], {})[row["stem_type"]] = row["file_path"]

    total = len(pending)
    for idx, song in enumerate(pending, 1):
        sid = song["id"]
        _track_header(idx, total, song, "Analysing")
        stems_for_song = stem_map.get(sid, {})
        if not stems_for_song:
            _track_note("No stems table rows — marking error_analysis")
            update_song_status(sid, "error_analysis")
            counters["error"] += 1
            continue

        per_stem = {}
        any_failure = False
        for stem_type in ("full", "vocals", "instrumental"):
            fp = stems_for_song.get(stem_type, "")
            audio_path = Path(fp) if fp else None
            if not audio_path or not audio_path.exists():
                _track_note(f"[{stem_type}] file missing on disk, skipping this stem")
                any_failure = True
                continue
            try:
                features = analyze_file(audio_path, trim_secs=BEAT_TRIM_SECS)
            except Exception:
                log.warning(f"         Exception analysing {stem_type}:\n{traceback.format_exc()}")
                any_failure = True
                continue
            if not features:
                _track_note(f"[{stem_type}] analyse_file returned empty features")
                any_failure = True
                continue
            upsert_features(sid, stem_type, features.copy())
            per_stem[stem_type] = features
            _track_note(
                f"[{stem_type}] BPM={features.get('bpm', '?')} "
                f"Key={features.get('key', '?')} {features.get('mode', '')} "
                f"Camelot={features.get('camelot', '?')}"
            )

        # Structure detection (sections with chorus/verse/drop timestamps).
        # Non-fatal: matching still works without sections, just less precise.
        full_fp = stems_for_song.get("full", "")
        if per_stem and full_fp and Path(full_fp).exists():
            vocals_fp = stems_for_song.get("vocals", "")
            vocals_path = Path(vocals_fp) if vocals_fp else None
            try:
                sections = detect_sections(Path(full_fp), vocals_path)
                if sections:
                    replace_sections(sid, sections)
                    _track_note(
                        f"[structure] {len(sections)} sections: "
                        + ", ".join(f"{s['label']} @{_fmt_secs(s['start_sec'])}"
                                    for s in sections)
                    )
            except Exception:
                log.warning(f"         Exception detecting structure:\n{traceback.format_exc()}")
                _track_note("[structure] detection failed — continuing without sections")

        if per_stem and not any_failure:
            update_song_status(sid, "analysed")
            results[sid] = per_stem
            counters["analysed"] += 1
        elif per_stem:
            update_song_status(sid, "analysed")
            results[sid] = per_stem
            counters["analysed"] += 1
            _track_note("Partial success — some stems missing but core features stored")
        else:
            update_song_status(sid, "error_analysis")
            counters["error"] += 1
            _track_note("All stems failed — marking error_analysis")

    _stage_summary(4, counters)
    return results


def run_reverify() -> dict:
    """Re-check every already-downloaded track for stale ~30s SoundCloud Go+
    previews (WORKFLOW_AUDIT ISSUE-1). Refreshes DB duration from the real file,
    and when a preview is swapped for a full-length track, resets that song to
    'downloaded' so a following stems/analysis run reprocesses the new audio."""
    from downloader.download import reverify_track

    _stage_header(0, "Re-verify cached downloads")
    songs = get_all_songs()
    counters = {"checked": 0, "duration refreshed": 0,
                "replaced (full re-download)": 0, "error": 0}

    for idx, song in enumerate(songs, 1):
        sid = song["id"]
        if song["status"] in ("queued", "error_download"):
            continue  # nothing downloaded yet — normal download stage handles it
        counters["checked"] += 1
        _track_header(idx, len(songs), song, "Re-verifying")
        try:
            res = reverify_track(sid, song["title"], song["source_url"],
                                 artist=song.get("artist") or "")
        except Exception:
            log.warning(f"         Exception during re-verify:\n{traceback.format_exc()}")
            counters["error"] += 1
            continue

        if not res.path:
            _track_note("No full-length version available")
            counters["error"] += 1
            continue

        if res.duration_secs:
            update_song_duration(sid, res.duration_secs)
            counters["duration refreshed"] += 1
        if res.replaced:
            update_song_status(sid, "downloaded")
            counters["replaced (full re-download)"] += 1
            _track_note(f"Preview replaced with full track "
                        f"({_fmt_secs(res.duration_secs)}) — reset to re-stem/analyse")
        else:
            _track_note(f"OK ({_fmt_secs(res.duration_secs)})")

    _stage_summary(0, counters)
    return counters


def run_match(seed_song_id: int = 1,
              seed_stem: str = "vocals",
              candidate_stem: str = "instrumental") -> dict:
    _stage_header(5, "Match")
    from matcher.match import score_all_pairs, find_matches, format_results
    from database.models import get_conn

    all_pairs = score_all_pairs()
    vi_count  = len(all_pairs["vocal_over_instrumental"])
    ii_count  = len(all_pairs["instrumental_over_instrumental"])
    log.info(f"  vocal->instrumental pairs scored: {vi_count}")
    log.info(f"  instrumental->instrumental pairs scored: {ii_count}")

    conn = get_conn()
    row = conn.execute("SELECT title, artist FROM songs WHERE id=?",
                       (seed_song_id,)).fetchone()
    conn.close()
    seed_title = f"{row['title']} — {row['artist']}" if row else f"Song #{seed_song_id}"

    vi_results = find_matches(seed_song_id, top_k=TOP_K_RESULTS,
                              seed_role="vocal",
                              combo_type="vocal_over_instrumental")
    print(format_results(vi_results, seed_title=seed_title,
                         combo_type="vocal_over_instrumental"))

    ii_results = find_matches(seed_song_id, top_k=TOP_K_RESULTS,
                              seed_role="vocal",
                              combo_type="instrumental_over_instrumental")
    print(format_results(ii_results, seed_title=seed_title,
                         combo_type="instrumental_over_instrumental"))

    return all_pairs


