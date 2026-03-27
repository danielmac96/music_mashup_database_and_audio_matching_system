"""
pipeline.py — Orchestrates the full mashup engine pipeline.

Stages:
  1. ingest   → fetch track metadata from SoundCloud playlist URL
  2. download → download audio files via yt-dlp
  3. stems    → separate vocals / instrumental with Demucs
  4. analyse  → extract audio features with librosa
  5. match    → find best mashup candidates for a seed song
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import BEAT_TRIM_SECS, TOP_K_RESULTS
from database.models import (
    init_db, upsert_song, update_song_status, update_song_duration,
    upsert_stem, upsert_features, get_all_songs,
    get_features_for_song,
)
from ingest.soundcloud   import fetch_playlist
from downloader.download import download_track
from stems.separate      import separate
from analysis.analyze    import analyze_file
from matcher.match       import find_matches, format_results

log = logging.getLogger(__name__)


def run_ingest(playlist_url: str) -> list:
    log.info("── Stage 1: Ingest ──────────────────────────────────────")
    tracks = fetch_playlist(playlist_url)
    song_ids = []
    for t in tracks:
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
        )
        song_ids.append(sid)
        log.info(f"  Ingested [{sid}] {t['title']} — {t['artist']}")
    log.info(f"  Total: {len(song_ids)} tracks in database")
    return song_ids


def run_download() -> dict:
    log.info("── Stage 2: Download ────────────────────────────────────")
    songs = [s for s in get_all_songs() if s["status"] in ("queued", "error")]
    downloaded = {}
    for song in songs:
        sid   = song["id"]
        title = song["title"]
        url   = song["source_url"]
        log.info(f"  Downloading [{sid}] {title}")
        outcome = download_track(sid, title, url, artist=song["artist"])
        if outcome and outcome.path.exists():
            update_song_status(sid, "downloaded", raw_path=str(outcome.path))
            if outcome.duration_secs is not None:
                update_song_duration(sid, outcome.duration_secs)
                log.info(f"    Updated duration to {outcome.duration_secs:.1f}s (YouTube source)")
            downloaded[sid] = outcome.path
            log.info(f"    ✓ {outcome.path.name}")
        else:
            update_song_status(sid, "error")
            log.warning(f"    ✗ Failed: {title}")
    log.info(f"  Downloaded: {len(downloaded)}/{len(songs)}")
    return downloaded


def run_stems() -> dict:
    log.info("── Stage 3: Stem separation ─────────────────────────────")
    songs = [s for s in get_all_songs()
             if s["raw_path"] and Path(s["raw_path"]).exists()]
    results = {}
    for song in songs:
        sid   = song["id"]
        title = song["title"]
        path  = Path(song["raw_path"])
        log.info(f"  Separating [{sid}] {title}")
        stems = separate(sid, title, path, artist=song["artist"])
        if stems:
            upsert_stem(sid, "vocals",        str(stems["vocals"]))
            upsert_stem(sid, "instrumental",  str(stems["instrumental"]))
            upsert_stem(sid, "full",          str(path))
            update_song_status(sid, "stemmed")
            results[sid] = stems
            log.info(f"    ✓ vocals + instrumental")
        else:
            update_song_status(sid, "error")
            log.warning(f"    ✗ Separation failed: {title}")
    log.info(f"  Stemmed: {len(results)}/{len(songs)}")
    return results


def run_analysis() -> dict:
    log.info("── Stage 4: Analysis ────────────────────────────────────")
    from database.models import get_conn
    conn = get_conn()
    stem_rows = conn.execute("SELECT song_id, stem_type, file_path FROM stems").fetchall()
    conn.close()

    stem_map = {}
    for row in stem_rows:
        if Path(row["file_path"]).exists():
            stem_map.setdefault(row["song_id"], {})[row["stem_type"]] = row["file_path"]

    songs   = [s for s in get_all_songs() if s["id"] in stem_map]
    results = {}

    for song in songs:
        sid   = song["id"]
        title = song["title"]
        log.info(f"  Analysing [{sid}] {title}")
        results[sid] = {}
        stems_for_song = stem_map.get(sid, {})

        for stem_type in ("full", "vocals", "instrumental"):
            fp = stems_for_song.get(stem_type, "")
            audio_path = Path(fp) if fp else None
            if audio_path and audio_path.exists():
                features = analyze_file(audio_path, trim_secs=BEAT_TRIM_SECS)
            else:
                log.warning(f"    No audio for stem '{stem_type}', skipping")
                continue
            upsert_features(sid, stem_type, features.copy())
            results[sid][stem_type] = features
            log.info(f"    [{stem_type}] BPM={features['bpm']}  "
                     f"Key={features['key']} {features['mode']}  "
                     f"Camelot={features['camelot']}")

        update_song_status(sid, "analysed")

    log.info(f"  Analysed: {len(results)}/{len(songs)}")
    return results


def run_match(seed_song_id: int = 1,
              seed_stem: str = "vocals",
              candidate_stem: str = "instrumental") -> dict:
    log.info("── Stage 5: Match ───────────────────────────────────────")
    from matcher.match import score_all_pairs, find_matches, format_results
    from database.models import get_conn

    all_pairs = score_all_pairs()
    vi_count  = len(all_pairs["vocal_over_instrumental"])
    ii_count  = len(all_pairs["instrumental_over_instrumental"])
    log.info(f"  vocal→instrumental pairs scored: {vi_count}")
    log.info(f"  instrumental→instrumental pairs scored: {ii_count}")

    conn = get_conn()
    row = conn.execute("SELECT title, artist FROM songs WHERE id=?",
                       (seed_song_id,)).fetchone()
    conn.close()
    seed_title = f"{row['title']} — {row['artist']}" if row else f"Song #{seed_song_id}"

    # Show vocal-over-instrumental matches
    vi_results = find_matches(seed_song_id, top_k=TOP_K_RESULTS,
                              seed_role="vocal",
                              combo_type="vocal_over_instrumental")
    print(format_results(vi_results, seed_title=seed_title,
                         combo_type="vocal_over_instrumental"))

    # Show instrumental-over-instrumental matches
    ii_results = find_matches(seed_song_id, top_k=TOP_K_RESULTS,
                              seed_role="vocal",
                              combo_type="instrumental_over_instrumental")
    print(format_results(ii_results, seed_title=seed_title,
                         combo_type="instrumental_over_instrumental"))

    return all_pairs