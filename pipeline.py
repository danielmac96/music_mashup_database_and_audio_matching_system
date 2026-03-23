"""
pipeline.py — Orchestrates the full mashup engine pipeline.

Each stage is independently skippable so you can re-run just one step
without re-processing everything.

Stages:
  1. ingest   → fetch track metadata from playlist URL or mock
  2. download → download audio files via yt-dlp
  3. stems    → separate vocals / instrumental with Demucs
  4. analyse  → extract audio features with librosa
  5. match    → find best mashup candidates for a seed song
"""
import sys
import logging
from pathlib import Path

# Ensure project root is importable regardless of working directory
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import BEAT_TRIM_SECS, TOP_K_RESULTS
from database.models import (
    init_db, upsert_song, update_song_status,
    upsert_stem, upsert_features, get_all_songs,
    get_features_for_song,
)
from ingest.soundcloud   import fetch_playlist
from downloader.download import download_track
from stems.separate      import separate
from analysis.analyze    import analyze_file
from matcher.match       import find_matches, format_results

log = logging.getLogger(__name__)


# ── Stage 1: Ingest ───────────────────────────────────────────────────────────

def run_ingest(playlist_url: str = "", use_mock: bool = False) -> list[int]:
    """
    Fetch playlist metadata and insert into the database.
    Returns list of inserted song IDs.
    """
    log.info("── Stage 1: Ingest ──────────────────────────────────────")
    tracks = fetch_playlist(playlist_url or "mock", use_mock=use_mock)
    song_ids = []
    for t in tracks:
        sid = upsert_song(
            title=t["title"], artist=t["artist"],
            source_url=t["source_url"],
            duration_secs=t.get("duration_secs", 0),
            genre=t.get("genre", ""),
            status="queued",
        )
        song_ids.append(sid)
        log.info(f"  Ingested [{sid}] {t['title']} — {t['artist']}")
    log.info(f"  Total: {len(song_ids)} tracks in database")
    return song_ids


# ── Stage 2: Download ─────────────────────────────────────────────────────────

def run_download(use_mock: bool = False) -> dict[int, Path]:
    """
    Download all queued tracks. Returns {song_id: audio_path}.
    """
    log.info("── Stage 2: Download ────────────────────────────────────")
    songs = [s for s in get_all_songs() if s["status"] in ("queued", "error")]
    downloaded = {}

    for song in songs:
        sid   = song["id"]
        title = song["title"]
        url   = song["source_url"]

        log.info(f"  Downloading [{sid}] {title}")
        path = download_track(sid, title, url, artist=song["artist"], use_mock=use_mock)

        if path and path.exists():
            update_song_status(sid, "downloaded", raw_path=str(path))
            downloaded[sid] = path
            log.info(f"    ✓ {path.name}")
        else:
            update_song_status(sid, "error")
            log.warning(f"    ✗ Failed: {title}")

    log.info(f"  Downloaded: {len(downloaded)}/{len(songs)}")
    return downloaded


# ── Stage 3: Stems ────────────────────────────────────────────────────────────

def run_stems(use_mock: bool = False) -> dict[int, dict]:
    """
    Separate all downloaded tracks into vocal + instrumental stems.
    Returns {song_id: {vocals: Path, instrumental: Path}}.
    """
    log.info("── Stage 3: Stem separation ─────────────────────────────")
    songs = [s for s in get_all_songs()
             if s["raw_path"] and Path(s["raw_path"]).exists()]
    results = {}

    for song in songs:
        sid   = song["id"]
        title = song["title"]
        path  = Path(song["raw_path"])

        if not path.exists():
            log.warning(f"  Audio file missing: {path}")
            continue

        log.info(f"  Separating [{sid}] {title}")
        stems = separate(sid, title, path, artist=song["artist"], use_mock=use_mock)

        if stems:
            # Register stems in DB
            upsert_stem(sid, "vocals",       str(stems["vocals"]))
            upsert_stem(sid, "instrumental", str(stems["instrumental"]))
            upsert_stem(sid, "full",         str(path))
            update_song_status(sid, "stemmed")
            results[sid] = stems
            log.info(f"    ✓ vocals + instrumental")
        else:
            update_song_status(sid, "error")
            log.warning(f"    ✗ Separation failed: {title}")

    log.info(f"  Stemmed: {len(results)}/{len(songs)}")
    return results


# ── Stage 4: Analyse ──────────────────────────────────────────────────────────

def run_analysis(use_mock: bool = False) -> dict[int, dict]:
    """
    Analyse all stemmed tracks (full + vocal + instrumental).
    Returns {song_id: {stem_type: features}}.
    """
    log.info("── Stage 4: Analysis ────────────────────────────────────")

    # Rebuild stem type → path map from DB
    from database.models import get_conn
    conn = get_conn()
    stem_rows = conn.execute("SELECT song_id, stem_type, file_path FROM stems").fetchall()
    conn.close()

    stem_map = {}  # type: dict
    for row in stem_rows:
        if Path(row["file_path"]).exists():
            stem_map.setdefault(row["song_id"], {})[row["stem_type"]] = row["file_path"]

    # Process any song that has stem files on disk regardless of status
    songs   = [s for s in get_all_songs() if s["id"] in stem_map]
    results = {}

    for song in songs:
        sid   = song["id"]
        title = song["title"]
        log.info(f"  Analysing [{sid}] {title}")
        results[sid] = {}

        stems_for_song = stem_map.get(sid, {})
        # Analyse full + vocals + instrumental
        for stem_type in ("full", "vocals", "instrumental"):
            fp = stems_for_song.get(stem_type, "")
            audio_path = Path(fp) if fp else None

            if use_mock:
                # Pass a seed string unique to this song+stem combo
                seed = f"{title}_{stem_type}"
                features = analyze_file(Path(seed), use_mock=True)
            elif audio_path and audio_path.exists():
                features = analyze_file(audio_path, trim_secs=BEAT_TRIM_SECS)
            else:
                log.warning(f"    No audio for stem '{stem_type}', using mock")
                seed = f"{title}_{stem_type}"
                features = analyze_file(Path(seed), use_mock=True)

            upsert_features(sid, stem_type, features.copy())
            results[sid][stem_type] = features
            log.info(f"    [{stem_type}] BPM={features['bpm']}  "
                     f"Key={features['key']} {features['mode']}  "
                     f"Camelot={features['camelot']}")

        update_song_status(sid, "analysed")

    log.info(f"  Analysed: {len(results)}/{len(songs)}")
    return results


# ── Stage 5: Match ────────────────────────────────────────────────────────────

def run_match(seed_song_id: int,
              seed_stem: str = "vocals",
              candidate_stem: str = "instrumental") -> list[dict]:
    """
    Find the best mashup candidates for the given seed song.

    For Two Friends-style mashups you typically want:
      seed_stem="vocals"          → use seed's vocal features
      candidate_stem="instrumental" → match against candidates' instrumentals
    """
    log.info("── Stage 5: Match ───────────────────────────────────────")

    # Get seed song info for display
    conn = __import__("database.models", fromlist=["get_conn"]).get_conn()
    row = conn.execute("SELECT title, artist FROM songs WHERE id=?",
                       (seed_song_id,)).fetchone()
    conn.close()
    seed_title = f"{row['title']} — {row['artist']}" if row else f"Song #{seed_song_id}"

    log.info(f"  Seed: {seed_title} ({seed_stem})")
    log.info(f"  Comparing against: {candidate_stem} stems")

    results = find_matches(seed_song_id, top_k=TOP_K_RESULTS,
                           seed_stem=seed_stem, candidate_stem=candidate_stem)
    print(format_results(results, seed_title=seed_title))
    return results


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_all(playlist_url: str = "", seed_song_id: int = 1,
            use_mock: bool = False):
    """Run all five stages end-to-end."""
    init_db()
    run_ingest(playlist_url, use_mock=use_mock)
    run_download(use_mock=use_mock)
    run_stems(use_mock=use_mock)
    run_analysis(use_mock=use_mock)
    return run_match(seed_song_id)