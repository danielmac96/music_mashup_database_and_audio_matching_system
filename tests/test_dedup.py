"""A.2 — near-duplicate suppression (matcher/dedup.py).

A SoundCloud library holds the same record many times. Those uploads agree on
every sub-score by construction, so without suppression they take the top of the
ranked list with pairings that are not mashups.

sqlite + numpy only; no audio, no network.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from matcher.dedup import (  # noqa: E402
    cluster_variants, normalise_artist, normalise_title, variant_key,
)


# ── normalise_title ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Levels", "levels"),
    ("Levels (Original Mix)", "levels"),
    ("Levels (Extended Mix)", "levels"),
    ("Levels [Radio Edit]", "levels"),
    ("Levels - Extended Mix", "levels"),
    ("Levels - Radio Edit", "levels"),
    ("Levels (Skrillex Remix)", "levels"),
    ("Levels (Dimitri Vegas Bootleg)", "levels"),
    ("Levels (VIP)", "levels"),
    ("Levels (Official Video)", "levels"),
    ("Levels (Remastered 2011)", "levels"),
    ("Levels (feat. Someone)", "levels"),
    ("Levels feat. Someone", "levels"),
    ("01. Levels", "levels"),
    ("Levels (Free Download)", "levels"),
    ("  LEVELS  ", "levels"),
    ("Levels (Extended Mix) [Free Download]", "levels"),
])
def test_normalise_title_strips_version_noise(raw, expected):
    assert normalise_title(raw) == expected


@pytest.mark.parametrize("raw", [
    "Levels (Live at Wembley)",
    "Levels (Acoustic)",
])
def test_normalise_title_keeps_genuinely_different_recordings(raw):
    """A live take or an acoustic version is a different performance, not a
    re-tag of the same master — it must stay pairable."""
    assert normalise_title(raw) != "levels"


def test_normalise_title_empty_is_unkeyable():
    assert normalise_title("") == ""
    assert normalise_title(None) == ""
    assert normalise_title("(Extended Mix)") == ""
    assert variant_key("Artist", "(Original Mix)") == ""


def test_normalise_artist():
    assert normalise_artist("Avicii") == "avicii"
    assert normalise_artist("Avicii feat. Someone") == "avicii"
    assert normalise_artist("Avicii Official") == "avicii"
    assert normalise_artist(None) == ""


# ── cluster_variants ─────────────────────────────────────────────────────────

def _song(sid, title, artist, mfcc=None):
    s = {"song_id": sid, "title": title, "artist": artist}
    if mfcc is not None:
        s["mfcc"] = mfcc
    return s


def test_same_artist_variants_cluster_without_audio():
    """A remix has a completely different timbre from its original, so an MFCC
    gate would reject exactly the pairs this exists to suppress. Same artist +
    same work is enough."""
    songs = [
        _song(1, "Levels", "Avicii"),
        _song(2, "Levels (Extended Mix)", "Avicii"),
        _song(3, "Levels (Skrillex Remix)", "Avicii"),
    ]
    clusters = cluster_variants(songs)
    assert clusters == {1: 1, 2: 1, 3: 1}


def test_different_songs_by_same_artist_do_not_cluster():
    songs = [
        _song(1, "Levels", "Avicii"),
        _song(2, "Wake Me Up", "Avicii"),
    ]
    assert cluster_variants(songs) == {}


def test_singletons_are_absent_from_the_mapping():
    """Absent means NULL means 'no known variants' — callers rely on it."""
    songs = [_song(1, "Levels", "Avicii"), _song(2, "Titanium", "Sia")]
    assert cluster_variants(songs) == {}


def test_reupload_by_a_different_uploader_needs_audio_agreement():
    """The uploader name is not the recording artist. Same title + different
    artist clusters only when the stored timbre agrees."""
    same = [float(i) for i in range(13)]
    songs = [
        _song(1, "Levels", "Avicii", mfcc=same),
        _song(2, "Levels", "EDM Uploads HQ", mfcc=same),
    ]
    assert cluster_variants(songs) == {1: 1, 2: 1}


def test_cover_version_does_not_cluster():
    """Same title, different artist, different timbre — a cover is a genuinely
    different recording and must stay pairable."""
    a = [0.0] + [1.0] * 12
    b = [0.0] + [-1.0] * 12
    songs = [
        _song(1, "Levels", "Avicii", mfcc=a),
        _song(2, "Levels", "Some Cover Band", mfcc=b),
    ]
    assert cluster_variants(songs) == {}


def test_same_title_different_artist_without_mfcc_does_not_cluster():
    """Absent audio is not evidence of sameness; the title alone was already
    judged insufficient here."""
    songs = [_song(1, "Levels", "Avicii"), _song(2, "Levels", "Someone Else")]
    assert cluster_variants(songs) == {}


def test_generic_titles_require_audio_even_for_the_same_artist():
    """'Intro' by one artist is routinely several different tracks."""
    songs = [_song(1, "Intro", "Some DJ"), _song(2, "Intro", "Some DJ")]
    assert cluster_variants(songs) == {}


def test_cluster_id_is_the_smallest_member_and_is_order_stable():
    songs = [
        _song(7, "Levels (Extended Mix)", "Avicii"),
        _song(3, "Levels", "Avicii"),
        _song(9, "Levels (Radio Edit)", "Avicii"),
    ]
    clusters = cluster_variants(songs)
    assert set(clusters.values()) == {3}
    assert cluster_variants(list(reversed(songs))) == clusters


def test_clustering_is_transitive():
    """A~B via artist and B~C via audio must put all three together."""
    same = [float(i) for i in range(13)]
    songs = [
        _song(1, "Levels", "Avicii", mfcc=same),
        _song(2, "Levels (Extended Mix)", "Avicii", mfcc=same),
        _song(3, "Levels", "Random Reupload", mfcc=same),
    ]
    clusters = cluster_variants(songs)
    assert clusters == {1: 1, 2: 1, 3: 1}


def test_untitled_rows_are_skipped_not_merged():
    """Every empty title must not become one giant cluster."""
    songs = [_song(1, "", "A"), _song(2, "", "B"), _song(3, "(Original Mix)", "C")]
    assert cluster_variants(songs) == {}


# ── End-to-end: the pair the ranked list must not contain ────────────────────

@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "test.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


def _add_song(db_path, title, artist, *, bpm, camelot, mfcc_seed):
    from database.models import upsert_features, upsert_song
    sid = upsert_song(title, artist, f"u://{title}/{artist}", 240,
                      status="analysed", db_path=db_path)
    for stem in ("full", "vocals", "instrumental"):
        upsert_features(sid, stem, {
            "bpm": bpm, "key": "C", "mode": "major", "camelot": camelot,
            "loudness_rms": 0.1, "energy": 0.5,
            "mfcc": [0.0] + [float(mfcc_seed)] * 12,
            "spectral_centroid": 2000.0, "spectral_rolloff": 4000.0,
            "zero_crossing_rate": 0.05,
        }, db_path=db_path)
    return sid


def test_variant_pairs_are_not_scored(db_path):
    """The regression this task exists for: Original Mix vocals over Extended
    Mix instrumental scores ~1.0 on every term and tops the ranked list."""
    from database.models import get_conn, init_db
    from matcher.dedup import rebuild_variant_clusters
    from matcher.match import score_all_pairs
    init_db(db_path)

    orig = _add_song(db_path, "Levels", "Avicii",
                     bpm=126.0, camelot="8A", mfcc_seed=1)
    ext = _add_song(db_path, "Levels (Extended Mix)", "Avicii",
                    bpm=126.0, camelot="8A", mfcc_seed=1)
    other = _add_song(db_path, "Titanium", "David Guetta",
                      bpm=126.0, camelot="8A", mfcc_seed=2)

    rebuild_variant_clusters(db_path=db_path)
    conn = get_conn(db_path)
    rows = {r["id"]: r["variant_cluster"] for r in
            conn.execute("SELECT id, variant_cluster FROM songs").fetchall()}
    conn.close()
    assert rows[orig] == rows[ext] == min(orig, ext)
    assert rows[other] is None

    score_all_pairs(db_path=db_path)
    conn = get_conn(db_path)
    pairs = {(r["vocal_song_id"], r["inst_song_id"]) for r in conn.execute(
        "SELECT vocal_song_id, inst_song_id FROM mashup_candidates").fetchall()}
    conn.close()

    assert (orig, ext) not in pairs and (ext, orig) not in pairs
    # The genuinely different track still pairs with both.
    assert (orig, other) in pairs
    assert (other, orig) in pairs


def test_rebuild_only_writes_changed_rows(db_path):
    """It runs after every analysis, so a no-op rebuild must not take a write
    lock on the whole table."""
    from database.models import init_db
    from matcher.dedup import rebuild_variant_clusters
    init_db(db_path)
    _add_song(db_path, "Levels", "Avicii", bpm=126.0, camelot="8A", mfcc_seed=1)
    _add_song(db_path, "Levels (Extended Mix)", "Avicii",
              bpm=126.0, camelot="8A", mfcc_seed=1)

    first = rebuild_variant_clusters(db_path=db_path)
    assert first["n_changed"] == 2 and first["n_clusters"] == 1
    again = rebuild_variant_clusters(db_path=db_path)
    assert again["n_changed"] == 0
