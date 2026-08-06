"""Tests for the stems separator provenance tag (Phase 5)."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def db(tmp_path):
    from database import models
    p = tmp_path / "test.db"
    models.init_db(p)
    return models, p


def _tag(models, db_path, song_id, stem_type):
    conn = models.get_conn(db_path)
    row = conn.execute(
        "SELECT separator FROM stems WHERE song_id=? AND stem_type=?",
        (song_id, stem_type)).fetchone()
    conn.close()
    return row["separator"] if row else None


def test_upsert_stem_separator_roundtrip_and_none_preserves(db):
    models, p = db
    sid = models.upsert_song(title="T", source_url="http://x/1", db_path=p)

    models.upsert_stem(sid, "vocals", "/f/v.wav", separator="demucs:htdemucs", db_path=p)
    assert _tag(models, p, sid, "vocals") == "demucs:htdemucs"

    # None means "don't change" (e.g. existing files reused untouched).
    models.upsert_stem(sid, "vocals", "/f/v.wav", separator=None, db_path=p)
    assert _tag(models, p, sid, "vocals") == "demucs:htdemucs"

    # A real re-separation overwrites the tag.
    models.upsert_stem(sid, "vocals", "/f/v.wav", separator="mdx:UVR-MDX-NET-Inst_HQ_3",
                       db_path=p)
    assert _tag(models, p, sid, "vocals") == "mdx:UVR-MDX-NET-Inst_HQ_3"


def test_separator_tag_helper():
    from stems.separate import separator_tag
    assert separator_tag("demucs").startswith("demucs:")
    assert separator_tag("mdx").startswith("mdx:")
