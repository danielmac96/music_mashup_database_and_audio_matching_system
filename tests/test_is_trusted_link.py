"""The training-data gate: which mix-track links are good enough to become
training positives (database.models.is_trusted_link).
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import (  # noqa: E402
    AUTO_LINK_MIN_ARTIST, AUTO_LINK_MIN_DURATION, AUTO_LINK_MIN_SCORE,
)
from database.models import is_trusted_link  # noqa: E402

_GOOD_SCORE = AUTO_LINK_MIN_SCORE + 0.1
_GOOD_DUR = AUTO_LINK_MIN_DURATION + 10


def test_human_and_page_sourced_links_are_trusted():
    for status in ("manual", "resolved", "scraped"):
        assert is_trusted_link(status, None, None) is True


def test_unresolved_and_failed_are_never_trusted():
    for status in ("unresolved", "failed", "", None):
        assert is_trusted_link(status, 1.0, 300.0) is False


def test_confident_auto_link_is_trusted():
    assert is_trusted_link("auto", _GOOD_SCORE, _GOOD_DUR, 1.0) is True


def test_auto_link_below_the_score_floor_is_not_trusted():
    assert is_trusted_link("auto", AUTO_LINK_MIN_SCORE - 0.01, _GOOD_DUR, 1.0) is False


def test_preview_length_auto_link_is_not_trusted():
    assert is_trusted_link("auto", _GOOD_SCORE, 30.0, 1.0) is False


def test_auto_link_without_artist_agreement_is_not_trusted():
    """A strong title-only match against a different artist — the mislink the
    artist floor exists to catch — must not become training data."""
    assert is_trusted_link("auto", _GOOD_SCORE, _GOOD_DUR, 0.0) is False
    assert is_trusted_link("auto", _GOOD_SCORE, _GOOD_DUR,
                           AUTO_LINK_MIN_ARTIST - 0.01) is False


def test_missing_artist_score_skips_the_artist_check():
    """Rows linked before the column existed have no artist score; they keep
    their old verdict rather than being retroactively distrusted."""
    assert is_trusted_link("auto", _GOOD_SCORE, _GOOD_DUR, None) is True
    assert is_trusted_link("auto", _GOOD_SCORE, _GOOD_DUR) is True
