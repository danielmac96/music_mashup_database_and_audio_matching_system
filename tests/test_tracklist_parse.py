"""Snapshot + unit tests for the pure tracklist parser (ingest/tracklist_parse).

These NEVER hit the network. Each fixture in tests/fixtures/tracklists/ has a
committed .expected.json snapshot; a site/format change that alters parsing
shows up here as a readable diff. To re-bless snapshots after an intentional
parser change, run this file with UPDATE_SNAPSHOTS=1 and eyeball the git diff.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingest.tracklist_parse import parse_line, parse_tracklist, split_artists  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tracklists"
FIXTURES = sorted(p for p in FIXTURE_DIR.iterdir()
                  if p.suffix in (".txt", ".html"))


# ── snapshots ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_fixture_snapshot(fixture):
    rows = parse_tracklist(fixture.read_text())
    snap_path = fixture.with_suffix(fixture.suffix + ".expected.json")
    if os.environ.get("UPDATE_SNAPSHOTS"):
        snap_path.write_text(json.dumps(rows, indent=1) + "\n")
    assert snap_path.exists(), f"missing snapshot {snap_path.name}"
    assert rows == json.loads(snap_path.read_text())


def test_fixtures_present():
    # The suite is only meaningful with the messy cases committed.
    assert len(FIXTURES) >= 5


# ── targeted behaviours ───────────────────────────────────────────────────────

def test_raw_label_is_untouched_original():
    line = "3. [4:05] Zedd & Grey - The Middle (Dzeko Remix)"
    row = parse_line(line)
    assert row["raw_label"] == line
    assert row["artist"] == "Zedd & Grey"
    assert row["title"] == "The Middle (Dzeko Remix)"
    assert row["remixer"] == "Dzeko"
    assert row["artists"] == ["Zedd", "Grey"]
    assert row["parse_confidence"] == 1.0


def test_id_track_flagged_low_confidence():
    row = parse_line("w/ ID - ID")
    assert row["is_id"] is True
    assert row["is_overlay"] is True
    assert row["parse_confidence"] == pytest.approx(0.2)


def test_vs_mashup_parts():
    row = parse_line("5. A Artist - One vs. B Artist - Two vs. C - Three")
    assert row["mashup_parts"] == ["A Artist - One", "B Artist - Two", "C - Three"]
    # Row itself stays linkable via the first component.
    assert row["artist"] == "A Artist" and row["title"] == "One"


def test_split_artists_variants():
    assert split_artists("Skrillex, Diplo & Justin Bieber") == \
        ["Skrillex", "Diplo", "Justin Bieber"]
    assert split_artists("Martin Garrix x Tiesto") == ["Martin Garrix", "Tiesto"]
    assert split_artists("Eminem feat. Rihanna") == ["Eminem", "Rihanna"]
    assert split_artists("deadmau5 and Kaskade") == ["deadmau5", "Kaskade"]
    # 'x' only splits as a word — artists with x inside a name survive.
    assert split_artists("Charli XCX") == ["Charli XCX"]
    assert split_artists("") == []


def test_remixer_bracket_styles():
    assert parse_line("Flume - Song [Disclosure Flip]")["remixer"] == "Disclosure"
    assert parse_line("A - B (Promise Land Rework)")["remixer"] == "Promise Land"
    assert parse_line("A - B (Extended)") ["remixer"] is None


def test_duplicate_ids_are_kept_other_dupes_dropped():
    rows = parse_tracklist("1. ID - ID\n2. A - B\n3. A - B\n4. ID - ID\n")
    labels = [(r["artist"], r["title"]) for r in rows]
    assert labels == [("ID", "ID"), ("A", "B"), ("ID", "ID")]


def test_no_network_imports():
    # The parser module must stay pure: importable without fastapi/yt-dlp,
    # no urllib/socket usage.
    import ingest.tracklist_parse as tp
    src = Path(tp.__file__).read_text()
    for banned in ("urllib", "requests", "socket", "http.client", "fastapi"):
        assert banned not in src
