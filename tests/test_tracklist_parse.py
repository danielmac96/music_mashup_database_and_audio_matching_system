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
    # encoding= is load-bearing here, not tidiness. festival_set.txt is UTF-8
    # and the only fixture with non-ASCII separators; decoded as the Windows
    # ANSI codepage its en dash arrives as three mojibake characters, _SPLIT_RE
    # then finds no " - " to split on, and every line parses as title-only with
    # parse_confidence 0.5 — a parser regression that never happened.
    rows = parse_tracklist(fixture.read_text(encoding="utf-8"))
    snap_path = fixture.with_suffix(fixture.suffix + ".expected.json")
    if os.environ.get("UPDATE_SNAPSHOTS"):
        snap_path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    assert snap_path.exists(), f"missing snapshot {snap_path.name}"
    assert rows == json.loads(snap_path.read_text(encoding="utf-8"))


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
    src = Path(tp.__file__).read_text(encoding="utf-8")
    for banned in ("urllib", "requests", "socket", "http.client", "fastapi"):
        assert banned not in src


# ── row furniture and the search credit ──────────────────────────────────────

from ingest.tracklist_parse import search_credit, search_query, strip_row_furniture

_ROW_DUMP = "Dominic Fike - 3 Nights (Acappella) COLUMBIA (SONY) 240 trioxide (17.4k) Save 18"


def test_row_furniture_is_stripped_from_a_copied_row():
    """Label, votes, IDer and "Save" — copied off 1001tracklists — are not the record."""
    assert strip_row_furniture(_ROW_DUMP) == "Dominic Fike - 3 Nights (Acappella)"
    row = parse_line("w/ [01:58] " + _ROW_DUMP)
    assert (row["artist"], row["title"], row["cue_secs"], row["is_overlay"]) == \
        ("Dominic Fike", "3 Nights (Acappella)", 118, True)
    assert row["raw_label"].endswith("Save 18"), "raw_label stays untouched"


def test_a_capital_title_is_not_mistaken_for_a_label():
    """The label is only removed when the rest of the furniture proves a row dump."""
    assert strip_row_furniture("Kendrick Lamar - HUMBLE.") == "Kendrick Lamar - HUMBLE."
    assert strip_row_furniture("Kendrick Lamar - HUMBLE. TDE 12 bob (1.2k) Save 3") \
        == "Kendrick Lamar - HUMBLE."
    assert strip_row_furniture("Queen - Song (2021)") == "Queen - Song (2021)"


def test_search_credit_names_the_record_not_the_cut():
    cases = {
        ("Dominic Fike", "3 Nights (Acappella)"): "Dominic Fike - 3 Nights",
        ("Whethan ft. Flux Pavilion & MAX", "Savage (Instrumental)"): "Whethan - Savage",
        ("Eminem", "Without Me (Acapella) [INTERSCOPE]"): "Eminem - Without Me",
        ("Avicii", "Levels (Extended Mix)"): "Avicii - Levels",
        ("A", "Song - Radio Edit"): "A - Song",
        ("A", "Wake Me Up ft. Aloe Blacc"): "A - Wake Me Up",
        # somebody's rework is a different record — kept
        ("Zedd & Grey", "The Middle (Dzeko Remix)"): "Zedd & Grey - The Middle (Dzeko Remix)",
        ("Flume", "Never Be Like You [Disclosure Flip]"): "Flume - Never Be Like You [Disclosure Flip]",
        ("Martin Garrix", "Animals (VIP) [SPINNIN]"): "Martin Garrix - Animals (VIP)",
        ("A", "Song (feat. B) (Skrillex Remix)"): "A - Song (Skrillex Remix)",
        ("A", "Song (Two Friends Intro Edit)"): "A - Song (Two Friends Intro Edit)",
        ("", "Just A Title"): "Just A Title",
    }
    for (artist, title), want in cases.items():
        assert search_query(artist, title) == want, (artist, title)


def test_search_credit_never_empties_a_title():
    assert search_credit("A", "(Instrumental)") == ("A", "(Instrumental)")
