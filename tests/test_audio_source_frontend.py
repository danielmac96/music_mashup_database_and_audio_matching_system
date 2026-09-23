"""Where a track's audio came from, pinned from Python.

A YouTube substitute used to be invisible: "Massive" showed its SoundCloud link
while its stems were cut from a remix. These pin the three surfaces that now say
so — the library chip, the detail line with its picker, the row menu — and that
the picker changes the link through the existing reset-and-reprocess route
rather than a second path.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


PICKER = _read("components/AudioSourcePicker.jsx")
DETAIL = _read("components/TrackDetail.jsx")
TABLE = _read("components/TrackTable.jsx")
ACTIONS = _read("components/TrackActions.jsx")
BULK = _read("components/BulkReprocess.jsx")
SOURCES = _read("sources.js")
API = _read("api.js")
CSS = _read("styles.css")


def test_api_exposes_the_candidates_route_and_a_pick_on_url_change():
    assert "/audio-candidates" in API
    assert re.search(r"updateTrackUrl:\s*\(id, sourceUrl, pick", API)


def test_the_picker_reuses_the_url_change_route_with_the_pick():
    assert "api.audioCandidates(track.id)" in PICKER
    assert "api.updateTrackUrl(track.id, c.url" in PICKER
    # The verdict shown is the server's, never re-derived in the browser.
    assert "c.passes" in PICKER and "c.reason" in PICKER


def test_detail_screen_shows_the_source_and_opens_the_picker():
    assert "<AudioSource " in DETAIL
    assert "<AudioSourcePicker " in DETAIL


def test_row_menu_offers_the_picker():
    assert "<AudioSourcePicker " in ACTIONS
    assert "Wrong audio?" in ACTIONS


def test_library_row_marks_substituted_audio():
    row = TABLE[TABLE.index("function TrackRow("):]
    assert "audioSubstitution(t)" in row
    assert "tt-src" in row


def test_substitution_reads_the_recorded_provenance():
    body = SOURCES[SOURCES.index("export function audioSubstitution"):]
    for via in ("youtube_fallback", "manual"):
        assert f'"{via}"' in body
    assert "unverified" in body


def test_sounds_right_confirms_and_settles_the_label():
    assert "/audio-confirm" in API
    assert "api.confirmAudio(track.id)" in PICKER
    assert "Sounds right" in PICKER
    # Non-greedy to the tag's close: the onPick arrow function contains a ">".
    assert re.search(r"<AudioSource .*?onChanged=\{onChanged\}.*?/>", DETAIL, re.S)
    body = SOURCES[SOURCES.index("export function audioSubstitution"):]
    # Confirmation turns YT? into YT, and only for the link it was given on.
    assert "p.url === track?.source_url" in body
    assert 'chip: "YT?"' in body


def test_bulk_bar_offers_the_suspect_audio_redownload():
    assert "suspect_audio" in BULK
    assert '"redownload_suspect"' in BULK


def test_every_new_class_is_styled():
    used = set()
    for src in (PICKER, TABLE):
        for m in re.finditer(r'className=[{"`]([^"`}]*)', src):
            used.update(w for w in re.split(r"[\s$]+", m.group(1))
                        if re.fullmatch(r"(asp|audio-src|tt-src)[\w-]*", w))
    assert used, "expected the provenance classes to be used"
    missing = sorted(c for c in used if f".{c}" not in CSS)
    assert not missing, f"unstyled classes: {missing}"
