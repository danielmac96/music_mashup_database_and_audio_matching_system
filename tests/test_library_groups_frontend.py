"""Library groups in the JSX, pinned from Python.

This repo has no JS test runner and adding one is a separate decision, so the
frontend invariants are asserted by reading the source — the same trick
tests/test_library_filters_frontend.py and tests/test_stale_frontend.py use.

Two rules carry this feature. **Filtering by a group must not fetch**: a group
is a set of song ids already in memory, exactly like every other control on that
bar. And **the groups are fetched once, in App.jsx**, because the rail counts
them, the filter bar names them, the table is narrowed by one and the row menu
writes to them — a second copy would still be showing the old shelf after the
first one added a track.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


HOOK = _read("hooks/useLibraryFilters.js")
GROUPS = _read("hooks/useLibraryGroups.js")
BAR = _read("components/LibraryFilters.jsx")
SCREEN = _read("components/LibraryScreen.jsx")
APP = _read("App.jsx")


def test_filtering_by_a_group_is_a_lookup_not_a_request():
    """GET /api/tracks is unpaginated and the membership is a list of ids, so
    narrowing to a shelf is arithmetic. A control that quietly re-queries turns
    "show me fewer" into "spend a request"."""
    for banned in ("fetch(", "api.", "XMLHttpRequest"):
        assert banned not in HOOK, banned
        assert banned not in BAR, banned
    apply_fn = HOOK[HOOK.index("export function applyLibraryFilters"):]
    assert "groups.has(filters.group, t.id)" in apply_fn


def test_the_groups_are_fetched_once_in_the_app():
    users = [rel for rel in ("App.jsx", "components/LibraryScreen.jsx",
                             "components/LibraryFilters.jsx",
                             "components/TrackActions.jsx",
                             "components/TrackDetail.jsx")
             if "useLibraryGroups(" in _read(rel)]
    assert users == ["App.jsx"], users
    assert "api.getLibraryGroups" in GROUPS
    assert GROUPS.count("api.getLibraryGroups") == 1


def test_the_rail_row_actually_filters():
    """The bug this replaces: CrateShelf listed the crates and its rows had no
    onClick at all, so a playlist you had saved was visible and unusable."""
    shelf = SCREEN[SCREEN.index("function GroupShelf"):]
    assert "onClick={() => onPick(g.id)}" in shelf
    assert "onPick" in SCREEN[:SCREEN.index("function GroupShelf")]


def test_the_rail_count_is_what_the_filter_will_show():
    """A crate's item_count includes tracks that are not in the library yet. The
    number next to a filter has to be the number of rows that filter returns, or
    clicking it looks broken."""
    shelf = SCREEN[SCREEN.index("function GroupShelf"):]
    assert "count={g.song_ids.length}" in shelf
    assert "count={g.item_count}" not in shelf


def test_a_group_and_the_rail_agree_on_which_one_is_active():
    """Ids arrive as numbers from the API and as strings from the rail, and ===
    between them is silently false — the filter would match nothing."""
    assert 'String(active) === String(g.id)' in SCREEN
    assert 'String(filters.group) === String(id)' in SCREEN
    assert "String(id) === String(groupId)" in GROUPS


def test_inside_a_group_unsorted_is_the_groups_own_order():
    """A crate's running order is most of why you saved it as one. With no
    group chosen, unsorted is import order."""
    fn = HOOK[HOOK.index("export function sortLibrary"):]
    fn = fn[:fn.index("\n}")]
    assert "if (!groupId) return rows;" in fn
    assert "groups.positionOf(groupId, t.id)" in fn


def test_the_importer_can_name_a_group_and_reports_it():
    src = _read("components/PlaylistImporter.jsx")
    assert "api.ingestTracks(kept, previewId, wantGroup)" in src
    assert "res.group" in src
    # Prefilled from the playlist's own name, which rode along on the preview.
    assert "data.playlist_title" in src


def test_the_importer_will_not_save_an_unnamed_group():
    src = _read("components/PlaylistImporter.jsx")
    save = src[src.index('<button className="save"'):]
    save = save[:save.index("</button>")]
    assert "groupOn && !groupName.trim()" in save


def test_discover_offers_the_group_only_on_a_playlist():
    """A search page or an artist's uploads is not a set; naming a group after
    "House" would make a shelf nobody asked for."""
    browser = _read("components/SoundCloudBrowser.jsx")
    assert 'here?.kind === "playlist" ? here.label : null' in browser
    assert "groupOn && playlistHere ? playlistHere : null" in browser
    dock = _read("components/ShortlistDock.jsx")
    assert "{playlistName && (" in dock


def test_an_import_elsewhere_tells_the_library_rail():
    """The rail is a screen away and only learns about a new group if someone
    says so."""
    browser = _read("components/SoundCloudBrowser.jsx")
    assert "onGroupsChanged?.()" in browser
    assert "onGroupsChanged={groups.refresh}" in APP
    assert "groups.refresh()" in SCREEN


def test_the_track_screen_chips_are_read_only():
    """Same decision as Discover's crate chip: the place that edits a group is
    the row menu in the library, next to the track you are deciding about."""
    src = _read("components/TrackDetail.jsx")
    block = src[src.index("{myGroups.map((g) => ("):]
    block = block[:block.index("))}")]
    assert "<button" not in block
    assert 'className="dt-crate"' in block


def test_removing_is_only_offered_for_the_group_you_can_see():
    """Taking a track off a shelf you are not looking at is how you lose it
    silently."""
    src = _read("components/TrackActions.jsx")
    picker = src[src.index("function GroupPicker"):]
    assert "groups.remove(active.id, [track.id])" in picker
    assert "active && on.has(String(active.id))" in picker
