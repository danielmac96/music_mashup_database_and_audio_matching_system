"""The suggestion engine behind Discovery's third pane.

Offline, using the same two patterns the rest of the SoundCloud layer uses: a
`_get` callable answering by substring match, and no real sleeps (`_get` bypasses
throttle, backoff and cache by design).

The ranking assertions here pin an ORDER, not membership. Ordering is the whole
product — a suggestion list that contains the right tracks in the wrong order is
a list nobody scrolls past the top of — so a change that reshuffles it should
fail a test rather than pass one.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ingest import soundcloud_api as sc_api          # noqa: E402
from ingest import soundcloud_browse as br           # noqa: E402
from ingest import soundcloud_recommend as rec       # noqa: E402
from ingest.soundcloud_browse import SoundCloudUnavailable   # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    sc_api._client_id = "test-client-id"
    br.reset_state()
    yield
    sc_api._client_id = None
    br.reset_state()


def hit(tid, *, user_id="9", username="Curator", genre="House", plays=0):
    """A v2 track payload, as SoundCloud returns it inside a collection."""
    return {"id": int(tid), "title": f"T{tid}", "duration": 200_000,
            "genre": genre, "playback_count": plays,
            "permalink_url": f"https://soundcloud.com/x/t{tid}",
            "user": {"id": int(user_id), "username": username,
                     "permalink_url": f"https://soundcloud.com/{username}"}}


def collection(items, next_href=None):
    return json.dumps({"collection": items, "next_href": next_href})


def server(related, *, users=None, sets=None, searches=None, calls=None):
    """A `_get` that answers the four endpoints the engine uses.

    `related` maps a seed track id to a list of hits (or to an exception to
    raise). Everything else defaults to empty so a test can name only the phase
    it cares about."""
    users, sets, searches = users or {}, sets or {}, searches or {}

    def _get(url):
        if calls is not None:
            calls.append(url)
        for tid, payload in related.items():
            if f"/tracks/{tid}/related" in url:
                if isinstance(payload, Exception):
                    raise payload
                return collection(payload)
        if "/search/playlists" in url:
            for term, payload in searches.items():
                if f"q={term}" in url:
                    return collection(_raise_if_error(payload))
            return collection([])
        for uid, payload in sets.items():
            if f"/users/{uid}/playlists" in url:
                return collection(_raise_if_error(payload))
        for uid, payload in users.items():
            if f"/users/{uid}" in url:
                return json.dumps(_raise_if_error(payload))
        raise AssertionError(f"unexpected request: {url}")
    return _get


def _raise_if_error(payload):
    """A mapping value may be an exception, meaning "this endpoint fails"."""
    if isinstance(payload, Exception):
        raise payload
    return payload


def seed(tid, title=None, artist_id=""):
    return {"track_id": str(tid), "title": title or f"Seed {tid}",
            "artist_id": artist_id,
            "source_url": f"https://soundcloud.com/me/s{tid}"}


TRACKS_ONLY = ("tracks",)


# ── seed preparation ─────────────────────────────────────────────────────────

def test_prepare_seeds_drops_songs_without_a_soundcloud_id():
    """The fan-out is /tracks/{id}/related; a song ingested through the mixes
    path has no id and there is nothing to substitute for it."""
    seeds = rec.prepare_seeds([
        {"track_id": "1", "title": "Keeps"},
        {"track_id": "", "title": "No id"},
        {"title": "No key at all"},
    ])
    assert [s["title"] for s in seeds] == ["Keeps"]


def test_prepare_seeds_dedups_and_caps():
    seeds = rec.prepare_seeds([{"track_id": "7"}] * 3)
    assert len(seeds) == 1

    many = rec.prepare_seeds([{"track_id": str(i)} for i in range(rec.MAX_SEEDS + 10)])
    assert len(many) == rec.MAX_SEEDS


def test_recommend_refuses_a_seedless_run():
    with pytest.raises(ValueError, match="track id"):
        rec.recommend([{"title": "no id"}], _get=server({}))


# ── ranking: the contract ────────────────────────────────────────────────────

def test_agreement_between_seeds_outranks_a_single_first_place():
    """The point of fusing rather than concatenating.

    T11 is second for one seed and first for another; T10 is first for one seed
    only. Two seeds agreeing must win — that is what makes the list a
    recommendation rather than a copy of one related-tracks page."""
    out = rec.recommend(
        [seed(1), seed(2)],
        kinds=TRACKS_ONLY,
        _get=server({"1": [hit(10), hit(11), hit(12)],
                     "2": [hit(11), hit(13)]}))

    assert [t["title"] for t in out["tracks"]] == ["T11", "T10", "T13", "T12"]
    assert out["tracks"][0]["votes"] == 2
    assert out["tracks"][1]["votes"] == 1


def test_rank_within_one_seed_is_preserved():
    out = rec.recommend([seed(1)], kinds=TRACKS_ONLY,
                        _get=server({"1": [hit(10), hit(11), hit(12)]}))
    assert [t["title"] for t in out["tracks"]] == ["T10", "T11", "T12"]


def test_ties_break_on_plays_so_the_order_is_stable():
    """Two rows at the same rank from different seeds score identically. Without
    a deterministic tiebreak they would swap places between runs, which reads as
    the list being random."""
    out = rec.recommend(
        [seed(1), seed(2)], kinds=TRACKS_ONLY,
        _get=server({"1": [hit(10, plays=5)], "2": [hit(11, plays=900)]}))
    assert [t["title"] for t in out["tracks"]] == ["T11", "T10"]


def test_because_names_the_seeds_that_led_there():
    out = rec.recommend(
        [seed(1, "My Anthem"), seed(2, "Other Tune")], kinds=TRACKS_ONLY,
        _get=server({"1": [hit(11)], "2": [hit(11)]}))
    assert out["tracks"][0]["because"] == ["My Anthem", "Other Tune"]


# ── what must not come back ──────────────────────────────────────────────────

def test_seeds_never_appear_in_their_own_results():
    """Related lists routinely contain the seed. Offering you a track you seeded
    from is the most obvious way for the feature to look broken."""
    out = rec.recommend([seed(1)], kinds=TRACKS_ONLY,
                        _get=server({"1": [hit(1), hit(11)]}))
    assert [t["title"] for t in out["tracks"]] == ["T11"]


def test_library_tracks_are_filtered_out():
    owned = lambda rows: {"11"}     # noqa: E731 — the injected library filter
    out = rec.recommend([seed(1)], kinds=TRACKS_ONLY, owned=owned,
                        _get=server({"1": [hit(10), hit(11), hit(12)]}))
    assert [t["title"] for t in out["tracks"]] == ["T10", "T12"]
    assert out["already_owned"] == 1


def test_already_owned_counts_records_not_identity_keys():
    """An owned row matches on BOTH its id and its URL, so counting the key set
    would tell the user two of their records were filtered when it was one."""
    owned = lambda rows: {"11", "https://soundcloud.com/x/t11"}   # noqa: E731
    out = rec.recommend([seed(1)], kinds=TRACKS_ONLY, owned=owned,
                        _get=server({"1": [hit(10), hit(11)]}))
    assert out["already_owned"] == 1


def test_a_row_owned_under_its_url_is_filtered_too():
    """The library matches url-first, id-second; the engine must honour both or a
    track whose permalink we stored but whose id we never learned reappears."""
    owned = lambda rows: {"https://soundcloud.com/x/t11"}   # noqa: E731
    out = rec.recommend([seed(1)], kinds=TRACKS_ONLY, owned=owned,
                        _get=server({"1": [hit(10), hit(11)]}))
    assert [t["title"] for t in out["tracks"]] == ["T10"]


def test_recommended_rows_stay_importable():
    """A suggestion must drop into POST /api/discovery/import unchanged — the
    same property track_row exists to guarantee. If this fails, the engine has
    started rewriting rows instead of annotating them."""
    from ingest.soundcloud import _normalise

    canonical = set(_normalise({
        "title": "t", "uploader": "a", "webpage_url": "u", "duration": 1,
    }))
    row = rec.recommend([seed(1)], kinds=TRACKS_ONLY,
                        _get=server({"1": [hit(11)]}))["tracks"][0]
    assert canonical <= set(row), canonical - set(row)


# ── failure ──────────────────────────────────────────────────────────────────

def test_one_dead_seed_does_not_lose_the_run():
    """Libraries age: uploads get deleted, go private, or region-lock. Losing
    every suggestion to one of them would break the feature over time."""
    boom = sc_api.SoundCloudAPIError("404 Not Found")
    out = rec.recommend([seed(1), seed(2)], kinds=TRACKS_ONLY,
                        _get=server({"1": [hit(11)], "2": boom}))

    assert [t["title"] for t in out["tracks"]] == ["T11"]
    assert out["seeds_used"] == 1
    assert out["skipped"][0]["track_id"] == "2"
    assert "404" in out["skipped"][0]["reason"]


def test_every_seed_failing_raises():
    boom = sc_api.SoundCloudAPIError("404 Not Found")
    with pytest.raises(sc_api.SoundCloudAPIError):
        rec.recommend([seed(1)], kinds=TRACKS_ONLY, _get=server({"1": boom}))


def test_an_open_breaker_stops_the_run_rather_than_being_skipped():
    """SoundCloudUnavailable means we are already backing off. Treating it as a
    per-seed failure would turn one refusal into a request storm against the
    client_id the frozen mixes resolver shares."""
    with pytest.raises(SoundCloudUnavailable):
        rec.recommend([seed(1), seed(2)], kinds=TRACKS_ONLY,
                      _get=server({"1": SoundCloudUnavailable("backing off"),
                                   "2": [hit(11)]}))


# ── artists ──────────────────────────────────────────────────────────────────

def test_artists_are_summed_from_who_uploaded_the_recommendations():
    out = rec.recommend(
        [seed(1)], kinds=("tracks", "artists"),
        _get=server(
            {"1": [hit(10, user_id="5", username="Loud"),
                   hit(11, user_id="5", username="Loud"),
                   hit(12, user_id="6", username="Quiet")]},
            users={"5": {"id": 5, "username": "Loud", "followers_count": 900,
                         "track_count": 40},
                   "6": {"id": 6, "username": "Quiet", "followers_count": 3,
                         "track_count": 2}}))

    assert [a["username"] for a in out["artists"]] == ["Loud", "Quiet"]
    assert out["artists"][0]["new_tracks"] == 2
    # Hydration is what stops the row rendering "0 followers".
    assert out["artists"][0]["followers"] == 900


def test_an_artist_you_own_keeps_their_score_and_gains_a_count():
    """Owning half their catalogue is evidence you like them, not a reason to
    score them as though they barely appeared."""
    owned = lambda rows: {"10"}     # noqa: E731
    out = rec.recommend(
        [seed(1)], kinds=("tracks", "artists"), owned=owned,
        _get=server({"1": [hit(10, user_id="5"), hit(11, user_id="5")]},
                    users={"5": {"id": 5, "username": "Curator"}}))

    assert out["artists"][0]["owned_tracks"] == 1
    assert out["artists"][0]["new_tracks"] == 1


def test_artists_you_seeded_from_are_not_suggested_back():
    out = rec.recommend(
        [seed(1, artist_id="5")], kinds=("tracks", "artists"),
        _get=server({"1": [hit(10, user_id="5"), hit(12, user_id="6")]},
                    users={"6": {"id": 6, "username": "Quiet"}}))
    assert [a["user_id"] for a in out["artists"]] == ["6"]


def test_a_failed_hydration_keeps_the_artist():
    out = rec.recommend(
        [seed(1)], kinds=("tracks", "artists"),
        _get=server({"1": [hit(10, user_id="5", username="Loud")]},
                    users={"5": sc_api.SoundCloudAPIError("boom")}))
    # The counts stay at zero, but the name from the track row survives, so the
    # artist is still listed rather than silently dropped.
    assert out["artists"][0]["username"] == "Loud"
    assert out["artists"][0]["followers"] == 0


# ── sets ─────────────────────────────────────────────────────────────────────

def playlist_hit(pid, title, user="Curator"):
    return {"id": int(pid), "title": title, "track_count": 12, "duration": 100,
            "permalink_url": f"https://soundcloud.com/{user}/sets/{pid}",
            "user": {"id": 9, "username": user}}


def test_sets_come_from_top_artists_and_then_from_genre_search():
    out = rec.recommend(
        [seed(1)],
        _get=server({"1": [hit(10, user_id="5", genre="Techno")]},
                    users={"5": {"id": 5, "username": "Loud"}},
                    sets={"5": [playlist_hit(100, "Their Set")]},
                    searches={"Techno": [playlist_hit(200, "A Techno Set")]}))

    titles = [p["title"] for p in out["playlists"]]
    # Artist-owned sets are grounded in the fan-out; a genre string match is not,
    # and must not outrank one.
    assert titles == ["Their Set", "A Techno Set"]
    assert out["playlists"][0]["source"] == "artist"
    assert out["playlists"][1]["source"] == "genre"


def test_the_same_set_found_twice_appears_once_as_the_stronger_source():
    out = rec.recommend(
        [seed(1)],
        _get=server({"1": [hit(10, user_id="5", genre="Techno")]},
                    users={"5": {"id": 5, "username": "Loud"}},
                    sets={"5": [playlist_hit(100, "Their Set")]},
                    searches={"Techno": [playlist_hit(100, "Their Set")]}))

    assert len(out["playlists"]) == 1
    assert out["playlists"][0]["source"] == "artist"


# ── request budget ───────────────────────────────────────────────────────────

def test_a_tracks_only_run_costs_one_request_per_seed():
    """`kinds` has to actually gate the phases: the artist and set tails are the
    expensive part, and a user who only wants tracks should not pay for them."""
    calls = []
    rec.recommend([seed(1), seed(2)], kinds=TRACKS_ONLY,
                  _get=server({"1": [hit(10)], "2": [hit(11)]}, calls=calls))
    assert len(calls) == 2
    assert all("/related" in c for c in calls)


def test_the_artist_phase_is_bounded_regardless_of_how_many_turn_up():
    ids = list(range(1, rec.TOP_ARTISTS + 6))
    uploaders = {str(i): {"id": i, "username": f"U{i}"} for i in ids}
    out = rec.recommend(
        [seed(1)], kinds=("tracks", "artists"),
        _get=server({"1": [hit(100 + i, user_id=str(i)) for i in ids]},
                    users=uploaders))
    assert len(out["artists"]) == rec.TOP_ARTISTS
