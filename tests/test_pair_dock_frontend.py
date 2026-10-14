"""The pair dock, pinned from Python.

Three invariants, all of which fail silently rather than loudly:

* a pair is identified by its four ids, NEVER by candidate.id. mashup_candidates
  is truncated on every "Score library" run, so an id survives exactly until the
  next re-score — and a rating keyed on one would attach itself to whatever row
  inherited that id;
* a rating is keyed on the SECTION indexes, so judging "chorus over drop" leaves
  your verdict on "verse over breakdown" alone. That is the P2.0 data-loss bug,
  restated on the client side;
* an unmeasured score term is drawn as unmeasured, not as zero.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


MODEL = _read("components/pairs/pairModel.js")
DOCK_HOOK = _read("hooks/usePairDock.js")
DOCK = _read("components/PairDock.jsx")
CARD = _read("components/PairCard.jsx")
RATINGS = _read("hooks/useRatings.js")
API = _read("api.js")


def test_a_pair_is_keyed_by_its_four_ids():
    fn = MODEL[MODEL.index("export const keyOf"):]
    fn = fn[:fn.index(";")]
    for part in ("vocal_song_id", "inst_song_id",
                 "vocal_section_idx", "inst_section_idx"):
        assert part in fn, part


def test_nothing_that_outlives_a_rescore_keys_on_candidate_id():
    """The table is truncated by score_all_pairs, so `id` is not an identity."""
    for rel, src in (("hooks/usePairDock.js", DOCK_HOOK),
                     ("hooks/useRatings.js", RATINGS),
                     ("components/PairDock.jsx", DOCK)):
        # A comment is allowed to name it — explaining why it is NOT the key is
        # most of the point of having one.
        code = "\n".join(l for l in src.split("\n")
                         if not l.strip().startswith(("//", "*", "/*")))
        assert not re.search(r"\bc\.id\b", code), rel
        assert not re.search(r"candidate\.id\b", code), rel


def test_the_dock_renders_one_card_per_key():
    assert "keyOf(c)" in DOCK
    assert "key={k}" in DOCK


def test_a_rating_carries_the_section_indexes():
    """Without them the server's unique index collapses every overlay of the
    same two records onto one row."""
    fn = RATINGS[RATINGS.index("const rate = useCallback"):]
    fn = fn[:fn.index("}, [byPair, verdicts, load])")]
    assert "vocalSection: candidate.vocal_section_idx" in fn
    assert "instSection: candidate.inst_section_idx" in fn
    assert "rating: stars" in fn


def test_re_sending_the_same_star_clears_the_rating():
    """The number keys make a stray judgement one keypress, so undoing one has
    to be as cheap. It lives in the hook rather than a call site so every
    writable StarRating undoes the same way, and it calls the DELETE — a null
    rating on the POST is COALESCEd into the star already stored."""
    fn = RATINGS[RATINGS.index("const rate = useCallback"):]
    fn = fn[:fn.index("}, [byPair, verdicts, load])")]
    assert "if (stars === prevR)" in fn
    assert "api.clearPairFeedback" in fn
    assert "vocalSection: candidate.vocal_section_idx" in fn
    api = _read("api.js")
    assert "clearPairFeedback" in api
    assert 'method: "DELETE"' in api[api.index("clearPairFeedback"):]


def test_the_feedback_key_matches_the_pair_key():
    """One built from a candidate, one from a stored feedback row, whose columns
    are named differently. They have to describe the same thing."""
    assert "export const feedbackKey" in MODEL
    fk = MODEL[MODEL.index("export const feedbackKey"):]
    fk = fk[:fk.index(";")]
    assert "f.vocal_section" in fk and "f.inst_section" in fk
    assert "?? -1" in fk


def test_a_star_reaches_the_api_as_a_rating():
    assert "rating = null" in API
    body = API[API.index("savePairFeedback:"):]
    body = body[:body.index("}),")]
    assert "verdict, rating" in body


def test_the_card_draws_the_four_weighted_terms():
    """LBL / DUR / VOI / PHR — the terms score_section is a sum of. rhythm and
    structure are deliberately absent: the first saturates and the second
    correlates .88 with label, so both sit at zero weight."""
    terms = MODEL[MODEL.index("export const SCORE_TERMS"):]
    terms = terms[:terms.index("];")]
    for k in ("score_label", "score_duration", "score_voice", "score_phrase"):
        assert k in terms, k
    for absent in ("score_rhythm", "score_structure"):
        assert absent not in terms, absent


def test_every_meter_explains_itself():
    """The expanded card's meters (VOI / HARM / RHY / ENGY) each carry their
    meaning and their value on hover; the section terms keep their plain names
    and explanations in pairModel."""
    terms = MODEL[MODEL.index("export const SCORE_TERMS"):]
    terms = terms[:terms.index("];")]
    assert terms.count("what:") == terms.count("key:") == terms.count("name:") == 4
    meters = CARD[CARD.index("function metersOf("):]
    meters = meters[:meters.index("\n}")]
    for name in ('"VOI"', '"HARM"', '"RHY"', '"ENGY"'):
        assert name in meters, name
    assert 'title={`${tip}: ${v == null ? "not measured"' in CARD


def test_each_side_shows_its_own_genre_key_and_bpm():
    """Variant C: each side on its own line with its own genre, key and BPM,
    before adjustment; the bed's genre turns amber when it differs."""
    assert 'title="Vocal BPM">{fmtBpm(cb.vocal ?? c.vocal_bpm)}' in CARD
    assert 'title="Bed BPM">{fmtBpm(cb.bed ?? c.inst_bpm)}' in CARD
    assert 'title="Vocal key (Camelot)"' in CARD and 'title="Bed key (Camelot)"' in CARD
    assert 'genreDiffers ? " differs" : ""' in CARD
    css = _read("styles.css")
    assert ".pc-bpm {" in css and ".pc-genre.differs" in css
    grid = css[css.index("\n.pc-grid {"):]
    assert "grid-template-columns: 26px minmax(0, 1fr) auto 30px 36px" in grid[:grid.index("}")]


def test_an_unmeasured_term_is_not_drawn_as_zero():
    """A candidate scored before those columns existed has no value for three of
    the four. An empty bar would claim it failed a test it was never given."""
    fn = MODEL[MODEL.index("export function termsOf"):]
    fn = fn[:fn.index("\n}")]
    assert "known" in fn and "null" in fn
    assert "value={v == null || !Number.isFinite(Number(v)) ? null : Number(v)}" in CARD
    prim = _read("components/ui/primitives.jsx")
    assert 'pct == null ? " unmeasured" : ""' in prim
    css = _read("styles.css")
    assert ".compat-meter .track.unmeasured" in css


def test_no_grid_is_not_a_measured_zero():
    """alignment_offset is null when neither side has a stored downbeat grid."""
    fn = MODEL[MODEL.index("export function nudgeLabel"):]
    fn = fn[:fn.index("\n}")]
    assert 'return "no grid"' in fn


def test_cards_scroll_rather_than_compress():
    """The list scrolls; a card never shrinks to fit (the list is a flex column
    of cards that keep their own height), and nothing scrolls sideways."""
    css = _read("styles.css")
    block = css[css.index("\n.pd-list {"):]
    block = block[:block.index("}")]
    assert "overflow-y: auto" in block and "overflow-x: hidden" in block
    card = css[css.index("\n.pair-card {"):]
    assert "flex-direction: column" in card[:card.index("}")]
    assert "min-height: 58px" in css[css.index("\n.pc-grid {"):]


def test_the_keyboard_is_gated_on_the_visible_screen():
    """A window listener that ignores which screen is showing is how two
    keyboard models end up fighting over the space bar. The dock's is bound
    only while the library is the visible route."""
    assert "bindKeys" in DOCK_HOOK
    app = _read("App.jsx")
    assert 'dock.bindKeys(route === "library")' in app
    assert "if (!enabled) return undefined;" in DOCK_HOOK


def test_the_keyboard_map_is_the_one_the_footer_advertises():
    keys = DOCK_HOOK[DOCK_HOOK.index("const bindKeys"):]
    for k in ("ArrowDown", "ArrowUp", '" "', "Enter", '"v"', '"b"'):
        assert k in keys, k
    assert 'e.key >= "1" && e.key <= "5"' in keys
    # ...and it does not fire while you are typing.
    assert 'el.tagName === "INPUT"' in keys


def test_soloing_a_stem_does_not_rearm_the_loop():
    """Re-arming restarts the clip and loses your place in the bar. Solo is a
    gain change on a running voice."""
    audition = _read("hooks/useHookAudition.js")
    fn = audition[audition.index("const applyStems = useCallback"):]
    fn = fn[:fn.index("}, [])")]
    assert "setVoiceGain" in fn
    assert "setVoice(" not in fn


def test_the_dock_scopes_to_the_selected_track_server_side():
    """GET /api/mashups already takes vocal_song_id / inst_song_id. Filtering a
    truncated page client-side would search the top of the list, not the
    library."""
    fn = DOCK_HOOK[DOCK_HOOK.index("useEffect(() => {"):]
    fn = fn[:fn.index("}, [selectedTrackId, role, order")]
    assert "opts.instSongId = selectedTrackId" in fn
    assert "opts.vocalSongId = selectedTrackId" in fn


def test_hiding_a_pair_is_reachable_and_reversible():
    """Hiding is a display preference, not a verdict: it is suppressed by SQL
    and never becomes training data. It is undone from the toast, from "Show
    hidden" in the dock's filters, and in the Tuning panel."""
    hook = DOCK_HOOK[DOCK_HOOK.index("const bindKeys"):]
    assert 'e.key === "h"' in hook
    assert "api.hidePair" in DOCK_HOOK and "api.unhidePair" in DOCK_HOOK
    assert 'label: "Undo", onClick: () => unhide(candidate)' in DOCK_HOOK
    assert "h hide" in DOCK, "the footer must advertise the key it binds"
    assert '"showHidden", `Show hidden pairs' in DOCK
    tuning = _read("components/TuningPanel.jsx")
    assert "api.getHidden" in tuning
    assert "api.unhidePair" in tuning and "api.includeTrack" in tuning


def test_loaded_pair_filters_never_refetch():
    """The Filters overlay's loaded-pairs half (max stretch, transposed beds,
    same genre, per-side genre / BPM / key, tier chips) narrows rows already
    fetched — it is not in the fetch effect's dependencies."""
    fetch = DOCK_HOOK[DOCK_HOOK.index("  useEffect(() => {\n    let cancelled = false;"):]
    deps = fetch[fetch.index("}, ["):]
    deps = deps[:deps.index("]")]
    for name in ("df", "tierOn", "hiddenKeys"):
        assert name not in deps, name
    assert "export function applyLoadedFilters" in DOCK_HOOK
    assert "applyLoadedFilters(out, df, hiddenKeys)" in DOCK_HOOK


def test_tier_chips_and_group_headers_read_the_api_tier():
    assert "c.confidence_tier" in DOCK_HOOK
    assert "className=\"pd-tierhead mono\"" in DOCK
    assert "toggleTier(t.id)" in DOCK


def test_the_dock_resizes_and_remembers():
    assert '"mashup.dockw.v1"' in DOCK
    assert "DOCK_MIN = 320" in DOCK and "DOCK_MAX = 640" in DOCK and "DOCK_DEFAULT = 400" in DOCK
    assert 'className="pd-resize"' in DOCK


def test_score_library_lives_with_the_knobs_it_re_applies():
    """Settings keeps the button beside the weights it re-applies, outside the
    collapsed body, and the dock's Re-score pill starts the SAME job: both go
    through requestRescore → useFreshness, the one place that calls
    api.startScoring, so there is one scoring job and the dock reloads when it
    lands. It used to sit in a Discover pane, a screen away from the weights."""
    tuning = _read("components/TuningPanel.jsx")
    assert "requestRescore" in tuning and "Score library" in tuning
    assert "MATCH_PRESETS" in tuning
    body = tuning[tuning.index("<h3>Scoring &amp; tuning</h3>"):]
    assert body.index("Score library") < body.index("{open && (")
    hook = _read("hooks/useFreshness.js")
    assert "api.startScoring" in hook and "onRescoreRequest" in hook
    # ...and nothing still sends the user to Discover to find it.
    for rel in ("components/PairDock.jsx", "components/TuningPanel.jsx",
                "components/PartnersRail.jsx"):
        src = _read(rel)
        assert "Score library" not in src or "Discover" not in src, rel
