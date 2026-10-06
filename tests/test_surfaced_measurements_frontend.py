"""Features the API already had that the UI could not reach, and measurements
that were stored but never drawn — found by the 100-persona simulation.

No JS test runner here; like the other *_frontend tests, the invariants are
asserted by reading the source."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


API = _read("api.js")
CSS = _read("styles.css")
DOCK = _read("components/PairDock.jsx")
DOCK_HOOK = _read("hooks/usePairDock.js")
CARD = _read("components/PairCard.jsx")
MODEL = _read("components/pairs/pairModel.js")


def _classes_styled(src: str):
    names = set()
    for cls in re.findall(r'className="([^"]+)"', src):
        names.update(c for c in cls.split() if c)
    return sorted(c for c in names if f".{c}" not in CSS)


# ── pair dock: filters, search, paging, per-vocal, batch export ──────────────

def test_dock_filters_go_to_the_server():
    """The dock used to send only limit + scope; every filter the list route
    accepts must now reach it, and none may be applied to the fetched page."""
    fn = DOCK_HOOK[DOCK_HOOK.index("function filterOpts"):]
    fn = fn[:fn.index("\n}\n")]
    for k in ("search", "minScore", "maxEffort", "genre", "era", "energy",
              "bpmBand", "vocalForward", "adventure"):
        assert k in fn, k
    assert "...filterOpts(filters)" in DOCK_HOOK
    for k in ("search", "offset"):
        assert f'params.set("{k}"' in API, k


def test_load_more_pages_by_offset_and_dedupes_on_the_pair_key():
    fn = DOCK_HOOK[DOCK_HOOK.index("const loadMore"):]
    fn = fn[:fn.index("}, [")]
    assert "offset: rowsRef.current.length" in fn
    assert "keyOf" in fn
    assert "loadMore" in DOCK and "hasMore" in DOCK


def test_best_bed_per_vocal_and_batch_export_are_reachable():
    assert "api.getBestBedPerVocal" in DOCK_HOOK
    assert '"/api/mashups/by-vocal' in API or "`/api/mashups/by-vocal" in API
    assert "api.startSessionBatch" in DOCK_HOOK
    assert '"/api/mashups/session/batch"' in API
    assert "exportBatch" in DOCK and "sessionArchiveUrl" in DOCK


def test_dock_menus_offer_only_what_the_library_contains():
    assert "api.getMashupFilters" in DOCK
    assert '"/api/mashups/filters"' in API


def test_new_dock_classes_are_styled():
    assert not _classes_styled(DOCK)


# ── measured harmony on the card and in Studio ───────────────────────────────

def test_measured_harmony_is_drawn_and_unmeasured_is_not():
    fn = MODEL[MODEL.index("export function harmonyOf"):]
    assert "c.harmonic_shift == null" in fn and "known: false" in fn
    assert "harmonyOf(c)" in CARD
    assert "h.known &&" in CARD
    assert "h.bassClash &&" in CARD and "BASS_CLASH_ADVICE" in CARD
    assert ".pc-tag.harmony" in CSS and ".pc-tag.clash" in CSS


def test_studio_shows_the_plans_harmony_and_bass_advice():
    studio = _read("components/MixStudio.jsx")
    assert "pairPlan?.harmony?.known" in studio
    assert "pairPlan?.harmony?.bass_clash" in studio
    assert ".align-chip.clash" in CSS


# ── section table ────────────────────────────────────────────────────────────

def test_section_table_shows_the_per_section_measurements():
    tab = _read("components/SectionTable.jsx")
    heads = tab[tab.index("const HEADS"):]
    heads = heads[:heads.index("];")]
    for h in ("ENERGY", "VOX", "SUNG", "PHR"):
        assert f'"{h}"' in heads, h
    cols = re.search(r'const COLS = "([^"]+)"', tab).group(1)
    assert len(cols.split()) == heads.count('"') // 2, "one grid column per header"
    for field in ("s.vocal_activity", "s.f0?.p10_midi", "s.phrase_length_bars",
                  "s.energy_trend", "s.provisional"):
        assert field in tab, field
    # Unmeasured draws as a dash, never as 0%.
    assert 's.vocal_activity == null ? "—"' in tab
    assert not _classes_styled(tab)


# ── library attribute filters ────────────────────────────────────────────────

def test_attribute_filters_exclude_unmeasured_rows():
    hook = _read("hooks/useLibraryFilters.js")
    fn = hook[hook.index("export function attrMatches"):]
    fn = fn[:fn.index("\n}\n")]
    assert "if (value == null) return false;" in fn
    assert "if (value == null || !Number.isFinite(x)) return false;" in fn
    assert "attrMatches(t.attrs?.[id], cond)" in hook
    assert "Object.keys(f.attrs || {}).length" in hook     # counts as active


def test_attribute_menu_is_built_from_rows_in_memory():
    bar = _read("components/LibraryFilters.jsx")
    assert "facets.attrValues" in bar and "facets.attrRange" in bar
    assert "catalogue={attributes?.catalogue" in _read("components/LibraryScreen.jsx")
    assert not _classes_styled(bar)


# ── queue ETA, analyser status, crate import ─────────────────────────────────

def test_queue_shows_typical_time_per_pool():
    q = _read("components/QueueScreen.jsx")
    assert "api.getJobTimings" in q and "/api/jobs/timings" in API
    assert "poolMedianSecs(timings, key)" in q
    assert not _classes_styled(q)


def test_analysis_shows_the_analyser_status():
    an = _read("components/AnalysisScreen.jsx")
    assert "api.getAnalysisStatus" in an and '"/api/analysis/status"' in API
    assert "az.blocked" in an
    # analysed_stems is { stem_type: n }; rendering the object crashed React.
    assert "{status.analysed_stems" not in an
    assert not _classes_styled(an)


def test_crates_import_from_pasted_links():
    cp = _read("components/CratePanel.jsx")
    assert "api.importCrateUrls" in cp and '"/api/crates/import"' in API
    assert not _classes_styled(cp)
