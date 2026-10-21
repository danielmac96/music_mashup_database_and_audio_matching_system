"""The Segments screen, pinned from Python (readme §4 Segments).

Most of these read the JSX and CSS, as the other frontend contract tests do.
The filtering model is also RUN: esbuild (Vite's own dependency) bundles
segmentModel.js and Node executes it, so "the match list keeps the server's
order under the table's filters" is checked on behaviour, not on spelling.
That test skips where Node or the frontend's node_modules are missing.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"
SEG = SRC / "components" / "segments"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


APP = _read("App.jsx")
SIDEBAR = _read("shell/Sidebar.jsx")
CSS = _read("styles.css")
FILES = {p.name: p.read_text(encoding="utf-8") for p in SEG.glob("*.js*")}
HOOKS = {n: _read(f"hooks/{n}") for n in ("useSegments.js", "useMiniStudio.js")}


def test_segments_sits_under_library_in_the_rail_and_renders():
    nav = SIDEBAR[SIDEBAR.index("const NAV = ["):]
    nav = nav[:nav.index("];")]
    ids = re.findall(r'\["(\w+)", "', nav)
    assert ids[:2] == ["library", "segments"], ids
    assert 'route === "segments"' in APP and "<SegmentsScreen" in APP
    # ⌘K's "Go to" is built from NAV_ITEMS, so it gets the screen for free.
    assert "NAV_ITEMS.map" in APP


def test_the_tables_are_virtualised_on_fixed_rows():
    table = FILES["SegmentTable.jsx"]
    assert "export const ROW_H = 32;" in table
    assert "rows.slice(first, first + count)" in table
    assert "height: rows.length * ROW_H" in table
    # Rows are memoised and call handlers through a ref.
    assert "memo(function Row" in table and "handlers.current" in table


def test_narrow_panes_drop_columns_by_priority_not_scroll_sideways():
    table = FILES["SegmentTable.jsx"]
    assert "prio" in table and "c.prio > 0" in table
    assert "c.flex ? slack : 0" in table


def test_filtering_never_fetches():
    model = FILES["segmentModel.js"]
    assert "api." not in model and "fetch(" not in model and "../../api" not in model
    filt = FILES["SegmentFilters.jsx"]
    assert "api." not in filt and "fetch(" not in filt


def test_the_search_box_reuses_the_library_parser():
    model = FILES["segmentModel.js"]
    assert 'import { parseQuery } from "../../hooks/useLibraryFilters"' in model


def test_matches_are_debounced_and_stale_answers_dropped():
    hook = HOOKS["useSegments.js"]
    assert "MATCH_DEBOUNCE_MS = 150" in hook
    assert "AbortController" in hook and "c.abort()" in hook
    assert "c.signal.aborted" in hook


def test_the_tier_is_never_computed_client_side():
    """§7: the frontend reads confidence_tier; theme.TIERS only names it."""
    for name, src in FILES.items():
        assert "TIER_MIN" not in src and "score_percentile >=" not in src, name
    assert "confidence_tier" in FILES["SegmentTable.jsx"]


def test_ratings_sets_and_notes_go_through_the_shared_hooks():
    screen = FILES["SegmentsScreen.jsx"]
    band = FILES["MiniStudio.jsx"]
    assert "ratings.rate(armed" in screen and "sets.addPair(armed)" in screen
    assert "ratings.rate(pair, n)" in band and "notes.save(pair" in band
    assert "sets.addPair(pair)" in band


def test_the_mini_studio_plays_what_the_card_prints():
    """The band's starting point is the pair row's own numbers — the dock's
    audition and Studio read the same ones (§7, one transpose per pair)."""
    hook = HOOKS["useMiniStudio.js"]
    for field in ("pair.semitone_shift", "r.bed_rate", "pair.stretch_factor",
                  "r.vocal_lane_gain", "r.bed_lane_gain", "r.bed_highpass_hz",
                  "pair.alignment_offset"):
        assert field in hook, field
    assert "hookUrl(pair.vocal_song_id" in hook and "hookUrl(pair.inst_song_id" in hook
    # Riding a fader must not restart the bar.
    assert "updateVoiceParams" in hook
    # Studio's lane gain is linear: the band prints 20·log10 (§7).
    assert "20 * Math.log10(g)" in FILES["MiniStudio.jsx"]


def test_open_in_studio_carries_the_tweaks():
    assert "onOpenStudio(pair, tweaks)" in FILES["MiniStudio.jsx"]
    assert "const pairToStudio = (c, tweaks = null)" in APP and "    tweaks,\n" in APP
    studio = _read("components/MixStudio.jsx")
    for k in ("tw?.semitones", "tw?.tempoPct", "tw?.nudgeDeltaSec", "tw?.vocalGain",
              "tw?.bedGain", "tw?.hpHz"):
        assert k in studio, k


def test_the_wav_render_places_the_trim_start_as_the_offset():
    """§7: build_mixdown puts a clip's first rendered sample at offset_sec."""
    band = FILES["MiniStudio.jsx"]
    assert "clip_start: pair.vocal_section_start" in band
    assert "clip_start: pair.inst_section_start" in band
    assert 'j.status === "completed"' in band


def test_one_sound_at_a_time():
    band = FILES["MiniStudio.jsx"]
    assert "player.stop()" in band and "ms.pause()" in band
    assert '|| route === "segments") return undefined;' in APP


def test_no_hex_and_every_class_is_styled():
    for name, src in {**FILES, **HOOKS}.items():
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", src), name
    used = set()
    for src in FILES.values():
        for m in re.finditer(r'className=(?:"([^"]+)"|\{`([^`]+)`\})', src):
            text = m.group(1) or re.sub(r"\$\{[^}]*\}", " ", m.group(2))
            used.update(t for t in text.split() if re.fullmatch(r"[a-z][\w-]*", t))
    # Role and state modifiers composed in templates, styled as compounds.
    shared = {"vocal", "bed", "on", "dim", "empty", "edited", "match", "lead",
              "warn", "val", "fill", "lbl", "glyph", "caps", "mono", "muted", "faint",
              "zebra", "selected", "pill"}
    missing = sorted(c for c in used - shared
                     if not re.search(r"\." + re.escape(c) + r"\b", CSS))
    assert not missing, missing


def test_help_names_the_screen_and_its_keys():
    help_ = _read("components/HelpPanel.jsx")
    assert '["Segments", ' in help_


# ── behaviour: the model, run under Node ─────────────────────────────────────

def _bundle(tmp_path):
    """segmentModel.js bundled to one ES module by esbuild's JS API — run
    through Node rather than node_modules/.bin, whose shim does not execute
    on Windows."""
    node = shutil.which("node")
    if not node or not (ROOT / "frontend" / "node_modules" / "esbuild").exists():
        pytest.skip("needs Node and frontend/node_modules (npm install)")
    out = tmp_path / "model.mjs"
    build = tmp_path / "build.mjs"
    build.write_text(
        "import { build } from 'esbuild';\n"
        f"await build({{ entryPoints: [{json.dumps(str(SEG / 'segmentModel.js'))}], "
        f"bundle: true, format: 'esm', platform: 'neutral', logLevel: 'error', "
        f"outfile: {json.dumps(str(out))} }});\n", encoding="utf-8")
    # Resolved from the frontend's node_modules: NODE_PATH does not apply to
    # ES modules, so the build script is copied next to them.
    local = ROOT / "frontend" / f".seg-test-build-{tmp_path.name}.mjs"
    local.write_text(build.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        subprocess.run([node, str(local)], check=True, cwd=ROOT / "frontend")
    finally:
        local.unlink(missing_ok=True)
    return node, out


def _run(tmp_path, body: str):
    node, model = _bundle(tmp_path)
    script = tmp_path / "run.mjs"
    script.write_text(f'import * as M from "{model.as_uri()}";\n{body}\n', encoding="utf-8")
    res = subprocess.run([node, str(script)], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


ROWS = [
    {"segment_id": 1, "song_id": 1, "section_index": 1, "title": "Alpha", "label": "chorus",
     "bpm": 124.0, "camelot": "8A", "bar_count": 16, "energy_rank": 0.9, "genre": "House", "usable": True},
    {"segment_id": 2, "song_id": 2, "section_index": 0, "title": "Beta", "label": "intro",
     "bpm": 62.0, "camelot": "9A", "bar_count": 8, "energy_rank": None, "genre": "Pop", "usable": False},
    {"segment_id": 3, "song_id": 3, "section_index": 2, "title": "Gamma", "label": "drop",
     "bpm": 128.0, "camelot": "8A", "bar_count": 16, "energy_rank": 0.4, "genre": "House", "usable": True},
    {"segment_id": 4, "song_id": 4, "section_index": 3, "title": "Delta", "label": "chorus",
     "bpm": 140.0, "camelot": "3B", "bar_count": 32, "energy_rank": 0.7, "genre": "Techno", "usable": True},
]


def test_unpacking_puts_song_fields_under_section_fields(tmp_path):
    body = {"columns": ["segment_id", "song_id", "label"],
            "songs": {"7": {"title": "T", "bpm": 120}},
            "rows": [[11, 7, "chorus"]]}
    out = _run(tmp_path, f"console.log(JSON.stringify(M.unpackSegments({json.dumps(body)})));")
    assert out == [{"title": "T", "bpm": 120, "segment_id": 11, "song_id": 7, "label": "chorus"}]


def test_search_tokens_and_filters(tmp_path):
    out = _run(tmp_path, f"""
const rows = {json.dumps(ROWS)};
const F = M.EMPTY_SEG_FILTERS;
const ids = (f) => M.applySegFilters(rows, {{...F, ...f}}, ["House", "Pop", "Techno"]).map((r) => r.segment_id);
console.log(JSON.stringify({{
  chorus: ids({{q: "chorus"}}),
  key: ids({{q: "8A"}}),
  bpmFold: ids({{q: "120-126"}}),
  bars: ids({{q: "32bars"}}),
  genre: ids({{q: "house drop"}}),
  energy: ids({{energyMin: 0.5}}),
  usable: ids({{usableOnly: true}}),
  text: ids({{q: "gam"}}),
}}));""")
    assert out["chorus"] == [1, 4]
    assert out["key"] == [1, 3]
    # 62 BPM reads as 124 at double time; 128 is inside 126 + 6%.
    assert out["bpmFold"] == [1, 2, 3]
    assert out["bars"] == [4]
    assert out["genre"] == [3]
    # Unmeasured energy never passes a filter that asks about it.
    assert out["energy"] == [1, 4]
    assert out["usable"] == [1, 3, 4]
    assert out["text"] == [3]


def test_match_mode_keeps_the_server_order_under_the_table_filters(tmp_path):
    matches = [{"segment_id": 3, "score_total": 0.9}, {"segment_id": 2, "score_total": 0.8},
               {"segment_id": 1, "score_total": 0.7}]
    out = _run(tmp_path, f"""
const rows = {json.dumps(ROWS)};
const m = {json.dumps(matches)};
const F = M.EMPTY_SEG_FILTERS;
const none = {{id: null, dir: null}};
const all = M.buildSideRows(rows, m, F, none);
const usable = M.buildSideRows(rows, m, {{...F, usableOnly: true}}, none);
const byBpm = M.buildSideRows(rows, m, F, {{id: "bpm", dir: "desc"}});
const cleared = M.buildSideRows(rows, null, F, none);
console.log(JSON.stringify({{
  all: all.rows.map((r) => r.segment_id), total: all.total,
  usable: usable.rows.map((r) => r.segment_id),
  fit: all.rows.map((r) => r._match.score_total),
  byBpm: byBpm.rows.map((r) => r.segment_id),
  cleared: cleared.rows.map((r) => r.segment_id), clearedTotal: cleared.total,
}}));""")
    assert out["all"] == [3, 2, 1] and out["total"] == 3      # server order, matches only
    assert out["usable"] == [3, 1]                            # filters on top, order kept
    assert out["fit"] == [0.9, 0.8, 0.7]
    assert out["byBpm"] == [3, 1, 2]                          # a header sort still sorts
    assert out["cleared"] == [1, 2, 3, 4] and out["clearedTotal"] == 4


def test_sorting_puts_unmeasured_last_both_ways_and_cycles(tmp_path):
    out = _run(tmp_path, f"""
const rows = {json.dumps(ROWS)};
const ids = (s) => M.sortSegments(rows, s).map((r) => r.segment_id);
console.log(JSON.stringify({{
  desc: ids({{id: "energy", dir: "desc"}}), asc: ids({{id: "energy", dir: "asc"}}),
  c1: M.cycleSegSort({{id: null, dir: null}}, "bpm"),
  c2: M.cycleSegSort({{id: "bpm", dir: "desc"}}, "bpm"),
  c3: M.cycleSegSort({{id: "bpm", dir: "asc"}}, "bpm"),
  t1: M.cycleSegSort({{id: null, dir: null}}, "title"),
  midi: [M.midiName(60), M.midiName(69.4), M.midiName(null)],
}}));""")
    assert out["desc"] == [1, 4, 3, 2] and out["asc"] == [3, 4, 1, 2]
    assert out["c1"] == {"id": "bpm", "dir": "desc"}
    assert out["c2"] == {"id": "bpm", "dir": "asc"}
    assert out["c3"] == {"id": None, "dir": None}
    assert out["t1"] == {"id": "title", "dir": "asc"}
    assert out["midi"] == ["C4", "A4", None]


def test_ways_in_from_track_detail_and_the_palette():
    table = _read("components/SectionTable.jsx")
    assert 'onMatch(s, "vocal")' in table and 'onMatch(s, "bed")' in table
    detail = _read("components/TrackDetail.jsx")
    assert "onOpenSegments({ songId: track.id, sectionIndex: s.section_index, role: asRole })" in detail
    assert "onOpenSegments={openSegments}" in APP and "seed={segmentsSeed}" in APP
    assert 'key: "a:segments"' in APP
    screen = FILES["SegmentsScreen.jsx"]
    assert "seeded.current === seed.at" in screen and "setLead(seed.role)" in screen


def test_the_mini_studio_declares_every_hook_before_its_early_return():
    """React error #310: a hook below `if (!pair) return …` changes the hook
    count the moment a pair is armed, and the whole screen goes blank."""
    band = FILES["MiniStudio.jsx"]
    body = band[band.index("export function MiniStudio"):]
    early = body.index("\n  if (!pair) {")
    assert not re.search(r"\buse(Effect|State|Ref|Memo|Callback|MiniStudio)\(", body[early:])


def test_segments_survives_a_trip_to_studio():
    assert "segmentsVisited &&" in APP and 'active={route === "segments"}' in APP
    assert ".seg-host.away { display: none; }" in CSS
    screen = FILES["SegmentsScreen.jsx"]
    # Hidden, it owns no keys and reports no status.
    assert screen.count("if (!active) return") + screen.count("|| !active) return") >= 4
    assert "if (active) onRailSlot" in screen
    assert "if (!active && ms.playing) ms.pause()" in FILES["MiniStudio.jsx"]


def test_no_component_is_declared_inside_a_render():
    """A component declared in a render is a new type every render; the loop
    position re-renders the band ~60×/s and its buttons were remounted mid-click."""
    band = FILES["MiniStudio.jsx"]
    assert "const Step = (" not in band
    assert not re.search(r"\n  const [A-Z]\w* = \(\{", band)
