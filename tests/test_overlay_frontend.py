"""The overlay in the browser (master plan Phase 4): frontend/src/components/
overlay/overlayModel.js run under Node against the real route on a synthetic
library — the score it shows for a placement is the priced row (criterion 4.4:
score_pair's at the library's landing), its time mapping is the server's
`on_bed`, its audition starts at the play spans — plus the JSX contracts: one
player, ratings through useRatings, keys only while the panel has focus, no
hex colours."""
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"
OV = SRC / "components" / "overlay"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _bundle(tmp_path, entry: Path, name: str):
    node = shutil.which("node")
    if not node or not (ROOT / "frontend" / "node_modules" / "esbuild").exists():
        pytest.skip("needs Node and frontend/node_modules (npm install)")
    out = tmp_path / f"{name}.mjs"
    local = ROOT / "frontend" / f".{name}-test-build-{tmp_path.name}.mjs"
    local.write_text(
        "import { build } from 'esbuild';\n"
        f"await build({{ entryPoints: [{json.dumps(str(entry))}], "
        "bundle: true, format: 'esm', platform: 'node', logLevel: 'error', "
        "jsx: 'automatic', loader: { '.js': 'jsx' }, "
        f"outfile: {json.dumps(str(out))} }});\n", encoding="utf-8")
    try:
        subprocess.run([node, str(local)], check=True, cwd=ROOT / "frontend")
    finally:
        local.unlink(missing_ok=True)
    return node, out


def run_node(tmp_path, entry: Path, body: str, data: dict, name: str = "ov"):
    node, mod = _bundle(tmp_path, entry, name)
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    script = tmp_path / "run.mjs"
    script.write_text(
        f'import * as M from "{mod.as_uri()}";\nimport fs from "node:fs";\n'
        f'const D = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "data.json"))}, "utf8"));\n'
        f"{body}\n", encoding="utf-8")
    res = subprocess.run([node, str(script)], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlay_fe")
    spec = importlib.util.spec_from_file_location("synth_library",
                                                  ROOT / "scripts" / "synth_library.py")
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    manifest = synth.generate(out, 60, seed=13, planted=6)
    mp = pytest.MonkeyPatch()
    mp.setenv("MASHUP_DB_PATH", str(out / "mashup.db"))
    mp.setenv("MASHUP_AUDIO_ROOT", str(out / "audio"))
    mp.setenv("MASHUP_DATA_DIR", str(out / "data"))
    mp.setenv("MASHUP_SETTINGS_DIR", str(out / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    import matcher.match as match
    importlib.reload(match)
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.routes.tracks as tracks
    importlib.reload(tracks)
    import api.server as server
    importlib.reload(server)
    yield {"m": m, "match": match, "manifest": manifest, "client": TestClient(server.app)}
    mp.undo()


def test_the_score_shown_for_a_placement_is_the_priced_row(lib, tmp_path):
    """Criterion 4.4: the overlay's live score for a placement equals
    score_pair's for the same inputs — the client looks the priced row up and
    never re-scores; at the library's landing that row IS score_pair's."""
    data = []
    for p in lib["manifest"]["planted"]:
        for r in lib["match"].score_pair(p["vocal_song_id"], p["inst_song_id"],
                                         db_path=lib["m"].DB_PATH):
            if r.get("vocal_section_idx") is None:
                continue
            got = lib["client"].get(
                f"/api/tracks/{p['inst_song_id']}/overlay",
                params={"vocal": p["vocal_song_id"], "vs": r["vocal_section_idx"],
                        "bs": r["inst_section_idx"]}).json()
            data.append({"payload": got, "want": r["total"]})
    assert len(data) >= 6
    out = run_node(tmp_path, OV / "overlayModel.js", """
      const res = D.map(({ payload, want }) => {
        const zero = M.placementAt(payload, 0);
        const offs = M.offsetsOf(payload);
        return {
          shown: zero.score_total, want,
          each: offs.every((o) => M.placementAt(payload, o).offset_bars === o),
          snap: M.snapOffset(payload, 0.4), step: M.stepOffset(payload, 0, 1),
          offs,
          land: offs.map((o) => { const r = M.placementAt(payload, o);
            return Math.abs(M.onBed(r.alignment_downbeat, r, payload.stretch) - r.bed_landing_sec); }),
          back: offs.map((o) => { const r = M.placementAt(payload, o);
            return Math.abs(M.onVocal(M.onBed(r.alignment_downbeat + 3.1, r, payload.stretch), r, payload.stretch) - (r.alignment_downbeat + 3.1)); }),
          phrases: payload.phrases.map((ph) => M.onBed(ph.downbeat, zero, payload.stretch)),
          incoming: zero.incoming_phrases.map((q) => q.at),
          aud: M.auditionCandidate(payload, zero, null, true),
          spans: zero.spans,
          errs: M.phraseErrorsMs(payload, zero),
        };
      });
      console.log(JSON.stringify(res));""", data)
    for o in out:
        assert o["shown"] == o["want"]
        assert o["each"] and o["snap"] == 0
        assert o["step"] == next((x for x in o["offs"] if x > 0), 0)
        assert max(o["land"]) < 1e-9 and max(o["back"]) < 1e-9
        # The same mapping the server drew the incoming phrase starts with.
        for t in o["phrases"]:
            if t >= o["incoming"][0] - 1e-3:
                assert min(abs(t - x) for x in o["incoming"]) < 1e-3
        # The audition starts on the play spans, with the ghost riding along.
        assert o["aud"]["vocal_play_start"] == pytest.approx(o["spans"]["vocal_play_start"])
        assert o["aud"]["bed_play_start"] == pytest.approx(o["spans"]["bed_play_start"])
        assert o["aud"]["ghost"]["stem"] == "vocals"
    errs = [e for o in out for e in o["errs"]]
    assert sum(e <= 46.44 for e in errs) / len(errs) >= 0.9


def test_an_audition_from_a_bar_maps_both_clips_to_that_bar(tmp_path):
    payload = {"stretch": 1.02, "placements": []}
    row = {"alignment_downbeat": 50.0, "bed_landing_sec": 30.0, "vocal_play_start": 49.5,
           "bed_play_start": 29.49, "vocal_section_end": 80.0, "inst_section_start": 30.0,
           "vocal_song_id": 1, "inst_song_id": 2, "vocal_section_idx": 3,
           "inst_section_idx": 4, "offset_bars": 2, "phrase": 1}
    out = run_node(tmp_path, OV / "overlayModel.js", """
      const a = M.auditionCandidate(D.payload, D.row, 34.08, false);
      const b = M.auditionCandidate(D.payload, D.row, 10.0, false);
      console.log(JSON.stringify({ a, b, k: M.auditionKey(D.row), k2: M.auditionKey(D.row, 34.08) }));""",
                   {"payload": payload, "row": row})
    assert out["a"]["bed_play_start"] == 34.08
    assert out["a"]["vocal_play_start"] == pytest.approx(50.0 + 4.08 / 1.02, abs=1e-4)
    assert out["a"]["ghost"] is None
    # A bar before the placement's own start plays the placement from its pickup.
    assert out["b"]["vocal_play_start"] == 49.5 and out["b"]["bed_play_start"] == 29.49
    assert out["k"] == "1:2:3:4@1/2" and out["k2"] != out["k"]


# ── contracts ────────────────────────────────────────────────────────────────

PANEL = _read(OV / "OverlayPanel.jsx")
CANVAS = _read(OV / "OverlayCanvas.jsx")
MODEL = _read(OV / "overlayModel.js")


def test_one_player_and_ratings_through_use_ratings():
    for src in (PANEL, CANVAS, MODEL):
        assert "new Audio" not in src and "MashupEngine" not in src \
            and "AudioContext" not in src
    assert 'kind: "pair"' in PANEL and "player.toggle(" in PANEL
    assert 'context: "track_detail"' in PANEL
    assert 'ratings.rate(row, n, "track_detail")' in PANEL
    assert "player.pair.setGhost" in PANEL and "player.pair.setStemMode" in PANEL
    hook = _read(SRC / "hooks" / "useHookAudition.js")
    assert "setGhost" in hook and 'e.setVoice("ghost"' in hook
    assert "ghostOn, setGhost" in hook


def test_keys_belong_to_the_panel_only_while_it_has_focus():
    assert "onKeyDown={onKeyDown}" in PANEL and "tabIndex={0}" in PANEL
    assert "window.addEventListener" not in PANEL
    assert "e.preventDefault(); e.stopPropagation();" in PANEL
    for k in ('k === " "', '"ArrowLeft"', 'k === "o"', 'k === "v"', 'k === "b"',
              'k >= "1" && k <= "5"', 'k === "["'):
        assert k in PANEL, k
    detail = _read(SRC / "components" / "TrackDetail.jsx")
    assert 'if (e.defaultPrevented || el?.closest?.(".ov-panel")) return;' in detail
    assert "<OverlayPanel" in detail


def test_the_canvas_reads_tokens_and_draws_on_one_time_axis():
    for src in (PANEL, CANVAS, MODEL):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", src), "no hex colours in components"
    assert "cssVar(" in CANVAS
    for token in ('"--tier-strong"', '"--tier-maybe"', '"--tier-long"'):
        assert token in MODEL
    for token in ('"--vocal"', '"--bed"', '"--muted"', '"--playhead"'):
        assert token in CANVAS
    # Seconds everywhere: the incoming vocal is mapped by onBed, x by xOf.
    assert "onBed(t, row, stretch)" in CANVAS and "xOf(t, view, W)" in CANVAS
    assert "bar_count" not in CANVAS


def test_studio_and_the_fl_export_carry_the_placement():
    pm = _read(SRC / "components" / "pairs" / "pairModel.js")
    assert "placement: c.offset_bars != null" in pm
    assert "vocal_play_start: c.vocal_play_start ?? null" in pm
    studio = _read(SRC / "components" / "MixStudio.jsx")
    assert "armed ? activeOption?.placement ?? null : null" in studio
    assert "...(placement ? { placement } : {})" in _read(SRC / "api.js")
    assert "{ phrase: row.phrase ?? 0, offset_bars: row.offset_bars ?? 0 }" in PANEL


def test_the_overflow_check_walks_the_overlay():
    ui = _read(ROOT / "tests" / "ui" / "ui_check.cjs")
    assert '["track-overlay"' in ui and ".ov-panel" in ui


def test_every_overlay_class_is_styled():
    css = _read(SRC / "styles.css")
    for cls in set(re.findall(r'className[=:]\s*[{"`]?[^"`}]*?\b(ov-[a-z-]+)', PANEL + CANVAS)):
        assert f".{cls}" in css, cls
