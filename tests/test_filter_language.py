"""One filter language (master plan 3B): the Library's search parser
(useLibraryFilters.parseQuery) is the one the parts tables and the pair dock
read, and the pair words — dropin, strong/good/maybe/longshot, free/light/heavy,
rated/unrated/loved, q50+ — are parsed once and applied where they mean
something, and named where they do not."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "frontend" / "src"


def _read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def _parse(tmp_path, queries):
    node = shutil.which("node")
    if not node or not (ROOT / "frontend" / "node_modules" / "esbuild").exists():
        pytest.skip("needs Node and frontend/node_modules (npm install)")
    out = tmp_path / "lf.mjs"
    local = ROOT / "frontend" / f".lf-test-build-{tmp_path.name}.mjs"
    local.write_text(
        "import { build } from 'esbuild';\n"
        f"await build({{ entryPoints: [{json.dumps(str(SRC / 'hooks' / 'useLibraryFilters.js'))}], "
        "bundle: true, format: 'esm', platform: 'node', logLevel: 'error', "
        f"outfile: {json.dumps(str(out))} }});\n", encoding="utf-8")
    try:
        subprocess.run([node, str(local)], check=True, cwd=ROOT / "frontend")
    finally:
        local.unlink(missing_ok=True)
    script = tmp_path / "run.mjs"
    script.write_text(
        f'import * as M from "{out.as_uri()}";\n'
        f"const Q = {json.dumps(queries)};\n"
        "console.log(JSON.stringify(Q.map((q) => M.parseQuery(q, ['House']))));\n",
        encoding="utf-8")
    res = subprocess.run([node, str(script)], capture_output=True, text=True, check=True,
                         cwd=ROOT / "frontend")
    return json.loads(res.stdout)


def test_pair_words_parse_once(tmp_path):
    got = _parse(tmp_path, ["dropin 8A~ strong longshot free loved q60+ house 124-128 Midnight"])
    p, rest = got[0]["parsed"], got[0]["rest"]
    assert p["dropin"] is True and p["tiers"] == ["strong", "long_shot"]
    assert p["effort"] == "free" and p["rated"] == "loved" and p["qualityMin"] == 0.6
    assert p["keyNeighbours"] == ["8A"] and p["genres"] == ["House"] and p["bpm"] == [124, 128]
    assert rest == "Midnight"


def test_each_surface_reads_the_one_parser():
    assert 'import { parseQuery } from "../../hooks/useLibraryFilters"' in _read(
        "components/segments/segmentModel.js")
    dock = _read("hooks/usePairDock.js")
    assert 'import { parseQuery } from "./useLibraryFilters";' in dock
    assert "export function pairQuery(q)" in dock
    pd = _read("components/PairDock.jsx")
    assert "pairQuery(search)" in pd and 'e.key === "Enter") { e.preventDefault(); pinSearch(); }' in pd
    lib = _read("hooks/useLibraryFilters.js")
    # Tracks answer "rated"; the words a track cannot answer are named.
    assert 'filters.rated === "loved" && !(t.rating >= 4)' in lib
    assert "a pair filter: type it in the dock" in lib
    model = _read("components/segments/segmentModel.js")
    assert "qualityMin: parsed.qualityMin ?? f.qualityMin" in model
