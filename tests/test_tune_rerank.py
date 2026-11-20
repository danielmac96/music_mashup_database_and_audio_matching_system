"""The dock's Tune drawer (master plan 3B, criterion 3.3): frontend/src/tune.js
re-ranks the loaded pairs client-side, and at the weights the server ranked
with its order IS the server's — run under Node against the real route on a
scored synthetic library. It also re-derives the rule's own total from the
per-term values on each row, and re-ranks 500 pairs inside 50 ms."""
import importlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _bundle(tmp_path):
    node = shutil.which("node")
    if not node or not (ROOT / "frontend" / "node_modules" / "esbuild").exists():
        pytest.skip("needs Node and frontend/node_modules (npm install)")
    out = tmp_path / "tune.mjs"
    local = ROOT / "frontend" / f".tune-test-build-{tmp_path.name}.mjs"
    local.write_text(
        "import { build } from 'esbuild';\n"
        f"await build({{ entryPoints: [{json.dumps(str(SRC / 'tune.js'))}], "
        "bundle: true, format: 'esm', platform: 'neutral', logLevel: 'error', "
        f"outfile: {json.dumps(str(out))} }});\n", encoding="utf-8")
    try:
        subprocess.run([node, str(local)], check=True, cwd=ROOT / "frontend")
    finally:
        local.unlink(missing_ok=True)
    return node, out


def _run(tmp_path, body: str, data: dict):
    node, mod = _bundle(tmp_path)
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    script = tmp_path / "run.mjs"
    script.write_text(
        f'import * as T from "{mod.as_uri()}";\nimport fs from "node:fs";\n'
        f'const D = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "data.json"))}, "utf8"));\n'
        f"{body}\n", encoding="utf-8")
    res = subprocess.run([node, str(script)], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    out = tmp_path_factory.mktemp("tune")
    spec = importlib.util.spec_from_file_location("synth_library",
                                                  ROOT / "scripts" / "synth_library.py")
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    synth.generate(out, 60, seed=5, planted=6)
    mp = pytest.MonkeyPatch()
    mp.setenv("MASHUP_DB_PATH", str(out / "mashup.db"))
    mp.setenv("MASHUP_AUDIO_ROOT", str(out / "audio"))
    mp.setenv("MASHUP_SETTINGS_DIR", str(out / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    from matcher.match import score_all_pairs
    score_all_pairs(db_path=m.DB_PATH)
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.server as server
    importlib.reload(server)
    client = TestClient(server.app)
    got = client.get("/api/mashups", params={
        "order": "score", "limit": 500, "max_per_song": 0,
        "combo_type": "vocal_over_instrumental"}).json()
    yield got
    mp.undo()


def test_the_listing_sends_the_weights_it_ranked_with(scored):
    tune = scored["tune"]
    assert set(tune) == {"match", "section", "slot", "effort", "dropinBonus", "alpha"}
    assert abs(sum(tune["match"].values()) - 1.0) < 1e-9
    assert tune["alpha"] == 0.0                      # no model: the rule's totals
    assert len(scored["candidates"]) > 40


def test_at_the_servers_weights_the_client_order_is_the_servers(scored, tmp_path):
    rows = [r for r in scored["candidates"] if not r.get("explore")]
    shuffled = rows[:]
    random.Random(3).shuffle(shuffled)
    out = _run(tmp_path, """
      const b = D.tune;
      const t = D.rows.map((r) => T.tunedTotal(r, b, b, b.alpha));
      const same = D.rows.every((r, i) => t[i] === r.score_total);
      console.log(JSON.stringify({ ids: T.tuneOrder(D.shuffled, b, b, b.alpha).map((r) => r.id), same }));
    """, {"rows": rows, "shuffled": shuffled, "tune": scored["tune"]})
    assert out["same"], "at the base weights every tuned total is the stored one, bit for bit"
    assert out["ids"] == [r["id"] for r in rows]


def test_the_client_arithmetic_is_the_rules(scored, tmp_path):
    """ruleOf(row, base, base), from the per-term values on the row, gives the
    rule's own stored total (to the stored terms' rounding)."""
    rows = [r for r in scored["candidates"] if r.get("score_section") is not None]
    out = _run(tmp_path, """
      const b = D.tune;
      console.log(JSON.stringify(D.rows.map((r) => T.ruleOf(r, b, b))));
    """, {"rows": rows, "tune": scored["tune"]})
    errs = [abs(x - r["score_rule"]) for x, r in zip(out, rows)]
    assert max(errs) < 2e-3, max(errs)


def test_tuning_moves_the_order_the_way_the_weight_says(scored, tmp_path):
    rows = [r for r in scored["candidates"] if not r.get("explore")]
    out = _run(tmp_path, """
      const b = D.tune;
      const heavy = { ...b, effort: 1.0 };
      const order = T.tuneOrder(D.rows, heavy, b, b.alpha);
      const eff = (rs) => rs.slice(0, 10).reduce((s, r) => s + (r.score_effort || 0), 0) / 10;
      console.log(JSON.stringify({ before: eff(D.rows), after: eff(order) }));
    """, {"rows": rows, "tune": scored["tune"]})
    assert out["after"] <= out["before"], "a heavier effort weight puts cheaper builds first"


def test_re_ranking_500_loaded_pairs_takes_under_50_ms(scored, tmp_path):
    base = [r for r in scored["candidates"] if not r.get("explore")]
    rows = [{**base[i % len(base)], "id": i} for i in range(500)]
    out = _run(tmp_path, """
      const b = D.tune;
      const w = { ...b, section: 0.4, slot: 0.35, effort: 0.1, dropinBonus: 0.05,
                  match: { ...b.match, key_score: 0.5 } };
      T.tuneOrder(D.rows, w, b, b.alpha);            // warm the JIT, as a drag does
      const times = [];
      for (let k = 0; k < 20; k += 1) {
        const t0 = performance.now();
        T.tuneOrder(D.rows, { ...w, effort: 0.1 + k / 100 }, b, b.alpha);
        times.push(performance.now() - t0);
      }
      times.sort((x, y) => x - y);
      console.log(JSON.stringify({ p95: times[18], max: times[19] }));
    """, {"rows": rows, "tune": scored["tune"]})
    print("tune re-rank 500 pairs:", out)
    assert out["p95"] < 50, out
