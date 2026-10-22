"""The allin1 work's frontend contract and its owner-run scripts' arithmetic
(readme §4 Analysis › Engines, Queue; §8 scripts). Frontend: read the source."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "frontend" / "src"
sys.path.insert(0, str(ROOT))


def _read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def test_analysis_has_an_engines_tab_fed_by_the_two_endpoints():
    screen = _read("components/AnalysisScreen.jsx")
    assert '["engines", "Engines"]' in screen and "<EngineCompare" in screen
    api = _read("api.js")
    assert "/api/analysis/engines/" in api and "/api/analysis/engine-report" in api
    eng = _read("components/EngineCompare.jsx")
    assert "getEngines" in eng and "getEngineReport" in eng
    # Pairs from both engines are rated through the one ratings hook.
    assert "ratings?.rate?.(c, stars)" in eng
    assert re.search(r"<AnalysisScreen[^>]*ratings={ratings}", _read("App.jsx"), re.S)


def test_every_engines_class_exists_in_the_stylesheet():
    css = _read("styles.css")
    eng = _read("components/EngineCompare.jsx")
    for cls in set(re.findall(r'className="([\w-]+)', eng)):
        if cls.startswith("eng-"):
            assert f".{cls}" in css, cls


def test_queue_shows_the_allin1_stage_and_its_pool():
    q = _read("hooks/useQueue.js")
    assert '["allin1", "allin1"]' in q
    assert 'return { state: "off" }' in q        # never a forever-"todo" in essentia mode
    screen = _read("components/QueueScreen.jsx")
    assert '["allin1", "allin1"]' in screen
    assert ".q-seg.off" in _read("styles.css")
    from api import queue_eta
    assert "allin1" in queue_eta.POOL_OF and "allin1" in queue_eta._STAGE_ORDER


def test_tuning_offers_the_two_switches_and_the_timing_knobs():
    t = _read("components/TuningPanel.jsx")
    for key in ("separation_pipeline", "segmentation_engine", "timing_tol_ms", "timing_weight"):
        assert key in t, key
    b = _read("components/BulkReprocess.jsx")
    assert 'run("allin1", "stale")' in b and 'run("switch_pipeline", "stale")' in b


def test_null_test_arithmetic():
    sys.path.insert(0, str(ROOT / "scripts"))
    from stem_null_test import null_test
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (44100, 2))
    same = null_test(a, a.copy())
    assert same["residual_dbfs"] < -200
    off = null_test(a, a + 0.001)
    assert off["residual_dbfs"] == pytest.approx(-60.0, abs=0.1)
    assert off["snr_db"] == pytest.approx(40.0, abs=0.5)


def test_phrase_check_scoring():
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_phrases import score
    detected = [{"onset_sec": 10.03}, {"onset_sec": 30.4}]
    s = score(detected, [10.0, 20.0, 30.0], beat=0.5)
    assert s["hits"] == 2 and s["within_50ms"] == 1
