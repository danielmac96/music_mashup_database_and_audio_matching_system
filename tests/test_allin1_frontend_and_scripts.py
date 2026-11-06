"""The allin1 work's frontend contract (readme §3 Queue, Settings). Frontend:
read the source."""
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
SRC = ROOT / "frontend" / "src"
sys.path.insert(0, str(ROOT))


def _read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def test_queue_shows_the_allin1_stage_and_its_pool():
    q = _read("hooks/useQueue.js")
    assert '["allin1", "allin1"]' in q
    screen = _read("components/QueueScreen.jsx")
    assert '["allin1", "allin1"]' in screen
    from api import queue_eta
    assert "allin1" in queue_eta.POOL_OF and "allin1" in queue_eta._STAGE_ORDER


def test_tuning_offers_the_timing_knobs_and_no_engine_switch():
    t = _read("components/TuningPanel.jsx")
    for key in ("timing_tol_ms", "timing_weight"):
        assert key in t, key
    for gone in ("separation_pipeline", "segmentation_engine", "stem_separator", "stem_mode"):
        assert gone not in t, gone
    b = _read("components/BulkReprocess.jsx")
    assert 'run("allin1", "stale")' in b and "switch_pipeline" not in b


def test_the_engines_tab_and_its_endpoints_are_gone():
    assert not (SRC / "components" / "EngineCompare.jsx").exists()
    assert '"engines"' not in _read("components/AnalysisScreen.jsx")
    api = _read("api.js")
    assert "/api/analysis/engines/" not in api and "engine-report" not in api


