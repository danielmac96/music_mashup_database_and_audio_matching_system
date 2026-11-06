"""Suite-wide defaults.

The app analyses with Essentia (readme §5.4), which has no Windows wheels. So
every test runs against a librosa-backed stand-in (tests/essentia_fake.py)
unless it is marked ``real_essentia`` (the real analyser; those run only where
Essentia imports) or ``real_analyzer_code`` (the real ``analyze_file_essentia``
driving stand-in algorithms the test installs itself). On the Linux CI leg
(``MASHUP_REQUIRE_ESSENTIA=1``) a missing Essentia is a failure, not a skip. Model downloads are switched off for the same run: an
analysis test must never fetch Essentia's models over the network."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import essentia_fake  # noqa: E402  (tests/essentia_fake.py)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_essentia: runs the real Essentia analyser (Linux/macOS only)")
    config.addinivalue_line(
        "markers", "real_analyzer_code: runs the real analyze_file_essentia over "
                   "stand-in Essentia algorithms (no install needed)")


def _essentia_importable() -> bool:
    try:
        import essentia  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def _suite_defaults(request, monkeypatch):
    monkeypatch.setenv("MASHUP_ESSENTIA_MODEL_FETCH", "0")
    # The preview's reachability check (ingest/errors.reachable) stays offline.
    monkeypatch.setenv("MASHUP_REACH_CHECK", "0")
    # Renders keep the phase vocoder under test (exact, ffmpeg-independent);
    # tests/test_phase6_export.py opts into Rubber Band.
    monkeypatch.setenv("MASHUP_STRETCH", "librosa")
    if request.node.get_closest_marker("real_essentia"):
        if not _essentia_importable():
            if os.environ.get("MASHUP_REQUIRE_ESSENTIA"):
                pytest.fail("Essentia must be installed on this CI leg")
            pytest.skip("Essentia is not installed here")
        return
    if request.node.get_closest_marker("real_analyzer_code"):
        return
    essentia_fake.install(monkeypatch)
