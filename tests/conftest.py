"""Suite-wide defaults.

The app analyses with Essentia (readme §9), which has no Windows wheels. The
suite pins the librosa analyser so the Windows CI leg keeps exercising the
librosa code that structure, band occupancy and stem quality still run on;
Essentia-mode tests set MASHUP_ANALYZER themselves, and monkeypatch undoes
this per test. Model downloads are switched off for the same run: an
analysis test must never fetch Essentia's models over the network."""
import pytest


@pytest.fixture(autouse=True)
def _pin_librosa_analyzer(monkeypatch):
    monkeypatch.setenv("MASHUP_ANALYZER", "librosa")
    monkeypatch.setenv("MASHUP_ESSENTIA_MODEL_FETCH", "0")
    # The preview's reachability check (ingest/errors.reachable) stays offline.
    monkeypatch.setenv("MASHUP_REACH_CHECK", "0")
    # Renders keep the phase vocoder under test (exact, ffmpeg-independent);
    # tests/test_phase6_export.py opts into Rubber Band.
    monkeypatch.setenv("MASHUP_STRETCH", "librosa")
    # The essentia engine and the legacy pipeline unless a test opts in: a
    # developer's settings.json with allin1 on must not change what the suite
    # scores (readme §5.5).
    monkeypatch.setenv("MASHUP_SEGMENTATION_ENGINE", "essentia")
    monkeypatch.setenv("MASHUP_SEPARATION_PIPELINE", "legacy_2stem")
