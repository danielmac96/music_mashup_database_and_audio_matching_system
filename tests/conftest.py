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
