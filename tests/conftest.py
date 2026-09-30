"""Suite-wide defaults.

The app analyses with Essentia (readme §9), which has no Windows wheels. The
suite pins the librosa analyser so the Windows CI leg keeps exercising the
librosa code that structure, band occupancy and stem quality still run on;
Essentia-mode tests set MASHUP_ANALYZER themselves, and monkeypatch undoes
this per test."""
import pytest


@pytest.fixture(autouse=True)
def _pin_librosa_analyzer(monkeypatch):
    monkeypatch.setenv("MASHUP_ANALYZER", "librosa")
