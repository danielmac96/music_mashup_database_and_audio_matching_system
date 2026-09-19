"""Mixes tab: a 501 on import asks for the Firecrawl key in place.

The endpoint answers 501 for exactly one thing — "a Firecrawl key would fix
this" — so the prompt keys off the status. Pinned from Python by reading the
JSX, the same trick tests/test_stale_frontend.py uses.
"""
import re
from pathlib import Path

SRC = (Path(__file__).parent.parent / "frontend" / "src" / "components"
       / "MixImporter.jsx").read_text(encoding="utf-8")


def test_prompt_appears_on_501_and_saves_the_key():
    assert 'startsWith("501")' in SRC
    assert "firecrawl_api_key" in SRC
    assert "api.saveSettings(" in SRC


def test_key_field_is_masked():
    assert re.search(r'type="password"', SRC)


def test_key_is_never_shown_back():
    for call in re.findall(r"(?:toast|setError)\(([^;]*)", SRC):
        assert "fcKey" not in call


def test_the_prompt_also_comes_back_when_the_job_rejects_the_key():
    # The scrape is a job now, so a bad key is discovered a request later than
    # the route's own 501. The job result carries needs_key for exactly that.
    assert "needs_key" in SRC
    assert re.search(r"setNeedsKey\(!!\(importJob\.result \|\| \{\}\)\.needs_key\)", SRC)


def test_import_and_ingest_are_polled_as_jobs():
    # Neither fits in an HTTP request: a stealth render of a heavy tracklist
    # takes minutes, and a 200-track ingest is 200 metadata fetches.
    for name in ("importJobId", "ingestJobId"):
        assert f"useJobPolling({name})" in SRC
    assert "res.job_id" in SRC and "setIngestJobId(res.job_id)" in SRC


def test_a_running_job_reports_its_progress():
    # A multi-minute wait with a frozen button is what this replaced.
    assert "ingestJob?.message" in SRC
    assert "importJob.message" in SRC


def test_the_import_button_never_passes_its_click_event_as_refresh():
    # onClick={importUrl} would hand React's event object to `refresh` and pay
    # for a fresh render on every click.
    assert "onClick={importUrl}" not in SRC
    assert "onClick={() => importUrl()}" in SRC
