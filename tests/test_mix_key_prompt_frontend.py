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
