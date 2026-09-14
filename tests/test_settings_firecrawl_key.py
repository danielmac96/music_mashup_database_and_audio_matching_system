"""The Firecrawl key is writable from the app, read live, and never sent back.

1001tracklists is Turnstile-walled, so importing a Big Bootie set needs a
Firecrawl key. Before this, the only way to supply one was an env var that
docker-compose never passed through — so the Mixes tab 501'd with no way out.
"""
import importlib
import json

import pytest
from fastapi import HTTPException

KEY = "fc-PLANTEDKEY0123456789"


@pytest.fixture()
def routes(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    import config
    importlib.reload(config)
    import api.routes.settings as settings_routes
    importlib.reload(settings_routes)
    return settings_routes, config


def test_saved_key_is_read_live_and_reported_as_presence_only(routes):
    settings_routes, config = routes
    assert config.current_firecrawl_api_key() == ""
    assert settings_routes.get_settings()["firecrawl_api_key"] == {
        "value": False, "source": "default"}

    saved = settings_routes.save_settings(
        settings_routes.SaveSettingsRequest(firecrawl_api_key=f"  {KEY}  "))

    # No reload: the key applies to the very next import.
    assert config.current_firecrawl_api_key() == KEY
    assert saved["restart_required"] is False
    got = settings_routes.get_settings()
    assert got["firecrawl_api_key"] == {"value": True, "source": "settings"}
    # These responses go to the browser.
    for body in (saved, got):
        assert KEY not in json.dumps(body)


def test_env_pinned_key_cannot_be_overwritten_from_the_app(routes, monkeypatch):
    settings_routes, config = routes
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-from-env")
    assert config.current_firecrawl_key_source() == "env"
    with pytest.raises(HTTPException) as exc:
        settings_routes.save_settings(
            settings_routes.SaveSettingsRequest(firecrawl_api_key=KEY))
    assert exc.value.status_code == 400
    assert config.current_firecrawl_api_key() == "fc-from-env"


def test_empty_env_passthrough_does_not_pin(routes, monkeypatch):
    # docker-compose passes ${FIRECRAWL_API_KEY:-}, i.e. "" when .env has none.
    settings_routes, config = routes
    monkeypatch.setenv("FIRECRAWL_API_KEY", "")
    settings_routes.save_settings(
        settings_routes.SaveSettingsRequest(firecrawl_api_key=KEY))
    assert config.current_firecrawl_api_key() == KEY
    assert config.current_firecrawl_key_source() == "settings"


@pytest.mark.parametrize("bad", ["   ", "fc-abc def"])
def test_blank_or_spaced_key_is_rejected(routes, bad):
    settings_routes, config = routes
    with pytest.raises(HTTPException) as exc:
        settings_routes.save_settings(
            settings_routes.SaveSettingsRequest(firecrawl_api_key=bad))
    assert exc.value.status_code == 400
    assert config.current_firecrawl_api_key() == ""
