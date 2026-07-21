import importlib


def test_firecrawl_key_from_env(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-123")
    import config
    importlib.reload(config)
    assert config.FIRECRAWL_API_KEY == "fc-test-123"
    assert config.FIRECRAWL_KEY_SOURCE == "env"
    assert config.FIRECRAWL_SCRAPE_URL.endswith("/v2/scrape")


def test_firecrawl_key_default_empty(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    import config
    importlib.reload(config)
    assert config.FIRECRAWL_API_KEY == ""
