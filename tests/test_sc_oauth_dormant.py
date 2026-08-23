"""The SoundCloud write layer while it is dormant — which is the normal state.

Registering a SoundCloud app is open and self-serve but requires an Artist Pro
subscription, so most installs will never have credentials. What matters is that
the feature explains itself instead of failing obscurely, that nothing leaks a
token to the browser, and that the flow is actually correct for the day a key
does appear.
"""
import importlib
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.delenv("SOUNDCLOUD_CLIENT_ID", raising=False)
    monkeypatch.delenv("SOUNDCLOUD_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("SC_CLIENT_ID", raising=False)

    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()

    import ingest.soundcloud_oauth as oauth
    importlib.reload(oauth)
    # settings.py does `import config` at module load, so without this reload it
    # keeps answering from the pre-reload config module and `configure()` below
    # would patch a module nobody reads.
    import api.routes.settings as settings_routes
    importlib.reload(settings_routes)
    import api.routes.playlists as pl
    importlib.reload(pl)
    import api.routes.discovery as disc
    importlib.reload(disc)
    import api.routes.crates as crates
    importlib.reload(crates)
    import api.server as server
    importlib.reload(server)

    return TestClient(server.app), oauth, config, models


def configure(config, monkeypatch, client_id="cid", secret="csecret"):
    """Pretend a registered app exists, without writing real credentials."""
    monkeypatch.setattr(config, "SOUNDCLOUD_CLIENT_ID", client_id)
    monkeypatch.setattr(config, "SOUNDCLOUD_CLIENT_SECRET", secret)


# ── dormant ──────────────────────────────────────────────────────────────────

def test_not_configured_without_credentials(env):
    _, oauth, _, _ = env
    assert oauth.is_configured() is False
    st = oauth.status()
    assert st["authorized"] is False
    assert "Artist Pro" in st["reason"]


def test_status_endpoint_says_read_yes_write_no(env):
    client, _, _, _ = env
    body = client.get("/api/discovery/status").json()
    assert body["read_enabled"] is True
    assert body["write_enabled"] is False


@pytest.mark.parametrize("method,path,payload", [
    ("post", "/api/discovery/account/authorize", {"redirect_uri": "http://localhost/cb"}),
    ("post", "/api/discovery/account/callback",
     {"code": "c", "verifier": "v", "redirect_uri": "http://localhost/cb"}),
    ("post", "/api/discovery/tracks/123/like", None),
])
def test_write_endpoints_are_501_with_setup_instructions(env, method, path, payload):
    """501, not 500 and not a silent no-op — and the message names what to set."""
    client, _, _, _ = env
    r = getattr(client, method)(path, json=payload) if payload else getattr(client, method)(path)
    assert r.status_code == 501
    assert "soundcloud_client_id" in r.json()["detail"]


def test_crate_push_is_501_when_dormant(env):
    client, _, _, _ = env
    cid = client.post("/api/crates", json={"name": "c"}).json()["id"]
    client.post(f"/api/crates/{cid}/items", json={"rows": [{
        "title": "T", "artist": "A", "track_id": "1001",
        "source_url": "https://soundcloud.com/a/t"}]})
    r = client.post(f"/api/crates/{cid}/push", json={})
    assert r.status_code == 501
    assert "soundcloud_client_id" in r.json()["detail"]


def test_push_without_track_ids_is_400_not_501(env, monkeypatch):
    """A crate of hand-typed URLs genuinely cannot be pushed; that is a different
    problem from the app not being set up, and says so."""
    client, _, config, _ = env
    configure(config, monkeypatch)
    cid = client.post("/api/crates", json={"name": "c"}).json()["id"]
    client.post(f"/api/crates/{cid}/items", json={"rows": [{
        "title": "T", "source_url": "https://soundcloud.com/a/t"}]})
    r = client.post(f"/api/crates/{cid}/push", json={})
    assert r.status_code == 400
    assert "track id" in r.json()["detail"]


def test_configured_but_not_connected_still_blocks(env, monkeypatch):
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    assert oauth.is_configured() is True
    assert oauth.is_authorized() is False

    r = client.post("/api/discovery/tracks/123/like")
    assert r.status_code == 501
    assert "no account is connected" in r.json()["detail"]


# ── secrets stay on the server ───────────────────────────────────────────────

def test_settings_never_exposes_the_secret_or_a_token(env, monkeypatch):
    """GET /api/settings is read by the browser."""
    client, oauth, config, _ = env
    configure(config, monkeypatch, client_id="SUPERSECRETID", secret="SUPERSECRETVALUE")
    oauth._write_token({"access_token": "PLANTEDTOKEN", "refresh_token": "PLANTEDREFRESH",
                        "expires_at": time.time() + 3600, "username": "me"})

    body = client.get("/api/settings").text
    for leak in ("SUPERSECRETVALUE", "PLANTEDTOKEN", "PLANTEDREFRESH"):
        assert leak not in body

    # Presence is reported, so the UI can show "configured" without the value.
    assert client.get("/api/settings").json()["soundcloud_client_secret"]["value"] is True


def test_account_status_never_returns_the_token(env, monkeypatch):
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    oauth._write_token({"access_token": "PLANTEDTOKEN", "refresh_token": "R",
                        "expires_at": time.time() + 3600, "username": "me"})
    body = client.get("/api/discovery/account").text
    assert "PLANTEDTOKEN" not in body
    assert client.get("/api/discovery/account").json()["username"] == "me"


def test_token_is_not_stored_in_settings_json(env, monkeypatch):
    """settings.json is served; the token file is not."""
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    oauth._write_token({"access_token": "PLANTEDTOKEN", "refresh_token": "R",
                        "expires_at": time.time() + 3600})
    settings_file = config.settings_path()
    if settings_file.exists():
        assert "PLANTEDTOKEN" not in settings_file.read_text(encoding="utf-8")
    assert "PLANTEDTOKEN" in config.soundcloud_token_path().read_text(encoding="utf-8")


# ── the flow is correct for the day a key exists ─────────────────────────────

def test_authorize_url_is_a_valid_pkce_request(env, monkeypatch):
    import base64
    import hashlib
    import urllib.parse

    client, oauth, config, _ = env
    configure(config, monkeypatch, client_id="MYAPPID")
    out = oauth.authorize_url("http://localhost:5173/cb")

    q = urllib.parse.parse_qs(urllib.parse.urlsplit(out["url"]).query)
    assert q["response_type"] == ["code"]
    assert q["client_id"] == ["MYAPPID"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["redirect_uri"] == ["http://localhost:5173/cb"]

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(out["verifier"].encode()).digest()).decode().rstrip("=")
    assert q["code_challenge"] == [expected]
    # The verifier is the secret half and must never appear in the URL.
    assert out["verifier"] not in out["url"]


def test_exchange_code_persists_the_token(env, monkeypatch):
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    sent = {}

    def fake_post(url, fields):
        sent.update(url=url, **fields)
        return {"access_token": "AT", "refresh_token": "RT", "expires_in": 3600,
                "scope": "non-expiring"}

    out = oauth.exchange_code("thecode", "theverifier", "http://cb", _post=fake_post)
    assert out["authorized"] is True
    assert sent["grant_type"] == "authorization_code"
    assert sent["code_verifier"] == "theverifier"
    assert oauth.is_authorized() is True


def test_refresh_keeps_the_refresh_token_when_the_response_omits_it(env, monkeypatch):
    """A refresh response may omit refresh_token; dropping it would silently turn
    a connected account into one that expires and cannot recover."""
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    oauth._write_token({"access_token": "OLD", "refresh_token": "KEEPME",
                        "expires_at": time.time() - 1, "username": "me"})

    oauth.refresh(_post=lambda url, fields: {"access_token": "NEW", "expires_in": 3600})
    stored = json.loads(config.soundcloud_token_path().read_text(encoding="utf-8"))
    assert stored["access_token"] == "NEW"
    assert stored["refresh_token"] == "KEEPME"
    assert stored["username"] == "me"


def test_expired_token_refreshes_before_a_write(env, monkeypatch):
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    oauth._write_token({"access_token": "OLD", "refresh_token": "RT",
                        "expires_at": time.time() - 1})
    refreshed = {"n": 0}

    def fake_post(url, fields):
        refreshed["n"] += 1
        return {"access_token": "NEW", "refresh_token": "RT", "expires_in": 3600}

    seen = {}
    oauth.like_track("123", _post=fake_post,
                     _request=lambda m, u, p, tok: seen.update(token=tok, url=u) or {})
    assert refreshed["n"] == 1
    assert seen["token"] == "NEW"


def test_writes_target_the_registered_app_api_not_api_v2(env, monkeypatch):
    """The read layer scrapes api-v2 anonymously; using that for writes would be
    a terms violation and, if noticed, would get READ blocked — which would take
    the frozen mixes auto-resolver down with it."""
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    oauth._write_token({"access_token": "AT", "refresh_token": "RT",
                        "expires_at": time.time() + 3600})
    seen = {}
    oauth.create_playlist("c", ["1", "2"],
                          _request=lambda m, u, p, tok: seen.update(url=u, payload=p) or {})
    assert seen["url"].startswith("https://api.soundcloud.com/")
    assert "api-v2" not in seen["url"]


def test_create_playlist_is_private_by_default(env, monkeypatch):
    """Pushing a shortlist must never publish to your followers by accident."""
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    oauth._write_token({"access_token": "AT", "refresh_token": "RT",
                        "expires_at": time.time() + 3600})
    seen = {}
    oauth.create_playlist("c", ["1"],
                          _request=lambda m, u, p, tok: seen.update(payload=p) or {})
    assert seen["payload"]["playlist"]["sharing"] == "private"
    assert seen["payload"]["playlist"]["tracks"] == [{"id": 1}]


def test_disconnect_clears_the_token(env, monkeypatch):
    client, oauth, config, _ = env
    configure(config, monkeypatch)
    oauth._write_token({"access_token": "AT", "refresh_token": "RT",
                        "expires_at": time.time() + 3600})
    assert client.post("/api/discovery/account/disconnect").json()["authorized"] is False
    assert oauth.is_authorized() is False
