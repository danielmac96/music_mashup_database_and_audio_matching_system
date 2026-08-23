"""SoundCloud OAuth 2.1 — the write path. Dormant until credentials exist.

Creating a playlist on your account, liking, reposting: all of it needs OAuth
against a *registered* app. Registration is open and self-serve — there is no
approval queue — but it is gated on a **SoundCloud Artist Pro subscription**.
The blocker is therefore a recurring cost this repo has not paid, not an
impossibility. So this module is built, wired and tested, and reports
``configured: false`` until a client id and secret appear in settings. Every
write endpoint answers 501 with an explanation rather than failing obscurely.

The flow below was checked against the live spec on 2026-08-23 and matches it:
OAuth 2.1 with PKCE (S256) required, ``secure.soundcloud.com`` for both
authorize and token, ~1h access tokens and single-use refresh tokens.

Two things are **unverified** and must be checked before any of this is
switched on, rather than assumed:

* whether ``http://localhost`` / ``http://127.0.0.1`` is an acceptable
  registered redirect URI. The docs do not say, and a local-only app has
  nowhere else to send the callback.
* whether the numeric track ids the scraped ``api-v2`` browse layer returns are
  the same id space ``api.soundcloud.com`` accepts in a playlist write. Crates
  freeze v2 ids, so if the spaces disagree every push writes the wrong tracks.

Deliberately different from the read layer in two ways, so nothing can leak
between them:

* it talks to ``https://api.soundcloud.com`` (the registered-app API), not
  ``api-v2`` (SoundCloud's internal endpoint the browse layer scrapes into);
* it sends ``Authorization: OAuth <token>``. The read layer sends no auth header
  at all, and must not start — using a scraped client_id to attempt writes would
  be a terms violation, would fail, and if noticed would get the READ path
  blocked, which would take the frozen mixes auto-resolver down with it.

Token storage is a separate file from settings.json, because GET /api/settings is
read by the browser and must never carry a bearer token.

Refresh is lazy, inside ``_token()``. No background thread: a refresh loop for a
feature that may never activate is pure liability.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

API = "https://api.soundcloud.com"
AUTHORIZE_URL = "https://secure.soundcloud.com/authorize"
TOKEN_URL = "https://secure.soundcloud.com/oauth/token"

# Refresh this many seconds before the token actually expires, so a request never
# races the boundary.
_REFRESH_MARGIN_SECS = 60

_SETUP_HINT = (
    "SoundCloud writing needs a registered app. Add soundcloud_client_id and "
    "soundcloud_client_secret in Settings (or set SOUNDCLOUD_CLIENT_ID and "
    "SOUNDCLOUD_CLIENT_SECRET), then connect your account. Registering an app "
    "is open and self-serve, but SoundCloud requires an Artist Pro "
    "subscription to issue credentials — reading, searching and building "
    "local crates all work without any of this."
)


class NotConfigured(RuntimeError):
    """No client credentials, or no connected account. Routes answer 501."""


class SoundCloudAuthError(RuntimeError):
    """The token exchange or refresh was rejected."""


# ── credentials + token storage ──────────────────────────────────────────────

def _credentials() -> tuple:
    """Read live rather than at import, so saving them in Settings takes effect
    without a restart — the same reason the scoring knobs are live-read."""
    import config
    return (getattr(config, "SOUNDCLOUD_CLIENT_ID", "") or "",
            getattr(config, "SOUNDCLOUD_CLIENT_SECRET", "") or "")


def _token_path():
    import config
    return config.soundcloud_token_path()


def _read_token() -> Dict[str, Any]:
    try:
        p = _token_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        log.warning("SoundCloud token file is unreadable; treating as disconnected")
    return {}


def _write_token(data: Dict[str, Any]) -> None:
    p = _token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)          # best effort; a no-op on most Windows setups
    except OSError:
        pass


def _clear_token() -> None:
    try:
        _token_path().unlink()
    except (OSError, FileNotFoundError):
        pass


def is_configured() -> bool:
    """Whether an app's credentials are present. Not whether it is connected."""
    client_id, client_secret = _credentials()
    return bool(client_id and client_secret)


def is_authorized() -> bool:
    """Whether we hold a token that can be used or refreshed."""
    tok = _read_token()
    return bool(tok.get("access_token") and (tok.get("refresh_token")
                                             or tok.get("expires_at", 0) > time.time()))


def status() -> Dict[str, Any]:
    """What the account panel shows. Pure local reads — no network, so this is
    cheap enough to call on every Discovery mount."""
    configured = is_configured()
    tok = _read_token() if configured else {}
    authorized = bool(configured and tok.get("access_token"))
    return {
        "configured": configured,
        "authorized": authorized,
        "username": tok.get("username", "") if authorized else "",
        "expires_at": tok.get("expires_at", 0) if authorized else 0,
        "reason": "" if configured else _SETUP_HINT,
    }


def require_ready() -> None:
    """Raise NotConfigured unless a write can actually be attempted."""
    if not is_configured():
        raise NotConfigured(_SETUP_HINT)
    if not is_authorized():
        raise NotConfigured(
            "SoundCloud credentials are set but no account is connected. "
            "Use Connect in Settings to authorise this app.")


# ── the PKCE flow ────────────────────────────────────────────────────────────

def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def authorize_url(redirect_uri: str, state: Optional[str] = None) -> Dict[str, str]:
    """Begin the flow. Returns {url, state, verifier}.

    The caller keeps the verifier and hands it back to exchange_code — it never
    goes over the wire, which is the whole point of PKCE."""
    if not is_configured():
        raise NotConfigured(_SETUP_HINT)
    client_id, _ = _credentials()
    verifier = secrets.token_urlsafe(64)
    state = state or secrets.token_urlsafe(16)
    query = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "code_challenge": _challenge(verifier),
        "code_challenge_method": "S256",
        "state": state,
    })
    return {"url": f"{AUTHORIZE_URL}?{query}", "state": state, "verifier": verifier}


def _post_form(url: str, fields: Dict[str, str], *, _post=None) -> Dict[str, Any]:
    if _post is not None:
        return _post(url, fields)
    body = urllib.parse.urlencode(fields).encode("ascii")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json; charset=utf-8",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300] if exc.fp else ""
        raise SoundCloudAuthError(f"SoundCloud rejected the token request "
                                  f"(HTTP {exc.code}). {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SoundCloudAuthError(f"Could not reach SoundCloud: {exc}") from exc


def _store(payload: Dict[str, Any], *, username: str = "") -> Dict[str, Any]:
    if not payload.get("access_token"):
        raise SoundCloudAuthError("SoundCloud returned no access token.")
    existing = _read_token()
    data = {
        "access_token": payload["access_token"],
        # A refresh response may omit refresh_token; losing it would silently
        # turn a connected account into one that expires and cannot recover.
        "refresh_token": payload.get("refresh_token") or existing.get("refresh_token", ""),
        "expires_at": time.time() + float(payload.get("expires_in") or 3600),
        "scope": payload.get("scope", ""),
        "username": username or existing.get("username", ""),
    }
    _write_token(data)
    return data


def exchange_code(code: str, verifier: str, redirect_uri: str, *, _post=None) -> Dict[str, Any]:
    """Finish the flow and persist the token."""
    if not is_configured():
        raise NotConfigured(_SETUP_HINT)
    client_id, client_secret = _credentials()
    payload = _post_form(TOKEN_URL, {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
        "code": code,
    }, _post=_post)
    data = _store(payload)
    return {"authorized": True, "username": data.get("username", "")}


def refresh(*, _post=None) -> Dict[str, Any]:
    if not is_configured():
        raise NotConfigured(_SETUP_HINT)
    tok = _read_token()
    if not tok.get("refresh_token"):
        raise NotConfigured("No refresh token — reconnect the SoundCloud account.")
    client_id, client_secret = _credentials()
    payload = _post_form(TOKEN_URL, {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": tok["refresh_token"],
    }, _post=_post)
    return _store(payload)


def disconnect() -> None:
    _clear_token()


def _token(*, _post=None) -> str:
    """A usable access token, refreshed lazily when it is close to expiring."""
    require_ready()
    tok = _read_token()
    if tok.get("expires_at", 0) - _REFRESH_MARGIN_SECS <= time.time():
        tok = refresh(_post=_post)
    return tok["access_token"]


# ── authed requests ──────────────────────────────────────────────────────────

def authed(method: str, path: str, payload: Optional[dict] = None,
           *, _request=None, _post=None) -> Any:
    """One authenticated call against api.soundcloud.com."""
    token = _token(_post=_post)
    url = f"{API}{path}"
    if _request is not None:
        return _request(method, url, payload, token)

    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method.upper(), headers={
        "Authorization": f"OAuth {token}",
        "Accept": "application/json; charset=utf-8",
        **({"Content-Type": "application/json; charset=utf-8"} if body else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300] if exc.fp else ""
        raise SoundCloudAuthError(
            f"SoundCloud API {method} {path} failed (HTTP {exc.code}). {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SoundCloudAuthError(f"Could not reach SoundCloud: {exc}") from exc


# ── writes ───────────────────────────────────────────────────────────────────
# Thin by design. The value of this module is the gate above, not these.

def create_playlist(title: str, track_ids, *, sharing: str = "private", **kw) -> dict:
    """Create a playlist. Private by default — pushing a crate should never
    publish to your followers unless you say so."""
    return authed("POST", "/playlists", {"playlist": {
        "title": title,
        "sharing": sharing,
        "tracks": [{"id": int(t)} for t in track_ids if str(t).isdigit()],
    }}, **kw)


def set_playlist_tracks(playlist_id: str, track_ids, **kw) -> dict:
    """Replace a playlist's tracks — this is also how you reorder one."""
    return authed("PUT", f"/playlists/{playlist_id}", {"playlist": {
        "tracks": [{"id": int(t)} for t in track_ids if str(t).isdigit()],
    }}, **kw)


def delete_playlist(playlist_id: str, **kw) -> dict:
    return authed("DELETE", f"/playlists/{playlist_id}", **kw)


def like_track(track_id: str, **kw) -> dict:
    return authed("POST", f"/likes/tracks/{track_id}", **kw)


def unlike_track(track_id: str, **kw) -> dict:
    return authed("DELETE", f"/likes/tracks/{track_id}", **kw)


def repost_track(track_id: str, **kw) -> dict:
    return authed("POST", f"/reposts/tracks/{track_id}", **kw)
