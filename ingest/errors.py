"""One vocabulary for "a fetch failed": what went wrong, in words a user can
act on, the same wherever a link is fetched — the download stage, the
paste-bar preview, Discover, the Mixes scrape.

The categories are the download stage's (drm / premium / geo / private /
removed / network / outdated / unknown), plus `tool` (yt-dlp missing) and
`blocked` / `rate_limited` / `upstream` for the HTTP layer. The 2026-10-15
persona simulation found an offline import answering, after 15 s, "check that
yt-dlp is installed" (it was), and Discover printing a raw
`<urlopen error Tunnel connection failed: 403 Forbidden>`.

Does not import `database` (readme §7).
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

# (regex, kind, user-facing message) — first match on the joined lowercased
# error text wins, so order encodes priority. {ver} = installed yt-dlp version.
ERROR_CLASSES: Tuple[Tuple[str, str, str], ...] = (
    (r"drm.protected", "drm",
     "SoundCloud serves this track DRM-protected — it cannot be downloaded directly."),
    (r"go\+|premium|subscribers", "premium",
     "This track is for SoundCloud Go+ subscribers only."),
    (r"not available in your country|geo.?restrict", "geo",
     "This track is geo-blocked in your region."),
    (r"private|sign in|log ?in|authentication", "private",
     "This track is private or requires a SoundCloud login."),
    (r"unable to download json metadata.{0,80}404", "outdated",
     "SoundCloud rejected the request (HTTP 404). The track may have been removed, "
     "or yt-dlp {ver} is outdated — use the Update yt-dlp button on the Import tab."),
    (r"http error 404|not found|no longer available|has been removed", "removed",
     "Track not found — it may have been removed from SoundCloud."),
    (r"getaddrinfo|timed out|connection|temporary failure|unable to download webpage"
     r"|tunnel|name or service not known|network is unreachable|urlopen error",
     "network",
     "Network error — check your connection and retry."),
)

SITES = {"soundcloud": ("SoundCloud", "https://soundcloud.com"),
         "youtube": ("YouTube", "https://www.youtube.com")}


def classify_text(text: str, ver: str = "") -> Tuple[str, str]:
    """(kind, message) for raw error text; ("unknown", the first line) when no
    class matches."""
    joined = (text or "").lower()
    for pattern, kind, template in ERROR_CLASSES:
        if re.search(pattern, joined):
            return kind, template.format(ver=ver or "?")
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return "unknown", first[:400] or "Something went wrong — see the server log."


def network_message(site: str = "SoundCloud") -> str:
    return f"Can't reach {site} — check your internet connection, then Retry."


def describe(exc: BaseException, site: str = "SoundCloud") -> Dict[str, str]:
    """{kind, message} for an exception from an HTTP fetch. HTTP status codes
    first (they are exact), then the text classes; a network failure always
    names the site and the one fix."""
    code = getattr(exc, "code", None)
    text = str(exc)
    m = re.search(r"\bHTTP (\d{3})\b", text)
    if code is None and m:
        code = int(m.group(1))
    if code == 429:
        return {"kind": "rate_limited",
                "message": f"{site} is rate-limiting requests — wait a minute, then Retry."}
    if code in (401, 403):
        return {"kind": "blocked",
                "message": f"{site} refused the request (HTTP {code}) — it may be private "
                           "or need a login, or yt-dlp may need updating."}
    if code == 404:
        return {"kind": "removed",
                "message": f"{site} answered HTTP 404 — the link or track is gone; if every "
                           "link does this, update yt-dlp (Library offers it)."}
    if code is not None and 500 <= int(code) < 600:
        return {"kind": "upstream",
                "message": f"{site} is having trouble (HTTP {code}) — try again shortly."}
    kind, msg = classify_text(text)
    if kind == "network":
        return {"kind": "network", "message": network_message(site)}
    if kind == "unknown":
        why = f" (HTTP {code})" if code is not None else ""
        return {"kind": "unknown", "message": f"{site} request failed{why} — try again, "
                                              "or check the server log."}
    return {"kind": kind, "message": msg}


_REACH: Dict[str, Tuple[float, bool]] = {}


def reachable(source: str = "soundcloud", timeout: float = 4.0,
              ttl: float = 30.0) -> bool:
    """Whether the site answers at all — any HTTP status counts — in at most
    `timeout` seconds; remembered for `ttl`. urllib honours the proxy
    environment, as yt-dlp does, so a proxied machine is not called offline."""
    import os
    if os.environ.get("MASHUP_REACH_CHECK", "1").strip().lower() in ("0", "false", "no"):
        return True       # the test suite sets this: a test never touches the network
    _, url = SITES.get(source, SITES["soundcloud"])
    now = time.monotonic()
    hit = _REACH.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    ok = True
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0 mashup-engine"})
        urllib.request.urlopen(req, timeout=timeout).close()
    except urllib.error.HTTPError:
        ok = True          # it answered
    except Exception:  # noqa: BLE001 — unreachable, refused, timed out, proxy said no
        ok = False
    _REACH[url] = (now, ok)
    return ok


def site_name(source: Optional[str]) -> str:
    return SITES.get(source or "soundcloud", SITES["soundcloud"])[0]
