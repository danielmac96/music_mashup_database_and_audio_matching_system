// sources.js — pure helpers for classifying pasted links and tidying titles.
// Mirrors ingest/sources.py (classify_url) — keep the two in sync.

export function classifyUrl(url) {
  const raw = (url || "").trim();
  if (!raw) return { source: "unknown", kind: "track" };
  let parsed;
  try {
    parsed = new URL(raw.includes("://") ? raw : `https://${raw}`);
  } catch {
    return { source: "unknown", kind: "track" };
  }
  const host = parsed.hostname.toLowerCase().replace(/^(www\.|m\.)/, "");
  const path = parsed.pathname || "";

  if (["soundcloud.com", "on.soundcloud.com", "api.soundcloud.com"].includes(host)) {
    return { source: "soundcloud", kind: path.includes("/sets/") ? "playlist" : "track" };
  }
  if (["youtube.com", "music.youtube.com", "youtu.be"].includes(host)) {
    const isList = parsed.searchParams.has("list") || path.startsWith("/playlist");
    return { source: "youtube", kind: isList ? "playlist" : "track" };
  }
  return { source: "unknown", kind: "track" };
}

// When a library track's audio is NOT the link it was imported from, say what it
// is: { kind, chip, label, title, url, confirmed }, or null when the audio is the
// link. Reads songs.audio_provenance (api/workers/stages._record_provenance) and
// mirrors the "overwritten SoundCloud link" test in database/models.py
// (SC_LINK_OVERWRITTEN_SQL) for tracks downloaded before provenance existed.
//
// `confirmed` is "✓ Sounds right" (POST /api/tracks/{id}/audio-confirm): you
// listened and it is the record. It is pinned to the link the audio came from,
// so a download from anywhere else starts unconfirmed again.
export function audioSubstitution(track) {
  const p = track?.audio_provenance;
  const confirmed = !!p?.confirmed && p.url === track?.source_url;
  const byYou = confirmed ? " · ✓ confirmed by you" : "";
  if (p?.via === "youtube_fallback") {
    const delta = p.duration_secs && p.expected_secs
      ? Math.round(p.duration_secs - p.expected_secs) : null;
    return {
      kind: "fallback",
      chip: "YT",
      label: `Audio: YouTube — ${p.title || "upload"}${p.uploader ? ` by ${p.uploader}` : ""}`
        + (delta != null ? ` (Δ ${delta >= 0 ? "+" : "−"}${Math.abs(delta)}s)` : "")
        + byYou,
      title: confirmed
        ? "A YouTube upload stood in for this track, and you confirmed it is the record."
        : "SoundCloud would not serve this track, so a YouTube upload that "
          + "passed the same-record check was downloaded instead.",
      url: p.url,
      confirmed,
    };
  }
  if (p?.via === "manual" && track.origin_url && p.url !== track.origin_url) {
    return {
      kind: "manual",
      chip: classifyUrl(p.url).source === "youtube" ? "YT" : "↗",
      label: `Audio: your pick — ${p.title || p.url}`,
      title: "You chose this upload. It is never flagged as suspect.",
      url: p.url,
      // A pick is already your decision; there is nothing left to confirm.
      confirmed: true,
    };
  }
  const scLooking = /^\d+$/.test(track?.track_id || "")
    || /sndcdn\.com/.test(track?.thumbnail || "");
  if (track && !track.origin_url && track.source === "youtube" && scLooking) {
    if (confirmed) {
      return {
        kind: "confirmed",
        chip: "YT",
        label: `Audio: YouTube${byYou}`,
        title: "Downloaded from YouTube before substitutes were checked — you "
          + "listened and confirmed it is the record.",
        url: track.source_url,
        confirmed,
      };
    }
    return {
      kind: "unverified",
      chip: "YT?",
      label: "Audio: YouTube — unverified",
      title: "Downloaded from YouTube before substitutes were checked, so it may "
        + "be a remix or a different cut. Listen, then ✓ Sounds right or Wrong "
        + "audio? — or re-download the suspect tracks from the library bar.",
      url: track.source_url,
      confirmed,
    };
  }
  return null;
}

// Junk suffixes YouTube uploaders bolt onto titles. Applied repeatedly so
// "Song (Official Video) [HQ]" fully unwraps.
const TITLE_JUNK = [
  /\s*[([【]\s*(official\s+)?(music\s+)?(video|audio|visuali[sz]er|lyric(s)?(\s+video)?|hd|hq|4k|remaster(ed)?( \d{4})?|explicit|clean|out now.*?|free (dl|download).*?)\s*[)\]】]\s*$/i,
  /\s*[|·-]\s*(official\s+(music\s+)?(video|audio)|lyrics?|monstercat( uncaged| instinct)? release|ncs release|premiere)\s*$/i,
];
const ARTIST_JUNK = [/vevo$/i, /\s*-\s*topic$/i, /official$/i];

// Tidy a noisy YouTube title/channel into {title, artist}. Non-destructive:
// falls back to the originals when cleaning would empty a field.
export function cleanYouTubeTitle(title, artist) {
  let t = (title || "").trim();
  let a = (artist || "").trim();

  for (let pass = 0; pass < 4; pass++) {
    const before = t;
    for (const re of TITLE_JUNK) t = t.replace(re, "").trim();
    if (t === before) break;
  }

  // "Artist - Title" in the video title beats the channel name as artist.
  const m = /^(.{1,80}?)\s+[-–—]\s+(.+)$/.exec(t);
  if (m) {
    a = m[1].trim() || a;
    t = m[2].trim();
  }

  for (const re of ARTIST_JUNK) a = a.replace(re, "").trim();

  return { title: t || title, artist: a || artist };
}
