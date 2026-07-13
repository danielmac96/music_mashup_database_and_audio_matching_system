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
