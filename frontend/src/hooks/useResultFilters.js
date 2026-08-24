import { useCallback, useMemo, useState } from "react";

/**
 * Filtering and sorting for a page of Discovery results.
 *
 * Scope is deliberately "what is currently loaded", not all of SoundCloud.
 * Pages arrive 20–50 at a time behind a Load more button, and the browse layer
 * shares one scraped client_id with the frozen mixes auto-resolver — spending
 * that rate limit to make a sort look global is exactly the trade this codebase
 * refuses. So the bar says "showing 12 of 47 loaded" and means it, and Load more
 * appends into the active filter rather than clearing it.
 *
 * `applyFilters` is exported as a plain function so the logic is testable
 * without a DOM. No fetching happens anywhere in here.
 */

const EMPTY_LIST = [];

export const EMPTY_FILTERS = {
  // "" is unsorted, and that is the default on purpose: SoundCloud's own
  // relevance order is meaningful and must survive until the user asks
  // otherwise.
  sort: "",
  dir: "asc",
  genre: "",
  inLibrary: "any",
  inCrate: "any",
  hideSnip: false,
  minMin: "",
  maxMin: "",
  album: "any",
  verified: "any",
  country: "",
};

/** The listing is homogeneous in practice, so the rows themselves say what it is. */
export function kindOf(items) {
  const first = (items || [])[0];
  if (!first) return "track";
  if (first.kind === "playlist") return "playlist";
  if (first.kind === "user") return "user";
  return "track";
}

// Sort options per kind. `text` sorts compare with localeCompare; `num` sorts
// push missing/zero to the bottom in BOTH directions — see makeCmp.
export const SORTS = {
  track: {
    title: { label: "Title", text: (r) => r.title },
    artist: { label: "Artist", text: (r) => r.artist },
    plays: { label: "Plays", num: (r) => r.plays },
    likes: { label: "Likes", num: (r) => r.likes },
    reposts: { label: "Reposts", num: (r) => r.reposts },
    duration: { label: "Duration", num: (r) => r.duration_secs },
    // upload_date is yt-dlp's YYYYMMDD, which sorts correctly as a number.
    upload: { label: "Upload date", num: (r) => Number(r.upload_date) || 0 },
  },
  playlist: {
    title: { label: "Title", text: (r) => r.title },
    artist: { label: "Artist", text: (r) => r.artist },
    tracks: { label: "Tracks", num: (r) => r.track_count },
    duration: { label: "Duration", num: (r) => r.duration_secs },
  },
  user: {
    username: { label: "Name", text: (r) => r.username },
    followers: { label: "Followers", num: (r) => r.followers },
    tracks: { label: "Tracks", num: (r) => r.track_count },
  },
};

function makeCmp(spec, dir) {
  const sign = dir === "desc" ? -1 : 1;
  if (spec.text) {
    return (a, b) => sign * String(spec.text(a) || "").localeCompare(
      String(spec.text(b) || ""), undefined, { sensitivity: "base" });
  }
  return (a, b) => {
    const av = Number(spec.num(a)) || 0;
    const bv = Number(spec.num(b)) || 0;
    // Missing sorts last whichever way the arrow points. A row with no play
    // count is unknown, not unpopular, and must never outrank a real number.
    if (!av && !bv) return 0;
    if (!av) return 1;
    if (!bv) return -1;
    return sign * (av - bv);
  };
}

const minsToSecs = (v) => {
  const n = parseFloat(v);
  return Number.isFinite(n) && n > 0 ? n * 60 : null;
};

/** Distinct non-empty values of `field` across the loaded rows, sorted. */
function presentValues(items, field) {
  const seen = new Set();
  for (const r of items || []) {
    const v = (r?.[field] || "").trim();
    if (v) seen.add(v);
  }
  return [...seen].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

/** Genre is a dropdown, never free text — SoundCloud genres are unbounded user input. */
export const genresIn = (items) => presentValues(items, "genre");
export const countriesIn = (items) => presentValues(items, "country");

/**
 * The crates represented among the loaded rows, for the in-crate facet.
 * Derived from the membership map rather than fetched, so the dropdown offers
 * no crate that could not possibly match, and costs no request.
 */
export function cratesIn(items, crateOf) {
  if (!crateOf) return EMPTY_LIST;
  const byId = new Map();
  for (const r of items || []) {
    for (const c of crateOf(r) || []) byId.set(String(c.crate_id), c.name);
  }
  return [...byId.entries()]
    .map(([crate_id, name]) => ({ crate_id, name }))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
}

/** Pure: rows in, the rows to render out. */
export function applyFilters(items, filters, crateOf) {
  const rows = items || [];
  const f = { ...EMPTY_FILTERS, ...(filters || {}) };
  const kind = kindOf(rows);
  const membership = crateOf || (() => EMPTY_LIST);

  const kept = rows.filter((r) => {
    if (f.genre && kind !== "user" && (r.genre || "") !== f.genre) return false;

    if (kind === "track") {
      if (f.inLibrary === "yes" && !r.in_library) return false;
      if (f.inLibrary === "no" && r.in_library) return false;
      if (f.hideSnip && r.is_snip) return false;

      const crates = membership(r) || EMPTY_LIST;
      if (f.inCrate === "none" && crates.length) return false;
      if (f.inCrate && f.inCrate !== "any" && f.inCrate !== "none"
          && !crates.some((c) => String(c.crate_id) === String(f.inCrate))) return false;

      const secs = Number(r.duration_secs) || 0;
      const lo = minsToSecs(f.minMin);
      const hi = minsToSecs(f.maxMin);
      if (lo !== null && secs < lo) return false;
      if (hi !== null && secs > hi) return false;
    }

    if (kind === "playlist") {
      if (f.album === "album" && !r.is_album) return false;
      if (f.album === "playlist" && r.is_album) return false;
    }

    if (kind === "user") {
      if (f.verified === "yes" && !r.verified) return false;
      if (f.country && (r.country || "") !== f.country) return false;
    }

    return true;
  });

  const spec = SORTS[kind]?.[f.sort];
  // Sort a copy — the caller's array is SoundCloud's ordering and other code
  // still reads it.
  return spec ? [...kept].sort(makeCmp(spec, f.dir)) : kept;
}

/** True when anything is actually narrowing or reordering the list. */
export function isActive(filters) {
  const f = { ...EMPTY_FILTERS, ...(filters || {}) };
  return Object.keys(EMPTY_FILTERS).some(
    (k) => k !== "dir" && f[k] !== EMPTY_FILTERS[k]);
}

export function useResultFilters(items, crateOf) {
  const [filters, setFilters] = useState(EMPTY_FILTERS);

  // Filters clear when the listing changes, matching how selection already
  // clears in SoundCloudBrowser.run().
  const reset = useCallback(() => setFilters(EMPTY_FILTERS), []);

  const visible = useMemo(
    () => applyFilters(items, filters, crateOf), [items, filters, crateOf]);

  return {
    filters,
    setFilters,
    reset,
    visible,
    total: (items || []).length,
    active: isActive(filters),
  };
}
