import { useMemo, useState } from "react";
import { isReadyToMash, isRecentlyAdded, needsAttention, parseCamelot } from "../theme";

// Filtering and sorting the library, as pure functions over rows already in
// memory.
//
// GET /api/tracks returns every song unpaginated, so there is nothing to fetch:
// this module makes NO API call, and a test greps it to keep it that way. It is
// the same rule ResultFilters follows for Discover, and for the same reason —
// a control that quietly re-queries turns "narrow the list" into "spend a
// request", and the browse layer shares its rate limit with the frozen mixes
// resolver.

export const EMPTY_FILTERS = {
  search: "",
  view: "",            // "" | ready | recent | attention — the rail's shelves
  group: "",           // a crate id — the library, narrowed to one shelf
  key: "",             // a Camelot key, e.g. "8A"
  keyTolerance: 1,     // +/- n steps around the wheel
  bpmMin: "", bpmMax: "",
  genres: [],          // multi-select; empty = any
  playsMin: "",        // a floor, e.g. 100000
  yearMin: "", yearMax: "",
  minStars: 0,
};

export const EMPTY_SORT = { primary: "", primaryDir: "desc",
                            secondary: "", secondaryDir: "desc" };

// Named bands, so the chips offer round numbers rather than a number pad.
export const PLAYS_BANDS = [
  ["", "Any"], ["1000", "1k+"], ["10000", "10k+"],
  ["100000", "100k+"], ["1000000", "1M+"],
];
export const BPM_BANDS = [
  ["", "", "Any"], ["90", "110", "90–110"], ["110", "125", "110–125"],
  ["125", "135", "125–135"], ["135", "150", "135–150"], ["150", "", "150+"],
];
export const KEY_TOLERANCES = [0, 1, 2];

export const VIEWS = [
  ["ready", "Ready to mash", "var(--green)", isReadyToMash],
  ["recent", "Recently added", "var(--cyan)", (t) => isRecentlyAdded(t)],
  ["attention", "Needs attention", "var(--amber)", needsAttention],
];

/* ── sort keys ───────────────────────────────────────────────────────────── */
// Deliberately no "best pair score": that would have to come from the ranked
// list, which is truncated, so sorting the whole library by it would order the
// rows it happened to fetch and silently mis-place the rest. Rating is the
// honest second key — it is the user's own, and complete.
export const SORT_KEYS = [
  ["", "unsorted"],
  ["added", "added"],
  ["title", "title"],
  ["artist", "artist"],
  ["genre", "genre"],
  ["bpm", "BPM"],
  ["key", "key"],
  ["year", "year"],
  ["plays", "plays"],
  ["duration", "length"],
  ["rating", "rating"],
  ["sections", "sections"],
  ["group", "group order"],
];

const feat = (t) => t?.features?.full || {};

// Group membership arrives as a pair of lookups over ids already in memory
// (hooks/useLibraryGroups.js). The default answers "no groups" so every function
// here stays callable with rows alone — a filter must never depend on a fetch
// having landed.
export const NO_GROUPS = { has: () => false, positionOf: () => null };

// Missing values sort LAST in both directions. A track with no play count is
// unknown, not unpopular — the same rule Discover's result filters use.
const NUMERIC = {
  added: (t) => t.id,
  bpm: (t) => feat(t).bpm,
  year: (t) => t.release_year || null,
  plays: (t) => t.plays || null,
  duration: (t) => t.duration_secs,
  sections: (t) => t.section_count || null,
};
const TEXT = {
  title: (t) => t.title || "",
  artist: (t) => t.artist || "",
  genre: (t) => t.genre || "",
  key: (t) => feat(t).camelot || "",
};

/* ── the Camelot wheel ───────────────────────────────────────────────────── */
// Distance in wheel steps, treating the relative major/minor as the same place.
// That is what makes "8A ±1" mean what a DJ means by it: 8A, 8B, 7A and 9A all
// mix, and none of them needs a pitch shift.
export function wheelDistance(a, b) {
  const x = parseCamelot(a), y = parseCamelot(b);
  if (!x || !y) return null;
  let d = Math.abs(x.num - y.num) % 12;
  if (d > 6) d = 12 - d;
  return d;
}

/* ── facets ──────────────────────────────────────────────────────────────── */
// Only offer values this library actually contains. SoundCloud genre strings
// are unbounded user input, so a hard-coded list is wrong the moment you import
// anything — the old Library toolbar shipped Pop / Hip Hop / Rap / EDM and
// silently hid everything else.
export function facetsOf(rows) {
  const genres = new Map();
  const keys = new Set();
  let yearLo = null, yearHi = null;
  for (const t of rows) {
    const g = (t.genre || "").trim();
    if (g) genres.set(g, (genres.get(g) || 0) + 1);
    const cam = feat(t).camelot;
    if (cam) keys.add(cam);
    const y = t.release_year || 0;
    if (y) {
      if (yearLo == null || y < yearLo) yearLo = y;
      if (yearHi == null || y > yearHi) yearHi = y;
    }
  }
  return {
    genres: [...genres.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([name, n]) => ({ name, n })),
    keys: [...keys].sort((a, b) => {
      const x = parseCamelot(a), y = parseCamelot(b);
      return (x?.num ?? 99) - (y?.num ?? 99) || a.localeCompare(b);
    }),
    yearLo, yearHi,
  };
}

export function isActive(f) {
  return !!(f.search || f.view || f.group || f.key || f.bpmMin || f.bpmMax
    || (f.genres && f.genres.length) || f.playsMin || f.yearMin
    || f.yearMax || f.minStars);
}

export function countView(rows, id) {
  const view = VIEWS.find((v) => v[0] === id);
  return view ? rows.filter(view[3]).length : 0;
}

/* ── apply ───────────────────────────────────────────────────────────────── */

export function applyLibraryFilters(rows, f, starOf = () => null,
                                    groups = NO_GROUPS) {
  const filters = { ...EMPTY_FILTERS, ...(f || {}) };
  const needle = filters.search.trim().toLowerCase();
  const genres = new Set(filters.genres.map((g) => g.toLowerCase()));
  const view = VIEWS.find((v) => v[0] === filters.view);
  const bpmMin = filters.bpmMin === "" ? null : Number(filters.bpmMin);
  const bpmMax = filters.bpmMax === "" ? null : Number(filters.bpmMax);
  const playsMin = filters.playsMin === "" ? null : Number(filters.playsMin);
  const yearMin = filters.yearMin === "" ? null : Number(filters.yearMin);
  const yearMax = filters.yearMax === "" ? null : Number(filters.yearMax);

  return rows.filter((t) => {
    if (view && !view[3](t)) return false;
    // A group is a set of song ids the app already has, so this is a lookup and
    // never a request — the same rule every other control on this bar follows.
    if (filters.group && !groups.has(filters.group, t.id)) return false;
    if (needle) {
      const hay = `${t.title || ""} ${t.artist || ""} ${t.genre || ""}`.toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    if (genres.size && !genres.has((t.genre || "").toLowerCase())) return false;

    const f0 = feat(t);
    if (bpmMin != null || bpmMax != null) {
      // An unanalysed track has no tempo to compare. It is excluded rather than
      // kept, because a BPM filter is a statement about tempo and this row has
      // none — leaving it in would put un-mixable rows in every band.
      if (f0.bpm == null) return false;
      if (bpmMin != null && f0.bpm < bpmMin) return false;
      if (bpmMax != null && f0.bpm > bpmMax) return false;
    }
    if (filters.key) {
      const d = wheelDistance(filters.key, f0.camelot);
      if (d == null || d > filters.keyTolerance) return false;
    }
    if (playsMin != null && (t.plays || 0) < playsMin) return false;
    if (yearMin != null || yearMax != null) {
      const y = t.release_year || 0;
      if (!y) return false;     // absent, not "before the range"
      if (yearMin != null && y < yearMin) return false;
      if (yearMax != null && y > yearMax) return false;
    }
    if (filters.minStars) {
      if ((starOf(t.id) || 0) < filters.minStars) return false;
    }
    return true;
  });
}

function compare(rows, key, dir, ctx) {
  const sign = dir === "asc" ? 1 : -1;
  const { starOf, groups, groupId } = ctx;
  // The order the tracks sit in inside the selected group — a saved playlist's
  // running order, which is most of why you saved it as one. With no group
  // selected every row is unplaced, so this sorts nothing rather than
  // inventing an order across shelves.
  if (key === "group") {
    return (a, b) => {
      const x = groupId ? groups.positionOf(groupId, a.id) : null;
      const y = groupId ? groups.positionOf(groupId, b.id) : null;
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x - y) * sign;
    };
  }
  if (key === "rating") {
    return (a, b) => {
      const x = starOf(a.id) || null, y = starOf(b.id) || null;
      if (x == null && y == null) return 0;
      if (x == null) return 1;          // unrated last, both directions
      if (y == null) return -1;
      return (x - y) * sign;
    };
  }
  if (NUMERIC[key]) {
    const get = NUMERIC[key];
    return (a, b) => {
      const x = get(a), y = get(b);
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x - y) * sign;
    };
  }
  if (TEXT[key]) {
    const get = TEXT[key];
    return (a, b) => {
      const x = get(a), y = get(b);
      if (!x && !y) return 0;
      if (!x) return 1;
      if (!y) return -1;
      return x.localeCompare(y) * sign;
    };
  }
  return null;
}

// Unsorted is the default and a real choice: rows arrive in import order, which
// is the order you added them, and that is often what you are looking for.
export function sortLibrary(rows, sort, starOf = () => null,
                            groups = NO_GROUPS, groupId = "") {
  const s = { ...EMPTY_SORT, ...(sort || {}) };
  const ctx = { starOf, groups, groupId };
  const first = compare(rows, s.primary, s.primaryDir, ctx);
  if (!first) return rows;
  const second = compare(rows, s.secondary, s.secondaryDir, ctx);
  const out = [...rows];
  out.sort((a, b) => first(a, b) || (second ? second(a, b) : 0));
  return out;
}

export function useLibraryFilters(rows, starOf, groups = NO_GROUPS) {
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [sort, setSort] = useState(EMPTY_SORT);

  const facets = useMemo(() => facetsOf(rows), [rows]);
  const visible = useMemo(
    () => sortLibrary(applyLibraryFilters(rows, filters, starOf, groups),
                      sort, starOf, groups, filters.group),
    [rows, filters, sort, starOf, groups],
  );

  return {
    filters, setFilters, sort, setSort, facets, visible,
    total: rows.length,
    active: isActive(filters),
    reset: () => setFilters(EMPTY_FILTERS),
    patch: (p) => setFilters((f) => ({ ...f, ...p })),
  };
}
