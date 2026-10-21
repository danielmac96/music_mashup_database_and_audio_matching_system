import { useMemo, useState } from "react";
import {
  compatibleKeys, isReadyToMash, isRecentlyAdded, needsAttention, parseCamelot,
} from "../theme";

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
  q: "",               // free text; tokens that read as filters are previewed
  bpm: null,           // [lo, hi]
  fold: false,         // ±6%, half and double time included
  keys: [],            // Camelot codes, e.g. "8A"
  keyCompat: false,    // + the selected track's key, ±1 and its relative
  genres: [],          // SoundCloud genre strings (top of the tree)
  styles: [],          // Discogs styles (the tree's children; top 3 count)
  year: null,          // [lo, hi]
  plays: null,         // a floor, e.g. 100000
  vocalForward: false, // sung share ≥ VOCAL_FORWARD_MIN
  hasStems: false,     // both stems separated
  status: [],          // STATUS keys
  // Analysed attributes (analysis/attributes.py), keyed by attribute id:
  // { min } for a number ("≥"), { value } for a category or a top-N list.
  attrs: {},
  view: "",            // a built-in shelf: ready | recent | attention
  group: "",           // a crate id — the library, narrowed to one shelf
};

// Tracks whose sung share reaches this are "vocal-forward".
export const VOCAL_FORWARD_MIN = 0.45;
// The Library's BPM slider spans this range.
export const BPM_LO = 60;
export const BPM_HI = 180;
const FOLD_TOL = 0.06;
export const YEAR_OPEN = 2100;   // "2015+" is stored as [2015, YEAR_OPEN]

// How deep into a top-N list (Discogs styles, moods/themes, instruments) a
// value may sit and still count: "Tech House" as a track's 5th style at 3%
// is not a tech-house track.
export const ATTR_TOP_DEPTH = 3;

const topLabels = (v) => (Array.isArray(v)
  ? v.slice(0, ATTR_TOP_DEPTH).map((x) => x?.label).filter(Boolean) : []);

// Whether one track passes one attribute condition. An unmeasured value never
// passes: an attribute filter is a statement about that attribute, the same
// rule the BPM and year filters follow.
export function attrMatches(value, cond) {
  if (!cond) return true;
  if (cond.value != null && cond.value !== "") {
    if (value == null) return false;
    if (Array.isArray(value)) return topLabels(value).includes(cond.value);
    return String(value) === String(cond.value);
  }
  const lo = cond.min === "" || cond.min == null ? null : Number(cond.min);
  const hi = cond.max === "" || cond.max == null ? null : Number(cond.max);
  if (lo == null && hi == null) return true;
  const x = Number(value);
  if (value == null || !Number.isFinite(x)) return false;
  if (lo != null && x < lo) return false;
  if (hi != null && x > hi) return false;
  return true;
}

// The rail's built-in shelves (Saved views lists these above your presets).
export const VIEWS = [
  ["ready", "Ready to mash", "var(--ok)", isReadyToMash],
  ["recent", "Recently added", "var(--link)", (t) => isRecentlyAdded(t)],
  ["attention", "Needs attention", "var(--warn)", needsAttention],
];

export function countView(rows, id) {
  const view = VIEWS.find((v) => v[0] === id);
  return view ? rows.filter(view[3]).length : 0;
}

/* ── status ──────────────────────────────────────────────────────────────── */
// The Status pill and filter, from songs.status plus whether a stage is running
// now. `processing` is a stage in flight (or stems done, analysis pending).
export const STATUS = [
  ["ready", "Ready"], ["nostems", "No stems"], ["processing", "Analysing…"],
  ["queued", "Queued"], ["failed", "Failed"],
];
export const STATUS_PILL = {   // theme.statusMeta's input for each key
  ready: "analysed", nostems: "downloaded", processing: "stemmed",
  queued: "queued", failed: "error",
};
export function statusKeyOf(t, running = false) {
  const s = String(t?.status || "").toLowerCase();
  if (s.startsWith("error")) return "failed";
  if (running || s === "stemmed") return "processing";
  if (s === "analysed") return "ready";
  if (s === "downloaded") return "nostems";
  return "queued";
}

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

const feat = (t) => t?.features?.full || {};
const norm = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
const GENRE_ALIASES = { rnb: "r&b", ukg: "uk garage", dnb: "drum & bass", hiphop: "hip-hop" };

/* ── the parsing search ──────────────────────────────────────────────────── */
// "8A 124-128 house 2015+ vox" reads as filters: a Camelot key, a BPM range (or
// "126bpm"), a genre this library contains, a year or year range, "vox" for
// vocal-forward and "stems" for has-stems. What is left is free text. The
// preview shows what was read; ⏎ pins it (pinQuery).
export function parseQuery(q, genreNames = []) {
  const byNorm = new Map(genreNames.map((g) => [norm(g), g]));
  const p = {};
  const rest = [];
  const tokens = String(q || "").trim().split(/\s+/).filter(Boolean);
  for (let i = 0; i < tokens.length; i += 1) {
    const tk = tokens[i];
    const lo = tk.toLowerCase();
    let m;
    if ((m = /^(\d{1,2})([ab])(~?)$/i.exec(tk)) && +m[1] >= 1 && +m[1] <= 12) {
      const code = `${+m[1]}${m[2].toUpperCase()}`;
      // "8A~": the key and what mixes with it — ±1 hour and the relative
      // (theme.compatibleKeys). A DJ typing a key usually means "mixable with".
      const add = m[3] ? compatibleKeys(code) : [code];
      p.keys = [...(p.keys || []), ...add];
      if (m[3]) (p.keyNeighbours = p.keyNeighbours || []).push(code);
      continue;
    }
    if ((m = /^(\d{2,3})[-–](\d{2,3})$/.exec(tk)) && +m[1] >= 50 && +m[2] <= 200) {
      p.bpm = [Math.min(+m[1], +m[2]), Math.max(+m[1], +m[2])]; continue;
    }
    if ((m = /^(\d{2,3})bpm$/i.exec(tk))) { p.bpm = [+m[1] - 2, +m[1] + 2]; continue; }
    if ((m = /^((?:19|20)\d\d)\+$/.exec(tk))) { p.year = [+m[1], YEAR_OPEN]; continue; }
    if ((m = /^((?:19|20)\d\d)[-–]((?:19|20)\d\d)$/.exec(tk))) {
      p.year = [Math.min(+m[1], +m[2]), Math.max(+m[1], +m[2])]; continue;
    }
    if (["vox", "vocal", "vocals", "vocal-forward"].includes(lo)) { p.vocalForward = true; continue; }
    if (lo === "stems") { p.hasStems = true; continue; }
    // Two-word genres first ("tech house"), then one word, then an alias.
    const two = i + 1 < tokens.length ? byNorm.get(norm(`${tk} ${tokens[i + 1]}`)) : null;
    if (two) { (p.genres = p.genres || []).push(two); i += 1; continue; }
    const one = byNorm.get(norm(tk)) || byNorm.get(norm(GENRE_ALIASES[norm(tk)]));
    if (one) { (p.genres = p.genres || []).push(one); continue; }
    rest.push(tk);
  }
  return { parsed: p, rest: rest.join(" ") };
}

// The filters with the search box's parsed tokens applied on top — what the
// table shows while you type.
export function effectiveFilters(f, genreNames = []) {
  const { parsed, rest } = parseQuery(f.q, genreNames);
  const uniq = (a) => [...new Set(a)];
  return {
    ...f,
    text: rest,
    keys: uniq([...(f.keys || []), ...(parsed.keys || [])]),
    genres: uniq([...(f.genres || []), ...(parsed.genres || [])]),
    bpm: parsed.bpm || f.bpm,
    year: parsed.year || f.year,
    vocalForward: f.vocalForward || !!parsed.vocalForward,
    hasStems: f.hasStems || !!parsed.hasStems,
  };
}

// ⏎ in the search box: the parsed tokens become real filters, the rest stays.
export function pinQuery(f, genreNames = []) {
  const e = effectiveFilters(f, genreNames);
  const { text, ...rest } = e;
  return { ...rest, q: text };
}

// What the search box read, for the preview popover.
export function describeParsed(q, genreNames = []) {
  const { parsed } = parseQuery(q, genreNames);
  const out = [];
  if (parsed.keyNeighbours) {
    out.push(`Mixes with ${parsed.keyNeighbours.join(", ")} (${parsed.keys.join(", ")})`);
  } else if (parsed.keys) out.push(`Key ${parsed.keys.join(", ")}`);
  if (parsed.bpm) out.push(`${parsed.bpm[0]}–${parsed.bpm[1]} BPM`);
  (parsed.genres || []).forEach((g) => out.push(g));
  if (parsed.year) out.push(yearLabel(parsed.year));
  if (parsed.vocalForward) out.push("Vocal-forward");
  if (parsed.hasStems) out.push("Has stems");
  return out;
}

export const yearLabel = (y) => (y[1] >= YEAR_OPEN ? `${y[0]} or later` : `${y[0]}–${y[1]}`);

/* ── facets ──────────────────────────────────────────────────────────────── */
// Only offer values this library actually contains. SoundCloud genre strings
// are unbounded user input, so a hard-coded list is wrong the moment you import
// anything. Genres carry the Discogs styles their tracks lead with as children.
export function facetsOf(rows) {
  const genres = new Map();     // genre -> { n, styles: Map(style -> n) }
  const styleTotals = new Map();
  let yearLo = null, yearHi = null;
  const attrValues = {};
  const attrRange = {};
  for (const t of rows) {
    for (const [id, v] of Object.entries(t.attrs || {})) {
      if (v == null) continue;
      if (Array.isArray(v) || typeof v === "string") {
        const m = attrValues[id] || (attrValues[id] = new Map());
        for (const label of (Array.isArray(v) ? topLabels(v) : [v])) {
          m.set(label, (m.get(label) || 0) + 1);
        }
      } else if (Number.isFinite(Number(v))) {
        const r = attrRange[id] || (attrRange[id] = { lo: Number(v), hi: Number(v) });
        r.lo = Math.min(r.lo, Number(v));
        r.hi = Math.max(r.hi, Number(v));
      }
    }
    const g = (t.genre || "").trim();
    const style = t.attrs?.style;
    if (g) {
      const e = genres.get(g) || { n: 0, styles: new Map() };
      e.n += 1;
      if (style) e.styles.set(style, (e.styles.get(style) || 0) + 1);
      genres.set(g, e);
    }
    for (const s of topLabels(t.attrs?.styles)) styleTotals.set(s, (styleTotals.get(s) || 0) + 1);
    const y = t.release_year || 0;
    if (y) {
      if (yearLo == null || y < yearLo) yearLo = y;
      if (yearHi == null || y > yearHi) yearHi = y;
    }
  }
  const byCount = (a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]));
  return {
    genres: [...genres.entries()]
      .sort((a, b) => b[1].n - a[1].n || a[0].localeCompare(b[0]))
      .map(([name, e]) => ({
        name, n: e.n,
        styles: [...e.styles.entries()].sort(byCount)
          .map(([s]) => ({ name: s, n: styleTotals.get(s) || e.styles.get(s) })),
      })),
    yearLo, yearHi,
    attrValues: Object.fromEntries(Object.entries(attrValues).map(([id, m]) => [
      id, [...m.entries()].sort(byCount).map(([name, n]) => ({ name, n }))])),
    attrRange,
  };
}

// Energy is on the analyser's scale, which means nothing across differently
// mastered records — so the column shows each track's rank in this library,
// 0–1 (the "measure, then rank against this library" rule).
export function energyRanks(rows) {
  const vals = rows.map((t) => feat(t).energy).filter((v) => v != null && Number.isFinite(v))
    .sort((a, b) => a - b);
  const out = new Map();
  if (!vals.length) return out;
  for (const t of rows) {
    const v = feat(t).energy;
    if (v == null || !Number.isFinite(v)) continue;
    let lo = 0, hi = vals.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (vals[mid] <= v) lo = mid + 1; else hi = mid; }
    out.set(t.id, vals.length > 1 ? (lo - 1) / (vals.length - 1) : 1);
  }
  return out;
}

export function isActive(f) {
  return !!(f.q || f.bpm || f.keys?.length || f.keyCompat || f.genres?.length
    || f.styles?.length || f.year || f.plays || f.vocalForward || f.hasStems
    || f.status?.length || Object.keys(f.attrs || {}).length || f.view || f.group);
}

/* ── apply ───────────────────────────────────────────────────────────────── */
// ctx: { selectedKey, groups, running(id) }. `f` should already be the
// effective filters (effectiveFilters) — the search box's tokens included.
export function applyLibraryFilters(rows, f, ctx = {}) {
  const filters = { ...EMPTY_FILTERS, text: "", ...(f || {}) };
  const { selectedKey = null, groups = NO_GROUPS, running = () => false } = ctx;
  const needle = (filters.text ?? filters.q ?? "").trim().toLowerCase();
  const genres = new Set(filters.genres.map((g) => g.toLowerCase()));
  const styles = new Set(filters.styles);
  const keys = new Set(filters.keys);
  if (filters.keyCompat && selectedKey) compatibleKeys(selectedKey).forEach((k) => keys.add(k));
  const status = new Set(filters.status);
  const view = VIEWS.find((v) => v[0] === filters.view);
  const attrConds = Object.entries(filters.attrs || {});

  return rows.filter((t) => {
    if (view && !view[3](t)) return false;
    // A group is a set of song ids the app already has, so this is a lookup and
    // never a request — the same rule every other control on this bar follows.
    if (filters.group && !groups.has(filters.group, t.id)) return false;
    if (needle) {
      const hay = `${t.title || ""} ${t.artist || ""} ${t.genre || ""}`.toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    if (genres.size || styles.size) {
      // Either side of the tree passes: the genre, or one of its top styles.
      const inGenre = genres.size && genres.has((t.genre || "").toLowerCase());
      const inStyle = styles.size && (styles.has(t.attrs?.style)
        || topLabels(t.attrs?.styles).some((s) => styles.has(s)));
      if (!inGenre && !inStyle) return false;
    }

    const f0 = feat(t);
    if (filters.bpm) {
      // An unanalysed track has no tempo to compare. It is excluded rather than
      // kept, because a BPM filter is a statement about tempo and this row has
      // none — leaving it in would put un-mixable rows in every band.
      if (f0.bpm == null) return false;
      const [lo, hi] = filters.bpm;
      const inRange = (b) => (filters.fold
        ? b >= lo * (1 - FOLD_TOL) && b <= hi * (1 + FOLD_TOL)
        : b >= lo && b <= hi);
      if (!(inRange(f0.bpm) || (filters.fold && (inRange(f0.bpm * 2) || inRange(f0.bpm / 2))))) {
        return false;
      }
    }
    if (keys.size) {
      const k = parseCamelot(f0.camelot);
      if (!k || !keys.has(`${k.num}${k.letter}`)) return false;
    }
    if (filters.plays != null && (t.plays || 0) < filters.plays) return false;
    if (filters.year) {
      const y = t.release_year || 0;
      if (!y) return false;     // absent, not "before the range"
      if (y < filters.year[0] || y > filters.year[1]) return false;
    }
    if (filters.vocalForward) {
      const v = t.mash?.vocal_coverage;
      if (v == null || v < VOCAL_FORWARD_MIN) return false;
    }
    if (filters.hasStems && !(t.stems?.vocals && t.stems?.instrumental)) return false;
    if (status.size && !status.has(statusKeyOf(t, running(t.id)))) return false;
    for (const [id, cond] of attrConds) {
      if (!attrMatches(t.attrs?.[id], cond)) return false;
    }
    return true;
  });
}

// Group membership arrives as a pair of lookups over ids already in memory
// (hooks/useLibraryGroups.js). The default answers "no groups" so every function
// here stays callable with rows alone — a filter must never depend on a fetch
// having landed.
export const NO_GROUPS = { has: () => false, positionOf: () => null };

/* ── sort ────────────────────────────────────────────────────────────────── */
// One key per column id; `first` is the direction a first click picks. A header
// cycles first → the other → off: import order (off) is a real order.
const camelotOrder = (t) => { const k = parseCamelot(feat(t).camelot); return k ? k.num * 2 + (k.letter === "B" ? 1 : 0) : null; };
const STATUS_ORDER = Object.fromEntries(STATUS.map(([k], i) => [k, i]));
export const SORTS = {
  title: { get: (t) => t.title || "", text: true, first: "asc" },
  artist: { get: (t) => t.artist || "", text: true, first: "asc" },
  bpm: { get: (t) => feat(t).bpm, first: "asc" },
  key: { get: camelotOrder, first: "asc" },
  energy: { get: (t) => t._energy, first: "desc" },
  genre: { get: (t) => t.genre || "", text: true, first: "asc" },
  year: { get: (t) => t.release_year || null, first: "desc" },
  len: { get: (t) => t.duration_secs, first: "desc" },
  vox: { get: (t) => t.mash?.vocal_coverage ?? null, first: "desc" },
  best: { get: (t) => t.mash?.best_pct ?? null, first: "desc" },
  pairs: { get: (t) => (t.mash ? (t.mash.as_vocal || 0) + (t.mash.as_bed || 0) : null), first: "desc" },
  plays: { get: (t) => t.plays || null, first: "desc" },
  rating: { get: (t) => t.rating || null, first: "desc" },
  status: { get: (t) => STATUS_ORDER[t._status] ?? null, first: "asc" },
  added: { get: (t) => t.id, first: "desc" },
  sections: { get: (t) => t.section_count || null, first: "desc" },
};

// The next sort after a header click: first direction → the other → off.
export function cycleSort(sort, id) {
  const first = sortSpec(id)?.first || "asc";
  const other = first === "asc" ? "desc" : "asc";
  if (sort?.id !== id || !sort?.dir) return { id, dir: first };
  if (sort.dir === first) return { id, dir: other };
  return { id: null, dir: null };
}

function sortSpec(id) {
  if (SORTS[id]) return SORTS[id];
  // A Library column added from the Analysis panel. A top-N list sorts by its
  // first label.
  if (id && id.startsWith("attr:")) {
    const aid = id.slice(5);
    return {
      get: (t) => { const v = t.attrs?.[aid]; return Array.isArray(v) ? v[0]?.label ?? null : v ?? null; },
      first: "desc",
    };
  }
  return null;
}

function compare(key, dir) {
  const spec = sortSpec(key);
  if (!spec) return null;
  const sign = dir === "asc" ? 1 : -1;
  // Missing values sort LAST in both directions — a track with no play count
  // is unknown, not unpopular. The sign applies only after the null checks.
  return (a, b) => {
    let x = spec.get(a), y = spec.get(b);
    if (spec.text) { x = x || null; y = y || null; }
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    if (typeof x === "number" && typeof y === "number") return (x - y) * sign;
    return String(x).localeCompare(String(y)) * sign;
  };
}

// Unsorted is the default and a real choice: rows arrive in import order, which
// is the order you added them — or, inside a group, the group's running order.
export function sortLibrary(rows, sort, groups = NO_GROUPS, groupId = "") {
  const cmp = sort?.id && sort?.dir ? compare(sort.id, sort.dir) : null;
  if (!cmp) {
    if (!groupId) return rows;
    const pos = (t) => groups.positionOf(groupId, t.id);
    return [...rows].sort((a, b) => (pos(a) ?? 1e9) - (pos(b) ?? 1e9));
  }
  return [...rows].sort(cmp);
}

export function useLibraryFilters(rows, { sort, selectedKey = null, groups = NO_GROUPS,
                                          running = () => false } = {}) {
  const [filters, setFilters] = useState(EMPTY_FILTERS);

  const facets = useMemo(() => facetsOf(rows), [rows]);
  const genreNames = useMemo(() => facets.genres.map((g) => g.name), [facets]);
  const effective = useMemo(() => effectiveFilters(filters, genreNames),
    [filters, genreNames]);
  const visible = useMemo(
    () => sortLibrary(applyLibraryFilters(rows, effective, { selectedKey, groups, running }),
      sort, groups, filters.group),
    [rows, effective, sort, selectedKey, groups, running, filters.group],
  );

  return {
    filters, setFilters, effective, facets, genreNames, visible,
    total: rows.length,
    active: isActive(filters),
    reset: () => setFilters(EMPTY_FILTERS),
    patch: (p) => setFilters((f) => ({ ...f, ...p })),
    pin: () => setFilters((f) => pinQuery(f, genreNames)),
  };
}
