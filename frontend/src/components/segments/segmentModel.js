// The Segments screen's data model: unpacking GET /api/segments, and filtering
// and sorting the two tables as pure functions over rows already in memory.
//
// Like hooks/useLibraryFilters.js this module makes NO API call (a test greps
// it): narrowing a list must never spend a request. The search box reads the
// Library's tokens (parseQuery: 8A, 124-128, 126bpm, a genre, 2015+) plus a
// section label ("chorus", "drop") and a bar count ("16bars").
import { parseQuery } from "../../hooks/useLibraryFilters";

export const ROLES = ["vocal", "bed"];
export const STEM_OF = { vocal: "vocals", bed: "instrumental" };
export const OTHER = { vocal: "bed", bed: "vocal" };

// The labels analysis/structure.py writes, in the order a DJ reaches for them.
export const LABELS = ["chorus", "drop", "verse", "build", "breakdown", "bridge", "intro", "outro"];

/** GET /api/segments → one object per row: the song's fields (sent once per
 *  song) under the section's own. */
export function unpackSegments(body) {
  const cols = body?.columns || [];
  const songs = body?.songs || {};
  return (body?.rows || []).map((packed) => {
    const row = {};
    for (let i = 0; i < cols.length; i += 1) row[cols[i]] = packed[i];
    return { ...songs[row.song_id], ...row };
  });
}

export const EMPTY_SEG_FILTERS = {
  q: "",
  labels: [],          // section labels
  keys: [],            // Camelot codes (the full mix's key)
  bpm: null,           // [lo, hi], half/double time counted
  bars: null,          // [lo, hi]
  energyMin: null,     // 0..1, the section's energy rank on its side
  voxMin: null,        // 0..1, share of the section that is sung (vocal side)
  qualityMin: null,    // 0..1, the stem's separation quality
  genres: [],
  storedOnly: false,   // used by at least one stored pair
  usableOnly: false,   // no intros / outros
};

const BPM_FOLD = 0.06;

// The search box's tokens on top of the pinned filters.
export function effectiveSegFilters(f, genreNames = []) {
  const tokens = String(f.q || "").trim().split(/\s+/).filter(Boolean);
  const labels = [];
  let bars = null;
  const rest = [];
  for (const tk of tokens) {
    const lo = tk.toLowerCase();
    let m;
    if (LABELS.includes(lo)) { labels.push(lo); continue; }
    if ((m = /^(\d{1,3})bars?$/.exec(lo))) { bars = [+m[1], +m[1]]; continue; }
    rest.push(tk);
  }
  const { parsed, rest: text } = parseQuery(rest.join(" "), genreNames);
  const uniq = (a) => [...new Set(a)];
  return {
    ...f,
    text,
    labels: uniq([...(f.labels || []), ...labels]),
    keys: uniq([...(f.keys || []), ...(parsed.keys || [])]),
    genres: uniq([...(f.genres || []), ...(parsed.genres || [])]),
    bpm: parsed.bpm || f.bpm,
    bars: bars || f.bars,
  };
}

// What the search box read, for its preview line.
export function describeSegQuery(q, genreNames = []) {
  const e = effectiveSegFilters({ ...EMPTY_SEG_FILTERS, q }, genreNames);
  const out = [];
  if (e.labels.length) out.push(e.labels.join(" / "));
  if (e.keys.length) out.push(`Key ${e.keys.join(", ")}`);
  if (e.bpm) out.push(`${e.bpm[0]}–${e.bpm[1]} BPM`);
  if (e.bars) out.push(`${e.bars[0]} bars`);
  e.genres.forEach((g) => out.push(g));
  return out;
}

function bpmIn(bpm, [lo, hi]) {
  if (bpm == null) return false;
  // ±6%, at half and double time too: what a DJ can actually play together.
  return [bpm, bpm * 2, bpm / 2].some((b) => b >= lo * (1 - BPM_FOLD) && b <= hi * (1 + BPM_FOLD));
}

const norm = (s) => String(s || "").toLowerCase();

/** The rows that pass `f`. An unmeasured value never passes a filter that
 *  asks about it — the Library's rule. */
export function applySegFilters(rows, f, genreNames = []) {
  const e = effectiveSegFilters(f, genreNames);
  const text = norm(e.text);
  return rows.filter((r) => {
    if (e.labels.length && !e.labels.includes(r.label || "verse")) return false;
    if (e.keys.length && !e.keys.includes(r.camelot)) return false;
    if (e.bpm && !bpmIn(r.bpm, e.bpm)) return false;
    if (e.bars) {
      if (r.bar_count == null) return false;
      const b = Math.round(r.bar_count);
      if (b < e.bars[0] || b > e.bars[1]) return false;
    }
    if (e.energyMin != null && !(r.energy_rank != null && r.energy_rank >= e.energyMin)) return false;
    if (e.voxMin != null && !(r.vocal_activity != null && r.vocal_activity >= e.voxMin)) return false;
    if (e.qualityMin != null && !(r.stem_quality != null && r.stem_quality >= e.qualityMin)) return false;
    if (e.genres.length && !e.genres.includes(r.genre)) return false;
    if (e.storedOnly && !(r.uses > 0)) return false;
    if (e.usableOnly && !r.usable) return false;
    if (text) {
      const hay = `${norm(r.title)} ${norm(r.artist)} ${norm(r.genre)} ${norm(r.style)} ${norm(r.line)}`;
      if (!text.split(/\s+/).every((w) => hay.includes(w))) return false;
    }
    return true;
  });
}

export function isSegFiltered(f) {
  return Boolean(f.q || f.labels.length || f.keys.length || f.bpm || f.bars
    || f.energyMin != null || f.voxMin != null || f.qualityMin != null
    || f.genres.length || f.storedOnly || f.usableOnly);
}

// Sorts. `fit` exists only in match mode (the other side has a selection) and
// is the server's order; everything else sorts the loaded rows.
const LABEL_RANK = Object.fromEntries(LABELS.map((l, i) => [l, i]));
export const SEG_SORTS = {
  title: (r) => norm(r.title),
  label: (r) => LABEL_RANK[r.label || "verse"] ?? 99,
  bpm: (r) => r.bpm,
  key: (r) => r.camelot ? parseInt(r.camelot, 10) * 2 + (r.camelot.endsWith("B") ? 1 : 0) : null,
  bars: (r) => r.bar_count,
  energy: (r) => r.energy_rank,
  vox: (r) => r.vocal_activity,
  sung: (r) => r.sung_median,
  quality: (r) => r.stem_quality,
  uses: (r) => r.uses,
  genre: (r) => norm(r.genre) || null,
  fit: (r) => r._match?.score_total,
};

/** Sorted copy. Unmeasured values sort last in either direction. */
export function sortSegments(rows, sort) {
  const get = sort?.id && SEG_SORTS[sort.id];
  if (!get || !sort.dir) return rows;
  const dir = sort.dir === "asc" ? 1 : -1;
  return rows
    .map((r, i) => [r, get(r), i])
    .sort((a, b) => {
      const x = a[1], y = b[1];
      if (x == null && y == null) return a[2] - b[2];
      if (x == null) return 1;
      if (y == null) return -1;
      if (x < y) return -dir;
      if (x > y) return dir;
      return a[2] - b[2];
    })
    .map(([r]) => r);
}

// First click sorts the column's natural way (text ascending, numbers
// descending), the second reverses, the third goes back to the load order.
export function cycleSegSort(sort, id) {
  const first = id === "title" || id === "label" || id === "genre" || id === "key" ? "asc" : "desc";
  if (sort?.id !== id) return { id, dir: first };
  if (sort.dir === first) return { id, dir: first === "asc" ? "desc" : "asc" };
  return { id: null, dir: null };
}

// MIDI note number → "C4".
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export function midiName(m) {
  if (m == null || !Number.isFinite(m)) return null;
  const n = Math.round(m);
  return `${NOTE_NAMES[((n % 12) + 12) % 12]}${Math.floor(n / 12) - 1}`;
}

export const segKey = (r) => `${r.song_id}:${r.section_index}`;

/** What one table lists. Without `matchRows` (null): the side's sections.
 *  With them (the other side leads): only the sections the server matched, in
 *  the server's order, each carrying its match row as `_match` — then the
 *  table's own filters, then its sort, as everywhere else. Returns
 *  { rows, total }, total being what the filters chose from. */
export function buildSideRows(sideRows, matchRows, filters, sort, genreNames = []) {
  let base = sideRows;
  if (matchRows != null) {
    const bySeg = new Map(matchRows.map((m, i) => [m.segment_id, { ...m, _rank: i }]));
    base = sideRows.filter((r) => bySeg.has(r.segment_id))
      .map((r) => ({ ...r, _match: bySeg.get(r.segment_id) }))
      .sort((a, b) => a._match._rank - b._match._rank);
  }
  return { rows: sortSegments(applySegFilters(base, filters, genreNames), sort),
           total: base.length };
}

/** A section's span as "1:04–1:36". */
export function fmtSpan(a, b) {
  const t = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  return a == null || b == null ? "—" : `${t(a)}–${t(b)}`;
}
