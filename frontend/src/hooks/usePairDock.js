import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { onNextForSet } from "../dockModes";
import { keyOf, SCORE_TERMS } from "../components/pairs/pairModel";
import { TIERS } from "../theme";

// The pair dock: what it shows, where the cursor is, and what the keyboard does.
//
// The dock is the reason the tabs became a rail. Judging a pair is the
// expensive part of this app, and it used to cost a tab switch plus finding
// your place again. Here the list is always beside the library, and a whole
// judgement is one keypress.
//
// Two kinds of narrowing, kept apart on purpose:
// - LIBRARY-WIDE (`filters`, `order`, search): sent to GET /api/mashups, so
//   they rank and narrow the whole library, not the page. Changing one
//   refetches.
// - LOADED PAIRS (`df`, tier chips): applied to the rows already fetched and
//   NEVER refetch — max stretch, transposed beds, same genre, per-side vocal /
//   bed genre, BPM and key. Fast to flick through, and honest about it ("n of
//   m loaded").
//
// ORDERS. Everything except "effort" is the server's, so it ranks the library
// rather than the page: uncertain surfaces the model's blind spots, where a
// verdict buys the most information per keypress, and the four section terms
// ask "which pairs agree on phrasing?" of the whole table.
const AUTO_ADVANCE_KEY = "mashup.dockAutoAdvance.v1";

export const ORDERS = [
  ["score", "Score", "Best fit first."],
  ["effort", "Effort", "Cheapest to build first, within the rows already loaded."],
  ["uncertain", "Uncertain",
   "Where the learned scorer is least sure — a verdict here teaches it the most."],
  ...SCORE_TERMS.map((t) => [t.order, t.name, `Best ${t.what}`, t.color]),
];

// The one order the server does not do. Everything else is sent as-is.
const CLIENT_ORDER = "effort";

const LIMIT = 40;
// One song may not own the dock. Three rows is enough to show that a pair works
// at more than one moment without a single strong vocal taking the page.
const MAX_PER_SONG = 3;

// The library-wide filters. Every one of them runs in SQL
// (get_candidates_enriched): narrowing a page the server already truncated
// would search the top 40, not the library. `perVocal` swaps the list for
// "best bed for each of my vocals" (GET /api/mashups/by-vocal).
export const DOCK_FILTERS = {
  search: "", minScore: 0, maxEffort: null, genre: "", era: "", energy: "",
  bpmBand: "", vocalForward: false, adventure: 0, perVocal: false,
  // Keepers ("rated" / "loved") and the opposite ("unrated"); the key the pair
  // LANDS in (the vocal's) within keyTol Camelot steps; one section type per
  // side. All four in SQL too.
  rated: "", key: "", keyTol: 1, vocalLabel: "", instLabel: "",
};

// The loaded-pairs filters (client side, never a request).
const EMPTY_SIDE = { genres: [], bpm: null, keys: [] };
export const LOADED_FILTERS = {
  maxDelta: null, allowShift: true, sameGenre: false, showHidden: false,
  v: EMPTY_SIDE, b: EMPTY_SIDE,
};

// How many library-wide filters differ from the defaults (search is shown on
// its own).
export function activeFilterCount(f) {
  return Object.keys(DOCK_FILTERS)
    .filter((k) => k !== "search" && k !== "keyTol" && f[k] !== DOCK_FILTERS[k]).length;
}
export const sideCount = (s) => (s.genres.length ? 1 : 0) + (s.bpm ? 1 : 0) + (s.keys.length ? 1 : 0);
export function loadedFilterCount(df) {
  return (df.maxDelta != null ? 1 : 0) + (!df.allowShift ? 1 : 0) + (df.sameGenre ? 1 : 0)
    + (df.showHidden ? 1 : 0) + sideCount(df.v) + sideCount(df.b);
}

// The list route's query options for a filter set — shared by the fetch and
// "load more", so page two is the same query as page one.
function filterOpts(f) {
  return {
    search: f.search.trim(), minScore: f.minScore, maxEffort: f.maxEffort,
    genre: f.genre, era: f.era, energy: f.energy, bpmBand: f.bpmBand,
    vocalForward: f.vocalForward, adventure: f.adventure,
    rated: f.rated, key: f.key, keyTol: f.keyTol,
    vocalLabel: f.vocalLabel, instLabel: f.instLabel,
  };
}

// One side of a pair against one side's loaded filters. The pair row carries
// genre (compat.genre), tempo and key per side.
function sidePasses(c, side, s) {
  const genre = side === "v" ? c.compat?.genre?.vocal ?? c.vocal_genre : c.compat?.genre?.bed ?? c.inst_genre;
  const bpm = side === "v" ? c.vocal_bpm : c.inst_bpm;
  const key = side === "v" ? c.vocal_camelot : c.inst_camelot;
  if (s.genres.length && !s.genres.includes(genre)) return false;
  if (s.bpm && (bpm == null || bpm < s.bpm[0] || bpm > s.bpm[1])) return false;
  if (s.keys.length && !s.keys.includes(key)) return false;
  return true;
}

export function applyLoadedFilters(rows, df, hiddenKeys) {
  return rows.filter((c) => {
    if (!df.showHidden && hiddenKeys.has(keyOf(c))) return false;
    const d = c.compat?.bpm?.delta_pct;
    if (df.maxDelta != null && (d == null || Math.abs(d) > df.maxDelta)) return false;
    if (!df.allowShift && c.semitone_shift) return false;
    if (df.sameGenre && c.compat?.genre?.same !== true) return false;
    return sidePasses(c, "v", df.v) && sidePasses(c, "b", df.b);
  });
}

const TIER_INDEX = Object.fromEntries(TIERS.map((t, i) => [t.id, i]));

// `player` is the app-wide one from usePlayer. The dock borrows it, so the bar
// at the bottom is showing the same pair the dock's cursor is on.
export function usePairDock({ selectedTrackId, role = "vocal", ratings,
                              onOpenStudio, player, onAddToSet = null,
                              onCompare = null }) {
  const [order, setOrder] = useState("score");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cursor, setCursor] = useState(0);
  const [expanded, setExpanded] = useState(null);       // a pair key, or null
  const [filters, setFiltersState] = useState(DOCK_FILTERS);
  const [df, setDfState] = useState(LOADED_FILTERS);
  const [tierOn, setTierOn] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  // Pairs hidden this session stay loaded, so "Show hidden" can bring them
  // back and Undo can put them back; the server already leaves them out.
  const [hiddenKeys, setHiddenKeys] = useState(() => new Set());
  // Re-fetch the same list — after something it shows changed elsewhere (a
  // section line typed on Track detail).
  const [nonce, setNonce] = useState(0);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  // "What comes next" for a set: the list becomes that set's next candidates
  // until it is closed (or the scope changes). { id, name } or null.
  const [nextFor, setNextFor] = useState(null);
  useEffect(() => onNextForSet((id, name) => setNextFor({ id, name })), []);
  useEffect(() => { setNextFor(null); }, [selectedTrackId, role]);
  const clearNextFor = useCallback(() => setNextFor(null), []);
  const [loadingMore, setLoadingMore] = useState(false);
  const setFilters = useCallback(
    (patch) => setFiltersState((f) => ({ ...f, ...patch })), []);
  const resetFilters = useCallback(() => setFiltersState(DOCK_FILTERS), []);
  const setDf = useCallback((patch) => { setDfState((f) => ({ ...f, ...patch })); setCursor(0); }, []);
  const setSide = useCallback((side, patch) => {
    setDfState((f) => ({ ...f, [side]: { ...f[side], ...patch } }));
    setCursor(0);
  }, []);
  const resetLoaded = useCallback(() => { setDfState(LOADED_FILTERS); setTierOn([]); }, []);
  const toggleTier = useCallback((id) => {
    setTierOn((t) => (t.includes(id) ? t.filter((x) => x !== id) : [...t, id]));
    setCursor(0);
  }, []);
  // "Best bed per vocal" is a library-wide view; scoped to one track it would
  // be that track's single best row, which the normal list already leads with.
  const perVocal = filters.perVocal && selectedTrackId == null;

  // Which pair the transport is actually on, keyed by the pair's four ids. It
  // is DERIVED from the player rather than tracked here: a local copy would go
  // stale the moment anything else took the audio away.
  const armedKey = player.kind === "pair" ? player.source.key : null;
  const audio = player.pair;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const opts = { limit: LIMIT, maxPerSong: MAX_PER_SONG, ...filterOpts(filters) };
    if (selectedTrackId != null) {
      // Scoping by role, because a track can be either side of a pair.
      if (role === "instrumental") opts.instSongId = selectedTrackId;
      else opts.vocalSongId = selectedTrackId;
    }
    if (order !== CLIENT_ORDER) opts.order = order;
    const request = nextFor
      ? api.getSetNext(nextFor.id, 20)
      : perVocal
        ? api.getBestBedPerVocal({ limit: LIMIT, minScore: filters.minScore })
        : api.getMashups(opts);
    request
      .then((d) => {
        if (cancelled) return;
        const got = d.candidates || [];
        setRows(got);
        setHasMore(!perVocal && !nextFor && got.length >= LIMIT);
        setCursor(0);
        setExpanded(null);
      })
      .catch((e) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [selectedTrackId, role, order, filters, perVocal, nonce, nextFor]);

  const rowsRef = useRef(rows);
  rowsRef.current = rows;

  // The next page of the same server-side list. The offset is applied after
  // the per-song cap, so page two continues exactly where page one stopped.
  const loadMore = useCallback(async () => {
    if (loadingMore || perVocal) return;
    setLoadingMore(true);
    const opts = { limit: LIMIT, maxPerSong: MAX_PER_SONG, ...filterOpts(filters),
                   offset: rowsRef.current.length };
    if (selectedTrackId != null) {
      if (role === "instrumental") opts.instSongId = selectedTrackId;
      else opts.vocalSongId = selectedTrackId;
    }
    if (order !== CLIENT_ORDER) opts.order = order;
    try {
      const d = await api.getMashups(opts);
      const got = d.candidates || [];
      setRows((rs) => {
        const seen = new Set(rs.map(keyOf));
        return [...rs, ...got.filter((r) => !seen.has(keyOf(r)))];
      });
      setHasMore(got.length >= LIMIT);
    } catch (e) {
      toast(`Could not load more pairs: ${e.message}`);
    } finally {
      setLoadingMore(false);
    }
  }, [loadingMore, perVocal, filters, selectedTrackId, role, order]);

  // Export the top N of what the dock is showing as FL session folders, one
  // zip. The filters go to the server rather than a list of rows, so the
  // diversity cap is applied the same way the list applied it.
  const exportBatch = useCallback((topN) => api.startSessionBatch({
    top_n: topN, min_score: filters.minScore, max_per_song: MAX_PER_SONG,
    max_effort: filters.maxEffort, genre: filters.genre, era: filters.era,
    energy: filters.energy, bpm_band: filters.bpmBand,
    vocal_forward: filters.vocalForward, search: filters.search.trim(),
  }), [filters]);

  // Rows after the client-side order and the loaded-pairs filters, with tier
  // counts taken BEFORE the tier chips narrow them (a chip says how many it
  // would show).
  const { filtered, tierCounts } = useMemo(() => {
    let out = rows;
    if (order === CLIENT_ORDER) {
      // Missing effort sorts last: an unscored row is not a cheap one.
      out = [...rows].sort((a, b) => {
        const x = a.score_effort, y = b.score_effort;
        if (x == null && y == null) return 0;
        if (x == null) return 1;
        if (y == null) return -1;
        return x - y;
      });
    }
    out = applyLoadedFilters(out, df, hiddenKeys);
    const counts = Object.fromEntries(TIERS.map((t) => [t.id, 0]));
    out.forEach((c) => { if (c.confidence_tier in counts) counts[c.confidence_tier] += 1; });
    return { filtered: out, tierCounts: counts };
  }, [rows, order, df, hiddenKeys]);

  // Under the Score order the cards are grouped under tier headers: the tier
  // is the API's confidence_tier, so a measured close-call can sit a tier below
  // its percentile — grouping keeps the order inside each tier.
  const grouped = order === "score" && !nextFor;
  const visible = useMemo(() => {
    let out = tierOn.length ? filtered.filter((c) => tierOn.includes(c.confidence_tier)) : filtered;
    if (grouped) {
      out = out.map((c, i) => [c, i]).sort((a, b) =>
        (TIER_INDEX[a[0].confidence_tier] ?? 9) - (TIER_INDEX[b[0].confidence_tier] ?? 9) || a[1] - b[1])
        .map(([c]) => c);
    }
    return out;
  }, [filtered, tierOn, grouped]);

  const visibleRef = useRef(visible);
  visibleRef.current = visible;
  const current = visible[Math.min(cursor, visible.length - 1)] || null;

  // Warm the next rows' clips so stepping down the list is instant rather than
  // a decode you can hear waiting for.
  useEffect(() => {
    if (visible.length) audio.prefetch(visible.slice(cursor, cursor + 3));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursor, visible]);

  const play = useCallback((candidate) => {
    if (!candidate) return;
    player.toggle({
      kind: "pair",
      key: keyOf(candidate),
      candidate,
      title: candidate.vocal_title,
      subtitle: `over ${candidate.inst_title}`,
    });
  }, [player]);

  // After a 1–5 verdict, step to the next pair and keep looping if a loop was
  // playing — judging a list is one keypress per pair. On unless switched off
  // (a per-browser choice, for when you want the WHY chips first).
  const [autoAdvance, setAutoAdvanceState] = useState(() => {
    try { return localStorage.getItem(AUTO_ADVANCE_KEY) !== "0"; } catch { return true; }
  });
  const setAutoAdvance = useCallback((on) => {
    setAutoAdvanceState(on);
    try { localStorage.setItem(AUTO_ADVANCE_KEY, on ? "1" : "0"); } catch { /* ignore */ }
  }, []);

  const move = useCallback((delta) => {
    setCursor((c) => Math.max(0, Math.min(visibleRef.current.length - 1, c + delta)));
  }, []);

  const toggleExpand = useCallback((candidate) => {
    if (!candidate) return;
    const k = keyOf(candidate);
    setExpanded((e) => (e === k ? null : k));
  }, []);

  // Rate, then (with auto-advance) move on — the loop follows the cursor if it
  // was playing. A click on a star never advances: you are looking at the card.
  const rateAndAdvance = useCallback((candidate, n) => {
    if (!candidate) return;
    ratings.rate(candidate, n);
    if (!autoAdvance) return;
    const list = visibleRef.current;
    const at = list.findIndex((c) => keyOf(c) === keyOf(candidate));
    if (at < 0 || at >= list.length - 1) return;
    const wasLooping = armedKey === keyOf(candidate) && audio.playing;
    setCursor(at + 1);
    if (expanded === keyOf(candidate)) setExpanded(keyOf(list[at + 1]));
    if (wasLooping) play(list[at + 1]);
  }, [ratings, autoAdvance, armedKey, audio, play, expanded]);

  // Hiding is a display preference, not a verdict: it is suppressed by the SQL
  // and never becomes training data. The card goes at once, and the toast's
  // Undo (or "Show hidden") brings it back.
  const unhide = useCallback(async (candidate) => {
    const k = keyOf(candidate);
    setHiddenKeys((s) => { const n = new Set(s); n.delete(k); return n; });
    try { await api.unhidePair(candidate.vocal_song_id, candidate.inst_song_id); } catch { /* shown again anyway */ }
  }, []);
  const hide = useCallback(async (candidate) => {
    if (!candidate) return;
    const k = keyOf(candidate);
    setHiddenKeys((s) => new Set(s).add(k));
    if (expanded === k) setExpanded(null);
    try {
      await api.hidePair(candidate.vocal_song_id, candidate.inst_song_id);
      toast(`Hidden: ${candidate.vocal_title} over ${candidate.inst_title}`,
        { label: "Undo", onClick: () => unhide(candidate) });
    } catch (e) {
      setHiddenKeys((s) => { const n = new Set(s); n.delete(k); return n; });
      toast(`Could not hide that pair: ${e.message}`);
    }
  }, [expanded, unhide]);

  const openStudio = useCallback((candidate) => {
    if (!candidate) return;
    player.stop();
    onOpenStudio(candidate);
  }, [player, onOpenStudio]);

  // The keyboard model, from the dock's own footer: ↑↓ move, space loop, 1–5
  // rate (and advance), ⏎ expand, V/B solo, h hide, a + set, c compare.
  // `enabled` stops it firing while another screen is on top. The Library
  // table stops its own keys, so these act when the table is not focused.
  const bindKeys = useCallback((enabled, { space = true } = {}) => {
    if (!enabled) return undefined;
    const onKey = (e) => {
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.tagName === "SELECT" || el.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
      const list = visibleRef.current;
      const row = list[Math.min(cursor, list.length - 1)] || null;

      if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); move(1); return; }
      if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); move(-1); return; }
      if (e.key === " ") { if (!space) return; e.preventDefault(); play(row); return; }
      if (e.key === "Enter") { e.preventDefault(); toggleExpand(row); return; }
      if ((e.key === "c" || e.key === "C") && row && onCompare) {
        e.preventDefault(); onCompare(row); return;
      }
      if ((e.key === "a" || e.key === "A") && row && onAddToSet) {
        e.preventDefault(); onAddToSet(row); return;
      }
      if (e.key === "v" || e.key === "V") {
        e.preventDefault();
        audio.setStemMode(audio.stemMode === "vox" ? "both" : "vox");
        return;
      }
      if (e.key === "b" || e.key === "B") {
        e.preventDefault();
        audio.setStemMode(audio.stemMode === "bed" ? "both" : "bed");
        return;
      }
      if (e.key === "h" || e.key === "H") { e.preventDefault(); hide(row); return; }
      if (e.key >= "1" && e.key <= "5" && row) {
        e.preventDefault();
        rateAndAdvance(row, Number(e.key));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, move, play, toggleExpand, hide, audio, onAddToSet, onCompare, rateAndAdvance]);

  return {
    order, setOrder, rows: visible, loaded: rows.length, filteredCount: filtered.length,
    loading, error,
    filters, setFilters, resetFilters, perVocal, hasMore, loadMore, loadingMore,
    exportBatch,
    df, setDf, setSide, resetLoaded, tierOn, toggleTier, tierCounts, grouped,
    hiddenCount: hiddenKeys.size,
    isHidden: (c) => hiddenKeys.has(keyOf(c)),
    cursor, setCursor, current, armedKey, expanded, setExpanded, toggleExpand,
    play, move, openStudio, hide, unhide, rateAndAdvance, bindKeys, audio,
    reload, autoAdvance, setAutoAdvance,
    nextFor, clearNextFor,
  };
}
