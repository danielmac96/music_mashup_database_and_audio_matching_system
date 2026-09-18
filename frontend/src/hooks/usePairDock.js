import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { keyOf, SCORE_TERMS } from "../components/pairs/pairModel";

// The pair dock: what it shows, where the cursor is, and what the keyboard does.
//
// The dock is the reason the tabs became a rail. Judging a pair is the
// expensive part of this app, and it used to cost a tab switch plus finding
// your place again. Here the list is always beside the library, and a whole
// judgement is one keypress.
//
// ORDERS. Everything except "effort" is the server's, so it ranks the library
// rather than the page: uncertain surfaces the model's blind spots, where a
// verdict buys the most information per keypress, and the four section terms
// ask "which pairs agree on phrasing?" of the whole table.
//
// "effort" alone sorts client-side over the page the server returned, and says
// so: re-asking the server for cheap-to-build pairs would be a different query
// (max_effort), not a re-ordering of this one.
export const ORDERS = [
  ["score", "Score", "Best fit first."],
  ["effort", "Effort", "Cheapest to build first, within the rows already loaded."],
  ["uncertain", "Uncertain",
   "Where the learned scorer is least sure — a verdict here teaches it the most."],
  ...SCORE_TERMS.map((t) => [t.order, t.label, `Best ${t.what}`, t.color]),
];

// The one order the server does not do. Everything else is sent as-is.
const CLIENT_ORDER = "effort";

const LIMIT = 40;
// One song may not own the dock. Three rows is enough to show that a pair works
// at more than one moment without a single strong vocal taking the page.
const MAX_PER_SONG = 3;

// `player` is the app-wide one from usePlayer. The dock used to build its own
// MashupEngine, which is how the app ended up with several players and one bar
// that belonged to none of them. It borrows the shared one now, so the bar at
// the bottom is showing the same pair the dock's cursor is on.
export function usePairDock({ selectedTrackId, role = "vocal", ratings,
                              onOpenStudio, player }) {
  const [order, setOrder] = useState("score");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cursor, setCursor] = useState(0);

  // Which pair the transport is actually on, keyed by the pair's four ids
  // rather than candidate.id — mashup_candidates is truncated on every
  // re-score, so an id does not survive one. It is DERIVED from the player
  // rather than tracked here: a local copy would go stale the moment anything
  // else (the bar's ✕, a library row, the Studio) took the audio away.
  const armedKey = player.kind === "pair" ? player.source.key : null;
  const audio = player.pair;
  const rowsRef = useRef(rows);
  rowsRef.current = rows;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const opts = { limit: LIMIT, maxPerSong: MAX_PER_SONG };
    if (selectedTrackId != null) {
      // Scoping by role, because a track can be either side of a pair. Picking
      // a row in the library asks "what beds fit this vocal"; the dock says so
      // in its scope line rather than leaving you to infer it.
      if (role === "instrumental") opts.instSongId = selectedTrackId;
      else opts.vocalSongId = selectedTrackId;
    }
    if (order !== CLIENT_ORDER) opts.order = order;
    api.getMashups(opts)
      .then((d) => {
        if (cancelled) return;
        setRows(d.candidates || []);
        setCursor(0);
      })
      .catch((e) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [selectedTrackId, role, order]);

  const visible = useMemo(() => {
    if (order !== CLIENT_ORDER) return rows;
    // Missing effort sorts last: an unscored row is not a cheap one.
    return [...rows].sort((a, b) => {
      const x = a.score_effort, y = b.score_effort;
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return x - y;
    });
  }, [rows, order]);

  const current = visible[cursor] || null;

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

  const move = useCallback((delta) => {
    setCursor((c) => {
      const next = Math.max(0, Math.min(rowsRef.current.length - 1, c + delta));
      return next;
    });
  }, []);

  // Hiding is a display preference, not a verdict: it is suppressed by the SQL
  // and never becomes training data. The row goes immediately rather than on a
  // refetch — waiting for the list to blink is what makes triage feel slow.
  const hide = useCallback(async (candidate) => {
    if (!candidate) return;
    const k = keyOf(candidate);
    const before = rowsRef.current;
    setRows((rs) => rs.filter((r) => keyOf(r) !== k));
    try {
      await api.hidePair(candidate.vocal_song_id, candidate.inst_song_id);
    } catch (e) {
      setRows(before);
      toast(`Could not hide that pair: ${e.message}`);
    }
  }, []);

  const openStudio = useCallback((candidate) => {
    if (!candidate) return;
    player.stop();
    onOpenStudio(candidate);
  }, [player, onOpenStudio]);

  // The keyboard model, from the dock's own footer: ↑↓ move, space loop,
  // 1-5 rate, V/B solo, h hide, ⏎ studio. `enabled` is what stops it firing while
  // another screen is on top — a window listener that ignores which screen is
  // showing is how two keyboard models end up fighting over the space bar.
  const bindKeys = useCallback((enabled) => {
    if (!enabled) return undefined;
    const onKey = (e) => {
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.tagName === "SELECT" || el.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const list = visible;
      const row = list[cursor] || null;

      if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); move(1); return; }
      if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); move(-1); return; }
      if (e.key === " ") { e.preventDefault(); play(row); return; }
      if (e.key === "Enter") { e.preventDefault(); openStudio(row); return; }
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
      if (e.key === "h" || e.key === "H") {
        e.preventDefault();
        hide(row);
        return;
      }
      if (e.key >= "1" && e.key <= "5" && row) {
        e.preventDefault();
        ratings.rate(row, Number(e.key));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, cursor, move, play, openStudio, hide, audio, ratings]);

  return {
    order, setOrder, rows: visible, loading, error,
    cursor, setCursor, current, armedKey,
    play, move, openStudio, hide, bindKeys, audio,
  };
}
