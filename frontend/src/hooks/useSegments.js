// useSegments — the Segments screen's two lists and its one live question.
//
// Both lists are fetched once when the screen opens (GET /api/segments is
// cached server-side per library state, so a re-open costs ~20 ms) and filtered
// in memory. The match list is the one thing that fetches on interaction: a
// selection asks the server to score that section against the whole other
// side. Debounced so arrowing down a table sends one request, and aborted when
// a newer selection replaces it — a late answer for a row you have left must
// never repaint the other table.
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { unpackSegments } from "../components/segments/segmentModel";

export const MATCH_DEBOUNCE_MS = 150;

export function useSegments() {
  const [rows, setRows] = useState({ vocal: [], bed: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [v, b] = await Promise.all([api.getSegments("vocal"), api.getSegments("bed")]);
      setRows({ vocal: unpackSegments(v), bed: unpackSegments(b) });
    } catch (e) {
      setError(e?.message || "Could not load the segments");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);
  return { ...rows, loading, error, refresh: load };
}

/** Matches for `anchor` ({segment_id, role} or null), with `opts` passed
 *  through to the server (max stretch, max shift, tiers). */
export function useSegmentMatches(anchor, opts) {
  const [state, setState] = useState({ rows: null, loading: false, error: null, meta: null });
  const ctrl = useRef(null);
  const id = anchor?.segment_id ?? null;
  const role = anchor?.role ?? null;
  const optKey = JSON.stringify(opts || {});

  useEffect(() => {
    ctrl.current?.abort();
    if (id == null) {
      setState({ rows: null, loading: false, error: null, meta: null });
      return undefined;
    }
    setState((s) => ({ ...s, loading: true, error: null }));
    const c = new AbortController();
    ctrl.current = c;
    const t = setTimeout(async () => {
      try {
        const out = await api.getSegmentMatches(id, role, JSON.parse(optKey), c.signal);
        if (c.signal.aborted) return;
        setState({ rows: out.rows || [], loading: false, error: null,
                   meta: { scored: out.scored, gated: out.gated, anchor: out.anchor } });
      } catch (e) {
        if (c.signal.aborted || e?.name === "AbortError") return;
        setState({ rows: [], loading: false, error: e?.message || "Could not rank the matches",
                   meta: null });
      }
    }, MATCH_DEBOUNCE_MS);
    return () => { clearTimeout(t); c.abort(); };
  }, [id, role, optKey]);

  return state;
}
