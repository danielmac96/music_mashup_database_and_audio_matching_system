import { useCallback, useMemo, useState } from "react";
import { rowKey } from "../components/ScRows";

// Discover's shortlist: the tracks you have ticked, in every mode, until you
// import them, add them to a crate or drop them. It lives at Discover's scope
// (not inside one pane) so a find from Suggestions and one from a search sit
// in the same list — and it survives navigating from a search to an artist to
// a set, which is the whole digging path.
//
// Every shortlisted row is LISTED in the dock, so nothing is ever imported
// that you cannot see. Rows already in the library are never shortlisted:
// there is nothing left to import.
export function useShortlist() {
  const [byKey, setByKey] = useState(() => new Map());

  const rows = useMemo(() => [...byKey.values()], [byKey]);
  const importable = useMemo(() => rows.filter((r) => !r.in_library), [rows]);

  const has = useCallback((r) => byKey.has(rowKey(r)), [byKey]);

  const toggle = useCallback((r) => {
    if (!r || r.kind || r.in_library) return;
    setByKey((m) => {
      const next = new Map(m);
      const k = rowKey(r);
      if (next.has(k)) next.delete(k); else next.set(k, r);
      return next;
    });
  }, []);

  // "Shortlist all shown": adds every shown track not already in the library.
  const addMany = useCallback((list) => {
    setByKey((m) => {
      const next = new Map(m);
      for (const r of list || []) {
        if (r && !r.kind && !r.in_library) next.set(rowKey(r), r);
      }
      return next;
    });
  }, []);

  const remove = useCallback((r) => {
    setByKey((m) => { const next = new Map(m); next.delete(rowKey(r)); return next; });
  }, []);

  const removeMany = useCallback((list) => {
    setByKey((m) => {
      const next = new Map(m);
      for (const r of list || []) next.delete(rowKey(r));
      return next;
    });
  }, []);

  const clear = useCallback(() => setByKey(new Map()), []);

  return { rows, importable, has, toggle, addMany, remove, removeMany, clear, count: rows.length };
}
