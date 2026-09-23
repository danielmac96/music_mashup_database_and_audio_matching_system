import { useCallback, useMemo, useState } from "react";

// Library column widths: the defaults, plus whatever the user has dragged.
//
// Keyed by a stable column id rather than position. An index would silently
// re-point every stored width the first time a column is added or moved, and
// the symptom — one narrow column somewhere else — looks nothing like the
// cause. Unknown ids are ignored on load and missing ones fall back to the
// default, so a stored layout can never be corrupted by a later column change.
//
// A dragged width is a fixed px that overrides the default, including for the
// flexible columns. Whatever slack is left still goes to the `fr` columns that
// have not been dragged, which is what keeps the table filling its frame.

const KEY = "mashup.library.cols.v1";
const MAX_W = 600;

function load() {
  try {
    const raw = localStorage.getItem(KEY);
    const o = raw ? JSON.parse(raw) : null;
    if (!o || typeof o !== "object" || Array.isArray(o)) return {};
    // Only finite positive numbers. A hand-edited or half-written value must
    // not be able to collapse a column to nothing.
    const clean = {};
    for (const [k, v] of Object.entries(o)) {
      if (Number.isFinite(Number(v)) && Number(v) > 0) clean[k] = Number(v);
    }
    return clean;
  } catch {
    return {};
  }
}

function save(widths) {
  try {
    localStorage.setItem(KEY, JSON.stringify(widths));
  } catch {
    // A private window with storage blocked still gets a working table; it
    // just forgets the drag on reload.
  }
}

/** cols: [{ id, w, min }] — `w` is the default track (any grid length). */
export function useColumnWidths(cols) {
  const [widths, setWidths] = useState(load);

  const template = useMemo(
    () => cols.map((c) => (widths[c.id] != null ? `${widths[c.id]}px` : c.w)).join(" "),
    [cols, widths],
  );

  const setWidth = useCallback((id, px) => {
    const col = cols.find((c) => c.id === id);
    const min = col?.min ?? 32;
    const next = Math.round(Math.max(min, Math.min(MAX_W, px)));
    setWidths((w) => {
      const out = { ...w, [id]: next };
      save(out);
      return out;
    });
  }, [cols]);

  const resetColumn = useCallback((id) => {
    setWidths((w) => {
      if (w[id] == null) return w;
      const out = { ...w };
      delete out[id];
      save(out);
      return out;
    });
  }, []);

  return { template, setWidth, resetColumn };
}
