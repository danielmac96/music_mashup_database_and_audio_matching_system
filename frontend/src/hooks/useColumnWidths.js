import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";

// The Library's view: which columns show, in what order, how wide, and the
// sort. Its home is the server (GET/PUT /api/prefs/library-view), so every
// browser sees the same table; a copy is cached in localStorage so the first
// paint is already right and a dropped request does not lose a drag.
//
// Keyed by a stable column id, never by position. An index would silently
// re-point every stored width the first time a column is added or moved, and
// the symptom — one narrow column somewhere else — looks nothing like the
// cause. Unknown ids are ignored and missing ones fall back to the default, so
// a stored layout can never be corrupted by a later column change.
//
// Attribute columns (`attr:<id>`) keep their width and place here, but WHETHER
// they show is the Analysis panel's attribute visibility (useAttributes), so
// the Columns menu and the Analysis switches cannot disagree.

const CACHE = "mashup.libview.v1";
const SAVE_DEBOUNCE_MS = 500;
const MIN_W = 32;
const MAX_W = 800;

function cleanWidth(v) {
  // Only finite positive numbers. A hand-edited or half-written value must
  // not be able to collapse a column to nothing.
  const n = Number(v);
  return v != null && Number.isFinite(n) && n > 0 ? Math.min(MAX_W, Math.max(MIN_W, n)) : null;
}

function cleanView(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const columns = {};
  for (const c of Array.isArray(raw.columns) ? raw.columns : []) {
    if (!c || typeof c.id !== "string") continue;
    columns[c.id] = { visible: c.visible !== false, width: cleanWidth(c.width) };
  }
  const order = Array.isArray(raw.order) ? raw.order.filter((x) => typeof x === "string") : [];
  const s = raw.sort || {};
  const sort = s.id && (s.dir === "asc" || s.dir === "desc") ? { id: s.id, dir: s.dir }
    : { id: null, dir: null };
  return { columns, order, sort };
}

function loadCache() {
  try {
    return cleanView(JSON.parse(localStorage.getItem(CACHE) || "null"));
  } catch {
    return null;
  }
}

function saveCache(view) {
  try {
    localStorage.setItem(CACHE, JSON.stringify(view));
  } catch {
    // A private window with storage blocked still gets a working table; the
    // server copy is the one that counts.
  }
}

const EMPTY = { columns: {}, order: [], sort: { id: null, dir: null } };

// The stored shape the API takes.
const toWire = (v) => ({
  columns: Object.entries(v.columns).map(([id, c]) => ({ id, visible: c.visible, width: c.width })),
  order: v.order, sort: v.sort,
});

/**
 * catalog: [{ id, w, min, defaultVisible, attr? }] — every column the table can
 * show, in its default order. attrVisible: ids of attribute columns shown.
 */
export function useLibraryView(catalog, attrVisible = []) {
  const [view, setView] = useState(() => loadCache() || EMPTY);
  const loaded = useRef(false);
  const timer = useRef(null);

  useEffect(() => {
    let live = true;
    api.getLibraryView()
      .then((v) => {
        const clean = cleanView(v);
        // An empty stored view means "defaults" — keep whatever the cache had.
        if (live && clean && (clean.order.length || Object.keys(clean.columns).length
          || clean.sort.id)) {
          setView(clean);
          saveCache(clean);
        }
      })
      .catch(() => {})
      .finally(() => { loaded.current = true; });
    return () => { live = false; clearTimeout(timer.current); };
  }, []);

  const update = useCallback((fn) => {
    setView((v) => {
      const next = fn(v);
      saveCache(next);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => {
        api.putLibraryView(toWire(next)).catch(() => {});
      }, SAVE_DEBOUNCE_MS);
      return next;
    });
  }, []);

  const attrSet = useMemo(() => new Set(attrVisible.map((id) => `attr:${id}`)), [attrVisible]);

  // The resolved columns, in order: catalogue defaults overlaid with the view.
  const columns = useMemo(() => {
    const byId = new Map(catalog.map((c) => [c.id, c]));
    const order = view.order.filter((id) => byId.has(id));
    const placed = new Set(order);
    // A column the stored order has never seen goes where its default puts it.
    catalog.forEach((c, i) => {
      if (placed.has(c.id)) return;
      const prev = catalog.slice(0, i).reverse().find((p) => placed.has(p.id));
      order.splice(prev ? order.indexOf(prev.id) + 1 : 0, 0, c.id);
      placed.add(c.id);
    });
    return order.map((id) => {
      const c = byId.get(id);
      const stored = view.columns[id];
      const visible = c.attr ? attrSet.has(id)
        : stored ? stored.visible : c.defaultVisible !== false;
      return { ...c, visible, width: stored?.width ?? c.w };
    });
  }, [catalog, view, attrSet]);

  const setWidth = useCallback((id, px) => {
    const col = catalog.find((c) => c.id === id);
    const w = Math.round(Math.max(col?.min ?? MIN_W, Math.min(MAX_W, px)));
    update((v) => ({ ...v, columns: { ...v.columns,
      [id]: { visible: v.columns[id]?.visible ?? col?.defaultVisible !== false, width: w } } }));
  }, [catalog, update]);

  const resetColumn = useCallback((id) => {
    update((v) => {
      if (!v.columns[id]) return v;
      return { ...v, columns: { ...v.columns, [id]: { ...v.columns[id], width: null } } };
    });
  }, [update]);

  const setVisible = useCallback((id, visible) => {
    update((v) => ({ ...v, columns: { ...v.columns,
      [id]: { width: v.columns[id]?.width ?? null, visible } } }));
  }, [update]);

  // Move `from` to just before `to` (a header dropped on another).
  const move = useCallback((from, to) => {
    if (!from || from === to) return;
    update((v) => {
      const order = columns.map((c) => c.id).filter((x) => x !== from);
      const at = order.indexOf(to);
      order.splice(at < 0 ? order.length : at, 0, from);
      return { ...v, order };
    });
  }, [columns, update]);

  const setSort = useCallback((sort) => update((v) => ({ ...v, sort })), [update]);
  const reset = useCallback(() => update(() => EMPTY), [update]);

  return { columns, sort: view.sort, setWidth, resetColumn, setVisible, move, setSort, reset,
           loaded: loaded.current };
}
