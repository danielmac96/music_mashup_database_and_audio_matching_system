import { useEffect, useRef } from "react";
import { api } from "../api";

// A mode's saved view (GET/PUT /api/prefs/mode-views): the Parts tables'
// filters and sorts, the dock's Tune weights. The Tracks view is
// useColumnWidths' (library-view). Read once per page load and shared; each
// mode writes only its own entry, debounced, and a failed write is dropped —
// a view is a convenience, never a record.
let loading = null;
function loadViews() {
  if (!loading) loading = api.getModeViews().catch(() => ({}));
  return loading;
}

const SAVE_MS = 600;

/** Restores `mode`'s saved view once (onRestore(view) when one is stored),
 *  then saves `value` whenever it changes — after the restore, never before,
 *  so a fresh mount cannot overwrite what was saved with its defaults. */
export function useModeView(mode, value, onRestore) {
  const ready = useRef(false);
  const restore = useRef(onRestore);
  restore.current = onRestore;
  useEffect(() => {
    let live = true;
    loadViews().then((views) => {
      if (!live) return;
      const v = views && views[mode];
      if (v) { try { restore.current(v); } catch { /* a stale shape is ignored */ } }
      ready.current = true;
    });
    return () => { live = false; };
  }, [mode]);
  const last = useRef(undefined);
  useEffect(() => {
    if (!ready.current) return undefined;
    const body = JSON.stringify(value ?? null);
    if (body === last.current) return undefined;
    const t = setTimeout(() => {
      last.current = body;
      loading = null;                    // the next mount reads what was saved
      api.putModeView(mode, value ?? null).catch(() => {});
    }, SAVE_MS);
    return () => clearTimeout(t);
  }, [mode, value]);
}
