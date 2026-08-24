import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";

/**
 * Crate chips for a page of Discovery rows.
 *
 * Fetched live rather than baked onto the rows, and that is the whole point of
 * the hook. Suggestion rows never pass through the server-side annotator — the
 * worker freezes them onto the job — so a badge written there would lie the
 * moment you shortlisted something. And even in the browser pane the result
 * list is not re-fetched after an add, so a baked-in badge would go stale
 * immediately. Re-running on `refreshKey`, which both panes already bump after
 * a successful add, is what makes the chip appear without a reload.
 *
 * Returns `(row) => crates[]`, empty for a row in no crate.
 */
export function useCrateMembership(items, refreshKey) {
  const [map, setMap] = useState(() => ({}));
  // Monotonic token, the same guard SoundCloudBrowser.run() uses: a slow
  // response for page 1 must not overwrite a fast one for page 2.
  const token = useRef(0);

  // Only tracks carry crate membership — a set or an artist row is a place to
  // go. `!i.kind` is the same predicate useRowSelection selects on.
  const trackRows = useMemo(() => (items || []).filter((i) => !i.kind), [items]);

  // Depend on the identity strings rather than the array, so a re-render that
  // produces an equal list does not re-fire the request.
  const urlKey = useMemo(
    () => trackRows.map((r) => r.source_url || "").join("\n"), [trackRows]);
  const idKey = useMemo(
    () => trackRows.map((r) => r.track_id || "").join("\n"), [trackRows]);

  useEffect(() => {
    const urls = urlKey.split("\n").filter(Boolean);
    const ids = idKey.split("\n").filter(Boolean);
    if (!urls.length && !ids.length) {
      setMap({});
      return;
    }
    const mine = ++token.current;
    api.crateMembership(urls, ids)
      .then((body) => {
        if (mine !== token.current) return;   // superseded
        setMap(body.membership || {});
      })
      // A chip is decoration. Failing quietly is correct here — it must never
      // raise a toast or blank the results behind it.
      .catch(() => { if (mine === token.current) setMap({}); });
  }, [urlKey, idKey, refreshKey]);

  return useMemo(
    () => (row) => map[row?.source_url] || map[row?.track_id] || EMPTY, [map]);
}

const EMPTY = [];
