import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";

// Sets (chosen mashups in running order) live in App, like the library and the
// judgements: the dock's "+ Set", Studio's "+ Set" and the Sets screen all
// write to the same ACTIVE set, and three copies of "which set is that" would
// disagree the moment one of them created a new one.
const ACTIVE_KEY = "mashup.activeSet.v1";

function readActive() {
  try { return Number(localStorage.getItem(ACTIVE_KEY)) || null; } catch { return null; }
}

export function useSets() {
  const [sets, setSets] = useState([]);
  const [activeId, setActiveIdState] = useState(readActive);

  const refresh = useCallback(async () => {
    try {
      const d = await api.getSets();
      setSets(d.sets || []);
      return d.sets || [];
    } catch { return []; }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const setActiveId = useCallback((id) => {
    setActiveIdState(id);
    try { localStorage.setItem(ACTIVE_KEY, id == null ? "" : String(id)); } catch { /* ignore */ }
  }, []);

  // The active set, or — when it was deleted — the most recent one.
  const active = sets.find((s) => s.id === activeId) || sets[0] || null;

  const create = useCallback(async (name) => {
    const s = await api.createSet(name);
    setActiveId(s.id);
    await refresh();
    return s;
  }, [refresh, setActiveId]);

  // Add a pair to the active set, making a first set if there is none, so the
  // first press of "+ Set" never asks a question. Two quick presses before the
  // first set exists share ONE creation: `active` is still null for the second
  // until the refresh lands, which used to make "My set" and "My set 2".
  const creating = useRef(null);
  // Once a set is active the pending creation has done its job; clearing it
  // here (not after the await) also covers a press that lands before the
  // re-render that makes the new set `active`.
  useEffect(() => { if (active) creating.current = null; }, [active]);
  const addPair = useCallback(async (c) => {
    if (!c) return null;
    try {
      let target = active;
      if (!target) {
        creating.current = creating.current || create("My set");
        target = await creating.current;
      }
      const res = await api.addToSet(target.id, c);
      setActiveId(target.id);
      refresh();
      toast(res.added?.duplicate
        ? `Already in “${target.name}”`
        : `Added to “${target.name}” — ${res.items.length} mashup${res.items.length === 1 ? "" : "s"}`);
      return res;
    } catch (e) {
      toast(`Could not add to the set: ${e.message}`);
      return null;
    }
  }, [active, create, refresh, setActiveId]);

  return { sets, active, activeId: active?.id ?? null, setActiveId, refresh, create, addPair };
}
