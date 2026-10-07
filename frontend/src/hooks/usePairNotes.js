import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { keyOf, feedbackKey } from "../components/pairs/pairModel";

// Notes on pairs ("opener", "needs a riser", "use the 2nd chorus"), fetched once
// in App like the ratings, so the dock, Studio and the Sets screen show the
// same note. Keyed by the pair's four ids (readme §7); never training data.
export function usePairNotes() {
  const [byPair, setByPair] = useState({});
  const load = useCallback(async () => {
    try {
      const d = await api.getPairNotes();
      const m = {};
      for (const n of d.notes || []) m[feedbackKey(n)] = n.note;
      setByPair(m);
    } catch { /* notes are a nicety; the screen works without them */ }
  }, []);
  useEffect(() => { load(); }, [load]);

  const noteOf = useCallback((c) => (c ? byPair[keyOf(c)] || "" : ""), [byPair]);
  const save = useCallback(async (c, note) => {
    const k = keyOf(c);
    const prev = byPair[k] || "";
    setByPair((m) => ({ ...m, [k]: note }));
    try {
      await api.savePairNote(c, note);
    } catch (e) {
      setByPair((m) => ({ ...m, [k]: prev }));
      toast(`Could not save the note: ${e.message}`);
    }
  }, [byPair]);
  return { noteOf, save, refresh: load };
}
