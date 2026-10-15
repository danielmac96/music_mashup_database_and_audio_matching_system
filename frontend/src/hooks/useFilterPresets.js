import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";

// Named Library filter presets (GET/POST/DELETE /api/prefs/filter-presets).
// Fetched once in App.jsx: the Library's Saved ▾, the rail's Saved views and
// the ⌘K palette all list the same presets. `filters` is opaque to the server.
export function useFilterPresets() {
  const [presets, setPresets] = useState([]);

  const refresh = useCallback(() => {
    api.getFilterPresets().then((r) => setPresets(r.presets || [])).catch(() => {});
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const save = useCallback(async (name, filters) => {
    try {
      const p = await api.saveFilterPreset(name, filters);
      setPresets((ps) => [...ps, p]);
      toast(`Saved "${p.name}"`);
      return p;
    } catch (e) {
      toast(`Could not save the preset: ${e.message}`);
      return null;
    }
  }, []);

  // Deleting is undoable: the toast's Undo saves it back under the same name.
  const remove = useCallback(async (p) => {
    try {
      await api.deleteFilterPreset(p.id);
      setPresets((ps) => ps.filter((x) => x.id !== p.id));
      toast(`Deleted "${p.name}"`, {
        label: "Undo",
        onClick: () => { api.saveFilterPreset(p.name, p.filters).then(refresh).catch(() => {}); },
      });
    } catch (e) {
      toast(`Could not delete the preset: ${e.message}`);
    }
  }, [refresh]);

  return { presets, save, remove, refresh };
}
