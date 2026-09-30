import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";

// The attribute catalogue and which attributes are shown (readme §9, C).
// Fetched once in App.jsx; a toggle saves server-side, so every browser and
// every rebuild agrees on it.
export function useAttributes() {
  const [data, setData] = useState({ attributes: [], categories: [],
                                     visibility: { library: [], detail: [] } });
  const refresh = useCallback(() => {
    api.getAttributes().then(setData).catch(() => {});
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  const byId = useMemo(
    () => Object.fromEntries(data.attributes.map((a) => [a.id, a])), [data.attributes]);
  const toggle = useCallback((where, id) => {
    setData((d) => {
      const cur = d.visibility[where] || [];
      const next = { ...d.visibility,
        [where]: cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id] };
      api.setAttributeVisibility(next).catch(refresh);
      return { ...d, visibility: next };
    });
  }, [refresh]);
  return { catalogue: data.attributes, categories: data.categories, byId,
           visibility: data.visibility, toggle, refresh };
}
