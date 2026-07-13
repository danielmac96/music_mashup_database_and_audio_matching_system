import { useEffect, useState } from "react";
import { api } from "../api";

// Fetch the engine's mashup plan (stretch/shift suggestion + section pairings)
// for a vocal/bed pair. Null until both sides are picked and distinct.
export function usePlan(vocalId, instId) {
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setPlan(null);
    setError(null);
    if (vocalId == null || instId == null || vocalId === instId) return;
    let cancelled = false;
    api.getMashupPlan(vocalId, instId)
      .then((p) => { if (!cancelled) setPlan(p); })
      .catch((e) => { if (!cancelled) setError(`Plan: ${e.message}`); });
    return () => { cancelled = true; };
  }, [vocalId, instId]);

  return { plan, error };
}
