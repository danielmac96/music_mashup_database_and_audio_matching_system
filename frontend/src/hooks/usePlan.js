import { useEffect, useState } from "react";
import { api } from "../api";

// Fetch the engine's mashup plan (stretch/shift suggestion + section pairings)
// for a vocal/bed pair. Null until both sides are picked and distinct.
//
// `pin` is the candidate row's own section pair and measured transpose
// ({ vocalSectionIdx, instSectionIdx, harmonicShift }). Pass it whenever the
// plan is being shown FOR a row: without it the server re-chooses the sections,
// so the Plan expander described a different moment than the row directly above
// it — and the FL export, which builds the same plan, rendered that other
// moment. Omit it where there is no row to be faithful to (Studio's ad-hoc
// pair), and the server picks as before.
export function usePlan(vocalId, instId, pin = null) {
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);

  const vSec = pin?.vocalSectionIdx ?? null;
  const iSec = pin?.instSectionIdx ?? null;
  const shift = pin?.harmonicShift ?? null;

  useEffect(() => {
    setPlan(null);
    setError(null);
    if (vocalId == null || instId == null || vocalId === instId) return;
    let cancelled = false;
    api.getMashupPlan(vocalId, instId, {
      vocalSectionIdx: vSec, instSectionIdx: iSec, harmonicShift: shift,
    })
      .then((p) => { if (!cancelled) setPlan(p); })
      .catch((e) => { if (!cancelled) setError(`Plan: ${e.message}`); });
    return () => { cancelled = true; };
    // Primitives, not the `pin` object: a fresh object literal on every render
    // would refetch the plan continuously.
  }, [vocalId, instId, vSec, iSec, shift]);

  return { plan, error };
}
