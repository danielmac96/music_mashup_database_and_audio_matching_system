import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { MATCH_PRESETS, loadPresetIdx } from "../matchPresets";
import { onPairsInputChanged, onRescoreRequest } from "../rescore";

// Whether the scored pairs still describe the library, and the one way to
// bring them up to date from where you work. GET /api/mashups/freshness counts
// analysed tracks whose scoring inputs changed since the last Score library run
// began (songs.pairs_input_at); "Score library" itself used to live only in
// Settings, behind a collapsed card, so nothing on the Library said the pairs
// were stale after a BPM fix or an import.
const POLL_MS = 20000;

export function useFreshness({ onScored = null } = {}) {
  const [fresh, setFresh] = useState(null);
  const [jobId, setJobId] = useState(null);

  const refresh = useCallback(() => {
    api.getFreshness()
      .then((f) => {
        setFresh(f);
        // A run started elsewhere (Settings, another tab) shows here too.
        if (f.running_job_id) setJobId((cur) => cur || f.running_job_id);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  const rescore = useCallback(async () => {
    if (jobId) return;
    const p = MATCH_PRESETS[loadPresetIdx()];
    try {
      const { job_id } = await api.startScoring({ bpmMaxDiff: p.bpm, keyMinScore: p.key });
      setJobId(job_id);
      toast(`Scoring library (${p.label} match)…`);
    } catch (e) {
      toast(`Could not start scoring: ${e.message}`);
    }
  }, [jobId]);

  useEffect(() => onRescoreRequest(rescore), [rescore]);
  useEffect(() => onPairsInputChanged(refresh), [refresh]);

  const onJobDone = useCallback((job) => {
    setJobId(null);
    refresh();
    if (job?.status === "completed") onScored?.();
  }, [refresh, onScored]);

  const changed = fresh?.changed_tracks || 0;
  return {
    fresh, jobId, rescore, onJobDone, refresh,
    // What the pill should say, or null when there is nothing to do.
    label: jobId ? null
      : fresh?.never_scored ? "Score library"
      : changed > 0 ? `Re-score · ${changed} changed` : null,
  };
}
