import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

// The library, fetched once and shared.
//
// GET /api/tracks returns every song with no pagination, so there is exactly
// one useful copy of it in the app. Before the revamp TrackList owned it, which
// was fine while the library was a tab; now the sidebar counts, the track
// table, the pair dock's scope and the detail view all read the same rows, and
// four independent fetches of the whole library would be four answers that can
// disagree mid-pipeline.
//
// The polling is TrackList's, moved rather than rewritten: while any pipeline
// job is active the list is silently re-pulled so statuses and pipeline dots
// advance without the user clicking anything, and once more on the falling edge
// so the final 'analysed' state lands after the last job completes.

const JOB_ACTIVE = new Set(["queued", "running"]);
const POLL_ACTIVE_MS = 1500;
const POLL_IDLE_MS = 5000;

export function useLibrary() {
  const [tracks, setTracks] = useState([]);
  const [pipeJobs, setPipeJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const data = await api.getTracks();
      setTracks(data.tracks);
    } catch (e) {
      setError(e.message);
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const prevActiveRef = useRef(false);
  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      try {
        const { jobs } = await api.getJobs({ kind: "pipeline" });
        if (cancelled) return;
        setPipeJobs(jobs);
        const anyActive = jobs.some((j) => JOB_ACTIVE.has(j.status));
        if (anyActive || prevActiveRef.current) await refresh(true);
        prevActiveRef.current = anyActive;
        timer = setTimeout(poll, anyActive ? POLL_ACTIVE_MS : POLL_IDLE_MS);
      } catch {
        if (!cancelled) timer = setTimeout(poll, POLL_IDLE_MS);
      }
    };
    poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [refresh]);

  return { tracks, pipeJobs, loading, error, refresh };
}
