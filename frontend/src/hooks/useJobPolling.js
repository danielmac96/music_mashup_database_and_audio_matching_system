import { useEffect, useState } from "react";
import { api } from "../api";

const TERMINAL = new Set(["completed", "failed"]);

export function useJobPolling(jobId, intervalMs = 1000) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setError(null);
      return;
    }

    let cancelled = false;
    let timer = null;

    const tick = async () => {
      try {
        const data = await api.getJob(jobId);
        if (cancelled) return;
        setJob(data);
        if (TERMINAL.has(data.status)) return;
      } catch (e) {
        if (!cancelled) setError(e.message);
        return;
      }
      timer = setTimeout(tick, intervalMs);
    };

    tick();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId, intervalMs]);

  return { job, error };
}
