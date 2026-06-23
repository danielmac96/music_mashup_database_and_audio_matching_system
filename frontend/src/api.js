async function jsonFetch(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* not json */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const api = {
  previewPlaylist: (url) =>
    jsonFetch("/api/playlists/preview", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  ingestTracks: (tracks) =>
    jsonFetch("/api/playlists/ingest", {
      method: "POST",
      body: JSON.stringify({ tracks }),
    }),

  getTracks: () => jsonFetch("/api/tracks"),

  startDownload: (id) =>
    jsonFetch(`/api/tracks/${id}/download`, { method: "POST" }),

  startSeparate: (id) =>
    jsonFetch(`/api/tracks/${id}/separate`, { method: "POST" }),

  startAnalyze: (id) =>
    jsonFetch(`/api/tracks/${id}/analyze`, { method: "POST" }),

  correctFeatures: (id, { bpm, key, mode } = {}) =>
    jsonFetch(`/api/tracks/${id}/features`, {
      method: "PATCH",
      body: JSON.stringify({ bpm, key, mode }),
    }),

  getJob: (jobId) => jsonFetch(`/api/jobs/${jobId}`),

  audioUrl: (id, stemType) => `/api/tracks/${id}/audio/${stemType}`,

  getSections: (id) => jsonFetch(`/api/tracks/${id}/sections`),

  getWaveform: (id, stem) => jsonFetch(`/api/tracks/${id}/waveform?stem=${stem}`),

  startScoring: () => jsonFetch("/api/mashups/score", { method: "POST" }),

  getMashups: ({
    comboType = "",
    minScore = 0,
    limit = 50,
    vocalSongId = null,
    instSongId = null,
  } = {}) => {
    const params = new URLSearchParams();
    if (comboType) params.set("combo_type", comboType);
    if (minScore) params.set("min_score", String(minScore));
    params.set("limit", String(limit));
    if (vocalSongId != null) params.set("vocal_song_id", String(vocalSongId));
    if (instSongId != null) params.set("inst_song_id", String(instSongId));
    return jsonFetch(`/api/mashups?${params}`);
  },

  getMashupPlan: (vocalId, instId) =>
    jsonFetch(`/api/mashups/plan?vocal_id=${vocalId}&inst_id=${instId}`),

  startPreview: (vocalId, instId, vocalStart = null, instStart = null) => {
    const params = new URLSearchParams({ vocal_id: vocalId, inst_id: instId });
    if (vocalStart != null) params.set("vocal_start", vocalStart.toFixed(3));
    if (instStart  != null) params.set("inst_start",  instStart.toFixed(3));
    return jsonFetch(`/api/mashups/preview?${params}`, { method: "POST" });
  },

  previewAudioUrl: (vocalId, instId) =>
    `/api/mashups/preview/audio?vocal_id=${vocalId}&inst_id=${instId}`,

  startAdjust: (vocalId, instId, anchor, stretch = null, shift = null) => {
    const params = new URLSearchParams({ vocal_id: vocalId, inst_id: instId, anchor });
    if (stretch != null) params.set("stretch", String(stretch));
    if (shift != null) params.set("shift", String(shift));
    return jsonFetch(`/api/mashups/adjust?${params}`, { method: "POST" });
  },

  adjustedAudioUrl: (vocalId, instId, anchor) =>
    `/api/mashups/adjust/audio?vocal_id=${vocalId}&inst_id=${instId}&anchor=${anchor}`,
};
