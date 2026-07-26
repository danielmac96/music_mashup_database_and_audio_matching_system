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

  // Poll progressive metadata hydration for a playlist preview session.
  getPreviewStatus: (previewId) =>
    jsonFetch(`/api/playlists/preview/${previewId}`),

  ingestTracks: (tracks, previewId = null) =>
    jsonFetch("/api/playlists/ingest", {
      method: "POST",
      body: JSON.stringify({ tracks, preview_id: previewId }),
    }),

  getTracks: () => jsonFetch("/api/tracks"),

  startDownload: (id) =>
    jsonFetch(`/api/tracks/${id}/download`, { method: "POST" }),

  startSeparate: (id) =>
    jsonFetch(`/api/tracks/${id}/separate`, { method: "POST" }),

  startAnalyze: (id) =>
    jsonFetch(`/api/tracks/${id}/analyze`, { method: "POST" }),

  startStructure: (id) =>
    jsonFetch(`/api/tracks/${id}/structure`, { method: "POST" }),

  // Run (or resume/retry) the full auto-chain pipeline for one track.
  processTrack: (id) =>
    jsonFetch(`/api/tracks/${id}/process`, { method: "POST" }),

  // Re-check a track for a stale ~30s Go+ preview and re-download full if needed.
  reverifyTrack: (id) =>
    jsonFetch(`/api/tracks/${id}/reverify`, { method: "POST" }),

  // All pipeline jobs (newest first) — drives live per-track progress + the
  // Library batch banner. activeOnly drops finished jobs.
  getJobs: ({ kind = "pipeline", activeOnly = false } = {}) => {
    const params = new URLSearchParams();
    if (kind) params.set("kind", kind);
    if (activeOnly) params.set("active_only", "true");
    return jsonFetch(`/api/jobs?${params}`);
  },

  correctFeatures: (id, { bpm, key, mode } = {}) =>
    jsonFetch(`/api/tracks/${id}/features`, {
      method: "PATCH",
      body: JSON.stringify({ bpm, key, mode }),
    }),

  getJob: (jobId) => jsonFetch(`/api/jobs/${jobId}`),

  // Whether ffmpeg/ffprobe/yt-dlp/demucs/librosa are available on the server.
  getDeps: () => jsonFetch("/api/health/deps"),

  // pip install -U yt-dlp on the server (stale yt-dlp breaks SoundCloud).
  updateYtdlp: () => jsonFetch("/api/health/update-ytdlp", { method: "POST" }),

  // Settings / first-run wizard.
  getSettings: () => jsonFetch("/api/settings"),

  validatePath: (path) =>
    jsonFetch("/api/settings/validate-path", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  saveSettings: ({ audioRoot = null, dbPath = null, pipelineWorkers = null,
                   stemSeparator = null } = {}) =>
    jsonFetch("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        audio_root: audioRoot,
        db_path: dbPath,
        pipeline_workers: pipelineWorkers,
        stem_separator: stemSeparator,
      }),
    }),

  audioUrl: (id, stemType) => `/api/tracks/${id}/audio/${stemType}`,

  getSections: (id) => jsonFetch(`/api/tracks/${id}/sections`),

  getWaveform: (id, stem) => jsonFetch(`/api/tracks/${id}/waveform?stem=${stem}`),

  startScoring: ({ bpmMaxDiff = null, keyMinScore = null } = {}) => {
    const params = new URLSearchParams();
    if (bpmMaxDiff != null) params.set("bpm_max_diff", String(bpmMaxDiff));
    if (keyMinScore != null) params.set("key_min_score", String(keyMinScore));
    const qs = params.toString();
    return jsonFetch(`/api/mashups/score${qs ? `?${qs}` : ""}`, { method: "POST" });
  },

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

  startExport: (vocalId, instId, anchor, stretch, shift, vocalOffset, instOffset,
                vocalGain = null, instGain = null) => {
    const params = new URLSearchParams({ vocal_id: vocalId, inst_id: instId, anchor });
    if (stretch != null) params.set("stretch", String(stretch));
    if (shift != null) params.set("shift", String(shift));
    if (vocalOffset != null) params.set("vocal_offset", vocalOffset.toFixed(3));
    if (instOffset != null) params.set("inst_offset", instOffset.toFixed(3));
    if (vocalGain != null) params.set("vocal_gain", vocalGain.toFixed(3));
    if (instGain != null) params.set("inst_gain", instGain.toFixed(3));
    return jsonFetch(`/api/mashups/export?${params}`, { method: "POST" });
  },

  exportAudioUrl: (vocalId, instId) =>
    `/api/mashups/export/audio?vocal_id=${vocalId}&inst_id=${instId}`,

  // ── Studio (DAW tab) ───────────────────────────────────────────────────────
  // Render the arrangement server-side (decoupled stretch/pitch per clip) to a
  // WAV. clips: [{ song_id, stem, offset_sec, rate, semitones, gain }]
  startMixdown: (clips) =>
    jsonFetch("/api/studio/mixdown", {
      method: "POST",
      body: JSON.stringify({ clips }),
    }),

  mixdownAudioUrl: (token) => `/api/studio/mixdown/${token}/audio`,

  // ── Mixes (1001tracklists ingestion) ──────────────────────────────────────
  importMix: (url) =>
    jsonFetch("/api/mixes/import", { method: "POST", body: JSON.stringify({ url }) }),

  importMixPaste: (content, url = "") =>
    jsonFetch("/api/mixes/import-paste", {
      method: "POST",
      body: JSON.stringify({ content, url }),
    }),

  getMixes: () => jsonFetch("/api/mixes"),

  getMix: (id) => jsonFetch(`/api/mixes/${id}`),

  resolveMixTrack: (trackId, url) =>
    jsonFetch(`/api/mixes/tracks/${trackId}/resolve`, {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  // trackIds: optional subset to resolve (omit/empty = every unlinked track).
  autoResolveMix: (id, platform = "both", trackIds = null) =>
    jsonFetch(`/api/mixes/${id}/auto-resolve`, {
      method: "POST",
      body: JSON.stringify(
        trackIds && trackIds.length ? { platform, track_ids: trackIds } : { platform }),
    }),

  confirmMixTrack: (trackId) =>
    jsonFetch(`/api/mixes/tracks/${trackId}/confirm`, { method: "POST" }),

  // Scrape the track's 1001tracklists detail page for its real SoundCloud/YouTube
  // URL. On-demand only (one Firecrawl call per click).
  scrapeMixTrackLink: (trackId) =>
    jsonFetch(`/api/mixes/tracks/${trackId}/scrape-link`, { method: "POST" }),

  ingestMix: (id) => jsonFetch(`/api/mixes/${id}/ingest`, { method: "POST" }),

  // Manually add a track (artist/title + optional SC/YT link). No link → the row
  // is left 'unresolved' for the Auto-link flow. Returns the new track row.
  addMixTrack: (id, { artist = "", title, link = "" }) =>
    jsonFetch(`/api/mixes/${id}/tracks`, {
      method: "POST",
      body: JSON.stringify({ artist, title, link }),
    }),

  // Remove a not-yet-ingested track (and its match pairs). Returns full detail.
  deleteMixTrack: (id, trackId) =>
    jsonFetch(`/api/mixes/${id}/tracks/${trackId}`, { method: "DELETE" }),

  reorderMixTracks: (id, trackIds) =>
    jsonFetch(`/api/mixes/${id}/reorder`, {
      method: "POST",
      body: JSON.stringify({ track_ids: trackIds }),
    }),

  // Bulk role + match save from the matching board. `roles` is
  // [{track_id, role}], `matches` is [{vocal_track_id, inst_track_id|null}].
  saveMixAssignments: (id, roles, matches) =>
    jsonFetch(`/api/mixes/${id}/assignments`, {
      method: "POST",
      body: JSON.stringify({ roles, matches }),
    }),

  // Discard manual edits and rebuild the original 'w/'-derived grouping.
  resetMixMatches: (id) =>
    jsonFetch(`/api/mixes/${id}/reset-matches`, { method: "POST" }),

  // ── Training data + learned model ─────────────────────────────────────────
  getDatasets: () => jsonFetch("/api/datasets"),

  buildDataset: ({ name = "bbm", negRatio = 5, seed = 42 } = {}) =>
    jsonFetch("/api/datasets/build", {
      method: "POST",
      body: JSON.stringify({ name, neg_ratio: negRatio, seed }),
    }),

  getModels: () => jsonFetch("/api/models"),

  trainModel: (datasetId) =>
    jsonFetch("/api/models/train", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId }),
    }),

  activateModel: (id) => jsonFetch(`/api/models/${id}/activate`, { method: "POST" }),

  getScorerStatus: () => jsonFetch("/api/mashups/scorer-status"),

  // Read-only DB browser (debug view). Maps to the api/routes/database.py
  // router mounted at /api/db.
  getDbTables: () => jsonFetch("/api/db/tables"),

  getDbTable: (table, limit = 100, offset = 0) =>
    jsonFetch(`/api/db/tables/${table}?limit=${limit}&offset=${offset}`),
};
