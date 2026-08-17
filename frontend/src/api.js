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

  // ── Pair judgments (T2.1) — the ✓/✗ made while triaging the ranked list.
  // Survives "Score library", unlike mashup_candidates.
  getPairFeedback: (verdict = "") =>
    jsonFetch(`/api/mashups/feedback${verdict ? `?verdict=${verdict}` : ""}`),

  savePairFeedback: ({ vocalSongId, instSongId, verdict,
                       vocalSection = null, instSection = null }) =>
    jsonFetch("/api/mashups/feedback", {
      method: "POST",
      body: JSON.stringify({
        vocal_song_id: vocalSongId, inst_song_id: instSongId, verdict,
        vocal_section: vocalSection, inst_section: instSection,
      }),
    }),

  // Which beat of the bar this stem's grid starts on (0-3). Set by alt+clicking
  // a beat line in Studio when detected bar lines don't match what you hear.
  setBeatPhase: (id, stem, phase) =>
    jsonFetch(`/api/tracks/${id}/beat-phase`, {
      method: "PATCH",
      body: JSON.stringify({ stem, phase }),
    }),

  // Remove a song from the library — deletes its DB rows AND audio/stem files.
  deleteTrack: (id) => jsonFetch(`/api/tracks/${id}`, { method: "DELETE" }),

  // Repoint a song at a corrected URL. Resets download/stems/analysis and
  // re-runs the pipeline from the new URL.
  updateTrackUrl: (id, sourceUrl) =>
    jsonFetch(`/api/tracks/${id}/url`, {
      method: "PATCH",
      body: JSON.stringify({ source_url: sourceUrl }),
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

  // Create a fresh empty library (db + audio folders) at `path` and make it
  // active. Takes effect on the next server restart (paths bind at import).
  newLibrary: (path, force = false) =>
    jsonFetch("/api/settings/new-library", {
      method: "POST",
      body: JSON.stringify({ path, force }),
    }),

  // Pass only the keys you are changing. Paths and worker counts need a
  // restart; the separator, the stem mode and every scoring knob are re-read on
  // use, so they apply to the next separation / re-score.
  saveSettings: (patch = {}) => {
    const map = {
      audioRoot: "audio_root", dbPath: "db_path",
      pipelineWorkers: "pipeline_workers", stemSeparator: "stem_separator",
      stemMode: "stem_mode",
    };
    const body = {};
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === undefined) continue;
      body[map[k] || k] = v;
    }
    return jsonFetch("/api/settings", { method: "POST", body: JSON.stringify(body) });
  },

  // Which generation of features the library is missing, per feature group.
  getStaleness: () => jsonFetch("/api/tracks/staleness"),

  // Re-run a pipeline stage across many tracks. action: analyze | separate |
  // process. scope: stale (only what needs it) | all | ids.
  bulkReprocess: ({ action = "analyze", scope = "stale", songIds = null } = {}) =>
    jsonFetch("/api/tracks/bulk", {
      method: "POST",
      body: JSON.stringify({ action, scope, song_ids: songIds }),
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
    // 0 = uncapped. Server-side (T3.4): capping a truncated 50 client-side
    // would just show fewer rows, not better ones.
    maxPerSong = 3,
    // T3.5 filters — also server-side, and for the same reason.
    genre = "", era = "", energy = "", bpmBand = "", vocalForward = false,
    // Phase C — cap on how much work a pair costs to build (0-1). null = any.
    maxEffort = null,
    // Phase F — "score" (best first) or "uncertain" (the model's blind spots,
    // where a verdict buys the most information per keypress).
    order = "score",
    // Phase F — 0 = safest fit first, 1 = most adventurous. Only reorders pairs
    // that already cleared every technical gate; it never surfaces a bad fit.
    adventure = 0,
    // C.2 — try a different balance without re-scoring. An object of the five
    // sub-score weights; the server re-ranks the WHOLE table on them and
    // returns re-weighted totals and percentiles. null = use the saved set.
    weights = null,
    // Server-side sort, so the export can ask for the page it is looking at.
    sort = "",
  } = {}) => {
    const params = new URLSearchParams();
    if (comboType) params.set("combo_type", comboType);
    if (minScore) params.set("min_score", String(minScore));
    params.set("limit", String(limit));
    if (vocalSongId != null) params.set("vocal_song_id", String(vocalSongId));
    if (instSongId != null) params.set("inst_song_id", String(instSongId));
    params.set("max_per_song", String(maxPerSong));
    if (maxEffort != null) params.set("max_effort", String(maxEffort));
    if (order && order !== "score") params.set("order", order);
    if (adventure > 0) params.set("adventure", String(adventure));
    if (genre) params.set("genre", genre);
    if (era) params.set("era", era);
    if (energy) params.set("energy", energy);
    if (bpmBand) params.set("bpm_band", bpmBand);
    if (vocalForward) params.set("vocal_forward", "true");
    if (weights) params.set("weights", JSON.stringify(weights));
    if (sort && sort !== "score") params.set("sort", sort);
    return jsonFetch(`/api/mashups?${params}`);
  },

  // Which filter values this library actually contains.
  getMashupFilters: (comboType = "") =>
    jsonFetch(`/api/mashups/filters${comboType ? `?combo_type=${comboType}` : ""}`),

  // "The best bed for each of my vocals" — every acapella gets a turn instead
  // of one well-placed vocal owning the page.
  getBestBedPerVocal: ({ limit = 50, perVocal = 1, minScore = 0 } = {}) => {
    const params = new URLSearchParams({
      limit: String(limit), per_vocal: String(perVocal),
    });
    if (minScore) params.set("min_score", String(minScore));
    return jsonFetch(`/api/mashups/by-vocal?${params}`);
  },

  // ── Hidden pairs / excluded tracks (T3.4) ─────────────────────────────────
  // Display preferences, not judgments: they survive "Score library" but are
  // deliberately not training data.
  getHidden: () => jsonFetch("/api/mashups/hidden"),

  hidePair: (vocalSongId, instSongId) =>
    jsonFetch("/api/mashups/hidden", {
      method: "POST",
      body: JSON.stringify({ vocal_song_id: vocalSongId, inst_song_id: instSongId }),
    }),

  unhidePair: (vocalSongId, instSongId) =>
    jsonFetch(`/api/mashups/hidden?vocal_song_id=${vocalSongId}`
      + `&inst_song_id=${instSongId}`, { method: "DELETE" }),

  excludeTrack: (songId) =>
    jsonFetch(`/api/mashups/excluded/${songId}`, { method: "POST" }),

  includeTrack: (songId) =>
    jsonFetch(`/api/mashups/excluded/${songId}`, { method: "DELETE" }),

  // `pin` ties the plan to a candidate row's own section pair and measured
  // transpose, so the recipe describes the moment that was auditioned rather
  // than one the server re-chooses. Omit it for an ad-hoc pair.
  getMashupPlan: (vocalId, instId, {
    vocalSectionIdx = null, instSectionIdx = null, harmonicShift = null,
  } = {}) => {
    const params = new URLSearchParams({
      vocal_id: String(vocalId), inst_id: String(instId),
    });
    if (vocalSectionIdx != null) params.set("vocal_section_idx", String(vocalSectionIdx));
    if (instSectionIdx != null) params.set("inst_section_idx", String(instSectionIdx));
    if (harmonicShift != null) params.set("harmonic_shift", String(harmonicShift));
    return jsonFetch(`/api/mashups/plan?${params}`);
  },

  // ── Studio (DAW tab) ───────────────────────────────────────────────────────
  // Render the arrangement server-side (decoupled stretch/pitch per clip) to a
  // WAV. clips: [{ song_id, stem, offset_sec, rate, semitones, gain }]
  startMixdown: (clips) =>
    jsonFetch("/api/studio/mixdown", {
      method: "POST",
      body: JSON.stringify({ clips }),
    }),

  mixdownAudioUrl: (token) => `/api/studio/mixdown/${token}/audio`,

  // Export a mashup as an FL Studio session folder: both stems conformed to the
  // target tempo and key and padded so bar 1 is at 0:00, plus a click, the
  // recipe, and a session.json in the mixdown clip shape. A mixdown is a bounce;
  // this is something you can actually mix.
  startSessionExport: (vocalSongId, instSongId) =>
    jsonFetch("/api/studio/session", {
      method: "POST",
      body: JSON.stringify({ vocal_song_id: vocalSongId, inst_song_id: instSongId }),
    }),

  // The same, for the top N of the currently filtered ranked list. Filters go to
  // the server rather than a list of ids so the export matches what is on
  // screen — including the diversity cap, which is applied after the SQL.
  startBatchSessionExport: (opts = {}) =>
    jsonFetch("/api/mashups/session/batch", {
      method: "POST",
      body: JSON.stringify(opts),
    }),

  sessionArchiveUrl: (token) => `/api/studio/session/${token}/archive`,

  // The Audition tab's export used to live here as startAuditionExport — a
  // fixed two-clip wrapper over this same endpoint, with its own duplicate of
  // mixdownAudioUrl. It went with the tab (T4.1): one arranger, one export
  // payload shape.

  // ── Mixes (1001tracklists ingestion) ──────────────────────────────────────
  importMix: (url) =>
    jsonFetch("/api/mixes/import", { method: "POST", body: JSON.stringify({ url }) }),

  getMixes: () => jsonFetch("/api/mixes"),

  getMix: (id) => jsonFetch(`/api/mixes/${id}`),

  resolveMixTrack: (trackId, url) =>
    jsonFetch(`/api/mixes/tracks/${trackId}/resolve`, {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  // trackIds: optional subset to resolve (omit/empty = every unlinked track).
  // relink: also re-search tracks a previous auto-link got wrong. Manual,
  // scraped and already-ingested links are never overwritten.
  autoResolveMix: (id, platform = "both", trackIds = null, relink = false) =>
    jsonFetch(`/api/mixes/${id}/auto-resolve`, {
      method: "POST",
      body: JSON.stringify({
        platform,
        ...(trackIds && trackIds.length ? { track_ids: trackIds } : {}),
        ...(relink ? { relink: true } : {}),
      }),
    }),

  // Ranked search hits for one track, so a wrong auto-link can be fixed by
  // picking the right one rather than hunting down a URL to paste. Normally
  // served instantly from what auto-link already fetched; `refresh` forces a
  // fresh search.
  mixTrackCandidates: (trackId, platform = "soundcloud", limit = 5, refresh = false) =>
    jsonFetch(`/api/mixes/tracks/${trackId}/candidates`
      + `?platform=${encodeURIComponent(platform)}&limit=${limit}`
      + (refresh ? "&refresh=true" : "")),

  // Clear links, returning tracks to "needs link". Omit trackIds to unlink every
  // linked track. Already-ingested tracks are skipped server-side.
  unlinkMixTracks: (id, trackIds = null) =>
    jsonFetch(`/api/mixes/${id}/unlink`, {
      method: "POST",
      body: JSON.stringify(trackIds && trackIds.length ? { track_ids: trackIds } : {}),
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

  deactivateModel: (id) =>
    jsonFetch(`/api/models/${id}/deactivate`, { method: "POST" }),

  deleteModel: (id) => jsonFetch(`/api/models/${id}`, { method: "DELETE" }),

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
