// A POST whose answer is a file (an export built from a body too large for a
// query string): fetch it and hand it to the browser as a download.
async function downloadPost(url, body, fallbackName) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(`${res.status} ${detail}`);
  }
  const cd = res.headers.get("Content-Disposition") || "";
  const name = (cd.match(/filename="([^"]+)"/) || [])[1] || fallbackName;
  const blob = await res.blob();
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 5000);
  return { name, skipped: Number(res.headers.get("X-Skipped-Tracks") || 0) };
}

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

  // `groupName` also files the import under a named library group (a crate), so
  // a SoundCloud set stays a set once it is in the library. The group holds the
  // WHOLE import, including tracks that were already here and came back as
  // skipped duplicates.
  ingestTracks: (tracks, previewId = null, groupName = null) =>
    jsonFetch("/api/playlists/ingest", {
      method: "POST",
      body: JSON.stringify({ tracks, preview_id: previewId,
                             group_name: groupName || null }),
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

  // Move a selected track and its likeliest partners ahead of the import queue.
  prefetchTrack: (id) =>
    jsonFetch(`/api/tracks/${id}/prefetch`, { method: "POST" }),

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

  // The pipeline pools (workers / busy / waiting per stage) and each waiting
  // job's place in line — the Queue screen.
  getQueue: () => jsonFetch("/api/jobs/queue"),

  correctFeatures: (id, { bpm, key, mode } = {}) =>
    jsonFetch(`/api/tracks/${id}/features`, {
      method: "PATCH",
      body: JSON.stringify({ bpm, key, mode }),
    }),

  // ── Pair judgments (T2.1) — the ✓/✗ made while triaging the ranked list.
  // Survives "Score library", unlike mashup_candidates.
  getPairFeedback: (verdict = "") =>
    jsonFetch(`/api/mashups/feedback${verdict ? `?verdict=${verdict}` : ""}`),

  // `rating` is the dock's 1-5 star; `verdict` is Discover's ✓/~/✗. Send
  // either — the server derives the other, so the learned scorer keeps seeing
  // the verdict it trains on whichever control the judgement came from.
  savePairFeedback: ({ vocalSongId, instSongId, verdict = null, rating = null,
                       vocalSection = null, instSection = null }) =>
    jsonFetch("/api/mashups/feedback", {
      method: "POST",
      body: JSON.stringify({
        vocal_song_id: vocalSongId, inst_song_id: instSongId, verdict, rating,
        vocal_section: vocalSection, inst_section: instSection,
      }),
    }),

  // Does each scored term agree with the ratings? (matcher/term_report.py)
  getTermReport: () => jsonFetch("/api/mashups/term-report"),

  // Why a judged pair got its star — keys of models.FEEDBACK_REASONS
  // (pairModel.VERDICT_REASONS). 404 when the pair is not rated.
  savePairReasons: ({ vocalSongId, instSongId, vocalSection = null,
                      instSection = null, reasons = [] }) =>
    jsonFetch("/api/mashups/feedback/reasons", {
      method: "POST",
      body: JSON.stringify({
        vocal_song_id: vocalSongId, inst_song_id: instSongId,
        vocal_section: vocalSection, inst_section: instSection, reasons,
      }),
    }),

  // Forget a judgement — the star, and the verdict it implied, together. The
  // four ids are the key; there is no "rating: 0", because the POST above
  // COALESCEs a null rating into the one already stored.
  clearPairFeedback: ({ vocalSongId, instSongId,
                        vocalSection = null, instSection = null }) =>
    jsonFetch("/api/mashups/feedback", {
      method: "DELETE",
      body: JSON.stringify({
        vocal_song_id: vocalSongId, inst_song_id: instSongId,
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
  // `pick` ({ title, uploader, duration_secs }) is set when the link came from
  // the "Wrong audio?" picker, and is recorded as where the audio came from.
  updateTrackUrl: (id, sourceUrl, pick = null) =>
    jsonFetch(`/api/tracks/${id}/url`, {
      method: "PATCH",
      body: JSON.stringify({ source_url: sourceUrl, pick }),
    }),

  // YouTube uploads that could be this track's audio, each with the verdict the
  // download fallback applies (passes / reason / duration_delta). Slow — two
  // live searches — so the picker shows that it is searching.
  audioCandidates: (id) => jsonFetch(`/api/tracks/${id}/audio-candidates`),

  // "✓ Sounds right": you listened and a substituted track's audio is the
  // record. Settles YT? to YT and drops it from the suspect-audio count.
  confirmAudio: (id) =>
    jsonFetch(`/api/tracks/${id}/audio-confirm`, { method: "POST" }),

  getJob: (jobId) => jsonFetch(`/api/jobs/${jobId}`),

  // Median / p90 wall time per pipeline stage (analysis_runs), for an ETA.
  getJobTimings: (since = "") =>
    jsonFetch(`/api/jobs/timings${since ? `?since=${encodeURIComponent(since)}` : ""}`),

  // Which analyser is configured and effective, model coverage, per-group
  // cache coverage and librosa ↔ Essentia agreement.
  getAnalysisStatus: () => jsonFetch("/api/analysis/status"),

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
    // Title/artist substring on either side, and paging past the first page —
    // both in SQL, so they search and page the library rather than the page.
    search = "", offset = 0,
    // Keepers / unrated, the landing key ± n Camelot steps, section types.
    rated = "", key = "", keyTol = 1, vocalLabel = "", instLabel = "",
  } = {}) => {
    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (offset > 0) params.set("offset", String(offset));
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
    if (rated) params.set("rated", rated);
    if (key) { params.set("key", key); params.set("key_tolerance", String(keyTol)); }
    if (vocalLabel) params.set("vocal_label", vocalLabel);
    if (instLabel) params.set("inst_label", instLabel);
    return jsonFetch(`/api/mashups?${params}`);
  },

  // Which genre / era / BPM / energy values the scored pairs actually contain,
  // so the dock's filter menus only offer what will match something.
  getMashupFilters: () => jsonFetch("/api/mashups/filters"),

  // "Best bed for each of my vocals": every acapella gets one row, ordered by
  // how good its best option is.
  getBestBedPerVocal: ({ limit = 40, perVocal = 1, minScore = 0 } = {}) =>
    jsonFetch(`/api/mashups/by-vocal?limit=${limit}&per_vocal=${perVocal}`
      + `&min_score=${minScore}`),

  // Render one candidate's two sections, conformed, to a WAV server-side —
  // the same maths the FL export uses, as a file you can keep or send.
  startCandidatePreview: (candidateId) =>
    jsonFetch(`/api/mashups/${candidateId}/preview`, { method: "POST" }),

  // Export the top N pairs (under the given filters) as one zip of FL session
  // folders. Body keys are the list route's: top_n, min_score, max_effort, …
  startSessionBatch: (body) =>
    jsonFetch("/api/mashups/session/batch", {
      method: "POST",
      body: JSON.stringify(body),
    }),

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

  getMashupPlan: (vocalId, instId) =>
    jsonFetch(`/api/mashups/plan?vocal_id=${vocalId}&inst_id=${instId}`),

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
  // `sections` ({ vocal, inst } section indexes) names the pairing to export —
  // Studio's armed timing; without it the server falls back to its own pick.
  startSessionExport: (vocalSongId, instSongId, sections = null) =>
    jsonFetch("/api/studio/session", {
      method: "POST",
      body: JSON.stringify({
        vocal_song_id: vocalSongId, inst_song_id: instSongId,
        vocal_section_idx: sections?.vocal ?? null,
        inst_section_idx: sections?.inst ?? null,
      }),
    }),

  sessionArchiveUrl: (token) => `/api/studio/session/${token}/archive`,

  // The Audition tab's export used to live here as startAuditionExport — a
  // fixed two-clip wrapper over this same endpoint, with its own duplicate of
  // mixdownAudioUrl. It went with the tab (T4.1): one arranger, one export
  // payload shape.

  // ── Mixes (1001tracklists ingestion) ──────────────────────────────────────
  // A 1001tracklists URL answers { job_id } — a stealth render of a heavy set
  // is minutes of work and never belonged inside the request. A plain-HTML
  // tracklist page still answers with the mix itself. `refresh` pays for a
  // fresh render instead of re-parsing the cached markdown.
  importMix: (url, refresh = false) =>
    jsonFetch("/api/mixes/import", {
      method: "POST",
      body: JSON.stringify({ url, refresh }),
    }),

  // The tracklist text, parsed server-side by the same line parser the scrape
  // path uses. No key, no render — it works when the scrape is walled, which
  // on 1001tracklists it currently always is. `url` is the set page this came
  // from: source_url is UNIQUE, so it is what makes a re-paste replace the mix
  // rather than add a second copy.
  importMixPaste: (content, url = "") =>
    jsonFetch("/api/mixes/import-paste", {
      method: "POST",
      body: JSON.stringify({ content, url }),
    }),

  // A capture from the browser bookmarklet. It IS the markdown a scrape would
  // have returned, so the server runs it through the same parser — and caches it
  // under `url`, which makes a later URL import succeed with no network call.
  importMixMarkdown: (markdown, url = "") =>
    jsonFetch("/api/mixes/import-markdown", {
      method: "POST",
      body: JSON.stringify({ markdown, url }),
    }),

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

  // Answers { job_id, queued }: a 200-track mix is 200 metadata fetches. Poll
  // the job; its result carries the counts.
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

  // The Analysis panel (readme §9, C): every attribute with its coverage, and
  // which ones the Library and Track detail show.
  getAttributes: () => jsonFetch("/api/analysis/attributes"),
  setAttributeVisibility: (vis) => jsonFetch("/api/analysis/attributes/visibility", {
    method: "PUT", body: JSON.stringify(vis),
  }),

  // ── Discovery (SoundCloud search/browse) + crates ──────────────────────────
  // Every track row comes back with `in_library` already resolved server-side,
  // so the browser never has to reconcile results against the library itself.

  // Where the library is thin: vocals with few beds, beds with few vocals.
  discoveryGaps: (maxPartners = 2) => jsonFetch(`/api/discovery/gaps?max_partners=${maxPartners}`),

  discoverySearch: (q, kind = "tracks", cursor = null, limit = 20) =>
    jsonFetch(`/api/discovery/search?${new URLSearchParams({
      q, kind, limit: String(limit), ...(cursor ? { cursor } : {}),
    })}`),

  // Paste any SoundCloud link: a track, a set, or an artist page. A set resolves
  // straight to its tracks and an artist to their uploads.
  discoveryResolve: (url) =>
    jsonFetch("/api/discovery/resolve", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  discoveryUserFeed: (userId, feed = "tracks", cursor = null) =>
    jsonFetch(`/api/discovery/users/${userId}/${feed}${cursor
      ? `?cursor=${encodeURIComponent(cursor)}` : ""}`),

  discoveryPlaylist: (playlistId) =>
    jsonFetch(`/api/discovery/playlists/${playlistId}`),

  discoveryRelated: (trackId, cursor = null) =>
    jsonFetch(`/api/discovery/tracks/${trackId}/related${cursor
      ? `?cursor=${encodeURIComponent(cursor)}` : ""}`),

  discoveryImport: (rows, groupName = null) =>
    jsonFetch("/api/discovery/import", {
      method: "POST",
      body: JSON.stringify({ rows, group_name: groupName || null }),
    }),

  discoveryStatus: () => jsonFetch("/api/discovery/status"),

  // Your profile. This IDENTIFIES a public account rather than logging in —
  // the write layer needs a registered app (Artist Pro) and is dormant, so only
  // public sets, likes and uploads are readable, and the UI says as much.
  discoveryProfile: () => jsonFetch("/api/discovery/profile"),

  discoverySetProfile: (url) =>
    jsonFetch("/api/discovery/profile", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  // Bookmarked profiles for Discover's sidebar. NOT "followed profiles":
  // followings need /me/followings, i.e. OAuth, which ships dormant. Saving is
  // a bookmark, and there is no "n new" count because nothing snapshots them.
  discoverySavedProfiles: () => jsonFetch("/api/discovery/saved-profiles"),

  discoverySaveProfile: (url) =>
    jsonFetch("/api/discovery/saved-profiles", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  discoveryForgetProfile: (userId) =>
    jsonFetch(`/api/discovery/saved-profiles/${userId}`, { method: "DELETE" }),

  discoveryDisconnect: () =>
    jsonFetch("/api/discovery/profile", { method: "DELETE" }),

  // Library tracks that can seed a suggestion run — i.e. the ones carrying a
  // SoundCloud track id, since the fan-out is /tracks/{id}/related.
  discoverySeeds: () => jsonFetch("/api/discovery/seeds"),

  // Returns { job_id, seed_count, offered }; poll the job for the result. It is
  // a job because one request per seed at the browse layer's deliberate pace is
  // tens of seconds.
  discoveryRecommend: (body) =>
    jsonFetch("/api/discovery/recommend", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getCrates: () => jsonFetch("/api/crates"),

  // Every crate as a LIBRARY GROUP: name, counts, and the ids of the library
  // songs it holds, in crate order. One request for the whole screen — the rail
  // draws every group's count at once, and filtering by one is then arithmetic
  // over rows already in memory rather than a query.
  getLibraryGroups: () => jsonFetch("/api/crates/groups"),

  // The Library's counterpart to addCrateItems, which takes browse rows for
  // tracks that may not be here yet. These are already in the library, so the
  // item is written already linked.
  addSongsToCrate: (id, songIds) =>
    jsonFetch(`/api/crates/${id}/songs`, {
      method: "POST",
      body: JSON.stringify({ song_ids: songIds }),
    }),

  removeSongsFromCrate: (id, songIds) =>
    jsonFetch(`/api/crates/${id}/songs/remove`, {
      method: "POST",
      body: JSON.stringify({ song_ids: songIds }),
    }),

  // Which crates already hold these rows. POST because a page of permalinks is
  // far too long for a query string. Keyed by the URL as sent, so the caller
  // never has to re-implement normalize_url in JS.
  crateMembership: (urls, trackIds = []) =>
    jsonFetch("/api/crates/membership", {
      method: "POST",
      body: JSON.stringify({ urls, track_ids: trackIds }),
    }),

  getCrate: (id) => jsonFetch(`/api/crates/${id}`),

  createCrate: (name, note = "") =>
    jsonFetch("/api/crates", { method: "POST", body: JSON.stringify({ name, note }) }),

  renameCrate: (id, name) =>
    jsonFetch(`/api/crates/${id}`, { method: "PATCH", body: JSON.stringify({ name }) }),

  deleteCrate: (id) => jsonFetch(`/api/crates/${id}`, { method: "DELETE" }),

  addCrateItems: (id, rows) =>
    jsonFetch(`/api/crates/${id}/items`, {
      method: "POST",
      body: JSON.stringify({ rows }),
    }),

  removeCrateItems: (id, itemIds) =>
    jsonFetch(`/api/crates/${id}/items/remove`, {
      method: "POST",
      body: JSON.stringify({ item_ids: itemIds }),
    }),

  reorderCrate: (id, itemIds) =>
    jsonFetch(`/api/crates/${id}/reorder`, {
      method: "POST",
      body: JSON.stringify({ item_ids: itemIds }),
    }),

  ingestCrate: (id) => jsonFetch(`/api/crates/${id}/ingest`, { method: "POST" }),

  // A plain link, not a fetch: the response is a file download.
  // A new crate from a pasted list of links (one per line) — resolved
  // server-side into the same frozen rows Discover adds.
  importCrateUrls: (name, urls) =>
    jsonFetch("/api/crates/import", {
      method: "POST",
      body: JSON.stringify({ name, urls }),
    }),

  crateExportUrl: (id, format = "urls") =>
    `/api/crates/${id}/export?format=${format}`,

  // Dormant until SoundCloud app credentials exist — answers 501 with setup
  // instructions, which the UI shows as the tooltip on the disabled button.
  pushCrate: (id, sharing = "private") =>
    jsonFetch(`/api/crates/${id}/push`, {
      method: "POST",
      body: JSON.stringify({ sharing }),
    }),

  getDbTables: () => jsonFetch("/api/db/tables"),

  getDbTable: (table, limit = 100, offset = 0) =>
    jsonFetch(`/api/db/tables/${table}?limit=${limit}&offset=${offset}`),

  // The lyric cue of one vocal section; empty text clears it.
  saveSectionLine: (songId, s, text) =>
    jsonFetch(`/api/tracks/${songId}/section-line`, {
      method: "POST",
      body: JSON.stringify({ start_sec: s.start_sec, end_sec: s.end_sec, text }),
    }),

  // ── Sets: chosen mashups in running order ──────────────────────────────────
  getSets: () => jsonFetch("/api/sets"),
  createSet: (name) =>
    jsonFetch("/api/sets", { method: "POST", body: JSON.stringify({ name }) }),
  getSet: (id) => jsonFetch(`/api/sets/${id}`),
  updateSet: (id, patch) =>
    jsonFetch(`/api/sets/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  deleteSet: (id) => jsonFetch(`/api/sets/${id}`, { method: "DELETE" }),
  // A pair is named by its four ids (readme §7), never by candidate.id.
  addToSet: (id, c) =>
    jsonFetch(`/api/sets/${id}/items`, {
      method: "POST",
      body: JSON.stringify({
        vocal_song_id: c.vocal_song_id, inst_song_id: c.inst_song_id,
        vocal_section: c.vocal_section_idx ?? null, inst_section: c.inst_section_idx ?? null,
      }),
    }),
  removeFromSet: (id, itemId) =>
    jsonFetch(`/api/sets/${id}/items/${itemId}`, { method: "DELETE" }),
  reorderSet: (id, itemIds) =>
    jsonFetch(`/api/sets/${id}/reorder`, {
      method: "POST", body: JSON.stringify({ item_ids: itemIds }),
    }),
  suggestSetOrder: (id, start = null) =>
    jsonFetch(`/api/sets/${id}/suggest-order${start != null ? `?start=${start}` : ""}`),
  setExportUrl: (id, format, base = "") => {
    const q = new URLSearchParams({ format });
    if (base) q.set("base", base);
    return `/api/sets/${id}/export?${q}`;
  },

  // ── Notes on a pair, and plain exports of the dock's pairs ─────────────────
  getPairNotes: () => jsonFetch("/api/mashups/notes"),
  savePairNote: (c, note) =>
    jsonFetch("/api/mashups/notes", {
      method: "POST",
      body: JSON.stringify({
        vocal_song_id: c.vocal_song_id, inst_song_id: c.inst_song_id,
        vocal_section: c.vocal_section_idx ?? null, inst_section: c.inst_section_idx ?? null,
        note,
      }),
    }),
  exportPairs: (rows, format, name = "pairs", base = "") =>
    downloadPost("/api/mashups/export", {
      format, name, base: base || null,
      pairs: rows.map((c) => ({
        vocal_song_id: c.vocal_song_id, inst_song_id: c.inst_song_id,
        vocal_section: c.vocal_section_idx ?? null, inst_section: c.inst_section_idx ?? null,
      })),
    }, `${name}.${format === "rekordbox" ? "xml" : format === "cue" ? "txt" : "csv"}`),
};
