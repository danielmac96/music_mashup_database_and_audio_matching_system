import { useEffect, useState } from "react";
import { DndContext, PointerSensor, useSensor, useSensors, closestCenter } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable, arrayMove } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { api } from "../api";
import { classifyUrl } from "../sources";
import { toast } from "../toast";
import { useJobPolling } from "../hooks/useJobPolling";
import { MlPanel } from "./MlPanel";
import { MixMatchBoard } from "./MixMatchBoard";

function SortableRow({ track, children }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: track.id });
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };
  return (
    <div ref={setNodeRef} style={style} className="mix-row">
      <span className="drag-handle" {...attributes} {...listeners}
        title="Drag to reorder" style={{ cursor: "grab", userSelect: "none" }}>⋮⋮</span>
      {children}
    </div>
  );
}

// Ingest pipeline stages a song passes through, in order. Drives the per-mix
// processing tracker so a heavy, multi-session ingest stays legible.
const INGEST_STAGES = ["queued", "downloaded", "stemmed", "analysed"];

function StageStepper({ status, error }) {
  const s = status || "queued";
  const isErr = String(s).startsWith("error");
  const reached = isErr ? -1 : INGEST_STAGES.indexOf(s);
  return (
    <span className="stage-stepper" title={isErr ? (error || s) : `stage: ${s}`}>
      {INGEST_STAGES.map((st, i) => (
        <span key={st}
          className={`stage-dot${i <= reached ? " on" : ""}${isErr ? " err" : ""}`} />
      ))}
      <span className={`stage-label${isErr ? " err" : ""}`}>{isErr ? "error" : s}</span>
    </span>
  );
}

// Mixes tab: import a DJ-set tracklist by URL, resolve each entry to a
// SoundCloud/YouTube link, then ingest the resolved tracks through the normal
// pipeline.

function ResolveInput({ track, onResolved }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    const u = url.trim();
    if (!u) return;
    if (classifyUrl(u).source === "unknown") {
      toast("Paste a SoundCloud or YouTube track link");
      return;
    }
    setBusy(true);
    try {
      const updated = await api.resolveMixTrack(track.id, u);
      onResolved(updated);
      setUrl("");
    } catch (e) {
      toast(`Resolve failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mix-resolve">
      <input
        placeholder="paste track link…"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
      />
      <button className="mini-btn" onClick={submit} disabled={busy || !url.trim()}>
        {busy ? "…" : "link"}
      </button>
    </div>
  );
}

const fmtPlays = (n) =>
  !n ? "" : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M plays`
    : n >= 1e3 ? `${Math.round(n / 1e3)}k plays` : `${n} plays`;

const fmtDur = (secs) => {
  if (!secs) return "";
  const s = Math.round(secs);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

// The ranked search hits behind a link. Auto-link picks the top one; when it
// picks wrong the right answer is usually right here, so let it be chosen
// directly instead of making the user hunt down a URL to paste. Normally served
// from what auto-link already fetched (track.has_candidates), so it's instant.
function CandidatePicker({ track, platform, onResolved }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [hits, setHits] = useState(null);
  const [cached, setCached] = useState(false);
  const [picking, setPicking] = useState(null);

  // 'both' is a resolve strategy, not a searchable platform — show SoundCloud,
  // which is what 'both' tries first and what we prefer for audio anyway.
  const searchPlatform = platform === "youtube" ? "youtube" : "soundcloud";

  const load = async (refresh = false) => {
    setLoading(true);
    try {
      const d = await api.mixTrackCandidates(track.id, searchPlatform, 5, refresh);
      setHits(d.candidates || []);
      setCached(!!d.cached);
      return true;
    } catch (e) {
      toast(`Search failed: ${e.message}`);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const toggle = async () => {
    if (open) { setOpen(false); return; }
    setOpen(true);
    if (hits) return;
    if (!(await load())) setOpen(false);
  };

  const pick = async (url) => {
    setPicking(url);
    try {
      onResolved(await api.resolveMixTrack(track.id, url));
    } catch (e) {
      toast(`Link failed: ${e.message}`);
    } finally {
      setPicking(null);
    }
  };

  return (
    <>
      <button className="mini-btn" onClick={toggle}
        title={track.has_candidates
          ? `Show the ${searchPlatform} results auto-link found for this track`
          : `Search ${searchPlatform} for other matches`}>
        {open ? "▴ hide" : "▾ others"}
      </button>
      {open && (
        <div className="mix-candidates">
          {loading && <span className="faint">searching {searchPlatform}…</span>}
          {!loading && hits && hits.length === 0 && (
            <span className="faint">No results — try pasting a link.</span>
          )}
          {!loading && hits && (
            <div className="mix-candidates-head faint">
              <span>{cached ? "from the last search" : `live ${searchPlatform} search`}</span>
              <button className="mini-btn" onClick={() => load(true)}
                title="Search the platform again instead of reusing the last results">
                ↻ search again
              </button>
            </div>
          )}
          {!loading && (hits || []).map((h) => (
            <button
              key={h.url}
              className={`mix-candidate${h.url === track.resolved_url ? " current" : ""}`}
              onClick={() => pick(h.url)}
              disabled={!!picking}
              title={h.url}
            >
              <span className="mix-candidate-title">{h.title || h.url}</span>
              <span className="faint">
                {[h.uploader, fmtDur(h.duration_secs), fmtPlays(h.playback_count),
                  `match ${Math.round((h.score || 0) * 100)}%`]
                  .filter(Boolean).join(" · ")}
                {h.url === track.resolved_url ? " · current" : ""}
              </span>
            </button>
          ))}
        </div>
      )}
    </>
  );
}

export function MixImporter() {
  const [mixes, setMixes] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState(null);
  const [platform, setPlatform] = useState("both");
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [viewMode, setViewMode] = useState("list"); // 'list' | 'match'
  const [newTrack, setNewTrack] = useState({ artist: "", title: "", link: "" });
  const [addingTrack, setAddingTrack] = useState(false);
  const [resolveJobId, setResolveJobId] = useState(null);
  const { job: resolveJob } = useJobPolling(resolveJobId);
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }));

  const onReorder = async ({ active, over }) => {
    if (!over || active.id === over.id || !detail) return;
    const ids = detail.tracks.map((t) => t.id);
    const next = arrayMove(ids, ids.indexOf(active.id), ids.indexOf(over.id));
    const byId = Object.fromEntries(detail.tracks.map((t) => [t.id, t]));
    setDetail({ ...detail, tracks: next.map((id, i) => ({ ...byId[id], idx: i })) });
    try {
      setDetail(await api.reorderMixTracks(detail.id, next));
    } catch (e) {
      toast(`Reorder failed: ${e.message}`);
      api.getMix(detail.id).then(setDetail).catch(() => {});
    }
  };

  const loadMixes = () =>
    api.getMixes().then((d) => setMixes(d.mixes)).catch((e) => setError(e.message));

  useEffect(() => { loadMixes(); }, []);

  useEffect(() => {
    setDetail(null);
    setSelectedIds(new Set());   // fresh selection per mix
    if (activeId == null) return;
    api.getMix(activeId).then(setDetail).catch((e) => setError(e.message));
  }, [activeId]);

  const toggleSelected = (id) =>
    setSelectedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  // Anything not yet in the library can be selected. One selection drives all
  // three bulk actions; each derives its own eligible subset below, so a button's
  // count always means what it says.
  const tracks = detail?.tracks || [];
  const idsWhere = (pred) => tracks.filter(pred).map((t) => t.id);
  const selected = (ids) => ids.filter((id) => selectedIds.has(id));

  const selectableIds = idsWhere((t) => !t.song_id);
  const allSelected = selectableIds.length > 0 &&
    selected(selectableIds).length === selectableIds.length;

  const selectedLinkable = selected(idsWhere((t) => !t.song_id && !t.resolved_url));
  const selectedRelinkable = selected(
    idsWhere((t) => !t.song_id && t.resolved_url && t.resolve_status === "auto"));
  const selectedUnlinkable = selected(idsWhere((t) => !t.song_id && t.resolved_url));

  // Auto-linked but low confidence, and not yet ingested — what "Re-link" falls
  // back to when nothing is selected (manual/scraped links are off-limits).
  const flaggedIds = idsWhere((t) => !t.song_id && t.resolved_url &&
                                     t.resolve_status === "auto" && !t.trusted);
  const relinkIds = selectedRelinkable.length ? selectedRelinkable : flaggedIds;

  const toggleSelectAll = () =>
    setSelectedIds(allSelected ? new Set() : new Set(selectableIds));

  // When an auto-resolve job finishes, pull the freshly-linked tracks back in.
  useEffect(() => {
    if (!resolveJob || !detail) return;
    if (resolveJob.status === "completed") {
      const r = resolveJob.result || {};
      toast(`${r.relink ? "Re-linked" : "Auto-linked"} ${r.resolved ?? 0} ` +
            `track${r.resolved === 1 ? "" : "s"} on ${r.platform || platform}` +
            (r.failed ? ` · ${r.failed} not found` : ""));
      api.getMix(detail.id).then(setDetail).catch((e) => setError(e.message));
      loadMixes();
      setResolveJobId(null);
    } else if (resolveJob.status === "failed") {
      setError(resolveJob.error || "Auto-resolve failed");
      setResolveJobId(null);
    }
  }, [resolveJob]); // eslint-disable-line react-hooks/exhaustive-deps

  const importUrl = async () => {
    setError(null);
    setBusy(true);
    try {
      const mix = await api.importMix(url.trim());
      toast(`Imported “${mix.title}” (${mix.track_count} tracks)`);
      setUrl("");
      await loadMixes();
      setActiveId(mix.id);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const autoResolve = async () => {
    if (!detail) return;
    setError(null);
    try {
      const res = await api.autoResolveMix(detail.id, platform, selectedLinkable);
      setResolveJobId(res.job_id);
      toast(`Searching ${res.platform} for ${res.queued} selected track${res.queued === 1 ? "" : "s"}…`);
    } catch (e) {
      setError(e.message);
    }
  };

  // Re-search links a previous auto-link got wrong. `trackIds` narrows it to one
  // row (the per-row ↻); omitted, it uses the selection, or every flagged row.
  const relink = async (trackIds = relinkIds) => {
    if (!detail || !trackIds.length) return;
    setError(null);
    try {
      const res = await api.autoResolveMix(detail.id, platform, trackIds, true);
      setResolveJobId(res.job_id);
      toast(`Re-linking ${res.queued} track${res.queued === 1 ? "" : "s"} on ${res.platform}…`);
    } catch (e) {
      setError(e.message);
    }
  };

  // Clear links so the tracks go back to "needs link". Destructive and there's no
  // undo, so anything beyond a single row asks first.
  const unlink = async (trackIds = selectedUnlinkable) => {
    if (!detail || !trackIds.length) return;
    if (trackIds.length > 1 &&
        !window.confirm(`Remove the links from ${trackIds.length} tracks? ` +
                        `They'll go back to "needs link" and you can auto-link them again.`)) {
      return;
    }
    setError(null);
    try {
      const res = await api.unlinkMixTracks(detail.id, trackIds);
      setDetail(res);
      setSelectedIds(new Set());
      toast(`Unlinked ${res.unlinked} track${res.unlinked === 1 ? "" : "s"}` +
            (res.skipped_ingested
              ? ` · ${res.skipped_ingested} already in the library, left alone` : ""));
      loadMixes();
    } catch (e) {
      setError(e.message);
    }
  };

  const resolving = !!resolveJobId &&
    (!resolveJob || resolveJob.status === "queued" || resolveJob.status === "running");

  const ingest = async () => {
    if (!detail) return;
    setIngesting(true);
    setError(null);
    try {
      const res = await api.ingestMix(detail.id);
      toast(`Auto-processing ${res.count} track${res.count === 1 ? "" : "s"} from this mix`);
      const d = await api.getMix(detail.id);
      setDetail(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setIngesting(false);
    }
  };

  const onTrackResolved = (updated) => {
    setDetail((d) => d && {
      ...d,
      resolved_count: d.tracks.filter((t) => (t.id === updated.id ? updated.resolved_url : t.resolved_url)).length,
      tracks: d.tracks.map((t) => (t.id === updated.id ? updated : t)),
    });
  };

  const addTrack = async () => {
    if (!detail) return;
    const title = newTrack.title.trim();
    if (!title) { toast("Enter a title for the track"); return; }
    const link = newTrack.link.trim();
    if (link && classifyUrl(link).source === "unknown") {
      toast("Link must be a SoundCloud or YouTube URL (or leave it blank)");
      return;
    }
    setAddingTrack(true);
    try {
      const row = await api.addMixTrack(detail.id, {
        artist: newTrack.artist.trim(), title, link });
      setDetail((d) => d && {
        ...d,
        tracks: [...d.tracks, row],
        track_count: (d.track_count ?? d.tracks.length) + 1,
        resolved_count: d.resolved_count + (row.resolved_url ? 1 : 0),
      });
      setNewTrack({ artist: "", title: "", link: "" });
      toast(link ? "Track added" : "Track added — use Auto-link to find its URL");
      loadMixes();
    } catch (e) {
      toast(`Add failed: ${e.message}`);
    } finally {
      setAddingTrack(false);
    }
  };

  const removeTrack = async (t) => {
    if (!detail) return;
    const label = `${t.artist ? `${t.artist} — ` : ""}${t.title}`;
    if (!window.confirm(`Remove “${label}” from this mix?`)) return;
    try {
      setDetail(await api.deleteMixTrack(detail.id, t.id));
      loadMixes();
    } catch (e) {
      toast(`Remove failed: ${e.message}`);
    }
  };

  // Live-refresh the ingest tracker while any of this mix's songs is still mid-
  // pipeline. Ingest runs after matching, so this rarely overlaps board editing.
  useEffect(() => {
    if (!detail) return;
    const busy = (detail.tracks || []).some((t) =>
      t.song_id && t.song_status && t.song_status !== "analysed" &&
      !String(t.song_status).startsWith("error"));
    if (!busy) return;
    const h = setInterval(() => {
      api.getMix(detail.id).then(setDetail).catch(() => {});
    }, 3000);
    return () => clearInterval(h);
  }, [detail]);

  return (
    <div className="page mixes">
      <div className="screen-head" style={{ display: "block" }}>
        <h1>Mixes — learn from real DJ sets</h1>
        <div className="hint" style={{ marginTop: 5 }}>
          Import a set's tracklist (Big Bootie, festival sets, 1001tracklists…),
          link each entry to a playable track, then ingest them into the library.
        </div>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}

      <div className="import-input-row">
        <div className="import-input">
          <span className="faint">🔗</span>
          <input
            type="url"
            placeholder="tracklist URL (e.g. a 1001tracklists set page)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>
        <button className="btn" onClick={importUrl} disabled={busy || !url.trim()}>
          {busy ? "…" : "Scrape URL"}
        </button>
      </div>
      <div className="faint" style={{ fontSize: 11, margin: "4px 0 10px" }}>
        Scraping a set page imports its whole tracklist — you can add or remove
        individual tracks afterwards.
      </div>

      <div className="mix-stack">
        <div className="mix-list-bar">
          <div className="mix-list-head">Imported mixes ({mixes.length})</div>
          {mixes.length === 0 ? (
            <div className="empty" style={{ padding: "6px 2px" }}>None yet.</div>
          ) : (
            <div className="mix-list-items">
              {mixes.map((m) => (
                <div
                  key={m.id}
                  className={`mix-list-item${m.id === activeId ? " active" : ""}`}
                  onClick={() => setActiveId(m.id)}
                >
                  <div className="mix-list-title">{m.title}</div>
                  <div className="faint" style={{ fontSize: 11 }}>
                    {m.track_count} tracks · {m.resolved_count} linked
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="mix-detail">
          {!detail ? (
            <p className="empty">Select a mix to see its tracklist.</p>
          ) : (
            <>
              <div className="mix-detail-head">
                <div>
                  <div style={{ fontWeight: 600 }}>{detail.title}</div>
                  <div className="faint" style={{ fontSize: 11 }}>
                    {detail.track_count} tracks · {detail.resolved_count} linked
                    {detail.source_url ? <> · <a href={detail.source_url} target="_blank" rel="noreferrer">source</a></> : null}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <div className="seg-toggle" role="tablist" aria-label="Mix view">
                    <button
                      className={`seg${viewMode === "list" ? " active" : ""}`}
                      role="tab" aria-selected={viewMode === "list"}
                      onClick={() => setViewMode("list")}
                    >
                      Tracklist
                    </button>
                    <button
                      className={`seg${viewMode === "match" ? " active" : ""}`}
                      role="tab" aria-selected={viewMode === "match"}
                      onClick={() => setViewMode("match")}
                      title="Match vocals to the instrumentals they were mashed over"
                    >
                      Match {detail.match_count ? `(${detail.match_count})` : ""}
                    </button>
                  </div>
                  {viewMode === "match" && (
                    <span className={`match-checkpoint${detail.match_count ? " ok" : ""}`}
                      title="Matches save automatically as you arrange them — you can leave and resume later, then Ingest.">
                      {detail.match_count
                        ? `✓ ${detail.match_count} match${detail.match_count === 1 ? "" : "es"} saved`
                        : "no matches yet"}
                    </span>
                  )}
                  {viewMode === "list" && selectableIds.length > 0 && (
                    <label className="faint" style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}
                      title="Select / deselect every track that isn't in the library yet">
                      <input type="checkbox" checked={allSelected}
                        onChange={toggleSelectAll} disabled={resolving} />
                      All ({selected(selectableIds).length}/{selectableIds.length})
                    </label>
                  )}
                  <select
                    className="mini-select"
                    value={platform}
                    onChange={(e) => setPlatform(e.target.value)}
                    disabled={resolving}
                    title="Which platform to search for links"
                  >
                    <option value="both">Both (SC → YT)</option>
                    <option value="soundcloud">SoundCloud</option>
                    <option value="youtube">YouTube</option>
                  </select>
                  <button
                    className="btn ghost"
                    onClick={autoResolve}
                    disabled={resolving || selectedLinkable.length === 0}
                    title="Search the chosen platform and auto-link the selected tracks"
                  >
                    {resolving ? "Searching…" : `⚡ Auto-link (${selectedLinkable.length})`}
                  </button>
                  {viewMode === "list" && relinkIds.length > 0 && (
                    <button
                      className="btn ghost"
                      onClick={() => relink()}
                      disabled={resolving}
                      title={selectedRelinkable.length
                        ? "Search again for the selected auto-linked tracks"
                        : "Search again for every low-confidence auto link. Links you pasted or confirmed are left alone."}
                    >
                      ↻ Re-link ({relinkIds.length}){selectedRelinkable.length ? "" : " flagged"}
                    </button>
                  )}
                  {viewMode === "list" && selectedUnlinkable.length > 0 && (
                    <button
                      className="btn ghost danger"
                      onClick={() => unlink()}
                      disabled={resolving}
                      title="Clear the links on the selected tracks, returning them to “needs link”. Tracks already in the library are left alone."
                    >
                      ✕ Unlink ({selectedUnlinkable.length})
                    </button>
                  )}
                  <button
                    className="btn"
                    onClick={ingest}
                    disabled={ingesting || detail.resolved_count === 0}
                    title="Save every linked track to the library and auto-process it"
                  >
                    {ingesting ? "Ingesting…" : `⇣ Ingest ${detail.resolved_count} linked`}
                  </button>
                </div>
              </div>
              {resolving && (
                <div className="hint" style={{ margin: "2px 0 8px" }}>
                  <span className="spin-dot" /> {resolveJob?.message || "Queued…"}
                  {typeof resolveJob?.progress === "number" && resolveJob.progress > 0
                    ? ` (${resolveJob.progress}%)` : ""}
                </div>
              )}
              {detail.ingested_count > 0 && (
                <div className="mix-ingest-tracker">
                  <div className="mix-ingest-head">
                    <strong>Processing</strong>
                    <span className="faint">
                      {detail.analysed_count}/{detail.ingested_count} analysed
                      {detail.analysed_count < detail.ingested_count
                        ? " · runs in the background, resumes automatically across sessions" : " ✓"}
                    </span>
                  </div>
                  <div className="mix-ingest-rows">
                    {detail.tracks.filter((t) => t.song_id).map((t) => (
                      <div key={t.id} className="mix-ingest-row">
                        <span className="mix-ingest-title" title={t.song_last_error || undefined}>
                          {t.artist ? `${t.artist} — ` : ""}{t.title}
                        </span>
                        <StageStepper status={t.song_status} error={t.song_last_error} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {viewMode === "match" && (
                <MixMatchBoard mix={detail} onMixUpdated={setDetail} />
              )}
              <div className="mix-rows" style={viewMode === "match" ? { display: "none" } : undefined}>
                <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onReorder}>
                <SortableContext items={detail.tracks.map((t) => t.id)} strategy={verticalListSortingStrategy}>
                {detail.tracks.map((t) => (
                  <SortableRow key={t.id} track={t}>
                    {!t.song_id ? (
                      <input
                        type="checkbox"
                        className="mix-select"
                        checked={selectedIds.has(t.id)}
                        onChange={() => toggleSelected(t.id)}
                        disabled={resolving}
                        title="Select this track for auto-link / re-link / unlink"
                      />
                    ) : (
                      <span className="mix-select" />
                    )}
                    <span className="mix-num">{t.idx + 1}</span>
                    <div className="mix-info">
                      <span className="mix-title" title={`${t.artist ? `${t.artist} — ` : ""}${t.title}`}>
                        {t.artist ? `${t.artist} — ` : ""}{t.title}
                      </span>
                      <span className="mix-tags">
                        {t.song_id ? <span className="mix-flag ok">in library #{t.song_id}</span>
                          : t.resolved_url && t.resolve_status === "auto" && t.trusted
                            ? <span className="mix-flag ok" title="Auto-found, high confidence — counts as training data">auto-linked ✓</span>
                          : t.resolved_url && t.resolve_status === "auto"
                            ? <span className="mix-flag auto" title="Auto-found but low confidence — verify it's the right track, then Confirm">auto-linked ⚠ verify</span>
                          : t.resolved_url ? <span className="mix-flag ok">linked</span>
                          : <span className="mix-flag failed">needs link</span>}
                        {t.resolved_url && (
                          <a className="mix-url" href={t.resolved_url} target="_blank"
                            rel="noreferrer" title={t.resolved_url}>
                            {t.resolved_url}
                          </a>
                        )}
                      </span>
                    </div>
                    <div className="mix-actions">
                      {!t.song_id && t.resolved_url && t.resolve_status === "auto" && !t.trusted && (
                        <button
                          className="mini-btn"
                          title="Confirm this auto-found link is correct (promotes it to trusted)"
                          onClick={async () => {
                            try {
                              const updated = await api.confirmMixTrack(t.id);
                              onTrackResolved(updated);
                            } catch (e) {
                              toast(`Confirm failed: ${e.message}`);
                            }
                          }}
                        >
                          ✓ confirm
                        </button>
                      )}
                      {!t.song_id && t.resolved_url && t.resolve_status === "auto" && (
                        <button
                          className="mini-btn"
                          disabled={resolving}
                          title="Search again for this track and replace the link"
                          onClick={() => relink([t.id])}
                        >
                          ↻
                        </button>
                      )}
                      {/* Available on every row, not just flagged ones: auto-link
                          already fetched these hits, so opening the list is free.
                          Keyed on the link so re-linking or unlinking drops the
                          panel's stale results instead of showing hits for a link
                          that's gone. */}
                      {!t.song_id && (
                        <CandidatePicker key={`cand-${t.id}-${t.resolved_url || ""}`}
                          track={t} platform={platform}
                          onResolved={onTrackResolved} />
                      )}
                      {!t.song_id && <ResolveInput track={t} onResolved={onTrackResolved} />}
                      {!t.song_id && (
                        <button className="mini-btn danger" title="Remove this track from the mix"
                          onClick={() => removeTrack(t)}>✕</button>
                      )}
                    </div>
                  </SortableRow>
                ))}
                </SortableContext>
                </DndContext>
                <div className="mix-add-track">
                  <span className="mix-add-label">+ Add track</span>
                  <input className="mix-add-in" placeholder="Artist (optional)"
                    value={newTrack.artist}
                    onChange={(e) => setNewTrack((n) => ({ ...n, artist: e.target.value }))} />
                  <input className="mix-add-in" placeholder="Title"
                    value={newTrack.title}
                    onChange={(e) => setNewTrack((n) => ({ ...n, title: e.target.value }))}
                    onKeyDown={(e) => e.key === "Enter" && addTrack()} />
                  <input className="mix-add-in wide" placeholder="SoundCloud/YouTube link (optional)"
                    value={newTrack.link}
                    onChange={(e) => setNewTrack((n) => ({ ...n, link: e.target.value }))}
                    onKeyDown={(e) => e.key === "Enter" && addTrack()} />
                  <button className="mini-btn" onClick={addTrack}
                    disabled={addingTrack || !newTrack.title.trim()}>
                    {addingTrack ? "…" : "Add"}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="mix-train-section" style={{ marginTop: 18 }}>
        <div className="screen-head" style={{ display: "block", marginBottom: 8 }}>
          <h2 style={{ margin: 0, fontSize: 15 }}>Train from these mixes</h2>
          <div className="hint" style={{ marginTop: 4 }}>
            The <code>w/</code> overlay lines in every imported mix are documented
            vocal-over-instrumental mashups. Once their tracks are ingested and
            analysed, build a dataset and train a model that scores new matches.
          </div>
        </div>
        <MlPanel />
      </div>
    </div>
  );
}
