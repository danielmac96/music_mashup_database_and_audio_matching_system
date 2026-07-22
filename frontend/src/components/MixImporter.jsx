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

// Mixes tab: import a DJ-set tracklist (paste the text — URL scraping needs the
// optional playwright stack), resolve each entry to a SoundCloud/YouTube link,
// then ingest the resolved tracks through the normal pipeline.

function ResolveInput({ track, onResolved }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [scraping, setScraping] = useState(false);

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

  const scrapeLink = async () => {
    setScraping(true);
    try {
      onResolved(await api.scrapeMixTrackLink(track.id));
    } catch (e) {
      toast(`Official link failed: ${e.message}`);
    } finally {
      setScraping(false);
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
      {track.tl_track_url ? (
        <button
          className="mini-btn"
          onClick={scrapeLink}
          disabled={scraping}
          title="Scrape the real SoundCloud/YouTube URL from the 1001tracklists track page"
        >
          {scraping ? "…" : "🔗 Official link"}
        </button>
      ) : null}
    </div>
  );
}

export function MixImporter() {
  const [mixes, setMixes] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [url, setUrl] = useState("");
  const [paste, setPaste] = useState("");
  const [showPaste, setShowPaste] = useState(true);
  const [busy, setBusy] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState(null);
  const [platform, setPlatform] = useState("both");
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [viewMode, setViewMode] = useState("list"); // 'list' | 'match'
  const [exporting, setExporting] = useState(false);
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

  // Tracks eligible for auto-linking: not yet in the library and still unlinked.
  const linkableIds = (detail?.tracks || [])
    .filter((t) => !t.song_id && !t.resolved_url)
    .map((t) => t.id);
  const selectedLinkable = linkableIds.filter((id) => selectedIds.has(id));
  const allLinkableSelected = linkableIds.length > 0 &&
    selectedLinkable.length === linkableIds.length;

  const toggleSelectAll = () =>
    setSelectedIds(allLinkableSelected ? new Set() : new Set(linkableIds));

  // When an auto-resolve job finishes, pull the freshly-linked tracks back in.
  useEffect(() => {
    if (!resolveJob || !detail) return;
    if (resolveJob.status === "completed") {
      const r = resolveJob.result || {};
      toast(`Auto-linked ${r.resolved ?? 0} track${r.resolved === 1 ? "" : "s"} on ${r.platform || platform}` +
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

  const importPaste = async () => {
    setError(null);
    setBusy(true);
    try {
      const mix = await api.importMixPaste(paste, url.trim());
      toast(`Parsed ${mix.track_count} tracks from paste`);
      setPaste("");
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

  return (
    <div className="page mid">
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
            placeholder="tracklist URL (optional — used as the mix's source link)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>
        <button className="btn" onClick={importUrl} disabled={busy || !url.trim()}>
          {busy ? "…" : "Scrape URL"}
        </button>
      </div>
      <div className="faint" style={{ fontSize: 11, margin: "4px 0 10px" }}>
        URL scraping needs the optional playwright stack —{" "}
        <button className="linklike" onClick={() => setShowPaste((s) => !s)}>
          pasting the tracklist text
        </button>{" "}
        always works.
      </div>

      {showPaste && (
        <div className="paste-panel">
          <textarea
            rows={7}
            placeholder={"Paste the tracklist here, e.g.\n\nTwo Friends Big Bootie Mix 20\n1. Artist - Title\n2. [12:34] Artist - Title\n…"}
            value={paste}
            onChange={(e) => setPaste(e.target.value)}
          />
          <div style={{ marginTop: 6, display: "flex", justifyContent: "flex-end" }}>
            <button className="btn" onClick={importPaste} disabled={busy || !paste.trim()}>
              {busy ? "Parsing…" : "Parse pasted tracklist"}
            </button>
          </div>
        </div>
      )}

      <div className="mix-layout">
        <div className="mix-list">
          <div className="mix-list-head">Imported mixes ({mixes.length})</div>
          {mixes.length === 0 && <div className="empty" style={{ padding: 14 }}>None yet.</div>}
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
                    <button
                      className="btn ghost"
                      disabled={exporting || !detail.match_count}
                      title="Write these matched pairs out as a training dataset"
                      onClick={async () => {
                        setExporting(true);
                        try {
                          const res = await api.exportMixTrainingSet(detail.id);
                          toast(`Exported ${res.n_pos} pair${res.n_pos === 1 ? "" : "s"} → ${res.name} v${res.version}`);
                        } catch (e) {
                          toast(`Export failed: ${e.message}`);
                        } finally {
                          setExporting(false);
                        }
                      }}
                    >
                      {exporting ? "Exporting…" : "⇪ Export training set"}
                    </button>
                  )}
                  {viewMode === "list" && linkableIds.length > 0 && (
                    <label className="faint" style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}
                      title="Select / deselect every unlinked track">
                      <input type="checkbox" checked={allLinkableSelected}
                        onChange={toggleSelectAll} disabled={resolving} />
                      All ({selectedLinkable.length}/{linkableIds.length})
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
              {viewMode === "match" && (
                <MixMatchBoard mix={detail} onMixUpdated={setDetail} />
              )}
              <div className="mix-rows" style={viewMode === "match" ? { display: "none" } : undefined}>
                <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onReorder}>
                <SortableContext items={detail.tracks.map((t) => t.id)} strategy={verticalListSortingStrategy}>
                {detail.tracks.map((t) => (
                  <SortableRow key={t.id} track={t}>
                    {!t.song_id && !t.resolved_url ? (
                      <input
                        type="checkbox"
                        className="mix-select"
                        checked={selectedIds.has(t.id)}
                        onChange={() => toggleSelected(t.id)}
                        disabled={resolving}
                        title="Select this track for auto-linking"
                      />
                    ) : (
                      <span className="mix-select" />
                    )}
                    <span className="mix-num">{t.idx + 1}</span>
                    <span className="mix-cue" />
                    <div className="mix-info">
                      <span className="mix-title">
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
                          <a className="faint" href={t.resolved_url} target="_blank" rel="noreferrer"
                            style={{ fontSize: 10, maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {t.resolved_url}
                          </a>
                        )}
                      </span>
                    </div>
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
                    {!t.song_id && <ResolveInput track={t} onResolved={onTrackResolved} />}
                  </SortableRow>
                ))}
                </SortableContext>
                </DndContext>
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
