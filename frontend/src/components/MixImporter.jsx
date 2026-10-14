import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { DndContext, PointerSensor, useSensor, useSensors, closestCenter } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable, arrayMove } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { api } from "../api";
import { GRAB_BOOKMARKLET } from "../bookmarklet/generated";
import { classifyUrl } from "../sources";
import { toast } from "../toast";
import { useJobPolling } from "../hooks/useJobPolling";
import { MixMatchBoard } from "./MixMatchBoard";
import { keyOf } from "./pairs/pairModel";
import { Disclosure, TierBadge } from "./ui/primitives";
import { fmtDur, fmtPlays } from "../theme";

// Mixes (revamp step 5): import a DJ set's tracklist (scrape, paste, or the
// bookmarklet when the scrape is walled), link each entry to SoundCloud or
// YouTube, match the vocals to the beds they went over, and ingest. The left
// column is how a mix GETS here; the main column is the mix you are working on.

// What tells a bookmarklet capture from plain pasted text: every captured row
// carries a link to the track's own 1001tracklists page, absolute or relative.
// Mirrors _TRACK_LINK_RE in ingest/firecrawl_scrape.py.
const CAPTURE_RE =
  /\]\((?:https?:\/\/(?:www\.)?1001tracklists\.com)?\/track\/|\[no track page\]/i;

// Ingest pipeline stages a song passes through, in order — the Processing
// disclosure's four dots.
const INGEST_STAGES = ["queued", "downloaded", "stemmed", "analysed"];

// A destructive action waits this long behind its Undo toast before it is
// sent — the toast's own lifetime when it carries an action.
const UNDO_MS = 7000;

const label = (t) => `${t.artist ? `${t.artist} — ` : ""}${t.title}`;
const isVocal = (t) => t.role === "vocal" || ((t.role || "unassigned") === "unassigned" && !!t.is_overlay);
const flaggedOf = (t) => !t.song_id && t.resolved_url && t.resolve_status === "auto" && !t.trusted;

// The filter chips. Each row is in exactly one of the four narrow groups.
const FILTERS = [
  ["all", "All", () => true],
  ["lib", "In library", (t) => !!t.song_id],
  ["linked", "Linked", (t) => !t.song_id && !!t.resolved_url && !flaggedOf(t)],
  ["flagged", "Check ⚠", flaggedOf],
  ["needs", "Needs link", (t) => !t.song_id && !t.resolved_url],
];

// The DJ's documented pairs, as the engine sees them: play one, open it in
// Studio, ask for pairs like it, and see whether the engine agrees — its
// score for the pair and where it ranks among that vocal's beds.
function DocumentedPairs({ pairs, player, onOpenStudio, onFindSimilar }) {
  if (!pairs?.length) return null;
  const playingKey = player?.kind === "pair" ? player.source?.key : null;
  const agreeing = pairs.filter((p) => p.engine_rank && p.engine_rank <= 3).length;
  const scored = pairs.filter((p) => p.engine).length;
  return (
    <section className="mx-doc">
      <div className="mx-sechead">
        <b>Documented pairs</b>
        <span className="muted small">
          {scored
            ? `the engine scores ${scored} of ${pairs.length}; ${agreeing} rank in that vocal’s top 3 beds`
            : "ingest and analyse both sides to hear and score them"}
        </span>
      </div>
      {pairs.map((p) => {
        const e = p.engine;
        const pct = e?.score_percentile != null ? Math.round(e.score_percentile * 100) : null;
        const k = e ? keyOf(e) : null;
        return (
          <div key={p.id} className="mx-docpair">
            <div className="mx-docnames">
              <span className="mx-side vox"><span className="tag mono">VOX</span>
                <span className="t">{p.vocal_artist ? `${p.vocal_artist} — ` : ""}{p.vocal_title}</span></span>
              <span className="mx-side bed"><span className="tag mono">BED</span>
                <span className="t">{p.inst_artist ? `${p.inst_artist} — ` : ""}{p.inst_title}</span></span>
            </div>
            <span className="mx-engine mono"
              title={e ? `${e.reason || ""}\nRank ${p.engine_rank} of ${p.engine_field} beds the engine scored for this vocal`
                : p.engine_state}>
              {e ? (
                <>
                  {e.confidence_tier && <TierBadge tier={e.confidence_tier} />}
                  <span>engine {pct}</span>
                  <span className="muted"> · #{p.engine_rank} of {p.engine_field}</span>
                </>
              ) : <span className="muted">{p.engine_state}</span>}
            </span>
            <div className="mx-docact">
              <button className="ui-btn" disabled={!e || !player} onClick={() => player.toggle({
                kind: "pair", key: k, candidate: e,
                title: e.vocal_title, subtitle: `over ${e.inst_title}`,
              })}>{k && playingKey === k ? "◍ stop" : "▶ loop"}</button>
              <button className="ui-btn" disabled={!(p.vocal_song_id && p.inst_song_id) || !onOpenStudio}
                onClick={() => onOpenStudio(e || {
                  vocal_song_id: p.vocal_song_id, inst_song_id: p.inst_song_id,
                })}>Studio</button>
              <button className="ui-btn" disabled={!p.vocal_song_id || !onFindSimilar}
                title="The pair dock, scoped to this vocal: every bed the engine likes under it"
                onClick={() => onFindSimilar(p.vocal_song_id)}>Similar →</button>
            </div>
          </div>
        );
      })}
    </section>
  );
}

function StageDots({ status, error }) {
  const s = status || "queued";
  const isErr = String(s).startsWith("error");
  const reached = isErr ? -1 : INGEST_STAGES.indexOf(s);
  const done = s === "analysed";
  return (
    <span className={`mx-stage${isErr ? " err" : done ? " done" : ""}`} title={isErr ? (error || s) : `stage: ${s}`}>
      {INGEST_STAGES.map((st, i) => <i key={st} className={i <= reached ? "on" : ""} />)}
      <span className="mono">{isErr ? "error" : s}</span>
    </span>
  );
}

// A URL that wraps at path boundaries instead of mid-word.
function BreakableUrl({ url }) {
  const parts = String(url).replace(/^https?:\/\/(www\.)?/, "").split("/");
  return (
    <>
      {parts.map((part, i) => (
        <Fragment key={i}>{i > 0 && "/"}{i > 0 && <wbr />}{part}</Fragment>
      ))}
    </>
  );
}

// How the link was arrived at and how much to trust it.
function LinkFlag({ track }) {
  if (track.song_id) return <span className="mx-flag lib" title={`In your library as #${track.song_id}`}>in library</span>;
  if (!track.resolved_url) return <span className="mx-flag needs" title="No link yet — auto-link, pick a result or paste one">needs link</span>;
  if (track.resolve_status === "auto") {
    return track.trusted
      ? <span className="mx-flag ok" title="Auto-found, high confidence — counts as training data">auto-linked ✓</span>
      : <span className="mx-flag check" title="Auto-found but low confidence — check it, then confirm">auto-linked ⚠</span>;
  }
  return (
    <span className="mx-flag ok" title={track.resolve_status === "scraped" ? "Scraped from the track’s 1001tracklists page"
      : track.resolve_status === "library" ? "Matched by name to a song in your library" : "Linked by you"}>linked</span>
  );
}

function SourceTag({ track }) {
  if (!track.resolved_url) return null;
  const source = track.link_platform || classifyUrl(track.resolved_url).source;
  if (source !== "youtube" && source !== "soundcloud") return null;
  return <span className={`mx-src ${source}`}>{source === "youtube" ? "YouTube" : "SoundCloud"}</span>;
}

// "▾ others": the ranked search hits behind a link (auto-link picks the top
// one; when it picks wrong the right answer is usually here), a box to paste a
// link, and — when the tracklist gave the track its own page — the exact link
// scraped from that page.
function OthersPanel({ track, platform, onResolved }) {
  const [loading, setLoading] = useState(true);
  const [hits, setHits] = useState(null);
  const [cached, setCached] = useState(false);
  const [picking, setPicking] = useState(null);
  const [url, setUrl] = useState("");
  const [scraping, setScraping] = useState(false);
  // 'both' is a resolve strategy, not a searchable platform — show SoundCloud,
  // which is what 'both' tries first.
  const searchPlatform = platform === "youtube" ? "youtube" : "soundcloud";

  const load = async (refresh = false) => {
    setLoading(true);
    try {
      const d = await api.mixTrackCandidates(track.id, searchPlatform, 5, refresh);
      setHits(d.candidates || []);
      setCached(!!d.cached);
    } catch (e) {
      toast(`Search failed: ${e.message}`);
      setHits([]);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { load(); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  const pick = async (u) => {
    setPicking(u);
    try { onResolved(await api.resolveMixTrack(track.id, u)); }
    catch (e) { toast(`Link failed: ${e.message}`); }
    finally { setPicking(null); }
  };
  const paste = async () => {
    const u = url.trim();
    if (!u) return;
    if (classifyUrl(u).source === "unknown") { toast("Paste a SoundCloud or YouTube track link"); return; }
    await pick(u);
    setUrl("");
  };
  const scrape = async () => {
    setScraping(true);
    try {
      onResolved(await api.scrapeMixTrackLink(track.id));
      toast("Linked from the track’s 1001tracklists page");
    } catch (e) { toast(`Scrape failed: ${e.message}`); }
    finally { setScraping(false); }
  };

  return (
    <div className="mx-others">
      <div className="mx-othershead muted small">
        <span>{loading ? `searching ${searchPlatform}…` : cached ? `from the last search · ${searchPlatform}` : `live search · ${searchPlatform}`}</span>
        <button className="ui-link" onClick={() => load(true)} disabled={loading}
          title="Search the platform again instead of reusing the last results">↻ search again</button>
        {track.tl_track_url && (
          <button className="ui-link" onClick={scrape} disabled={scraping}
            title="Read the exact link off this track’s 1001tracklists page (one Firecrawl request)">
            {scraping ? "scraping…" : "⤓ scrape its link"}
          </button>
        )}
      </div>
      {!loading && hits?.length === 0 && <span className="muted small">No results — paste a link below.</span>}
      {!loading && (hits || []).map((h) => (
        <button key={h.url} className={`mx-cand${h.url === track.resolved_url ? " current" : ""}`}
          onClick={() => pick(h.url)} disabled={!!picking} title={h.url}>
          <span className="t">{h.title || h.url}</span>
          <span className="muted">{h.uploader}</span>
          <span className="mono muted">{[fmtDur(h.duration_secs), h.playback_count ? fmtPlays(h.playback_count) : null].filter(Boolean).join(" · ")}</span>
          <span className="mono">{Math.round((h.score || 0) * 100)}%</span>
        </button>
      ))}
      <div className="mx-pasterow">
        <input className="mx-in" value={url} aria-label="Paste a track link"
          placeholder="…or paste a SoundCloud / YouTube link"
          onChange={(e) => setUrl(e.target.value)} onKeyDown={(e) => e.key === "Enter" && paste()} />
        <button className="ui-btn" onClick={paste} disabled={!!picking || !url.trim()}>Link</button>
      </div>
    </div>
  );
}

function SortableRow({ id, className, children }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  return (
    <div ref={setNodeRef} className={className}
      style={{ transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1 }}>
      <span className="mx-grip" {...attributes} {...listeners} title="Drag to reorder">⋮⋮</span>
      {children}
    </div>
  );
}

export function MixImporter({ player = null, onOpenStudio = null, onFindSimilar = null,
                              onOpenTrack = null } = {}) {
  const [mixes, setMixes] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState(null);
  // Paste is the path that does not depend on anyone letting us render their
  // page. It opens by itself when a scrape comes back walled.
  const [paste, setPaste] = useState("");
  const [showPaste, setShowPaste] = useState(false);
  const [showBm, setShowBm] = useState(false);
  const [copiedBm, setCopiedBm] = useState(false);
  // /import answers 501 for exactly one thing — a Firecrawl key would fix it —
  // so that status is what raises the key prompt.
  const [needsKey, setNeedsKey] = useState(false);
  const [fcKey, setFcKey] = useState("");
  const [platform, setPlatform] = useState("both");
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [viewMode, setViewMode] = useState("list"); // 'list' | 'match'
  const [filter, setFilter] = useState("all");
  const [openCand, setOpenCand] = useState(null);
  const [newTrack, setNewTrack] = useState({ artist: "", title: "", link: "" });
  const [addingTrack, setAddingTrack] = useState(false);
  const [resolveJobId, setResolveJobId] = useState(null);
  const { job: resolveJob } = useJobPolling(resolveJobId);
  // Import and Ingest are jobs too: a stealth render and a 200-track metadata
  // sweep are both far too long to hold a request open for.
  const [importJobId, setImportJobId] = useState(null);
  const { job: importJob, error: importPollError } = useJobPolling(importJobId);
  const [ingestJobId, setIngestJobId] = useState(null);
  const { job: ingestJob, error: ingestPollError } = useJobPolling(ingestJobId);
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }));

  // Undo instead of confirm: a removal or an unlink shows at once, and is only
  // sent when its Undo toast has run out. `pending` holds what is waiting.
  const [pending, setPending] = useState({ removed: new Set(), unlinked: new Set() });
  const pendingOps = useRef(new Map());   // op id → { timer, commit }
  const activeRef = useRef(activeId);
  activeRef.current = activeId;

  const loadMixes = () =>
    api.getMixes().then((d) => setMixes(d.mixes)).catch((e) => setError(e.message));
  useEffect(() => { loadMixes(); }, []);

  const flushPending = async () => {
    const ops = [...pendingOps.current.values()];
    pendingOps.current.clear();
    for (const op of ops) { clearTimeout(op.timer); await op.commit(); }
  };
  // Leaving the screen sends what is waiting rather than dropping it.
  useEffect(() => () => { flushPending(); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  const defer = (message, apply, revert, commit) => {
    const id = Math.random().toString(36).slice(2);
    apply();
    const run = async () => {
      pendingOps.current.delete(id);
      try { await commit(); } catch (e) { toast(`${e.message}`); revert(); }
    };
    pendingOps.current.set(id, { timer: setTimeout(run, UNDO_MS), commit: run });
    toast(message, {
      label: "Undo",
      onClick: () => {
        const op = pendingOps.current.get(id);
        if (!op) return;
        clearTimeout(op.timer);
        pendingOps.current.delete(id);
        revert();
      },
    });
  };
  const pendingAdd = (kind, ids) => setPending((p) => ({ ...p, [kind]: new Set([...p[kind], ...ids]) }));
  const pendingDrop = (kind, ids) => setPending((p) => {
    const next = new Set(p[kind]);
    ids.forEach((i) => next.delete(i));
    return { ...p, [kind]: next };
  });

  const selectMix = async (id) => {
    await flushPending();
    setActiveId(id);
  };

  useEffect(() => {
    setDetail(null);
    setSelectedIds(new Set());   // fresh selection per mix
    setOpenCand(null);
    setFilter("all");
    if (activeId == null) return;
    api.getMix(activeId).then(setDetail).catch((e) => setError(e.message));
  }, [activeId]);

  // The mix as it will be once the waiting removals and unlinks land.
  const tracks = useMemo(() => (detail?.tracks || [])
    .filter((t) => !pending.removed.has(t.id))
    .map((t) => (pending.unlinked.has(t.id) && !t.song_id
      ? { ...t, resolved_url: null, link_platform: null, resolve_status: "unresolved", trusted: false }
      : t)), [detail, pending]);

  const toggleSelected = (id) => setSelectedIds((prev) => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  // One selection drives the bulk actions; each derives its own eligible
  // subset, so a button's count always means what it says.
  const idsWhere = (pred) => tracks.filter(pred).map((t) => t.id);
  const selected = (ids) => ids.filter((id) => selectedIds.has(id));
  const selectableIds = idsWhere((t) => !t.song_id);
  const allSelected = selectableIds.length > 0 && selected(selectableIds).length === selectableIds.length;
  const selectedLinkable = selected(idsWhere((t) => !t.song_id && !t.resolved_url));
  const selectedRelinkable = selected(idsWhere((t) => !t.song_id && t.resolved_url && t.resolve_status === "auto"));
  const selectedUnlinkable = selected(idsWhere((t) => !t.song_id && t.resolved_url));
  // Auto-linked but low confidence — what Re-link falls back to when nothing
  // is selected (manual and scraped links are off-limits).
  const flaggedIds = idsWhere(flaggedOf);
  const relinkIds = selectedRelinkable.length ? selectedRelinkable : flaggedIds;
  const toggleSelectAll = () => setSelectedIds(allSelected ? new Set() : new Set(selectableIds));
  const ingestable = tracks.filter((t) => t.resolved_url && !t.song_id).length;

  // When an auto-resolve job finishes, pull the freshly-linked tracks back in.
  useEffect(() => {
    if (!resolveJob || !detail) return;
    if (resolveJob.status === "completed") {
      const r = resolveJob.result || {};
      toast(`${r.relink ? "Re-linked" : "Auto-linked"} ${r.resolved ?? 0} `
            + `track${r.resolved === 1 ? "" : "s"} on ${r.platform || platform}`
            + (r.failed ? ` · ${r.failed} not found` : ""));
      api.getMix(detail.id).then(setDetail).catch((e) => setError(e.message));
      loadMixes();
      setResolveJobId(null);
    } else if (resolveJob.status === "failed") {
      setError(resolveJob.error || "Auto-resolve failed");
      setResolveJobId(null);
    }
  }, [resolveJob]); // eslint-disable-line react-hooks/exhaustive-deps

  // The scraper says when a page is walled in its own words
  // (ingest/firecrawl_scrape.py); match on that rather than re-deriving it.
  const failImport = (msg) => {
    setError(msg);
    if (/paste the tracklist/i.test(msg || "")) setShowPaste(true);
  };

  // A capture is recognised by the per-track link; plain pasted text has none.
  const isCapture = CAPTURE_RE.test(paste);

  // One box, two payloads — branching on the content beats asking which kind
  // of paste you are holding, and a capture's links are what "scrape its
  // link" needs later.
  const importPaste = async () => {
    setError(null);
    setBusy(true);
    try {
      const mix = isCapture
        ? await api.importMixMarkdown(paste, url.trim())
        : await api.importMixPaste(paste, url.trim());
      await finishImport(mix);
      setPaste("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const finishImport = async (mix) => {
    toast(`Imported “${mix.title}” (${mix.track_count} tracks)`);
    setUrl("");
    await loadMixes();
    await selectMix(mix.id);
  };

  const importUrl = async (refresh = false) => {
    setError(null);
    setNeedsKey(false);
    setBusy(true);
    try {
      const res = await api.importMix(url.trim(), refresh);
      if (res.job_id) {
        setImportJobId(res.job_id);
        return;   // `busy` is cleared by the job effect, not here
      }
      await finishImport(res);
    } catch (e) {
      failImport(e.message);
      setNeedsKey(e.message.startsWith("501"));
      setBusy(false);
      return;
    }
    setBusy(false);
  };

  useEffect(() => {
    if (!importPollError) return;
    setError(importPollError);
    setImportJobId(null);
    setBusy(false);
  }, [importPollError]);

  useEffect(() => {
    if (!ingestPollError) return;
    setError(ingestPollError);
    setIngestJobId(null);
    setIngesting(false);
  }, [ingestPollError]);

  // The scrape job. A bad Firecrawl key is only discovered once the job runs,
  // so the key prompt is raised off result.needs_key here too.
  useEffect(() => {
    if (!importJob) return;
    if (importJob.status === "completed") {
      const r = importJob.result || {};
      setImportJobId(null);
      setBusy(false);
      api.getMix(r.mix_id).then((mix) => finishImport(mix)).catch((e) => setError(e.message));
    } else if (importJob.status === "failed") {
      failImport(importJob.error || "Tracklist scrape failed");
      setNeedsKey(!!(importJob.result || {}).needs_key);
      setImportJobId(null);
      setBusy(false);
    }
  }, [importJob]); // eslint-disable-line react-hooks/exhaustive-deps

  // The key is read live server-side, so the retry needs no restart. It leaves
  // component state as soon as it is saved, and is never echoed back.
  const saveKeyAndRetry = async () => {
    const key = fcKey.trim();
    if (!key || !url.trim()) return;
    setError(null);
    setBusy(true);
    try {
      await api.saveSettings({ firecrawl_api_key: key });
    } catch (e) {
      setError(e.message);
      setBusy(false);
      return;
    }
    setFcKey("");
    await importUrl();
  };

  const autoResolve = async () => {
    if (!detail || !selectedLinkable.length) return;
    setError(null);
    await flushPending();
    try {
      const res = await api.autoResolveMix(detail.id, platform, selectedLinkable);
      setResolveJobId(res.job_id);
      toast(`Searching ${res.platform} for ${res.queued} selected track${res.queued === 1 ? "" : "s"}…`);
    } catch (e) {
      setError(e.message);
    }
  };

  // Tracks you already own, linked by name with no search at all.
  const findInLibrary = async () => {
    if (!detail) return;
    setError(null);
    await flushPending();
    try {
      const res = await api.matchMixToLibrary(detail.id);
      setDetail(res);
      toast(res.linked ? `Linked ${res.linked} track${res.linked === 1 ? "" : "s"} already in your library`
        : "No other tracks of this mix are in your library");
      loadMixes();
    } catch (e) {
      setError(e.message);
    }
  };

  // Re-search links a previous auto-link got wrong. `trackIds` narrows it to
  // one row (the per-row ↻); omitted, the selection, or every flagged row.
  const relink = async (trackIds = relinkIds) => {
    if (!detail || !trackIds.length) return;
    setError(null);
    await flushPending();
    try {
      const res = await api.autoResolveMix(detail.id, platform, trackIds, true);
      setResolveJobId(res.job_id);
      toast(`Re-linking ${res.queued} track${res.queued === 1 ? "" : "s"} on ${res.platform}…`);
    } catch (e) {
      setError(e.message);
    }
  };

  // Clear links so the tracks go back to "needs link" — undoable for as long
  // as its toast shows.
  const unlink = (trackIds = selectedUnlinkable) => {
    if (!detail || !trackIds.length) return;
    const mixId = detail.id;
    setSelectedIds(new Set());
    defer(`Unlinked ${trackIds.length} track${trackIds.length === 1 ? "" : "s"}`,
      () => pendingAdd("unlinked", trackIds),
      () => pendingDrop("unlinked", trackIds),
      async () => {
        const res = await api.unlinkMixTracks(mixId, trackIds);
        if (activeRef.current === mixId) setDetail(res);
        pendingDrop("unlinked", trackIds);
        loadMixes();
      });
  };

  const removeTrack = (t) => {
    if (!detail) return;
    const mixId = detail.id;
    defer(`Removed “${label(t)}”`,
      () => pendingAdd("removed", [t.id]),
      () => pendingDrop("removed", [t.id]),
      async () => {
        const res = await api.deleteMixTrack(mixId, t.id);
        if (activeRef.current === mixId) setDetail(res);
        pendingDrop("removed", [t.id]);
        loadMixes();
      });
  };

  const resolving = !!resolveJobId &&
    (!resolveJob || resolveJob.status === "queued" || resolveJob.status === "running");

  const ingest = async () => {
    if (!detail) return;
    setIngesting(true);
    setError(null);
    await flushPending();
    try {
      const res = await api.ingestMix(detail.id);
      setIngestJobId(res.job_id);
      toast(res.queued
        ? `Saving ${res.queued} track${res.queued === 1 ? "" : "s"} to the library…`
        : "Every linked track is already in the library");
    } catch (e) {
      setError(e.message);
      setIngesting(false);
    }
  };

  // The ingest job walks the mix in batches, so `message` is the live
  // "n–m of 206" and the mix is only re-read once it is done.
  useEffect(() => {
    if (!ingestJob) return;
    if (ingestJob.status === "completed") {
      const r = ingestJob.result || {};
      toast(`Auto-processing ${r.count ?? 0} track${r.count === 1 ? "" : "s"}`
            + (r.linked_count ? ` · ${r.linked_count} already in the library` : "")
            + (r.unresolvable_count ? ` · ${r.unresolvable_count} unusable` : ""));
      setIngestJobId(null);
      setIngesting(false);
      if (detail) api.getMix(detail.id).then(setDetail).catch((e) => setError(e.message));
      loadMixes();
    } else if (ingestJob.status === "failed") {
      setError(ingestJob.error || "Ingest failed");
      setIngestJobId(null);
      setIngesting(false);
      if (detail) api.getMix(detail.id).then(setDetail).catch(() => {});
    }
  }, [ingestJob]); // eslint-disable-line react-hooks/exhaustive-deps

  const onTrackResolved = (updated) => {
    setOpenCand(null);
    setDetail((d) => d && { ...d, tracks: d.tracks.map((t) => (t.id === updated.id ? updated : t)) });
    loadMixes();
  };

  const confirm = async (t) => {
    try { onTrackResolved(await api.confirmMixTrack(t.id)); }
    catch (e) { toast(`Confirm failed: ${e.message}`); }
  };

  const onReorder = async ({ active, over }) => {
    if (!over || active.id === over.id || !detail) return;
    await flushPending();
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
      const row = await api.addMixTrack(detail.id, { artist: newTrack.artist.trim(), title, link });
      setDetail((d) => d && { ...d, tracks: [...d.tracks, row] });
      setNewTrack({ artist: "", title: "", link: "" });
      toast(link ? "Track added" : "Track added — use Auto-link to find its URL");
      loadMixes();
    } catch (e) {
      toast(`Add failed: ${e.message}`);
    } finally {
      setAddingTrack(false);
    }
  };

  // Live-refresh the Processing dots while any of this mix's songs is still
  // mid-pipeline.
  useEffect(() => {
    if (!detail) return;
    const moving = (detail.tracks || []).some((t) =>
      t.song_id && t.song_status && t.song_status !== "analysed"
      && !String(t.song_status).startsWith("error"));
    if (!moving) return;
    const h = setInterval(() => { api.getMix(detail.id).then(setDetail).catch(() => {}); }, 3000);
    return () => clearInterval(h);
  }, [detail]);

  // ── derived for the main column ─────────────────────────────────────────────
  const counts = Object.fromEntries(FILTERS.map(([id, , f]) => [id, tracks.filter(f).length]));
  const shown = tracks.filter(FILTERS.find(([id]) => id === filter)[2]);
  let bedNo = 0;
  const bedNumber = {};
  tracks.forEach((t) => { if (!isVocal(t)) bedNumber[t.id] = ++bedNo; });
  const ingested = tracks.filter((t) => t.song_id);
  const analysed = ingested.filter((t) => t.song_status === "analysed").length;
  const host = detail?.source_url ? detail.source_url.replace(/^https?:\/\/(www\.)?/, "").split("/")[0] : null;
  const jobLine = ingesting ? ingestJob : resolving ? resolveJob : null;

  return (
    <div className="mx-screen">
      {/* The importer: how a mix GETS here, not something used while working
          on one, so it sits in its own column. */}
      <aside className="mx-sidecol">
        <div className="mx-import">
          <span className="caps">Import a set</span>
          <div className="mx-row">
            <input className="mx-in" type="url" aria-label="Tracklist URL"
              placeholder="1001tracklists / set page URL"
              value={url} onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && url.trim()) importUrl(); }} />
            <button className="btn-primary" onClick={() => importUrl()} disabled={busy || !url.trim()}
              title="Scrape the tracklist">{busy && !paste ? "…" : "Scrape"}</button>
          </div>
          {busy && importJob && <span className="mx-progress small">◍ {importJob.message || "Queued…"}</span>}
          <div className="mx-row">
            <button className="ui-link" aria-expanded={showPaste} onClick={() => setShowPaste((v) => !v)}>
              {showPaste ? "Hide paste box" : "Paste a tracklist"}
            </button>
            <button className="ui-link" aria-expanded={showBm} onClick={() => setShowBm((v) => !v)}>
              Scrape blocked?
            </button>
          </div>
          {showPaste && (
            <>
              <textarea className="mx-paste" rows={6} value={paste} aria-label="Pasted tracklist"
                onChange={(e) => setPaste(e.target.value)}
                placeholder={"1. Artist - Title\n2. [12:34] Artist - Title\n   w/ Artist - Title"} />
              <button className="ui-btn" onClick={importPaste} disabled={busy || !paste.trim()}>
                {busy ? "Parsing…" : isCapture ? "Import captured tracklist" : "Parse pasted tracklist"}
              </button>
              <span className="muted small">
                {isCapture
                  ? "A bookmarklet capture — each track keeps its 1001tracklists link and cue time."
                  : "Numbered lines are beds; w/ lines are the vocals over them."}
                {" "}Fill in the URL too and this stays the same mix — re-pasting replaces its
                tracks and keeps the links you already resolved.
              </span>
            </>
          )}
          {/* The way in when the scrape is blocked: your browser renders these
              pages because you are a person, and the bookmarklet reads the
              tracklist out of the page you are already looking at. */}
          {showBm && (
            <>
              <div className="mx-row">
                <a className="mx-bookmarklet" href={GRAB_BOOKMARKLET} onClick={(e) => e.preventDefault()}
                  title="Drag me to your bookmarks bar">⤓ Grab tracklist</a>
                <button className="ui-btn" onClick={() => {
                  navigator.clipboard.writeText(GRAB_BOOKMARKLET)
                    .then(() => setCopiedBm(true), () => setCopiedBm(false));
                }}>{copiedBm ? "Copied ✓" : "Copy"}</button>
              </div>
              <span className="muted small">
                Drag it to your bookmarks bar. On a set page, click it, copy what it found and paste it above.
              </span>
            </>
          )}
          {error && <div className="error-text small">{error}</div>}
          {needsKey && (
            <div className="mx-key">
              <span className="caps">Firecrawl API key</span>
              <input className="mx-in" type="password" autoComplete="off" aria-label="Firecrawl API key"
                placeholder="fc-…" value={fcKey}
                onChange={(e) => setFcKey(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && fcKey.trim()) saveKeyAndRetry(); }} />
              <button className="ui-btn" onClick={saveKeyAndRetry} disabled={busy || !fcKey.trim() || !url.trim()}>
                {busy ? "Scraping…" : "Save key & retry"}
              </button>
              <span className="muted small">
                Get one at <a href="https://www.firecrawl.dev/app/api-keys" target="_blank" rel="noreferrer">firecrawl.dev</a>.
                Saved on this machine; a set costs about 9 credits to scrape.
              </span>
            </div>
          )}
        </div>

        <div className="mx-listhead"><b>Mixes</b><span className="mono muted">{mixes.length}</span></div>
        <div className="mx-cards">
          {mixes.length === 0 && (
            <div className="muted small mx-pad">No sets yet. Import one above — a scrape brings the whole running order.</div>
          )}
          {mixes.map((m) => {
            const s = m.summary || {};
            const n = s.tracks || m.track_count || 0;
            const lib = s.library != null ? s.ingested || 0 : 0;
            const linked = Math.max(0, (s.linked ?? m.resolved_count ?? 0) - lib);
            const w = (k) => (n ? `${Math.round((k / n) * 100)}%` : "0%");
            return (
              <button key={m.id} className={`mx-card${m.id === activeId ? " on" : ""}`}
                aria-current={m.id === activeId} onClick={() => selectMix(m.id)}>
                <span className="t">{m.title}</span>
                <span className="mono m">
                  <span>{n} tracks</span>
                  <span className="lnk">{linked + lib} linked</span>
                  <span className="lib">{lib} in library</span>
                </span>
                <span className="mx-bar"><i className="lib" style={{ width: w(lib) }} /><i className="lnk" style={{ width: w(linked) }} /></span>
              </button>
            );
          })}
        </div>
      </aside>

      <main className="mx-main">
        {!detail ? (
          <div className="mx-empty">
            <b>{activeId == null ? "Pick a mix, or import one." : "Loading the mix…"}</b>
            <span className="muted">Numbered entries are beds; w/ lines are the vocals the DJ laid over
              them. Link them, match them, then Ingest — each documented pair becomes training data.</span>
          </div>
        ) : (
          <>
            <header className="mx-head">
              <div className="mx-title">
                <h1>{detail.title}</h1>
                <span className="mono muted small">
                  {tracks.length} tracks · {tracks.filter((t) => t.resolved_url || t.song_id).length} linked
                  · {ingested.length} in library · {detail.match_count || 0} documented pairs
                  {host && <> · <a href={detail.source_url} target="_blank" rel="noreferrer">{host}</a></>}
                </span>
              </div>
              <div className="d-tabs" role="tablist" aria-label="Mix view">
                <button role="tab" aria-selected={viewMode === "list"} className={viewMode === "list" ? "on" : ""}
                  title="Every track and its link" onClick={() => setViewMode("list")}>Tracklist</button>
                <button role="tab" aria-selected={viewMode === "match"} className={viewMode === "match" ? "on" : ""}
                  title="Match vocals to the beds they were mashed over" onClick={() => setViewMode("match")}>
                  Match{detail.match_count ? ` (${detail.match_count})` : ""}
                </button>
              </div>
              <button className="btn-primary" onClick={ingest} disabled={ingesting || !ingestable}
                title="Save every linked track to the library and auto-process it">
                {ingesting ? "Ingesting…" : ingestable ? `⇣ Ingest ${ingestable} linked`
                  : ingested.length ? "All linked tracks in library" : "⇣ Ingest — link tracks first"}
              </button>
            </header>

            {viewMode === "list" && (
              <div className="mx-toolbar">
                <button className="ui-btn" onClick={toggleSelectAll} disabled={resolving || !selectableIds.length}
                  title="Select every track not in the library yet">
                  <span className={`cbox${allSelected ? " on" : ""}`}>{allSelected ? "✓" : ""}</span>
                  All ({selected(selectableIds).length}/{selectableIds.length})
                </button>
                <select className="mx-select" value={platform} aria-label="Search platform"
                  onChange={(e) => setPlatform(e.target.value)} disabled={resolving}>
                  <option value="both">Both (SC → YT)</option>
                  <option value="soundcloud">SoundCloud</option>
                  <option value="youtube">YouTube</option>
                </select>
                <button className="ui-btn" onClick={findInLibrary} disabled={resolving}
                  title="Link rows to songs already in your library, by artist and title — no search, no network">
                  Find in library
                </button>
                <button className="ui-btn" onClick={autoResolve} disabled={resolving || !selectedLinkable.length}
                  title={selectedLinkable.length ? "Search the chosen platform and auto-link the selected tracks"
                    : "Select tracks without a link first"}>
                  {resolving ? "Searching…" : `⚡ Auto-link (${selectedLinkable.length})`}
                </button>
                <button className="ui-btn" onClick={() => relink()} disabled={resolving || !relinkIds.length}
                  title="Search again for low-confidence auto links. Links you pasted or confirmed are left alone.">
                  ↻ Re-link ({relinkIds.length}){selectedRelinkable.length ? "" : " flagged"}
                </button>
                <button className="ui-btn danger" onClick={() => unlink()} disabled={resolving || !selectedUnlinkable.length}
                  title="Clear the links on the selected tracks, returning them to “needs link”. Undo on the toast.">
                  ✕ Unlink ({selectedUnlinkable.length})
                </button>
                <span className="grow" />
                <div className="mx-chips" role="group" aria-label="Show">
                  {FILTERS.map(([id, text]) => (
                    <button key={id} className={`mx-chip ${id}${filter === id ? " on" : ""}`}
                      aria-pressed={filter === id} title="Show only these"
                      onClick={() => setFilter(filter === id && id !== "all" ? "all" : id)}>
                      {text}<span className="mono">{counts[id]}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {jobLine && (
              <div className="mx-busy small">
                <span className="bar"><i style={{ width: `${Math.max(4, Number(jobLine.progress) || 0)}%` }} /></span>
                {jobLine.message || "Queued…"}
              </div>
            )}

            {viewMode === "match" ? (
              <MixMatchBoard mix={detail} onMixUpdated={setDetail} />
            ) : (
              <div className="mx-scroll">
                {ingested.length > 0 && (
                  <div className="mx-pad">
                    <Disclosure id="processing" storageKey="mashup.mixes.v1" title="Processing"
                      meta={`${analysed}/${ingested.length} analysed${analysed < ingested.length
                        ? " · runs in the background and resumes across sessions" : " ✓"}`}>
                      <div className="mx-proc">
                        {ingested.map((t) => (
                          <div key={t.id} className="mx-procrow" title={t.song_last_error || undefined}>
                            <span className="t">{label(t)}</span>
                            <StageDots status={t.song_status} error={t.song_last_error} />
                          </div>
                        ))}
                      </div>
                    </Disclosure>
                  </div>
                )}

                <DocumentedPairs pairs={detail.pairs} player={player}
                  onOpenStudio={onOpenStudio} onFindSimilar={onFindSimilar} />

                <div className="mx-table">
                  <div className="mx-thead caps">
                    <span /><span /><span>#</span><span>Track · link</span><span>Status</span><span>Source</span><span />
                  </div>
                  <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onReorder}>
                    <SortableContext items={shown.map((t) => t.id)} strategy={verticalListSortingStrategy}>
                      {shown.map((t) => {
                        const vocal = isVocal(t);
                        const open = openCand === t.id;
                        return (
                          <div key={t.id} className="mx-rowwrap">
                            <SortableRow id={t.id}
                              className={`mx-trow${vocal ? " vocal" : ""}${selectedIds.has(t.id) ? " picked" : ""}`}>
                              {!t.song_id ? (
                                <button className={`cbox${selectedIds.has(t.id) ? " on" : ""}`} aria-label="Select"
                                  disabled={resolving} onClick={() => toggleSelected(t.id)}
                                  title="Select for auto-link / re-link / unlink">{selectedIds.has(t.id) ? "✓" : ""}</button>
                              ) : <span />}
                              <span className="mono muted">{vocal ? "" : bedNumber[t.id]}</span>
                              <div className="mx-track">
                                <span className="mx-name" title={t.raw_label || label(t)}>
                                  {vocal && <span className="mx-wv mono" title="A w/ line — the vocal over the bed above">w/</span>}
                                  {t.cue_secs != null && <span className="mono muted">{fmtDur(t.cue_secs) || "0:00"}</span>}
                                  {t.is_id ? <span className="mx-id mono">ID</span> : null}
                                  <span className="t">{label(t)}</span>
                                </span>
                                {t.resolved_url
                                  ? <a className="mx-url mono" href={t.resolved_url} target="_blank" rel="noreferrer"
                                      title={t.resolved_url}><BreakableUrl url={t.resolved_url} /></a>
                                  : <span className="mx-url mono none">no link yet</span>}
                              </div>
                              <LinkFlag track={t} />
                              <SourceTag track={t} />
                              <div className="mx-act">
                                {flaggedOf(t) && (
                                  <button className="ui-btn" onClick={() => confirm(t)}
                                    title="Confirm this auto-found link is right (promotes it to trusted)">✓ confirm</button>
                                )}
                                {!t.song_id && t.resolved_url && t.resolve_status === "auto" && (
                                  <button className="ui-btn" disabled={resolving} aria-label="Re-link"
                                    title="Search again and replace the link" onClick={() => relink([t.id])}>↻</button>
                                )}
                                {!t.song_id && (
                                  <>
                                    <button className={`ui-btn${open ? " on" : ""}`} aria-expanded={open}
                                      title="Other search results, or paste a link"
                                      onClick={() => setOpenCand(open ? null : t.id)}>{open ? "▴ hide" : "▾ others"}</button>
                                    <button className="ui-btn danger" aria-label="Remove from the mix"
                                      title="Remove from the mix (Undo on the toast)" onClick={() => removeTrack(t)}>✕</button>
                                  </>
                                )}
                                {t.song_id && onOpenTrack && (
                                  <button className="ui-btn" onClick={() => onOpenTrack(t.song_id)}>Open track</button>
                                )}
                              </div>
                            </SortableRow>
                            {open && (
                              <OthersPanel key={`cand-${t.id}-${t.resolved_url || ""}`} track={t}
                                platform={platform} onResolved={onTrackResolved} />
                            )}
                          </div>
                        );
                      })}
                    </SortableContext>
                  </DndContext>
                  {!shown.length && <div className="muted small mx-pad">Nothing in this filter.</div>}
                  <div className="mx-add">
                    <span className="muted small">+ Add track</span>
                    <input className="mx-in" placeholder="Artist (optional)" aria-label="Artist"
                      value={newTrack.artist}
                      onChange={(e) => setNewTrack((n) => ({ ...n, artist: e.target.value }))} />
                    <input className="mx-in" placeholder="Title" aria-label="Title" value={newTrack.title}
                      onChange={(e) => setNewTrack((n) => ({ ...n, title: e.target.value }))}
                      onKeyDown={(e) => e.key === "Enter" && addTrack()} />
                    <input className="mx-in wide" placeholder="SoundCloud / YouTube link (optional)" aria-label="Link"
                      value={newTrack.link}
                      onChange={(e) => setNewTrack((n) => ({ ...n, link: e.target.value }))}
                      onKeyDown={(e) => e.key === "Enter" && addTrack()} />
                    <button className="ui-btn" onClick={addTrack} disabled={addingTrack || !newTrack.title.trim()}>
                      {addingTrack ? "…" : "Add"}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
