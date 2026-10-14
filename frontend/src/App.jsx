import { useCallback, useEffect, useMemo, useState } from "react";
import { MixImporter } from "./components/MixImporter";
import { LibraryScreen } from "./components/LibraryScreen";
import { QueueScreen } from "./components/QueueScreen";
import { AnalysisScreen } from "./components/AnalysisScreen";
import { PairDock } from "./components/PairDock";
import { scoredOptionOf } from "./components/pairs/pairModel";
import { PlayerBar } from "./components/PlayerBar";
import { TrackDetail } from "./components/TrackDetail";
import { Discovery } from "./components/Discovery";
import { MixStudio } from "./components/MixStudio";
import { SetScreen } from "./components/SetScreen";
import { HelpPanel } from "./components/HelpPanel";
import { DatabaseBrowser } from "./components/DatabaseBrowser";
import { TuningPanel } from "./components/TuningPanel";
import { MlPanel } from "./components/MlPanel";
import { BulkReprocess } from "./components/BulkReprocess";
import { SetupWizard } from "./components/SetupWizard";
import { NAV_ITEMS, Sidebar, usePipelineReadout, useRailCollapsed } from "./shell/Sidebar";
import { CommandPalette } from "./shell/CommandPalette";
import { Styleguide } from "./shell/Styleguide";
import { useLibrary } from "./hooks/useLibrary";
import { useLibraryGroups } from "./hooks/useLibraryGroups";
import { useRatings } from "./hooks/useRatings";
import { useAttributes } from "./hooks/useAttributes";
import { usePairDock } from "./hooks/usePairDock";
import { useSets } from "./hooks/useSets";
import { usePairNotes } from "./hooks/usePairNotes";
import { usePlayer } from "./hooks/usePlayer";
import { isActiveJob } from "./hooks/useQueue";
import { api } from "./api";
import { onToast } from "./toast";
import { useFreshness } from "./hooks/useFreshness";

// ?styleguide — a dev-build route (tokens and primitives); never in production.
const STYLEGUIDE = Boolean(import.meta.env.DEV);
const initialRoute = () => {
  try {
    if (STYLEGUIDE && new URLSearchParams(window.location.search).has("styleguide")) {
      return "styleguide";
    }
  } catch { /* no location */ }
  return "library";
};

// The global search box for a route: Discover's own, else the Library's.
const SEARCH_FOR = { discovery: "discover" };

// Statuses still waiting on a stage that prefetch can move up the line.
const PREFETCH_STATUSES = new Set(["queued", "downloaded", "stemmed"]);

// The four tabs became a sidebar. Not cosmetics: with navigation down the left,
// the Library screen has room for a permanent pair dock on the right, so
// judging a pair and browsing the library stop being two places you switch
// between. The order is still the order the work happens in — get tracks in,
// tag the documented mixes, find pairs, build them.

function Toast() {
  const [msg, setMsg] = useState("");
  const [action, setAction] = useState(null);
  useEffect(() => {
    let timer = null;
    const off = onToast((m, a) => {
      setMsg(m);
      setAction(a || null);
      clearTimeout(timer);
      // A toast that offers something stays long enough to take it.
      timer = setTimeout(() => { setMsg(""); setAction(null); }, a ? 7000 : 2600);
    });
    return () => { off(); clearTimeout(timer); };
  }, []);
  if (!msg) return null;
  return (
    <div className="toast" role="status">
      <span className="dot">●</span>
      {msg}
      {action && (
        <button className="toast-action"
          onClick={() => { action.onClick(); setMsg(""); setAction(null); }}>
          {action.label}
        </button>
      )}
    </div>
  );
}

export default function App() {
  const [route, setRouteState] = useState(initialRoute);
  const rail = useRailCollapsed();
  const [paletteOpen, setPaletteOpen] = useState(false);
  // null = still loading, true/false = configured flag from GET /api/settings.
  const [configured, setConfigured] = useState(null);
  // Pair handed to Studio from Library/Discover. `at` is bumped on every send
  // so re-sending the same pair still re-seeds.
  const [studioSeed, setStudioSeed] = useState({ vocalId: null, instId: null });
  // Right-side header status readout — each screen reports its own.
  const [headerStatus, setHeaderStatus] = useState(null); // { locked, text }
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  // "New here?" shows until help has been opened once.
  const [helpSeen, setHelpSeen] = useState(() => {
    try { return localStorage.getItem("mashup.helpSeen.v1") === "1"; } catch { return true; }
  });
  const openHelp = useCallback(() => {
    setHelpOpen(true); setHelpSeen(true);
    try { localStorage.setItem("mashup.helpSeen.v1", "1"); } catch { /* ignore */ }
  }, []);
  useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT"
        || el.isContentEditable)) return;
      if (e.key === "?") { e.preventDefault(); openHelp(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openHelp]);
  // The library row the pair dock is scoped to, and the track the detail view
  // is open on. Selecting re-scopes; opening is a separate, deliberate act.
  const [selectedTrackId, setSelectedTrackId] = useState(null);
  // Which side of a pair the selected track is being looked at as. A track can
  // be the vocal on top or the bed underneath, and the dock has to be told
  // which question you are asking.
  const [dockRole, setDockRole] = useState("vocal");
  // What the rail's per-route slot is showing. Each screen registers its own
  // block here rather than the rail knowing every screen's internals.
  const [railSlot, setRailSlot] = useState(null);
  // A SoundCloud search handed to Discover from elsewhere (Track detail's
  // "Find acapella / instrumental").
  const [discoverNav, setDiscoverNav] = useState(null);
  const discoverSearch = (q, role = "") => {
    setDiscoverNav({ kind: "search", q, role, at: Date.now() });
    setRoute("discovery");
  };

  // ONE player for the whole app, and it lives here for the same reason the
  // library and the judgements do: every screen starts audio, and four
  // independent players meant nothing could stop the other three, and the bar
  // at the bottom could only ever describe one of them.
  const player = usePlayer();

  // The library is fetched once, here, because the rail counts it, the table
  // lists it and (from the next phase) the dock scopes to a row of it. Four
  // independent fetches of the same unpaginated endpoint would be four answers
  // that can disagree while the pipeline is running.
  const library = useLibrary();
  // Judgements are loaded once, here: the library's star column, the pair dock
  // and the track screen's partners rail all read the same map, and three
  // copies of it would disagree the moment one of them posted a rating.
  const ratings = useRatings();
  const attributes = useAttributes();
  // Groups (crates, seen from the library side) are fetched once here for the
  // same reason: the rail counts them, the filter bar names them, the table is
  // narrowed by one and the row menu writes to them. A second copy would still
  // be showing the old shelf after the first one added a track to it.
  const groups = useLibraryGroups();
  // Sets (mashups in running order) — the dock, Studio and the Sets screen all
  // add to the same active set.
  const sets = useSets();
  const notes = usePairNotes();

  useEffect(() => {
    if (!settingsOpen) return undefined;
    const onKey = (e) => { if (e.key === "Escape") setSettingsOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsOpen]);

  // On load, check whether the app has been configured (first-run wizard gate).
  // If /api/settings is unreachable, assume configured so a transient error
  // does not wall off the whole UI.
  useEffect(() => {
    api.getSettings()
      .then((s) => setConfigured(Boolean(s.configured)))
      .catch(() => setConfigured(true));
  }, []);

  const setRoute = (next) => {
    setHeaderStatus(null);
    setRailSlot(null);
    setRouteState(next);
  };

  // Stable, because the Library puts it inside the header status it reports
  // from an effect — a fresh function every render would re-fire that effect
  // forever.
  const openQueue = useCallback(() => {
    setHeaderStatus(null);
    setRailSlot(null);
    setRouteState("queue");
  }, []);

  // Each send is its own instruction, not a patch over the last one: a pair
  // from the pair dock opens as a pair, and a single track from Library is added as
  // one lane to whatever is already arranged.
  const sendToStudio = (patch) => {
    setStudioSeed({ ...patch, at: Date.now() });
    setRoute("studio");
  };

  // A pair goes to Studio with both tracks in full and the suggestion marked as
  // a region — nothing is trimmed away. `scoredOption` rides along because
  // top_section_pairs is capped at six, so the row you are looking at need not
  // be among the options Studio re-fetches for itself.
  const pairToStudio = (c) => sendToStudio({
    vocalId: c.vocal_song_id,
    instId: c.inst_song_id,
    semitoneShift: c.semitone_shift ?? 0,
    recipe: c.recipe ?? null,
    vocalSectionStart: c.vocal_section_start ?? 0,
    instSectionStart: c.inst_section_start ?? 0,
    scoredOption: scoredOptionOf(c),
  });

  const dock = usePairDock({
    selectedTrackId, role: dockRole, ratings, onOpenStudio: pairToStudio, player,
    onAddToSet: sets.addPair,
  });
  // Stale pairs and the one scoring job (the dock's pill, Settings' button and
  // a toast's "Re-score now" all start it). A finished run reloads the dock and
  // the library's MASH / PAIRS columns.
  const freshness = useFreshness({
    onScored: () => { dock.reload(); library.refresh(); },
  });

  // A whole set on one Studio timeline, back to back — what "Next pair" could
  // never do, because it replaces the arrangement.
  const chainToStudio = (items) => sendToStudio({
    chain: items.map((c) => ({
      vocalId: c.vocal_song_id, instId: c.inst_song_id,
      semitoneShift: c.semitone_shift ?? 0, recipe: c.recipe ?? null,
      // On a set's tempo curve, each mashup lands at its own point on it.
      targetBpm: c.set_bpm ?? null,
      scoredOption: scoredOptionOf(c),
    })),
  });

  // "Next pair" in Studio walks the DOCK's list, so the order you are working
  // through is the order you chose there — Studio has no list of its own and
  // inventing a second one would give the two screens different ideas about
  // what comes next.
  const nextPair = (append = false) => {
    const at = Math.min(dock.rows.length - 1, dock.cursor + 1);
    const next = dock.rows[at];
    if (!next) return;
    dock.setCursor(at);
    if (!append) { pairToStudio(next); return; }
    // Append: lay the next pair after what is already arranged, to hear the
    // transition between two mashups instead of replacing the first.
    sendToStudio({ chain: [{
      vocalId: next.vocal_song_id, instId: next.inst_song_id,
      semitoneShift: next.semitone_shift ?? 0, recipe: next.recipe ?? null,
      scoredOption: scoredOptionOf(next),
    }], append: true });
  };

  // The dock owns the keyboard only while the Library screen is the one you are
  // looking at. Discover has its own model over the same rows, and two window
  // listeners racing for the space bar is exactly the bug this avoids.
  useEffect(() => dock.bindKeys(route === "library"), [dock.bindKeys, route]);

  // Studio's timeline IS its player, and it drives an engine of its own. Two
  // transports on screen would fight over the space bar and over the speakers,
  // so arriving there stops whatever the bar was playing.
  useEffect(() => {
    if (route === "studio") player.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route]);

  // Space is the app-wide play/pause everywhere the dock is not already using
  // it for its own "loop this pair".
  useEffect(() => {
    if (route === "library" || route === "studio") return undefined;
    const onKey = (e) => {
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.isContentEditable)) return;
      if (e.code !== "Space" || !player.source) return;
      e.preventDefault();
      if (player.playing) player.pause(); else player.resume();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [route, player]);

  // `/` focuses the search on screen — Discover's on Discover, the Library's
  // everywhere else (going there first).
  const focusSearch = useCallback(() => {
    const target = SEARCH_FOR[route] || "library";
    if (target === "library" && route !== "library") setRoute("library");
    setTimeout(() => document.querySelector(`[data-search="${target}"]`)?.focus(), 40);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route]);

  // ⌘K / ctrl+K toggles the palette from anywhere, typing included. `/` and
  // esc only when no field has the keyboard; esc steps back one layer
  // (palette → settings → help close themselves; the track screen → Library).
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
        return;
      }
      if (paletteOpen || e.defaultPrevented) return;
      const el = e.target;
      const typing = el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.tagName === "SELECT" || el.isContentEditable);
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "/") { e.preventDefault(); focusSearch(); }
      else if (e.key === "Escape" && route === "styleguide") setRoute("library");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paletteOpen, focusSearch, route]);

  const pipeline = usePipelineReadout(library.tracks);

  const counts = useMemo(() => {
    const active = library.pipeJobs.filter(isActiveJob).length;
    return {
      library: library.tracks.length,
      // Only while something is processing: an idle "0" is noise.
      queue: active || null,
    };
  }, [library.tracks, library.pipeJobs]);

  // Selecting a row scopes the dock; it does not navigate. Clearing it is what
  // puts the dock back on "the best pairs in the library".
  // A click selects; it never toggles. Clicking the selected row again used to
  // clear it, sending the dock back to "Best in library" while the search still
  // showed that one row (68 of 100 simulated users hit it). Clearing is the ✕
  // on the dock's scope line, or Esc on the Library.
  const selectTrack = (id) => setSelectedTrackId(id);
  const clearSelection = () => setSelectedTrackId(null);

  useEffect(() => {
    if (route !== "library" || selectedTrackId == null || settingsOpen || helpOpen) {
      return undefined;
    }
    const onKey = (e) => {
      const el = e.target;
      if (e.key !== "Escape" || (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.tagName === "SELECT" || el.isContentEditable))) return;
      setSelectedTrackId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [route, selectedTrackId, settingsOpen, helpOpen]);

  const selectedTrack = useMemo(
    () => library.tracks.find((t) => t.id === selectedTrackId) || null,
    [library.tracks, selectedTrackId],
  );

  // Selecting a track (row or detail screen) while anything is still waiting
  // on stems moves it and its likeliest partners to the front of the line.
  // Debounced so arrowing through rows sends one request, not one per row.
  const unfinished = useMemo(
    () => library.tracks.some((t) => PREFETCH_STATUSES.has(t.status)),
    [library.tracks],
  );
  useEffect(() => {
    if (selectedTrackId == null || !unfinished) return undefined;
    const t = setTimeout(() => { api.prefetchTrack(selectedTrackId).catch(() => {}); }, 600);
    return () => clearTimeout(t);
  }, [selectedTrackId, unfinished]);

  if (configured === false) {
    return (
      <div className="app-shell setup">
        <header className="topbar">
          <div className="brand">
            <span className="diamond">◈</span> Mashup Engine
          </div>
        </header>
        <SetupWizard onConfigured={() => window.location.reload()} />
      </div>
    );
  }

  return (
    <div className={`app-shell${rail.collapsed ? " rail-collapsed" : ""}`}>
      <Sidebar route={route === "track" ? "library" : route}
        onRoute={setRoute} counts={counts}
        collapsed={rail.collapsed} onToggle={rail.toggle}
        pipeline={pipeline}
        onOpenPalette={() => setPaletteOpen(true)}
        onStyleguide={STYLEGUIDE ? () => setRoute("styleguide") : null}
        settingsOpen={settingsOpen}
        onOpenSettings={() => setSettingsOpen((v) => !v)}
        judged={ratings.count}
        onHelp={openHelp} helpSeen={helpSeen}
        status={headerStatus?.text ? (
          // The screen's status readout lives in the rail, not floating over
          // the top-right of the screen — there it covered the pair dock's
          // header and Studio's export buttons.
          <div className={`float-status${headerStatus.locked ? " locked" : ""}`
              + `${headerStatus.onClick ? " clickable" : ""}`}
            onClick={headerStatus.onClick || undefined}
            title={headerStatus.onClick ? "Open the queue" : headerStatus.text}>
            {headerStatus.locked && <span className="dot pulse" />}
            <span className="txt">
              {headerStatus.locked ? `◈ ${headerStatus.text}` : headerStatus.text}
            </span>
          </div>
        ) : null}>
        {railSlot}
      </Sidebar>

      <div className="app-main">

        {route === "styleguide" && STYLEGUIDE && <Styleguide />}

        {route === "mixes" && (
          <MixImporter player={player} onOpenStudio={pairToStudio}
            onFindSimilar={(vocalId) => {
              setSelectedTrackId(vocalId); setDockRole("vocal"); setRoute("library");
            }} />
        )}
        {route === "library" && (
          <div className="lib-layout">
            <main className="lib-main">
              <LibraryScreen
                library={library}
                attributes={attributes}
                ratings={ratings}
                groups={groups}
                player={player}
                selectedId={selectedTrackId}
                onSelect={selectTrack}
                onOpen={(id) => { setSelectedTrackId(id); setRoute("track"); }}
                onRailSlot={setRailSlot}
                onStatus={setHeaderStatus}
                onOpenQueue={openQueue}
              />
            </main>
            <PairDock dock={dock} ratings={ratings}
              onAddToSet={sets.addPair} setName={sets.active?.name || null}
              notes={notes}
              scopeTitle={selectedTrack?.title || null} onClearScope={clearSelection}
              freshness={freshness}
              role={dockRole} onRole={setDockRole} />
          </div>
        )}
        {route === "queue" && (
          <QueueScreen
            library={library}
            onOpen={(id) => { setSelectedTrackId(id); setRoute("track"); }}
            onRailSlot={setRailSlot}
          />
        )}
        {route === "analysis" && (
          <AnalysisScreen attributes={attributes} onRailSlot={setRailSlot}
            tracks={library.tracks} onPlayPair={dock.play} onOpenPair={pairToStudio} />
        )}
        {route === "track" && (
          <TrackDetail
            track={selectedTrack}
            attributes={attributes}
            tracks={library.tracks}
            ratings={ratings}
            groups={groups}
            player={player}
            role={dockRole}
            onRole={setDockRole}
            onBack={() => setRoute("library")}
            onStudio={pairToStudio}
            onOpenTrack={setSelectedTrackId}
            onStatus={setHeaderStatus}
            onChanged={() => library.refresh(true)}
            onDiscover={discoverSearch}
            onLinesChanged={dock.reload}
          />
        )}
        {route === "discovery" && (
          <Discovery
            player={player}
            onGroupsChanged={groups.refresh}
            onStatus={setHeaderStatus}
            onOpenLibrary={() => setRoute("library")}
            onRailSlot={setRailSlot}
            externalNav={discoverNav}
          />
        )}
        {route === "sets" && (
          <SetScreen sets={sets} player={player}
            onOpenStudio={pairToStudio} onOpenChain={chainToStudio}
            onRailSlot={setRailSlot} onOpenLibrary={() => setRoute("library")} />
        )}
        {route === "studio" && (
          <MixStudio
            seed={studioSeed}
            onSeedConsumed={() => setStudioSeed({ vocalId: null, instId: null })}
            onStatus={setHeaderStatus}
            onNextPair={dock.rows.length > 1 ? () => nextPair(false) : null}
            onAppendNext={dock.rows.length > 1 ? () => nextPair(true) : null}
            onAddToSet={sets.addPair} notes={notes} setName={sets.active?.name || null}
          />
        )}

        {/* Outside the route switch on purpose. setRoute clears the header
            status and the rail slot; the player is the one thing that must
            survive navigation, because "it stopped when I changed tab" is the
            bug. Studio is the exception — it has a transport of its own. */}
        {route !== "studio" && (
          <PlayerBar player={player} ratings={ratings}
            onStudio={(c) => dock.openStudio(c)} />
        )}
      </div>

      {settingsOpen && (
        <>
          <div className="drawer-scrim" onClick={() => setSettingsOpen(false)} />
          <aside className="settings-drawer">
            <div className="drawer-head">
              <h2>Settings</h2>
              <span className="spacer" style={{ flex: 1 }} />
              <button className="drawer-x" onClick={() => setSettingsOpen(false)}
                title="Close (esc)">✕</button>
            </div>

            {/* Renders nothing (heading included) when no track needs it. */}
            <BulkReprocess onQueued={() => library.refresh()} />

            <div className="drawer-body">
              <TuningPanel />
            </div>

            <div className="drawer-section">
              <h3>Train from the imported mixes</h3>
              <span className="hint">
                The <code>w/</code> overlay lines in every imported mix are
                documented vocal-over-instrumental mashups. Once their tracks are
                ingested and analysed, build a dataset and train a model that
                scores new matches. It lives here rather than on Mixes because it
                acts on the whole library, not on the set you are matching.
              </span>
            </div>
            <div className="drawer-body">
              <MlPanel />
            </div>

            <div className="drawer-section">
              <span className="hint">
                The database browser is a debugging window, not a step in the
                workflow — which is why it lives here rather than in the rail.
              </span>
            </div>
            <div className="drawer-body">
              <DatabaseBrowser />
            </div>
          </aside>
        </>
      )}

      {paletteOpen && (
        <CommandPalette onClose={() => setPaletteOpen(false)}
          goTo={[
            ...NAV_ITEMS.map(([id, label, glyph]) => ({
              key: `go:${id}`, glyph, label, run: () => setRoute(id) })),
            ...(STYLEGUIDE ? [{ key: "go:styleguide", glyph: "◇", label: "Styleguide",
              hint: "tokens and primitives", run: () => setRoute("styleguide") }] : []),
          ]}
          actions={[
            { key: "a:search", glyph: "⌕", label: "Focus search", hint: "/", run: focusSearch },
            ...(selectedTrackId != null ? [{ key: "a:clear", glyph: "✕",
              label: "Clear the selected track", hint: "esc", run: clearSelection }] : []),
            { key: "a:rail", glyph: "«", label: "Toggle rail", run: rail.toggle },
            ...(player.source ? [{ key: "a:play", glyph: "▶", label: "Play / pause",
              hint: "space", run: () => (player.playing ? player.pause() : player.resume()) }] : []),
            { key: "a:help", glyph: "?", label: "How it works", hint: "?", run: openHelp },
            { key: "a:settings", glyph: "⚙", label: "Settings", run: () => setSettingsOpen(true) },
          ]}
          tracks={library.tracks}
          pairs={dock.rows}
          onOpenTrack={(id) => { setSelectedTrackId(id); setRoute("track"); }}
          onOpenPair={(c) => {
            setRoute("library");
            const at = dock.rows.indexOf(c);
            if (at >= 0) dock.setCursor(at);
            dock.play(c);
          }} />
      )}

      <Toast />
      {helpOpen && <HelpPanel onClose={() => setHelpOpen(false)} />}
    </div>
  );
}
