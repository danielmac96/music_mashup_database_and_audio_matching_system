import { useCallback, useEffect, useRef, useState } from "react";
import { MashupSuggestions } from "./MashupSuggestions";
import { SoundCloudBrowser } from "./SoundCloudBrowser";
import { Suggestions } from "./Suggestions";
import { ScreenHeader } from "../shell/ScreenHeader";
import { RailRow, RailSection } from "../shell/Sidebar";
import { api } from "../api";

// Discovery is three questions that share a tab because they are one job at
// three scales: "what should I add to the library?", "what would I like that I
// don't have?", and "what should I build out of what I have?". Each feeds the
// next — suggestions come from the library, and the mashup list is what tells
// you which kind of track you are short of.
const MODES = [
  ["tracks", "Find tracks"],
  ["suggest", "Suggestions"],
  ["mashups", "Find mashups"],
];

const HINTS = {
  tracks: "Search SoundCloud, shortlist into a crate, then import the lot.",
  suggest: "More like the records you already have — tracks, artists and sets.",
  mashups: "Ranked section pairs from the tracks you already have.",
};

const MODE_KEY = "mashup.discovery.mode.v1";

function loadMode() {
  try {
    const saved = localStorage.getItem(MODE_KEY);
    return MODES.some(([id]) => id === saved) ? saved : "tracks";
  } catch {
    return "tracks";
  }
}

export function Discovery({ player, seed, onClearSeed, onAudition, onStatus,
                            showInstOverInst, onOpenLibrary, onRailSlot,
                            onGroupsChanged }) {
  const [mode, setMode] = useState(loadMode);
  // Mashups is expensive to mount — it fetches a ranked list, filter
  // vocabularies, scorer status and every stored verdict. Once visited it stays
  // mounted and is hidden with CSS so toggling back is instant and keeps its
  // filters, cursor and scroll position (the same trick MixImporter uses for
  // its match board).
  const [mashupsMounted, setMashupsMounted] = useState(() => loadMode() === "mashups");
  // Suggestions is cheap to mount, but its RESULT costs a job of tens of
  // seconds. Unmounting on a tab switch would throw that away, so once visited
  // it stays mounted and is hidden with CSS — the same trick as the mashups pane.
  const [suggestMounted, setSuggestMounted] = useState(() => loadMode() === "suggest");
  // An artist or set clicked in Suggestions opens in the browser pane.
  const [nav, setNav] = useState(null);

  const modeRef = useRef(mode);
  modeRef.current = mode;

  const switchMode = (next) => {
    if (next === mode) return;
    if (next === "mashups") setMashupsMounted(true);
    if (next === "suggest") setSuggestMounted(true);
    onStatus?.(null);          // each pane owns the header readout while visible
    setMode(next);
    try { localStorage.setItem(MODE_KEY, next); } catch { /* full */ }
  };

  // A seed means "find beds for this track", sent from Library. It is only
  // meaningful in the mashups pane, so honour it by switching — and persist the
  // mode like any other switch, so arriving by seed and arriving by click leave
  // you in the same place next time.
  useEffect(() => {
    if (!seed) return;
    setMashupsMounted(true);
    setMode("mashups");
    try { localStorage.setItem(MODE_KEY, "mashups"); } catch { /* full */ }
  }, [seed]);

  // A hidden pane still runs its effects, and would otherwise push its status
  // into the header while you are looking at another one. One stable callback
  // each: reading `mode` directly — or calling a factory inline in the JSX —
  // would change the callback's identity every render and churn the child's
  // effects, which is why this goes through the ref.
  const mashupStatus = useCallback((status) => {
    if (modeRef.current === "mashups") onStatus?.(status);
  }, [onStatus]);

  const suggestStatus = useCallback((status) => {
    if (modeRef.current === "suggest") onStatus?.(status);
  }, [onStatus]);

  // The rail's per-route furniture. SOURCES are the four ways in — they map
  // onto the modes and the browser's `kind`, so the rail is a shortcut to a
  // state the panes already have rather than a fifth thing to keep in sync.
  useEffect(() => {
    onRailSlot?.(
      <>
        <RailSection label="SOURCES">
          <RailRow label="Search" active={mode === "tracks" && !nav}
            onClick={() => switchMode("tracks")} />
          <RailRow label="Similar to library" active={mode === "suggest"}
            onClick={() => switchMode("suggest")}
            title="Ranked from the tracks you already own" />
          <RailRow label="Find mashups" active={mode === "mashups"}
            onClick={() => switchMode("mashups")}
            title="Pairs inside the library you already have" />
        </RailSection>
        <SavedProfiles onOpen={(profile) => {
          setNav({ kind: "user", id: profile.user_id, label: profile.username });
          switchMode("tracks");
        }} />
      </>,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, nav]);

  return (
    <>
      <ScreenHeader title="Discover">
        <div className="pd-seg">
          {MODES.map(([id, label]) => (
            <button key={id} className={mode === id ? "on" : ""}
              onClick={() => switchMode(id)}>
              {label}
            </button>
          ))}
        </div>
        <span className="hint">{HINTS[mode]}</span>
      </ScreenHeader>

      <div className="disc-body">
        {mode === "tracks" && (
          <SoundCloudBrowser onStatus={onStatus} onOpenLibrary={onOpenLibrary}
            player={player}
            nav={nav} onNavDone={() => setNav(null)}
            onGroupsChanged={onGroupsChanged} />
        )}

      {suggestMounted && (
        <div className="disc-pane"
          style={mode === "suggest" ? undefined : { display: "none" }}>
          <Suggestions
            player={player}
            onStatus={suggestStatus}
            onOpenLibrary={onOpenLibrary}
            onNavigate={(target) => {
              if (!target?.id) return;
              setNav(target);
              switchMode("tracks");
            }}
          />
        </div>
      )}

      {mashupsMounted && (
        <div className="disc-pane"
          style={mode === "mashups" ? undefined : { display: "none" }}>
          <MashupSuggestions
            player={player}
            seed={seed}
            onClearSeed={onClearSeed}
            onAudition={onAudition}
            onStatus={mashupStatus}
            showInstOverInst={showInstOverInst}
            active={mode === "mashups"}
          />
        </div>
      )}
      </div>
    </>
  );
}

// The rail's profile shelf. NOT "followed profiles": followings need
// /me/followings, which needs OAuth, which ships dormant — and the browse layer
// has no followings scrape. These are bookmarks, kept in app_prefs, and there
// is no "n new" badge because nothing snapshots a profile to diff against.
function SavedProfiles({ onOpen }) {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    let live = true;
    api.discoverySavedProfiles()
      .then((b) => { if (live) setRows(b.profiles || []); })
      .catch(() => { if (live) setRows([]); });
    return () => { live = false; };
  }, []);
  if (!rows.length) return null;
  return (
    <RailSection label="SAVED PROFILES" scroll>
      {rows.map((p) => (
        <RailRow key={p.user_id} label={p.username} count={p.track_count}
          glyph="◍" onClick={() => onOpen(p)}
          title={`${p.track_count || 0} public tracks`} />
      ))}
    </RailSection>
  );
}
