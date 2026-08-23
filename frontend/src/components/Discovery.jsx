import { useCallback, useEffect, useRef, useState } from "react";
import { MashupSuggestions } from "./MashupSuggestions";
import { SoundCloudBrowser } from "./SoundCloudBrowser";
import { Suggestions } from "./Suggestions";

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

export function Discovery({ seed, onClearSeed, onAudition, onStatus,
                            showInstOverInst, onOpenLibrary }) {
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
  // meaningful in the mashups pane, so honour it by switching.
  useEffect(() => {
    if (!seed) return;
    setMashupsMounted(true);
    setMode("mashups");
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

  return (
    <>
      <div className="discovery-modebar">
        <div className="seg">
          {MODES.map(([id, label]) => (
            <button key={id} className={mode === id ? "active" : ""}
              onClick={() => switchMode(id)}>
              {label}
            </button>
          ))}
        </div>
        <span className="hint">{HINTS[mode]}</span>
      </div>

      {mode === "tracks" && (
        <SoundCloudBrowser onStatus={onStatus} onOpenLibrary={onOpenLibrary}
          nav={nav} onNavDone={() => setNav(null)} />
      )}

      {suggestMounted && (
        <div style={mode === "suggest" ? undefined : { display: "none" }}>
          <Suggestions
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
        <div style={mode === "mashups" ? undefined : { display: "none" }}>
          <MashupSuggestions
            seed={seed}
            onClearSeed={onClearSeed}
            onAudition={onAudition}
            onStatus={mashupStatus}
            showInstOverInst={showInstOverInst}
          />
        </div>
      )}
    </>
  );
}
