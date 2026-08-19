import { useCallback, useEffect, useRef, useState } from "react";
import { MashupSuggestions } from "./MashupSuggestions";
import { SoundCloudBrowser } from "./SoundCloudBrowser";

// Discovery is two questions that share a tab because they are the same job at
// two scales: "what should I add to the library?" and "what should I build out
// of it?". Finding tracks feeds finding mashups, and the mashup list is what
// tells you which kind of track you are short of.
const MODES = [
  ["tracks", "Find tracks"],
  ["mashups", "Find mashups"],
];

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

  const modeRef = useRef(mode);
  modeRef.current = mode;

  const switchMode = (next) => {
    if (next === mode) return;
    if (next === "mashups") setMashupsMounted(true);
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

  // A hidden MashupSuggestions still runs its effects, and would otherwise push
  // its status into the header while you are looking at search results. The ref
  // keeps this callback stable — deriving it from `mode` on each render would
  // change its identity and churn the child's effects.
  const gatedStatus = useCallback((status) => {
    if (modeRef.current === "mashups") onStatus?.(status);
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
        <span className="hint">
          {mode === "tracks"
            ? "Search SoundCloud, shortlist into a crate, then import the lot."
            : "Ranked section pairs from the tracks you already have."}
        </span>
      </div>

      {mode === "tracks" && (
        <SoundCloudBrowser onStatus={onStatus} onOpenLibrary={onOpenLibrary} />
      )}

      {mashupsMounted && (
        <div style={mode === "mashups" ? undefined : { display: "none" }}>
          <MashupSuggestions
            seed={seed}
            onClearSeed={onClearSeed}
            onAudition={onAudition}
            onStatus={gatedStatus}
            showInstOverInst={showInstOverInst}
          />
        </div>
      )}
    </>
  );
}
