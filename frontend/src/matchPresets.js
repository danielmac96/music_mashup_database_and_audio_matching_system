// Pre-filter width presets passed to "Score library" (bpm = max BPM diff,
// key = min Camelot score). Tight = only clean matches; Wide = more candidates.
//
// Balanced and Wide no longer gate on key at all (P1.1). Transposing a bed by a
// semitone or two is an ordinary move and the effort penalty already prices it,
// so a key gate on top deleted the pair before scoring AND would have demoted
// it if it had survived. Tight keeps the gate for the days you only want pairs
// that need no transpose at all.
//
// Shared by Settings → Tuning and the dock's Re-score pill, so a re-score from
// either uses the width last chosen. The choice is a per-browser convenience.
export const MATCH_PRESETS = [
  { label: "Tight", bpm: 8, key: 0.75 },
  { label: "Balanced", bpm: 16, key: 0 },
  { label: "Wide", bpm: 24, key: 0 },
];

const KEY = "mashup.matchPreset.v1";
const DEFAULT_IDX = 1;   // Balanced

export function loadPresetIdx() {
  try {
    const raw = localStorage.getItem(KEY);
    const n = raw == null ? NaN : Number(raw);
    return Number.isInteger(n) && n >= 0 && n < MATCH_PRESETS.length ? n : DEFAULT_IDX;
  } catch { return DEFAULT_IDX; }
}

export function savePresetIdx(i) {
  try { localStorage.setItem(KEY, String(i)); } catch { /* ignore */ }
}
