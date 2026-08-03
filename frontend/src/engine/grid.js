// Beat-grid math shared by both studios (T1.4).
//
// beat_times says WHERE the beats are; it does not say which of them starts a
// bar. librosa's tracker latches wherever the onset evidence is strongest and
// is just as happy to start on beat 3, so "every 4th beat from the first
// detected beat" silently puts the bar lines 1–3 beats off. beat_phase (stored
// per features row) is the index, 0–3, of the first detected beat within its
// bar; analysis picks it by summing onset strength at each candidate phase.
//
// Tracks analysed before beat_phase existed store 0, which reproduces the old
// behaviour exactly — so every call site can adopt this with no visual change
// until the track is re-analysed.

export const BEATS_PER_BAR = 4;

/** Is beat `i` of a beat_times array a downbeat, given the track's phase? */
export function isDownbeat(i, phase = 0) {
  // JS % keeps the sign of the dividend, so i=0 with phase=2 gives -2, not 2.
  // Normalising is what makes phases 1–3 work at the start of the array.
  return (((i - phase) % BEATS_PER_BAR) + BEATS_PER_BAR) % BEATS_PER_BAR === 0;
}

/** The downbeat times from a beat_times array, in the same units. */
export function downbeatsOf(beatTimes, phase = 0) {
  return (beatTimes || []).filter((_, i) => isDownbeat(i, phase));
}

/** beat_phase off a features payload, clamped to a real bar position. */
export function beatPhaseOf(feature) {
  const p = Number(feature?.beat_phase);
  return Number.isInteger(p) && p >= 0 && p < BEATS_PER_BAR ? p : 0;
}

/**
 * The phase that makes beat `i` a downbeat — for the alt+click override, where
 * the user points at the beat they hear as bar 1.
 */
export function phaseForDownbeatAt(i) {
  return ((i % BEATS_PER_BAR) + BEATS_PER_BAR) % BEATS_PER_BAR;
}
