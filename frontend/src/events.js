// Implicit pair events (readme §5.13): what is done with a pair, buffered here
// and posted to POST /api/events every few seconds and on unload.
//
// The one rule: NOTHING in this file may be felt. No call here throws, awaits,
// toasts or returns anything a caller could depend on. A failed post drops its
// batch — the events are a learning signal, never a record the user made. So
// every entry point is wrapped, the fetch's rejection is swallowed, and no UI
// state lives here. (tests/test_pair_events.py runs this file under Node with
// a network that always fails.)
//
// A pair is keyed by its four ids, like everywhere else (pairModel.keyOf).

export const EVENTS_URL = "/api/events";
export const FLUSH_MS = 5000;
// `shown` is sampled: the top of each dock fetch, not every row scrolled past.
export const SHOWN_SAMPLE = 20;
// A loop stopped sooner than this was skipped, not listened to.
export const SKIP_SECS = 3;
// A buffer nobody is draining (the server is down) must not grow for ever.
export const MAX_BUFFER = 400;

let buffer = [];
let timer = null;
let listening = null;      // the pair loop being timed, or null
let wired = false;

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

function scorerVersion(c) {
  if (!c || !c.scorer) return null;
  return c.model_version ? `${c.scorer}:${c.model_version}` : String(c.scorer);
}

/** The wire shape of one event, or null when `c` does not name a pair. */
export function eventOf(event, c, { value = null, context = null, rank = null } = {}) {
  const vocal = c?.vocal_song_id ?? c?.vocalSongId;
  const inst = c?.inst_song_id ?? c?.instSongId;
  if (!event || vocal == null || inst == null) return null;
  return {
    event,
    vocal_song_id: vocal,
    inst_song_id: inst,
    vocal_section: c.vocal_section_idx ?? c.vocal_section ?? null,
    inst_section: c.inst_section_idx ?? c.inst_section ?? null,
    value: typeof value === "number" && Number.isFinite(value) ? value : null,
    context,
    scorer_version: scorerVersion(c),
    rank_shown: Number.isInteger(rank) && rank > 0 ? rank : null,
  };
}

function wire() {
  if (wired || typeof window === "undefined") return;
  wired = true;
  timer = setInterval(() => flush(), FLUSH_MS);
  const leave = () => { closeListening(); flush({ unload: true }); };
  window.addEventListener("pagehide", leave);
  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") leave();
    });
  }
}

/** Record one event about a pair. Fire-and-forget: returns nothing, never throws. */
export function track(event, candidate, opts = {}) {
  try {
    const e = eventOf(event, candidate, opts);
    if (!e) return;
    if (buffer.length >= MAX_BUFFER) buffer.shift();
    buffer.push(e);
    wire();
  } catch { /* an event is never worth an error */ }
}

/** `shown` for the top of a freshly fetched list, with each row's rank. */
export function trackShown(rows, context = "dock") {
  try {
    (rows || []).slice(0, SHOWN_SAMPLE).forEach((c, i) => track("shown", c, { context, rank: i + 1 }));
  } catch { /* never felt */ }
}

/** One event per pair of an export. */
export function trackEach(event, rows, context = "dock") {
  try {
    (rows || []).forEach((c, i) => track(event, c, { context, rank: i + 1 }));
  } catch { /* never felt */ }
}

/** Post what is buffered. A failure drops the batch; nothing is retried,
 *  awaited or reported. `unload` uses sendBeacon so the page can go. */
export function flush({ unload = false } = {}) {
  try {
    if (!buffer.length) return;
    const body = JSON.stringify({ events: buffer });
    buffer = [];
    if (unload && typeof navigator !== "undefined" && navigator.sendBeacon) {
      navigator.sendBeacon(EVENTS_URL, new Blob([body], { type: "application/json" }));
      return;
    }
    const sent = fetch(EVENTS_URL, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body, keepalive: true,
    });
    if (sent && sent.catch) sent.catch(() => {});
  } catch { /* an event is never worth an error */ }
}

function closeListening() {
  const l = listening;
  listening = null;
  if (!l) return;
  if (l.since != null) l.secs += (now() - l.since) / 1000;
  if (l.secs <= 0) return;                 // it never sounded
  track(l.secs < SKIP_SECS ? "skip" : "loop_play", l.candidate,
    { value: Math.round(l.secs * 10) / 10, context: l.context, rank: l.rank });
}

/** Called by the app-wide player whenever its pair source or its playing flag
 *  changes. Times how long each pair loop actually sounded (pauses excluded)
 *  and, when the player moves off it, records `loop_play` with the seconds —
 *  or `skip` when that was under SKIP_SECS. `source` is the player's pair
 *  source ({ key, candidate, context, rank }) or null. */
export function noteListening(source, playing) {
  try {
    const key = source ? source.key : null;
    if (listening && listening.key !== key) closeListening();
    if (!source) return;
    if (!listening) {
      listening = { key, candidate: source.candidate, context: source.context || "dock",
                    rank: source.rank ?? null, secs: 0, since: null };
    }
    if (playing && listening.since == null) listening.since = now();
    else if (!playing && listening.since != null) {
      listening.secs += (now() - listening.since) / 1000;
      listening.since = null;
    }
  } catch { /* never felt */ }
}

/** For tests: what is waiting to be posted. */
export const pending = () => buffer.slice();
