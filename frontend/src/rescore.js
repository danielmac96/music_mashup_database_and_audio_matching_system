// "Re-score now" can be asked for from anywhere (a toast after a BPM fix, the
// dock's pill); one listener — useFreshness, mounted once in App — runs it, so
// there is a single scoring job and a single place that knows about it.
const listeners = new Set();

export function requestRescore() {
  for (const cb of listeners) cb();
}

export function onRescoreRequest(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
