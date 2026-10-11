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

// Something a pair's score reads just changed here (a BPM or key edit, a bar 1
// fix): the freshness check runs now rather than at its next poll, so the
// dock's Re-score strip appears with the edit.
const changed = new Set();

export function notifyPairsInputChanged() {
  for (const cb of changed) cb();
}

export function onPairsInputChanged(cb) {
  changed.add(cb);
  return () => changed.delete(cb);
}
