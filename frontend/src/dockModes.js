// "Show me what comes next in this set" can be asked from a toast (after + Set)
// as well as from the dock; usePairDock listens and swaps its list for the
// set's next candidates (GET /api/sets/{id}/next). A set built by pressing `a`
// down one track's list was that track over four beds — nothing pointed at the
// pairs that move the set on.
const listeners = new Set();

export function requestNextForSet(setId, setName = "") {
  for (const cb of listeners) cb(setId, setName);
}

export function onNextForSet(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
