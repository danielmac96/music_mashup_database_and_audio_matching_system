// Tiny pub/sub toast bus. Components call toast("message"); the <Toast/>
// element in App.jsx subscribes via onToast and renders the latest message.
const listeners = new Set();

// An optional action ({ label, onClick }) renders as a button on the toast —
// "Re-score now" after a BPM fix, rather than a message naming a button that
// lives somewhere else.
export function toast(message, action = null) {
  for (const cb of listeners) cb(message, action);
}

export function onToast(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
