// Tiny pub/sub toast bus. Components call toast("message"); the <Toast/>
// element in App.jsx subscribes via onToast and renders the latest message.
const listeners = new Set();

export function toast(message) {
  for (const cb of listeners) cb(message);
}

export function onToast(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
