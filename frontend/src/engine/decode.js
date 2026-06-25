// Fetch a stem URL and decode it into an AudioBuffer, caching by URL so the
// same stem isn't re-downloaded/re-decoded when toggling tracks or replaying.
const cache = new Map(); // url -> Promise<AudioBuffer>

export function decodeStem(ctx, url) {
  if (cache.has(url)) return cache.get(url);
  const p = fetch(url)
    .then((res) => {
      if (!res.ok) throw new Error(`audio fetch ${res.status} for ${url}`);
      return res.arrayBuffer();
    })
    // decodeAudioData detaches the ArrayBuffer; each URL decodes once and the
    // resulting AudioBuffer is immutable + safe to reuse across plays.
    .then((buf) => ctx.decodeAudioData(buf))
    .catch((err) => {
      cache.delete(url); // let a later attempt retry instead of caching the failure
      throw err;
    });
  cache.set(url, p);
  return p;
}

export function clearDecodeCache() {
  cache.clear();
}
