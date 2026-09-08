import { useEffect, useState } from "react";
import { api } from "../api";

// Waveforms come from the SERVER, not from decoding audio in the browser.
//
// `features.waveform_rms_json` holds 360 normalised RMS points per stem,
// measured during analysis, and GET /api/tracks/{id}/waveform serves them with
// the beat grid. Drawing from those points costs one small JSON request instead
// of fetching and decoding a ~40MB stem — which is what MixStudio already does,
// and the reason its lanes paint before the audio has finished loading.
//
// There is no batch endpoint, so this is one request per (song, stem). Fine for
// a screen showing one track's two stems; do NOT reach for it from a list.

const cache = new Map();   // `${songId}:${stem}` -> Promise<payload>

function fetchWave(songId, stem) {
  const key = `${songId}:${stem}`;
  if (!cache.has(key)) {
    // The promise is cached, not the value, so two callers mounting at once
    // share one request. A failure is evicted so it is not cached forever.
    cache.set(key, api.getWaveform(songId, stem).catch((e) => {
      cache.delete(key);
      throw e;
    }));
  }
  return cache.get(key);
}

export function useWaveform(songId, stem) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (songId == null) { setData(null); return undefined; }
    let live = true;
    setError(null);
    fetchWave(songId, stem)
      .then((d) => { if (live) setData(d); })
      .catch((e) => { if (live) { setData(null); setError(e.message); } });
    return () => { live = false; };
  }, [songId, stem]);

  return { waveform: data?.waveform || null, beatTimes: data?.beat_times || [],
           beatPhase: data?.beat_phase || 0, error };
}

// A closed envelope for an SVG polygon: across the top, then back along the
// bottom, mirrored about the centre line.
export function envelopePoints(values, width = 1000, height = 78, gain = 1) {
  if (!values || values.length < 2) return "";
  const mid = height / 2;
  const n = values.length;
  const top = [];
  const bottom = [];
  for (let i = 0; i < n; i += 1) {
    const x = (i / (n - 1)) * width;
    const v = Math.max(0, Math.min(1, (values[i] || 0) * gain));
    top.push(`${x.toFixed(2)},${(mid - v * mid).toFixed(2)}`);
    bottom.push(`${x.toFixed(2)},${(mid + v * mid).toFixed(2)}`);
  }
  bottom.reverse();
  return `${top.join(" ")} ${bottom.join(" ")}`;
}
