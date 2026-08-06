// useHookAudition — arm a candidate pair's 16-bar hooks and loop them (T1.7).
//
// The point of the ranked list is that judging a candidate costs a keypress
// rather than 30-60 seconds. That budget only works if:
//   * we play PRE-CUT hook clips (T1.6), not whole ~40 MB stems;
//   * the next rows' clips are already decoding before you reach them;
//   * moving to a new row cancels the previous one instead of queueing behind it.
//
// Coordinate space (see engine/MashupEngine.js): everything here is in global
// display seconds, both voices start at 0, and the bed's `rate` is raw seconds
// per display second — so conforming the bed to the vocal's tempo is exactly
// the stretch_factor the server sends with the row.
import { useCallback, useEffect, useRef, useState } from "react";
import { MashupEngine } from "../engine/MashupEngine";
import { decodeStem } from "../engine/decode";

const PREFETCH_AHEAD = 2;

// start/end (seconds) ask for that exact span instead of the track's own hook.
// Rounded to milliseconds so two rows naming the same section produce the same
// URL, which is what keeps both the server's clip cache and decodeStem's
// AudioBuffer cache hitting.
export const hookUrl = (songId, stem, start, end) => {
  const base = `/api/tracks/${songId}/hook/audio?stem=${stem}`;
  if (start == null || end == null || !(end > start)) return base;
  return `${base}&start=${start.toFixed(3)}&end=${end.toFixed(3)}`;
};

/** The two clips a candidate needs: the vocal on top, the instrumental beneath.
 *
 * Scoring stores the winning (vocal section x bed section) on the row (T3.3),
 * so the preview plays the moment the pair was actually chosen for. Rows scored
 * before that existed — or whose tracks have no structure yet — have no section
 * columns and fall back to each track's 16-bar hook. */
export function hookUrlsFor(c) {
  if (!c) return [];
  return [
    hookUrl(c.vocal_song_id, "vocals", c.vocal_section_start, c.vocal_section_end),
    hookUrl(c.inst_song_id, "instrumental", c.inst_section_start, c.inst_section_end),
  ];
}

export function useHookAudition() {
  const engineRef = useRef(null);
  const [playingId, setPlayingId] = useState(null);
  const [error, setError] = useState(null);
  // Monotonic token: a late-arriving decode from a row you have already moved
  // past must not hijack the transport. Compared on every await boundary.
  const armToken = useRef(0);

  const engine = useCallback(() => {
    if (!engineRef.current) engineRef.current = new MashupEngine();
    return engineRef.current;
  }, []);

  const stop = useCallback(() => {
    armToken.current += 1;           // invalidate anything mid-flight
    const e = engineRef.current;
    if (e) {
      e.stop();
      e.removeVoice("vocal");
      e.removeVoice("inst");
    }
    setPlayingId(null);
  }, []);

  /** Warm the decode cache for upcoming rows so stepping down is instant. */
  const prefetch = useCallback(async (candidates) => {
    const e = engine();
    try {
      await e.init();
    } catch {
      return; // no AudioContext yet (no user gesture) — prefetch is best-effort
    }
    for (const c of (candidates || []).slice(0, PREFETCH_AHEAD)) {
      for (const url of hookUrlsFor(c)) {
        // decodeStem caches by URL and swallows nothing, so a 404 on one clip
        // must not reject the whole warm-up.
        decodeStem(e.ctx, url).catch(() => {});
      }
    }
  }, [engine]);

  /**
   * Play `candidate`'s hooks, bed conformed to the vocal's tempo and pitch.
   * Safe to call repeatedly while arrowing — each call supersedes the last.
   */
  const audition = useCallback(async (candidate) => {
    if (!candidate) return;
    const token = ++armToken.current;
    setError(null);

    const e = engine();
    try {
      await e.init();
      if (armToken.current !== token) return;

      const [vocalUrl, bedUrl] = hookUrlsFor(candidate);
      const [vocalBuf, bedBuf] = await Promise.all([
        decodeStem(e.ctx, vocalUrl),
        decodeStem(e.ctx, bedUrl),
      ]);
      if (armToken.current !== token) return;   // user moved on mid-decode

      e.stop();
      e.setVoice("vocal", { buffer: vocalBuf, offsetSec: 0, rate: 1, semitones: 0, gain: 0.95 });
      e.setVoice("inst", {
        buffer: bedBuf,
        offsetSec: 0,
        // rate > 1 reads the bed faster, i.e. conforms it up to the vocal's
        // tempo; the worklet compensates pitch so only the shift below moves it.
        rate: candidate.stretch_factor || 1,
        semitones: candidate.semitone_shift || 0,
        gain: 0.8,
      });

      // Loop the shorter of the two so the cycle never runs into silence.
      const bedDisplay = bedBuf.duration / (candidate.stretch_factor || 1);
      const len = Math.max(1, Math.min(vocalBuf.duration, bedDisplay));
      e.setLoop({ start: 0, end: len });

      await e.play(0);
      if (armToken.current !== token) { e.stop(); return; }
      setPlayingId(candidate.id);
    } catch (err) {
      if (armToken.current !== token) return;
      setError(err?.message || "Could not load this pair's hooks");
      setPlayingId(null);
    }
  }, [engine]);

  const toggle = useCallback((candidate) => {
    if (candidate && playingId === candidate.id) stop();
    else audition(candidate);
  }, [audition, playingId, stop]);

  // Tear down on unmount AND on tab switch — an engine left running keeps an
  // AudioContext and a worklet alive behind whatever the user opened next.
  useEffect(() => () => {
    armToken.current += 1;
    engineRef.current?.dispose();
    engineRef.current = null;
  }, []);

  return { audition, toggle, stop, prefetch, playingId, error };
}
