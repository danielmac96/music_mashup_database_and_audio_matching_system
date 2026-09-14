// usePlayer — the ONE thing in this app that is allowed to make a sound.
//
// Before this, four players ran side by side and none of them owned the bar:
// LibraryScreen had a hidden <audio>, TrackDetail had a SECOND hidden <audio>,
// the pair dock had a MashupEngine, and Discover's mashup pane built a THIRD
// AudioContext of its own. Nothing arbitrated, so two could sound at once — and
// the bar at the bottom was wired only to the dock, so a song played from a
// library row had no play/pause, no scrub and no time readout at all. Worse, the
// <audio> elements were rendered by the screens that started them, so navigating
// away unmounted the element and killed playback silently.
//
// So: one hook, at App scope, three backends behind one interface.
//
//   track  <audio>       a whole song or stem, optionally looping a window
//   pair   MashupEngine  two conformed stems          — scrubs within the loop
//   sc     SC widget     an un-imported Discover row  — scrubs the full duration
//
// The <audio> element is created with `new Audio()` and held in a ref, NOT
// rendered as JSX. That is the fix, not an implementation detail: an element
// owned by a screen dies with that screen, which is the bug.
//
// There used to be a fourth backend, `clip`, which played a section as a
// server-cut WAV containing ONLY that section. It had to go: a file that holds
// twelve seconds cannot be scrubbed anywhere else in the record, which is
// exactly the "clicking the bar just resets the loop" report. A section is now
// a LOOP WINDOW on the whole file — `loop: {start, end}` on a `track` source —
// so the strip spans the song, the playhead can be dragged anywhere, and
// switching stems is a src swap rather than a fresh server render.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useHookAudition } from "./useHookAudition";
import { createScPlayer } from "./useScWidget";
import { toast } from "../toast";

/** A source's identity, for "is this row the one playing?" checks. Two sources
 *  naming the same audio must produce the same key, and a pair is keyed by its
 *  four ids for the reason pairModel spells out — never by candidate.id.
 *
 *  The STEM is deliberately absent from a track's key. The stem is a live
 *  control on the bar now, not part of what you picked: switching from Full to
 *  Vox is still the same thing you are listening to, and putting it in the key
 *  would drop every row highlight in the app mid-song. The loop window IS in
 *  the key, because "this record" and "the chorus of this record" are two
 *  different things to be playing and `toggle` has to tell them apart. */
export function sourceKey(s) {
  if (!s) return null;
  if (s.kind === "track") {
    const w = s.loop
      ? `${(s.loop.start ?? 0).toFixed(3)}-${(s.loop.end ?? 0).toFixed(3)}`
      : "all";
    return `track:${s.songId}:${w}`;
  }
  if (s.kind === "sc") return `sc:${s.trackId || s.permalink}`;
  if (s.kind === "pair") return `pair:${s.key}`;
  return null;
}

// How often the rAF ticker is allowed to push a position into React. 60ms is
// ~16/s: smooth enough for the detail screen's playhead, cheap enough not to
// re-render the shell every frame.
const POSITION_PUSH_MS = 60;

export function usePlayer() {
  // The pair backend is the existing auditioner, unchanged — it already owns the
  // engine, the stem solo and the loop counting.
  const pair = useHookAudition();

  const elRef = useRef(null);
  const scRef = useRef(null);
  const [source, setSource] = useState(null);
  // Read by the rAF ticker and by switchStem, neither of which should
  // re-subscribe every time the source is patched.
  const sourceRef = useRef(null);
  sourceRef.current = source;
  // Element/widget transport. The pair backend reports its own; the readouts
  // below pick whichever backend is actually live.
  const [elState, setElState] = useState({ playing: false, position: 0, duration: 0 });
  const [error, setError] = useState(null);

  const audio = useCallback(() => {
    if (!elRef.current) {
      const el = new Audio();
      el.preload = "metadata";
      elRef.current = el;
    }
    return elRef.current;
  }, []);

  const sc = useCallback(() => {
    if (!scRef.current) {
      scRef.current = createScPlayer((patch) => {
        // An error keeps the SOURCE: the bar has to stay up, because its
        // ◎ SoundCloud link is the way through when the widget will not play.
        if ("error" in patch) {
          setError(patch.error);
          if (patch.error) { setElState((s) => ({ ...s, playing: false })); return; }
        }
        if (patch.ended) setSource(null);
        const { error: _e, ended: _d, ...rest } = patch;
        setElState((s) => {
          const next = { ...s, ...rest };
          // The widget reports the PREVIOUS track's length when the new one
          // fails to load, so its duration is only trusted when we have none of
          // our own. The row's duration_secs is SoundCloud's own full_duration
          // and is the better number anyway.
          if (rest.duration != null && s.duration > 0) next.duration = s.duration;
          return next;
        });
      });
    }
    return scRef.current;
  }, []);

  // A seek that survives being asked too early. Setting currentTime before the
  // element knows its duration is thrown away silently, which is every seek
  // issued in the same tick as a src assignment — i.e. landing on a loop start
  // and restoring the clock across a stem switch, both of which do exactly that.
  const pendingSeekRef = useRef(null);
  const seekWhenReady = useCallback((secs) => {
    const el = elRef.current;
    if (!el) return;
    const target = Math.max(0, secs || 0);
    if (pendingSeekRef.current) {
      el.removeEventListener("loadedmetadata", pendingSeekRef.current);
      pendingSeekRef.current = null;
    }
    if (el.readyState >= 1) { el.currentTime = target; return; }
    const once = () => {
      el.removeEventListener("loadedmetadata", once);
      pendingSeekRef.current = null;
      el.currentTime = target;
    };
    pendingSeekRef.current = once;
    el.addEventListener("loadedmetadata", once);
  }, []);

  // Called before every play. Mutual exclusion is the point of this hook: three
  // backends exist, exactly one of them may be sounding.
  const silence = useCallback((keep) => {
    if (keep !== "element" && elRef.current) {
      elRef.current.pause();
      elRef.current.removeAttribute("src");
      elRef.current.load();
    }
    if (keep !== "sc" && scRef.current) scRef.current.stop();
    if (keep !== "pair") pair.stop();
  }, [pair]);

  const stop = useCallback(() => {
    silence(null);
    setSource(null);
    setElState({ playing: false, position: 0, duration: 0 });
    setError(null);
  }, [silence]);

  const play = useCallback(async (next) => {
    if (!next) return;
    setError(null);

    if (next.kind === "pair") {
      silence("pair");
      setElState({ playing: false, position: 0, duration: 0 });
      setSource(next);
      pair.audition(next.candidate);
      return;
    }

    if (next.kind === "sc") {
      // Answered before the click costs anything: a row SoundCloud will not
      // embed gets the old behaviour and a reason, not a button that fails.
      if (next.embeddable === false) {
        toast("SoundCloud won't embed this one — opening it there instead");
        window.open(next.permalink, "_blank", "noopener");
        return;
      }
      silence("sc");
      setElState({ playing: false, position: 0, duration: next.duration || 0 });
      setSource(next);
      try {
        await sc().play(next.permalink);
      } catch (e) {
        setError(e?.message || "Could not play this on SoundCloud");
        setSource(null);
      }
      return;
    }

    // track — the one <audio> element, playing a whole file. A section is this
    // with `loop` set; the file is the same either way, which is what lets the
    // strip span the record and the playhead leave the section.
    silence("element");
    const el = audio();
    el.src = api.audioUrl(next.songId, next.stem || "full");
    // NEVER the native loop: that loops the whole FILE, and a window inside one
    // has to be wrapped by the ticker below.
    el.loop = false;
    const at = next.loop ? (next.loop.start || 0) : (next.startAt || 0);
    // A track's length is seeded from the library row so the strip is not
    // zero-width for the first frame; loadedmetadata replaces it with the truth.
    setElState({ playing: false, position: at, duration: next.duration || 0 });
    setSource(next);
    if (at > 0) seekWhenReady(at);
    try {
      await el.play();
    } catch (e) {
      // A missing download or an autoplay refusal both land here.
      setError(e?.message || "Could not play that");
      setSource(null);
    }
  }, [audio, pair, sc, seekWhenReady, silence]);

  useEffect(() => {
    const el = audio();
    const sync = () => setElState((s) => ({
      ...s,
      playing: !el.paused && !el.ended,
      position: el.currentTime,
      duration: Number.isFinite(el.duration) && el.duration > 0 ? el.duration : s.duration,
    }));
    const onEnd = () => { setElState((s) => ({ ...s, playing: false })); setSource(null); };
    const onErr = () => { setError("That audio could not be loaded"); setSource(null); };
    const events = ["timeupdate", "loadedmetadata", "play", "pause", "seeked",
                    "durationchange", "progress"];
    events.forEach((e) => el.addEventListener(e, sync));
    el.addEventListener("ended", onEnd);
    el.addEventListener("error", onErr);
    return () => {
      events.forEach((e) => el.removeEventListener(e, sync));
      el.removeEventListener("ended", onEnd);
      el.removeEventListener("error", onErr);
    };
  }, [audio]);

  // The loop wrap, and a smooth playhead.
  //
  // `timeupdate` fires about four times a second, which is both a visibly steppy
  // playhead and — far worse — up to 250ms of overshoot past the end of a loop.
  // One rAF does the wrap within a frame and pushes a position at ~16/s. The
  // pre-cut clip looped natively in the audio thread and so was gapless; one
  // frame of slop is what scrubbing the whole record costs.
  useEffect(() => {
    if (source?.kind !== "track" || !elState.playing) return undefined;
    let raf = 0;
    let lastPush = 0;
    const tick = (t) => {
      raf = requestAnimationFrame(tick);
      const el = elRef.current;
      if (!el) return;
      const loop = sourceRef.current?.loop;
      if (loop && el.currentTime >= loop.end) el.currentTime = loop.start || 0;
      if (t - lastPush < POSITION_PUSH_MS) return;
      lastPush = t;
      const now = el.currentTime;
      setElState((s) => (Math.abs(s.position - now) < 0.001 ? s : { ...s, position: now }));
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [source?.kind, elState.playing]);

  // Tear down once, with the app shell.
  useEffect(() => () => {
    elRef.current?.pause();
    scRef.current?.dispose();
  }, []);

  const kind = source?.kind || null;
  const onPair = kind === "pair";

  const playing = onPair ? pair.playing : elState.playing;
  const position = onPair ? pair.position : elState.position;
  const duration = onPair ? pair.loopLength : elState.duration;

  const pause = useCallback(() => {
    if (kind === "pair") pair.pause();
    else if (kind === "sc") scRef.current?.pause();
    else elRef.current?.pause();
  }, [kind, pair]);

  const resume = useCallback(() => {
    if (kind === "pair") pair.resume();
    else if (kind === "sc") scRef.current?.resume();
    else elRef.current?.play().catch(() => setError("Could not resume"));
  }, [kind, pair]);

  /** Seek in the SAME seconds the bar displays: absolute song seconds for a
   *  track (loop or no loop), position-in-loop for a pair.
   *
   *  Seeking OUT of an armed loop releases it. The alternative — snapping back
   *  at the loop end — means dropping the playhead at 2:40 while the 1:04 chorus
   *  is looping jumps you back instantly, which reads as the drag having failed.
   *  Releasing says what happened: you asked to hear somewhere else. */
  const seek = useCallback((secs) => {
    if (kind === "pair") { pair.seek(secs); return; }
    if (kind === "sc") { scRef.current?.seek(secs); return; }
    const el = elRef.current;
    if (!el) return;
    const target = Math.max(0, secs);
    const loop = source?.loop;
    if (loop && (target < (loop.start || 0) - 0.001 || target > loop.end + 0.001)) {
      setSource((s) => (s ? { ...s, loop: null } : s));
    }
    seekWhenReady(target);
    setElState((s) => ({ ...s, position: target }));
  }, [kind, pair, seekWhenReady, source]);

  /** Swap which stem of the CURRENT track is sounding, keeping the clock, the
   *  loop and whether it was playing. The loop lives on the source rather than
   *  on the file, so it survives the swap untouched. */
  const switchStem = useCallback((stemId) => {
    const s = sourceRef.current;
    if (!s || s.kind !== "track" || !stemId || s.stem === stemId) return;
    const el = elRef.current;
    if (!el) return;
    const at = el.currentTime;
    const wasPlaying = !el.paused && !el.ended;
    el.src = api.audioUrl(s.songId, stemId);
    el.loop = false;
    seekWhenReady(at);
    setSource({ ...s, stem: stemId });
    setElState((st) => ({ ...st, position: at }));
    if (wasPlaying) el.play().catch(() => setError("Could not switch stem"));
  }, [seekWhenReady]);

  const key = useMemo(() => sourceKey(source), [source]);

  const toggle = useCallback((next) => {
    const nextKey = sourceKey(next);
    if (nextKey && nextKey === key) {
      if (playing) pause(); else resume();
      return;
    }
    play(next);
  }, [key, pause, play, playing, resume]);

  const isPlaying = useCallback((k) => Boolean(k) && k === key, [key]);

  return {
    source, kind, key, playing, position, duration,
    error: error || pair.error,
    play, toggle, pause, resume, seek, stop, isPlaying, switchStem,
    // Pair-only passthrough. The dock and the suggestions pane drive these, and
    // the bar shows them only while a pair is the source.
    pair,
  };
}
