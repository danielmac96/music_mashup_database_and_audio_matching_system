import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { KeyChip } from "./KeyChip";
import { TrackArt } from "./TrackArt";
import { MashupEngine } from "../engine/MashupEngine";
import { decodeStem } from "../engine/decode";
import { downbeatsOf, isDownbeat } from "../engine/grid";
import { fmtTime, keyRel } from "../theme";
import { toast } from "../toast";

const SECTION_COLORS = {
  intro: "#6b7280", verse: "#3b82f6", chorus: "#ec4899", drop: "#f59e0b",
  breakdown: "#14b8a6", bridge: "#22c55e", outro: "#6b7280",
};

const TIMELINE_WIDTH = 1200;
const TIMELINE_HEIGHT = 100;
const LOOP_BARS = [1, 2, 4, 8];
const STEMS = [
  { value: "vocals", label: "Vocal" },
  { value: "instrumental", label: "Inst" },
  { value: "full", label: "Full" },
];
// Below this, vocal-stem beat/tempo tracking is unreliable — fall back to full.
const VOCAL_BPM_CONFIDENCE_MIN = 0.35;

function eventPos(e, el, pps) {
  const rect = el.getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  return (frac * TIMELINE_WIDTH) / pps;
}

function dragFraction(e, apply) {
  const rect = e.currentTarget.getBoundingClientRect();
  const move = (ev) => {
    let f = (ev.clientX - rect.left) / rect.width;
    f = Math.max(0, Math.min(1, f));
    apply(f);
  };
  move(e);
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}

function lowerBoundIndex(sortedArr, val) {
  let lo = 0, hi = sortedArr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedArr[mid] < val) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function visibleBeatWindow(beatTimes, offsetSecs, pps, marginPx = 150) {
  if (beatTimes.length === 0) return beatTimes;
  const loSec = -marginPx / pps - offsetSecs;
  const hiSec = (TIMELINE_WIDTH + marginPx) / pps - offsetSecs;
  return beatTimes.slice(lowerBoundIndex(beatTimes, loSec), lowerBoundIndex(beatTimes, hiSec));
}

// Snap the drag offset so the closest on-screen beat/downbeat of the dragged
// deck lines up with the other deck's nearest beat/downbeat once within snapPx.
function snapOffsetToBeats(rawOffsetSecs, ownBeats, otherBeats, otherOffsetSecs, pps, mode, snapPx = 10) {
  const stride = mode === "bars" ? 4 : 1;
  const own = stride > 1 ? ownBeats.filter((_, i) => i % stride === 0) : ownBeats;
  const oth = stride > 1 ? otherBeats.filter((_, i) => i % stride === 0) : otherBeats;
  const a = visibleBeatWindow(own, rawOffsetSecs, pps);
  const b = visibleBeatWindow(oth, otherOffsetSecs, pps);
  if (a.length === 0 || b.length === 0) return rawOffsetSecs;
  let i = 0, j = 0, bestAbsDiff = Infinity, bestDelta = 0;
  while (i < a.length && j < b.length) {
    const av = a[i] + rawOffsetSecs;
    const bv = b[j] + otherOffsetSecs;
    const diff = bv - av;
    if (Math.abs(diff) < bestAbsDiff) { bestAbsDiff = Math.abs(diff); bestDelta = diff; }
    if (av < bv) i++; else j++;
  }
  const snapSecs = snapPx / pps;
  return bestAbsDiff <= snapSecs ? rawOffsetSecs + bestDelta : rawOffsetSecs;
}

function shiftCamelot(camelot, semitones) {
  if (!camelot) return "—";
  const n = parseInt(camelot, 10) || 1;
  const letter = camelot.slice(-1);
  const num = ((n - 1 + 7 * semitones) % 12 + 12) % 12 + 1;
  return `${num}${letter}`;
}

function trackLabel(t) {
  return `${t.title}${t.artist ? ` — ${t.artist}` : ""}`;
}

const nearest = (arr, target) =>
  arr.reduce((best, t) => (Math.abs(t - target) < Math.abs(best - target) ? t : best), arr[0]);

// ── One waveform lane (section ribbon + canvas waveform/beat-grid + playhead) ─
function Lane({
  deck, sections, durationSecs, pps, offsetSecs, onOffsetChange,
  waveform = [], beatTimes = [], beatPhase = 0, onPlay, isPlaying, playheadPos, loop,
  otherBeatTimes = [], otherOffsetSecs = 0, snapMode = "beats", beatSecs = 0,
}) {
  const [dragging, setDragging] = useState(false);
  const [snapped, setSnapped] = useState(false);
  const dragRef = useRef(null);
  const canvasRef = useRef(null);
  const isA = deck === "a";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const offsetPx = offsetSecs * pps;

    if (waveform.length > 0) {
      const segSecs = durationSecs / waveform.length;
      const midY = H / 2;
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < waveform.length; i++) {
        const x = (i + 0.5) * segSecs * pps + offsetPx;
        if (x < -2 || x > W + 2) { first = true; continue; }
        const amp = waveform[i] * (midY - 6);
        if (first) { ctx.moveTo(x, midY - amp); first = false; }
        else ctx.lineTo(x, midY - amp);
      }
      for (let i = waveform.length - 1; i >= 0; i--) {
        const x = (i + 0.5) * segSecs * pps + offsetPx;
        if (x < -2 || x > W + 2) continue;
        ctx.lineTo(x, midY + waveform[i] * (midY - 6));
      }
      ctx.closePath();
      ctx.fillStyle = isA ? "rgba(56,189,248,0.18)" : "rgba(245,166,35,0.18)";
      ctx.fill();
      ctx.strokeStyle = isA ? "rgba(56,189,248,0.85)" : "rgba(245,166,35,0.85)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (beatTimes.length > 0) {
      const barColor = isA ? "rgba(56,189,248,0.55)" : "rgba(245,158,11,0.55)";
      const beatColor = isA ? "rgba(56,189,248,0.22)" : "rgba(245,158,11,0.22)";
      for (let i = 0; i < beatTimes.length; i++) {
        const x = Math.round(beatTimes[i] * pps + offsetPx) + 0.5;
        if (x < 0 || x > W) continue;
        const isBar = isDownbeat(i, beatPhase);
        // Downbeats sit visibly stronger than beats so a wrong phase shows.
        ctx.strokeStyle = isBar ? barColor : beatColor;
        ctx.lineWidth = isBar ? 2.5 : 1;
        ctx.beginPath();
        ctx.moveTo(x, isBar ? 13 : TIMELINE_HEIGHT * 0.3);
        ctx.lineTo(x, isBar ? TIMELINE_HEIGHT : TIMELINE_HEIGHT * 0.75);
        ctx.stroke();
      }
    }
  }, [waveform, beatTimes, beatPhase, pps, durationSecs, offsetSecs, isA]);

  const handleMouseDown = (e) => {
    if (!onOffsetChange) return;
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startOffset: offsetSecs };
    setDragging(true);
    const onMove = (me) => {
      if (!dragRef.current) return;
      const raw = dragRef.current.startOffset + (me.clientX - dragRef.current.startX) / pps;
      const next = snapMode === "off"
        ? raw
        : snapOffsetToBeats(raw, beatTimes, otherBeatTimes, otherOffsetSecs, pps, snapMode);
      setSnapped(next !== raw);
      onOffsetChange(next);
    };
    const onUp = () => {
      setDragging(false); setSnapped(false); dragRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const playheadPct = playheadPos == null ? null : (playheadPos * pps / TIMELINE_WIDTH) * 100;

  return (
    <div className="lane">
      <div className={`lane-side ${isA ? "vocal" : "bed"}`}>
        <div className="row">
          <button className={`lane-play ${isA ? "vocal" : "bed"}`} onClick={onPlay} disabled={!onPlay}
            title={isPlaying ? "Pause" : "Solo this deck"}>
            {isPlaying ? "❚❚" : "▶"}
          </button>
          <span className="name">{isA ? "Deck A" : "Deck B"}</span>
        </div>
        <div className="offset">offset {offsetSecs >= 0 ? "+" : ""}{offsetSecs.toFixed(2)}s</div>
        {onOffsetChange && (
          <div className="nudge-row">
            <button className="nudge-btn" title={`Nudge back one beat${beatSecs ? ` (${(beatSecs * 1000).toFixed(0)} ms)` : ""}`}
              onClick={() => onOffsetChange(offsetSecs - (beatSecs || 0.1))}>«</button>
            <button className="nudge-btn" title="Nudge back 10 ms"
              onClick={() => onOffsetChange(offsetSecs - 0.01)}>‹</button>
            <button className="nudge-btn" title="Nudge forward 10 ms"
              onClick={() => onOffsetChange(offsetSecs + 0.01)}>›</button>
            <button className="nudge-btn" title={`Nudge forward one beat${beatSecs ? ` (${(beatSecs * 1000).toFixed(0)} ms)` : ""}`}
              onClick={() => onOffsetChange(offsetSecs + (beatSecs || 0.1))}>»</button>
          </div>
        )}
      </div>
      <div
        className={`lane-track${dragging ? " dragging" : ""}${snapped ? " snapped" : ""}`}
        onMouseDown={handleMouseDown}
      >
        <canvas ref={canvasRef} width={TIMELINE_WIDTH} height={TIMELINE_HEIGHT} />
        <div className="section-ribbon" style={{ width: durationSecs * pps, transform: `translateX(${offsetSecs * pps}px)` }}>
          {sections.map((sec) => (
            <span
              key={sec.id}
              style={{
                left: sec.start_sec * pps,
                width: Math.max((sec.end_sec - sec.start_sec) * pps, 3),
                background: SECTION_COLORS[sec.label] ?? "#3b82f6",
              }}
              title={`${sec.label} ${fmtTime(sec.start_sec)}–${fmtTime(sec.end_sec)}`}
            />
          ))}
        </div>
        {loop && (
          <div className="loop" style={{
            position: "absolute", top: 0, bottom: 0, pointerEvents: "none",
            left: `${(loop.start * pps / TIMELINE_WIDTH) * 100}%`,
            width: `${((loop.end - loop.start) * pps / TIMELINE_WIDTH) * 100}%`,
            background: isA ? "rgba(56,189,248,0.12)" : "rgba(245,166,35,0.12)",
            borderLeft: `2px solid ${isA ? "rgba(56,189,248,0.7)" : "rgba(245,166,35,0.7)"}`,
            borderRight: `2px solid ${isA ? "rgba(56,189,248,0.7)" : "rgba(245,166,35,0.7)"}`,
          }} />
        )}
        {playheadPct != null && playheadPct >= 0 && playheadPct <= 100 && (
          <div className="lane-playhead" style={{ left: `${playheadPct}%` }} />
        )}
      </div>
    </div>
  );
}

export function AuditionStudio({ seed, onStatus }) {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);

  // ── Per-deck source: track id + which stem to play ────────────────────────
  const [aId, setAId] = useState(seed?.vocalId ?? null);
  const [bId, setBId] = useState(seed?.instId ?? null);
  const [aStem, setAStem] = useState("vocals");
  const [bStem, setBStem] = useState("instrumental");

  // Per-deck tempo (target BPM), key (semitones), mix (vol + mute)
  const [aTarget, setATarget] = useState(null);
  const [bTarget, setBTarget] = useState(null);
  const [aShift, setAShift] = useState(0);
  const [bShift, setBShift] = useState(0);
  const [aVol, setAVol] = useState(0.85);
  const [bVol, setBVol] = useState(0.8);
  const [aMute, setAMute] = useState(false);
  const [bMute, setBMute] = useState(false);
  const [cross, setCross] = useState(0.5);

  // Per-deck alignment + loops
  const [aOffset, setAOffset] = useState(0);
  const [bOffset, setBOffset] = useState(0);
  const [aLoopBars, setALoopBars] = useState(0); // 0 = off
  const [bLoopBars, setBLoopBars] = useState(0);
  const [aLoop, setALoop] = useState(null);
  const [bLoop, setBLoop] = useState(null);
  const [snapMode, setSnapMode] = useState("beats");

  const [aSections, setASections] = useState([]);
  const [bSections, setBSections] = useState([]);
  const [aWave, setAWave] = useState({ waveform: [], beat_times: [] });
  const [bWave, setBWave] = useState({ waveform: [], beat_times: [] });

  const [menu, setMenu] = useState(null); // "a" | "b" | null
  const [pickerSearch, setPickerSearch] = useState("");

  // Audio engine + decoded buffers
  const engineRef = useRef(null);
  const [aBuffer, setABuffer] = useState(null);
  const [bBuffer, setBBuffer] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null);

  // Transport
  const [position, setPosition] = useState(null);
  const [deckPos, setDeckPos] = useState({ a: null, b: null });
  const [isPlaying, setIsPlaying] = useState(false);
  const [soloDeck, setSoloDeck] = useState(null); // null = both, "a" | "b"

  // Export
  const [exportJobId, setExportJobId] = useState(null);
  const [exportReady, setExportReady] = useState(null);

  useEffect(() => {
    const engine = new MashupEngine();
    engineRef.current = engine;
    engine.init().catch((e) => setAudioError(`Audio engine: ${e.message}`));
    engine.onTick((pos, playing) => {
      setPosition(pos); setIsPlaying(playing);
      setDeckPos({ a: engine.voicePosition("a"), b: engine.voicePosition("b") });
    });
    return () => { engineRef.current?.dispose(); engineRef.current = null; };
  }, []);

  useEffect(() => {
    api.getTracks().then((d) => setTracks(d.tracks)).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (seed?.vocalId != null) { setAId(seed.vocalId); setAStem("vocals"); }
    if (seed?.instId != null) { setBId(seed.instId); setBStem("instrumental"); }
  }, [seed]);

  const analysedTracks = useMemo(() => tracks.filter((t) => t.features?.full), [tracks]);
  const aTrack = useMemo(() => tracks.find((t) => t.id === aId), [tracks, aId]);
  const bTrack = useMemo(() => tracks.find((t) => t.id === bId), [tracks, bId]);

  // Native BPM for a deck's chosen stem (stems-first with fallback for vocals).
  const nativeBpm = (track, stem) => {
    if (!track) return null;
    const full = track.features?.full;
    if (stem === "vocals") {
      const vf = track.features?.vocals;
      if (vf?.bpm && (vf?.bpm_confidence ?? 0) >= VOCAL_BPM_CONFIDENCE_MIN) return vf.bpm;
      return full?.bpm ?? null;
    }
    if (stem === "instrumental") return track.features?.instrumental?.bpm ?? full?.bpm ?? null;
    return full?.bpm ?? null;
  };
  const stemAvailable = (track, stem) =>
    stem === "full" ? !!track?.stems?.full : !!track?.stems?.[stem];

  const aNativeBpm = nativeBpm(aTrack, aStem);
  const bNativeBpm = nativeBpm(bTrack, bStem);
  const aCamelot = aTrack?.features?.full?.camelot;
  const bCamelot = bTrack?.features?.full?.camelot;

  // Default the target BPM to native when a deck (re)loads.
  useEffect(() => { setATarget(aNativeBpm ?? null); }, [aId, aStem, aNativeBpm]);
  useEffect(() => { setBTarget(bNativeBpm ?? null); }, [bId, bStem, bNativeBpm]);

  const clampRate = (r) => Math.max(0.5, Math.min(2, r));
  const aRate = aTarget && aNativeBpm ? clampRate(aTarget / aNativeBpm) : 1;
  const bRate = bTarget && bNativeBpm ? clampRate(bTarget / bNativeBpm) : 1;
  const aFactor = 1 / aRate; // content secs → display secs
  const bFactor = 1 / bRate;
  const aEffBpm = aNativeBpm ? aNativeBpm * aRate : null;
  const bEffBpm = bNativeBpm ? bNativeBpm * bRate : null;

  const aShiftC = Math.max(-24, Math.min(24, Math.round(Number(aShift) || 0)));
  const bShiftC = Math.max(-24, Math.min(24, Math.round(Number(bShift) || 0)));

  const aAudioSecs = aBuffer?.duration ?? aTrack?.duration_secs ?? 0;
  const bAudioSecs = bBuffer?.duration ?? bTrack?.duration_secs ?? 0;
  const aDisplayDur = aAudioSecs * aFactor;
  const bDisplayDur = bAudioSecs * bFactor;

  const pps = useMemo(() => {
    const maxDur = Math.max(aDisplayDur, bDisplayDur, 1);
    return TIMELINE_WIDTH / maxDur;
  }, [aDisplayDur, bDisplayDur]);

  const centerGlobal = TIMELINE_WIDTH / (2 * pps);
  const playPos = position ?? centerGlobal;

  const aDisplaySections = useMemo(
    () => aSections.map((s) => ({ ...s, start_sec: s.start_sec * aFactor, end_sec: s.end_sec * aFactor })),
    [aSections, aFactor]);
  const bDisplaySections = useMemo(
    () => bSections.map((s) => ({ ...s, start_sec: s.start_sec * bFactor, end_sec: s.end_sec * bFactor })),
    [bSections, bFactor]);
  const aBeatDisplay = useMemo(
    () => (aWave.beat_times || []).map((t) => t * aFactor), [aWave.beat_times, aFactor]);
  const bBeatDisplay = useMemo(
    () => (bWave.beat_times || []).map((t) => t * bFactor), [bWave.beat_times, bFactor]);
  // Which beat of the bar the grid starts on (T1.4); 0 = pre-beat_phase track.
  const aBeatPhase = aWave.beat_phase || 0;
  const bBeatPhase = bWave.beat_phase || 0;

  // ── Load metadata + decode audio when a deck's track/stem changes ─────────
  const loadDeck = (id, stem, setSections, setWave, setBuffer) => {
    setSections([]); setWave({ waveform: [], beat_times: [] }); setBuffer(null);
    if (!id) return () => {};
    api.getSections(id).then((d) => setSections(d.sections)).catch(() => {});
    api.getWaveform(id, stem).then(setWave).catch(() => {});
    let cancelled = false;
    setAudioLoading(true); setAudioError(null);
    engineRef.current.init()
      .then(() => decodeStem(engineRef.current.ctx, api.audioUrl(id, stem)))
      .then((buf) => { if (!cancelled) setBuffer(buf); })
      .catch((e) => { if (!cancelled) setAudioError(`Deck audio: ${e.message}`); })
      .finally(() => { if (!cancelled) setAudioLoading(false); });
    return () => { cancelled = true; };
  };

  useEffect(() => {
    engineRef.current?.stop(); setALoopBars(0); setALoop(null); setAOffset(0); setPosition(null);
    return loadDeck(aId, aStem, setASections, setAWave, setABuffer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aId, aStem]);

  useEffect(() => {
    engineRef.current?.stop(); setBLoopBars(0); setBLoop(null); setBOffset(0); setPosition(null);
    return loadDeck(bId, bStem, setBSections, setBWave, setBBuffer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bId, bStem]);

  useEffect(() => { setExportJobId(null); setExportReady(null); }, [aId, bId, aStem, bStem]);

  // Effective per-voice gains from the mix bus (+ optional solo mask).
  const gainsFor = (solo = soloDeck) => {
    const xfA = Math.min(1, 2 * (1 - cross));
    const xfB = Math.min(1, 2 * cross);
    let a = aMute ? 0 : aVol * xfA;
    let b = bMute ? 0 : bVol * xfB;
    if (solo === "a") b = 0;
    if (solo === "b") a = 0;
    return { a, b };
  };

  const rateRef = useRef({ a: aRate, b: bRate });
  rateRef.current = { a: aRate, b: bRate };
  const shiftRef = useRef({ a: aShiftC, b: bShiftC });
  shiftRef.current = { a: aShiftC, b: bShiftC };
  const gainsRef = useRef(gainsFor());
  gainsRef.current = gainsFor();

  // Structural sync: buffers + alignment offsets → engine voices.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    const { a: ra, b: rb } = rateRef.current;
    const { a: sa, b: sb } = shiftRef.current;
    const { a: ga, b: gb } = gainsRef.current;
    if (aBuffer) engine.setVoice("a", { buffer: aBuffer, offsetSec: aOffset, rate: ra, semitones: sa, gain: ga });
    else engine.removeVoice("a");
    if (bBuffer) engine.setVoice("b", { buffer: bBuffer, offsetSec: bOffset, rate: rb, semitones: sb, gain: gb });
    else engine.removeVoice("b");
    engine.refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aBuffer, bBuffer, aOffset, bOffset, soloDeck]);

  // Mix bus: apply gains live on every slider/mute/crossfade change.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    const { a, b } = gainsFor();
    engine.setVoiceGain("a", a);
    engine.setVoiceGain("b", b);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aVol, bVol, cross, aMute, bMute, soloDeck]);

  // Pitch: live, no re-arm.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    engine.updateVoiceParams("a", { semitones: aShiftC });
    engine.updateVoiceParams("b", { semitones: bShiftC });
  }, [aShiftC, bShiftC]);

  // Tempo: debounce (re-arms the voice — audible if fired per tick).
  const rateDebounceRef = useRef(null);
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    if (rateDebounceRef.current) clearTimeout(rateDebounceRef.current);
    rateDebounceRef.current = setTimeout(() => {
      engine.updateVoiceParams("a", { rate: aRate });
      engine.updateVoiceParams("b", { rate: bRate });
    }, 150);
    return () => clearTimeout(rateDebounceRef.current);
  }, [aRate, bRate]);

  // Per-deck loops → engine.
  useEffect(() => { engineRef.current?.setVoiceLoop("a", aLoop); }, [aLoop]);
  useEffect(() => { engineRef.current?.setVoiceLoop("b", bLoop); }, [bLoop]);

  // ── Beat-lock / status readout ────────────────────────────────────────────
  const matched = (() => {
    if (!aEffBpm || !bEffBpm) return false;
    const ratios = [1, 2, 0.5, 1.5, 2 / 3, 4 / 3, 3 / 4];
    return ratios.some((r) => Math.abs(bEffBpm * r - aEffBpm) / aEffBpm < 0.02);
  })();

  useEffect(() => {
    if (aEffBpm && bEffBpm) {
      onStatus?.(matched
        ? { locked: true, text: `BEAT-LOCKED · A ${aEffBpm.toFixed(1)} · B ${bEffBpm.toFixed(1)} BPM` }
        : { text: `Match tempo to lock · A ${aEffBpm.toFixed(1)} vs B ${bEffBpm.toFixed(1)}` });
    } else {
      onStatus?.({ text: "Load two decks" });
    }
  }, [matched, aEffBpm, bEffBpm, onStatus]);

  // ── Transport ─────────────────────────────────────────────────────────────
  const startPlayback = async (solo) => {
    setError(null); setAudioError(null);
    setSoloDeck(solo);
    const engine = engineRef.current;
    if (!engine) return;
    const { a, b } = gainsFor(solo);
    engine.setVoiceGain("a", a);
    engine.setVoiceGain("b", b);
    try { await engine.play(position ?? centerGlobal); }
    catch (e) { setAudioError(`Playback: ${e.message}`); }
  };
  const handlePlayBoth = () => {
    if (isPlaying && soloDeck === null) { engineRef.current?.pause(); return; }
    startPlayback(null);
  };
  const handleSolo = (deck) => {
    if (isPlaying && soloDeck === deck) { engineRef.current?.pause(); return; }
    startPlayback(deck);
  };

  const handleScrub = (e, el) => {
    const pos = eventPos(e, el, pps);
    setPosition(pos); engineRef.current?.seek(pos);
    const onMove = (me) => { const p2 = eventPos(me, el, pps); setPosition(p2); engineRef.current?.seek(p2); };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const swapDecks = () => {
    setAId(bId); setBId(aId); setAStem(bStem); setBStem(aStem);
  };

  // ── Tempo/key helpers ───────────────────────────────────────────────────
  const matchTempoToOther = (deck) => {
    if (!aEffBpm || !bEffBpm || !aNativeBpm || !bNativeBpm) { toast("Analyze both decks first"); return; }
    if (deck === "a") setATarget(Number((bEffBpm).toFixed(2)));
    else setBTarget(Number((aEffBpm).toFixed(2)));
    toast("Tempo matched");
  };
  const kr = keyRel(shiftCamelot(aCamelot, aShiftC), shiftCamelot(bCamelot, bShiftC));
  const matchKeyToOther = (deck) => {
    // Move `deck` so its (shifted) key equals the OTHER deck's (shifted) key.
    const base = keyRel(aCamelot, bCamelot);
    if (!aCamelot || !bCamelot) { toast("Key data missing"); return; }
    if (deck === "b") setBShift(aShiftC + (base.suggest ?? 0));
    else setAShift(bShiftC - (base.suggest ?? 0));
    toast("Key matched");
  };

  // ── Per-deck loop (1/2/4/8 bars) ──────────────────────────────────────────
  const beatSecsA = aEffBpm ? (60 / aNativeBpm) * aFactor : 0; // == 60/aEffBpm
  const beatSecsB = bEffBpm ? (60 / bNativeBpm) * bFactor : 0;
  const toggleLoop = (deck) => (bars) => {
    const isA = deck === "a";
    const curBars = isA ? aLoopBars : bLoopBars;
    const setBars = isA ? setALoopBars : setBLoopBars;
    const setLoopWin = isA ? setALoop : setBLoop;
    const effBpm = isA ? aEffBpm : bEffBpm;
    const beats = isA ? aBeatDisplay : bBeatDisplay;
    const offset = isA ? aOffset : bOffset;
    if (curBars === bars) { setBars(0); setLoopWin(null); return; } // toggle off
    if (!effBpm) { toast("Analyze this deck first"); return; }
    const barSecs = (4 * 60) / effBpm; // display seconds per bar
    // Snap the loop start to this deck's nearest downbeat at/under the playhead.
    const phase = isA ? aBeatPhase : bBeatPhase;
    const downbeats = downbeatsOf(beats, phase).map((t) => t + offset);
    let start = playPos;
    if (downbeats.length) {
      const before = downbeats.filter((d) => d <= playPos + 0.05);
      start = before.length ? before[before.length - 1] : downbeats[0];
    }
    setBars(bars);
    setLoopWin({ start, end: start + bars * barSecs });
  };

  const alignDownbeats = () => {
    const aDown = downbeatsOf(aBeatDisplay, aBeatPhase).map((t) => t + aOffset);
    const bDown = downbeatsOf(bBeatDisplay, bBeatPhase).map((t) => t + bOffset);
    if (aDown.length === 0 || bDown.length === 0) { toast("Beat data missing — analyze both decks"); return; }
    const aAt = nearest(aDown, playPos);
    const bAt = nearest(bDown, playPos);
    setAOffset(aOffset + (bAt - aAt)); // move A onto B's downbeat
    toast("Deck A downbeat aligned to Deck B");
  };

  // ── Keyboard shortcuts (space play, arrows nudge A) ────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        if (aBuffer && bBuffer) handlePlayBoth();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const dir = e.key === "ArrowLeft" ? -1 : 1;
        const step = e.shiftKey ? 0.01 : (beatSecsA || 0.1);
        setAOffset((o) => o + dir * step);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const resetAll = () => {
    engineRef.current?.stop();
    setAOffset(0); setBOffset(0); setPosition(null);
    setALoopBars(0); setBLoopBars(0); setALoop(null); setBLoop(null);
    setAVol(0.85); setBVol(0.8); setCross(0.5); setAMute(false); setBMute(false);
    setAShift(0); setBShift(0);
    setATarget(aNativeBpm ?? null); setBTarget(bNativeBpm ?? null);
    toast("Reset to defaults");
  };

  const handleExport = async () => {
    setError(null);
    if (aId == null || bId == null) return;
    try {
      const { a, b } = gainsFor(null);
      const { job_id } = await api.startAuditionExport({
        aId, bId, aStem, bStem, aRate, bRate, aShift: aShiftC, bShift: bShiftC,
        aOffset, bOffset, aGain: a, bGain: b,
      });
      setExportReady(null); setExportJobId(job_id);
      toast("Rendering mashup WAV…");
    } catch (e) { setError(e.message); }
  };

  // ── Derived UI ─────────────────────────────────────────────────────────────
  const bothLoaded = aTrack && bTrack;
  const samePair = aId != null && bId === aId && aStem === bStem;
  const dbLabel = (vol, mute) => (mute ? "MUTE" : `${(vol * 24 - 12).toFixed(1)}dB`);

  const pickerList = analysedTracks
    .filter((t) => !pickerSearch || trackLabel(t).toLowerCase().includes(pickerSearch.toLowerCase()));

  const StemPicker = ({ track, stem, onPick }) => (
    <div className="stem-seg" style={{ display: "inline-flex", gap: 2 }}>
      {STEMS.map((s) => {
        const ok = stemAvailable(track, s.value);
        return (
          <button key={s.value}
            className={`snap-seg-btn${stem === s.value ? " active" : ""}`}
            disabled={!ok}
            title={ok ? `Play the ${s.label.toLowerCase()} stem` : `${s.label} not separated for this track`}
            onClick={(e) => { e.stopPropagation(); onPick(s.value); }}
            style={{
              fontSize: 11, padding: "2px 8px", borderRadius: 5, cursor: ok ? "pointer" : "not-allowed",
              opacity: ok ? 1 : 0.4,
              border: "1px solid var(--border-ctrl)",
              background: stem === s.value ? "var(--accent, #38bdf8)" : "transparent",
              color: stem === s.value ? "#04121b" : "var(--text)",
            }}>
            {s.label}
          </button>
        );
      })}
    </div>
  );

  const Selector = ({ deck, track, stem, onStem, bpm, effBpm, onOpen }) => {
    const isA = deck === "a";
    return (
      <div className={`track-select ${isA ? "vocal" : "bed"}`}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }} onClick={onOpen}>
          <span className="role">{isA ? "DECK A" : "DECK B"}</span>
          <div className="info">
            <div className="t">{track ? track.title : "— select track —"}</div>
            <div className="a">{track?.artist || ""}</div>
          </div>
          {bpm != null && (
            <div className="bpm">
              {(effBpm ?? bpm).toFixed(1)} BPM
              {(bpm < 80 || bpm > 170) && (
                <span className="bpm-warn" title="Unusual tempo — likely a half/double-time detection error.">{" ⚠"}</span>
              )}
            </div>
          )}
          {track?.features?.full?.camelot && (
            <KeyChip camelot={track.features.full.camelot} style={{ padding: "4px 9px" }} />
          )}
          <span className="caret">▾</span>
        </div>
        {track && (
          <div style={{ marginTop: 6 }}>
            <StemPicker track={track} stem={stem} onPick={onStem} />
          </div>
        )}
      </div>
    );
  };

  const TempoModule = ({ deck, effBpm, target, setTarget }) => (
    <div className="module">
      <div className="module-head">
        <span className="micro-label">TEMPO · {deck === "a" ? "A" : "B"}</span>
        <span className="val">{effBpm ? effBpm.toFixed(1) : "—"}<span className="u"> BPM</span></span>
      </div>
      <div className="slider-row">
        <button className="step-btn" onClick={() => setTarget((t) => Math.max(40, (Number(t) || 0) - 0.5))}>−</button>
        <input type="number" step="0.5" value={target != null ? Number(target).toFixed(1) : ""}
          onChange={(e) => setTarget(parseFloat(e.target.value) || 0)}
          style={{ width: 72, textAlign: "center" }} />
        <button className="step-btn" onClick={() => setTarget((t) => (Number(t) || 0) + 0.5)}>+</button>
      </div>
      <div className="module-cta cyan" onClick={() => matchTempoToOther(deck)}>⚡ Match to Deck {deck === "a" ? "B" : "A"}</div>
    </div>
  );

  const KeyModule = ({ deck, camelot, shift, setShift }) => {
    const shown = shiftCamelot(camelot, deck === "a" ? aShiftC : bShiftC);
    return (
      <div className="module">
        <div className="module-head">
          <span className="micro-label">KEY · {deck === "a" ? "A" : "B"}</span>
          <span className="val">{shift > 0 ? "+" : ""}{shift}<span className="u"> st</span></span>
        </div>
        <div className="key-map">
          <KeyChip camelot={camelot} as="span" fallback="?" style={{ padding: "5px 10px", opacity: 0.6 }} />
          <span className="arrow">→</span>
          <KeyChip camelot={shown} as="span" fallback="?" style={{ padding: "5px 10px" }} />
        </div>
        <div className="slider-row">
          <button className="step-btn" onClick={() => setShift((s) => Math.max(-24, (Number(s) || 0) - 1))}>−</button>
          <div className="track-bar" onPointerDown={(e) => dragFraction(e, (f) => setShift(Math.round(-24 + f * 48)))}>
            <div className="knob violet" style={{ left: `${((shift + 24) / 48) * 100}%` }} />
          </div>
          <button className="step-btn" onClick={() => setShift((s) => Math.min(24, (Number(s) || 0) + 1))}>+</button>
        </div>
        <div className="module-cta ghost" onClick={() => matchKeyToOther(deck)}>Match key to Deck {deck === "a" ? "B" : "A"}</div>
      </div>
    );
  };

  const LoopSeg = ({ bars, onBar }) => (
    <div className="snap-row">
      <span>Loop</span>
      <div className="snap-seg">
        {LOOP_BARS.map((n) => (
          <button key={n} className={bars === n ? "active" : ""} onClick={() => onBar(n)}>{n}</button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="page audition">
      {error && <div className="error-text" style={{ marginBottom: 8 }}>{error}</div>}
      {audioError && <div className="error-text" style={{ marginBottom: 8 }}>{audioError}</div>}

      {/* deck selectors */}
      <div className="aud-selectors">
        <Selector deck="a" track={aTrack} stem={aStem} onStem={setAStem} bpm={aNativeBpm} effBpm={aEffBpm}
          onOpen={() => { setMenu(menu === "a" ? null : "a"); setPickerSearch(""); }} />
        <div className="swap-btn" onClick={swapDecks} title="Swap decks">⇄</div>
        <Selector deck="b" track={bTrack} stem={bStem} onStem={setBStem} bpm={bNativeBpm} effBpm={bEffBpm}
          onOpen={() => { setMenu(menu === "b" ? null : "b"); setPickerSearch(""); }} />

        {menu && <div className="picker-overlay" onClick={() => setMenu(null)} />}
        {menu && (
          <div className="picker-menu" style={{ left: menu === "a" ? 0 : "auto", right: menu === "b" ? 0 : "auto" }}>
            <div className="search-box" style={{ width: "auto", margin: "2px 4px 6px" }}>
              <span className="ico">⌕</span>
              <input autoFocus placeholder="Search…" value={pickerSearch}
                onChange={(e) => setPickerSearch(e.target.value)} onClick={(e) => e.stopPropagation()} />
            </div>
            {pickerList.map((t) => {
              const selected = menu === "a" ? t.id === aId : t.id === bId;
              return (
                <div key={t.id} className={`picker-row${selected ? " selected" : ""}`}
                  onClick={() => { if (menu === "a") setAId(t.id); else setBId(t.id); setMenu(null); }}>
                  <TrackArt id={t.id} thumbnail={t.thumbnail} className="art" />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="t">{t.title}</div>
                    <div className="a">{t.artist || "—"}</div>
                  </div>
                  <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{t.features?.full?.bpm?.toFixed(1)}</div>
                  {t.features?.full?.camelot && (
                    <KeyChip camelot={t.features.full.camelot} style={{ fontSize: 11, padding: "2px 6px" }} />
                  )}
                </div>
              );
            })}
            {pickerList.length === 0 && <div className="empty" style={{ padding: 12 }}>No analysed tracks.</div>}
          </div>
        )}
      </div>

      {analysedTracks.length === 0 && (
        <p className="empty">No analysed tracks yet. Separate and analyze tracks in the Library tab first.</p>
      )}
      {analysedTracks.length > 0 && !bothLoaded && (
        <div className="aud-empty">
          <div className="step"><span className="n">1</span> Pick a track for <b className="v">Deck A</b> and <b className="b">Deck B</b>, then choose a stem (Vocal / Inst / Full) per deck.</div>
          <div className="step"><span className="n">2</span> Press <span className="kbd">space</span> to play both; drag a lane (or <span className="kbd">←</span><span className="kbd">→</span>) to line up beats.</div>
          <div className="step"><span className="n">3</span> Set each deck's BPM, key and volume, arm a 1/2/4/8-bar loop, then export as WAV.</div>
        </div>
      )}
      {samePair && <div className="error-text" style={{ marginBottom: 8 }}>Both decks are the same track + stem — pick a different source.</div>}
      {audioLoading && <p className="hint" style={{ marginBottom: 8 }}>Decoding stems for playback…</p>}

      {bothLoaded && (
        <>
          {/* WAVEFORM STACK */}
          <div className="wave-panel">
            <div className="wave-legend">
              {["verse", "chorus", "drop", "breakdown"].map((l) => (
                <span key={l} className="sw"><i style={{ background: SECTION_COLORS[l] }} />{l}</span>
              ))}
              <button className="align-btn" onClick={alignDownbeats}
                title="Snap Deck A's nearest downbeat to Deck B's">⇥ Align downbeats</button>
              <span className="hint">
                Drag a lane to nudge · <span className="kbd">space</span> play ·{" "}
                <span className="kbd">←</span><span className="kbd">→</span> nudge Deck A (shift = fine)
              </span>
            </div>

            <div className="ruler" onMouseDown={(e) => handleScrub(e, e.currentTarget)}
              title="Click or drag to scrub">
              <div className="playhead" style={{ left: `${Math.min(100, Math.max(0, (playPos * pps / TIMELINE_WIDTH) * 100))}%` }} />
            </div>

            <Lane
              deck="a" sections={aDisplaySections} durationSecs={aDisplayDur}
              pps={pps} offsetSecs={aOffset} onOffsetChange={setAOffset}
              waveform={aWave.waveform} beatTimes={aBeatDisplay} beatPhase={aBeatPhase} loop={aLoop}
              onPlay={aBuffer ? () => handleSolo("a") : null}
              isPlaying={isPlaying && soloDeck === "a"} playheadPos={deckPos.a ?? playPos}
              otherBeatTimes={bBeatDisplay} otherOffsetSecs={bOffset} snapMode={snapMode}
              beatSecs={beatSecsA}
            />
            <Lane
              deck="b" sections={bDisplaySections} durationSecs={bDisplayDur}
              pps={pps} offsetSecs={bOffset} onOffsetChange={setBOffset}
              waveform={bWave.waveform} beatTimes={bBeatDisplay} beatPhase={bBeatPhase} loop={bLoop}
              onPlay={bBuffer ? () => handleSolo("b") : null}
              isPlaying={isPlaying && soloDeck === "b"} playheadPos={deckPos.b ?? playPos}
              otherBeatTimes={aBeatDisplay} otherOffsetSecs={aOffset} snapMode={snapMode}
              beatSecs={beatSecsB}
            />

            <div className="wave-readout">
              A {fmtTime(Math.max(0, (deckPos.a ?? playPos) - aOffset))} · B {fmtTime(Math.max(0, (deckPos.b ?? playPos) - bOffset))} ·{" "}
              <span style={{ color: matched ? "var(--green)" : "var(--amber-light)" }}>
                {matched ? "grids aligned ✓" : "grids drift — match tempo"}
              </span>
            </div>
          </div>

          {/* COMMAND BAR */}
          <div className="command-bar">
            <TempoModule deck="a" effBpm={aEffBpm} target={aTarget} setTarget={setATarget} />
            <KeyModule deck="a" camelot={aCamelot} shift={aShiftC} setShift={setAShift} />
            <TempoModule deck="b" effBpm={bEffBpm} target={bTarget} setTarget={setBTarget} />
            <KeyModule deck="b" camelot={bCamelot} shift={bShiftC} setShift={setBShift} />

            {/* MIX */}
            <div className="module mix">
              <div className="module-head"><span className="micro-label">MIX</span></div>
              <div className="mix-row">
                <span className="lab v">A</span>
                <div className="track-bar thick" onPointerDown={(e) => dragFraction(e, setAVol)}>
                  <div className="fill" style={{ width: `${aVol * 100}%`, background: "var(--cyan)" }} />
                  <div className="knob cyan" style={{ left: `${aVol * 100}%`, boxShadow: "none" }} />
                </div>
                <span className="db">{dbLabel(aVol, aMute)}</span>
                <button className={`mute-btn${aMute ? " on" : ""}`} onClick={() => setAMute((m) => !m)}>M</button>
              </div>
              <div className="mix-row">
                <span className="lab b">B</span>
                <div className="track-bar thick" onPointerDown={(e) => dragFraction(e, setBVol)}>
                  <div className="fill" style={{ width: `${bVol * 100}%`, background: "var(--amber)" }} />
                  <div className="knob" style={{ left: `${bVol * 100}%`, background: "var(--amber)" }} />
                </div>
                <span className="db">{dbLabel(bVol, bMute)}</span>
                <button className={`mute-btn${bMute ? " on" : ""}`} onClick={() => setBMute((m) => !m)}>M</button>
              </div>
              <div className="crossfade-head">
                <span className="v">DECK A</span><span>CROSSFADE</span><span className="b">DECK B</span>
              </div>
              <div className="crossfade" onPointerDown={(e) => dragFraction(e, setCross)}>
                <div className="knob" style={{ left: `${cross * 100}%` }} />
              </div>
            </div>

            {/* TRANSPORT + LOOPS */}
            <div className="module transport">
              <span className="micro-label">TRANSPORT</span>
              <button className={`play-btn ${isPlaying && soloDeck === null ? "playing" : "stopped"}`}
                onClick={handlePlayBoth} disabled={!aBuffer || !bBuffer}>
                {isPlaying && soloDeck === null ? "❚❚ Pause" : "▶ Play both"}
              </button>
              <LoopSeg bars={aLoopBars} onBar={toggleLoop("a")} />
              <LoopSeg bars={bLoopBars} onBar={toggleLoop("b")} />
              <div className="snap-row">
                <span>Snap</span>
                <div className="snap-seg">
                  {["beats", "bars", "off"].map((o) => (
                    <button key={o} className={snapMode === o ? "active" : ""} onClick={() => setSnapMode(o)}>{o}</button>
                  ))}
                </div>
              </div>
              <div className="btn-row">
                <button className="reset-btn" onClick={resetAll}>↺ Reset</button>
              </div>
              <div className="spacer" style={{ flex: 1 }} />
              <button className="export-btn" onClick={handleExport} disabled={exportJobId != null || samePair}>
                ↓ Export mashup WAV
              </button>
              {exportJobId && (
                <JobBadge jobId={exportJobId} onComplete={(job) => {
                  const token = exportJobId; // studio mixdown token == job id
                  setExportJobId(null);
                  if (job.status === "completed") setExportReady({ token });
                }} />
              )}
              {exportReady && exportReady.token && (
                <a href={api.auditionExportAudioUrl(exportReady.token)} target="_blank" rel="noreferrer"
                  className="muted" style={{ fontSize: 12, textAlign: "center" }}>
                  ↓ download / play export
                </a>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
