import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { MashupEngine } from "../engine/MashupEngine";
import { decodeStem } from "../engine/decode";
import { artGradient, camelotColor, fmtTime, keyRel, SHOW_NOTES } from "../theme";
import { toast } from "../toast";

const SECTION_COLORS = {
  intro: "#6b7280", verse: "#3b82f6", chorus: "#ec4899", drop: "#f59e0b",
  breakdown: "#14b8a6", bridge: "#22c55e", outro: "#6b7280",
};

const TIMELINE_WIDTH = 1200;
const TIMELINE_HEIGHT = 100;

const BEAT_LOCK_RATIOS = [
  { label: "1:1", value: 1 }, { label: "2:1", value: 2 }, { label: "1:2", value: 0.5 },
  { label: "3:2", value: 1.5 }, { label: "2:3", value: 2 / 3 },
  { label: "4:3", value: 4 / 3 }, { label: "3:4", value: 3 / 4 },
];

// Logical-second position under a pointer event, mapped through the element's
// rendered width so it lines up with the canvas drawn at TIMELINE_WIDTH.
function eventPos(e, el, pps) {
  const rect = el.getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  return (frac * TIMELINE_WIDTH) / pps;
}

// 0..1 fraction under a pointer, for the custom module sliders.
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
// track lines up with the other track's nearest beat/downbeat once within snapPx.
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
  const num = ((n - 1 + Math.round(semitones / 7)) % 12 + 12) % 12 + 1;
  return `${num}${letter}`;
}

function trackLabel(t) {
  return `${t.title}${t.artist ? ` — ${t.artist}` : ""}`;
}

// ── One waveform lane (section ribbon + canvas waveform/beat-grid + playhead) ─
function Lane({
  sections, durationSecs, role, pps, offsetSecs, onOffsetChange,
  waveform = [], beatTimes = [], onPlay, isPlaying, playheadPos,
  otherBeatTimes = [], otherOffsetSecs = 0, snapMode = "beats",
}) {
  const [dragging, setDragging] = useState(false);
  const [snapped, setSnapped] = useState(false);
  const dragRef = useRef(null);
  const canvasRef = useRef(null);
  const isVocal = role === "vocal";

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
      ctx.fillStyle = isVocal ? "rgba(56,189,248,0.18)" : "rgba(245,166,35,0.18)";
      ctx.fill();
      ctx.strokeStyle = isVocal ? "rgba(56,189,248,0.85)" : "rgba(245,166,35,0.85)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (beatTimes.length > 0) {
      const barColor = isVocal ? "rgba(56,189,248,0.55)" : "rgba(245,158,11,0.55)";
      const beatColor = isVocal ? "rgba(56,189,248,0.22)" : "rgba(245,158,11,0.22)";
      for (let i = 0; i < beatTimes.length; i++) {
        const x = Math.round(beatTimes[i] * pps + offsetPx) + 0.5;
        if (x < 0 || x > W) continue;
        const isBar = i % 4 === 0;
        ctx.strokeStyle = isBar ? barColor : beatColor;
        ctx.lineWidth = isBar ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(x, isBar ? 13 : TIMELINE_HEIGHT * 0.3);
        ctx.lineTo(x, isBar ? TIMELINE_HEIGHT : TIMELINE_HEIGHT * 0.75);
        ctx.stroke();
      }
    }
  }, [waveform, beatTimes, pps, durationSecs, offsetSecs, isVocal]);

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
      <div className={`lane-side ${role}`}>
        <div className="row">
          <button className={`lane-play ${role}`} onClick={onPlay} disabled={!onPlay}
            title={isPlaying ? "Pause" : "Solo this stem"}>
            {isPlaying ? "❚❚" : "▶"}
          </button>
          <span className="name">{isVocal ? "Vocal" : "Bed"}</span>
        </div>
        <div className="offset">offset {offsetSecs >= 0 ? "+" : ""}{offsetSecs.toFixed(2)}s</div>
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
        {playheadPct != null && playheadPct >= 0 && playheadPct <= 100 && (
          <div className="lane-playhead" style={{ left: `${playheadPct}%` }} />
        )}
      </div>
    </div>
  );
}

const VOCAL_BPM_CONFIDENCE_MIN = 0.35;

export function AuditionStudio({ seed, onStatus }) {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [vocalId, setVocalId] = useState(seed?.vocalId ?? null);
  const [instId, setInstId] = useState(seed?.instId ?? null);
  const [plan, setPlan] = useState(null);
  const [anchor, setAnchor] = useState("instrumental");
  const [stretchInput, setStretchInput] = useState(1);
  const [shiftInput, setShiftInput] = useState(0);

  const [vocalSections, setVocalSections] = useState([]);
  const [instSections, setInstSections] = useState([]);
  const [vocalOffset, setVocalOffset] = useState(0);
  const [instOffset, setInstOffset] = useState(0);
  const [vocalWaveform, setVocalWaveform] = useState({ waveform: [], beat_times: [] });
  const [instWaveform, setInstWaveform] = useState({ waveform: [], beat_times: [] });
  const [snapMode, setSnapMode] = useState("beats");

  // Mix bus (client-side, driven into the engine's per-voice gain nodes)
  const [vocalVol, setVocalVol] = useState(0.8);
  const [bedVol, setBedVol] = useState(0.7);
  const [cross, setCross] = useState(0.5);
  const [vocalMute, setVocalMute] = useState(false);
  const [bedMute, setBedMute] = useState(false);

  const [menu, setMenu] = useState(null); // "vocal" | "bed" | null
  const [pickerSearch, setPickerSearch] = useState("");
  const [loop8, setLoop8] = useState(false);

  // Audio engine + decoded buffers
  const engineRef = useRef(null);
  const [vocalBuffer, setVocalBuffer] = useState(null);
  const [instBuffer, setInstBuffer] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null);

  // Transport
  const [position, setPosition] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [soloRole, setSoloRole] = useState(null); // null = mashup, "vocal" | "inst"
  const [loop, setLoop] = useState(null);

  // Export
  const [exportJobId, setExportJobId] = useState(null);
  const [exportReady, setExportReady] = useState(null);

  useEffect(() => {
    engineRef.current = new MashupEngine();
    engineRef.current.init().catch((e) => setAudioError(`Audio engine: ${e.message}`));
    engineRef.current.onTick((pos, playing) => { setPosition(pos); setIsPlaying(playing); });
    return () => { engineRef.current?.dispose(); engineRef.current = null; };
  }, []);

  useEffect(() => {
    api.getTracks().then((d) => setTracks(d.tracks)).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (seed?.vocalId != null) setVocalId(seed.vocalId);
    if (seed?.instId != null) setInstId(seed.instId);
  }, [seed]);

  const vocalOptions = useMemo(() => tracks.filter((t) => t.stems?.vocals && t.features?.full), [tracks]);
  const instOptions = useMemo(() => tracks.filter((t) => t.stems?.instrumental && t.features?.full), [tracks]);
  const vocalTrack = useMemo(() => tracks.find((t) => t.id === vocalId), [tracks, vocalId]);
  const instTrack = useMemo(() => tracks.find((t) => t.id === instId), [tracks, instId]);

  // Stems-first BPM with fallback (see backend beat-grid rationale).
  const vocalStemFeat = vocalTrack?.features?.vocals;
  const vocalFullFeat = vocalTrack?.features?.full;
  const useVocalStemBpm = vocalStemFeat?.bpm && (vocalStemFeat?.bpm_confidence ?? 0) >= VOCAL_BPM_CONFIDENCE_MIN;
  const vocalBpm = useVocalStemBpm ? vocalStemFeat.bpm : vocalFullFeat?.bpm;
  const instStemFeat = instTrack?.features?.instrumental;
  const instFullFeat = instTrack?.features?.full;
  const instBpm = instStemFeat?.bpm ?? instFullFeat?.bpm;

  const vocalCamelot = vocalFullFeat?.camelot;
  const instCamelot = instFullFeat?.camelot;

  const appliedStretch = (() => {
    const s = Number(stretchInput);
    return Number.isFinite(s) && s > 0 ? Math.max(0.5, Math.min(2, s)) : 1;
  })();
  const appliedShift = (() => {
    const s = Math.round(Number(shiftInput));
    return Number.isFinite(s) ? Math.max(-24, Math.min(24, s)) : 0;
  })();

  const vocalDisplayFactor = anchor === "vocal" ? 1 / appliedStretch : 1;
  const instDisplayFactor = anchor === "instrumental" ? 1 / appliedStretch : 1;
  // Lay out the timeline against the ACTUAL decoded stem length, not the song's
  // metadata duration_secs — those can disagree badly (e.g. a SoundCloud 30s
  // preview value on a full ~3min stem), which would compact the waveform into
  // a sliver while the audio plays on for minutes. The playhead follows the
  // buffer, so basing the layout on buffer.duration keeps waveform, section
  // ribbon and playhead aligned. Fall back to metadata until the buffer decodes.
  const vocalAudioSecs = vocalBuffer?.duration ?? vocalTrack?.duration_secs ?? 0;
  const instAudioSecs = instBuffer?.duration ?? instTrack?.duration_secs ?? 0;
  const vocalDisplayDuration = vocalAudioSecs * vocalDisplayFactor;
  const instDisplayDuration = instAudioSecs * instDisplayFactor;

  const pps = useMemo(() => {
    const maxDur = Math.max(vocalDisplayDuration, instDisplayDuration, 1);
    return TIMELINE_WIDTH / maxDur;
  }, [vocalDisplayDuration, instDisplayDuration]);

  const centerGlobal = TIMELINE_WIDTH / (2 * pps);
  const playPos = position ?? centerGlobal;
  const vocalCenterTime = Math.max(0, Math.min(vocalDisplayDuration, playPos - vocalOffset));
  const instCenterTime = Math.max(0, Math.min(instDisplayDuration, playPos - instOffset));

  const vocalDisplaySections = useMemo(
    () => vocalSections.map((s) => ({ ...s, start_sec: s.start_sec * vocalDisplayFactor, end_sec: s.end_sec * vocalDisplayFactor })),
    [vocalSections, vocalDisplayFactor]);
  const instDisplaySections = useMemo(
    () => instSections.map((s) => ({ ...s, start_sec: s.start_sec * instDisplayFactor, end_sec: s.end_sec * instDisplayFactor })),
    [instSections, instDisplayFactor]);
  const vocalDisplayBeatTimes = useMemo(
    () => (vocalWaveform.beat_times || []).map((t) => t * vocalDisplayFactor), [vocalWaveform.beat_times, vocalDisplayFactor]);
  const instDisplayBeatTimes = useMemo(
    () => (instWaveform.beat_times || []).map((t) => t * instDisplayFactor), [instWaveform.beat_times, instDisplayFactor]);

  // ── Load metadata + decode audio when a side changes ──────────────────────
  useEffect(() => {
    setVocalSections([]); setVocalWaveform({ waveform: [], beat_times: [] });
    setVocalOffset(0); setVocalBuffer(null);
    engineRef.current?.stop(); setLoop(null); setLoop8(false); setPosition(null);
    if (!vocalId) return;
    api.getSections(vocalId).then((d) => setVocalSections(d.sections)).catch(() => {});
    api.getWaveform(vocalId, "vocals").then(setVocalWaveform).catch(() => {});
    let cancelled = false;
    setAudioLoading(true); setAudioError(null);
    engineRef.current.init()
      .then(() => decodeStem(engineRef.current.ctx, api.audioUrl(vocalId, "vocals")))
      .then((buf) => { if (!cancelled) setVocalBuffer(buf); })
      .catch((e) => { if (!cancelled) setAudioError(`Vocal audio: ${e.message}`); })
      .finally(() => { if (!cancelled) setAudioLoading(false); });
    return () => { cancelled = true; };
  }, [vocalId]);

  useEffect(() => {
    setInstSections([]); setInstWaveform({ waveform: [], beat_times: [] });
    setInstOffset(0); setInstBuffer(null);
    engineRef.current?.stop(); setLoop(null); setLoop8(false); setPosition(null);
    if (!instId) return;
    api.getSections(instId).then((d) => setInstSections(d.sections)).catch(() => {});
    api.getWaveform(instId, "instrumental").then(setInstWaveform).catch(() => {});
    let cancelled = false;
    setAudioLoading(true); setAudioError(null);
    engineRef.current.init()
      .then(() => decodeStem(engineRef.current.ctx, api.audioUrl(instId, "instrumental")))
      .then((buf) => { if (!cancelled) setInstBuffer(buf); })
      .catch((e) => { if (!cancelled) setAudioError(`Instrumental audio: ${e.message}`); })
      .finally(() => { if (!cancelled) setAudioLoading(false); });
    return () => { cancelled = true; };
  }, [instId]);

  useEffect(() => {
    setPlan(null); setExportJobId(null); setExportReady(null);
    if (vocalId == null || instId == null || vocalId === instId) return;
    let cancelled = false;
    api.getMashupPlan(vocalId, instId)
      .then((p) => !cancelled && setPlan(p))
      .catch((e) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, [vocalId, instId]);

  // Engine-suggested stretch/pitch defaults for the current anchor side.
  useEffect(() => {
    if (!plan) return;
    const planStretch = plan.stretch_factor || 1;
    const planShift = plan.semitone_shift || 0;
    if (anchor === "instrumental") { setStretchInput(planStretch); setShiftInput(planShift); }
    else { setStretchInput(planStretch ? 1 / planStretch : 1); setShiftInput(-planShift); }
  }, [anchor, plan]);

  // Effective per-voice gains from the mix bus (+ optional solo mask).
  const gainsFor = (solo = soloRole) => {
    const xfV = Math.min(1, 2 * (1 - cross));
    const xfB = Math.min(1, 2 * cross);
    let v = vocalMute ? 0 : vocalVol * xfV;
    let i = bedMute ? 0 : bedVol * xfB;
    if (solo === "vocal") i = 0;
    if (solo === "inst") v = 0;
    return { v, i };
  };

  const stretchShiftRef = useRef({ stretch: appliedStretch, shift: appliedShift });
  stretchShiftRef.current = { stretch: appliedStretch, shift: appliedShift };
  const gainsRef = useRef(gainsFor());
  gainsRef.current = gainsFor();

  // Structural sync: buffers, alignment offsets, anchor side.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    const { stretch, shift } = stretchShiftRef.current;
    const { v, i } = gainsRef.current;
    if (vocalBuffer) {
      engine.setVoice("vocal", {
        buffer: vocalBuffer, offsetSec: vocalOffset,
        rate: anchor === "vocal" ? stretch : 1,
        semitones: anchor === "vocal" ? shift : 0, gain: v,
      });
    } else engine.removeVoice("vocal");
    if (instBuffer) {
      engine.setVoice("inst", {
        buffer: instBuffer, offsetSec: instOffset,
        rate: anchor === "instrumental" ? stretch : 1,
        semitones: anchor === "instrumental" ? shift : 0, gain: i,
      });
    } else engine.removeVoice("inst");
    engine.refresh();
  }, [vocalBuffer, instBuffer, vocalOffset, instOffset, anchor, soloRole]);

  // Mix bus: apply gains live on every slider/mute/crossfade change.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    const { v, i } = gainsFor();
    engine.setVoiceGain("vocal", v);
    engine.setVoiceGain("inst", i);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vocalVol, bedVol, cross, vocalMute, bedMute, soloRole]);

  // Pitch: live, no re-arm.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    engine.updateVoiceParams("vocal", { semitones: anchor === "vocal" ? appliedShift : 0 });
    engine.updateVoiceParams("inst", { semitones: anchor === "instrumental" ? appliedShift : 0 });
  }, [appliedShift, anchor]);

  // Stretch: debounce (re-arms both voices — audible if fired per tick).
  const stretchDebounceRef = useRef(null);
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    if (stretchDebounceRef.current) clearTimeout(stretchDebounceRef.current);
    stretchDebounceRef.current = setTimeout(() => {
      engine.updateVoiceParams("vocal", { rate: anchor === "vocal" ? appliedStretch : 1 });
      engine.updateVoiceParams("inst", { rate: anchor === "instrumental" ? appliedStretch : 1 });
    }, 150);
    return () => clearTimeout(stretchDebounceRef.current);
  }, [appliedStretch, anchor]);

  useEffect(() => { engineRef.current?.setLoop(loop); }, [loop]);

  // ── Beat-lock / status readout ────────────────────────────────────────────
  const effBedBpm = instBpm != null ? instBpm * (anchor === "instrumental" ? appliedStretch : 1) : null;
  const effVocalBpm = vocalBpm != null ? vocalBpm * (anchor === "vocal" ? appliedStretch : 1) : null;
  const matched = (() => {
    if (!vocalBpm || !instBpm) return false;
    const a = anchor === "instrumental" ? effBedBpm : effVocalBpm;
    const b = anchor === "instrumental" ? vocalBpm : instBpm;
    return Math.abs(a - b) / b < 0.02;
  })();

  useEffect(() => {
    if (vocalBpm && instBpm) {
      onStatus?.(matched
        ? { locked: true, text: `BEAT-LOCKED · ${vocalBpm.toFixed(1)} BPM · ${vocalCamelot || "?"}` }
        : { text: `Match tempo to lock · ${vocalBpm.toFixed(1)} vs ${instBpm.toFixed(1)}` });
    } else {
      onStatus?.({ text: "Load a vocal + a bed" });
    }
  }, [matched, vocalBpm, instBpm, vocalCamelot, onStatus]);

  // ── Transport handlers ────────────────────────────────────────────────────
  const startPlayback = async (solo) => {
    setError(null); setAudioError(null);
    setSoloRole(solo);
    const engine = engineRef.current;
    if (!engine) return;
    const { v, i } = gainsFor(solo);
    engine.setVoiceGain("vocal", v);
    engine.setVoiceGain("inst", i);
    try { await engine.play(position ?? centerGlobal); }
    catch (e) { setAudioError(`Playback: ${e.message}`); }
  };
  const handlePlayMashup = () => {
    if (isPlaying && soloRole === null) { engineRef.current?.pause(); return; }
    startPlayback(null);
  };
  const handleSolo = (role) => {
    if (isPlaying && soloRole === role) { engineRef.current?.pause(); return; }
    startPlayback(role);
  };

  const handleScrub = (e, el) => {
    const pos = eventPos(e, el, pps);
    if (e.shiftKey) {
      const startPos = pos;
      const onMove = (me) => {
        const p2 = eventPos(me, el, pps);
        setLoop({ start: Math.min(startPos, p2), end: Math.max(startPos, p2) });
        setLoop8(false);
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      return;
    }
    setPosition(pos); engineRef.current?.seek(pos);
    const onMove = (me) => { const p2 = eventPos(me, el, pps); setPosition(p2); engineRef.current?.seek(p2); };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const setAnchorSide = (next) => { if (next !== anchor) { setAnchor(next); setError(null); } };
  const swapTracks = () => { const v = vocalId; setVocalId(instId); setInstId(v); };

  // ── TEMPO helpers ─────────────────────────────────────────────────────────
  const beatPills = useMemo(() => {
    if (!vocalBpm || !instBpm) return [];
    return BEAT_LOCK_RATIOS.map(({ label, value }) => {
      const stretch = anchor === "instrumental" ? (vocalBpm * value) / instBpm : (instBpm * value) / vocalBpm;
      const inRange = stretch >= 0.5 && stretch <= 2;
      return { label, stretch, inRange, active: inRange && Math.abs(stretch - appliedStretch) < 0.01 };
    });
  }, [vocalBpm, instBpm, anchor, appliedStretch]);

  const autoMatchTempo = () => {
    if (!vocalBpm || !instBpm) return;
    const s = anchor === "instrumental" ? vocalBpm / instBpm : instBpm / vocalBpm;
    setStretchInput(Math.max(0.5, Math.min(2, s)).toFixed(4));
    toast("Tempo matched — beat-locked");
  };

  // ── KEY helpers ───────────────────────────────────────────────────────────
  const kr = keyRel(vocalCamelot, instCamelot);
  const bedShiftedKey = shiftCamelot(instCamelot, appliedShift);
  const suggestShift = plan?.semitone_shift != null
    ? (anchor === "instrumental" ? plan.semitone_shift : -plan.semitone_shift)
    : kr.suggest;
  const applySuggestedPitch = () => {
    setShiftInput(suggestShift);
    toast(suggestShift === 0 ? "Keys already match" : `Pitch ${suggestShift > 0 ? "+" : ""}${suggestShift} st applied`);
  };

  // ── Loop 8 bars ───────────────────────────────────────────────────────────
  const toggleLoop8 = () => {
    if (loop8) { setLoop8(false); setLoop(null); return; }
    const bpm = vocalBpm || instBpm;
    if (!bpm) return;
    const barSecs = (4 * 60 / (vocalBpm || bpm)) * vocalDisplayFactor;
    const start = playPos;
    setLoop({ start, end: start + 8 * barSecs });
    setLoop8(true);
  };

  const resetAll = () => {
    engineRef.current?.stop();
    setVocalOffset(0); setInstOffset(0);
    setLoop(null); setLoop8(false); setPosition(null);
    setVocalVol(0.8); setBedVol(0.7); setCross(0.5);
    setVocalMute(false); setBedMute(false);
    const planStretch = plan?.stretch_factor || 1;
    const planShift = plan?.semitone_shift || 0;
    if (anchor === "instrumental") { setStretchInput(planStretch); setShiftInput(planShift); }
    else { setStretchInput(planStretch ? 1 / planStretch : 1); setShiftInput(-planShift); }
    toast("Reset to defaults");
  };

  const handleExport = async () => {
    setError(null);
    if (vocalId == null || instId == null || vocalId === instId) return;
    try {
      const { job_id } = await api.startExport(vocalId, instId, anchor, appliedStretch, appliedShift, vocalOffset, instOffset);
      setExportReady(null); setExportJobId(job_id);
      toast("Rendering mashup WAV…");
    } catch (e) { setError(e.message); }
  };

  // ── Derived UI values ─────────────────────────────────────────────────────
  const samePair = vocalId != null && instId === vocalId;
  const bothLoaded = vocalTrack && instTrack;
  const stretchPct = ((appliedStretch - 0.5) / 1.5) * 100;
  const pitchPct = ((appliedShift + 24) / 48) * 100;
  const targetBpm = anchor === "instrumental" ? vocalBpm : instBpm;
  const dbLabel = (vol, mute) => (mute ? "MUTE" : `${(vol * 24 - 12).toFixed(1)}dB`);

  const pickerList = (menu === "vocal" ? vocalOptions : menu === "bed" ? instOptions : [])
    .filter((t) => !pickerSearch || trackLabel(t).toLowerCase().includes(pickerSearch.toLowerCase()));

  const analysisChip = (track) => {
    const f = track?.features?.full?.metrics || {};
    const items = [
      ["Tempo", f.tempo], ["Key", f.key], ["Dynamics", f.dynamics],
      ["Timbre", f.timbre], ["Structure", (track?.section_count || 0) > 0],
    ];
    return items.map(([label, on], i) => (
      <span key={label} className={on ? "" : "off"}>
        {label}{i < items.length - 1 ? " · " : ""}
      </span>
    ));
  };

  const Selector = ({ role, track, bpm, onOpen }) => (
    <div className={`track-select ${role === "vocal" ? "vocal" : "bed"}`} onClick={onOpen}>
      <span className="role">{role === "vocal" ? "VOCAL" : "BED"}</span>
      <div className="info">
        <div className="t">{track ? track.title : role === "vocal" ? "— select vocal —" : "— select bed —"}</div>
        <div className="a">{track?.artist || ""}</div>
      </div>
      {bpm != null && <div className="bpm">{bpm.toFixed(1)} BPM</div>}
      {track?.features?.full?.camelot && (
        <div className="key-chip" style={{ background: camelotColor(track.features.full.camelot), padding: "4px 9px" }}>
          {track.features.full.camelot}
        </div>
      )}
      <span className="caret">▾</span>
    </div>
  );

  return (
    <div className="page audition">
      {error && <div className="error-text" style={{ marginBottom: 8 }}>{error}</div>}
      {audioError && <div className="error-text" style={{ marginBottom: 8 }}>{audioError}</div>}

      {/* track selectors */}
      <div className="aud-selectors">
        <Selector role="vocal" track={vocalTrack} bpm={vocalBpm}
          onOpen={() => { setMenu(menu === "vocal" ? null : "vocal"); setPickerSearch(""); }} />
        <div className="swap-btn" onClick={swapTracks} title="Swap vocal / bed">⇄</div>
        <Selector role="bed" track={instTrack} bpm={instBpm}
          onOpen={() => { setMenu(menu === "bed" ? null : "bed"); setPickerSearch(""); }} />

        {menu && (
          <div className="picker-menu" style={{ left: menu === "vocal" ? 0 : "auto", right: menu === "bed" ? 0 : "auto" }}>
            <div className="search-box" style={{ width: "auto", margin: "2px 4px 6px" }}>
              <span className="ico">⌕</span>
              <input autoFocus placeholder="Search…" value={pickerSearch}
                onChange={(e) => setPickerSearch(e.target.value)} onClick={(e) => e.stopPropagation()} />
            </div>
            {pickerList.map((t) => {
              const selected = menu === "vocal" ? t.id === vocalId : t.id === instId;
              return (
                <div key={t.id} className={`picker-row${selected ? " selected" : ""}`}
                  onClick={() => { if (menu === "vocal") setVocalId(t.id); else setInstId(t.id); setMenu(null); }}>
                  <div className="art" style={{ background: t.thumbnail ? `url(${t.thumbnail})` : artGradient(t.id) }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="t">{t.title}</div>
                    <div className="a">{t.artist || "—"}</div>
                  </div>
                  <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{t.features?.full?.bpm?.toFixed(1)}</div>
                  {t.features?.full?.camelot && (
                    <div className="key-chip" style={{ background: camelotColor(t.features.full.camelot), fontSize: 11, padding: "2px 6px" }}>
                      {t.features.full.camelot}
                    </div>
                  )}
                </div>
              );
            })}
            {pickerList.length === 0 && <div className="empty" style={{ padding: 12 }}>No analysed tracks.</div>}
          </div>
        )}
      </div>

      {/* analysis chips */}
      {bothLoaded && (
        <div className="analysis-chips">
          <span className="vk">Vocal analysed:</span>
          <span className="list">{analysisChip(vocalTrack)}</span>
          <span className="divider" />
          <span className="bk">Bed analysed:</span>
          <span className="list">{analysisChip(instTrack)}</span>
        </div>
      )}

      {vocalOptions.length === 0 && (
        <p className="empty">No tracks with separated + analysed vocals yet. Separate and analyze tracks in the Library tab first.</p>
      )}
      {samePair && <div className="error-text" style={{ marginBottom: 8 }}>Pick two different tracks.</div>}
      {audioLoading && <p className="hint" style={{ marginBottom: 8 }}>Decoding stems for playback…</p>}

      {bothLoaded && !samePair && (
        <>
          {/* WAVEFORM STACK */}
          <div className="wave-panel">
            <div className="wave-legend">
              {["verse", "chorus", "drop", "breakdown"].map((l) => (
                <span key={l} className="sw"><i style={{ background: SECTION_COLORS[l] }} />{l}</span>
              ))}
              <span className="hint">Drag a lane to nudge · click ruler to move playhead · shift-drag = loop</span>
            </div>

            <div className="ruler" onMouseDown={(e) => handleScrub(e, e.currentTarget)}
              title="Click or drag to scrub · Shift-drag to set a loop">
              <div className="playhead" style={{ left: `${(playPos * pps / TIMELINE_WIDTH) * 100}%` }} />
              {loop && (
                <div className="loop" style={{
                  left: `${(loop.start * pps / TIMELINE_WIDTH) * 100}%`,
                  width: `${((loop.end - loop.start) * pps / TIMELINE_WIDTH) * 100}%`,
                }} />
              )}
            </div>

            <Lane
              role="vocal" sections={vocalDisplaySections} durationSecs={vocalDisplayDuration}
              pps={pps} offsetSecs={vocalOffset} onOffsetChange={setVocalOffset}
              waveform={vocalWaveform.waveform} beatTimes={vocalDisplayBeatTimes}
              onPlay={vocalBuffer ? () => handleSolo("vocal") : null}
              isPlaying={isPlaying && soloRole === "vocal"} playheadPos={playPos}
              otherBeatTimes={instDisplayBeatTimes} otherOffsetSecs={instOffset} snapMode={snapMode}
            />
            <Lane
              role="bed" sections={instDisplaySections} durationSecs={instDisplayDuration}
              pps={pps} offsetSecs={instOffset} onOffsetChange={setInstOffset}
              waveform={instWaveform.waveform} beatTimes={instDisplayBeatTimes}
              onPlay={instBuffer ? () => handleSolo("inst") : null}
              isPlaying={isPlaying && soloRole === "inst"} playheadPos={playPos}
              otherBeatTimes={vocalDisplayBeatTimes} otherOffsetSecs={vocalOffset} snapMode={snapMode}
            />

            <div className="wave-readout">
              Playhead — Vocal {fmtTime(vocalCenterTime)} · Bed {fmtTime(instCenterTime)} ·{" "}
              <span style={{ color: matched ? "var(--green)" : "var(--amber-light)" }}>
                {matched ? "grids aligned ✓" : "grids drift — match tempo"}
              </span>
            </div>
          </div>

          {/* COMMAND BAR */}
          <div className="command-bar">
            {/* TEMPO */}
            <div className="module">
              <div className="module-head">
                <span className="micro-label">TEMPO</span>
                <span className="val">{targetBpm ? targetBpm.toFixed(1) : "—"}<span className="u"> BPM</span></span>
              </div>
              <div className="anchor-row">
                <button className={`anchor-btn${anchor === "instrumental" ? " active" : ""}`} onClick={() => setAnchorSide("instrumental")}>Stretch Bed→Vocal</button>
                <button className={`anchor-btn${anchor === "vocal" ? " active" : ""}`} onClick={() => setAnchorSide("vocal")}>Vocal→Bed</button>
              </div>
              <div className="slider-row">
                <button className="step-btn" onClick={() => setStretchInput((appliedStretch - 0.01).toFixed(4))}>−</button>
                <div className="track-bar" onPointerDown={(e) => dragFraction(e, (f) => setStretchInput((0.5 + f * 1.5).toFixed(4)))}>
                  <div className="knob cyan" style={{ left: `${stretchPct}%` }} />
                </div>
                <button className="step-btn" onClick={() => setStretchInput((appliedStretch + 0.01).toFixed(4))}>+</button>
                <span className="slider-val">×{appliedStretch.toFixed(3)}</span>
              </div>
              <div className="beat-pills">
                {beatPills.map((b) => (
                  <span key={b.label}
                    className={`beat-pill${b.active ? " active" : ""}`}
                    style={{ opacity: b.inRange ? 1 : 0.35, cursor: b.inRange ? "pointer" : "not-allowed" }}
                    onClick={() => b.inRange && setStretchInput(b.stretch.toFixed(4))}
                    title={`×${b.stretch.toFixed(3)}`}>
                    {b.label}
                  </span>
                ))}
              </div>
              <div className="module-cta cyan" onClick={autoMatchTempo}>⚡ Auto-match tempo</div>
            </div>

            {/* KEY / PITCH */}
            <div className="module">
              <div className="module-head">
                <span className="micro-label">KEY / PITCH</span>
                <span className="val">{appliedShift > 0 ? "+" : ""}{appliedShift}<span className="u"> st</span></span>
              </div>
              <div className="key-map">
                <span className="key-chip" style={{ background: camelotColor(vocalCamelot), padding: "5px 10px" }}>{vocalCamelot || "?"}</span>
                <span className="arrow" style={{ color: kr.color }}>{kr.arrow}</span>
                <span className="key-chip" style={{ background: camelotColor(bedShiftedKey), padding: "5px 10px" }}>{bedShiftedKey}</span>
              </div>
              <div className="slider-row">
                <button className="step-btn" onClick={() => setShiftInput(Math.max(-24, appliedShift - 1))}>−</button>
                <div className="track-bar" onPointerDown={(e) => dragFraction(e, (f) => setShiftInput(Math.round(-24 + f * 48)))}>
                  <div className="knob violet" style={{ left: `${pitchPct}%` }} />
                </div>
                <button className="step-btn" onClick={() => setShiftInput(Math.min(24, appliedShift + 1))}>+</button>
              </div>
              <div className="key-rel-text">{kr.text}</div>
              <div className="module-cta ghost" onClick={applySuggestedPitch}>
                {suggestShift === 0 ? "Keys match — no shift" : `Apply suggested pitch (${suggestShift > 0 ? "+" : ""}${suggestShift} st)`}
              </div>
            </div>

            {/* MIX */}
            <div className="module mix">
              <div className="module-head">
                <span className="micro-label">MIX</span>
                {SHOW_NOTES && <span className="needs-bus">NEEDS MIX BUS</span>}
              </div>
              <div className="mix-row">
                <span className="lab v">♪V</span>
                <div className="track-bar thick" onPointerDown={(e) => dragFraction(e, setVocalVol)}>
                  <div className="fill" style={{ width: `${vocalVol * 100}%`, background: "var(--cyan)" }} />
                  <div className="knob cyan" style={{ left: `${vocalVol * 100}%`, boxShadow: "none" }} />
                </div>
                <span className="db">{dbLabel(vocalVol, vocalMute)}</span>
                <button className={`mute-btn${vocalMute ? " on" : ""}`} onClick={() => setVocalMute((m) => !m)}>M</button>
              </div>
              <div className="mix-row">
                <span className="lab b">♪B</span>
                <div className="track-bar thick" onPointerDown={(e) => dragFraction(e, setBedVol)}>
                  <div className="fill" style={{ width: `${bedVol * 100}%`, background: "var(--amber)" }} />
                  <div className="knob" style={{ left: `${bedVol * 100}%`, background: "var(--amber)" }} />
                </div>
                <span className="db">{dbLabel(bedVol, bedMute)}</span>
                <button className={`mute-btn${bedMute ? " on" : ""}`} onClick={() => setBedMute((m) => !m)}>M</button>
              </div>
              <div className="crossfade-head">
                <span className="v">VOCAL</span><span>CROSSFADE</span><span className="b">BED</span>
              </div>
              <div className="crossfade" onPointerDown={(e) => dragFraction(e, setCross)}>
                <div className="knob" style={{ left: `${cross * 100}%` }} />
              </div>
            </div>

            {/* TRANSPORT */}
            <div className="module transport">
              <span className="micro-label">TRANSPORT</span>
              <button className={`play-btn ${isPlaying && soloRole === null ? "playing" : "stopped"}`}
                onClick={handlePlayMashup} disabled={!vocalBuffer || !instBuffer}>
                {isPlaying && soloRole === null ? "❚❚ Pause" : "▶ Play mashup"}
              </button>
              <div className="btn-row">
                <button className={`loop-btn${loop8 ? " on" : ""}`} onClick={toggleLoop8}>⟲ Loop 8 bars</button>
                <button className="reset-btn" onClick={resetAll}>↺ Reset</button>
              </div>
              <div className="snap-row">
                <span>Snap</span>
                <div className="snap-seg">
                  {["beats", "bars", "off"].map((o) => (
                    <button key={o} className={snapMode === o ? "active" : ""} onClick={() => setSnapMode(o)}>{o}</button>
                  ))}
                </div>
              </div>
              <div className="spacer" style={{ flex: 1 }} />
              <button className="export-btn" onClick={handleExport} disabled={exportJobId != null}>
                ↓ Export mashup WAV
              </button>
              {exportJobId && (
                <JobBadge jobId={exportJobId} onComplete={(job) => {
                  setExportJobId(null);
                  if (job.status === "completed") setExportReady(job.result || {});
                }} />
              )}
              {exportReady && (
                <a href={api.exportAudioUrl(vocalId, instId)} target="_blank" rel="noreferrer"
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
