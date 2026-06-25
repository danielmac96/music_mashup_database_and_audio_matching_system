import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { MashupEngine } from "../engine/MashupEngine";
import { decodeStem } from "../engine/decode";
import { MetricGrid } from "./MetricIndicators";

const SECTION_COLORS = {
  intro:     "#4b5563",
  verse:     "#3b82f6",
  chorus:    "#ec4899",
  drop:      "#f59e0b",
  breakdown: "#14b8a6",
  bridge:    "#22c55e",
  outro:     "#6b7280",
};

const TIMELINE_WIDTH  = 1200;
const TIMELINE_HEIGHT = 100;

function fmt(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// Logical-second position under a pointer event, mapped through the element's
// rendered width so it lines up with the canvas drawn at TIMELINE_WIDTH.
function eventPos(e, el, pps) {
  const rect = el.getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  return (frac * TIMELINE_WIDTH) / pps;
}

function SectionTimeline({
  sections, durationSecs, label, pps, offsetSecs, onOffsetChange,
  selectedId, onSectionClick, waveform = [], beatTimes = [], trackRole = "vocal",
  onPlay, isPlaying, playheadPos, loopBand,
  otherBeatTimes = [], otherOffsetSecs = 0, snapMode = "beat", bpmText,
}) {
  const isDraggable = onOffsetChange != null;
  const [dragging, setDragging] = useState(false);
  const [snapped, setSnapped] = useState(false);
  const dragRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
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
        const amp = waveform[i] * (midY - 6);
        ctx.lineTo(x, midY + amp);
      }
      ctx.closePath();
      ctx.fillStyle = trackRole === "vocal"
        ? "rgba(99,179,255,0.18)"
        : "rgba(251,191,36,0.18)";
      ctx.fill();
      ctx.strokeStyle = trackRole === "vocal"
        ? "rgba(99,179,255,0.4)"
        : "rgba(251,191,36,0.4)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    if (beatTimes.length > 0) {
      const barColor  = trackRole === "vocal" ? "rgba(6,182,212,0.75)"  : "rgba(245,158,11,0.75)";
      const beatColor = trackRole === "vocal" ? "rgba(6,182,212,0.35)"  : "rgba(245,158,11,0.35)";
      for (let i = 0; i < beatTimes.length; i++) {
        const x = Math.round(beatTimes[i] * pps + offsetPx) + 0.5;
        if (x < 0 || x > W) continue;
        const isBar = i % 4 === 0;
        ctx.strokeStyle = isBar ? barColor : beatColor;
        ctx.lineWidth = isBar ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(x, isBar ? 0 : TIMELINE_HEIGHT * 0.25);
        ctx.lineTo(x, isBar ? TIMELINE_HEIGHT : TIMELINE_HEIGHT * 0.75);
        ctx.stroke();
      }
    }
  }, [waveform, beatTimes, pps, durationSecs, offsetSecs, trackRole]);

  const handleMouseDown = (e) => {
    if (!isDraggable) return;
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startOffset: offsetSecs };
    setDragging(true);

    const onMove = (me) => {
      if (!dragRef.current) return;
      const dx = me.clientX - dragRef.current.startX;
      const raw = dragRef.current.startOffset + dx / pps;
      const snappedOffset = snapMode === "off"
        ? raw
        : snapOffsetToBeats(raw, beatTimes, otherBeatTimes, otherOffsetSecs, pps, snapMode);
      setSnapped(snappedOffset !== raw);
      onOffsetChange(snappedOffset);
    };
    const onUp = () => {
      setDragging(false);
      setSnapped(false);
      dragRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const playheadPct = (() => {
    if (playheadPos == null) return null;
    const pct = (playheadPos * pps / TIMELINE_WIDTH) * 100;
    return pct >= 0 && pct <= 100 ? pct : null;
  })();

  const loopPct = (() => {
    if (!loopBand) return null;
    const l = (loopBand.start * pps / TIMELINE_WIDTH) * 100;
    const r = (loopBand.end   * pps / TIMELINE_WIDTH) * 100;
    return { left: Math.max(0, l), width: Math.min(100, r) - Math.max(0, l) };
  })();

  return (
    <div className="timeline-row">
      <button
        className={`track-play-btn${isPlaying ? " playing" : ""}`}
        onClick={onPlay}
        disabled={!onPlay}
        title={isPlaying ? "Pause" : "Solo this stem from the playhead"}
      >
        {isPlaying ? "⏸" : "▶"}
      </button>
      <div className="timeline-row-label">
        {label}
        {bpmText && <div className="bpm-tag">{bpmText}</div>}
      </div>
      <div
        className={`timeline-track${isDraggable ? " draggable" : ""}${dragging ? " dragging" : ""}${snapped ? " snapped" : ""}`}
        onMouseDown={handleMouseDown}
      >
        <canvas
          ref={canvasRef}
          width={TIMELINE_WIDTH}
          height={TIMELINE_HEIGHT}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
                   zIndex: 1, pointerEvents: "none" }}
        />
        {loopPct != null && loopPct.width > 0 && (
          <div className="loop-band" style={{ left: `${loopPct.left}%`, width: `${loopPct.width}%` }} />
        )}
        <div
          style={{
            position: "absolute",
            top: 0,
            height: "100%",
            width: durationSecs * pps,
            transform: `translateX(${offsetSecs * pps}px)`,
            zIndex: 2,
          }}
        >
          {sections.map((sec) => (
            <div
              key={sec.id}
              className={`section-block${selectedId === sec.id ? " selected" : ""}`}
              style={{
                left: sec.start_sec * pps,
                width: Math.max((sec.end_sec - sec.start_sec) * pps, 4),
                background: SECTION_COLORS[sec.label] ?? "#4b5563",
              }}
              title={`${sec.label}  ${fmt(sec.start_sec)}–${fmt(sec.end_sec)}`}
              onClick={(e) => { e.stopPropagation(); onSectionClick(sec); }}
            />
          ))}
        </div>
        {playheadPct != null && (
          <div className="playhead" style={{ left: `${playheadPct}%` }} />
        )}
        <div className="center-marker" />
      </div>
    </div>
  );
}

function trackLabel(t) {
  return `${t.title}${t.artist ? ` — ${t.artist}` : ""}`;
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

// Snap the drag offset so the closest on-screen beat (mode "beat") or downbeat
// (mode "bar") of the dragged track lines up exactly with the other track's
// nearest beat/downbeat, once within `snapPx`.
function snapOffsetToBeats(rawOffsetSecs, ownBeats, otherBeats, otherOffsetSecs, pps, mode, snapPx = 10) {
  const stride = mode === "bar" ? 4 : 1;
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

const BEAT_LOCK_RATIOS = [
  { label: "1:1", value: 1 },
  { label: "2:1", value: 2 },
  { label: "1:2", value: 0.5 },
  { label: "3:2", value: 1.5 },
  { label: "2:3", value: 2 / 3 },
  { label: "4:3", value: 4 / 3 },
  { label: "3:4", value: 3 / 4 },
];

const VOCAL_GAIN = 0.95;
const INST_GAIN  = 0.8;

export function AuditionStudio({ seed }) {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [vocalId, setVocalId] = useState(seed?.vocalId ?? null);
  const [instId, setInstId] = useState(seed?.instId ?? null);
  const [plan, setPlan] = useState(null);
  const [anchor, setAnchor] = useState("instrumental"); // which side gets stretched/pitched
  const [stretchInput, setStretchInput] = useState(1);
  const [shiftInput, setShiftInput] = useState(0);

  const [vocalSections, setVocalSections] = useState([]);
  const [instSections, setInstSections] = useState([]);
  const [vocalOffset, setVocalOffset] = useState(0);
  const [instOffset, setInstOffset] = useState(0);
  const [selVocal, setSelVocal] = useState(null);
  const [selInst, setSelInst] = useState(null);
  const [vocalWaveform, setVocalWaveform] = useState({ waveform: [], beat_times: [] });
  const [instWaveform,  setInstWaveform]  = useState({ waveform: [], beat_times: [] });
  const [snapMode, setSnapMode] = useState("beat"); // beat | bar | off

  // Audio engine + decoded buffers
  const engineRef = useRef(null);
  const [vocalBuffer, setVocalBuffer] = useState(null);
  const [instBuffer, setInstBuffer]   = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null);

  // Transport
  const [position, setPosition] = useState(null); // global display sec; null = center
  const [isPlaying, setIsPlaying] = useState(false);
  const [soloRole, setSoloRole] = useState(null);  // null = mashup, "vocal" | "inst"
  const [loop, setLoop] = useState(null);          // global display { start, end }

  // Export
  const [exportJobId, setExportJobId] = useState(null);
  const [exportReady, setExportReady] = useState(null); // job result payload

  useEffect(() => {
    engineRef.current = new MashupEngine();
    engineRef.current.init().catch((e) => setAudioError(`Audio engine: ${e.message}`));
    engineRef.current.onTick((pos, playing) => {
      setPosition(pos);
      setIsPlaying(playing);
    });
    return () => { engineRef.current?.dispose(); engineRef.current = null; };
  }, []);

  useEffect(() => {
    api.getTracks()
      .then((d) => setTracks(d.tracks))
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (seed?.vocalId != null) setVocalId(seed.vocalId);
    if (seed?.instId != null) setInstId(seed.instId);
  }, [seed]);

  const vocalOptions = useMemo(
    () => tracks.filter((t) => t.stems?.vocals && t.features?.full),
    [tracks]
  );
  const instOptions = useMemo(
    () => tracks.filter((t) => t.stems?.instrumental && t.features?.full),
    [tracks]
  );

  const vocalTrack = useMemo(() => tracks.find((t) => t.id === vocalId), [tracks, vocalId]);
  const instTrack  = useMemo(() => tracks.find((t) => t.id === instId),  [tracks, instId]);

  // Stems-first with fallback: trust the vocal stem's own tempo only when its
  // bpm_confidence clears the threshold the backend also uses for beat grids
  // (vocals aren't percussive, so low-confidence stem tracking falls back to
  // the full-mix BPM). The instrumental stem is reliable, so prefer it
  // outright and only fall back to full-mix if it's missing.
  const VOCAL_BPM_CONFIDENCE_MIN = 0.35;
  const vocalStemFeat = vocalTrack?.features?.vocals;
  const vocalFullFeat = vocalTrack?.features?.full;
  const useVocalStemBpm = vocalStemFeat?.bpm
    && (vocalStemFeat?.bpm_confidence ?? 0) >= VOCAL_BPM_CONFIDENCE_MIN;
  const vocalBpm  = useVocalStemBpm ? vocalStemFeat.bpm : vocalFullFeat?.bpm;
  const vocalConf = useVocalStemBpm ? vocalStemFeat.bpm_confidence : vocalFullFeat?.bpm_confidence;

  const instStemFeat = instTrack?.features?.instrumental;
  const instFullFeat = instTrack?.features?.full;
  const instBpm  = instStemFeat?.bpm ?? instFullFeat?.bpm;
  const instConf = instStemFeat?.bpm ? instStemFeat?.bpm_confidence : instFullFeat?.bpm_confidence;

  // Live, decoupled stretch/pitch applied in real time to the anchor side.
  const appliedStretch = (() => {
    const s = Number(stretchInput);
    return Number.isFinite(s) && s > 0 ? s : 1;
  })();
  const appliedShift = (() => {
    const s = Math.round(Number(shiftInput));
    return Number.isFinite(s) ? Math.max(-24, Math.min(24, s)) : 0;
  })();

  // time_stretch(rate) shrinks duration by `rate`, so a point at original time
  // t now plays at t / rate — display = orig * (1 / rate).
  const vocalDisplayFactor = anchor === "vocal" ? 1 / appliedStretch : 1;
  const instDisplayFactor  = anchor === "instrumental" ? 1 / appliedStretch : 1;

  const vocalDisplayDuration = (vocalTrack?.duration_secs ?? 0) * vocalDisplayFactor;
  const instDisplayDuration  = (instTrack?.duration_secs ?? 0) * instDisplayFactor;

  const pps = useMemo(() => {
    const maxDur = Math.max(vocalDisplayDuration, instDisplayDuration, 1);
    return TIMELINE_WIDTH / maxDur;
  }, [vocalDisplayDuration, instDisplayDuration]);

  const centerGlobal = TIMELINE_WIDTH / (2 * pps);
  const playPos = position ?? centerGlobal;

  // Content time under the playhead for each track (display-time).
  const vocalCenterTime = Math.max(0, Math.min(vocalDisplayDuration, playPos - vocalOffset));
  const instCenterTime  = Math.max(0, Math.min(instDisplayDuration, playPos - instOffset));

  const vocalDisplaySections = useMemo(
    () => vocalSections.map((s) => ({
      ...s,
      start_sec: s.start_sec * vocalDisplayFactor,
      end_sec: s.end_sec * vocalDisplayFactor,
    })),
    [vocalSections, vocalDisplayFactor]
  );
  const instDisplaySections = useMemo(
    () => instSections.map((s) => ({
      ...s,
      start_sec: s.start_sec * instDisplayFactor,
      end_sec: s.end_sec * instDisplayFactor,
    })),
    [instSections, instDisplayFactor]
  );
  const vocalDisplayBeatTimes = useMemo(
    () => (vocalWaveform.beat_times || []).map((t) => t * vocalDisplayFactor),
    [vocalWaveform.beat_times, vocalDisplayFactor]
  );
  const instDisplayBeatTimes = useMemo(
    () => (instWaveform.beat_times || []).map((t) => t * instDisplayFactor),
    [instWaveform.beat_times, instDisplayFactor]
  );

  // ── Load metadata + decode audio when a side changes ──────────────────────
  useEffect(() => {
    setVocalSections([]); setSelVocal(null);
    setVocalWaveform({ waveform: [], beat_times: [] });
    setVocalOffset(0); setVocalBuffer(null);
    engineRef.current?.stop();
    setLoop(null); setPosition(null);
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
    setInstSections([]); setSelInst(null);
    setInstWaveform({ waveform: [], beat_times: [] });
    setInstOffset(0); setInstBuffer(null);
    engineRef.current?.stop();
    setLoop(null); setPosition(null);
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
    setPlan(null);
    setExportJobId(null); setExportReady(null);
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
    if (anchor === "instrumental") {
      setStretchInput(planStretch);
      setShiftInput(planShift);
    } else {
      setStretchInput(planStretch ? 1 / planStretch : 1);
      setShiftInput(-planShift);
    }
  }, [anchor, plan]);

  // Keep stretch/shift readable inside effects below without making every
  // tiny slider tick re-run the buffer/offset sync effect.
  const stretchShiftRef = useRef({ stretch: appliedStretch, shift: appliedShift });
  stretchShiftRef.current = { stretch: appliedStretch, shift: appliedShift };

  // Structural sync: buffers, alignment offsets, anchor side, solo muting.
  // Reads the *current* stretch/shift via the ref so it doesn't need them as
  // deps (those get their own live/debounced handling below).
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    const { stretch, shift } = stretchShiftRef.current;
    const vGain = soloRole === "inst" ? 0 : VOCAL_GAIN;
    const iGain = soloRole === "vocal" ? 0 : INST_GAIN;
    if (vocalBuffer) {
      engine.setVoice("vocal", {
        buffer: vocalBuffer,
        offsetSec: vocalOffset,
        rate: anchor === "vocal" ? stretch : 1,
        semitones: anchor === "vocal" ? shift : 0,
        gain: vGain,
      });
    } else engine.removeVoice("vocal");
    if (instBuffer) {
      engine.setVoice("inst", {
        buffer: instBuffer,
        offsetSec: instOffset,
        rate: anchor === "instrumental" ? stretch : 1,
        semitones: anchor === "instrumental" ? shift : 0,
        gain: iGain,
      });
    } else engine.removeVoice("inst");
    engine.refresh();
  }, [vocalBuffer, instBuffer, vocalOffset, instOffset, anchor, soloRole]);

  // Pitch: applies instantly, live, with no re-arm (SoundTouch handles the
  // semitone change in place) — safe to fire on every slider movement.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    engine.updateVoiceParams("vocal", { semitones: anchor === "vocal" ? appliedShift : 0 });
    engine.updateVoiceParams("inst",  { semitones: anchor === "instrumental" ? appliedShift : 0 });
  }, [appliedShift, anchor]);

  // Stretch: changing the rate remaps the whole timeline and re-arms both
  // voices, which is audible as a glitch if it fires on every drag tick — so
  // debounce until the slider settles (or the user releases it).
  const stretchDebounceRef = useRef(null);
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    if (stretchDebounceRef.current) clearTimeout(stretchDebounceRef.current);
    stretchDebounceRef.current = setTimeout(() => {
      engine.updateVoiceParams("vocal", { rate: anchor === "vocal" ? appliedStretch : 1 });
      engine.updateVoiceParams("inst",  { rate: anchor === "instrumental" ? appliedStretch : 1 });
    }, 150);
    return () => clearTimeout(stretchDebounceRef.current);
  }, [appliedStretch, anchor]);

  useEffect(() => {
    engineRef.current?.setLoop(loop);
  }, [loop]);

  // ── Transport handlers ────────────────────────────────────────────────────
  const startPlayback = async (solo) => {
    setError(null); setAudioError(null);
    setSoloRole(solo);
    const engine = engineRef.current;
    if (!engine) return;
    // Apply solo muting immediately so play reflects it without waiting a frame.
    engine.setVoiceGain("vocal", solo === "inst" ? 0 : VOCAL_GAIN);
    engine.setVoiceGain("inst",  solo === "vocal" ? 0 : INST_GAIN);
    try {
      await engine.play(position ?? centerGlobal);
    } catch (e) {
      setAudioError(`Playback: ${e.message}`);
    }
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
      // Shift-drag on the ruler sets a loop region.
      const startPos = pos;
      const onMove = (me) => {
        const p2 = eventPos(me, el, pps);
        setLoop({ start: Math.min(startPos, p2), end: Math.max(startPos, p2) });
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      return;
    }
    setPosition(pos);
    engineRef.current?.seek(pos);
    const onMove = (me) => {
      const p2 = eventPos(me, el, pps);
      setPosition(p2);
      engineRef.current?.seek(p2);
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const handleSetAnchor = (next) => {
    if (next === anchor) return;
    setAnchor(next);
    setError(null);
  };

  const loopFromSelection = () => {
    const sel = selVocal ? { sec: selVocal, off: vocalOffset } :
                selInst ? { sec: selInst, off: instOffset } : null;
    if (!sel) return;
    setLoop({ start: sel.sec.start_sec + sel.off, end: sel.sec.end_sec + sel.off });
  };

  const applySuggestedPitch = () => {
    if (!plan || plan.semitone_shift == null) return;
    setShiftInput(anchor === "instrumental" ? plan.semitone_shift : -plan.semitone_shift);
  };

  const resetAll = () => {
    engineRef.current?.stop();
    setVocalOffset(0); setInstOffset(0);
    setLoop(null); setPosition(null);
    setSelVocal(null); setSelInst(null);
    const planStretch = plan?.stretch_factor || 1;
    const planShift = plan?.semitone_shift || 0;
    if (anchor === "instrumental") { setStretchInput(planStretch); setShiftInput(planShift); }
    else { setStretchInput(planStretch ? 1 / planStretch : 1); setShiftInput(-planShift); }
  };

  const handleExport = async () => {
    setError(null);
    if (vocalId == null || instId == null || vocalId === instId) return;
    try {
      const { job_id } = await api.startExport(
        vocalId, instId, anchor, appliedStretch, appliedShift, vocalOffset, instOffset,
      );
      setExportReady(null);
      setExportJobId(job_id);
    } catch (e) {
      setError(e.message);
    }
  };

  const stretchSuggestions = useMemo(() => {
    if (!vocalBpm || !instBpm) return [];
    return BEAT_LOCK_RATIOS
      .map(({ label, value }) => ({
        label,
        stretch: anchor === "instrumental"
          ? (vocalBpm * value) / instBpm
          : (instBpm * value) / vocalBpm,
      }))
      .filter((s) => s.stretch >= 0.5 && s.stretch <= 2)
      .sort((a, b) => Math.abs(a.stretch - 1) - Math.abs(b.stretch - 1));
  }, [vocalBpm, instBpm, anchor]);

  const samePair = vocalId != null && instId === vocalId;
  const showTimeline = vocalSections.length > 0 || instSections.length > 0;

  const alignmentText = (() => {
    if (!vocalTrack || !instTrack) {
      return "Drag either track to align sections under the playhead ↕";
    }
    return `Playhead — Vocal: ${fmt(vocalCenterTime)}  |  Instrumental: ${fmt(instCenterTime)}`;
  })();

  const bpmTag = (bpm, conf) => {
    if (!bpm) return null;
    const c = conf != null ? ` · ${Math.round(conf * 100)}%` : "";
    return `${bpm.toFixed(1)} BPM${c}`;
  };

  return (
    <div className="panel">
      <h2 style={{ margin: 0 }}>Audition Studio</h2>
      <p className="muted" style={{ marginTop: 4 }}>
        Pick a vocal and an instrumental, drag either waveform to line up sections under the
        playhead, then play them together. Tempo and pitch are matched live (decoupled — no
        chipmunking) and nothing is written until you export.
      </p>

      {error && <div className="error-text" style={{ marginTop: 8 }}>{error}</div>}
      {audioError && <div className="error-text" style={{ marginTop: 8 }}>{audioError}</div>}

      <div className="audition-pickers" style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 12 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="muted">Vocal (top){vocalBpm ? ` — ${bpmTag(vocalBpm, vocalConf)}` : ""}</span>
          <select
            value={vocalId ?? ""}
            onChange={(e) => setVocalId(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">— select vocal —</option>
            {vocalOptions.map((t) => (
              <option key={t.id} value={t.id}>{trackLabel(t)}</option>
            ))}
          </select>
        </label>

        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="muted">Instrumental (bed){instBpm ? ` — ${bpmTag(instBpm, instConf)}` : ""}</span>
          <select
            value={instId ?? ""}
            onChange={(e) => setInstId(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">— select instrumental —</option>
            {instOptions.map((t) => (
              <option key={t.id} value={t.id}>{trackLabel(t)}</option>
            ))}
          </select>
        </label>
      </div>

      {(vocalTrack || instTrack) && (
        <div className="audition-metrics" style={{ display: "flex", gap: 32, flexWrap: "wrap", marginTop: 10 }}>
          {vocalTrack && (
            <div>
              <span className="muted" style={{ fontSize: "0.75rem" }}>Vocal — what's analysed</span>
              <MetricGrid
                stems={vocalTrack.stems}
                features={vocalTrack.features}
                sectionCount={vocalTrack.section_count}
              />
            </div>
          )}
          {instTrack && (
            <div>
              <span className="muted" style={{ fontSize: "0.75rem" }}>Instrumental — what's analysed</span>
              <MetricGrid
                stems={instTrack.stems}
                features={instTrack.features}
                sectionCount={instTrack.section_count}
              />
            </div>
          )}
        </div>
      )}

      {vocalOptions.length === 0 && (
        <p className="muted" style={{ marginTop: 8 }}>
          No tracks with separated + analysed vocals yet. Separate and analyze
          tracks in the Library tab first.
        </p>
      )}

      {samePair && (
        <div className="error-text" style={{ marginTop: 8 }}>
          Pick two different tracks.
        </div>
      )}

      {audioLoading && (
        <p className="muted" style={{ marginTop: 8 }}>Decoding stems for playback…</p>
      )}

      {showTimeline && (
        <div className="timeline-panel" style={{ marginTop: 20 }}>
          <div className="section-legend">
            {Object.entries(SECTION_COLORS).map(([lbl, color]) => (
              <span key={lbl} className="legend-item">
                <span className="legend-swatch" style={{ background: color }} />
                {lbl}
              </span>
            ))}
          </div>

          {/* Scrub ruler: click/drag to move the playhead, shift-drag to set a loop. */}
          <div
            className="transport-ruler"
            onMouseDown={(e) => handleScrub(e, e.currentTarget)}
            title="Click or drag to scrub · Shift-drag to set a loop region"
          >
            <div className="ruler-playhead" style={{ left: `${(playPos * pps / TIMELINE_WIDTH) * 100}%` }} />
            {loop && (
              <div
                className="ruler-loop"
                style={{
                  left: `${(loop.start * pps / TIMELINE_WIDTH) * 100}%`,
                  width: `${((loop.end - loop.start) * pps / TIMELINE_WIDTH) * 100}%`,
                }}
              />
            )}
          </div>

          <SectionTimeline
            sections={vocalDisplaySections}
            durationSecs={vocalDisplayDuration}
            label="Vocal"
            bpmText={bpmTag(vocalBpm, vocalConf)}
            pps={pps}
            offsetSecs={vocalOffset}
            onOffsetChange={setVocalOffset}
            selectedId={selVocal?.id}
            onSectionClick={setSelVocal}
            waveform={vocalWaveform.waveform}
            beatTimes={vocalDisplayBeatTimes}
            trackRole="vocal"
            onPlay={vocalBuffer ? () => handleSolo("vocal") : null}
            isPlaying={isPlaying && soloRole === "vocal"}
            playheadPos={playPos}
            loopBand={loop}
            otherBeatTimes={instDisplayBeatTimes}
            otherOffsetSecs={instOffset}
            snapMode={snapMode}
          />

          <SectionTimeline
            sections={instDisplaySections}
            durationSecs={instDisplayDuration}
            label="Inst"
            bpmText={bpmTag(instBpm, instConf)}
            pps={pps}
            offsetSecs={instOffset}
            onOffsetChange={setInstOffset}
            selectedId={selInst?.id}
            onSectionClick={setSelInst}
            waveform={instWaveform.waveform}
            beatTimes={instDisplayBeatTimes}
            trackRole="instrumental"
            onPlay={instBuffer ? () => handleSolo("inst") : null}
            isPlaying={isPlaying && soloRole === "inst"}
            playheadPos={playPos}
            loopBand={loop}
            otherBeatTimes={vocalDisplayBeatTimes}
            otherOffsetSecs={vocalOffset}
            snapMode={snapMode}
          />

          <div className="alignment-readout">
            {alignmentText}
            <label style={{ marginLeft: 14, display: "inline-flex", alignItems: "center", gap: 4, fontSize: "0.75rem" }}>
              Snap while dragging:
              <select value={snapMode} onChange={(e) => setSnapMode(e.target.value)} style={{ fontSize: "0.75rem" }}>
                <option value="beat">beats</option>
                <option value="bar">bars</option>
                <option value="off">off</option>
              </select>
            </label>
          </div>

          <div className="anchor-toggle" style={{ display: "flex", gap: 8, marginTop: 8, alignItems: "center", flexWrap: "wrap" }}>
            <button
              className={anchor === "instrumental" ? "active" : "secondary"}
              onClick={() => handleSetAnchor("instrumental")}
              disabled={samePair || vocalId == null || instId == null}
            >
              Stretch instrumental → vocal
            </button>
            <button
              className={anchor === "vocal" ? "active" : "secondary"}
              onClick={() => handleSetAnchor("vocal")}
              disabled={samePair || vocalId == null || instId == null}
            >
              Stretch vocal → instrumental
            </button>

            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem" }}>
              <span className="muted">Stretch ×</span>
              <button
                className="secondary"
                style={{ padding: "1px 7px" }}
                onClick={() => setStretchInput((Number(stretchInput) - 0.01).toFixed(4))}
                title="Nudge stretch down 0.01"
              >−</button>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.001"
                value={appliedStretch}
                onChange={(e) => setStretchInput(e.target.value)}
                style={{ width: 110 }}
              />
              <button
                className="secondary"
                style={{ padding: "1px 7px" }}
                onClick={() => setStretchInput((Number(stretchInput) + 0.01).toFixed(4))}
                title="Nudge stretch up 0.01"
              >+</button>
              <span className="muted" style={{ minWidth: 52, fontVariantNumeric: "tabular-nums" }}>
                ×{appliedStretch.toFixed(3)}
              </span>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem" }}>
              <span className="muted">Pitch (st)</span>
              <button
                className="secondary"
                style={{ padding: "1px 7px" }}
                onClick={() => setShiftInput(Math.max(-24, appliedShift - 1))}
                title="Nudge pitch down 1 semitone"
              >−</button>
              <input
                type="range"
                min="-24"
                max="24"
                step="1"
                value={appliedShift}
                onChange={(e) => setShiftInput(e.target.value)}
                style={{ width: 90 }}
              />
              <button
                className="secondary"
                style={{ padding: "1px 7px" }}
                onClick={() => setShiftInput(Math.min(24, appliedShift + 1))}
                title="Nudge pitch up 1 semitone"
              >+</button>
              <span className="muted" style={{ minWidth: 30, fontVariantNumeric: "tabular-nums" }}>
                {appliedShift > 0 ? "+" : ""}{appliedShift}
              </span>
            </label>
            <button
              className="secondary"
              style={{ fontSize: "0.75rem" }}
              onClick={() => startPlayback(null)}
              title="Re-audition from the current marker with the latest stretch/pitch"
            >
              ▶ Play from marker
            </button>
            {plan?.stretch_factor && (
              <button
                className="secondary"
                style={{ fontSize: "0.75rem" }}
                onClick={() => setStretchInput(anchor === "instrumental"
                  ? plan.stretch_factor
                  : (plan.stretch_factor ? 1 / plan.stretch_factor : 1))}
                title="Auto tempo-match using the detected BPMs"
              >
                Auto tempo-match
              </button>
            )}
          </div>

          {stretchSuggestions.length > 0 && (
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6, alignItems: "center" }}>
              <span className="muted" style={{ fontSize: "0.75rem" }}>Beat-lock ratios:</span>
              {stretchSuggestions.map((s) => (
                <button
                  key={s.label}
                  className="secondary"
                  style={{ fontSize: "0.7rem", padding: "2px 7px" }}
                  onClick={() => setStretchInput(s.stretch.toFixed(4))}
                  title={`Set stretch to ×${s.stretch.toFixed(4)} so the beat grids align at a ${s.label} ratio`}
                >
                  {s.label} (×{s.stretch.toFixed(3)})
                </button>
              ))}
            </div>
          )}

          {plan && (
            <div className="key-hint" style={{ marginTop: 6, fontSize: "0.78rem" }}>
              <span className="muted">Key:</span>{" "}
              {plan.vocal?.key ?? "?"} {plan.vocal?.mode ?? ""} (vocal) vs{" "}
              {plan.inst?.key ?? "?"} {plan.inst?.mode ?? ""} (inst) — {plan.key_relation}
              {plan.semitone_shift ? (
                <button
                  className="secondary"
                  style={{ fontSize: "0.7rem", padding: "2px 7px", marginLeft: 8 }}
                  onClick={applySuggestedPitch}
                  title="Set pitch to the suggested semitone offset to bring the keys together"
                >
                  Apply suggested pitch ({plan.semitone_shift >= 0 ? "+" : ""}{plan.semitone_shift} st)
                </button>
              ) : null}
            </div>
          )}

          <div className="preview-play-row" style={{ marginTop: 10 }}>
            <button
              className={`preview-play-btn${isPlaying && soloRole === null ? " playing" : ""}`}
              onClick={handlePlayMashup}
              disabled={samePair || !vocalBuffer || !instBuffer}
              title="Play both stems together, tempo/pitch-matched live, from the playhead"
            >
              {isPlaying && soloRole === null ? "⏸ Stop mashup" : "▶ Play mashup"}
            </button>

            <button
              className="secondary"
              onClick={loopFromSelection}
              disabled={!selVocal && !selInst}
              title="Loop the selected section"
            >
              Loop selection
            </button>
            {loop && (
              <button className="secondary" onClick={() => setLoop(null)}>
                Clear loop ({fmt(loop.start)}–{fmt(loop.end)})
              </button>
            )}

            <button
              className="secondary"
              onClick={resetAll}
              title="Reset alignment, stretch, pitch and loop to defaults"
            >
              Reset all
            </button>
          </div>

          <div className="preview-play-row" style={{ marginTop: 8 }}>
            <button
              onClick={handleExport}
              disabled={samePair || vocalId == null || instId == null || exportJobId != null}
              title="Render the current mashup (alignment + stretch + pitch) to a WAV"
            >
              Export mashup WAV
            </button>
            {exportJobId && (
              <JobBadge
                jobId={exportJobId}
                onComplete={(job) => {
                  setExportJobId(null);
                  if (job.status === "completed") setExportReady(job.result || {});
                }}
              />
            )}
            {exportReady && (
              <a
                href={api.exportAudioUrl(vocalId, instId)}
                target="_blank"
                rel="noreferrer"
                className="muted"
                style={{ fontSize: "0.8rem" }}
              >
                ↓ download / play export
              </a>
            )}
          </div>

          {(selVocal || selInst) && (
            <div className="mash-point">
              Mash point →
              {selVocal
                ? ` Vocal ${selVocal.label} (${fmt(selVocal.start_sec)}–${fmt(selVocal.end_sec)})`
                : " —"}
              {" + "}
              {selInst
                ? `Inst ${selInst.label} (${fmt(selInst.start_sec)}–${fmt(selInst.end_sec)})`
                : " —"}
            </div>
          )}
        </div>
      )}

      {plan && (
        <div style={{ marginTop: 16 }}>
          <div className="plan-summary" style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <div>
              <div className="muted">Target tempo</div>
              <strong>{plan.target_bpm ? `${plan.target_bpm.toFixed(1)} BPM` : "—"}</strong>
            </div>
            <div>
              <div className="muted">Stretch instrumental</div>
              <strong>{plan.stretch_factor ? `×${plan.stretch_factor}` : "—"}</strong>
            </div>
            <div>
              <div className="muted">Pitch instrumental</div>
              <strong>
                {plan.semitone_shift != null
                  ? `${plan.semitone_shift >= 0 ? "+" : ""}${plan.semitone_shift} st`
                  : "—"}
              </strong>
            </div>
            <div>
              <div className="muted">Key relation</div>
              <strong>{plan.key_relation}</strong>
            </div>
          </div>

          {plan.pairings?.length > 0 && (
            <p className="muted" style={{ marginTop: 12, fontSize: "0.8rem" }}>
              Auto-aligned on: {plan.pairings[0].note}.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
