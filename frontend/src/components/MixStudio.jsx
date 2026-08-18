import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { KeyChip } from "./KeyChip";
import { TrackArt } from "./TrackArt";
import { MashupEngine } from "../engine/MashupEngine";
import { decodeStem } from "../engine/decode";
import { downbeatsOf, isDownbeat, phaseForDownbeatAt } from "../engine/grid";
import { fmtTime, keyRel } from "../theme";
import { toast } from "../toast";

// ── Studio: multi-track DAW-style arrangement view ───────────────────────────
// N lanes (any stem of any analysed track) on one zoomable/scrollable timeline
// driven by MashupEngine (which is already N-voice: one SoundTouch worklet per
// lane, all locked to a single AudioContext clock). Per lane: tempo-sync to the
// project BPM (decoupled stretch), pitch shift, gain/mute/solo, drag with
// snap-to-grid. Export renders the same clip math server-side to a WAV.
//
// This is the ONLY arranger (T4.1). It absorbed the two-deck Audition view,
// which was a near-duplicate over the same engine and meant two export payload
// shapes and two playback sync paths. What came across from it:
//   * the A/B crossfader, which now rides the first two lanes;
//   * switching a lane's stem in place, instead of removing and re-adding it;
//   * a manual stretch factor for a lane that is not tempo-synced;
//   * "match key to lane 1" and "align to the bar grid" as one-click moves;
//   * loop lengths of 1/2/4/8 bars rather than only 8.
// Audition's nudge buttons did not come across as buttons: the arrow keys here
// already nudge the selected lane by a beat, and shift+arrow by 10 ms, which is
// what those buttons did.
//
// `seed` ({ vocalId, instId, ... }) opens the arranger on one pair straight
// from Discover — that is what the Audition tab used to be.

const SECTION_COLORS = {
  intro: "#6b7280", verse: "#3b82f6", chorus: "#ec4899", drop: "#f59e0b",
  breakdown: "#14b8a6", bridge: "#22c55e", outro: "#6b7280",
};

// Per-lane accent colours, cycled as lanes are added.
const LANE_COLORS = [
  { line: "56,189,248" },   // cyan
  { line: "245,166,35" },   // amber
  { line: "167,139,250" },  // violet
  { line: "52,211,153" },   // green
  { line: "244,114,182" },  // pink
  { line: "96,165,250" },   // blue
  { line: "251,146,60" },   // orange
  { line: "163,230,53" },   // lime
];

const STEM_LABEL = {
  vocals: "VOX", instrumental: "INST", full: "FULL",
  drums: "DRM", bass: "BAS", other: "OTH",
};
// Order the stem switches offer. drums/bass/other only exist for tracks
// separated in four-stem mode (Phase D); the buttons disable themselves
// otherwise, which is also how the user discovers the mode exists. They are
// what make the real producer moves possible — drop the bed's bass and keep the
// vocal track's, or swap the bed's drums for a tighter kit.
const STEM_ORDER = ["vocals", "instrumental", "drums", "bass", "other", "full"];
const VOCAL_BPM_CONFIDENCE_MIN = 0.35; // mirror of backend fallback threshold
const MIN_PPS = 4, MAX_PPS = 240;
const SNAP_PX = 12;
const HEADER_W = 236; // lane-header column width (must match .studio-grid CSS)
const STORAGE_KEY = "mashup.studio.project.v1";

let laneUid = 1;

// Source BPM for a lane, stems-first with the vocal-confidence fallback the
// waveform endpoint also applies (its beat grid follows the same rule).
function laneBpmFor(track, stem) {
  const feats = track?.features || {};
  if (stem === "vocals") {
    const v = feats.vocals;
    if (v?.bpm && (v.bpm_confidence ?? 0) >= VOCAL_BPM_CONFIDENCE_MIN) return v.bpm;
    return feats.full?.bpm ?? null;
  }
  return feats[stem]?.bpm ?? feats.full?.bpm ?? null;
}

// Rate (playback-speed factor) that conforms a lane to the project tempo,
// half/double-time aware: pick the ×½/×1/×2 target whose stretch stays
// closest to 1 so the stem is warped as little as possible.
function syncRateFor(laneBpm, projectBpm) {
  if (!laneBpm || !projectBpm) return null;
  let best = null;
  for (const mul of [0.5, 1, 2]) {
    const r = (projectBpm * mul) / laneBpm;
    if (r < 0.5 || r > 2) continue;
    if (!best || Math.abs(Math.log(r)) < Math.abs(Math.log(best))) best = r;
  }
  return best;
}

// One semitone = 7 Camelot positions (7×7 ≡ 1 mod 12).
function shiftCamelot(camelot, semitones) {
  if (!camelot || !semitones) return camelot || null;
  const m = /^(\d{1,2})\s*([ABab])$/.exec(String(camelot).trim());
  if (!m) return camelot;
  const num = ((parseInt(m[1], 10) - 1 + 7 * semitones) % 12 + 12) % 12 + 1;
  return `${num}${m[2].toUpperCase()}`;
}

function fmtBarsBeats(pos, bpm) {
  if (!bpm) return "—";
  const beat = 60 / bpm;
  const totalBeats = Math.floor(Math.max(0, pos) / beat);
  return `${Math.floor(totalBeats / 4) + 1}.${(totalBeats % 4) + 1}`;
}

// ── canvas painters (window rendering: only [viewStart, viewStart+W/pps]) ────

function paintLane(canvas, lane, viewStart, pps, selected) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr; canvas.height = H * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const rawDur = lane.buffer?.duration ?? lane.rawDur ?? 0;
  if (!rawDur) return;
  const dispDur = rawDur / lane.rate;
  const x0 = (lane.offsetSec - viewStart) * pps;
  const x1 = (lane.offsetSec + dispDur - viewStart) * pps;
  if (x1 < 0 || x0 > W) return;

  const rgb = LANE_COLORS[lane.colorIdx % LANE_COLORS.length].line;

  // Clip body — makes the clip's extent visible even where audio is quiet.
  ctx.fillStyle = `rgba(${rgb},${selected ? 0.10 : 0.06})`;
  ctx.strokeStyle = `rgba(${rgb},${selected ? 0.8 : 0.35})`;
  ctx.lineWidth = 1;
  const bx = Math.max(-2, x0), bw = Math.min(W + 2, x1) - bx;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(bx, 9, bw, H - 10, 4);
  else ctx.rect(bx, 9, bw, H - 10);
  ctx.fill();
  ctx.stroke();

  // Section ribbon (structure labels), scaled into display time.
  for (const sec of lane.sections || []) {
    const sx = (lane.offsetSec + sec.start_sec / lane.rate - viewStart) * pps;
    const sw = ((sec.end_sec - sec.start_sec) / lane.rate) * pps;
    if (sx + sw < 0 || sx > W) continue;
    ctx.fillStyle = SECTION_COLORS[sec.label] ?? "#3b82f6";
    ctx.globalAlpha = 0.85;
    ctx.fillRect(Math.max(0, sx), 2, Math.max(2, Math.min(W, sx + sw) - Math.max(0, sx)), 4);
    ctx.globalAlpha = 1;
  }

  // Waveform envelope.
  const wf = lane.waveform || [];
  if (wf.length > 0) {
    const midY = 9 + (H - 10) / 2;
    const segDisp = dispDur / wf.length;
    const iLo = Math.max(0, Math.floor((viewStart - lane.offsetSec) / segDisp) - 1);
    const iHi = Math.min(wf.length, Math.ceil((viewStart + W / pps - lane.offsetSec) / segDisp) + 1);
    ctx.beginPath();
    let started = false;
    for (let i = iLo; i < iHi; i++) {
      const x = (lane.offsetSec + (i + 0.5) * segDisp - viewStart) * pps;
      const amp = wf[i] * (midY - 12);
      if (!started) { ctx.moveTo(x, midY - amp); started = true; }
      else ctx.lineTo(x, midY - amp);
    }
    for (let i = iHi - 1; i >= iLo; i--) {
      const x = (lane.offsetSec + (i + 0.5) * segDisp - viewStart) * pps;
      ctx.lineTo(x, midY + wf[i] * (midY - 12));
    }
    if (started) {
      ctx.closePath();
      ctx.fillStyle = `rgba(${rgb},0.20)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${rgb},0.85)`;
      ctx.stroke();
    }
  }

  // Beat grid of THIS lane (its own beats, warped by its rate).
  const beats = lane.beatTimes || [];
  if (beats.length > 0 && pps > 8) {
    for (let i = 0; i < beats.length; i++) {
      const t = lane.offsetSec + beats[i] / lane.rate;
      const x = Math.round((t - viewStart) * pps) + 0.5;
      if (x < 0) continue;
      if (x > W) break;
      const isBar = isDownbeat(i, lane.beatPhase);
      if (!isBar && pps < 24) continue;
      // Downbeats are drawn well clear of beat lines so a wrong phase is
      // obvious by eye — that is the only way to catch a bad detection.
      ctx.strokeStyle = `rgba(${rgb},${isBar ? 0.75 : 0.15})`;
      ctx.lineWidth = isBar ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(x, isBar ? 9 : H * 0.45);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
  }
}

function paintRuler(canvas, viewStart, pps, projectBpm, loop) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr; canvas.height = H * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  if (loop) {
    const lx = (loop.start - viewStart) * pps;
    const lw = (loop.end - loop.start) * pps;
    ctx.fillStyle = "rgba(122,162,247,0.20)";
    ctx.fillRect(lx, 0, lw, H);
    ctx.fillStyle = "rgba(122,162,247,0.7)";
    ctx.fillRect(lx, 0, 2, H);
    ctx.fillRect(lx + lw - 2, 0, 2, H);
  }

  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  if (projectBpm) {
    const beat = 60 / projectBpm, bar = beat * 4;
    // Label stride: keep bar numbers ≥ 44px apart.
    const stride = Math.max(1, Math.ceil(44 / (bar * pps)));
    const first = Math.max(0, Math.floor(viewStart / bar));
    const last = Math.ceil((viewStart + W / pps) / bar);
    for (let b = first; b <= last; b++) {
      const x = Math.round((b * bar - viewStart) * pps) + 0.5;
      const labelled = b % stride === 0;
      ctx.strokeStyle = labelled ? "rgba(160,170,192,0.55)" : "rgba(160,170,192,0.25)";
      ctx.beginPath(); ctx.moveTo(x, labelled ? 2 : H * 0.5); ctx.lineTo(x, H); ctx.stroke();
      if (labelled) {
        ctx.fillStyle = "rgba(160,170,192,0.8)";
        ctx.fillText(String(b + 1), x + 3, 2);
      }
      if (beat * pps > 14 && stride === 1) {
        for (let k = 1; k < 4; k++) {
          const bx = Math.round(((b * bar) + k * beat - viewStart) * pps) + 0.5;
          ctx.strokeStyle = "rgba(160,170,192,0.15)";
          ctx.beginPath(); ctx.moveTo(bx, H * 0.65); ctx.lineTo(bx, H); ctx.stroke();
        }
      }
    }
  } else {
    // No BPM yet: plain seconds ruler.
    const step = pps > 40 ? 1 : pps > 12 ? 5 : 15;
    const first = Math.max(0, Math.floor(viewStart / step));
    const last = Math.ceil((viewStart + W / pps) / step);
    for (let s = first; s <= last; s++) {
      const x = Math.round((s * step - viewStart) * pps) + 0.5;
      ctx.strokeStyle = "rgba(160,170,192,0.35)";
      ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, H); ctx.stroke();
      ctx.fillStyle = "rgba(160,170,192,0.8)";
      ctx.fillText(fmtTime(s * step), x + 3, 2);
    }
  }
}

// Snap a proposed offset so the lane's nearest visible downbeat clicks onto
// the project grid once within SNAP_PX. Returns { offset, snapped }.
function snapToGrid(rawOffset, lane, projectBpm, snapMode, viewStart, viewSecs, pps) {
  if (snapMode === "off" || !projectBpm) return { offset: rawOffset, snapped: false };
  const step = snapMode === "bar" ? (60 / projectBpm) * 4 : 60 / projectBpm;
  const beats = lane.beatTimes || [];
  // Candidate lane anchors: its downbeats (every 4th beat) in display time —
  // or the clip start when the lane has no beat data.
  const anchors = [];
  if (beats.length > 0) {
    for (let i = 0; i < beats.length; i += 4) {
      const t = rawOffset + beats[i] / lane.rate;
      if (t >= viewStart - viewSecs * 0.5 && t <= viewStart + viewSecs * 1.5) anchors.push(t);
    }
  }
  if (anchors.length === 0) anchors.push(rawOffset);

  let bestDelta = Infinity;
  for (const t of anchors) {
    const nearest = Math.round(t / step) * step;
    const d = nearest - t;
    if (Math.abs(d) < Math.abs(bestDelta)) bestDelta = d;
  }
  const snapSecs = SNAP_PX / pps;
  if (Math.abs(bestDelta) <= snapSecs) return { offset: rawOffset + bestDelta, snapped: true };
  return { offset: rawOffset, snapped: false };
}

// ── main component ────────────────────────────────────────────────────────────

export function MixStudio({ onStatus, seed, onSeedConsumed }) {
  const [tracks, setTracks] = useState([]);
  const [lanes, setLanes] = useState([]);
  const [projectBpm, setProjectBpm] = useState(null);
  const [snapMode, setSnapMode] = useState("bar"); // "bar" | "beat" | "off"
  const [selectedId, setSelectedId] = useState(null);
  const [soloId, setSoloId] = useState(null);
  // Crossfader between the first two lanes (from Audition). 0.5 is a no-op, so
  // a project that never touches it behaves exactly as before.
  const [cross, setCross] = useState(0.5);
  const [loopBars, setLoopBars] = useState(8);

  // Viewport (zoom + horizontal scroll, in seconds/px-per-second).
  const [pps, setPps] = useState(28);
  const [viewStart, setViewStart] = useState(0);
  const [viewW, setViewW] = useState(1000);

  // Transport.
  const [position, setPosition] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loop, setLoop] = useState(null);

  const [picker, setPicker] = useState(false);
  const [pickerSearch, setPickerSearch] = useState("");
  const [error, setError] = useState(null);
  const [exportJobId, setExportJobId] = useState(null);
  const [exportToken, setExportToken] = useState(null);
  const [sessionJobId, setSessionJobId] = useState(null);
  const [sessionToken, setSessionToken] = useState(null);
  const [dragSnapped, setDragSnapped] = useState(false);
  const [restored, setRestored] = useState(false);

  const engineRef = useRef(null);
  const lanesRef = useRef(lanes); lanesRef.current = lanes;
  const viewRef = useRef(null);
  const laneCanvasRefs = useRef(new Map());
  const rulerRef = useRef(null);

  // ── engine lifecycle ────────────────────────────────────────────────────
  useEffect(() => {
    engineRef.current = new MashupEngine();
    engineRef.current.init().catch((e) => setError(`Audio engine: ${e.message}`));
    engineRef.current.onTick((pos, playing) => {
      setPosition(pos);
      setIsPlaying(playing);
    });
    return () => { engineRef.current?.dispose(); engineRef.current = null; };
  }, []);

  useEffect(() => {
    api.getTracks().then((d) => setTracks(d.tracks)).catch((e) => setError(e.message));
  }, []);

  // ── viewport sizing ─────────────────────────────────────────────────────
  // viewW is the LANE AREA width (timeline minus the lane-header column) — all
  // px↔seconds math is relative to the canvases, not the whole grid.
  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    const measure = () => setViewW(Math.max(100, (el.clientWidth || 1000) - HEADER_W));
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const viewSecs = viewW / pps;
  const projectLen = useMemo(() => {
    let end = 60;
    for (const l of lanes) {
      const rawDur = l.buffer?.duration ?? l.rawDur ?? 0;
      end = Math.max(end, l.offsetSec + rawDur / l.rate);
    }
    return end + 30;
  }, [lanes]);

  // ── lane loading ────────────────────────────────────────────────────────
  // Waveform, structure and decoded audio for one (song, stem). Shared by
  // addLane and the in-place stem switch so both land in the same state — a
  // second copy is how a lane ends up drawing one stem and playing another.
  const loadLaneMedia = useCallback(async (laneId, songId, stem) => {
    const engine = engineRef.current;
    if (!engine) return;
    const patch = (p) => setLanes((ls) => ls.map((l) => (l.id === laneId ? { ...l, ...p } : l)));

    api.getWaveform(songId, stem)
      .then((d) => patch({ waveform: d.waveform || [], beatTimes: d.beat_times || [],
                           beatPhase: d.beat_phase || 0 }))
      .catch(() => {});
    api.getSections(songId)
      .then((d) => patch({ sections: d.sections }))
      .catch(() => {});
    try {
      await engine.init();
      const buf = await decodeStem(engine.ctx, api.audioUrl(songId, stem));
      patch({ buffer: buf, loading: false, loadError: null });
    } catch (e) {
      patch({ loading: false, loadError: `audio: ${e.message}` });
    }
  }, []);

  const addLane = useCallback(async (track, stem, saved = {}) => {
    const engine = engineRef.current;
    if (!engine) return;
    const id = `lane${laneUid++}`;
    const lane = {
      id,
      songId: track.id,
      stem,
      title: track.title,
      artist: track.artist || "",
      thumbnail: track.thumbnail,
      camelot: track.features?.full?.camelot || null,
      bpm: laneBpmFor(track, stem),
      colorIdx: (saved.colorIdx != null ? saved.colorIdx : laneUid) % LANE_COLORS.length,
      offsetSec: saved.offsetSec ?? 0,
      rate: saved.rate ?? 1,
      semitones: saved.semitones ?? 0,
      gain: saved.gain ?? 0.8,
      muted: saved.muted ?? false,
      synced: saved.synced ?? false,
      waveform: [], beatTimes: [], beatPhase: 0, sections: [],
      buffer: null, rawDur: track.duration_secs || 0,
      loading: true, loadError: null,
    };
    setLanes((ls) => [...ls, lane]);
    setSelectedId(id);
    await loadLaneMedia(id, track.id, stem);
    return id;
  }, [loadLaneMedia]);

  // Switch a lane's stem without losing its placement (from Audition, where the
  // stem picker sat on the deck). Offset/rate/pitch/gain all survive, because
  // the point of the switch is to hear the same arrangement with a different
  // layer — dropping the lane and re-adding it would reset all of that.
  const setLaneStem = useCallback((lane, stem) => {
    if (stem === lane.stem) return;
    engineRef.current?.removeVoice(lane.id);
    setLanes((ls) => ls.map((l) => (l.id === lane.id
      ? { ...l, stem, buffer: null, waveform: [], beatTimes: [], beatPhase: 0,
          sections: [], loading: true, loadError: null,
          bpm: laneBpmFor(tracks.find((t) => t.id === l.songId), stem) }
      : l)));
    loadLaneMedia(lane.id, lane.songId, stem);
  }, [loadLaneMedia, tracks]);

  const removeLane = (id) => {
    engineRef.current?.removeVoice(id);
    setLanes((ls) => ls.filter((l) => l.id !== id));
    if (selectedId === id) setSelectedId(null);
    if (soloId === id) setSoloId(null);
  };

  const patchLane = (id, patch) =>
    setLanes((ls) => ls.map((l) => (l.id === id ? { ...l, ...patch } : l)));

  const moveLane = (id, dir) =>
    setLanes((ls) => {
      const i = ls.findIndex((l) => l.id === id);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= ls.length) return ls;
      const next = [...ls];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });

  // First lane with a BPM seeds the project tempo.
  useEffect(() => {
    if (projectBpm == null) {
      const withBpm = lanes.find((l) => l.bpm);
      if (withBpm) setProjectBpm(Math.round(withBpm.bpm));
    }
  }, [lanes, projectBpm]);

  // Tempo-synced lanes follow project BPM changes.
  useEffect(() => {
    if (!projectBpm) return;
    setLanes((ls) => ls.map((l) => {
      if (!l.synced || !l.bpm) return l;
      const r = syncRateFor(l.bpm, projectBpm);
      return r && Math.abs(r - l.rate) > 1e-6 ? { ...l, rate: r } : l;
    }));
  }, [projectBpm]);

  // ── engine sync (structure: buffers/offsets/rates/pitch/gain) ───────────
  // The crossfader rides the first two lanes only — with N lanes there is no
  // single "other side" to fade towards, and the two-lane case is the pair the
  // Discover hand-off opens. At 0.5 both factors are 1, so it is inert until
  // touched. Lanes 3+ are never attenuated by it.
  const crossFactor = useCallback((idx) => {
    if (lanes.length < 2) return 1;
    if (idx === 0) return Math.min(1, 2 * (1 - cross));
    if (idx === 1) return Math.min(1, 2 * cross);
    return 1;
  }, [cross, lanes.length]);

  const gainFor = useCallback((l, idx) => {
    if (soloId) return l.id === soloId ? l.gain : 0;
    return l.muted ? 0 : l.gain * crossFactor(idx);
  }, [soloId, crossFactor]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    const seen = new Set();
    let structural = false;
    lanes.forEach((l, idx) => {
      if (!l.buffer) return;
      seen.add(l.id);
      if (!engine.voices.has(l.id)) {
        engine.setVoice(l.id, {
          buffer: l.buffer, offsetSec: l.offsetSec, rate: l.rate,
          semitones: l.semitones, gain: gainFor(l, idx),
        });
        structural = true; // new voice needs arming if we're mid-playback
      } else {
        // Live path: gain/pitch apply in place; rate/offset re-arm only when
        // they actually changed (updateVoiceParams checks) — so volume rides
        // and metadata refreshes never stutter playback.
        engine.updateVoiceParams(l.id, {
          rate: l.rate, offsetSec: l.offsetSec,
          semitones: l.semitones, gain: gainFor(l, idx),
        });
      }
    });
    for (const role of [...engine.voices.keys()]) {
      if (!seen.has(role)) { engine.removeVoice(role); structural = true; }
    }
    if (structural) engine.refresh();
  }, [lanes, gainFor]);

  useEffect(() => { engineRef.current?.setLoop(loop); }, [loop]);

  // ── lane moves ported from Audition ─────────────────────────────────────
  // Audition's "match key to the other deck" and "align downbeats" were its two
  // one-click alignment moves. With N lanes the reference is lane 1: it is the
  // first thing you added and, from Discover, the vocal everything else sits
  // under.
  const referenceLane = lanes[0] || null;

  const matchKeyToReference = (lane) => {
    if (!referenceLane || lane.id === referenceLane.id) return;
    const ref = shiftCamelot(referenceLane.camelot, referenceLane.semitones);
    if (!ref || !lane.camelot) { toast("Key data missing — analyze both tracks"); return; }
    // keyRel().suggest is the shift that takes the second key to the first, so
    // it is applied on top of whatever this lane is already shifted by.
    const suggest = keyRel(ref, shiftCamelot(lane.camelot, lane.semitones)).suggest ?? 0;
    const next = Math.max(-12, Math.min(12, lane.semitones + suggest));
    patchLane(lane.id, { semitones: next });
    toast(suggest ? `Pitched to ${ref} (${next > 0 ? "+" : ""}${next} st)` : "Already in key");
  };

  // Snap the lane so its nearest downbeat lands on the nearest project bar line.
  // Dragging already snaps, but only while you drag — this fixes a lane that
  // arrived pre-placed (from Discover) or drifted after a tempo change.
  const alignLaneToGrid = (lane) => {
    if (!projectBpm) { toast("Set a project BPM first"); return; }
    const downs = downbeatsOf(lane.beatTimes || [], lane.beatPhase)
      .map((t) => lane.offsetSec + t / lane.rate);
    if (!downs.length) { toast("No beat grid — analyze this track first"); return; }
    const bar = (60 / projectBpm) * 4;
    const anchor = downs.reduce((best, t) =>
      (Math.abs(t - position) < Math.abs(best - position) ? t : best), downs[0]);
    const delta = Math.round(anchor / bar) * bar - anchor;
    patchLane(lane.id, { offsetSec: lane.offsetSec + delta });
    toast(`Aligned to bar ${Math.round(anchor / bar) + 1}`);
  };

  const resetLane = (lane) => {
    patchLane(lane.id, { offsetSec: 0, rate: 1, semitones: 0, gain: 0.8,
                         muted: false, synced: false });
    toast("Lane reset");
  };

  // ── persistence (localStorage, debounced) ───────────────────────────────
  // Saving stays OFF until the restore finishes — otherwise the empty initial
  // state could overwrite the stored project before the async restore lands.
  const restoreRan = useRef(false);
  const seedAtMount = useRef(seed);
  useEffect(() => {
    if (restoreRan.current) return; // StrictMode double-invoke guard
    restoreRan.current = true;
    // Arriving with a PAIR from Discover means the user asked for that pair —
    // restoring the previous project first would only be thrown away. A single
    // track is added to the existing project, so that one still restores.
    const s = seedAtMount.current;
    if (s?.vocalId != null && s?.instId != null) { setRestored(true); return; }
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { /* corrupt */ }
    if (!saved?.lanes?.length) { setRestored(true); return; }
    // Restore once the library list is available so lanes get titles/BPM.
    api.getTracks().then((d) => {
      const byId = new Map(d.tracks.map((t) => [t.id, t]));
      if (saved.projectBpm) setProjectBpm(saved.projectBpm);
      if (saved.snapMode) setSnapMode(saved.snapMode);
      let n = 0;
      for (const sl of saved.lanes) {
        const t = byId.get(sl.songId);
        if (t) { addLane(t, sl.stem, sl); n++; }
      }
      if (n) toast(`Restored studio project (${n} lane${n === 1 ? "" : "s"})`);
    }).catch(() => {}).finally(() => setRestored(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!restored) return;
    const t = setTimeout(() => {
      const payload = {
        projectBpm, snapMode,
        lanes: lanes.map((l) => ({
          songId: l.songId, stem: l.stem, offsetSec: l.offsetSec, rate: l.rate,
          semitones: l.semitones, gain: l.gain, muted: l.muted, synced: l.synced,
          colorIdx: l.colorIdx,
        })),
      };
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch { /* full */ }
    }, 400);
    return () => clearTimeout(t);
  }, [lanes, projectBpm, snapMode, restored]);

  // ── Discover hand-off — what the Audition tab used to be (T4.1) ─────────
  // Open on one pair: vocal over bed, bed conformed to the vocal's tempo and
  // pitched by the shift the ranked list already computed, both lanes placed so
  // the winning section pair (T3.3) starts at the same instant.
  const seededFor = useRef(null);
  useEffect(() => {
    if (!tracks.length) return;
    if (seed?.vocalId == null && seed?.instId == null) return;
    const isPair = seed.vocalId != null && seed.instId != null;
    // A single track is appended to the saved project, so it has to wait for
    // the restore — otherwise it lands at index 0 and the restored lanes queue
    // up behind it, which silently reassigns the crossfader's A side and the
    // key reference. A pair replaces the project, so it never waits.
    if (!isPair && !restored) return;
    // `at` is a timestamp the sender bumps, so re-sending the SAME pair still
    // re-seeds instead of being swallowed as a no-op.
    const token = `${seed.vocalId ?? ""}:${seed.instId ?? ""}:${seed.at ?? ""}`;
    if (seededFor.current === token) return;
    seededFor.current = token;

    // One id only — "send this track to Studio" from Library. Add it as a lane
    // alongside whatever is already arranged; replacing the project would throw
    // away work the user never asked to lose.
    if (!isPair) {
      const only = seed.vocalId ?? seed.instId;
      const track = tracks.find((t) => t.id === only);
      if (!track) { toast("That track is not in the library"); return; }
      const want = seed.vocalId != null ? "vocals" : "instrumental";
      addLane(track, track.stems?.[want] ? want : "full");
      setRestored(true);
      onSeedConsumed?.();
      return;
    }

    const vTrack = tracks.find((t) => t.id === seed.vocalId);
    const bTrack = tracks.find((t) => t.id === seed.instId);
    if (!vTrack || !bTrack) { toast("Those tracks are not in the library"); return; }

    const vStem = vTrack.stems?.vocals ? "vocals" : "full";
    const bStem = bTrack.stems?.instrumental ? "instrumental" : "full";
    const vBpm = laneBpmFor(vTrack, vStem);
    const bBpm = laneBpmFor(bTrack, bStem);
    // The vocal sets the project tempo: it is the layer whose timing the ear
    // holds on to, and it is what stretch_factor on the row is relative to.
    const bpm = vBpm ? Math.round(vBpm) : projectBpm;
    const vRate = (bpm && vBpm && syncRateFor(vBpm, bpm)) || 1;
    const bRate = (bpm && bBpm && syncRateFor(bBpm, bpm)) || 1;

    // Line the two chosen sections up on the same project instant. Both offsets
    // stay ≥ 0 by pushing the pair forward instead of hanging one lane off the
    // front of the timeline, where the viewport cannot scroll to it.
    const vAt = (seed.vocalSectionStart ?? 0) / vRate;
    const bAt = (seed.instSectionStart ?? 0) / bRate;
    const base = Math.max(vAt, bAt);

    engineRef.current?.stop();
    for (const l of lanesRef.current) engineRef.current?.removeVoice(l.id);
    setLanes([]);
    setLoop(null); setSoloId(null); setCross(0.5);
    if (bpm) setProjectBpm(bpm);
    setPosition(base);
    engineRef.current?.seek(base);

    addLane(vTrack, vStem, { offsetSec: base - vAt, rate: vRate, semitones: 0,
                             gain: 0.85, synced: Boolean(bpm && vBpm), colorIdx: 0 });
    addLane(bTrack, bStem, { offsetSec: base - bAt, rate: bRate,
                             semitones: seed.semitoneShift ?? 0,
                             gain: 0.8, synced: Boolean(bpm && bBpm), colorIdx: 1 });
    setRestored(true);
    // Hand the seed back: leaving it live would re-seed — and throw away the
    // user's edits — every time they leave Studio and come back.
    onSeedConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, tracks, addLane, restored]);

  const clearProject = () => {
    engineRef.current?.stop();
    for (const l of lanesRef.current) engineRef.current?.removeVoice(l.id);
    setLanes([]); setLoop(null); setPosition(0); setSoloId(null); setSelectedId(null);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
    toast("Project cleared");
  };

  // ── painting ────────────────────────────────────────────────────────────
  useEffect(() => {
    for (const l of lanes) {
      const c = laneCanvasRefs.current.get(l.id);
      if (c) paintLane(c, l, viewStart, pps, l.id === selectedId);
    }
    if (rulerRef.current) paintRuler(rulerRef.current, viewStart, pps, projectBpm, loop);
  }, [lanes, viewStart, pps, viewW, selectedId, projectBpm, loop]);

  // Follow the playhead while playing.
  useEffect(() => {
    if (!isPlaying) return;
    if (position > viewStart + viewSecs * 0.92) setViewStart(position - viewSecs * 0.1);
    else if (position < viewStart - 0.5) setViewStart(Math.max(0, position - viewSecs * 0.1));
  }, [position, isPlaying, viewStart, viewSecs]);

  // ── wheel: pan / ctrl+zoom (non-passive listener so preventDefault works) ─
  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const rect = el.getBoundingClientRect();
        const px = Math.max(0, e.clientX - rect.left - HEADER_W);
        const anchorSec = viewStart + px / pps;
        const next = Math.min(MAX_PPS, Math.max(MIN_PPS, pps * (e.deltaY < 0 ? 1.18 : 1 / 1.18)));
        setPps(next);
        setViewStart(Math.max(0, anchorSec - px / next));
      } else {
        const d = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / pps;
        setViewStart((v) => Math.max(0, v + d));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [pps, viewStart]);

  // ── transport ───────────────────────────────────────────────────────────
  const togglePlay = useCallback(async () => {
    const engine = engineRef.current;
    if (!engine) return;
    if (engine.isPlaying) { engine.pause(); return; }
    setError(null);
    try { await engine.play(position); }
    catch (e) { setError(`Playback: ${e.message}`); }
  }, [position]);

  const stopPlayback = () => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.pause();
    const home = loop ? loop.start : 0;
    setPosition(home);
    engine.seek(home);
  };

  const seekTo = (pos) => {
    setPosition(pos);
    engineRef.current?.seek(pos);
  };

  // Audition looped 1/2/4/8 bars per deck; here the loop is the project's, so
  // the length is a project setting. Clicking the armed length disarms it.
  const toggleLoopBars = (bars) => {
    if (loop && bars === loopBars) { setLoop(null); return; }
    if (!projectBpm) { toast("Set a project BPM first"); return; }
    const bar = (60 / projectBpm) * 4;
    const start = Math.round(position / bar) * bar;
    setLoopBars(bars);
    setLoop({ start, end: start + bars * bar });
  };

  // Ruler: click/drag scrub; shift-drag sets the loop.
  const handleRulerDown = (e) => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    const posAt = (ev) => Math.max(0, viewStart + (ev.clientX - rect.left) / pps);
    if (e.shiftKey) {
      const anchor = posAt(e);
      const onMove = (me) => {
        const p = posAt(me);
        setLoop({ start: Math.min(anchor, p), end: Math.max(anchor, p) });
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      return;
    }
    seekTo(posAt(e));
    const onMove = (me) => seekTo(posAt(me));
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  // Lane drag: horizontal move with snap; click selects.
  // alt+click a beat line to declare it bar 1. Onset-strength phase detection
  // is fooled by syncopation and quiet intros; the ear is not, so the manual
  // value wins and is persisted against the stem the lane is showing.
  const claimDownbeat = (e, lane) => {
    const beats = lane.beatTimes || [];
    if (!beats.length) { toast("No beat grid — analyze this track first"); return; }
    const rect = e.currentTarget.getBoundingClientRect();
    const t = viewStart + (e.clientX - rect.left) / pps;
    // Lane beats are stored raw; the canvas draws them warped by rate + offset.
    const rawT = (t - lane.offsetSec) * lane.rate;
    let nearest = 0;
    for (let i = 1; i < beats.length; i++) {
      if (Math.abs(beats[i] - rawT) < Math.abs(beats[nearest] - rawT)) nearest = i;
    }
    const phase = phaseForDownbeatAt(nearest);
    patchLane(lane.id, { beatPhase: phase });
    api.setBeatPhase(lane.songId, lane.stem, phase)
      .then(() => toast(`Bar 1 set to beat ${nearest + 1} (phase ${phase})`))
      .catch((err) => toast(err.message || "Could not save downbeat"));
  };

  const handleLaneDown = (e, lane) => {
    e.preventDefault();
    setSelectedId(lane.id);
    if (e.altKey) { claimDownbeat(e, lane); return; }
    const startX = e.clientX;
    const startOffset = lane.offsetSec;
    const onMove = (me) => {
      const raw = startOffset + (me.clientX - startX) / pps;
      const { offset, snapped } = snapToGrid(
        raw, lanesRef.current.find((l) => l.id === lane.id) || lane,
        projectBpm, snapMode, viewStart, viewSecs, pps);
      setDragSnapped(snapped);
      patchLane(lane.id, { offsetSec: offset });
    };
    const onUp = () => {
      setDragSnapped(false);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  // ── keyboard ────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && selectedId) {
        e.preventDefault();
        const lane = lanesRef.current.find((l) => l.id === selectedId);
        if (!lane) return;
        const dir = e.key === "ArrowLeft" ? -1 : 1;
        const beat = projectBpm ? 60 / projectBpm : 0.1;
        patchLane(selectedId, { offsetSec: lane.offsetSec + dir * (e.shiftKey ? 0.01 : beat) });
      } else if (e.key === "l" || e.key === "L") {
        toggleLoopBars(loopBars);
      } else if ((e.key === "Delete" || e.key === "Backspace") && selectedId) {
        removeLane(selectedId);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ── header status ───────────────────────────────────────────────────────
  const audible = lanes.filter((l) => l.buffer && (soloId ? l.id === soloId : !l.muted) && l.gain > 0);
  const locked = projectBpm && audible.length >= 2 && audible.every((l) => {
    if (!l.bpm) return false;
    const eff = l.bpm * l.rate;
    return [0.5, 1, 2].some((m) => Math.abs(eff - projectBpm * m) / (projectBpm * m) < 0.02);
  });
  useEffect(() => {
    if (!onStatus) return;
    if (lanes.length === 0) onStatus({ text: "Studio — add tracks to begin" });
    else if (locked) onStatus({ locked: true, text: `GRID-LOCKED · ${projectBpm} BPM · ${lanes.length} lanes` });
    else onStatus({ text: `${lanes.length} lane${lanes.length === 1 ? "" : "s"}${projectBpm ? ` · ${projectBpm} BPM` : ""}` });
  }, [lanes.length, locked, projectBpm, onStatus]);

  // ── export ──────────────────────────────────────────────────────────────
  // One clip payload, shared by the bounce and the session export, so what you
  // hear here, what the mixdown renders and what lands in FL are three views of
  // the same arrangement rather than three chances to disagree.
  const audibleClips = () => lanes
    .filter((l) => !l.muted && l.gain > 0)
    .map((l) => ({
      song_id: l.songId, stem: l.stem, offset_sec: l.offsetSec,
      rate: l.rate, semitones: l.semitones, gain: l.gain,
    }));

  const handleExport = async () => {
    const clips = audibleClips();
    if (clips.length === 0) { toast("Nothing audible to export"); return; }
    setError(null);
    try {
      const { job_id } = await api.startMixdown(clips);
      setExportToken(null);
      setExportJobId(job_id);
      toast("Rendering mixdown WAV…");
    } catch (e) { setError(e.message); }
  };

  // A.4 — export the arrangement you built.
  //
  // This used to send only two song ids and let the server re-plan the pair,
  // which threw away every offset, rate, pitch and gain set here. Nudging a
  // lane until it sat right produced nothing you could take to FL: the only
  // ways out were a bounce (which you cannot mix) and a session rebuilt from
  // the engine's own opinion of the pair. It also refused any arrangement that
  // was not exactly one vocal over one instrumental, so a three-way or a
  // drums-swap could not be exported at all.
  const handleSessionExport = async () => {
    const clips = audibleClips();
    if (clips.length === 0) { toast("Nothing audible to export"); return; }
    setError(null);
    try {
      const { job_id } = await api.startSessionExport({
        clips, targetBpm: projectBpm || null,
      });
      setSessionToken(null);
      setSessionJobId(job_id);
      toast(`Rendering FL session (${clips.length} conformed lane`
        + `${clips.length === 1 ? "" : "s"} + click)…`);
    } catch (e) { setError(e.message); }
  };

  // ── derived / picker ────────────────────────────────────────────────────
  const playheadX = (position - viewStart) * pps;
  const pickerList = tracks
    .filter((t) => t.stems?.full || t.stems?.vocals || t.stems?.instrumental)
    .filter((t) => {
      const q = pickerSearch.toLowerCase();
      return !q || `${t.title} ${t.artist || ""}`.toLowerCase().includes(q);
    });

  const zoom = (f) => {
    const center = viewStart + viewSecs / 2;
    const next = Math.min(MAX_PPS, Math.max(MIN_PPS, pps * f));
    setPps(next);
    setViewStart(Math.max(0, center - viewW / next / 2));
  };
  const zoomFit = () => {
    const next = Math.min(MAX_PPS, Math.max(MIN_PPS, viewW / projectLen));
    setPps(next);
    setViewStart(0);
  };

  const scrollMax = Math.max(0, projectLen - viewSecs);

  return (
    <div className="page studio">
      {error && <div className="error-text" style={{ marginBottom: 8 }}>{error}</div>}

      {/* ── toolbar ── */}
      <div className="studio-toolbar">
        <button className={`play-btn ${isPlaying ? "playing" : "stopped"}`}
          onClick={togglePlay} disabled={lanes.every((l) => !l.buffer)}
          title="Play / pause (space)">
          {isPlaying ? "❚❚" : "▶"}
        </button>
        <button className="studio-btn" onClick={stopPlayback} title="Stop — return to start">■</button>
        <span className="studio-clock mono" title="bar.beat · time">
          {fmtBarsBeats(position, projectBpm)} <span className="faint">|</span> {fmtTime(position)}
        </span>

        <span className="studio-sep" />

        <span className="micro-label">BPM</span>
        <button className="studio-btn" onClick={() => projectBpm && setProjectBpm(projectBpm - 1)}>−</button>
        <input
          className="studio-bpm mono"
          type="number" min={40} max={220}
          value={projectBpm ?? ""}
          placeholder="—"
          onChange={(e) => {
            const v = Number(e.target.value);
            setProjectBpm(Number.isFinite(v) && v >= 40 && v <= 220 ? v : projectBpm);
          }}
          title="Project tempo — synced lanes stretch to follow"
        />
        <button className="studio-btn" onClick={() => projectBpm && setProjectBpm(projectBpm + 1)}>+</button>

        <span className="studio-sep" />

        <span className="micro-label">SNAP</span>
        <div className="snap-seg">
          {["bar", "beat", "off"].map((o) => (
            <button key={o} className={snapMode === o ? "active" : ""} onClick={() => setSnapMode(o)}>{o}</button>
          ))}
        </div>

        <span className="micro-label">LOOP</span>
        <div className="snap-seg">
          {[1, 2, 4, 8].map((n) => (
            <button key={n} className={loop && loopBars === n ? "active" : ""}
              onClick={() => toggleLoopBars(n)}
              title={`${n}-bar loop at the playhead — click again to clear · shift-drag the ruler for a custom loop`}>
              {n}
            </button>
          ))}
        </div>

        <span className="studio-sep" />

        {lanes.length >= 2 && (
          <>
            <span className="micro-label" title={`${lanes[0].title} ⟷ ${lanes[1].title}`}>
              A/B
            </span>
            <input className="studio-cross" type="range" min={0} max={1} step={0.01}
              value={cross} onChange={(e) => setCross(Number(e.target.value))}
              title="Crossfade between the first two lanes. Centred = both at full level." />
            {cross !== 0.5 && (
              <button className="studio-btn" onClick={() => setCross(0.5)}
                title="Centre the crossfader">·|·</button>
            )}
            <span className="studio-sep" />
          </>
        )}

        <span className="micro-label">ZOOM</span>
        <button className="studio-btn" onClick={() => zoom(1 / 1.4)} title="Zoom out (or ctrl+wheel)">−</button>
        <button className="studio-btn" onClick={() => zoom(1.4)} title="Zoom in (or ctrl+wheel)">+</button>
        <button className="studio-btn" onClick={zoomFit} title="Fit project">fit</button>

        <span className="spacer" style={{ flex: 1 }} />

        <button className="studio-btn" onClick={clearProject} disabled={lanes.length === 0}
          title="Remove all lanes and clear the saved project">✕ clear</button>
        <button className="export-btn" onClick={handleExport}
          disabled={exportJobId != null || lanes.length === 0}>
          ↓ Export WAV
        </button>
        {exportJobId && (
          <JobBadge jobId={exportJobId} onComplete={(job) => {
            setExportJobId(null);
            if (job.status === "completed") setExportToken(job.id);
          }} />
        )}
        {exportToken && (
          <a href={api.mixdownAudioUrl(exportToken)} target="_blank" rel="noreferrer"
            className="muted" style={{ fontSize: 12 }}>
            ↓ download mixdown
          </a>
        )}
        <button className="export-btn" onClick={handleSessionExport}
          disabled={sessionJobId != null || lanes.length === 0}
          title="Export every audible lane as its own conformed WAV — the arrangement you built, with your stretches, transposes and placements baked in so they all drop into FL at 0:00 together. Includes a click track and a README listing each lane's settings.">
          ↓ Export FL session
        </button>
        {sessionJobId && (
          <JobBadge jobId={sessionJobId} onComplete={(job) => {
            setSessionJobId(null);
            if (job.status === "completed") setSessionToken(job.id);
          }} />
        )}
        {sessionToken && (
          <a href={api.sessionArchiveUrl(sessionToken)} target="_blank" rel="noreferrer"
            className="muted" style={{ fontSize: 12 }}>
            ↓ download session
          </a>
        )}
      </div>

      {/* ── timeline ── */}
      <div className="studio-timeline" ref={viewRef}>
        <div className="studio-grid">
          {/* header row: corner + ruler */}
          <div className="studio-corner">
            <button className="studio-add" onClick={() => { setPicker(true); setPickerSearch(""); }}>
              ＋ Add track
            </button>
          </div>
          <div className="studio-ruler" onMouseDown={handleRulerDown}
            title="Click/drag to scrub · shift-drag to set a loop">
            <canvas ref={rulerRef} />
          </div>

          {lanes.map((l) => {
            const effBpm = l.bpm ? l.bpm * l.rate : null;
            const rgb = LANE_COLORS[l.colorIdx % LANE_COLORS.length].line;
            const isSel = l.id === selectedId;
            const syncRate = projectBpm ? syncRateFor(l.bpm, projectBpm) : null;
            return (
              <Fragment key={l.id}>
                {/* lane header */}
                <div className={`studio-lanehead${isSel ? " selected" : ""}`}
                  style={{ borderLeft: `3px solid rgba(${rgb},0.9)` }}
                  onClick={() => setSelectedId(l.id)}>
                  <div className="lh-row">
                    <TrackArt id={l.songId} thumbnail={l.thumbnail} className="lh-art" />
                    <div className="lh-id">
                      <div className="lh-title" title={`${l.title}${l.artist ? ` — ${l.artist}` : ""}`}>
                        {l.title}
                      </div>
                      <div className="lh-meta">
                        {/* Stem switch in place (from Audition's deck picker):
                            the lane keeps its placement, so you can hear the
                            same arrangement with a different layer. */}
                        <span className="lh-stem-seg" onClick={(e) => e.stopPropagation()}>
                          {STEM_ORDER.map((s) => {
                            const src = tracks.find((t) => t.id === l.songId);
                            const ok = Boolean(src?.stems?.[s]);
                            // Hide four-stem buttons entirely for two-stem
                            // tracks rather than showing five dead controls.
                            if (!ok && !["vocals", "instrumental", "full"].includes(s)) return null;
                            return (
                              <button key={s} disabled={!ok || l.stem === s}
                                className={l.stem === s ? "active" : ""}
                                style={l.stem === s ? { color: `rgb(${rgb})`, borderColor: `rgba(${rgb},0.6)` } : undefined}
                                title={ok ? `Play the ${s} stem in this lane` : `No ${s} audio for this track`}
                                onClick={() => setLaneStem(l, s)}>
                                {STEM_LABEL[s]}
                              </button>
                            );
                          })}
                        </span>
                        {effBpm ? <span className="mono">{effBpm.toFixed(1)}</span> : <span className="faint">no BPM</span>}
                        <KeyChip camelot={shiftCamelot(l.camelot, l.semitones)} fallback="?"
                          style={{ fontSize: 10, padding: "1px 5px" }} />
                      </div>
                    </div>
                    <div className="lh-order">
                      <button onClick={(e) => { e.stopPropagation(); moveLane(l.id, -1); }} title="Move up">▲</button>
                      <button onClick={(e) => { e.stopPropagation(); moveLane(l.id, 1); }} title="Move down">▼</button>
                    </div>
                  </div>

                  <div className="lh-row lh-controls">
                    <button className={`lh-btn${l.muted ? " on" : ""}`}
                      onClick={(e) => { e.stopPropagation(); patchLane(l.id, { muted: !l.muted }); }}
                      title="Mute">M</button>
                    <button className={`lh-btn solo${soloId === l.id ? " on" : ""}`}
                      onClick={(e) => { e.stopPropagation(); setSoloId(soloId === l.id ? null : l.id); }}
                      title="Solo">S</button>
                    <input className="lh-gain" type="range" min={0} max={1.25} step={0.01}
                      value={l.gain}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => patchLane(l.id, { gain: Number(e.target.value) })}
                      title={`Gain ${(l.gain * 24 - 12).toFixed(1)} dB`} />
                    <button className="lh-x"
                      onClick={(e) => { e.stopPropagation(); removeLane(l.id); }}
                      title="Remove lane">✕</button>
                  </div>

                  <div className="lh-row lh-controls">
                    <button
                      className={`lh-btn sync${l.synced ? " on" : ""}`}
                      disabled={!l.bpm || !projectBpm || !syncRate}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (l.synced) patchLane(l.id, { synced: false, rate: 1 });
                        else patchLane(l.id, { synced: true, rate: syncRate });
                      }}
                      title={l.bpm && syncRate
                        ? `Tempo-sync to ${projectBpm} BPM (stretch ×${syncRate.toFixed(3)})`
                        : "Needs BPM analysis to sync"}>
                      SYNC
                    </button>
                    {/* Editable while unsynced — Audition let you type a target
                        BPM per deck, and SYNC alone cannot express "play this
                        at 0.97× because it drags". */}
                    <input className="lh-rate mono" type="number" step={0.005}
                      min={0.5} max={2} value={Number(l.rate.toFixed(3))}
                      disabled={l.synced}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (Number.isFinite(v) && v >= 0.5 && v <= 2) patchLane(l.id, { rate: v });
                      }}
                      title={l.synced ? "Synced to the project tempo — turn SYNC off to set this by hand"
                                      : "Stretch factor (speed). 1 = original tempo."} />
                    <span className="lh-pitch">
                      <button className="lh-btn" title="Pitch −1 st"
                        onClick={(e) => { e.stopPropagation(); patchLane(l.id, { semitones: Math.max(-12, l.semitones - 1) }); }}>−</button>
                      <span className="mono" style={{ minWidth: 30, textAlign: "center" }}>
                        {l.semitones > 0 ? "+" : ""}{l.semitones}st
                      </span>
                      <button className="lh-btn" title="Pitch +1 st"
                        onClick={(e) => { e.stopPropagation(); patchLane(l.id, { semitones: Math.min(12, l.semitones + 1) }); }}>+</button>
                    </span>
                  </div>
                  <div className="lh-row lh-controls">
                    <button className="lh-btn"
                      disabled={!referenceLane || l.id === referenceLane.id || !l.camelot}
                      onClick={(e) => { e.stopPropagation(); matchKeyToReference(l); }}
                      title={referenceLane && l.id !== referenceLane.id
                        ? `Pitch this lane into ${referenceLane.title}'s key`
                        : "The first lane is the key reference"}>
                      ⚡ key
                    </button>
                    <button className="lh-btn"
                      onClick={(e) => { e.stopPropagation(); alignLaneToGrid(l); }}
                      title="Snap this lane's nearest downbeat onto the bar grid">
                      ⇥ grid
                    </button>
                    <button className="lh-btn"
                      onClick={(e) => { e.stopPropagation(); resetLane(l); }}
                      title="Reset this lane's position, tempo, pitch and level">
                      ↺
                    </button>
                  </div>
                  {(l.loading || l.loadError) && (
                    <div className="lh-note">{l.loadError || "decoding…"}</div>
                  )}
                </div>

                {/* lane canvas */}
                <div className={`studio-lane${dragSnapped && isSel ? " snapped" : ""}`}
                  onMouseDown={(e) => handleLaneDown(e, l)}>
                  <canvas ref={(c) => {
                    if (c) laneCanvasRefs.current.set(l.id, c);
                    else laneCanvasRefs.current.delete(l.id);
                  }} />
                </div>
              </Fragment>
            );
          })}

          {lanes.length === 0 && (
            <div className="studio-empty">
              <div className="step"><span className="n">1</span> Hit <b>＋ Add track</b> and drop in a vocal from one song and an instrumental from another (any number of lanes works).</div>
              <div className="step"><span className="n">2</span> Press <b>SYNC</b> on each lane to conform it to the project BPM, then drag clips — downbeats click onto the bar grid.</div>
              <div className="step"><span className="n">3</span> <span className="kbd">space</span> to play · <span className="kbd">←</span><span className="kbd">→</span> nudge the selected lane · <span className="kbd">L</span> 8-bar loop · pitch ± per lane to match keys · Export WAV when it slaps.</div>
            </div>
          )}
        </div>

        {/* playhead overlay */}
        {playheadX >= 0 && playheadX <= viewW && lanes.length > 0 && (
          <div className="studio-playhead" style={{ left: HEADER_W + playheadX }} />
        )}
      </div>

      {/* horizontal scrollbar */}
      <input
        className="studio-scroll"
        type="range" min={0} max={Math.max(0.001, scrollMax)} step={0.05}
        value={Math.min(viewStart, scrollMax)}
        onChange={(e) => setViewStart(Number(e.target.value))}
      />
      <div className="hint" style={{ marginTop: 4 }}>
        wheel = pan · ctrl+wheel = zoom · drag a clip to move it (snap: {snapMode}) ·{" "}
        <span className="kbd">space</span> play · <span className="kbd">←</span><span className="kbd">→</span> nudge ·{" "}
        <span className="kbd">L</span> loop · <span className="kbd">⌫</span> remove lane
      </div>

      {/* ── track picker ── */}
      {picker && <div className="picker-overlay" onClick={() => setPicker(false)} />}
      {picker && (
        <div className="picker-menu studio-picker">
          <div className="search-box" style={{ width: "auto", margin: "2px 4px 6px" }}>
            <span className="ico">⌕</span>
            <input autoFocus placeholder="Search library…" value={pickerSearch}
              onChange={(e) => setPickerSearch(e.target.value)} />
          </div>
          {pickerList.map((t) => (
            <div key={t.id} className="picker-row studio-picker-row">
              <TrackArt id={t.id} thumbnail={t.thumbnail} className="art" />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="t">{t.title}</div>
                <div className="a">{t.artist || "—"}</div>
              </div>
              <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                {t.features?.full?.bpm ? t.features.full.bpm.toFixed(1) : "—"}
              </div>
              {t.features?.full?.camelot && (
                <KeyChip camelot={t.features.full.camelot} style={{ fontSize: 11, padding: "2px 6px" }} />
              )}
              <div className="studio-stem-btns">
                {STEM_ORDER.filter((s) => t.stems?.[s]
                    || ["vocals", "instrumental", "full"].includes(s)).map((s) => (
                  <button key={s} disabled={!t.stems?.[s]}
                    title={t.stems?.[s] ? `Add ${s} lane` : `No ${s} audio yet`}
                    onClick={() => { addLane(t, s); setPicker(false); }}>
                    {STEM_LABEL[s]}
                  </button>
                ))}
              </div>
            </div>
          ))}
          {pickerList.length === 0 && <div className="empty" style={{ padding: 12 }}>No processed tracks yet — import some in the Import tab.</div>}
        </div>
      )}
    </div>
  );
}
