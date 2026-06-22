import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";

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

function SectionTimeline({
  sections, durationSecs, label, pps, offsetSecs, onOffsetChange,
  selectedId, onSectionClick, waveform = [], beatTimes = [], trackRole = "vocal",
  onPlay, isPlaying, playheadTime,
}) {
  const isDraggable = onOffsetChange != null;
  const [dragging, setDragging] = useState(false);
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
      // Colored fill
      ctx.fillStyle = trackRole === "vocal"
        ? "rgba(99,179,255,0.18)"
        : "rgba(251,191,36,0.18)";
      ctx.fill();
      // Colored inner stroke
      ctx.strokeStyle = trackRole === "vocal"
        ? "rgba(99,179,255,0.4)"
        : "rgba(251,191,36,0.4)";
      ctx.lineWidth = 1;
      ctx.stroke();
      // Bold white outline on top
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
      onOffsetChange(dragRef.current.startOffset + dx / pps);
    };
    const onUp = () => {
      setDragging(false);
      dragRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  // Red playhead position as CSS percentage
  const playheadPct = (() => {
    if (playheadTime == null) return null;
    const pct = ((playheadTime + offsetSecs) * pps / TIMELINE_WIDTH) * 100;
    return pct >= 0 && pct <= 100 ? pct : null;
  })();

  return (
    <div className="timeline-row">
      <button
        className={`track-play-btn${isPlaying ? " playing" : ""}`}
        onClick={onPlay}
        disabled={!onPlay}
        title={isPlaying ? "Pause" : "Play from marker"}
      >
        {isPlaying ? "⏸" : "▶"}
      </button>
      <div className="timeline-row-label">{label}</div>
      <div
        className={`timeline-track${isDraggable ? " draggable" : ""}${dragging ? " dragging" : ""}`}
        onMouseDown={handleMouseDown}
      >
        <canvas
          ref={canvasRef}
          width={TIMELINE_WIDTH}
          height={TIMELINE_HEIGHT}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
                   zIndex: 1, pointerEvents: "none" }}
        />
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

export function AuditionStudio({ seed }) {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [vocalId, setVocalId] = useState(seed?.vocalId ?? null);
  const [instId, setInstId] = useState(seed?.instId ?? null);
  const [plan, setPlan] = useState(null);
  const [previewJobId, setPreviewJobId] = useState(null);
  const [previewTs, setPreviewTs] = useState(null);

  const [vocalSections, setVocalSections] = useState([]);
  const [instSections, setInstSections] = useState([]);
  const [vocalOffset, setVocalOffset] = useState(0);
  const [instOffset, setInstOffset] = useState(0);
  const [selVocal, setSelVocal] = useState(null);
  const [selInst, setSelInst] = useState(null);
  const [vocalWaveform, setVocalWaveform] = useState({ waveform: [], beat_times: [] });
  const [instWaveform,  setInstWaveform]  = useState({ waveform: [], beat_times: [] });

  // Audio playback
  const vocalAudioRef   = useRef(null);
  const instAudioRef    = useRef(null);
  const previewAudioRef = useRef(null);
  const previewStopRef  = useRef(null);
  const rafRef          = useRef(null);

  const [vocalPlaying,   setVocalPlaying]   = useState(false);
  const [instPlaying,    setInstPlaying]    = useState(false);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [playheadTimes,  setPlayheadTimes]  = useState({ vocal: null, inst: null });

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

  const pps = useMemo(() => {
    const maxDur = Math.max(vocalTrack?.duration_secs ?? 0, instTrack?.duration_secs ?? 0, 1);
    return TIMELINE_WIDTH / maxDur;
  }, [vocalTrack, instTrack]);

  // Timestamp at the center marker for each track
  const vocalCenterTime = useMemo(() => {
    if (!vocalTrack) return 0;
    return Math.max(0, Math.min(vocalTrack.duration_secs, TIMELINE_WIDTH / (2 * pps) - vocalOffset));
  }, [pps, vocalOffset, vocalTrack]);

  const instCenterTime = useMemo(() => {
    if (!instTrack) return 0;
    return Math.max(0, Math.min(instTrack.duration_secs, TIMELINE_WIDTH / (2 * pps) - instOffset));
  }, [pps, instOffset, instTrack]);

  useEffect(() => {
    setVocalSections([]);
    setSelVocal(null);
    setVocalWaveform({ waveform: [], beat_times: [] });
    setVocalOffset(0);
    if (vocalAudioRef.current) { vocalAudioRef.current.pause(); }
    setVocalPlaying(false);
    if (!vocalId) return;
    api.getSections(vocalId)
      .then((d) => setVocalSections(d.sections))
      .catch(() => {});
    api.getWaveform(vocalId, "vocals")
      .then(setVocalWaveform)
      .catch(() => {});
  }, [vocalId]);

  useEffect(() => {
    setInstSections([]);
    setInstOffset(0);
    setSelInst(null);
    setInstWaveform({ waveform: [], beat_times: [] });
    if (instAudioRef.current) { instAudioRef.current.pause(); }
    setInstPlaying(false);
    if (!instId) return;
    api.getSections(instId)
      .then((d) => setInstSections(d.sections))
      .catch(() => {});
    api.getWaveform(instId, "instrumental")
      .then(setInstWaveform)
      .catch(() => {});
  }, [instId]);

  useEffect(() => {
    setPlan(null);
    setPreviewTs(null);
    setPreviewJobId(null);
    if (previewAudioRef.current) { previewAudioRef.current.pause(); }
    setPreviewPlaying(false);
    if (vocalId == null || instId == null || vocalId === instId) return;
    let cancelled = false;
    api.getMashupPlan(vocalId, instId)
      .then((p) => !cancelled && setPlan(p))
      .catch((e) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, [vocalId, instId]);

  useEffect(() => {
    if (!previewTs || !previewAudioRef.current) return;
    previewAudioRef.current.src = `${api.previewAudioUrl(vocalId, instId)}&t=${previewTs}`;
    previewAudioRef.current.load();
  }, [previewTs, vocalId, instId]);

  // rAF loop for red playhead while audio is playing
  useEffect(() => {
    if (!vocalPlaying && !instPlaying && !previewPlaying) {
      cancelAnimationFrame(rafRef.current);
      setPlayheadTimes({ vocal: null, inst: null });
      return;
    }
    const tick = () => {
      const next = { vocal: null, inst: null };
      if (vocalPlaying && vocalAudioRef.current) {
        next.vocal = vocalAudioRef.current.currentTime;
      }
      if (instPlaying && instAudioRef.current) {
        next.inst = instAudioRef.current.currentTime;
      }
      if (previewPlaying && previewAudioRef.current) {
        const t = previewAudioRef.current.currentTime;
        next.vocal = vocalCenterTime + t;
        next.inst  = instCenterTime + t;
      }
      setPlayheadTimes(next);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [vocalPlaying, instPlaying, previewPlaying, vocalCenterTime, instCenterTime]);

  const stopAllAudio = () => {
    if (vocalAudioRef.current)   { vocalAudioRef.current.pause(); }
    if (instAudioRef.current)    { instAudioRef.current.pause(); }
    if (previewAudioRef.current) { previewAudioRef.current.pause(); }
    if (previewStopRef.current && previewAudioRef.current) {
      previewAudioRef.current.removeEventListener("timeupdate", previewStopRef.current);
      previewStopRef.current = null;
    }
    setVocalPlaying(false);
    setInstPlaying(false);
    setPreviewPlaying(false);
  };

  const handlePlayVocal = () => {
    const audio = vocalAudioRef.current;
    if (!audio || !vocalId) return;
    if (vocalPlaying) {
      audio.pause();
      setVocalPlaying(false);
    } else {
      stopAllAudio();
      audio.currentTime = vocalCenterTime;
      audio.play().catch(() => {});
      setVocalPlaying(true);
    }
  };

  const handlePlayInst = () => {
    const audio = instAudioRef.current;
    if (!audio || !instId) return;
    if (instPlaying) {
      audio.pause();
      setInstPlaying(false);
    } else {
      stopAllAudio();
      audio.currentTime = instCenterTime;
      audio.play().catch(() => {});
      setInstPlaying(true);
    }
  };

  const handlePlayPreview = () => {
    const audio = previewAudioRef.current;
    if (!audio || !previewTs) return;
    if (previewPlaying) {
      audio.pause();
      if (previewStopRef.current) {
        audio.removeEventListener("timeupdate", previewStopRef.current);
        previewStopRef.current = null;
      }
      setPreviewPlaying(false);
    } else {
      stopAllAudio();
      audio.currentTime = 0;
      const stopAt = 30;
      const checkStop = () => {
        if (audio.currentTime >= stopAt) {
          audio.pause();
          audio.removeEventListener("timeupdate", checkStop);
          previewStopRef.current = null;
          setPreviewPlaying(false);
        }
      };
      previewStopRef.current = checkStop;
      audio.addEventListener("timeupdate", checkStop);
      audio.play().catch(() => {});
      setPreviewPlaying(true);
    }
  };

  const renderPreview = async () => {
    setError(null);
    try {
      const { job_id } = await api.startPreview(vocalId, instId, vocalCenterTime, instCenterTime);
      setPreviewJobId(job_id);
    } catch (e) {
      setError(e.message);
    }
  };

  const samePair = vocalId != null && instId === vocalId;
  const showTimeline = vocalSections.length > 0 || instSections.length > 0;

  const alignmentText = (() => {
    if (!vocalTrack || !instTrack) {
      return "Drag either track to align sections under the center marker ↕";
    }
    return `Marker — Vocal: ${fmt(vocalCenterTime)}  |  Instrumental: ${fmt(instCenterTime)}`;
  })();

  return (
    <div className="panel">
      <h2 style={{ margin: 0 }}>Audition Studio</h2>
      <p className="muted" style={{ marginTop: 4 }}>
        Pick a vocal and an instrumental, drag either waveform to line up sections under the
        center marker, then preview the pair with the instrumental time-stretched to the
        vocal tempo.
      </p>

      {error && <div className="error-text" style={{ marginTop: 8 }}>{error}</div>}

      <div className="audition-pickers" style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 12 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="muted">Vocal (top)</span>
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
          <span className="muted">Instrumental (bed)</span>
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

          <SectionTimeline
            sections={vocalSections}
            durationSecs={vocalTrack?.duration_secs ?? 0}
            label="Vocal"
            pps={pps}
            offsetSecs={vocalOffset}
            onOffsetChange={setVocalOffset}
            selectedId={selVocal?.id}
            onSectionClick={setSelVocal}
            waveform={vocalWaveform.waveform}
            beatTimes={vocalWaveform.beat_times}
            trackRole="vocal"
            onPlay={vocalId ? handlePlayVocal : null}
            isPlaying={vocalPlaying}
            playheadTime={playheadTimes.vocal}
          />

          <SectionTimeline
            sections={instSections}
            durationSecs={instTrack?.duration_secs ?? 0}
            label="Inst"
            pps={pps}
            offsetSecs={instOffset}
            onOffsetChange={setInstOffset}
            selectedId={selInst?.id}
            onSectionClick={setSelInst}
            waveform={instWaveform.waveform}
            beatTimes={instWaveform.beat_times}
            trackRole="instrumental"
            onPlay={instId ? handlePlayInst : null}
            isPlaying={instPlaying}
            playheadTime={playheadTimes.inst}
          />

          <div className="alignment-readout">{alignmentText}</div>

          <div className="preview-play-row">
            <button
              className={`preview-play-btn${previewPlaying ? " playing" : ""}`}
              onClick={handlePlayPreview}
              disabled={!previewTs}
              title={previewTs ? "Play 30s of mashup from center marker" : "Render a preview first"}
            >
              {previewPlaying ? "⏸ Pause preview" : "▶ Play 30s mashup from marker"}
            </button>
            {!previewTs && <span className="muted" style={{ fontSize: "0.75rem" }}>render a preview below first</span>}
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

          {(vocalOffset !== 0 || instOffset !== 0) && (
            <button
              className="secondary"
              style={{ alignSelf: "flex-start", marginTop: 4, fontSize: "0.75rem", padding: "3px 8px" }}
              onClick={() => { setVocalOffset(0); setInstOffset(0); }}
            >
              Reset alignment
            </button>
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

          <div className="actions" style={{ marginTop: 16, alignItems: "center" }}>
            {previewJobId ? (
              <JobBadge
                jobId={previewJobId}
                onComplete={() => {
                  setPreviewJobId(null);
                  setPreviewTs(Date.now());
                }}
              />
            ) : (
              <button onClick={renderPreview} disabled={samePair}>
                {previewTs ? "Re-render preview" : "Render preview"}
              </button>
            )}
          </div>

          {plan.pairings?.length > 0 && (
            <p className="muted" style={{ marginTop: 12, fontSize: "0.8rem" }}>
              Auto-aligned on: {plan.pairings[0].note}.
            </p>
          )}
        </div>
      )}

      {/* Hidden audio elements */}
      <audio
        ref={vocalAudioRef}
        preload="none"
        src={vocalId ? api.audioUrl(vocalId, "vocals") : ""}
        onEnded={() => setVocalPlaying(false)}
        style={{ display: "none" }}
      />
      <audio
        ref={instAudioRef}
        preload="none"
        src={instId ? api.audioUrl(instId, "instrumental") : ""}
        onEnded={() => setInstPlaying(false)}
        style={{ display: "none" }}
      />
      <audio
        ref={previewAudioRef}
        preload="none"
        onEnded={() => { setPreviewPlaying(false); previewStopRef.current = null; }}
        style={{ display: "none" }}
      />
    </div>
  );
}
