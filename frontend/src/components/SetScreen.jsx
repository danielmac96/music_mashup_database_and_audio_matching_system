import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { camelotColor, cssVar, parseCamelot } from "../theme";
import { ScreenHeader } from "../shell/ScreenHeader";
import { KeyChip } from "./KeyChip";
import { keyOf, spanLabel } from "./pairs/pairModel";
import { RecipeStrip } from "./pairs/RecipeStrip";
import { useJobPolling } from "../hooks/useJobPolling";

// Sets (revamp step 6): one active set — a mix's running order. A slot is a
// pair, a whole track, a break or a gap (matcher/setflow). Slots sit on a
// timeline as wide as they last; a dot on every boundary says whether the move
// fits (±1 Camelot step and ±4% tempo); a flow graph under it draws the tempo,
// the key on the wheel and the energy curve you drag; the inspector on the
// right changes with what is selected — the set, a slot, a gap (fill it from
// /suggest) or a transition (hear it, rendered server-side).

const BASE_KEY = "mashup.rekordboxBase.v1";
const ZOOM_KEY = "mashup.setZoom.v1";
const PAD = 16;              // the timeline's side gutter; the flow graph shares it
const FLOW_H = 190;
const MIN_SLOT_PX = 14;

// How a mashup comes in from the one before it. Keys are
// matcher/setflow.TRANSITIONS (a test pins the two equal); the server
// suggests one per move and the user can override it.
export const MOVES = [
  ["bed_swap", "bed swap"], ["vocal_swap", "vocal swap"], ["blend", "blend"],
  ["echo_out", "echo out"], ["cut", "cut"],
];
const BAR_CHOICES = [4, 8, 16];

const PRESETS = [
  ["Warm-up", [{ t: 0, e: 0.25 }, { t: 0.7, e: 0.5 }, { t: 1, e: 0.62 }]],
  ["Peak time", [{ t: 0, e: 0.65 }, { t: 0.4, e: 0.88 }, { t: 1, e: 0.95 }]],
  ["Journey", [{ t: 0, e: 0.35 }, { t: 0.3, e: 0.55 }, { t: 0.62, e: 0.88 },
    { t: 0.8, e: 0.7 }, { t: 1, e: 0.92 }]],
];

const FIT = {
  ok: ["✓", "ok", "within ±1 key step and ±4% tempo"],
  warn: ["!", "warn", "one rule broken"],
  bad: ["!", "bad", "key and tempo both jump"],
  unknown: ["·", "none", "unmeasured"],
  none: ["·", "none", "a break or a gap — nothing to match"],
};

const mmss = (s) => {
  if (s == null || !Number.isFinite(s)) return "—";
  const n = Math.max(0, Math.round(s));
  return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`;
};
const hmmss = (s) => {
  const n = Math.max(0, Math.round(s || 0));
  return `${Math.floor(n / 3600)}:${String(Math.floor((n % 3600) / 60)).padStart(2, "0")}:${String(n % 60).padStart(2, "0")}`;
};
const pct = (x) => (x == null ? "?" : `${x > 0 ? "+" : ""}${x.toFixed(1)}%`);

// What a slot shows: its two lines, and a name for tooltips and toasts.
function slotText(item) {
  const k = item.kind || "pair";
  if (k === "pair") {
    return { t1: item.vocal_title || "?", t2: item.inst_title || "?",
             name: `${item.vocal_title} over ${item.inst_title}` };
  }
  if (k === "track") return { t1: item.title || "?", t2: item.artist || "", name: item.title || "track" };
  if (k === "break") return { t1: item.label || "Break", t2: "pause", name: item.label || "break" };
  return { t1: "Gap", t2: "fill me", name: "gap" };
}

// Re-adding a removed slot exactly where it was: the Undo of a removal.
function slotBody(item, position) {
  const k = item.kind || "pair";
  if (k === "pair") {
    return { kind: k, vocal_song_id: item.vocal_song_id, inst_song_id: item.inst_song_id,
             vocal_section: item.vocal_section_idx ?? null, inst_section: item.inst_section_idx ?? null,
             position };
  }
  if (k === "track") return { kind: k, song_id: item.song_id, position };
  return { kind: k, label: item.label || null, duration_secs: item.duration_secs ?? null, position };
}

// The tempo curve: every slot that plays lands at its point between start and
// end BPM, both sides stretched to it. Blank start = off: each mashup lands at
// its vocal's own tempo.
function TempoControl({ plan, onSave }) {
  const [start, setStart] = useState(plan?.start_bpm ?? "");
  const [end, setEnd] = useState(plan?.end_bpm ?? "");
  useEffect(() => { setStart(plan?.start_bpm ?? ""); setEnd(plan?.end_bpm ?? ""); }, [plan]);
  const commit = () => {
    const s0 = Number(start), e0 = Number(end);
    if (!start) { if (plan) onSave(null); return; }
    if (!(s0 > 40 && s0 < 220)) return;
    const next = { start_bpm: s0, end_bpm: end && e0 > 40 && e0 < 220 ? e0 : null };
    if (next.start_bpm !== plan?.start_bpm || next.end_bpm !== (plan?.end_bpm ?? null)) onSave(next);
  };
  const key = (e) => { if (e.key === "Enter") e.currentTarget.blur(); };
  return (
    <div className="set-tempo mono"
      title="A tempo curve for the set: each slot lands at its point between start and end, vocal and bed both stretched to it. Leave start blank to land each at its own tempo.">
      <span className="muted">tempo</span>
      <input value={start} placeholder="off" onChange={(e) => setStart(e.target.value)}
        onBlur={commit} onKeyDown={key} aria-label="Start BPM" />
      <span className="muted">→</span>
      <input value={end} placeholder={start ? "hold" : "—"} onChange={(e) => setEnd(e.target.value)}
        onBlur={commit} onKeyDown={key} aria-label="End BPM" disabled={!start} />
      {plan && <button className="ui-link" onClick={() => onSave(null)}>off</button>}
    </div>
  );
}

// The timeline: slots left to right, each as wide as it lasts, with the fit
// dot on every boundary. Drag a slot onto another to move it there.
function SetTimeline({ items, f, layout, pps, sel, onSelect, onMove }) {
  const drag = useRef(null);
  const total = layout.total;
  const step = pps * 60 > 70 ? 60 : pps * 300 > 70 ? 300 : 900;
  const ticks = [];
  for (let t = 0; t <= total; t += step) ticks.push(t);
  return (
    <div className="set-timeline" aria-label="Set timeline" style={{ width: layout.width }}>
      <div className="ss-ticks">
        {ticks.map((t) => <span key={t} className="mono" style={{ left: t * pps }}>{Math.floor(t / 60)}:00</span>)}
      </div>
      <div className="ss-slots">
        {items.map((it, k) => {
          const kind = it.kind || "pair";
          const land = f.landings[k] || {};
          const txt = slotText(it);
          const on = sel?.kind === "slot" && sel.id === it.item_id;
          return (
            <div key={it.item_id} className={`set-tl-block ${kind}${on ? " on" : ""}${it.stale ? " stale" : ""}`}
              style={{ left: layout.x[k] * pps, width: Math.max(MIN_SLOT_PX, layout.w[k] * pps) }}
              draggable title={`${k + 1}. ${txt.name}`}
              onClick={() => onSelect({ kind: "slot", id: it.item_id })}
              onDragStart={(e) => { drag.current = it.item_id; try { e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", String(it.item_id)); } catch { /* ignore */ } }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); const from = drag.current; drag.current = null; if (from != null && from !== it.item_id) onMove(from, k); }}>
              <div className="ss-card">
                <div className="ss-band" />
                <div className="ss-body">
                  <div className="ss-top">
                    <span className="mono muted">{k + 1}</span>
                    <span className="ss-kind mono">{kind.toUpperCase()}</span>
                    {land.camelot && <KeyChip camelot={land.camelot} />}
                  </div>
                  <span className="ss-t1">{txt.t1}</span>
                  <span className="ss-t2">{txt.t2}</span>
                  <span className="grow" />
                  <div className="ss-foot mono">
                    <span>{land.bpm ? land.bpm.toFixed(1) : "—"}</span>
                    <span>{mmss(layout.w[k])}</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        {f.transitions.map((t, i) => {
          const [glyph, tone, why] = FIT[t.fit] || FIT.unknown;
          const on = sel?.kind === "trans" && sel.i === i;
          const tip = t.fit === "none" ? `Transition ${i + 1} → ${i + 2}: ${why}`
            : `Transition ${i + 1} → ${i + 2}: ${t.from_key || "?"} → ${t.to_key || "?"}`
              + `${t.key_steps != null ? ` (${t.key_steps} step${t.key_steps === 1 ? "" : "s"})` : ""} · tempo ${pct(t.tempo_pct)} — ${why}`;
          return (
            <button key={i} className={`ss-dot ${tone}${on ? " on" : ""}`} title={tip} aria-label={tip}
              style={{ left: layout.x[i + 1] * pps }}
              onClick={() => onSelect({ kind: "trans", i })}>{glyph}</button>
          );
        })}
      </div>
    </div>
  );
}

// Tempo (top band), the key as a dot on the wheel (bottom band; a jump over
// one step drawn amber), each slot's energy as a tick, and the energy curve —
// a dashed line of points you drag (shift-click adds, double-click removes).
function FlowGraph({ items, f, layout, pps, curve, sel, onCurve, onCommit }) {
  const ref = useRef(null);
  const drag = useRef(null);
  const W = layout.width;
  const S = layout.span;
  const yE = (e) => 10 + (1 - e) * 170;
  const centre = (k) => (layout.x[k] + layout.w[k] / 2) * pps;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(W * dpr) || cv.height !== FLOW_H * dpr) {
      cv.width = Math.round(W * dpr); cv.height = FLOW_H * dpr;
    }
    const c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, FLOW_H);
    const ink = { band: cssVar("--well"), grid: cssVar("--line-soft"), label: cssVar("--faint"),
                  fg: cssVar("--fg"), key: cssVar("--run"), warn: cssVar("--warn"),
                  sel: cssVar("--accent-bg"), bg: cssVar("--bg") };
    c.fillStyle = ink.band; c.fillRect(0, 10, W, 80); c.fillRect(0, 100, W, 80);
    c.font = `9.5px ${cssVar("--font-mono") || "monospace"}`; c.textBaseline = "top";
    c.fillStyle = ink.label; c.fillText("BPM", 4, 12); c.fillText("KEY", 4, 102);
    const playable = items.map((it, k) => ({ it, k, land: f.landings[k] || {} }));
    // Selected slot behind everything.
    if (sel?.kind === "slot") {
      const k = items.findIndex((it) => it.item_id === sel.id);
      if (k >= 0) { c.fillStyle = ink.sel; c.fillRect(layout.x[k] * pps, 0, layout.w[k] * pps, FLOW_H); }
    }
    const bp = playable.map((p) => p.land.bpm).filter(Boolean);
    const lo = Math.min(...bp, 120) - 2, hi = Math.max(...bp, 128) + 2;
    const yB = (b) => 86 - ((b - lo) / (hi - lo)) * 72;
    [lo + 2, (lo + hi) / 2, hi - 2].forEach((b) => {
      c.fillStyle = ink.label; c.fillText(String(Math.round(b)), W - 26, yB(b) - 5);
      c.fillStyle = ink.grid; c.fillRect(30, yB(b), W - 60, 1);
    });
    // The tempo line breaks at a break or a gap.
    c.strokeStyle = ink.fg; c.lineWidth = 1.5; c.beginPath();
    let pen = false;
    playable.forEach(({ k, land }) => {
      if (!land.bpm) { pen = false; return; }
      const x = centre(k), y = yB(land.bpm);
      if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y);
    });
    c.stroke();
    playable.forEach(({ k, land }) => {
      if (!land.bpm) return;
      c.fillStyle = ink.fg; c.beginPath(); c.arc(centre(k), yB(land.bpm), 2.5, 0, 6.3); c.fill();
    });
    const yK = (code) => {
      const p = parseCamelot(code);
      return p ? 104 + ((p.num - 1) / 11) * 64 + (p.letter === "B" ? 6 : 0) : null;
    };
    for (let n = 1; n <= 12; n += 1) {
      const y = 104 + ((n - 1) / 11) * 64;
      c.fillStyle = ink.grid; c.fillRect(30, y + 3, W - 60, 1);
      if (n % 3 === 1) { c.fillStyle = ink.label; c.fillText(String(n), W - 18, y - 2); }
    }
    let prev = null;
    playable.forEach(({ k, land }) => {
      if (!land.camelot) { prev = null; return; }
      const x = centre(k), y = yK(land.camelot);
      const t = f.transitions[k - 1];
      if (prev && t) {
        const jump = t.key_ok === false;
        c.strokeStyle = jump ? ink.warn : ink.key; c.globalAlpha = jump ? 1 : 0.6;
        c.lineWidth = jump ? 2 : 1.2; c.beginPath(); c.moveTo(prev.x, prev.y); c.lineTo(x, y); c.stroke();
        c.globalAlpha = 1;
      }
      prev = { x, y };
    });
    playable.forEach(({ k, land }) => {
      if (!land.camelot) return;
      c.fillStyle = camelotColor(land.camelot); c.beginPath(); c.arc(centre(k), yK(land.camelot), 4.5, 0, 6.3); c.fill();
    });
    // Each slot's own energy, a tick beside it.
    c.fillStyle = ink.warn; c.globalAlpha = 0.55;
    items.forEach((it, k) => { if (it.energy != null) c.fillRect(centre(k) - 5, yE(it.energy) - 1, 10, 2); });
    c.globalAlpha = 1;
    if (curve?.length) {
      c.setLineDash([5, 4]); c.strokeStyle = ink.warn; c.lineWidth = 1.5; c.beginPath();
      const pts = curve.length === 1 ? [{ t: 0, e: curve[0].e }, { t: 1, e: curve[0].e }] : curve;
      pts.forEach((p, i) => { const x = p.t * S, y = yE(p.e); if (i === 0) c.moveTo(x, y); else c.lineTo(x, y); });
      c.stroke(); c.setLineDash([]);
      curve.forEach((p) => {
        c.fillStyle = ink.bg; c.strokeStyle = ink.warn; c.lineWidth = 2;
        c.beginPath(); c.arc(p.t * S, yE(p.e), 5, 0, 6.3); c.fill(); c.stroke();
      });
    }
  });

  const at = (e) => {
    const r = ref.current.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const hit = (x, y) => (curve || []).findIndex((p) => Math.hypot(p.t * S - x, yE(p.e) - y) < 9);
  const clamp = (v) => Math.max(0, Math.min(1, v));

  const onDown = (e) => {
    const { x, y } = at(e);
    const h = hit(x, y);
    if (h >= 0) {
      e.preventDefault();
      drag.current = { idx: h, curve: [...curve] };
      const move = (ev) => {
        const r = ref.current.getBoundingClientRect();
        const next = [...drag.current.curve];
        next[drag.current.idx] = { t: clamp((ev.clientX - r.left) / S), e: clamp(1 - (ev.clientY - r.top - 10) / 170) };
        drag.current.curve = next;
        onCurve(next);
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        const done = [...drag.current.curve].sort((a, b) => a.t - b.t);
        drag.current = null;
        onCommit(done);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      return;
    }
    if (e.shiftKey) {
      const next = [...(curve || []), { t: clamp(x / S), e: clamp(1 - (y - 10) / 170) }].sort((a, b) => a.t - b.t);
      onCommit(next);
    }
  };
  const onDbl = (e) => {
    const { x, y } = at(e);
    const h = hit(x, y);
    if (h >= 0) onCommit(curve.filter((_, j) => j !== h));
  };
  const onHover = (e) => {
    if (drag.current) return;
    const { x, y } = at(e);
    e.currentTarget.style.cursor = hit(x, y) >= 0 ? "grab" : "crosshair";
  };
  return (
    <canvas ref={ref} className="ss-flow" style={{ width: W, height: FLOW_H }}
      aria-label="Tempo, key and energy across the set — drag the energy points; shift-click adds one, double-click removes it"
      onPointerDown={onDown} onDoubleClick={onDbl} onPointerMove={onHover} />
  );
}

// What each rule a move or candidate breaks reads as (matcher/setflow).
const VIOLATION_TEXT = { key: "key", bpm: "tempo", repeat: "record repeats" };

// Studio lays a set on two A/B deck rows, many clips each, decoding only the
// clips near the playhead — so a whole set opens (82 lanes for a 41-pair set
// before decks, 2026-10-15). The cap only bounds a runaway set.
export const STUDIO_CHAIN_MAX = 64;

// How long a slot plays, in bars at its landing tempo — the set's clock, the
// cue sheet and the energy curve all run on it (setflow.DEFAULT_PLAY_BARS).
const PLAY_BARS = [16, 32, 64];
function PlayLength({ item, land, onSave }) {
  const bars = item.play_bars || null;
  const shown = land?.play_bars || bars;
  return (
    <div className="ss-moverow" title="How long this mashup plays in the set — the vocal section is what is layered, not how long it stays on">
      <span className="caps">Plays</span>
      <div className="d-seg">
        {PLAY_BARS.map((n) => (
          <button key={n} className={shown === n ? "on" : ""}
            onClick={() => onSave({ play_bars: n === 32 && !bars ? null : n })}>{n} bars</button>
        ))}
      </div>
      {land?.secs ? <span className="mono muted small">{mmss(land.secs)}</span> : null}
    </div>
  );
}

// What could come next: good pairs that are also an easy move from where the
// set lands now (GET /api/sets/{id}/next → matcher/setflow.next_candidates).
function NextUp({ setId, version, onAdd, onPlay, playingKey }) {
  const [rows, setRows] = useState(null);
  useEffect(() => {
    let live = true;
    api.getSetNext(setId, 6).then((d) => { if (live) setRows(d.candidates || []); })
      .catch(() => { if (live) setRows([]); });
    return () => { live = false; };
  }, [setId, version]);
  if (!rows || !rows.length) return null;
  return (
    <div className="ss-sec">
      <span className="caps">What comes next</span>
      <span className="muted small">
        {rows.some((c) => !c.next_relaxed)
          ? "Good on their own and an easy move from the last landing (±1 key step, ±4% tempo)."
          : "Nothing fits every rule from here — the closest, with what each breaks."}</span>
      {rows.map((c) => (
        <div key={keyOf(c)} className="set-next-row ss-cand">
          <PairLines item={c} />
          <div className="ss-why">
            <span className={`ss-chip ${c.next_grade === "smooth" ? "ok" : c.next_grade === "jump" ? "warn" : ""}`}>{c.next_why}</span>
            <span className="ss-chip mono">{Math.round((c.score_percentile ?? 0) * 100)}%</span>
            {c.next_relaxed && (
              <span className="ss-chip warn" title="Outside the set's fit rule">
                ⚠ {(c.next_violations || []).map((v) => VIOLATION_TEXT[v] || v).join(" · ")}</span>
            )}
            <span className="grow" />
            <button className="ui-btn" aria-label="Loop" onClick={() => onPlay(c)}>{playingKey === keyOf(c) ? "◍" : "▶"}</button>
            <button className="ui-btn" onClick={() => onAdd(c)}>+ add</button>
          </div>
        </div>
      ))}
    </div>
  );
}

function PairLines({ item, spans = false }) {
  return (
    <div className="ss-lines">
      <div className="ss-line vox"><span className="tag mono">VOX</span>
        <span className="t">{item.vocal_title}{item.vocal_artist ? ` · ${item.vocal_artist}` : ""}</span>
        {spans && <span className="mono muted">{spanLabel(item.vocal_section_label, item.vocal_section_start, item.vocal_section_end)}</span>}</div>
      <div className="ss-line bed"><span className="tag mono">BED</span>
        <span className="t">{item.inst_title}{item.inst_artist ? ` · ${item.inst_artist}` : ""}</span>
        {spans && <span className="mono muted">{spanLabel(item.inst_section_label, item.inst_section_start, item.inst_section_end)}</span>}</div>
    </div>
  );
}

function FitRows({ rows }) {
  if (!rows.length) return null;
  return (
    <div className="ss-sec">
      <span className="caps">Fit with neighbours</span>
      {rows.map((r) => (
        <div key={r.side} className="ss-fit">
          <span className="muted">{r.side}</span>
          <span className="t">{r.name}</span>
          <span className={`mono ${r.tone}`}>{r.txt}</span>
        </div>
      ))}
    </div>
  );
}

// The gap's suggestions (GET /api/sets/{id}/suggest).
function GapFill({ setId, position, version, onFill, onPlay, playingKey }) {
  const [out, setOut] = useState(null);
  useEffect(() => {
    let live = true;
    setOut(null);
    api.getSetSuggest(setId, position).then((d) => { if (live) setOut(d); })
      .catch((e) => { if (live) setOut({ candidates: [], error: e.message }); });
    return () => { live = false; };
  }, [setId, position, version]);
  if (!out) return <span className="muted small">Finding pairs that fit both sides…</span>;
  return (
    <div className="ss-sec">
      <div className="ss-sechead">
        <span className="caps">Suggestions</span>
        <span className="muted small">{out.relaxed ? "nothing fits every rule — closest, rule breaks marked"
          : "key ±1 of both neighbours · tempo ±4% · nearest the energy curve"}</span>
      </div>
      {out.error && <span className="error-text small">{out.error}</span>}
      {!out.candidates.length && !out.error && (
        <span className="muted small">Nothing in the library fits both neighbours within ±1 key step and ±4% tempo.
          Try moving the gap, or loosen the energy curve.</span>
      )}
      {out.candidates.map((c) => {
        const keyBad = c.violations.includes("key"), bpmBad = c.violations.includes("bpm");
        const prev = c.fit?.from_prev, next = c.fit?.into_next;
        const path = `${prev?.from_key || "·"}→${c.vocal_camelot || "?"}→${next?.to_key || "·"}`;
        const tempo = prev?.tempo_pct ?? next?.tempo_pct;
        return (
          <div key={keyOf(c)} className="ss-cand">
            <PairLines item={c} />
            <div className="ss-why">
              <span className={`ss-chip mono${keyBad ? " warn" : ""}`}
                style={keyBad ? undefined : { background: camelotColor(c.vocal_camelot), color: "var(--key-ink)" }}>
                {path}{keyBad ? " ⚠" : ""}</span>
              <span className={`ss-chip mono${bpmBad ? " warn" : ""}`}>{pct(tempo)}{bpmBad ? " ⚠" : ""}</span>
              {c.energy != null && c.energy_target != null && (
                <span className="ss-chip mono energy" title="This pair's energy against the curve here">
                  E {c.energy.toFixed(2)}/{c.energy_target.toFixed(2)}</span>
              )}
              <span className="ss-chip mono">{Math.round((c.score_percentile ?? 0) * 100)}%</span>
            {c.next_relaxed && (
              <span className="ss-chip warn" title="Outside the set's fit rule">
                ⚠ {(c.next_violations || []).map((v) => VIOLATION_TEXT[v] || v).join(" · ")}</span>
            )}
              <span className="grow" />
              <button className="ui-btn" aria-label="Loop" onClick={() => onPlay(c)}>{playingKey === keyOf(c) ? "◍" : "▶"}</button>
              <button className="btn-primary" onClick={() => onFill(c)}>Fill</button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function BreakEditor({ item, onSave }) {
  const [label, setLabel] = useState(item.label || "");
  const [secs, setSecs] = useState(item.duration_secs || 30);
  useEffect(() => { setLabel(item.label || ""); setSecs(item.duration_secs || 30); }, [item.item_id, item.label, item.duration_secs]);
  return (
    <div className="ss-sec">
      <input className="mx-in" value={label} aria-label="Break label"
        placeholder="e.g. talk to the crowd · air horn · drop to silence"
        onChange={(e) => setLabel(e.target.value)}
        onBlur={() => { if (label !== (item.label || "")) onSave({ label }); }}
        onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
      <div className="ss-fit"><span>Length</span><span className="grow" /><span className="mono">{mmss(secs)}</span></div>
      <input type="range" min={5} max={240} step={5} value={secs} aria-label="Break length"
        onChange={(e) => setSecs(Number(e.target.value))}
        onPointerUp={() => { if (secs !== item.duration_secs) onSave({ duration_secs: secs }); }}
        onKeyUp={() => { if (secs !== item.duration_secs) onSave({ duration_secs: secs }); }} />
    </div>
  );
}

function MoveEditor({ t, onMove }) {
  const mv = t.move;
  if (!mv) return null;
  return (
    <div className="ss-sec set-move" title={mv.text}>
      <span className="caps">The move</span>
      <div className="ss-moverow">
        <select className="mx-select" value={mv.type} aria-label="Move"
          onChange={(e) => onMove({ type: e.target.value, bars: null })}>
          {MOVES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
        {(mv.type === "blend" || mv.type === "bed_swap" || mv.type === "vocal_swap") && (
          <select className="mx-select" value={mv.bars} aria-label="Bars"
            onChange={(e) => onMove({ type: mv.type, bars: Number(e.target.value) })}>
            {[...new Set([...BAR_CHOICES, mv.bars])].sort((a, b) => a - b)
              .map((b) => <option key={b} value={b}>{b} bars</option>)}
          </select>
        )}
        {mv.suggested
          ? <span className="muted small">suggested</span>
          : <button className="ui-link" title={`Back to the suggestion: ${mv.suggestion?.type?.replace("_", " ")}`}
              onClick={() => onMove(null)}>reset</button>}
      </div>
      <span className="muted small">{mv.text}</span>
    </div>
  );
}

function PairNote({ item, onNote }) {
  const [note, setNote] = useState(item.note || "");
  useEffect(() => { setNote(item.note || ""); }, [item.item_id, item.note]);
  return (
    <input className="mx-in" value={note} placeholder="A note — ‘opener’, ‘needs a riser’…" aria-label="Note"
      title="A note on this pair. Never training data."
      onChange={(e) => setNote(e.target.value)}
      onBlur={() => { if (note !== (item.note || "")) onNote(item, note); }}
      onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
  );
}

export function SetScreen({ sets, player, onOpenStudio, onOpenChain, onRailSlot, onOpenLibrary,
                            onOpenTrack, dockPair = null, selectedTrack = null }) {
  const active = sets.active;
  const [data, setData] = useState(null);
  const [name, setName] = useState("");
  const [sel, setSel] = useState(null);       // {kind:'slot', id} | {kind:'trans', i} | null
  const [curve, setCurve] = useState(null);   // the energy curve while it is dragged
  const [showExport, setShowExport] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [flJob, setFlJob] = useState(null);
  const { job: flState } = useJobPolling(flJob?.job_id);
  const [base, setBase] = useState(() => {
    try { return localStorage.getItem(BASE_KEY) || ""; } catch { return ""; }
  });
  const [pps, setPps] = useState(() => {
    try { return Number(localStorage.getItem(ZOOM_KEY)) || 3; } catch { return 3; }
  });
  const scrollRef = useRef(null);

  // Sets keep their furniture in the main column; the rail slot stays empty.
  useEffect(() => { onRailSlot?.(null); }, [onRailSlot]);
  useEffect(() => { try { localStorage.setItem(ZOOM_KEY, String(pps)); } catch { /* ignore */ } }, [pps]);

  const load = useCallback(async () => {
    if (!active) { setData(null); return; }
    try {
      const d = await api.getSet(active.id);
      setData(d); setName(d.name);
    } catch (e) { toast(`Could not load the set: ${e.message}`); }
  }, [active]);
  useEffect(() => { load(); }, [load, active?.item_count]);
  useEffect(() => { setCurve(data?.energy_plan || null); }, [data?.energy_plan]);

  // FL sessions: a job; the zip downloads when it lands.
  useEffect(() => {
    if (!flState || !flJob) return;
    if (flState.status === "completed") {
      window.location.href = flJob.archive_url;
      toast(`${flJob.pair_count} FL session${flJob.pair_count === 1 ? "" : "s"} exported`
        + (flJob.skipped ? ` · ${flJob.skipped} over the limit left out` : ""));
      setFlJob(null);
    } else if (flState.status === "failed") {
      toast(`FL export failed: ${flState.error || "unknown error"}`);
      setFlJob(null);
    }
  }, [flState]);   // eslint-disable-line react-hooks/exhaustive-deps

  const { create, setActiveId } = sets;
  const apply = (d) => { setData(d); sets.refresh(); };
  const items = data?.items || [];
  const f = data?.flow || { landings: [], transitions: [], starts: [], total_secs: 0, grades: {}, fits: {} };

  // Layout time: each slot as wide as it lasts, laid end to end (overlaps are
  // in the running time the header prints, not in the picture).
  const layout = useMemo(() => {
    const w = items.map((it, k) => {
      const s = f.landings[k]?.secs;
      return s && s > 0 ? s : (it.duration_secs || 30);
    });
    const x = [];
    let t = 0;
    w.forEach((d) => { x.push(t); t += d; });
    x.push(t);
    // `width` is the drawing width (never narrower than 600px); `span` is how
    // far the set's own time reaches in it — the energy curve's t runs over it.
    return { x, w, total: t, width: Math.max(600, Math.ceil(t * pps)), span: Math.max(1, t * pps) };
  }, [items, f.landings, pps]);

  const selIdx = sel?.kind === "slot" ? items.findIndex((it) => it.item_id === sel.id) : -1;
  const selItem = selIdx >= 0 ? items[selIdx] : null;
  const after = selIdx >= 0 ? selIdx + 1 : items.length;

  const newSet = async () => {
    try { const s = await create("New set"); toast(`Made “${s.name}” — name it in the header`); }
    catch (e) { toast(e.message); }
  };

  const rename = async () => {
    if (!data || !name.trim() || name === data.name) return;
    try { apply(await api.updateSet(data.id, { name })); } catch (e) { toast(e.message); }
  };

  const insert = async (body, label) => {
    try {
      const d = await api.addSetSlot(data.id, body);
      apply(d);
      if (d.added?.duplicate) { toast("That pair is already in the set"); return; }
      setSel({ kind: "slot", id: d.added.id });
      if (label) toast(label);
    } catch (e) { toast(`Could not add: ${e.message}`); }
  };

  const addPair = async () => {
    let c = dockPair;
    if (!c && items.length) {
      try { c = (await api.getSetNext(data.id, 1)).candidates?.[0] || null; } catch { c = null; }
    }
    if (!c) { toast("Pick a pair in the Library’s pair dock first — + Pair adds the one under its cursor"); return; }
    insert({ kind: "pair", vocal_song_id: c.vocal_song_id, inst_song_id: c.inst_song_id,
             vocal_section: c.vocal_section_idx ?? null, inst_section: c.inst_section_idx ?? null,
             position: after }, `Added ${c.vocal_title} over ${c.inst_title}`);
  };
  const addTrack = () => {
    if (!selectedTrack) { toast("Select a track in the Library first — + Track adds it whole"); return; }
    insert({ kind: "track", song_id: selectedTrack.id, position: after }, `Added ${selectedTrack.title}`);
  };
  const addBreak = () => insert({ kind: "break", position: after });
  const addGap = (pos = after) => insert({ kind: "gap", position: pos });

  // Remove now; Undo puts the slot back where it was (its chosen move too).
  const remove = async (item) => {
    const k = items.findIndex((it) => it.item_id === item.item_id);
    try {
      apply(await api.removeFromSet(data.id, item.item_id));
      setSel(null);
      const setId = data.id;
      toast(`Removed slot ${k + 1}`, {
        label: "Undo",
        onClick: async () => {
          try {
            const d = await api.addSetSlot(setId, slotBody(item, k));
            if (item.transition && d.added?.id) {
              apply(await api.setItemTransition(setId, d.added.id, item.transition));
            } else apply(d);
          } catch (e) { toast(`Undo failed: ${e.message}`); }
        },
      });
    } catch (e) { toast(e.message); }
  };

  const reorderTo = async (ids) => {
    setData({ ...data, items: ids.map((id) => items.find((i) => i.item_id === id)) });
    try { apply(await api.reorderSet(data.id, ids)); } catch (e) { toast(e.message); load(); }
  };
  const moveSlot = (fromId, toIndex) => {
    const ids = items.map((i) => i.item_id).filter((id) => id !== fromId);
    ids.splice(toIndex, 0, fromId);
    reorderTo(ids);
  };
  const swap = (i) => {
    const ids = items.map((it) => it.item_id);
    [ids[i], ids[i + 1]] = [ids[i + 1], ids[i]];
    reorderTo(ids);
  };

  const autoOrder = async () => {
    try {
      const s = await api.suggestSetOrder(data.id, items.find((i) => (i.kind || "pair") !== "break" && (i.kind || "pair") !== "gap")?.item_id);
      const before = f.fits || {}, nxt = s.flow.fits || {};
      apply(await api.reorderSet(data.id, s.item_ids));
      toast(`Re-ordered: ${nxt.ok || 0} fitting moves, ${(nxt.warn || 0) + (nxt.bad || 0)} rough `
        + `(was ${before.ok || 0} / ${(before.warn || 0) + (before.bad || 0)})`);
    } catch (e) { toast(`Could not re-order: ${e.message}`); }
  };

  const saveNote = async (item, note) => {
    try { await api.savePairNote(item, note); toast(note ? "Note saved" : "Note removed"); load(); }
    catch (e) { toast(`Could not save the note: ${e.message}`); }
  };
  const saveTempo = async (plan) => {
    try {
      apply(await api.updateSet(data.id, plan ? { tempo_plan: plan } : { clear_tempo: true }));
      toast(plan ? `Tempo curve ${plan.start_bpm}${plan.end_bpm ? ` → ${plan.end_bpm}` : ""} BPM`
        : "Tempo curve off — each slot lands at its own tempo");
    } catch (e) { toast(`Could not set the tempo: ${e.message}`); }
  };
  const saveCurve = async (pts) => {
    setCurve(pts);
    try { apply(await api.updateSet(data.id, { energy_plan: pts })); }
    catch (e) { toast(`Could not save the energy curve: ${e.message}`); }
  };
  const setMove = async (item, move) => {
    try { apply(await api.setItemTransition(data.id, item.item_id, move)); }
    catch (e) { toast(`Could not change the move: ${e.message}`); }
  };
  // Every pair in Studio, each at its own landing tempo (the curve's point, or
  // the vocal's), from the selected slot on — at most STUDIO_CHAIN_MAX at once.
  const openChain = () => {
    const from = Math.max(0, selIdx);
    const pairs = items.map((it, k) => ({ it, k }))
      .filter(({ it, k }) => (it.kind || "pair") === "pair" && k >= from)
      .map(({ it, k }) => ({ ...it, set_bpm: it.set_bpm ?? f.landings[k]?.bpm ?? null,
                             // The move into this slot overlaps the previous one.
                             overlap_secs: k > from ? (f.transitions[k - 1]?.overlap_secs ?? 0) : 0 }));
    const all = items.filter((it) => (it.kind || "pair") === "pair").length;
    const chain = pairs.slice(0, STUDIO_CHAIN_MAX);
    if (!chain.length) return;
    if (chain.length < all) {
      toast(`Opening ${chain.length} of ${all} mashups — Studio arranges up to ${STUDIO_CHAIN_MAX} at a time; `
        + "select a later slot to open from there");
    }
    onOpenChain(chain);
  };
  const editSlot = async (item, patch) => {
    try { apply(await api.updateSetItem(data.id, item.item_id, patch)); }
    catch (e) { toast(e.message); }
  };
  const fill = async (gap, c) => {
    const k = items.findIndex((it) => it.item_id === gap.item_id);
    try {
      const d = await api.addSetSlot(data.id, {
        kind: "pair", vocal_song_id: c.vocal_song_id, inst_song_id: c.inst_song_id,
        vocal_section: c.vocal_section_idx ?? null, inst_section: c.inst_section_idx ?? null, position: k,
      });
      if (d.added?.duplicate) { toast("That pair is already in the set"); return; }
      apply(await api.removeFromSet(data.id, gap.item_id));
      setSel({ kind: "slot", id: d.added.id });
      toast(`Filled slot ${k + 1} with ${c.vocal_title} over ${c.inst_title}`);
    } catch (e) { toast(`Could not fill: ${e.message}`); }
  };

  const playPair = (item) => player.toggle({
    kind: "pair", key: keyOf(item), candidate: item,
    title: item.vocal_title, subtitle: `over ${item.inst_title}`,
  });
  const playTrack = (item) => player.toggle({
    kind: "track", songId: item.song_id, stem: "full", title: item.title, subtitle: item.artist,
    duration: item.song_duration,
  });
  const previewTransition = (i) => {
    const a = items[i], b = items[i + 1];
    if (!a || !b) return;
    if (f.transitions[i]?.fit === "none") { toast("A break or a gap sits on this boundary — nothing to blend"); return; }
    toast("Rendering the last 8 bars into the first 8…");
    player.toggle({
      kind: "track", url: api.setTransitionAudioUrl(data.id, b.item_id, 8), songId: null,
      title: `Transition ${i + 1} → ${i + 2}`, subtitle: `${slotText(a).name} into ${slotText(b).name}`,
    });
  };
  const playingKey = player.kind === "pair" ? player.source?.key : null;

  const exportAs = async (format) => {
    setShowExport(false);
    if (format === "fl") {
      try { setFlJob(await api.exportSetFl(data.id)); toast("Rendering FL sessions — the zip downloads when it is ready"); }
      catch (e) { toast(`FL export: ${e.message}`); }
      return;
    }
    const usesBase = ["rekordbox", "serato", "traktor"].includes(format);
    if (usesBase) {
      try { localStorage.setItem(BASE_KEY, base); } catch { /* ignore */ }
    }
    window.location.href = api.setExportUrl(data.id, format, usesBase ? base : "");
  };

  const del = async () => {
    if (!confirmDelete) { setConfirmDelete(true); setTimeout(() => setConfirmDelete(false), 4000); return; }
    await api.deleteSet(data.id);
    setConfirmDelete(false);
    sets.setActiveId(null);
    await sets.refresh();
    toast(`Deleted the set — its pairs stay scored and rated`);
  };

  // ←/→ select · alt+←/→ move · t preview · g gap · del remove · esc deselect
  const onKey = (e) => {
    if (!items.length) return;
    const i = selIdx;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const d = e.key === "ArrowRight" ? 1 : -1;
      if (e.altKey && i >= 0) {
        const j = i + d;
        if (j < 0 || j >= items.length) return;
        const ids = items.map((it) => it.item_id);
        [ids[i], ids[j]] = [ids[j], ids[i]];
        reorderTo(ids);
        return;
      }
      const base0 = i >= 0 ? i : sel?.kind === "trans" ? (d > 0 ? sel.i : sel.i + 1) : -1;
      const n = Math.max(0, Math.min(items.length - 1, base0 + d));
      setSel({ kind: "slot", id: items[n].item_id });
      const sc = scrollRef.current;
      if (sc) {
        const x = PAD + layout.x[n] * pps;
        if (x < sc.scrollLeft || x > sc.scrollLeft + sc.clientWidth - 80) sc.scrollLeft = x - 80;
      }
    } else if (e.key === "t" || e.key === "T") {
      if (sel?.kind === "trans") previewTransition(sel.i);
      else if (i >= 0 && items[i + 1]) previewTransition(i);
    } else if ((e.key === "g" || e.key === "G") && i >= 0) {
      addGap(i + 1);
    } else if ((e.key === "Delete" || e.key === "Backspace") && selItem) {
      e.preventDefault();
      remove(selItem);
    } else if (e.key === "Escape" && sel) {
      e.stopPropagation();
      setSel(null);
    }
  };

  if (!active) {
    return (
      <div className="set-screen">
        <ScreenHeader title="Sets" />
        <div className="ss-empty">
          <b>Plan a mix from the pairs you like</b>
          <span className="muted">A set is a running order: pairs, whole tracks, breaks and gaps. Add a
            pair from the Library’s pair dock with <b>+ Set</b> (or press <span className="kbd">a</span>),
            or from Studio. Here you order them, see whether each move fits, shape the energy, fill the
            gaps and export the lot.</span>
          <div className="ss-row">
            <button className="btn-primary" onClick={newSet}>＋ New set</button>
            <button className="ui-btn" onClick={onOpenLibrary}>Go to the pair dock</button>
          </div>
        </div>
      </div>
    );
  }
  // Never a blank page while a set loads: the header and its name show at once.
  if (!data || data.id !== active.id) {
    return (
      <div className="set-screen">
        <ScreenHeader title="Set" sub={`${active.item_count ?? 0} slots`} />
        <div className="ss-empty muted">Loading “{active.name}”…</div>
      </div>
    );
  }

  const bpms = f.landings.map((l) => l.bpm).filter(Boolean);
  const gaps = items.filter((it) => it.kind === "gap").length;
  const rough = f.transitions.map((t, i) => ({ t, i })).filter(({ t }) => t.fit === "warn" || t.fit === "bad");
  const sub = `${items.length} slot${items.length === 1 ? "" : "s"} · ${hmmss(f.total_secs)}`
    + (bpms.length ? ` · ${Math.round(bpms[0])}→${Math.round(bpms[bpms.length - 1])}` : "")
    + (gaps ? ` · ${gaps} gap${gaps > 1 ? "s" : ""}` : "");
  const flBusy = flJob && (!flState || flState.status === "queued" || flState.status === "running");

  // ── the inspector ───────────────────────────────────────────────────────────
  let inspector;
  if (selItem) {
    const k = selIdx;
    const kind = selItem.kind || "pair";
    const land = f.landings[k] || {};
    const into = f.transitions[k], from = f.transitions[k - 1];
    const fitRow = (side, t, other) => (t && t.fit !== "none" ? {
      side, name: `${other + 1}. ${slotText(items[other]).name}`,
      txt: `${t.from_key || "?"}→${t.to_key || "?"} · ${pct(t.tempo_pct)}`,
      tone: t.fit === "ok" ? "ok" : t.fit === "unknown" ? "muted" : "warn",
    } : null);
    const fits = [fitRow("from", from, k - 1), fitRow("into", into, k + 1)].filter(Boolean);
    inspector = (
      <>
        <div className="ss-ihead"><span className="caps">Slot {k + 1} · {kind}</span><span className="grow" />
          <button className="ui-link" aria-label="Close" onClick={() => setSel(null)}>✕</button></div>
        <div className="ss-ibody">
          {kind === "pair" && (
            <>
              <PairLines item={selItem} spans />
              <span className="mono muted small">
                lands {land.camelot || "?"} · {land.bpm ? land.bpm.toFixed(1) : "?"} BPM · {mmss(layout.w[k])}
                {selItem.score_percentile != null ? ` · ${Math.round(selItem.score_percentile * 100)}%` : ""}
                {selItem.energy != null ? ` · energy ${selItem.energy.toFixed(2)}` : ""}
                {selItem.stale ? " · stale (no longer scored)" : ""}
              </span>
              <PlayLength item={selItem} land={land} onSave={(patch) => editSlot(selItem, patch)} />
              <div className="set-recipe"><RecipeStrip recipe={selItem.recipe} compact /></div>
              <PairNote item={selItem} onNote={saveNote} />
              <FitRows rows={fits} />
              <div className="ss-acts">
                <button className="ui-btn" onClick={() => playPair(selItem)}>{playingKey === keyOf(selItem) ? "◍ Stop" : "▶ Loop"}</button>
                <button className="btn-primary" onClick={() => onOpenStudio(selItem)}>Open in Studio</button>
                <button className="ui-btn" onClick={() => addGap(k + 1)}>Insert gap after</button>
                <button className="ui-btn danger" onClick={() => remove(selItem)}>Remove</button>
              </div>
            </>
          )}
          {kind === "track" && (
            <>
              <span className="ss-ttl">{selItem.title} · {selItem.artist}</span>
              <span className="mono muted small">{selItem.camelot || "—"} · {selItem.bpm ? selItem.bpm.toFixed(1) : "—"} BPM · {selItem.play_bars ? `plays ${selItem.play_bars} bars` : "plays whole"}
                {selItem.energy != null ? ` · energy ${selItem.energy.toFixed(2)}` : ""}</span>
              <FitRows rows={fits} />
              <div className="ss-acts">
                <button className="ui-btn" onClick={() => playTrack(selItem)}>▶ Preview</button>
                {onOpenTrack && <button className="ui-btn" onClick={() => onOpenTrack(selItem.song_id)}>Open track</button>}
                <button className="ui-btn" onClick={() => addGap(k + 1)}>Insert gap after</button>
                <button className="ui-btn danger" onClick={() => remove(selItem)}>Remove</button>
              </div>
            </>
          )}
          {kind === "break" && (
            <>
              <span className="ss-ttl">{selItem.label || "Break"}</span>
              <span className="muted small">A pause or a note in the running order.</span>
              <BreakEditor item={selItem} onSave={(patch) => editSlot(selItem, patch)} />
              <div className="ss-acts"><button className="ui-btn danger" onClick={() => remove(selItem)}>Remove</button></div>
            </>
          )}
          {kind === "gap" && (
            <>
              <span className="ss-ttl warn">Find a pair for this slot</span>
              <GapFill setId={data.id} position={k}
                version={`${items.map((it) => it.item_id).join(",")}:${JSON.stringify(data.energy_plan)}:${JSON.stringify(data.tempo_plan)}`}
                onFill={(c) => fill(selItem, c)} onPlay={playPair} playingKey={playingKey} />
              <div className="ss-acts"><button className="ui-btn danger" onClick={() => remove(selItem)}>Remove gap</button></div>
            </>
          )}
        </div>
      </>
    );
  } else if (sel?.kind === "trans" && f.transitions[sel.i]) {
    const i = sel.i;
    const t = f.transitions[i];
    const none = t.fit === "none";
    inspector = (
      <>
        <div className="ss-ihead"><span className="caps">Transition {i + 1} → {i + 2}</span><span className="grow" />
          <button className="ui-link" aria-label="Close" onClick={() => setSel(null)}>✕</button></div>
        <div className="ss-ibody">
          <span className="ss-ttl">{slotText(items[i]).name}</span>
          <span className="ss-ttl"><span className="mono muted">INTO </span>{slotText(items[i + 1]).name}</span>
          {none ? <span className="muted small">A break or a gap — no key or tempo to match.</span> : (
            <>
              <FitRows rows={[
                { side: "Key", name: t.key_ok ? "within ±1 step" : t.key_ok === false ? "jumps — consider a bridge" : "unmeasured",
                  txt: `${t.from_key || "?"}→${t.to_key || "?"}`, tone: t.key_ok ? "ok" : t.key_ok === false ? "warn" : "muted" },
                { side: "Tempo", name: t.bpm_ok ? "within ±4%" : t.bpm_ok === false ? "a big tempo change" : "unmeasured",
                  txt: pct(t.tempo_pct), tone: t.bpm_ok ? "ok" : t.bpm_ok === false ? "warn" : "muted" },
              ]} />
              <MoveEditor t={t} onMove={(move) => setMove(items[i + 1], move)} />
            </>
          )}
          <div className="ss-acts">
            {!none && (
              <button className="btn-primary" onClick={() => previewTransition(i)}
                title="The last 8 bars of this slot into the first 8 of the next (t)">▶ Preview transition</button>
            )}
            <button className="ui-btn" onClick={() => addGap(i + 1)}>Insert gap here</button>
            <button className="ui-btn" onClick={() => swap(i)}>Swap the two</button>
          </div>
        </div>
      </>
    );
  } else {
    const path = items.map((it, k) => ({ k, code: f.landings[k]?.camelot, jump: f.transitions[k - 1]?.key_ok === false }))
      .filter((p) => p.code);
    inspector = (
      <>
        <div className="ss-ihead"><span className="caps">Set</span></div>
        <div className="ss-ibody">
          <span className="ss-ttl">{data.name}</span>
          <span className="mono muted small">{sub}</span>
          {path.length > 0 && (
            <div className="ss-sec">
              <span className="caps">Key path</span>
              <div className="ss-path">
                {path.map((p) => (
                  <span key={p.k} className={p.jump ? "jump" : ""} title={`${p.k + 1}. ${slotText(items[p.k]).name}`}>
                    <KeyChip camelot={p.code} /></span>
                ))}
              </div>
            </div>
          )}
          <div className="ss-sec">
            <span className="caps">Needs attention</span>
            {rough.length === 0
              ? <span className="ok small">✓ Every transition is within ±1 key step and ±4% tempo.</span>
              : rough.map(({ t, i }) => (
                <button key={i} className={`ss-warn ${t.fit}`} onClick={() => setSel({ kind: "trans", i })}>
                  <span>⚠</span>
                  <span className="t">{i + 1}→{i + 2}: {t.key_ok === false ? `key jumps ${t.key_steps} steps (${t.from_key}→${t.to_key})` : ""}
                    {t.key_ok === false && t.bpm_ok === false ? " · " : ""}{t.bpm_ok === false ? `tempo ${pct(t.tempo_pct)}` : ""}
                    {(t.key_ok === false || t.bpm_ok === false) && t.violations?.includes("repeat") ? " · " : ""}
                    {t.violations?.includes("repeat") ? "a record comes back too soon" : ""}</span>
                </button>
              ))}
          </div>
          {items.some((it) => (it.kind || "pair") === "pair") && (
            <NextUp setId={data.id}
              version={`${items.length}:${items.at(-1)?.item_id}:${JSON.stringify(data.tempo_plan)}`}
              onAdd={(c) => insert({ kind: "pair", vocal_song_id: c.vocal_song_id, inst_song_id: c.inst_song_id,
                vocal_section: c.vocal_section_idx ?? null, inst_section: c.inst_section_idx ?? null }, "Added to the end of the set")}
              onPlay={playPair} playingKey={playingKey} />
          )}
          <div className="ss-acts ss-danger">
            <button className={`ui-btn danger${confirmDelete ? " on" : ""}`} onClick={del}
              title="Delete this set (click twice)">{confirmDelete ? "Click again to delete" : "Delete"}</button>
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="set-screen ss">
      <div className="ss-main">
        <header className="ss-head">
          <input className="ss-name" value={name} aria-label="Set name" size={Math.max(5, name.length)}
            onChange={(e) => setName(e.target.value)} onBlur={rename}
            onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
          <select className="mx-select ss-pick" aria-label="Active set" value={active.id}
            title="One active set: the dock’s + Set and Studio add to it"
            onChange={(e) => (e.target.value === "new" ? newSet() : setActiveId(Number(e.target.value)))}>
            {sets.sets.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.item_count})</option>)}
            <option value="new">＋ New set…</option>
          </select>
          <span className="mono muted small">{sub}</span>
          {rough.length > 0 && (
            <button className="ss-rough" onClick={() => setSel(null)}
              title="Moves outside ±1 key step or ±4% tempo — listed in the inspector">
              ⚠ {rough.length} rough</button>
          )}
          <span className="grow" />
          <button className="ui-btn" onClick={addPair} title="The pair under the dock cursor (or the best next one), after the selected slot">+ Pair</button>
          <button className="ui-btn" onClick={addTrack} title="The selected Library track, played whole">+ Track</button>
          <button className="ui-btn" onClick={addBreak} title="A note or a pause in the running order">+ Break</button>
          <button className="ui-btn" onClick={() => addGap()} title="An empty slot the engine suggests pairs for (g)">+ Gap</button>
          <span className="ss-exportwrap">
            <button className="ui-btn" disabled={!items.length || flBusy} aria-expanded={showExport}
              onClick={() => setShowExport((v) => !v)}>
              {flBusy ? `Exporting ${flState?.progress ? `${Math.round(flState.progress)}%` : "…"}` : "Export ▾"}</button>
            {showExport && (
              <>
                <div className="ss-scrim" onClick={() => setShowExport(false)} />
                <div className="ss-menu">
                  <button onClick={() => exportAs("cue")}><b>Cue sheet</b><span>Titles, moves and start times, for a recorded mix</span></button>
                  <div className="ss-menurb">
                    <button onClick={() => exportAs("rekordbox")}><b>rekordbox XML</b><span>A playlist in this order, with beat grids and hot cues</span></button>
                    <input className="mx-in" value={base} onChange={(e) => setBase(e.target.value)}
                      placeholder="audio folder as rekordbox sees it (Docker: the host’s ./data/audio)"
                      title="Leave blank when rekordbox runs on the machine the library is on" />
                  </div>
                  <button onClick={() => exportAs("serato")}><b>Serato crate</b><span>The same files in order — drop into _Serato_/Subcrates</span></button>
                  <button onClick={() => exportAs("traktor")}><b>Traktor NML</b><span>A playlist with tempo, key and hot cues — Import Collection</span></button>
                  <button onClick={() => exportAs("fl")}><b>FL Studio sessions</b><span>One session folder per pair, aligned like Studio</span></button>
                  <button onClick={() => exportAs("csv")}><b>CSV</b><span>Every slot with its key, tempo, move and note</span></button>
                </div>
              </>
            )}
          </span>
        </header>

        <div className="ss-toolbar small">
          <div className="ss-zoom">
            <button className="ui-btn" aria-label="Zoom out" onClick={() => setPps((p) => Math.max(0.1, p / 1.4))}>−</button>
            <button className="ui-btn" aria-label="Zoom in" onClick={() => setPps((p) => Math.min(20, p * 1.4))}>+</button>
            <button className="ui-btn" onClick={() => {
              const w = (scrollRef.current?.clientWidth || 900) - PAD * 2 - 8;
              setPps(Math.max(0.08, w / (layout.total || 1)));
            }}>Fit</button>
          </div>
          <span className="ss-legend"><i className="bpm" />BPM</span>
          <span className="ss-legend" title="Each slot's landing key, at its position on the Camelot wheel"><i className="key" />Key</span>
          <span className="ss-legend" title="Drag its points · shift-click adds one · double-click removes one"><i className="energy" />Energy</span>
          <TempoControl plan={data.tempo_plan} onSave={saveTempo} />
          <div className="d-seg" role="group" aria-label="Energy preset">
            {PRESETS.map(([label, pts]) => (
              <button key={label} onClick={() => saveCurve(pts)}>{label}</button>
            ))}
          </div>
          <span className="grow" />
          <button className="ui-btn" onClick={autoOrder} disabled={items.length < 3}
            title="Re-order to keep each move small, from the first slot; breaks and gaps keep their places">⇅ Auto-order</button>
          <button className="ui-btn" disabled={!items.some((it) => (it.kind || "pair") === "pair")}
            onClick={openChain}
            title="Every pair on one Studio timeline, back to back, each at its point on the tempo curve">◫ Studio</button>
        </div>

        <div className="ss-scroll" ref={scrollRef} tabIndex={0} onKeyDown={onKey} aria-label="Set timeline — arrow keys select">
          <div className="ss-canvaswrap" style={{ width: layout.width + PAD * 2 }}>
            <SetTimeline items={data.items} f={f} layout={layout} pps={pps} sel={sel}
              onSelect={setSel} onMove={moveSlot} />
            {items.length === 0 && (
              <div className="ss-empty inline muted">
                The set is empty. Press <b>a</b> on a pair in the dock, <b>+ Set</b> in Studio, or add a pair,
                a track, a break or a gap above.
              </div>
            )}
            <FlowGraph items={items} f={f} layout={layout} pps={pps} curve={curve} sel={sel}
              onCurve={setCurve} onCommit={saveCurve} />
          </div>
        </div>
        <div className="ss-keys mono">←/→ select · alt+←/→ move slot · t preview transition · g insert gap after · del remove · drag slots to reorder · energy: drag a point, shift-click adds, double-click removes</div>
      </div>

      <aside className="ss-inspector" aria-label="Set inspector">{inspector}</aside>
    </div>
  );
}
