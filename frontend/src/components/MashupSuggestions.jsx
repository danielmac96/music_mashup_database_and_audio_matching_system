import { Fragment, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import {
  artGradient, bpmTag, camelotColor, fmtTime, keyRel, pct, tierFor,
} from "../theme";
import { toast } from "../toast";

const MIN_MATCHES = [50, 65, 75, 85];

function PlanDetails({ vocalId, instId }) {
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api.getMashupPlan(vocalId, instId)
      .then((p) => !cancelled && setPlan(p))
      .catch((e) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, [vocalId, instId]);

  if (error) return <div className="plan-detail error-text">{error}</div>;
  if (!plan) return <div className="plan-detail muted">Loading plan…</div>;

  return (
    <div className="plan-detail">
      <strong>Recipe</strong>
      <ol>
        {plan.steps.map((s, i) => <li key={i}>{s.replace(/^\d+\.\s*/, "")}</li>)}
      </ol>
      {plan.pairings?.length > 0 && (
        <div className="mono-grid" style={{ gridTemplateColumns: "1.2fr 1fr 1.2fr 1fr 1fr" }}>
          <div className="muted">VOCAL SECTION</div>
          <div className="muted">TIME</div>
          <div className="muted">BED SECTION</div>
          <div className="muted">TIME</div>
          <div className="muted">DURATION FIT</div>
          {plan.pairings.map((p, i) => (
            <Fragment key={i}>
              <div>{p.vocal_label}</div>
              <div>{fmtTime(p.vocal_start)}–{fmtTime(p.vocal_end)}</div>
              <div>{p.inst_label}</div>
              <div>{fmtTime(p.inst_start)}–{fmtTime(p.inst_end)}</div>
              <div>{p.vocal_duration}s / {p.inst_duration_stretched}s</div>
            </Fragment>
          ))}
        </div>
      )}
      <div className="muted mono" style={{ marginTop: 8, fontSize: 11.5 }}>
        {plan.key_relation}
        {plan.stretch_factor ? ` · stretch bed ×${plan.stretch_factor}` : ""}
        {plan.semitone_shift != null
          ? ` · pitch bed ${plan.semitone_shift >= 0 ? "+" : ""}${plan.semitone_shift} st`
          : ""}
      </div>
    </div>
  );
}

export function MashupSuggestions({ seed, onClearSeed, onAudition, onStatus }) {
  const [candidates, setCandidates] = useState([]);
  const [comboType, setComboType] = useState("vocal_over_instrumental");
  const [minMatch, setMinMatch] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scoreJobId, setScoreJobId] = useState(null);
  const [expanded, setExpanded] = useState(null);

  const refresh = async (type = comboType, activeSeed = seed, min = minMatch) => {
    setLoading(true);
    setError(null);
    try {
      const opts = { comboType: type, minScore: min / 100, limit: 50 };
      if (activeSeed?.songId != null) {
        if (activeSeed.role === "instrumental") opts.instSongId = activeSeed.songId;
        else opts.vocalSongId = activeSeed.songId;
      }
      const data = await api.getMashups(opts);
      setCandidates(data.candidates);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh(comboType, seed, minMatch);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed]);

  useEffect(() => {
    onStatus?.({ text: `${candidates.length} scored pair${candidates.length === 1 ? "" : "s"}` });
  }, [candidates.length, onStatus]);

  const seedTitle = useMemo(() => {
    if (seed?.songId == null) return null;
    const c = candidates[0];
    if (!c) return `#${seed.songId}`;
    return seed.role === "instrumental" ? c.inst_title : c.vocal_title;
  }, [seed, candidates]);

  const startScoring = async () => {
    try {
      const { job_id } = await api.startScoring();
      setScoreJobId(job_id);
      toast("Scoring library…");
    } catch (e) {
      setError(e.message);
    }
  };

  const switchType = (type) => { setComboType(type); setExpanded(null); refresh(type, seed, minMatch); };
  const cycleMin = () => {
    const next = MIN_MATCHES[(MIN_MATCHES.indexOf(minMatch) + 1) % MIN_MATCHES.length];
    setMinMatch(next);
    refresh(comboType, seed, next);
  };

  return (
    <div className="page">
      <div className="screen-head">
        <h1>Mashups</h1>
        <span className="sub">
          {loading ? "loading…" : `Ranked best-first${seedTitle ? " · seed fixed" : ""}`}
        </span>
      </div>

      <div className="toolbar">
        <div className="seg">
          <button className={comboType === "vocal_over_instrumental" ? "active" : ""}
            onClick={() => switchType("vocal_over_instrumental")}>Vocal / Instrumental</button>
          <button className={comboType === "instrumental_over_instrumental" ? "active" : ""}
            onClick={() => switchType("instrumental_over_instrumental")}>Instr. / Instr.</button>
        </div>
        <div className="chip" onClick={cycleMin}>
          <span className="k">Min match</span><span className="mono">{minMatch}%</span><span className="caret">▾</span>
        </div>
        {seedTitle && (
          <div className="chip seed">
            Seeded: <b>{seedTitle}</b> as {seed.role === "instrumental" ? "bed" : "vocal"}
            <span className="x" onClick={() => { onClearSeed?.(); refresh(comboType, null, minMatch); }}>✕</span>
          </div>
        )}
        <div className="spacer" />
        {scoreJobId ? (
          <JobBadge jobId={scoreJobId} onComplete={() => { setScoreJobId(null); refresh(); }} />
        ) : (
          <button className="btn" onClick={startScoring}>↻ Score library</button>
        )}
      </div>

      <div className="legend">
        <span>Sub-scores:</span>
        <span className="sw"><i style={{ background: "var(--cyan)" }} />BPM</span>
        <span className="sw"><i style={{ background: "var(--violet)" }} />Key</span>
        <span className="sw"><i style={{ background: "var(--amber)" }} />Energy</span>
        <span className="sw"><i style={{ background: "var(--green)" }} />Timbre</span>
        <span className="weights">Weighted: Key 30 · BPM 25 · Timbre 25 · Energy 20</span>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}

      {candidates.length === 0 && !loading ? (
        <p className="empty">
          {minMatch > MIN_MATCHES[0]
            ? `No pairs score ${minMatch}% or better — click "Min match" to lower the bar, or re-score after analyzing more tracks.`
            : "No scored pairs yet. Analyze your tracks (Library tab), then click “Score library”."}
        </p>
      ) : (
        <div className="pair-list">
          {candidates.map((c, i) => {
            const total = Math.round((c.score_total || 0) * 100);
            const { tier, color, textColor } = tierFor(total);
            const kr = keyRel(c.vocal_camelot, c.inst_camelot);
            const w = (v) => `${Math.round((v || 0) * 100)}%`;
            const isVI = comboType === "vocal_over_instrumental";
            return (
              <Fragment key={c.id}>
                <div className={`pair${i === 0 ? " top" : ""}`}>
                  <div className="pair-rank">{i + 1}</div>
                  <div className="pair-side">
                    <div className="pair-art" style={{ background: artGradient(c.vocal_song_id) }}>♪</div>
                    <div style={{ minWidth: 0 }}>
                      <div className="pair-role vocal">{isVI ? "VOCAL (TOP)" : "INSTR (TOP)"}</div>
                      <div className="pair-title" title={c.vocal_title}>{c.vocal_title}</div>
                      <div className="pair-meta">{c.vocal_artist || "—"} · {c.vocal_bpm?.toFixed(1)} · {c.vocal_camelot || "?"}</div>
                    </div>
                  </div>
                  <div className="score-cluster">
                    <div className="score-hero">
                      <span className="pctv" style={{ color }}>{total}%</span>
                      <span className="tier-badge" style={{ color: textColor, background: color }}>{tier}</span>
                    </div>
                    <div className="subscores">
                      <div className="cell"><span style={{ width: w(c.score_bpm), background: "var(--cyan)" }} /></div>
                      <div className="cell"><span style={{ width: w(c.score_key), background: "var(--violet)" }} /></div>
                      <div className="cell"><span style={{ width: w(c.score_energy), background: "var(--amber)" }} /></div>
                      <div className="cell"><span style={{ width: w(c.score_timbre), background: "var(--green)" }} /></div>
                    </div>
                    <div className="relation-chips">
                      <span className="rel-chip" style={{ color: kr.tagColor, background: kr.tagBg }}>{kr.tag}</span>
                      <span className="rel-chip bpm">{bpmTag(c.vocal_bpm, c.inst_bpm)}</span>
                    </div>
                  </div>
                  <div className="pair-side bed">
                    <div style={{ minWidth: 0 }}>
                      <div className="pair-role bed">BED (INST)</div>
                      <div className="pair-title" title={c.inst_title}>{c.inst_title}</div>
                      <div className="pair-meta">{c.inst_artist || "—"} · {c.inst_bpm?.toFixed(1)} · {c.inst_camelot || "?"}</div>
                    </div>
                    <div className="pair-art" style={{ background: artGradient(c.inst_song_id + 3) }}>♪</div>
                  </div>
                  <div className="pair-actions">
                    <button className="plan" onClick={() => setExpanded(expanded === c.id ? null : c.id)}>
                      {expanded === c.id ? "Hide ▴" : "Plan ▾"}
                    </button>
                    {isVI && (
                      <button className="audition" onClick={() => onAudition?.(c.vocal_song_id, c.inst_song_id)}>
                        ▶ Audition
                      </button>
                    )}
                  </div>
                </div>
                {expanded === c.id && (
                  <PlanDetails vocalId={c.vocal_song_id} instId={c.inst_song_id} />
                )}
              </Fragment>
            );
          })}
        </div>
      )}
    </div>
  );
}
