import { Fragment, useEffect, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";

function fmtTs(secs) {
  const s = Math.round(secs || 0);
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function pct(v) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}

function PlanDetails({ vocalId, instId }) {
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getMashupPlan(vocalId, instId)
      .then((p) => !cancelled && setPlan(p))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [vocalId, instId]);

  if (error) return <div className="error-text">{error}</div>;
  if (!plan) return <div className="muted">Loading plan…</div>;

  return (
    <div style={{ fontSize: "0.8rem", padding: "8px 0" }}>
      <div>
        <strong>Recipe</strong>
        <ol style={{ margin: "4px 0 8px 18px", padding: 0 }}>
          {plan.steps.map((s, i) => (
            <li key={i}>{s.replace(/^\d+\.\s*/, "")}</li>
          ))}
        </ol>
      </div>

      {plan.pairings.length > 0 && (
        <div>
          <strong>Section pairings</strong>
          <table style={{ marginTop: 4 }}>
            <thead>
              <tr>
                <th>Vocal section</th>
                <th>Timestamps</th>
                <th>Instrumental section</th>
                <th>Timestamps</th>
                <th>Duration fit</th>
              </tr>
            </thead>
            <tbody>
              {plan.pairings.map((p, i) => (
                <tr key={i}>
                  <td>{p.vocal_label}</td>
                  <td>
                    {fmtTs(p.vocal_start)}–{fmtTs(p.vocal_end)}
                  </td>
                  <td>{p.inst_label}</td>
                  <td>
                    {fmtTs(p.inst_start)}–{fmtTs(p.inst_end)}
                  </td>
                  <td>
                    {p.vocal_duration}s vs {p.inst_duration_stretched}s
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="muted" style={{ marginTop: 6 }}>
        Key relation: {plan.key_relation}
        {plan.stretch_factor
          ? ` · stretch instrumental ×${plan.stretch_factor}`
          : ""}
        {plan.semitone_shift != null
          ? ` · pitch instrumental ${plan.semitone_shift >= 0 ? "+" : ""}${plan.semitone_shift} st`
          : ""}
      </div>
      {(plan.files.vocals || plan.files.instrumental) && (
        <div className="muted" style={{ marginTop: 4, wordBreak: "break-all" }}>
          {plan.files.vocals && <div>Vocal stem: {plan.files.vocals}</div>}
          {plan.files.instrumental && (
            <div>Inst stem: {plan.files.instrumental}</div>
          )}
        </div>
      )}
    </div>
  );
}

export function MashupSuggestions({ seed, onClearSeed, onAudition }) {
  const [candidates, setCandidates] = useState([]);
  const [comboType, setComboType] = useState("vocal_over_instrumental");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scoreJobId, setScoreJobId] = useState(null);
  const [expanded, setExpanded] = useState(null); // candidate id

  const refresh = async (type = comboType, activeSeed = seed) => {
    setLoading(true);
    setError(null);
    try {
      const opts = { comboType: type, limit: 50 };
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

  // Refetch whenever a directed-search seed arrives from the Library tab.
  useEffect(() => {
    refresh(comboType, seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed]);

  const startScoring = async () => {
    try {
      const { job_id } = await api.startScoring();
      setScoreJobId(job_id);
    } catch (e) {
      setError(e.message);
    }
  };

  const switchType = (type) => {
    setComboType(type);
    setExpanded(null);
    refresh(type);
  };

  return (
    <div className="panel">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h2 style={{ margin: 0 }}>Mashup Suggestions</h2>
        <div className="actions">
          {scoreJobId ? (
            <JobBadge
              jobId={scoreJobId}
              onComplete={() => {
                setScoreJobId(null);
                refresh();
              }}
            />
          ) : (
            <button onClick={startScoring}>Score library</button>
          )}
          <button className="secondary" onClick={() => refresh()} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      <div className="tabs" style={{ marginTop: 8 }}>
        <button
          className={comboType === "vocal_over_instrumental" ? "active" : ""}
          onClick={() => switchType("vocal_over_instrumental")}
        >
          Vocal / Instrumental
        </button>
        <button
          className={
            comboType === "instrumental_over_instrumental" ? "active" : ""
          }
          onClick={() => switchType("instrumental_over_instrumental")}
        >
          Instrumental / Instrumental
        </button>
      </div>

      {seed?.songId != null && (
        <div className="muted" style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
          Showing matches where track #{seed.songId} is the{" "}
          {seed.role === "instrumental" ? "instrumental" : "vocal"}.
          <button className="secondary" onClick={() => { onClearSeed?.(); refresh(comboType, null); }}>
            Clear filter
          </button>
        </div>
      )}

      {error && (
        <div className="error-text" style={{ marginTop: 8 }}>
          {error}
        </div>
      )}

      {candidates.length === 0 && !loading ? (
        <p className="muted">
          No scored pairs yet. Analyze your tracks (Library tab), then click
          “Score library”.
        </p>
      ) : (
        <table style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>#</th>
              <th>Top (vocals)</th>
              <th>Bed (instrumental)</th>
              <th>Score</th>
              <th>BPM / Key / Energy / Timbre</th>
              <th>Genre · Year · Popularity</th>
              <th>Plan</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c, i) => (
              <Fragment key={c.id}>
                <tr>
                  <td>{i + 1}</td>
                  <td>
                    <div>{c.vocal_title}</div>
                    <div className="muted" style={{ fontSize: "0.75rem" }}>
                      {c.vocal_artist || "—"} · {c.vocal_bpm?.toFixed(1)} BPM ·{" "}
                      {c.vocal_camelot}
                    </div>
                  </td>
                  <td>
                    <div>{c.inst_title}</div>
                    <div className="muted" style={{ fontSize: "0.75rem" }}>
                      {c.inst_artist || "—"} · {c.inst_bpm?.toFixed(1)} BPM ·{" "}
                      {c.inst_camelot}
                    </div>
                  </td>
                  <td>
                    <strong>{c.score_total?.toFixed(3)}</strong>
                  </td>
                  <td style={{ fontSize: "0.75rem" }}>
                    {c.score_bpm?.toFixed(2)} / {c.score_key?.toFixed(2)} /{" "}
                    {c.score_energy?.toFixed(2)} / {c.score_timbre?.toFixed(2)}
                  </td>
                  <td style={{ fontSize: "0.75rem" }}>
                    <div>
                      {c.vocal_genre || "?"} · {c.vocal_year || "?"} ·{" "}
                      {pct(c.vocal_popularity)}
                    </div>
                    <div className="muted">
                      {c.inst_genre || "?"} · {c.inst_year || "?"} ·{" "}
                      {pct(c.inst_popularity)}
                    </div>
                  </td>
                  <td>
                    <div className="actions">
                      <button
                        className="secondary"
                        onClick={() =>
                          setExpanded(expanded === c.id ? null : c.id)
                        }
                        title={
                          c.vocal_section_count && c.inst_section_count
                            ? "Section-level mashup plan"
                            : "Plan available — analyze both tracks for section timestamps"
                        }
                      >
                        {expanded === c.id ? "Hide" : "Plan"}
                      </button>
                      {comboType === "vocal_over_instrumental" && (
                        <button
                          onClick={() => onAudition?.(c.vocal_song_id, c.inst_song_id)}
                          title="Render and hear this mashup in the Audition tab"
                        >
                          Audition
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
                {expanded === c.id && (
                  <tr>
                    <td colSpan={7}>
                      <PlanDetails
                        vocalId={c.vocal_song_id}
                        instId={c.inst_song_id}
                      />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
