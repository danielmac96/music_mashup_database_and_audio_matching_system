import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import {
  artGradient, camelotColor, fmtDur, isAnalysed, pipelineDots, statusMeta,
} from "../theme";
import { toast } from "../toast";

const KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const GENRES = ["All", "Pop", "Hip Hop", "Rap", "EDM"];
const SORTS = ["Popularity", "BPM", "Title", "Energy"];

// Auto-chain pipeline stage → the "kind" the PipelineDots component lights up.
const STAGE_KIND = {
  download: "download", stems: "separate",
  analyze: "analyze", structure: "structure",
};
const JOB_ACTIVE = new Set(["queued", "running"]);

// A detected tempo outside this range is very often a half/double-time octave
// error — the single most common way auto-analysis silently poisons matches.
// (bpm_confidence from librosa is a beats/frame density, not a 0-1 quality, so
// an out-of-range BPM is the honest signal to flag for a manual sanity-check.)
const bpmLooksOff = (f) => f?.bpm != null && (f.bpm < 80 || f.bpm > 170);

function FeatureEditor({ track, onSaved, onCancel }) {
  const feats = track.features?.full || {};
  const [bpm, setBpm] = useState(feats.bpm != null ? String(feats.bpm) : "");
  const [key, setKey] = useState(feats.key || "C");
  const [mode, setMode] = useState(feats.mode || "major");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const payload = { key, mode };
      const bpmNum = parseFloat(bpm);
      if (!Number.isNaN(bpmNum) && bpmNum > 0) payload.bpm = bpmNum;
      await api.correctFeatures(track.id, payload);
      toast("Features corrected — re-score to update matches");
      onSaved();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="feat-edit">
      <label>
        <span className="muted" style={{ width: 34 }}>BPM</span>
        <input type="number" step="0.1" min="1" value={bpm}
          onChange={(e) => setBpm(e.target.value)} style={{ width: 72 }} />
      </label>
      <label>
        <span className="muted" style={{ width: 34 }}>Key</span>
        <select value={key} onChange={(e) => setKey(e.target.value)}>
          {KEY_NAMES.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="major">major</option>
          <option value="minor">minor</option>
        </select>
      </label>
      {error && <div className="error-text">{error}</div>}
      <div className="mini-actions">
        <button className="mini-btn" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>
        <button className="mini-btn" onClick={onCancel} disabled={saving}>Cancel</button>
      </div>
    </div>
  );
}

function PipelineDots({ track, runningKind }) {
  const p = pipelineDots(track, runningKind);
  return (
    <div className="pipeline">
      <span className="dot" style={{ color: p.dl }}>●</span>DL
      <span className="dot" style={{ color: p.stems }}>●</span>Stems
      <span className="dot" style={{ color: p.analyse }}>●</span>Analyse
      <span className="dot" style={{ color: p.structure }}>●</span>Structure
    </div>
  );
}

const STAGE_LABEL = {
  download: "Downloading", stems: "Separating stems",
  analyze: "Analysing", structure: "Detecting structure",
};

// Live progress for a track being auto-processed by the pipeline queue.
function PipelineProgress({ job }) {
  const running = job.status === "running";
  const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
  const label = running ? (STAGE_LABEL[job.stage] || "Processing") : "Queued";
  return (
    <div className="pipe-progress">
      <div className="row" style={{ justifyContent: "space-between", fontSize: 12 }}>
        <span>{label}{running ? "…" : ""}</span>
        {running && <span className="mono faint">{pct}%</span>}
      </div>
      <div className="progress-bar" aria-label={`progress ${pct}%`}>
        <div className="fill" style={{ width: `${running ? pct : 0}%` }} />
      </div>
      {job.message && (
        <div className="muted" style={{ fontSize: 11, marginTop: 3 }}>{job.message}</div>
      )}
    </div>
  );
}

function StatusTag({ status }) {
  const m = statusMeta(status);
  return (
    <span
      className={`status-tag${m.pulse ? " pulse" : ""}`}
      style={{ color: m.color, background: m.bg, border: `1px solid ${m.border}` }}
    >
      {m.tag}
    </span>
  );
}

function metaLine(t) {
  const genre = t.genre || "—";
  const year = t.release_year || "—";
  const plays = t.plays ? `${t.plays.toLocaleString()} ▶` : "0 ▶";
  return `${genre} · ${year} · ${plays}`;
}

export function TrackList({ refreshKey, onSendToAudition, onFindMatches, onStatus }) {
  const [tracks, setTracks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [jobs, setJobs] = useState({}); // trackId -> { kind, jobId }  (manual single-stage)
  const [pipeJobs, setPipeJobs] = useState([]); // live auto-chain pipeline jobs
  const [editing, setEditing] = useState(null);

  // filters
  const [search, setSearch] = useState("");
  const [genre, setGenre] = useState("All");
  const [readyOnly, setReadyOnly] = useState(false);
  const [sort, setSort] = useState("Popularity");
  const [view, setView] = useState("cards");

  // preview playback (full mix), single shared audio element
  const audioRef = useRef(null);
  const [playingId, setPlayingId] = useState(null);

  const refresh = async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const data = await api.getTracks();
      setTracks(data.tracks);
    } catch (e) {
      setError(e.message);
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, [refreshKey]);

  // Poll the auto-chain pipeline jobs so imported tracks show live DL→Stems→
  // Analyse→Structure progress without the user clicking anything. While work
  // is active we also silently re-pull the track list so statuses/dots advance.
  const prevActiveRef = useRef(false);
  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      try {
        const { jobs: pj } = await api.getJobs({ kind: "pipeline" });
        if (cancelled) return;
        setPipeJobs(pj);
        const anyActive = pj.some((j) => JOB_ACTIVE.has(j.status));
        // Refresh while work runs, and once more on the falling edge so the
        // final 'analysed' state / dots land after the last job completes.
        if (anyActive || prevActiveRef.current) await refresh(true);
        prevActiveRef.current = anyActive;
        timer = setTimeout(poll, anyActive ? 1500 : 5000);
      } catch {
        if (!cancelled) timer = setTimeout(poll, 5000);
      }
    };
    poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  useEffect(() => () => { audioRef.current?.pause(); }, []);

  const startJob = async (id, kind, fn) => {
    try {
      const { job_id } = await fn(id);
      setJobs((prev) => ({ ...prev, [id]: { kind, jobId: job_id } }));
    } catch (e) {
      setError(e.message);
    }
  };

  const onJobDone = (id) => {
    setJobs((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });
    refresh();
  };

  const togglePreview = (t) => {
    if (playingId === t.id) {
      audioRef.current?.pause();
      setPlayingId(null);
      return;
    }
    audioRef.current?.pause();
    const a = new Audio(api.audioUrl(t.id, "full"));
    a.onended = () => setPlayingId(null);
    a.play().then(() => { audioRef.current = a; setPlayingId(t.id); }).catch(() => {
      toast("Couldn't play preview — is the track downloaded?");
    });
  };

  // ── filter + sort ──────────────────────────────────────────────────────
  const filtered = useMemo(() => {
    let out = tracks.filter((x) => {
      if (readyOnly && !isAnalysed(x)) return false;
      if (genre !== "All" && (x.genre || "").toLowerCase() !== genre.toLowerCase()) return false;
      if (search) {
        const q = search.toLowerCase();
        if (!`${x.title} ${x.artist || ""} ${x.genre || ""}`.toLowerCase().includes(q)) return false;
      }
      return true;
    });
    const bpmOf = (x) => x.features?.full?.bpm ?? Infinity;
    const enOf = (x) => x.features?.full?.energy ?? -1;
    if (sort === "Popularity") out = [...out].sort((a, b) => (b.plays || 0) - (a.plays || 0));
    else if (sort === "BPM") out = [...out].sort((a, b) => bpmOf(a) - bpmOf(b));
    else if (sort === "Title") out = [...out].sort((a, b) => a.title.localeCompare(b.title));
    else if (sort === "Energy") out = [...out].sort((a, b) => enOf(b) - enOf(a));
    return out;
  }, [tracks, readyOnly, genre, search, sort]);

  const readyCount = useMemo(() => tracks.filter(isAnalysed).length, [tracks]);

  // Most-recent pipeline job per song (pipeJobs is newest-first) + batch tally.
  const pipeBySong = useMemo(() => {
    const m = {};
    for (const j of pipeJobs) {
      if (j.song_id != null && !(j.song_id in m)) m[j.song_id] = j;
    }
    return m;
  }, [pipeJobs]);

  const batch = useMemo(() => {
    const active = pipeJobs.filter((j) => JOB_ACTIVE.has(j.status));
    return {
      active: active.length,
      running: active.filter((j) => j.status === "running").length,
      queued: active.filter((j) => j.status === "queued").length,
    };
  }, [pipeJobs]);

  useEffect(() => {
    if (batch.active > 0) {
      onStatus?.({
        locked: true,
        text: `Processing ${batch.active} track${batch.active === 1 ? "" : "s"} · ${batch.running} running · ${batch.queued} queued`,
      });
    } else {
      onStatus?.({ text: `${filtered.length} of ${tracks.length} tracks · ${readyCount} ready` });
    }
  }, [batch, filtered.length, tracks.length, readyCount, onStatus]);

  const cycle = (arr, cur, setter) => setter(arr[(arr.indexOf(cur) + 1) % arr.length]);

  // ── run-step availability (preserves existing pipeline gating) ─────────
  const gating = (t) => {
    const job = jobs[t.id];
    const pipe = pipeBySong[t.id];
    const pipelining = !!pipe && JOB_ACTIVE.has(pipe.status);
    const analysed = isAnalysed(t);
    const hasStructure = (t.section_count || 0) > 0;
    const busy = !!job || pipelining;
    return {
      job, pipe, pipelining,
      canDownload: !busy && (t.status === "queued" || t.status?.startsWith("error")),
      canSeparate: !busy && t.stems.full && (!t.stems.vocals || !t.stems.instrumental),
      // Already-analysed / already-structured tracks stay disabled — the data is
      // in the DB, so re-running is just wasted work (re-run via Edit if needed).
      canAnalyze: !busy && t.stems.full && !analysed,
      canStructure: !busy && t.stems.full && !hasStructure,
      canRetry: !busy && !!t.status?.startsWith("error"),
      // A downloaded track still ~30s long is likely a SoundCloud Go+ preview.
      canReverify: !busy && t.stems.full && t.duration_secs > 0 && t.duration_secs <= 40,
      analysed,
      hasStructure,
    };
  };

  const retryTrack = async (id) => {
    try {
      await api.processTrack(id);
      toast("Re-processing from the failed stage…");
      refresh(true);
    } catch (e) {
      setError(e.message);
    }
  };

  const RunActions = ({ t }) => {
    const g = gating(t);
    if (g.pipelining) return <PipelineProgress job={g.pipe} />;
    if (g.job) return <JobBadge jobId={g.job.jobId} onComplete={() => onJobDone(t.id)} />;
    return (
      <div className="mini-actions">
        {g.canRetry && t.last_error && (
          <div className="track-error" style={{ flexBasis: "100%" }} title={t.last_error}>
            ⚠ {t.last_error}
          </div>
        )}
        {g.canRetry && (
          <button className="mini-btn" title="Re-run the pipeline from where it failed"
            onClick={() => retryTrack(t.id)}>↻ Retry</button>
        )}
        <button className="mini-btn" disabled={!g.canDownload}
          onClick={() => startJob(t.id, "download", api.startDownload)}>Download</button>
        <button className="mini-btn" disabled={!g.canSeparate}
          onClick={() => startJob(t.id, "separate", api.startSeparate)}>Separate</button>
        <button className="mini-btn" disabled={!g.canAnalyze}
          title={g.analysed ? "Already analysed" : !t.stems.full ? "Download first" : "Extract tempo/key/waveform"}
          onClick={() => startJob(t.id, "analyze", api.startAnalyze)}>Analyze</button>
        <button className="mini-btn" disabled={!g.canStructure}
          title={g.hasStructure ? "Structure already detected" : !t.stems.full ? "Download first" : "Detect intro/verse/chorus/drop"}
          onClick={() => startJob(t.id, "structure", api.startStructure)}>Structure</button>
        {g.canReverify && (
          <button className="mini-btn" title="Looks like a 30s preview — re-download the full track"
            onClick={() => startJob(t.id, "reverify", api.reverifyTrack)}>⟳ Fix preview</button>
        )}
        <button className="mini-btn" disabled={!g.analysed}
          title={g.analysed ? "Find scored beds for this vocal" : "Analyze first"}
          onClick={() => onFindMatches?.(t.id, "vocal")}>Find beds</button>
        {g.analysed && (
          <button className="mini-btn" onClick={() => setEditing(editing === t.id ? null : t.id)}>
            {editing === t.id ? "Close" : "Edit"}
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="page">
      <div className="screen-head">
        <h1>Library</h1>
        <span className="sub">{loading ? "refreshing…" : `${readyCount} ready to mash`}</span>
      </div>

      {/* filter toolbar */}
      <div className="toolbar">
        <div className="search-box">
          <span className="ico">⌕</span>
          <input placeholder="Search title, artist, genre…" value={search}
            onChange={(e) => setSearch(e.target.value)} />
        </div>
        <div className="chip" onClick={() => cycle(GENRES, genre, setGenre)}>
          <span className="k">Genre</span><span>{genre}</span><span className="caret">▾</span>
        </div>
        <div className={`chip toggle${readyOnly ? " on" : ""}`} onClick={() => setReadyOnly((v) => !v)}>
          <span style={{ width: 9, height: 9, borderRadius: 3, background: "var(--green)", display: "inline-block" }} />
          Ready to mash
        </div>
        <div className="spacer" />
        <div className="chip" onClick={() => cycle(SORTS, sort, setSort)}>
          <span className="k">Sort</span><span>{sort}</span><span className="caret">▾</span>
        </div>
        <div className="seg">
          <button className={view === "cards" ? "active" : ""} onClick={() => setView("cards")}>▦ Cards</button>
          <button className={view === "table" ? "active" : ""} onClick={() => setView("table")}>≣ Table</button>
        </div>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}

      {filtered.length === 0 && !loading ? (
        <p className="empty">No tracks match. Import a SoundCloud URL on the Import tab.</p>
      ) : view === "cards" ? (
        <div className="card-grid">
          {filtered.map((t) => {
            const f = t.features?.full || {};
            const g = gating(t);
            return (
              <div key={t.id} className="card">
                <div className="card-top">
                  <div className="card-art" style={{ background: t.thumbnail ? `url(${t.thumbnail})` : artGradient(t.id) }}>
                    {t.thumbnail ? "" : "♪"}
                  </div>
                  <div className="card-id">
                    <div className="card-title" title={t.title}>{t.title}</div>
                    <div className="card-artist">{t.artist || "—"}</div>
                    <div className="card-statusrow">
                      <StatusTag status={t.status} />
                      <span className="card-dur">{t.duration_str || fmtDur(t.duration_secs)}</span>
                      {!!t.metadata_partial && (
                        <span className="badge metadata-partial" title="Full metadata couldn't be fetched.">
                          partial
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="metrics-row">
                  <div className="bpm-chip"><span className="u">BPM </span>{f.bpm != null ? f.bpm.toFixed(1) : "—"}</div>
                  {bpmLooksOff(f) && (
                    <span className="bpm-warn"
                      title={`Unusual tempo (${f.bpm.toFixed(1)} BPM) — often a half/double-time detection error. Verify with Edit; a wrong BPM skews every match.`}>
                      ⚠
                    </span>
                  )}
                  <div className="key-chip" style={{ background: camelotColor(f.camelot) }}>
                    {f.camelot || "—"}
                  </div>
                  <div className="energy-wrap">
                    <div className="l">Energy</div>
                    <div className="energy-bar"><span style={{ width: `${Math.round((f.energy || 0) * 100)}%` }} /></div>
                  </div>
                </div>

                <div className="pipeline" style={{ justifyContent: "space-between" }}>
                  <PipelineDots track={t} runningKind={g.job?.kind || (g.pipelining ? STAGE_KIND[g.pipe?.stage] : undefined)} />
                  <span className="faint">{metaLine(t)}</span>
                </div>

                <div className="card-actions">
                  <button className={`act${playingId === t.id ? " on" : ""}`} disabled={!t.stems.full}
                    onClick={() => togglePreview(t)}>
                    {playingId === t.id ? "❚❚ Playing" : "▶ Preview"}
                  </button>
                  <button className="act vocal" disabled={!g.analysed || !t.stems.vocals}
                    title={g.analysed && t.stems.vocals ? "Load into Audition as the vocal" : "Needs analysed vocal stem"}
                    onClick={() => onSendToAudition?.({ vocalId: t.id })}>♪ Vocal</button>
                  <button className="act bed" disabled={!g.analysed || !t.stems.instrumental}
                    title={g.analysed && t.stems.instrumental ? "Load into Audition as the bed" : "Needs analysed instrumental stem"}
                    onClick={() => onSendToAudition?.({ instId: t.id })}>♪ Bed</button>
                </div>

                <RunActions t={t} />
                {editing === t.id && (
                  <FeatureEditor track={t}
                    onSaved={() => { setEditing(null); refresh(); }}
                    onCancel={() => setEditing(null)} />
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="data-table">
          <div className="data-head" style={{ gridTemplateColumns: "40px 2fr 100px 70px 70px 1.4fr 1fr" }}>
            <div>ID</div><div>TITLE / ARTIST</div><div>STATUS</div><div>BPM</div>
            <div>KEY</div><div>PIPELINE</div><div style={{ textAlign: "right" }}>ACTIONS</div>
          </div>
          {filtered.map((t) => {
            const f = t.features?.full || {};
            const g = gating(t);
            return (
              <div key={t.id} className="data-row" style={{ gridTemplateColumns: "40px 2fr 100px 70px 70px 1.4fr 1fr" }}>
                <div className="mono faint">{t.id}</div>
                <div>
                  <div className="t">{t.title}</div>
                  <div className="a">{t.artist || "—"}</div>
                </div>
                <div><StatusTag status={t.status} /></div>
                <div className="mono" style={{ color: "var(--text-2)" }}>{f.bpm != null ? f.bpm.toFixed(1) : "—"}</div>
                <div>
                  <span className="key-chip" style={{ background: camelotColor(f.camelot), fontSize: 12, padding: "3px 7px" }}>
                    {f.camelot || "—"}
                  </span>
                </div>
                <div><PipelineDots track={t} runningKind={g.job?.kind} /></div>
                <div className="row-actions">
                  <button className="row-act vocal" disabled={!g.analysed || !t.stems.vocals}
                    onClick={() => onSendToAudition?.({ vocalId: t.id })}>Vocal</button>
                  <button className="row-act bed" disabled={!g.analysed || !t.stems.instrumental}
                    onClick={() => onSendToAudition?.({ instId: t.id })}>Bed</button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
