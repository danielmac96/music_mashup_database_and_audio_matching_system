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
  const [jobs, setJobs] = useState({}); // trackId -> { kind, jobId }
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

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getTracks();
      setTracks(data.tracks);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, [refreshKey]);

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

  useEffect(() => {
    onStatus?.({ text: `${filtered.length} of ${tracks.length} tracks · ${readyCount} ready` });
  }, [filtered.length, tracks.length, readyCount, onStatus]);

  const cycle = (arr, cur, setter) => setter(arr[(arr.indexOf(cur) + 1) % arr.length]);

  // ── run-step availability (preserves existing pipeline gating) ─────────
  const gating = (t) => {
    const job = jobs[t.id];
    const analysed = isAnalysed(t);
    const hasStructure = (t.section_count || 0) > 0;
    return {
      job,
      canDownload: !job && (t.status === "queued" || t.status?.startsWith("error")),
      canSeparate: !job && t.stems.full && (!t.stems.vocals || !t.stems.instrumental),
      // Already-analysed / already-structured tracks stay disabled — the data is
      // in the DB, so re-running is just wasted work (re-run via Edit if needed).
      canAnalyze: !job && t.stems.full && !analysed,
      canStructure: !job && t.stems.full && !hasStructure,
      analysed,
      hasStructure,
    };
  };

  const RunActions = ({ t }) => {
    const g = gating(t);
    if (g.job) return <JobBadge jobId={g.job.jobId} onComplete={() => onJobDone(t.id)} />;
    return (
      <div className="mini-actions">
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
                      {t.metadata_partial && (
                        <span className="badge metadata-partial" title="Full metadata couldn't be fetched.">
                          partial
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="metrics-row">
                  <div className="bpm-chip"><span className="u">BPM </span>{f.bpm != null ? f.bpm.toFixed(1) : "—"}</div>
                  <div className="key-chip" style={{ background: camelotColor(f.camelot) }}>
                    {f.camelot || "—"}
                  </div>
                  <div className="energy-wrap">
                    <div className="l">Energy</div>
                    <div className="energy-bar"><span style={{ width: `${Math.round((f.energy || 0) * 100)}%` }} /></div>
                  </div>
                </div>

                <div className="pipeline" style={{ justifyContent: "space-between" }}>
                  <PipelineDots track={t} runningKind={g.job?.kind} />
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
