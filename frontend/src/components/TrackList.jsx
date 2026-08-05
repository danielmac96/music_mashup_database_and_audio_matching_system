import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { KeyChip } from "./KeyChip";
import { PlayerBar } from "./PlayerBar";
import { PlaylistImporter } from "./PlaylistImporter";
import { TrackArt } from "./TrackArt";
import { SourceBadge } from "./SourceBadge";
import {
  fmtDur, isAnalysed, pipelineDots, statusMeta,
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

// key_confidence (T1.3) is margin-over-runner-up x chroma peakiness, so it is
// low both when two keys are effectively tied and when there is no tonal centre
// to find. Calibrated against a re-analysed 90-stem library, where it runs
// p25=0.010, p50=0.023, p90=0.066, max=0.128 — real music simply does not
// produce confident key estimates very often. The threshold marks the worst
// ~quartile: flagging the median would put a ⚠ on three tracks in four and
// train the eye to ignore it. null = analysed before this existed, say nothing.
const KEY_CONFIDENCE_MIN = 0.012;
const keyLooksOff = (f) =>
  f?.key_confidence != null && f.key_confidence < KEY_CONFIDENCE_MIN;

function TrackEditor({ track, onSaved, onCancel }) {
  const feats = track.features?.full || {};
  const analysed = isAnalysed(track);
  const [bpm, setBpm] = useState(feats.bpm != null ? String(feats.bpm) : "");
  const [key, setKey] = useState(feats.key || "C");
  const [mode, setMode] = useState(feats.mode || "major");
  const [url, setUrl] = useState(track.source_url || "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const saveFeatures = async () => {
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

  const saveUrl = async () => {
    const next = url.trim();
    if (!next || next === (track.source_url || "")) { onCancel(); return; }
    if (!window.confirm(
      "Change the source URL?\n\nThis resets download, stems and analysis for this "
      + "track and re-processes it from the new link.")) return;
    setSaving(true);
    setError(null);
    try {
      await api.updateTrackUrl(track.id, next);
      toast("URL updated — re-processing from the new link");
      onSaved();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="feat-edit">
      <label style={{ flexBasis: "100%", display: "flex", gap: 6, alignItems: "center" }}>
        <span className="muted" style={{ width: 34 }}>URL</span>
        <input type="url" value={url} onChange={(e) => setUrl(e.target.value)}
          placeholder="soundcloud.com/…  ·  youtube.com/watch?v=…"
          style={{ flex: 1, minWidth: 0 }} />
        <button className="mini-btn" onClick={saveUrl}
          disabled={saving || !url.trim() || url.trim() === (track.source_url || "")}
          title="Repoint this track at a corrected URL (resets & re-processes)">
          Save URL
        </button>
      </label>
      {analysed && (
        <>
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
          <div className="mini-actions">
            <button className="mini-btn" onClick={saveFeatures} disabled={saving}>
              {saving ? "Saving…" : "Save features"}
            </button>
          </div>
        </>
      )}
      {error && <div className="error-text">{error}</div>}
      <div className="mini-actions">
        <button className="mini-btn" onClick={onCancel} disabled={saving}>Close</button>
      </div>
    </div>
  );
}

function PipelineDots({ track, runningKind }) {
  const p = pipelineDots(track, runningKind);
  const sepTag = track.stems?.separator;
  return (
    <div className="pipeline">
      <span className="dot" style={{ color: p.dl }}>●</span>DL
      <span className="dot" style={{ color: p.stems }}>●</span>
      <span title={sepTag ? `Separated by ${sepTag}` : undefined}>Stems</span>
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

  // Bottom player bar: which track + stem is loaded (null = bar hidden).
  const [player, setPlayer] = useState(null); // { trackId, stem }

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

  const playTrack = (t, stem = "full") => {
    if (!t.stems?.[stem]) {
      toast(stem === "full"
        ? "Couldn't play — is the track downloaded?"
        : "That stem isn't separated yet");
      return;
    }
    setPlayer({ trackId: t.id, stem });
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

  const removeTrack = async (t) => {
    if (!window.confirm(
      `Remove “${t.title}” from the library?\n\nThis deletes the song from the `
      + `database and its downloaded audio + stems from disk.`)) return;
    try {
      await api.deleteTrack(t.id);
      toast(`Removed “${t.title}”`);
      if (player?.trackId === t.id) setPlayer(null);
      if (editing === t.id) setEditing(null);
      refresh();
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
        <button className="mini-btn"
          title="Edit the source URL or correct BPM/key"
          onClick={() => setEditing(editing === t.id ? null : t.id)}>
          {editing === t.id ? "Close" : "Edit"}
        </button>
        <button className="mini-btn danger" disabled={g.pipelining}
          title="Remove this song and its audio from the library"
          onClick={() => removeTrack(t)}
          style={{ color: "var(--red, #ff6b6b)" }}>
          🗑 Remove
        </button>
      </div>
    );
  };

  return (
    <div className="page" style={{ paddingBottom: player ? 92 : undefined }}>
      <div className="screen-head">
        <h1>Library</h1>
        <span className="sub">{loading ? "refreshing…" : `${readyCount} ready to mash`}</span>
      </div>

      {/* Importing is not a place you go — it is a thing you do to the library
          (T4.2). Paste, preview, save; the tracks appear in the list below and
          start processing, with no tab change. The dependency-health banner
          rides along, so a missing ffmpeg is visible where the work happens. */}
      <PlaylistImporter embedded onIngested={() => refresh()} />

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
        <p className="empty">No tracks match. Paste a SoundCloud or YouTube link above to add some.</p>
      ) : view === "cards" ? (
        <div className="card-grid">
          {filtered.map((t) => {
            const f = t.features?.full || {};
            const g = gating(t);
            return (
              <div key={t.id} className="card">
                <div className="card-top">
                  <TrackArt id={t.id} thumbnail={t.thumbnail} className="card-art">♪</TrackArt>
                  <div className="card-id">
                    <div className="card-title" title={t.title}>{t.title}</div>
                    <div className="card-artist">{t.artist || "—"}</div>
                    <div className="card-statusrow">
                      <StatusTag status={t.status} />
                      <SourceBadge source={t.source} />
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
                  <KeyChip camelot={f.camelot} />
                  {keyLooksOff(f) && (
                    <span className="bpm-warn"
                      title={`Key uncertain (${f.key_confidence.toFixed(3)} confidence) — verify before trusting the suggested pitch shift. Key is the heaviest match weight.`}>
                      ⚠
                    </span>
                  )}
                  <div className="bpm-chip" title="Energy (0–100)">
                    <span className="u">EN </span>{f.energy != null ? Math.round(f.energy * 100) : "—"}
                  </div>
                </div>

                <div className="pipeline" style={{ justifyContent: "space-between" }}>
                  <PipelineDots track={t} runningKind={g.job?.kind || (g.pipelining ? STAGE_KIND[g.pipe?.stage] : undefined)} />
                  <span className="faint">{metaLine(t)}</span>
                </div>

                <div className="card-actions">
                  <button className={`act${player?.trackId === t.id && player.stem === "full" ? " on" : ""}`}
                    disabled={!t.stems.full}
                    onClick={() => playTrack(t, "full")}>
                    ▶ Preview
                  </button>
                  <button className={`act icon${player?.trackId === t.id && player.stem === "vocals" ? " on" : ""}`}
                    disabled={!t.stems.vocals} title="Play the vocal stem"
                    onClick={() => playTrack(t, "vocals")}>▶</button>
                  <button className="act vocal" disabled={!g.analysed || !t.stems.vocals}
                    title={g.analysed && t.stems.vocals ? "Load into Audition as the vocal" : "Needs analysed vocal stem"}
                    onClick={() => onSendToAudition?.({ vocalId: t.id })}>♪ Vocal</button>
                  <button className={`act icon${player?.trackId === t.id && player.stem === "instrumental" ? " on" : ""}`}
                    disabled={!t.stems.instrumental} title="Play the instrumental stem"
                    onClick={() => playTrack(t, "instrumental")}>▶</button>
                  <button className="act bed" disabled={!g.analysed || !t.stems.instrumental}
                    title={g.analysed && t.stems.instrumental ? "Load into Audition as the bed" : "Needs analysed instrumental stem"}
                    onClick={() => onSendToAudition?.({ instId: t.id })}>♪ Bed</button>
                </div>

                <RunActions t={t} />
                {editing === t.id && (
                  <TrackEditor track={t}
                    onSaved={() => { setEditing(null); refresh(); }}
                    onCancel={() => setEditing(null)} />
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="data-table">
          <div className="data-head" style={{ gridTemplateColumns: "34px 40px 2fr 100px 70px 70px 54px 1.4fr 1fr" }}>
            <div />
            <div>ID</div><div>TITLE / ARTIST</div><div>STATUS</div><div>BPM</div>
            <div>KEY</div><div>EN</div><div>PIPELINE</div><div style={{ textAlign: "right" }}>ACTIONS</div>
          </div>
          {filtered.map((t) => {
            const f = t.features?.full || {};
            const g = gating(t);
            return (
              <div key={t.id}>
                <div className="data-row" style={{ gridTemplateColumns: "34px 40px 2fr 100px 70px 70px 54px 1.4fr 1fr" }}>
                  <div>
                    <button className={`row-act play${player?.trackId === t.id ? " vocal" : ""}`}
                      disabled={!t.stems.full} title="Play in the bottom player"
                      onClick={() => playTrack(t, "full")}>▶</button>
                  </div>
                  <div className="mono faint">{t.id}</div>
                  <div>
                    <div className="t">{t.title}</div>
                    <div className="a">{t.artist || "—"}</div>
                  </div>
                  <div><StatusTag status={t.status} /></div>
                  <div className="mono" style={{ color: "var(--text-2)" }}>{f.bpm != null ? f.bpm.toFixed(1) : "—"}</div>
                  <div>
                    <KeyChip camelot={f.camelot} as="span" style={{ fontSize: 12, padding: "3px 7px" }} />
                  </div>
                  <div className="mono" style={{ color: "var(--text-2)" }} title="Energy (0–100)">
                    {f.energy != null ? Math.round(f.energy * 100) : "—"}
                  </div>
                  <div><PipelineDots track={t} runningKind={g.job?.kind} /></div>
                  <div className="row-actions">
                    <button className="row-act vocal" disabled={!g.analysed || !t.stems.vocals}
                      onClick={() => onSendToAudition?.({ vocalId: t.id })}>Vocal</button>
                    <button className="row-act bed" disabled={!g.analysed || !t.stems.instrumental}
                      onClick={() => onSendToAudition?.({ instId: t.id })}>Bed</button>
                    <button className="row-act" title="Edit source URL / BPM / key"
                      onClick={() => setEditing(editing === t.id ? null : t.id)}>Edit</button>
                    <button className="row-act" disabled={g.pipelining}
                      title="Remove this song and its audio" style={{ color: "var(--red, #ff6b6b)" }}
                      onClick={() => removeTrack(t)}>🗑</button>
                  </div>
                </div>
                {editing === t.id && (
                  <div style={{ padding: "0 8px 10px" }}>
                    <TrackEditor track={t}
                      onSaved={() => { setEditing(null); refresh(); }}
                      onCancel={() => setEditing(null)} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {player && (
        <PlayerBar
          track={filtered.find((x) => x.id === player.trackId)
            || tracks.find((x) => x.id === player.trackId)}
          stem={player.stem}
          onStemChange={(stem) => setPlayer((p) => ({ ...p, stem }))}
          onClose={() => setPlayer(null)}
        />
      )}
    </div>
  );
}
