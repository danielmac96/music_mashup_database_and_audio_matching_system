import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useHookAudition } from "../hooks/useHookAudition";
import { JobBadge } from "./JobBadge";
import { TrackArt } from "./TrackArt";
import { usePlan } from "../hooks/usePlan";
import {
  bpmTag, fmtTime, keyRel, tierFor,
} from "../theme";
import { toast } from "../toast";

const MIN_MATCHES = [50, 65, 75, 85];

// Pre-filter width presets passed to "Score library" (bpm = max BPM diff,
// key = min Camelot score). Tight = only clean matches; Wide = more candidates.
const MATCH_PRESETS = [
  { label: "Tight", bpm: 6, key: 0.75 },
  { label: "Balanced", bpm: 10, key: 0.55 },
  { label: "Wide", bpm: 16, key: 0.4 },
];
const SORTS = ["Score", "Popularity"];
// How many rows any one song may occupy. 0 = uncapped, which is what the list
// did before T3.4 — and why a single 128 BPM 8A vocal could hold 40 of 50 rows.
const PER_SONG_CAPS = [3, 2, 1, 0];
const popOf = (c) => (c.vocal_popularity || 0) + (c.inst_popularity || 0);

// Key drives 30% of the score and the suggested pitch shift, so an unreliable
// one has to be visible on the row you are about to judge. See TrackList.jsx for
// how key_confidence is derived; null means analysed before it existed.
// Calibrated against the real library — see TrackList.jsx for the distribution.
const KEY_CONFIDENCE_MIN = 0.012;
const keyLooksOff = (kc) => kc != null && kc < KEY_CONFIDENCE_MIN;
const keyWarnTitle = (kc) =>
  `Key uncertain (${kc.toFixed(3)} confidence) — the suggested pitch shift may be wrong.`;

function PlanDetails({ vocalId, instId, candidate }) {
  const { plan, error } = usePlan(vocalId, instId);

  if (error) return <div className="plan-detail error-text">{error}</div>;
  if (!plan) return <div className="plan-detail muted">Loading plan…</div>;

  const sc = candidate || {};
  const pc = (v) => `${Math.round((v || 0) * 100)}%`;

  return (
    <div className="plan-detail">
      {candidate && (
        <div className="raw-scores mono"
          title="Raw sub-scores behind the percentile shown on the row">
          composite <b>{pc(sc.score_total)}</b>
          {" · bpm "}{pc(sc.score_bpm)}
          {" · key "}{pc(sc.score_key)}
          {" · energy "}{pc(sc.score_energy)}
          {" · timbre "}{pc(sc.score_timbre)}
        </div>
      )}
      {sc.score_section != null && (
        <div className="raw-scores mono"
          title="The section pair this candidate was chosen for — what the preview plays">
          plays <b>{sc.vocal_section_label || "vocal"}</b>{" "}
          {fmtTime(sc.vocal_section_start)}–{fmtTime(sc.vocal_section_end)}
          {" over "}<b>{sc.inst_section_label || "bed"}</b>{" "}
          {fmtTime(sc.inst_section_start)}–{fmtTime(sc.inst_section_end)}
          {" · fit "}{pc(sc.score_section)}
        </div>
      )}
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
  const [presetIdx, setPresetIdx] = useState(1); // Balanced
  const [sortMode, setSortMode] = useState("Score");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scoreJobId, setScoreJobId] = useState(null);
  const [expanded, setExpanded] = useState(null);
  const [scorer, setScorer] = useState(null); // { scorer, model_version, auc }
  // ── T3.4 diversity: one 128 BPM 8A vocal otherwise owns the whole page ────
  const [maxPerSong, setMaxPerSong] = useState(3);
  const [grouped, setGrouped] = useState(false);
  const [hiddenCount, setHiddenCount] = useState(0);
  // ── T3.5 filters. "" / false is off; they compose, and all of them are
  // applied by the server over the whole table, not over the visible 50.
  const [filters, setFilters] = useState(
    { genre: "", era: "", energy: "", bpmBand: "", vocalForward: false });
  const [filterOpts, setFilterOpts] = useState(null);

  // ── T1.7 triage: highlighted row, verdicts, shortlist, shortcut legend ────
  const [cursor, setCursor] = useState(0);
  const [verdicts, setVerdicts] = useState({});   // "vocalId:instId" -> love|ok|no
  const [shortlist, setShortlist] = useState(() => new Set());
  const [showKeys, setShowKeys] = useState(false);
  const rowRefs = useRef(new Map());
  // `auditioning` is the user's intent (space toggles it); playingId is what the
  // engine actually has armed. Keeping them separate is what lets the audio
  // FOLLOW the cursor: arrowing while auditioning re-arms on the new row
  // instead of making you press play again on every candidate.
  const [auditioning, setAuditioning] = useState(false);
  const { audition, stop, prefetch, playingId, error: audioError } = useHookAudition();

  const refreshScorer = () => api.getScorerStatus().then(setScorer).catch(() => setScorer(null));
  useEffect(() => { refreshScorer(); }, []);

  // Verdicts persist server-side (T2.1), so a reload shows what you already judged.
  useEffect(() => {
    api.getPairFeedback()
      .then((d) => setVerdicts(Object.fromEntries(
        (d.feedback || []).map((f) => [`${f.vocal_song_id}:${f.inst_song_id}`, f.verdict]))))
      .catch(() => {});
  }, []);

  const refresh = async (type = comboType, activeSeed = seed, min = minMatch,
                        cap = maxPerSong, group = grouped, f = filters) => {
    setLoading(true);
    setError(null);
    try {
      // "Best bed per vocal" answers a different question from the flat list —
      // what can I do with each acapella, rather than what is the single best
      // pair — so it has its own endpoint and ignores the seed and the cap.
      if (group && type === "vocal_over_instrumental" && activeSeed?.songId == null) {
        const data = await api.getBestBedPerVocal({ limit: 50, minScore: min / 100 });
        setCandidates(data.candidates);
        return;
      }
      const opts = { comboType: type, minScore: min / 100, limit: 50,
                     maxPerSong: cap, ...f };
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
    const n = candidates.length;
    const filtered = Object.values(filters).some(Boolean);
    onStatus?.({
      text: `${n} ${filtered ? "matching" : "scored"} pair${n === 1 ? "" : "s"}`
        + (filtered ? ` · ${activeFilters} filter${activeFilters === 1 ? "" : "s"}` : ""),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidates.length, onStatus, filters]);

  const seedTitle = useMemo(() => {
    if (seed?.songId == null) return null;
    const c = candidates[0];
    if (!c) return `#${seed.songId}`;
    return seed.role === "instrumental" ? c.inst_title : c.vocal_title;
  }, [seed, candidates]);

  const startScoring = async () => {
    try {
      const p = MATCH_PRESETS[presetIdx];
      const { job_id } = await api.startScoring({ bpmMaxDiff: p.bpm, keyMinScore: p.key });
      setScoreJobId(job_id);
      toast(`Scoring library (${p.label} match)…`);
    } catch (e) {
      setError(e.message);
    }
  };

  const sortedCandidates = useMemo(() => {
    if (sortMode === "Popularity") {
      return [...candidates].sort((a, b) => popOf(b) - popOf(a));
    }
    return candidates; // server already returns score-descending
  }, [candidates, sortMode]);

  // ── T1.7 keyboard triage ─────────────────────────────────────────────────
  const keyOf = (c) => `${c.vocal_song_id}:${c.inst_song_id}`;
  const current = sortedCandidates[cursor] || null;

  // Keep the highlight on a real row when the list changes underneath it
  // (re-score, filter, sort) instead of pointing past the end.
  useEffect(() => {
    setCursor((i) => Math.min(i, Math.max(0, sortedCandidates.length - 1)));
  }, [sortedCandidates.length]);

  useEffect(() => {
    rowRefs.current.get(current?.id)?.scrollIntoView({ block: "nearest" });
    // Warm the NEXT rows, not this one — this one is already decoding.
    prefetch(sortedCandidates.slice(cursor + 1, cursor + 3));
  }, [cursor, current?.id, sortedCandidates, prefetch]);

  // Audio follows the cursor while auditioning; silence when toggled off.
  useEffect(() => {
    if (auditioning && current) audition(current);
    else stop();
    // Intentionally keyed on the row IDENTITY, not the object — re-renders
    // from unrelated state must not restart playback mid-listen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auditioning, current?.id]);

  const judge = useCallback(async (c, verdict) => {
    if (!c) return;
    const k = keyOf(c);
    const prev = verdicts[k];
    setVerdicts((v) => ({ ...v, [k]: verdict }));   // optimistic — keep triage fast
    try {
      await api.savePairFeedback({
        vocalSongId: c.vocal_song_id, instSongId: c.inst_song_id, verdict,
        // The verdict is about the sections that were actually playing (T3.3),
        // so record which ones — a "no" on one chorus is not a "no" on the song.
        vocalSection: c.vocal_section_idx ?? null,
        instSection: c.inst_section_idx ?? null,
      });
    } catch (e) {
      setVerdicts((v) => ({ ...v, [k]: prev }));    // put it back if it didn't stick
      toast(e.message || "Could not save that verdict");
    }
  }, [verdicts]);

  const judgeAndAdvance = useCallback((verdict) => {
    const c = sortedCandidates[cursor];
    if (!c) return;
    judge(c, verdict);
    // Judging is a decision, so move on — that is what makes 50 candidates
    // a two-minute pass rather than a chore.
    setCursor((i) => Math.min(i + 1, sortedCandidates.length - 1));
  }, [cursor, judge, sortedCandidates]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (!sortedCandidates.length) return;

      switch (e.key) {
        case "j": case "ArrowDown":
          e.preventDefault();
          setCursor((i) => Math.min(i + 1, sortedCandidates.length - 1));
          break;
        case "k": case "ArrowUp":
          e.preventDefault();
          setCursor((i) => Math.max(i - 1, 0));
          break;
        case " ":
          e.preventDefault();
          setAuditioning((a) => !a);
          break;
        case "f": e.preventDefault(); judgeAndAdvance("love"); break;
        case "d": e.preventDefault(); judgeAndAdvance("no"); break;
        case "s": {
          e.preventDefault();
          const c = sortedCandidates[cursor];
          if (c) setShortlist((s) => {
            const n = new Set(s);
            n.has(c.id) ? n.delete(c.id) : n.add(c.id);
            return n;
          });
          break;
        }
        case "h": {
          e.preventDefault();
          const c = sortedCandidates[cursor];
          // Hiding removes the row, so the cursor already points at the next
          // pair — no advance, or you skip one.
          if (c) hide(c);
          break;
        }
        case "?": e.preventDefault(); setShowKeys((v) => !v); break;
        case "Escape": setAuditioning(false); setShowKeys(false); break;
        default: break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, sortedCandidates, judgeAndAdvance, hide]);

  // Switching away from the tab must not leave an AudioContext playing.
  useEffect(() => stop, [stop]);

  const switchType = (type) => { setComboType(type); setExpanded(null); refresh(type, seed, minMatch); };
  const cycleMin = () => {
    const next = MIN_MATCHES[(MIN_MATCHES.indexOf(minMatch) + 1) % MIN_MATCHES.length];
    setMinMatch(next);
    refresh(comboType, seed, next);
  };
  const cycleCap = () => {
    const next = PER_SONG_CAPS[(PER_SONG_CAPS.indexOf(maxPerSong) + 1) % PER_SONG_CAPS.length];
    setMaxPerSong(next);
    refresh(comboType, seed, minMatch, next);
  };
  const toggleGrouped = () => {
    const next = !grouped;
    setGrouped(next);
    setExpanded(null);
    refresh(comboType, seed, minMatch, maxPerSong, next);
  };

  // Only offer values this library actually contains — a Genre chip cycling
  // through forty genres you own none of is worse than no chip.
  useEffect(() => {
    api.getMashupFilters(comboType).then(setFilterOpts).catch(() => setFilterOpts(null));
  }, [comboType]);

  // Chips cycle: "" (off) → each value → back to off.
  const cycleFilter = (key, values) => {
    const list = ["", ...(values || [])];
    const next = list[(list.indexOf(filters[key]) + 1) % list.length];
    const f = { ...filters, [key]: next };
    setFilters(f);
    refresh(comboType, seed, minMatch, maxPerSong, grouped, f);
  };

  const toggleVocalForward = () => {
    const f = { ...filters, vocalForward: !filters.vocalForward };
    setFilters(f);
    refresh(comboType, seed, minMatch, maxPerSong, grouped, f);
  };

  const clearFilters = () => {
    const f = { genre: "", era: "", energy: "", bpmBand: "", vocalForward: false };
    setFilters(f);
    refresh(comboType, seed, minMatch, maxPerSong, grouped, f);
  };

  const activeFilters = Object.entries(filters).filter(([, v]) => v).length;

  const refreshHiddenCount = useCallback(() => {
    api.getHidden()
      .then((d) => setHiddenCount((d.pairs?.length || 0) + (d.tracks?.length || 0)))
      .catch(() => {});
  }, []);
  useEffect(() => { refreshHiddenCount(); }, [refreshHiddenCount]);

  // Hiding is a display preference, not a verdict — it never becomes training
  // data. Drop the row immediately: waiting for a refetch to make it disappear
  // is what makes triage feel slow.
  const hide = useCallback(async (c) => {
    if (!c) return;
    setCandidates((rows) => rows.filter((r) => r.id !== c.id));
    try {
      await api.hidePair(c.vocal_song_id, c.inst_song_id);
      refreshHiddenCount();
    } catch (e) {
      toast(e.message || "Could not hide that pair");
      refresh();
    }
  }, [refreshHiddenCount]);   // eslint-disable-line react-hooks/exhaustive-deps

  const excludeSong = useCallback(async (songId, title) => {
    if (songId == null) return;
    setCandidates((rows) => rows.filter(
      (r) => r.vocal_song_id !== songId && r.inst_song_id !== songId));
    try {
      await api.excludeTrack(songId);
      refreshHiddenCount();
      toast(`Excluded "${title || songId}" from Discover`);
    } catch (e) {
      toast(e.message || "Could not exclude that track");
      refresh();
    }
  }, [refreshHiddenCount]);   // eslint-disable-line react-hooks/exhaustive-deps

  const unhideAll = useCallback(async () => {
    try {
      const d = await api.getHidden();
      await Promise.all([
        ...(d.pairs || []).map((p) => api.unhidePair(p.vocal_song_id, p.inst_song_id)),
        ...(d.tracks || []).map((t) => api.includeTrack(t.song_id)),
      ]);
      refreshHiddenCount();
      refresh();
      toast("Restored every hidden pair and excluded track");
    } catch (e) {
      toast(e.message || "Could not restore");
    }
  }, [refreshHiddenCount]);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="page">
      <div className="screen-head">
        <h1>Mashups</h1>
        <span className="sub">
          {loading ? "loading…" : `Ranked best-first${seedTitle ? " · seed fixed" : ""}`}
        </span>
        <span style={{ flex: 1 }} />
        {scorer && (
          <span className="scorer-badge" title={
            scorer.scorer === "model"
              ? "Scored by the learned model. Presets set the BPM window; the key gate is dropped for the model."
              : "Scored by the hand-weighted heuristic. Train + activate a model on the Database tab to switch."
          }>
            Scorer: {scorer.scorer === "model"
              ? `Model ${scorer.model_version || ""}${scorer.auc != null ? ` (AUC ${scorer.auc})` : ""}`
              : "Heuristic"}
          </span>
        )}
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
        <div className="chip" onClick={() => setPresetIdx((presetIdx + 1) % MATCH_PRESETS.length)}
          title="Pre-filter width used by 'Score library' — re-score to apply">
          <span className="k">Match width</span><span>{MATCH_PRESETS[presetIdx].label}</span><span className="caret">▾</span>
        </div>
        <div className="chip" onClick={() => setSortMode(SORTS[(SORTS.indexOf(sortMode) + 1) % SORTS.length])}>
          <span className="k">Sort</span><span>{sortMode}</span><span className="caret">▾</span>
        </div>
        <div className="chip" onClick={cycleCap}
          title="How many rows one song may occupy, counting both sides. Without a cap, one well-placed vocal takes the whole page.">
          <span className="k">Per song</span>
          <span className="mono">{maxPerSong === 0 ? "∞" : `≤${maxPerSong}`}</span>
          <span className="caret">▾</span>
        </div>
        {comboType === "vocal_over_instrumental" && seed?.songId == null && (
          <div className={`chip${grouped ? " active" : ""}`} onClick={toggleGrouped}
            title="One row per vocal: the best bed for each of your acapellas, instead of the best pairs overall.">
            <span className="k">View</span><span>{grouped ? "Per vocal" : "Flat"}</span>
            <span className="caret">▾</span>
          </div>
        )}
        {!grouped && (
          <>
            <div className="chip" onClick={() => cycleFilter(
              "genre", (filterOpts?.genres || []).map((g) => g.genre))}
              title="Pairs containing a track of this genre, on either side">
              <span className="k">Genre</span>
              <span>{filters.genre || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className="chip" onClick={() => cycleFilter("era", filterOpts?.eras)}
              title="Pairs containing a record from this era, on either side">
              <span className="k">Era</span>
              <span>{filters.era || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className="chip" onClick={() => cycleFilter("bpmBand", filterOpts?.bpm_bands)}
              title="Target tempo — the vocal's BPM, which the bed is conformed to">
              <span className="k">BPM</span>
              <span className="mono">{filters.bpmBand || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className="chip" onClick={() => cycleFilter("energy", filterOpts?.energy_bands)}
              title="How hard the bed hits, ranked within your library">
              <span className="k">Energy</span>
              <span>{filters.energy || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className={`chip${filters.vocalForward ? " active" : ""}`}
              onClick={toggleVocalForward}
              title="Only pairs whose vocal section is properly sung, not an ad-lib">
              <span className="k">Vocal-forward</span>
              <span>{filters.vocalForward ? "On" : "Off"}</span>
            </div>
            {activeFilters > 0 && (
              <div className="chip" onClick={clearFilters} title="Clear every filter">
                <span className="k">Filters</span>
                <span className="mono">{activeFilters}</span><span className="x">✕</span>
              </div>
            )}
          </>
        )}
        {hiddenCount > 0 && (
          <div className="chip" onClick={unhideAll}
            title="Restore every pair you hid and every track you excluded">
            <span className="k">Hidden</span><span className="mono">{hiddenCount}</span>
            <span className="x">↺</span>
          </div>
        )}
        {seedTitle && (
          <div className="chip seed">
            Seeded: <b>{seedTitle}</b> as {seed.role === "instrumental" ? "bed" : "vocal"}
            <span className="x" onClick={() => { onClearSeed?.(); refresh(comboType, null, minMatch); }}>✕</span>
          </div>
        )}
        <div className="spacer" />
        <div className="chip" onClick={() => setShowKeys((v) => !v)}
          title="Keyboard shortcuts for judging by ear">
          <span className="k">Keys</span><span className="mono">?</span>
        </div>
        {scoreJobId ? (
          <JobBadge jobId={scoreJobId} onComplete={() => { setScoreJobId(null); refresh(); }} />
        ) : (
          <button className="btn" onClick={startScoring}>↻ Score library</button>
        )}
      </div>

      {showKeys && (
        <div className="key-legend">
          <b>Judge by ear</b>
          <span><kbd>j</kbd>/<kbd>k</kbd> or <kbd>↑</kbd><kbd>↓</kbd> move</span>
          <span><kbd>space</kbd> play / stop the 16-bar hooks</span>
          <span><kbd>f</kbd> ✓ keep · <kbd>d</kbd> ✗ reject (both advance)</span>
          <span><kbd>s</kbd> shortlist · <kbd>h</kbd> hide this pair</span>
          <span><kbd>esc</kbd> stop · <kbd>?</kbd> close</span>
          <span className="faint">
            The bed is conformed to the vocal's tempo and pitch automatically.
            Verdicts are saved and train the scorer.
          </span>
        </div>
      )}

      {audioError && (
        <p className="error-text">
          {audioError} — the hook clip may not be rendered yet. Analyze the track,
          or open it in Studio to audition the full stems.
        </p>
      )}

      <div className="legend">
        <span>Sub-scores:</span>
        <span className="sw"><i style={{ background: "var(--cyan)" }} />BPM</span>
        <span className="sw"><i style={{ background: "var(--violet)" }} />Key</span>
        <span className="sw"><i style={{ background: "var(--amber)" }} />Energy</span>
        <span className="sw"><i style={{ background: "var(--green)" }} />Timbre</span>
        <span className="weights">Weighted: Key 30 · BPM 25 · Timbre 25 · Energy 20</span>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}

      {sortedCandidates.length === 0 && !loading ? (
        <p className="empty">
          {activeFilters > 0
            ? "No pairs match every filter — clear one, or lower Min match."
            : minMatch > MIN_MATCHES[0]
            ? `No pairs score ${minMatch}% or better — click "Min match" to lower the bar, or re-score after analyzing more tracks.`
            : "No scored pairs yet. Analyze your tracks (Library tab), then click “Score library”."}
        </p>
      ) : (
        <div className="pair-list">
          {sortedCandidates.map((c, i) => {
            const total = Math.round((c.score_total || 0) * 100);
            // The headline is the pair's rank within YOUR library, not the raw
            // composite — a raw 0-1 reads ~78% for almost everything and so
            // tells you nothing about whether this pair is worth your time.
            // The raw value stays in the tooltip and the Plan expander.
            const pct = Math.round((c.score_percentile ?? 0) * 100);
            const { tier, color, textColor } = tierFor(total);
            const kr = keyRel(c.vocal_camelot, c.inst_camelot);
            const w = (v) => `${Math.round((v || 0) * 100)}%`;
            const isVI = comboType === "vocal_over_instrumental";
            const verdict = verdicts[keyOf(c)];
            const isCursor = i === cursor;
            return (
              <Fragment key={c.id}>
                <div
                  ref={(el) => { el ? rowRefs.current.set(c.id, el) : rowRefs.current.delete(c.id); }}
                  onClick={() => setCursor(i)}
                  className={`pair${i === 0 ? " top" : ""}${isCursor ? " cursor" : ""}`
                    + `${verdict ? ` judged-${verdict}` : ""}`
                    + `${playingId === c.id ? " playing" : ""}`}>
                  <div className="pair-rank">
                    {shortlist.has(c.id) ? "★" : i + 1}
                    {verdict && (
                      <span className={`verdict-chip ${verdict}`}
                        title={`You judged this "${verdict}" — press f/d to change`}>
                        {verdict === "no" ? "✗" : verdict === "love" ? "✓" : "~"}
                      </span>
                    )}
                  </div>
                  <div className="pair-side">
                    <TrackArt id={c.vocal_song_id} className="pair-art">♪</TrackArt>
                    <div style={{ minWidth: 0 }}>
                      <div className="pair-role vocal">{isVI ? "VOCAL (TOP)" : "INSTR (TOP)"}</div>
                      <div className="pair-title" title={c.vocal_title}>{c.vocal_title}</div>
                      <div className="pair-meta">
                        {c.vocal_artist || "—"} · {c.vocal_bpm?.toFixed(1)} · {c.vocal_camelot || "?"}
                        {keyLooksOff(c.vocal_key_confidence) && (
                          <span className="bpm-warn" title={keyWarnTitle(c.vocal_key_confidence)}>
                            {" ⚠"}
                          </span>
                        )}
                        {c.vocal_popularity != null && (
                          <span className="pop" title="Popularity percentile in your library">
                            {" · ★"}{Math.round(c.vocal_popularity * 100)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="score-cluster">
                    <div className="score-hero"
                      title={`Top ${Math.max(1, 100 - pct)}% of your scored pairs. Raw composite ${total}% — see Plan for the breakdown.`}>
                      <span className="pctv" style={{ color }}>{pct}</span>
                      <span className="pct-suffix">pctl</span>
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
                      <div className="pair-meta">
                        {c.inst_artist || "—"} · {c.inst_bpm?.toFixed(1)} · {c.inst_camelot || "?"}
                        {keyLooksOff(c.inst_key_confidence) && (
                          <span className="bpm-warn" title={keyWarnTitle(c.inst_key_confidence)}>
                            {" ⚠"}
                          </span>
                        )}
                        {c.inst_popularity != null && (
                          <span className="pop" title="Popularity percentile in your library">
                            {" · ★"}{Math.round(c.inst_popularity * 100)}
                          </span>
                        )}
                      </div>
                    </div>
                    <TrackArt id={c.inst_song_id + 3} className="pair-art">♪</TrackArt>
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
                    <button className="plan" onClick={() => hide(c)}
                      title="Hide this pairing (h). Kept out of every future list until you restore it.">
                      ⊘ Hide
                    </button>
                    <button className="plan"
                      onClick={() => excludeSong(c.vocal_song_id, c.vocal_title)}
                      title={`Exclude "${c.vocal_title}" from Discover entirely, on either side of a pair`}>
                      ⊘ Top track
                    </button>
                  </div>
                </div>
                {expanded === c.id && (
                  <PlanDetails vocalId={c.vocal_song_id} instId={c.inst_song_id} candidate={c} />
                )}
              </Fragment>
            );
          })}
        </div>
      )}
    </div>
  );
}
