import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import { StructureStrip } from "./StructureStrip";
import { SectionTable } from "./SectionTable";
import { PartnersRail } from "./PartnersRail";
import { keyOf } from "./pairs/pairModel";
import { hookUrl } from "../hooks/useHookAudition";
import { camelotColor, fmtDur, fmtPlays, fmtYear } from "../theme";

// Screen 1b. The visuals the library table deliberately does not carry, on
// demand: what this track is made of, and what it goes with.
//
// Everything here already exists in the database — sections since P2.1,
// waveforms since analysis, partners since scoring. Nothing on this screen is a
// new measurement.

const STEMS = [["full", "Full"], ["vocals", "Vocals"], ["instrumental", "Bed"]];

// Scoped to one song with the per-song cap OFF, so the PAIRS column counts
// every scored pairing rather than the three the ranked list would show. One
// song against a 30-track library is on the order of a hundred rows.
const PAIR_LIMIT = 500;

export function TrackDetail({ track, tracks, ratings, groups, role, onRole,
                              onBack, onStudio, onOpenTrack, onStatus }) {
  const [sections, setSections] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [stem, setStem] = useState("full");
  const [loop, setLoop] = useState(null);       // { start, end, index }
  const [position, setPosition] = useState(null);
  const [playingKey, setPlayingKey] = useState(null);
  const [error, setError] = useState(null);
  const audioRef = useRef(null);
  const mainRef = useRef(null);

  useEffect(() => {
    if (!track) return undefined;
    let live = true;
    setSections([]);
    setCandidates([]);
    setLoop(null);
    setPosition(null);
    api.getSections(track.id)
      .then((d) => { if (live) setSections(d.sections || []); })
      .catch((e) => { if (live) setError(e.message); });
    const opts = { limit: PAIR_LIMIT, maxPerSong: 0 };
    if (role === "instrumental") opts.instSongId = track.id;
    else opts.vocalSongId = track.id;
    api.getMashups(opts)
      .then((d) => { if (live) setCandidates(d.candidates || []); })
      .catch(() => { if (live) setCandidates([]); });
    return () => { live = false; };
  }, [track, role]);

  // Walking to a partner replaces the whole screen without unmounting it, so
  // the scroll position survives — and you land halfway down a page you have
  // not seen, below a hero that is now somebody else's.
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0 });
  }, [track?.id]);

  useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.isContentEditable)) return;
      if (e.key === "Escape") { e.preventDefault(); onBack(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onBack]);

  // This screen publishes NO float status. The pill is absolutely positioned at
  // the top right of the main column, which is exactly where this screen's
  // `song #N` and its `esc` button are — so anything reported here buries the
  // way out. Nothing is lost: the count it used to show is the SECTIONS tile,
  // and the partner count is the rail's own sub-line.
  useEffect(() => { onStatus(null); }, [track, onStatus]);

  // Energy is a raw analysis number whose scale means nothing on its own, so it
  // is shown as this library's own percentile — which is the only reading that
  // answers "is this a big record or a quiet one".
  const energyPct = useMemo(() => {
    const mine = track?.features?.full?.energy;
    if (mine == null) return null;
    const all = tracks.map((t) => t?.features?.full?.energy)
      .filter((v) => v != null).sort((a, b) => a - b);
    if (all.length < 2) return null;
    const below = all.filter((v) => v < mine).length;
    return Math.round((below / (all.length - 1)) * 100);
  }, [track, tracks]);

  const pairsBySection = useMemo(() => {
    const out = {};
    const field = role === "instrumental" ? "inst_section_idx" : "vocal_section_idx";
    for (const c of candidates) {
      const i = c[field];
      if (i == null) continue;
      out[i] = (out[i] || 0) + 1;
    }
    return out;
  }, [candidates, role]);

  // Sections and pairs both play through the same pre-cut clip endpoint the
  // dock uses, so a span auditioned here sounds like the same span auditioned
  // there — and the server's clip cache is shared.
  const playSpan = (songId, stemType, start, end, key) => {
    const el = audioRef.current;
    if (!el) return;
    if (playingKey === key) {
      el.pause();
      setPlayingKey(null);
      setPosition(null);
      return;
    }
    el.src = hookUrl(songId, stemType, start, end);
    el.loop = true;
    el.play().then(() => setPlayingKey(key)).catch((e) => setError(e.message));
    setLoop({ start, end });
  };

  const playSection = (s) => {
    // "Full" is not a separated stem, so a section auditioned on Full plays the
    // mix; Vocals and Bed play what the separator actually produced.
    playSpan(track.id, stem, s.start_sec, s.end_sec, `sec:${s.section_index}`);
  };

  // Opening a partner flips which side of a pair you are asking about. You got
  // here by looking at this track as the vocal, so the partner you clicked is a
  // bed — open it as one. Without the flip its rail would immediately re-scope
  // to ITS beds, which is a different question than the one you clicked.
  const openPartner = (id) => {
    onRole(role === "instrumental" ? "vocal" : "instrumental");
    onOpenTrack(id);
  };

  const playPair = (c) => {
    const mine = role === "instrumental"
      ? [c.inst_song_id, "instrumental", c.inst_section_start, c.inst_section_end]
      : [c.vocal_song_id, "vocals", c.vocal_section_start, c.vocal_section_end];
    playSpan(mine[0], mine[1], mine[2], mine[3], keyOf(c));
    setLoop({ start: mine[2], end: mine[3] });
  };

  // The playhead is drawn against the WHOLE track, but the clip only contains
  // the span — so its currentTime has to be put back where it came from.
  useEffect(() => {
    const el = audioRef.current;
    if (!el || !loop) return undefined;
    let raf = 0;
    const tick = () => {
      if (!el.paused) setPosition(loop.start + el.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [loop, playingKey]);

  if (!track) return null;

  const f = track.features?.full || {};
  const sepTag = track.stems?.separator || null;
  const stemCount = ["vocals", "instrumental", "drums", "bass", "other"]
    .filter((k) => track.stems?.[k]).length;
  const star = ratings.bySong[track.id] ?? null;
  const myGroups = (groups?.groupsOf(track.id) || [])
    .map((id) => groups.byId(id)).filter(Boolean);

  const sectionPlaying = playingKey?.startsWith("sec:")
    ? Number(playingKey.slice(4)) : null;

  return (
    <div className="detail">
      <header className="screen-bar">
        <button className="crumb" onClick={onBack}>Library</button>
        <span className="crumb-sep">/</span>
        <span className="crumb-here">{track.title}</span>
        <span className="mono detail-id">song #{track.id}</span>
        <button className="head-btn" onClick={onBack}>esc</button>
      </header>

      <div className="detail-body">
        <div className="detail-main" ref={mainRef}>
          <div className="hero">
            <TrackArt id={track.id} thumbnail={track.thumbnail} className="hero-art" />
            <div className="hero-text">
              <h1>{track.title}</h1>
              <div className="hero-meta">
                {[track.artist, track.genre, fmtYear(track.release_year),
                  track.plays ? `${fmtPlays(track.plays)} plays` : null]
                  .filter(Boolean).join(" · ")}
              </div>
              {/* Which shelves this track is on. Read-only, like Discover's
                  crate chip: the place that edits a group is the row menu in
                  the library, next to the track you are deciding about. */}
              {myGroups.length > 0 && (
                <div className="hero-groups">
                  {myGroups.map((g) => (
                    <span key={g.id} className="crate-chip" title="A library group">
                      ▨ {g.name}
                    </span>
                  ))}
                </div>
              )}
              <div className="tiles">
                <Tile label="BPM" value={f.bpm != null ? f.bpm.toFixed(1) : "—"} />
                <Tile label="KEY">
                  {f.camelot
                    ? <span className="tile-key mono"
                        style={{ background: camelotColor(f.camelot) }}>{f.camelot}</span>
                    : <span className="tile-dim">—</span>}
                  <span className="tile-note">
                    {[f.key, f.mode].filter(Boolean).join(" ") || "unmeasured"}
                  </span>
                </Tile>
                <Tile label="ENERGY" value={energyPct != null ? energyPct : "—"}
                  colour="var(--amber)"
                  title="This track's energy as a percentile of your own library — the raw number has no meaning on its own." />
                <Tile label="SECTIONS" value={sections.length || "—"} />
                <Tile label="LENGTH" value={fmtDur(track.duration_secs)} />
              </div>
            </div>
            <div className="hero-right">
              <div className="pd-seg">
                {STEMS.map(([id, label]) => (
                  <button key={id} className={stem === id ? "on" : ""}
                    onClick={() => setStem(id)}
                    disabled={id !== "full" && !track.stems?.[id]}
                    title={id !== "full" && !track.stems?.[id]
                      ? "That stem has not been separated yet" : `Audition on ${label}`}>
                    {label}
                  </button>
                ))}
              </div>
              <StarRating value={star} size={15}
                title={star ? `Best pairing rated ${star} of 5`
                  : "No pairing of this track has been rated"} />
              <span className="mono hero-sep">
                {sepTag ? `${sepTag.split(":")[0]} · ${stemCount} stems` : "no stems yet"}
              </span>
            </div>
          </div>

          {error && <div className="error-text">{error}</div>}

          <StructureStrip songId={track.id} sections={sections}
            duration={track.duration_secs} loop={loop} position={position}
            onPickSection={playSection} />

          <SectionTable sections={sections} pairsBySection={pairsBySection}
            playingIndex={sectionPlaying} onPlay={playSection} />
        </div>

        <PartnersRail candidates={candidates} role={role} ratings={ratings}
          judgedCount={ratings.count}
          playingKey={playingKey} onPlay={playPair} onStudio={onStudio}
          onOpenTrack={openPartner} />
      </div>

      <audio ref={audioRef} style={{ display: "none" }}
        onEnded={() => { setPlayingKey(null); setPosition(null); }} />
    </div>
  );
}

function Tile({ label, value, children, colour, title }) {
  return (
    <div className="tile" title={title}>
      <span className="tile-label mono">{label}</span>
      {children || (
        <span className="tile-value mono" style={colour ? { color: colour } : undefined}>
          {value}
        </span>
      )}
    </div>
  );
}
