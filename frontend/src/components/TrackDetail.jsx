import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import { StructureStrip } from "./StructureStrip";
import { SectionTable } from "./SectionTable";
import { PartnersRail } from "./PartnersRail";
import { AudioSource, AudioSourcePicker } from "./AudioSourcePicker";
import { keyOf } from "./pairs/pairModel";
import { camelotColor, fmtDur, fmtPlays, fmtYear } from "../theme";

// Screen 1b. The visuals the library table deliberately does not carry, on
// demand: what this track is made of, and what it goes with.
//
// Everything here already exists in the database — sections since P2.1,
// waveforms since analysis, partners since scoring. Nothing on this screen is a
// new measurement.

const STEMS = [["full", "Full"], ["vocals", "Vocals"], ["instrumental", "Bed"]];
const STEM_LABEL = Object.fromEntries(STEMS);

// Scoped to one song with the per-song cap OFF, so the PAIRS column counts
// every scored pairing rather than the three the ranked list would show. One
// song against a 30-track library is on the order of a hundred rows.
const PAIR_LIMIT = 500;

export function TrackDetail({ track, tracks, ratings, groups, player, role,
                              onRole, onBack, onStudio, onOpenTrack, onStatus,
                              onChanged }) {
  const [sections, setSections] = useState([]);
  // The "Wrong audio?" picker. Closed on every walk to another track: it holds
  // one track's search results, and they must not sit under another's title.
  const [picking, setPicking] = useState(false);
  useEffect(() => { setPicking(false); }, [track?.id]);
  const [candidates, setCandidates] = useState([]);
  const [stem, setStem] = useState("full");
  const [error, setError] = useState(null);
  const mainRef = useRef(null);

  // The playhead, the loop window and "which row is playing" all come off the
  // app-wide player now. This screen used to own a SECOND hidden <audio> of its
  // own — which is why the bar at the bottom never knew a section was looping,
  // and why starting a section here did not stop the song the library started.
  // `detailKey` rides along on the source so this screen can still ask "is THIS
  // row the one playing" without owning the audio.
  const src = player.source;
  const playingKey = src?.detailKey ?? null;
  // A section is a LOOP WINDOW on the whole file now, not a pre-cut clip, so the
  // loop comes straight off the source and the playhead runs the length of the
  // record rather than the length of the span.
  const onThisTrack = player.kind === "track" && src?.songId === track?.id;
  const position = onThisTrack || playingKey ? player.position : null;
  const loop = onThisTrack && src.loop
    ? { start: src.loop.start, end: src.loop.end } : null;

  // The bar can change the stem too (1b). Mirror it, or the hero segment ends up
  // claiming Full while the speakers are playing the vocal.
  useEffect(() => {
    if (onThisTrack && src.stem && src.stem !== stem) setStem(src.stem);
  }, [onThisTrack, src?.stem]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!track) return undefined;
    let live = true;
    setSections([]);
    setCandidates([]);
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

  // Sections and pairs both loop a WINDOW on the whole stem file. They used to
  // play a server-cut WAV holding only the span, which meant the playhead could
  // not be dragged anywhere else in the record and the bar's strip had nothing
  // but twelve seconds on it. Same audio, same moment — reachable surroundings.
  const playSpan = (songId, stemType, start, end, key, title, subtitle,
                    stems, duration) => {
    player.toggle({
      kind: "track",
      songId, stem: stemType,
      loop: { start, end },
      detailKey: key, title, subtitle,
      stems, duration,
    });
  };

  const playSection = (s) => {
    // "Full" is not a separated stem, so a section auditioned on Full plays the
    // mix; Vocals and Bed play what the separator actually produced.
    playSpan(track.id, stem, s.start_sec, s.end_sec, `sec:${s.section_index}`,
      `${s.label || "section"} · ${track.title}`,
      `${STEM_LABEL[stem]} · ${fmtDur(s.end_sec - s.start_sec)}`,
      track.stems, track.duration_secs);
  };

  // The whole track, on whichever stem is selected. The library row plays the
  // full mix and this screen had no way to do the same — you could audition a
  // twelve-second section of a record here but never the record.
  const playWhole = () => {
    player.toggle({
      kind: "track",
      songId: track.id,
      stem,
      // The stem is NOT part of the identity any more — it is a live control on
      // the bar, so switching it must not read as a different thing playing.
      detailKey: "whole",
      title: track.title,
      subtitle: [STEM_LABEL[stem], track.artist].filter(Boolean).join(" · "),
      duration: track.duration_secs,
      stems: track.stems,
    });
  };

  // Two controls for one fact, so they have to agree: the hero segment switches
  // the live audio when the player is on this track, and mirrors the bar's own
  // switch back into local state when the bar is the one that moved it.
  const pickStem = (id) => {
    setStem(id);
    if (onThisTrack) player.switchStem(id);
  };

  // Click or drag anywhere on the waveform. On an idle track this STARTS it
  // there rather than doing nothing, which is the only reading of a click on a
  // waveform that is not a dead end.
  const seekTo = (secs) => {
    if (onThisTrack) { player.seek(secs); return; }
    player.play({
      kind: "track",
      songId: track.id,
      stem,
      startAt: secs,
      detailKey: "whole",
      title: track.title,
      subtitle: [STEM_LABEL[stem], track.artist].filter(Boolean).join(" · "),
      duration: track.duration_secs,
      stems: track.stems,
    });
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
    // A scored pair exists only where both stems were separated, so the
    // partner's vox and bed are known to be on disk even though this screen
    // never fetched its row.
    playSpan(mine[0], mine[1], mine[2], mine[3], keyOf(c),
      role === "instrumental" ? c.inst_title : c.vocal_title,
      role === "instrumental" ? `bed · under ${c.vocal_title}`
        : `vocal · over ${c.inst_title}`,
      { full: true, vocals: true, instrumental: true }, null);
  };

  if (!track) return null;

  const f = track.features?.full || {};
  const sepTag = track.stems?.separator || null;
  const stemCount = ["vocals", "instrumental", "drums", "bass", "other"]
    .filter((k) => track.stems?.[k]).length;
  const star = ratings.bySong[track.id] ?? null;
  const myGroups = (groups?.groupsOf(track.id) || [])
    .map((id) => groups.byId(id)).filter(Boolean);
  // DERIVED from the live loop, not from the detailKey the row was started with.
  // Scrubbing out of a loop releases it, and a key kept on the source would then
  // leave a section row claiming to be playing while the playhead is a minute
  // away from it. Same reason usePairDock derives armedKey rather than storing it.
  const sectionPlaying = loop
    ? (sections.find((s) => Math.abs(s.start_sec - loop.start) < 0.01
                         && Math.abs(s.end_sec - loop.end) < 0.01)?.section_index ?? null)
    : null;
  const wholePlaying = playingKey === "whole" && !loop && player.playing;

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
              {/* Where the audio came from. A substituted download used to be
                  invisible here — the link said SoundCloud while the stems
                  were cut from a YouTube remix. */}
              <AudioSource track={track} onPick={() => setPicking((v) => !v)}
                onChanged={onChanged} />
              {picking && (
                <AudioSourcePicker track={track} onClose={() => setPicking(false)}
                  onChanged={onChanged} />
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
              <div className="hero-transport">
                {/* The library row plays a whole record and this screen could
                    only play twelve-second slices of one. Same player, same
                    bar — it just needed a way in. */}
                <button className="hero-play" onClick={playWhole}
                  title={wholePlaying ? "Pause"
                    : `Play the whole track (${STEM_LABEL[stem]})`}>
                  {wholePlaying ? "❚❚" : "▶"}
                </button>
                <div className="pd-seg">
                  {STEMS.map(([id, label]) => (
                    <button key={id} className={stem === id ? "on" : ""}
                      onClick={() => pickStem(id)}
                      disabled={id !== "full" && !track.stems?.[id]}
                      title={id !== "full" && !track.stems?.[id]
                        ? "That stem has not been separated yet"
                        : `Audition on ${label} — switches live while playing`}>
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              <StarRating value={star} size={15}
                title={star ? `Best pairing rated ${star} of 5`
                  : "No pairing of this track has been rated"} />
              <span className="mono hero-sep">
                {sepTag ? `${sepTag.split(":")[0]} · ${stemCount} stems` : "no stems yet"}
              </span>
            </div>
          </div>

          {(error || player.error) && (
            <div className="error-text">{error || player.error}</div>
          )}

          <StructureStrip songId={track.id} sections={sections}
            duration={track.duration_secs} loop={loop} position={position}
            stem={stem} onSeek={seekTo} onPickSection={playSection} />

          <SectionTable sections={sections} pairsBySection={pairsBySection}
            playingIndex={sectionPlaying} onPlay={playSection} />
        </div>

        <PartnersRail candidates={candidates} role={role} ratings={ratings}
          judgedCount={ratings.count}
          playingKey={playingKey} onPlay={playPair} onStudio={onStudio}
          onOpenTrack={openPartner} />
      </div>
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
