import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import { StructureStrip } from "./StructureStrip";
import { SectionTable } from "./SectionTable";
import { TrackEditor } from "./TrackEditor";
import { fmtAttr } from "../attributes";
import { AudioSource, AudioSourcePicker } from "./AudioSourcePicker";
import { Disclosure, KeyChip, StatusPill } from "./ui/primitives";
import { keyOf, pctOf, tierOf } from "./pairs/pairModel";
import { STATUS, STATUS_PILL, statusKeyOf } from "../hooks/useLibraryFilters";
import { notifyPairsInputChanged, requestRescore } from "../rescore";
import { compatColor, fmtDur, fmtYear, keyLooksOff } from "../theme";

// Track detail (revamp 1c). It replaces the Library's table and leaves the
// pair dock beside it: the breadcrumb, a hero (art with a hook ▶, title,
// artist, source, status), a stat strip, the structure canvas with its
// Full / Vocals / Bed switch, the section table, and five disclosure groups —
// Analysis attributes, Stems & quality, Source & provenance, Partners and
// Actions — all collapsed until opened, and remembered (mashup.detailGroups.v1).
//
// Everything here already exists in the database: sections, waveforms,
// partners, attributes. Nothing on this screen is a new measurement.

const STEMS = [["full", "Full"], ["vocals", "Vocals"], ["instrumental", "Bed"]];
const STEM_LABEL = Object.fromEntries(STEMS);
const GROUPS_KEY = "mashup.detailGroups.v1";
const STATUS_LABEL = Object.fromEntries(STATUS);

// Scoped to one song with the per-song cap OFF, so the section table's PAIRS
// counts every scored pairing rather than the three the ranked list shows.
const PAIR_LIMIT = 500;
const PARTNER_ROWS = 4;

const sourceOf = (url) => {
  const u = String(url || "");
  if (/youtu\.?be/i.test(u)) return "YouTube";
  if (/soundcloud/i.test(u)) return "SoundCloud";
  return u ? "Link" : "—";
};

function Stat({ label, tip, sub = null, subTone = null, children }) {
  return (
    <div className="dt-stat" title={tip}>
      <span className="caps">{label}</span>
      <div className="dt-statval">{children}</div>
      {sub != null && <span className="dt-statsub" style={subTone ? { color: subTone } : undefined}>{sub}</span>}
    </div>
  );
}
const Num = ({ v, unit, color }) => (v == null || v === "—"
  ? <span className="dt-num mono faint">—</span>
  : <><span className="dt-num mono" style={color ? { color } : undefined}>{v}</span>
    {unit && <span className="dt-unit">{unit}</span>}</>);

export function TrackDetail({ track, tracks, attributes = null, ratings, groups, player, role,
                              onRole, onBack, onStudio, onOpenTrack, onStatus,
                              onChanged, onDiscover = null, onLinesChanged = null,
                              onPlayPair = null, onCompareWith = null, onChooseAttributes = null,
                              onOpenQueue = null, onOpenSegments = null, running = false }) {
  const [sections, setSections] = useState([]);
  // The "Wrong audio?" picker and the key editor. Closed on every walk to
  // another track: they hold one track's state, which must not sit under
  // another's title.
  const [picking, setPicking] = useState(false);
  const [editing, setEditing] = useState(false);
  useEffect(() => { setPicking(false); setEditing(false); }, [track?.id]);
  const [candidates, setCandidates] = useState([]);
  const [others, setOthers] = useState([]);
  const [stem, setStem] = useState("full");
  const [error, setError] = useState(null);
  const mainRef = useRef(null);

  // The playhead, the loop window and "which row is playing" all come off the
  // app-wide player. `detailKey` rides along on the source so this screen can
  // ask "is THIS row the one playing" without owning the audio.
  const src = player.source;
  const playingKey = src?.detailKey ?? null;
  // A section is a LOOP WINDOW on the whole file, so the loop comes straight
  // off the source and the playhead runs the length of the record.
  const onThisTrack = player.kind === "track" && src?.songId === track?.id;
  const position = onThisTrack ? player.position : null;
  const loop = onThisTrack && src.loop ? { start: src.loop.start, end: src.loop.end } : null;

  // The bar can change the stem too. Mirror it, or the switch here ends up
  // claiming Full while the speakers are playing the vocal.
  useEffect(() => {
    if (onThisTrack && src.stem && src.stem !== stem) setStem(src.stem);
  }, [onThisTrack, src?.stem]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!track) return undefined;
    let live = true;
    setSections([]);
    setCandidates([]);
    setOthers([]);
    api.getSections(track.id)
      .then((d) => { if (live) setSections(d.sections || []); })
      .catch((e) => { if (live) setError(e.message); });
    // This track's pairs on the side it is looked at as (all of them, for the
    // section table's PAIRS), and the best few on the other side.
    const mine = { limit: PAIR_LIMIT, maxPerSong: 0 };
    const theirs = { limit: PARTNER_ROWS, maxPerSong: 0 };
    if (role === "instrumental") { mine.instSongId = track.id; theirs.vocalSongId = track.id; }
    else { mine.vocalSongId = track.id; theirs.instSongId = track.id; }
    api.getMashups(mine).then((d) => { if (live) setCandidates(d.candidates || []); })
      .catch(() => { if (live) setCandidates([]); });
    api.getMashups(theirs).then((d) => { if (live) setOthers(d.candidates || []); })
      .catch(() => { if (live) setOthers([]); });
    return () => { live = false; };
  }, [track, role]);

  // Walking to a partner replaces the whole screen without unmounting it, so
  // start at the top rather than halfway down a page you have not seen.
  useEffect(() => { mainRef.current?.scrollTo({ top: 0 }); }, [track?.id]);

  // This screen publishes no status of its own; the rail's pipeline readout
  // already says what is processing.
  useEffect(() => { onStatus(null); }, [track, onStatus]);

  // Energy as this library's own rank, 0–10 — the raw number has no meaning on
  // its own (the Library's Energy column reads the same).
  const energy10 = useMemo(() => {
    const mine = track?.features?.full?.energy;
    if (mine == null) return null;
    const all = tracks.map((t) => t?.features?.full?.energy).filter((v) => v != null).sort((a, b) => a - b);
    if (all.length < 2) return null;
    return (all.filter((v) => v < mine).length / (all.length - 1)) * 10;
  }, [track, tracks]);

  const pairsBySection = useMemo(() => {
    const out = {};
    const field = role === "instrumental" ? "inst_section_idx" : "vocal_section_idx";
    for (const c of candidates) {
      const i = c[field];
      if (i != null) out[i] = (out[i] || 0) + 1;
    }
    return out;
  }, [candidates, role]);

  // Sections loop a WINDOW on the whole stem file, so the playhead can still be
  // dragged anywhere in the record.
  const playSection = (s) => {
    player.toggle({
      kind: "track", songId: track.id, stem,
      loop: { start: s.start_sec, end: s.end_sec },
      detailKey: `sec:${s.section_index}`,
      title: `${s.label || "section"} · ${track.title}`,
      subtitle: `${STEM_LABEL[stem]} · ${fmtDur(s.end_sec - s.start_sec)}`,
      stems: track.stems, duration: track.duration_secs,
    });
  };

  // The hero ▶ (and space): loop the track's hook — the 16 bars worth
  // previewing — or play the record from the top when it has none yet.
  const playHook = async () => {
    if (onThisTrack && player.playing) { player.pause(); return; }
    if (onThisTrack) { player.resume(); return; }
    const base = {
      kind: "track", songId: track.id, stem, detailKey: "whole",
      title: track.title, subtitle: [STEM_LABEL[stem], track.artist].filter(Boolean).join(" · "),
      duration: track.duration_secs, stems: track.stems,
    };
    let hook = null;
    try {
      const h = await api.getHook(track.id, (track.mash?.vocal_coverage ?? 0) >= 0.3 ? "vocal" : "bed");
      if (h?.hook_end > h?.hook_start) hook = { start: h.hook_start, end: h.hook_end };
    } catch { /* not analysed yet */ }
    player.play(hook ? { ...base, loop: hook, detailKey: "hook" } : base);
  };

  // Space plays / pauses here (esc goes back): the dock's own keys still work
  // beside it, except space.
  useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT"
        || el.isContentEditable)) return;
      if (e.key === "Escape") { e.preventDefault(); onBack(); }
      else if (e.code === "Space" && !e.metaKey && !e.ctrlKey) { e.preventDefault(); playHook(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Two controls for one fact, so they have to agree: the switch here moves
  // the live audio when the player is on this track, and mirrors the bar's.
  const pickStem = (id) => {
    setStem(id);
    if (onThisTrack) player.switchStem(id);
  };

  // Click or drag on the structure. On an idle track this STARTS it there.
  const seekTo = (secs) => {
    if (onThisTrack) { player.seek(secs); return; }
    player.play({
      kind: "track", songId: track.id, stem, startAt: secs, detailKey: "whole",
      title: track.title, subtitle: [STEM_LABEL[stem], track.artist].filter(Boolean).join(" · "),
      duration: track.duration_secs, stems: track.stems,
    });
  };

  // Opening a partner flips which side of a pair you are asking about: you got
  // here looking at this track as the vocal, so the partner is a bed.
  const openPartner = (id, asRole) => {
    onRole(asRole);
    onOpenTrack(id);
  };

  const rerun = async (label, fn) => {
    try {
      await fn(track.id);
      toast(`Queued: ${label} for ${track.title}`,
        onOpenQueue ? { label: "Open queue", onClick: onOpenQueue } : null);
      onChanged?.();
    } catch (e) { toast(`Could not queue ${label}: ${e.message}`); }
  };
  const fixBpm = async (bpm) => {
    try {
      await api.correctFeatures(track.id, { bpm: Math.round(bpm * 10) / 10 });
      notifyPairsInputChanged();
      toast(`${track.title}: ${bpm.toFixed(1)} BPM — its pairs are stale until a re-score`,
        { label: "Re-score now", onClick: requestRescore });
      onChanged?.();
    } catch (e) { toast(`Could not correct the tempo: ${e.message}`); }
  };

  if (!track) return null;

  const f = track.features?.full || {};
  const attrs = track.attrs || {};
  const lowKey = keyLooksOff(f);
  const status = statusKeyOf(track, running);
  const sepTag = track.stems?.separator || null;
  const star = ratings.bySong[track.id] ?? null;
  const myGroups = (groups?.groupsOf(track.id) || []).map((id) => groups.byId(id)).filter(Boolean);
  const vox = track.mash?.vocal_coverage;
  const tags = (v) => (Array.isArray(v) ? v.slice(0, 2).map((x) => x.label) : []);
  const genreTags = [track.genre, ...tags(attrs.styles)].filter(Boolean).slice(0, 3);
  const moodTags = tags(attrs.moodtheme);
  const best = track.mash?.best_pct != null ? Math.round(track.mash.best_pct * 100) : null;
  const sectionPlaying = loop
    ? (sections.find((s) => Math.abs(s.start_sec - loop.start) < 0.01
                         && Math.abs(s.end_sec - loop.end) < 0.01)?.section_index ?? null)
    : null;
  const totalBars = sections.reduce((n, s) => n + (s.bar_count || 0), 0);
  const detailAttrs = attributes
    ? attributes.visibility.detail.map((id) => attributes.byId[id]).filter(Boolean) : [];
  const partnerRows = [
    ...(role === "instrumental" ? others : candidates).slice(0, PARTNER_ROWS)
      .map((c) => ({ c, side: "VOX", partner: [c.inst_song_id, c.inst_title, c.inst_artist, c.inst_camelot], as: "instrumental" })),
    ...(role === "instrumental" ? candidates : others).slice(0, PARTNER_ROWS)
      .map((c) => ({ c, side: "BED", partner: [c.vocal_song_id, c.vocal_title, c.vocal_artist, c.vocal_camelot], as: "vocal" })),
  ];
  const qual = (id) => (attrs[id] != null ? `${Math.round(attrs[id] * 100)}%` : null);
  const kv = (k, v, tone = null) => (
    <div key={k} className="dt-kv"><span className="muted">{k}</span>
      <span className="mono" style={tone ? { color: tone } : undefined}>{v ?? "—"}</span></div>
  );

  return (
    <div className="detail" ref={mainRef}>
      <div className="dt-crumb">
        <button className="ui-btn" onClick={onBack} title="Back to Library (esc)">← Library</button>
        <span className="faint">/</span>
        <span className="dt-crumbtitle">{track.title}</span>
        <span className="grow" />
        <span className="mono faint small">space play · esc back</span>
      </div>

      <div className="dt-hero">
        <div className="dt-art">
          <TrackArt id={track.id} thumbnail={track.thumbnail} className="dt-artimg" />
          <button className="dt-artplay" onClick={playHook} aria-label="Play the hook"
            title="Play the hook (space)">{onThisTrack && player.playing ? "❚❚" : "▶"}</button>
        </div>
        <div className="dt-herotext">
          <div className="dt-titlerow">
            <h1>{track.title}</h1>
            <span className="dt-artist">{track.artist}</span>
            <span className="dt-src mono" title="Where the audio came from">{sourceOf(track.source_url)}</span>
            <StatusPill status={STATUS_PILL[status]} label={STATUS_LABEL[status]}
              title={track.last_error || STATUS_LABEL[status]} />
            <StarRating value={star} size={13}
              title={star ? `Best pairing rated ${star} of 5` : "No pairing of this track has been rated"} />
            {myGroups.map((g) => (
              <span key={g.id} className="dt-crate" title="A crate this track is in">▨ {g.name}</span>
            ))}
          </div>
          <div className="dt-stats">
            <Stat label="BPM" tip="Detected tempo"
              sub={f.bpm_confidence != null ? `confidence ${Math.round(f.bpm_confidence * 100)}%` : f.bpm == null ? "not analysed" : null}>
              <Num v={f.bpm != null ? f.bpm.toFixed(1) : null} unit="BPM" />
            </Stat>
            <Stat label="Key" tip="Camelot key"
              sub={f.key_confidence != null ? (lowKey ? "⚠ low confidence" : `confidence ${f.key_confidence.toFixed(3)}`) : "not analysed"}
              subTone={lowKey ? "var(--star)" : null}>
              <KeyChip camelot={f.camelot} />
            </Stat>
            <Stat label="Energy" tip="Overall energy, ranked against this library">
              <Num v={energy10 != null ? energy10.toFixed(1) : null} unit="/ 10" />
            </Stat>
            <Stat label="Loudness" tip="Integrated loudness (EBU R128)" sub="integrated">
              <Num v={(attrs.lufs ?? f.lufs) != null ? Number(attrs.lufs ?? f.lufs).toFixed(1) : null} unit="LUFS" />
            </Stat>
            <Stat label="Vocal share" tip="Share of the track that is sung"
              sub={vox != null ? (vox >= 0.45 ? "vocal-forward" : "bed-leaning") : "needs stems"}>
              <Num v={vox != null ? Math.round(vox * 100) : null} unit="%" color="var(--vocal-label)" />
            </Stat>
            <Stat label="Genre / style" tip="SoundCloud genre and the Discogs styles it leads with">
              {genreTags.length ? genreTags.map((g) => <span key={g} className="dt-tag">{g}</span>)
                : <span className="faint">—</span>}
            </Stat>
            <Stat label="Mood" tip="Top moods / themes">
              {moodTags.length ? moodTags.map((g) => <span key={g} className="dt-tag">{g}</span>)
                : <span className="faint">—</span>}
            </Stat>
            <Stat label="Year" tip="Release year" sub={track.release_year ? null : "not on upload"}>
              <Num v={track.release_year ? fmtYear(track.release_year) : null} />
            </Stat>
            <Stat label="Best pair" tip="This track's best pairing, as a percentile of every scored pair">
              <Num v={best} unit={best != null ? "%" : ""} color={best != null ? compatColor(best) : null} />
            </Stat>
            <Stat label="Partners" tip="Pairs as vocal and as bed" sub="as vocal · as bed">
              <Num v={track.mash ? `${track.mash.as_vocal} · ${track.mash.as_bed}` : null} unit="V · B" />
            </Stat>
          </div>
        </div>
      </div>

      {(error || player.error) && <div className="error-text dt-pad">{error || player.error}</div>}

      <div className="dt-struct">
        <div className="dt-structhead">
          <b>Structure</b>
          <span className="mono muted small">
            {sections.length ? `${sections.length} sections · ${totalBars ? `${Math.round(totalBars)} bars · ` : ""}${fmtDur(track.duration_secs)}`
              : "not analysed"}
          </span>
          <span className="grow" />
          <span className="dt-legend"><i className="vox" />Vocal</span>
          <span className="dt-legend"><i className="bed" />Bed</span>
          <span className="dt-legend"><i className="nrg" />Energy</span>
          <div className="dt-seg" role="group" aria-label="Waveform stem">
            {STEMS.map(([id, label]) => (
              <button key={id} className={`${id}${stem === id ? " on" : ""}`} aria-pressed={stem === id}
                onClick={() => pickStem(id)} disabled={id !== "full" && !track.stems?.[id]}
                title={id !== "full" && !track.stems?.[id] ? "That stem has not been separated yet"
                  : `${label} — switches live while playing`}>{label}</button>
            ))}
          </div>
        </div>
        <StructureStrip songId={track.id} sections={sections}
          duration={track.duration_secs} loop={loop} position={position}
          stem={stem} onSeek={seekTo} onPickSection={playSection} keyCamelot={f.camelot} />
      </div>

      <SectionTable sections={sections} pairsBySection={pairsBySection}
        playingIndex={sectionPlaying} onPlay={playSection}
        onMatch={onOpenSegments
          ? (s, asRole) => onOpenSegments({ songId: track.id, sectionIndex: s.section_index, role: asRole })
          : null}
        onSaveLine={async (s, text) => {
          try {
            await api.saveSectionLine(track.id, s, text);
            setSections((ss) => ss.map((x) => (x.section_index === s.section_index
              ? { ...x, line: text || null } : x)));
            onLinesChanged?.();
          } catch (e) { toast(`Could not save the line: ${e.message}`); }
        }} />

      <div className="dt-groups">
        <Disclosure id="attrs" storageKey={GROUPS_KEY} title="Analysis attributes"
          meta={`${detailAttrs.length} shown`}>
          <div className="dt-kvgrid">
            {detailAttrs.map((a) => (
              <div key={a.id} className="dt-kv" title={a.description}>
                <span className="muted">{a.label}</span>
                {a.kind === "top" && Array.isArray(attrs[a.id])
                  ? <span className="dt-chips">{attrs[a.id].slice(0, 3).map((x) => (
                      <span key={x.label} className="dt-tag">{x.label} {Math.round(x.p * 100)}%</span>))}</span>
                  : <span className="mono">{fmtAttr(a, attrs[a.id])}</span>}
              </div>
            ))}
            {!detailAttrs.length && <span className="muted small">None chosen yet.</span>}
          </div>
          <div className="dt-actions">
            {onChooseAttributes && (
              <button className="ui-btn" onClick={onChooseAttributes}
                title="Pick what shows here in Analysis (the Detail switches)">Choose attributes…</button>
            )}
            {onCompareWith && (
              <button className="ui-btn" onClick={() => onCompareWith(track.id)}
                title="Compare this track with another in Analysis">Compare with…</button>
            )}
          </div>
        </Disclosure>

        <Disclosure id="stems" storageKey={GROUPS_KEY} title="Stems & quality"
          meta={track.stems?.vocals || track.stems?.instrumental
            ? [sepTag ? sepTag.split(":").slice(0, 2).join(" ") : null,
              track.stems?.drums ? "4 stems" : "2 stems",
              qual("vocal_quality") ? `vocal quality ${qual("vocal_quality")}` : null].filter(Boolean).join(" · ")
            : "not separated"}>
          <div className="dt-kvgrid">
            {kv("Separator", sepTag)}
            {kv("Vocal stem", track.stems?.vocals ? "✓ separated" : "—", track.stems?.vocals ? "var(--ok)" : null)}
            {kv("Instrumental stem", track.stems?.instrumental ? "✓ separated" : "—", track.stems?.instrumental ? "var(--ok)" : null)}
            {track.stems?.drums && kv("Drums / bass / other", "✓ four-stem", "var(--ok)")}
            {kv("Vocal stem quality", qual("vocal_quality"))}
            {kv("Bed stem quality", qual("bed_quality"))}
            {kv("Residual vocal in bed", qual("residual_vocal"))}
          </div>
        </Disclosure>

        <Disclosure id="src" storageKey={GROUPS_KEY} title="Source & provenance"
          meta={sourceOf(track.source_url)}>
          {/* Where the audio came from — a substituted download used to be
              invisible: the link said SoundCloud while the stems were cut from
              a YouTube upload. */}
          <AudioSource track={track} onPick={() => setPicking((v) => !v)} onChanged={onChanged} />
          {picking && (
            <AudioSourcePicker track={track} onClose={() => setPicking(false)} onChanged={onChanged} />
          )}
          <div className="dt-kvgrid">
            {kv("Added", String(track.created_at || "").slice(0, 10) || null)}
            {kv("Plays", track.plays ? track.plays.toLocaleString() : null)}
          </div>
          {track.last_error && <span className="dt-note">{track.last_error}</span>}
          <div className="dt-actions">
            {track.source_url && (
              <a className="ui-btn" href={track.source_url} target="_blank" rel="noreferrer">Open source page ↗</a>
            )}
            {/* Separated stems are good; an official acapella or instrumental
                is better. One click to look for one. */}
            {onDiscover && (
              <>
                <button className="ui-btn" onClick={() => onDiscover(`${track.artist || ""} ${track.title}`.trim(), "acapella")}
                  title="Search SoundCloud for an acapella of this track">Dig: acapella ↗</button>
                <button className="ui-btn" onClick={() => onDiscover(`${track.artist || ""} ${track.title}`.trim(), "instrumental")}
                  title="Search SoundCloud for an instrumental of this track">Dig: instrumental ↗</button>
              </>
            )}
          </div>
        </Disclosure>

        <Disclosure id="partners" storageKey={GROUPS_KEY} title="Partners"
          meta={track.mash ? `${track.mash.as_vocal} as vocal · ${track.mash.as_bed} as bed` : "none yet"}>
          {partnerRows.length ? partnerRows.map(({ c, side, partner, as }) => {
            const [pid, ptitle, partist, pkey] = partner;
            const tier = tierOf(c);
            return (
              <div key={keyOf(c)} className="dt-partner">
                <span className={`mono dt-role ${side === "VOX" ? "vox" : "bed"}`}
                  title={side === "VOX" ? "This track as the vocal — the partner is the bed" : "This track as the bed — the partner is the vocal"}>
                  {side}</span>
                <button className="dt-partnername" onClick={() => openPartner(pid, as)}
                  title={`Open ${ptitle} — as the ${as === "vocal" ? "vocal" : "bed"}`}>
                  {ptitle} <span className="muted">{partist}</span>
                </button>
                <KeyChip camelot={pkey} />
                <span className="mono" style={{ color: tier.color }}>{pctOf(c)}%</span>
                {onPlayPair && (
                  <button className="icon-btn" onClick={() => onPlayPair(c)} title="Loop this pair">▶</button>
                )}
                <button className="icon-btn" onClick={() => onStudio(c)} title="Open in Studio">◫</button>
              </div>
            );
          }) : <span className="dt-note">Pairs appear once stems and analysis finish, and the library is scored.</span>}
        </Disclosure>

        <Disclosure id="actions" storageKey={GROUPS_KEY} title="Actions"
          meta="Re-run stages, fix BPM or key">
          <div className="dt-actions">
            <button className="ui-btn" onClick={() => rerun("stems", api.startSeparate)}
              title="Queue the Stems stage again">Re-run stems</button>
            <button className="ui-btn" onClick={() => rerun("analysis", api.startAnalyze)}
              title="Queue the Analyse stage again">Re-analyse</button>
            <button className="ui-btn" onClick={() => rerun("structure", api.startStructure)}
              title="Queue the Structure stage again">Re-run structure</button>
            {String(track.status || "").startsWith("error") && (
              <button className="ui-btn" onClick={() => rerun("the pipeline", api.processTrack)}
                title="Retry from the stage that failed">Retry the pipeline</button>
            )}
            {f.bpm != null && (
              <>
                <button className="ui-btn" onClick={() => fixBpm(f.bpm * 2)}
                  title="Fix a half-time detection">BPM ×2 → {(f.bpm * 2).toFixed(1)}</button>
                <button className="ui-btn" onClick={() => fixBpm(f.bpm / 2)}
                  title="Fix a double-time detection">BPM ÷2 → {(f.bpm / 2).toFixed(1)}</button>
              </>
            )}
            <button className="ui-btn" onClick={() => setEditing((v) => !v)}
              title="Override the detected BPM or key, or change the source link">Edit key…</button>
          </div>
          {editing && (
            <TrackEditor track={track}
              onSaved={() => { setEditing(false); onChanged?.(); }}
              onCancel={() => setEditing(false)} />
          )}
        </Disclosure>
      </div>
    </div>
  );
}
