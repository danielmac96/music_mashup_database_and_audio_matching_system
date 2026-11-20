import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api";
import { track as trackEvent } from "../../events";
import { toast } from "../../toast";
import { KeyChip, TierBadge } from "../ui/primitives";
import { StarRating } from "../StarRating";
import { keyOf, pctOf } from "../pairs/pairModel";
import { compatColor } from "../../theme";
import { OverlayCanvas } from "./OverlayCanvas";
import {
  auditionCandidate, auditionKey, phraseErrorsMs, placementAt,
  placementSummary, stepOffset,
} from "./overlayModel";

// Track detail's ghost-slot overlay — "this vocal over that bed" (readme §3).
//
// The picker lists this track's ranked partners on the other side: the same
// listing the pair dock reads for this track, Drop-ins first
// (order=dropin), with a search box. Choosing one loads the overlay
// (GET /api/tracks/{bed}/overlay): every landing of the vocal's phrase on the
// bed section, priced by the scorer. Dragging the vocal, or Shift+←/→, moves
// it to another landing and the score and the fit band follow — they are that
// landing's priced row, never a client-side estimate.
//
// Sound goes through the ONE player (usePlayer → useHookAudition): space
// auditions the bed and the conformed vocal from the chosen bar, V / B solo,
// O swaps in the bed's original vocal for an A/B. Rating writes the normal
// pair_feedback row for the four ids (useRatings.rate) and its `rate` event.
// Keys are taken only while the panel has focus.

const PARTNERS = 40;

function partnerOf(row, asBed) {
  return asBed
    ? { id: row.vocal_song_id, title: row.vocal_title, artist: row.vocal_artist,
        camelot: row.vocal_camelot, bpm: row.vocal_bpm }
    : { id: row.inst_song_id, title: row.inst_title, artist: row.inst_artist,
        camelot: row.inst_camelot, bpm: row.inst_bpm };
}

export function OverlayPanel({ track, role, player, ratings, onStudio }) {
  const asBed = role === "instrumental";
  const [query, setQuery] = useState("");
  const [partners, setPartners] = useState([]);
  const [sel, setSel] = useState(0);
  const [payload, setPayload] = useState(null);
  const [phrase, setPhrase] = useState(0);
  const [offset, setOffset] = useState(0);
  const [fromBed, setFromBed] = useState(null);
  const [ghost, setGhost] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const panelRef = useRef(null);

  // The partners: this track's ranked pairs on the other side, Drop-ins first.
  useEffect(() => {
    if (!track) return undefined;
    let live = true;
    const t = setTimeout(() => {
      api.getMashups({
        [asBed ? "instSongId" : "vocalSongId"]: track.id,
        comboType: "vocal_over_instrumental", order: "dropin", limit: PARTNERS,
        maxPerSong: 1, search: query.trim(),
      }).then((d) => {
        if (!live) return;
        const rows = (d.candidates || []).filter((r) => r.vocal_section_idx != null
          && r.inst_section_idx != null);
        setPartners(rows);
        setSel(0);
      }).catch(() => { if (live) setPartners([]); });
    }, query ? 250 : 0);
    return () => { live = false; clearTimeout(t); };
  }, [track, asBed, query]);

  const pick = partners[Math.min(sel, partners.length - 1)] || null;
  // A partner (or phrase) change loads that section pair's overlay.
  useEffect(() => { setPhrase(0); }, [pick ? keyOf(pick) : null]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!pick) { setPayload(null); return undefined; }
    let live = true;
    setLoading(true);
    setError(null);
    api.getOverlay(pick.inst_song_id, { vocalId: pick.vocal_song_id,
      vs: pick.vocal_section_idx, bs: pick.inst_section_idx, phrase })
      .then((d) => {
        if (!live) return;
        setPayload(d);
        setOffset(d.current ?? 0);
        setFromBed(null);
      })
      .catch((e) => { if (live) { setPayload(null); setError(e.message); } })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [pick ? keyOf(pick) : null, phrase]); // eslint-disable-line react-hooks/exhaustive-deps

  const row = useMemo(() => placementAt(payload, offset), [payload, offset]);
  const key = auditionKey(row, fromBed);
  const playing = player.isPlaying(`pair:${key}`) && player.playing;
  const onPair = player.kind === "pair" && String(player.source?.key || "").startsWith(
    row ? `${row.vocal_song_id}:${row.inst_song_id}:${row.vocal_section_idx}:${row.inst_section_idx}@` : "\u0000");
  // The playhead on the bed's timeline: the loop clock is display seconds
  // from the clips' start; the bed runs at `stretch` × speed under it.
  const playhead = onPair && row
    ? (player.source.candidate?.bed_play_start ?? 0) + player.position * (payload?.stretch || 1)
    : null;

  const audition = useCallback((at = fromBed) => {
    if (!row) return;
    const cand = auditionCandidate(payload, row, at, true);
    player.toggle({
      kind: "pair", key: auditionKey(row, at), candidate: cand,
      title: row.vocal_title, subtitle: `over ${row.inst_title} · overlay`,
      context: "track_detail", rank: null,
    });
  }, [row, payload, player, fromBed]);

  // The ghost follows the player while this pair is armed.
  useEffect(() => { if (onPair) player.pair.setGhost?.(ghost); }, [ghost, onPair]); // eslint-disable-line react-hooks/exhaustive-deps
  // A moved vocal re-arms the audition at its new place.
  const wasPlaying = useRef(false);
  wasPlaying.current = onPair && player.playing;
  useEffect(() => {
    if (wasPlaying.current && row) audition(fromBed);
  }, [offset, phrase]); // eslint-disable-line react-hooks/exhaustive-deps

  const rate = (n) => {
    if (!row) return;
    ratings.rate(row, n, "track_detail");
    toast(`${n}★ — ${row.vocal_title} over ${row.inst_title}`);
  };

  const studio = () => { if (row) onStudio?.(row); };

  const flExport = async () => {
    if (!row) return;
    trackEvent("fl_export", row, { context: "track_detail" });
    try {
      const { job_id } = await api.startSessionExport(row.vocal_song_id, row.inst_song_id,
        { vocal: row.vocal_section_idx, inst: row.inst_section_idx }, null,
        { phrase: row.phrase ?? 0, offset_bars: row.offset_bars ?? 0 });
      toast("Rendering the FL session at this placement…");
      const poll = setInterval(async () => {
        try {
          const j = await api.getJob(job_id);
          if (j.status === "completed" || j.status === "failed") {
            clearInterval(poll);
            if (j.status === "completed") {
              const a = document.createElement("a");
              a.href = j.result?.archive_url || `/api/studio/session/${job_id}/archive`;
              a.download = "";
              document.body.appendChild(a); a.click(); a.remove();
              toast("FL session ready");
            } else toast(j.error || j.message || "FL export failed");
          }
        } catch { clearInterval(poll); }
      }, 1200);
    } catch (e) { toast(`Could not export: ${e.message}`); }
  };

  const onKeyDown = (e) => {
    const el = e.target;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); el.blur(); panelRef.current?.focus(); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    let handled = true;
    if (k === " ") audition();
    else if ((k === "ArrowLeft" || k === "ArrowRight") && e.shiftKey) {
      const next = stepOffset(payload, offset, k === "ArrowRight" ? 1 : -1);
      if (next != null) setOffset(next);
    } else if (k === "ArrowLeft" || k === "ArrowRight") {
      setSel((i) => Math.max(0, Math.min(partners.length - 1, i + (k === "ArrowRight" ? 1 : -1))));
    } else if (k === "[" || k === "]") {
      const n = payload?.phrases?.length || 1;
      setPhrase((p) => Math.max(0, Math.min(n - 1, p + (k === "]" ? 1 : -1))));
    } else if (k === "v" || k === "V") {
      player.pair.setStemMode(player.pair.stemMode === "vox" ? "both" : "vox");
    } else if (k === "b" || k === "B") {
      player.pair.setStemMode(player.pair.stemMode === "bed" ? "both" : "bed");
    } else if (k === "o" || k === "O") setGhost((g) => !g);
    else if (k >= "1" && k <= "5") rate(Number(k));
    else if (k === "Enter") studio();
    else if (k === "Escape") panelRef.current?.blur();
    else handled = false;
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  };

  if (!track) return null;
  const title = asBed ? "Try a vocal over this" : "Try over a bed";
  const star = row ? ratings.ratingOf(row) : null;
  const errs = phraseErrorsMs(payload, row);
  const onFrame = errs.length ? errs.filter((x) => x <= (payload?.frame_secs || 0.0464) * 1000).length : null;

  return (
    <section className="ov-panel" ref={panelRef} tabIndex={0} onKeyDown={onKeyDown}
      aria-label={`${title}: the overlay`}>
      <div className="ov-head">
        <b>{title}</b>
        <span className="muted small">
          {asBed ? "a partner's vocal, placed by bars over this track's bed"
            : "this vocal, placed by bars over a partner's bed"}
        </span>
        <span className="grow" />
        <span className="mono faint small">← → partner · ⇧← ⇧→ move · [ ] phrase · space play · V B solo · O original · 1–5 rate · ⏎ Studio</span>
      </div>

      <div className="ov-body">
        <div className="ov-picker">
          <input className="ov-input" placeholder="Search partners…" value={query}
            onChange={(e) => setQuery(e.target.value)} aria-label="Search partners" />
          <div className="ov-list" role="listbox" aria-label="Partners, Drop-ins first">
            {partners.map((r, i) => {
              const p = partnerOf(r, asBed);
              return (
                <button key={keyOf(r)} role="option" aria-selected={i === sel}
                  className={`ov-card${i === sel ? " on" : ""}`} onClick={() => setSel(i)}
                  title={`${p.title} — ${p.artist}`}>
                  <span className="ov-cardtitle">{p.title}</span>
                  <span className="ov-cardartist muted">{p.artist}</span>
                  <span className="ov-cardmeta">
                    <KeyChip camelot={p.camelot} />
                    <span className="mono small">{p.bpm ? p.bpm.toFixed(1) : "—"}</span>
                    {r.dropin && <span className="pc-dropin mono" title={`Drop-in — ${r.dropin_reason || ""}`}>⤓ Drop-in</span>}
                    <TierBadge tier={r.confidence_tier} />
                  </span>
                </button>
              );
            })}
            {!partners.length && (
              <span className="dt-note">No scored partners on this side yet{query ? " for that search" : ""}.</span>
            )}
          </div>
        </div>

        <div className="ov-stage">
          {error && <div className="error-text">{error}</div>}
          {!payload && !error && (
            <div className="dt-note">{loading ? "Placing the vocal…" : "Pick a partner to place its vocal."}</div>
          )}
          {payload && (
            <>
              <div className="ov-score">
                <span className="ov-pair">
                  <span className="ov-vox">{payload.vocal?.title}</span>
                  <span className="muted"> {payload.vocal_section?.label} over </span>
                  <span className="ov-bed">{payload.bed?.title}</span>
                  <span className="muted"> {payload.bed_section?.label}</span>
                </span>
                <span className="grow" />
                {row && (
                  <>
                    <span className="mono ov-total" style={{ color: compatColor(pctOf(row)) }}
                      title="This landing's score as a percentile of every stored pair — the scorer's own arithmetic for this placement">
                      {pctOf(row)}%</span>
                    <TierBadge tier={row.confidence_tier} />
                    {row.dropin && <span className="pc-dropin mono" title={row.dropin_reason}>⤓ Drop-in</span>}
                    {row.gated && <span className="ov-gated mono" title="The library run would not store this landing">gated</span>}
                    <StarRating value={star} onRate={rate} size={13} title="Rate this pairing (1–5)" />
                  </>
                )}
              </div>
              <OverlayCanvas payload={payload} row={row} offset={offset} ghostOn={ghost}
                fromBed={fromBed} playhead={playhead}
                onOffset={setOffset} onFromBed={setFromBed} />
              <div className="ov-readout">
                <span className="small">{placementSummary(payload, row)}</span>
                {row?.slot_reason && <span className="small muted">{row.slot_reason}</span>}
                {onFrame != null && (
                  <span className="small muted" title="Incoming phrase starts drawn within one 46 ms analysis frame of the bed's original phrase starts">
                    {onFrame} of {errs.length} phrase starts on the original's</span>
                )}
                {row?.window && (
                  <span className="small muted">
                    build {row.window.window_bars} bars: vocal bar {row.window.vocal_window_bar + 1} over bed bar {row.window.bed_window_bar + 1}
                  </span>
                )}
              </div>
              <div className="ov-controls">
                <button className="btn-primary" onClick={() => audition()}
                  title={fromBed != null ? "Play from the marked bar (space)" : "Play from the vocal's pickup (space)"}>
                  {playing ? "❚❚ Pause" : "▶ Play"}{fromBed != null ? " from bar" : ""}</button>
                <div className="ov-seg" role="group" aria-label="Solo">
                  {[["both", "Both"], ["vox", "V Vocal"], ["bed", "B Bed"]].map(([m, l]) => (
                    <button key={m} className={player.pair.stemMode === m ? "on" : ""}
                      aria-pressed={player.pair.stemMode === m}
                      onClick={() => player.pair.setStemMode(m)}>{l}</button>
                  ))}
                </div>
                <button className={`ui-btn${ghost ? " on" : ""}`} aria-pressed={ghost}
                  onClick={() => setGhost((g) => !g)}
                  title="O — hear (and show) the bed's original vocal instead of the incoming one">
                  O Original vocal</button>
                {payload.phrases?.length > 1 && (
                  <label className="ov-phrase small">Phrase
                    <select className="ov-select" value={phrase}
                      onChange={(e) => setPhrase(Number(e.target.value))}>
                      {payload.phrases.map((p) => (
                        <option key={p.index} value={p.index}>
                          {`${p.index + 1} · bar ${p.bar != null ? p.bar + 1 : "?"}${p.pickup_ms ? ` · pickup ${Math.round(p.pickup_ms)} ms` : ""}`}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <button className="ui-btn" onClick={() => {
                  const next = stepOffset(payload, offset, -1); if (next != null) setOffset(next);
                }} title="One landing earlier (Shift+←)">◂ Bar</button>
                <button className="ui-btn" onClick={() => {
                  const next = stepOffset(payload, offset, 1); if (next != null) setOffset(next);
                }} title="One landing later (Shift+→)">Bar ▸</button>
                {offset !== 0 && (
                  <button className="ui-btn" onClick={() => setOffset(0)}
                    title="Back to where the library placed it">↺ Library's landing</button>
                )}
                <span className="grow" />
                <button className="ui-btn" onClick={studio}
                  title="Open in Studio at this placement (⏎)">◫ Open in Studio</button>
                <button className="ui-btn" onClick={flExport}
                  title="Export an FL Studio session at this placement">⤓ FL export</button>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
