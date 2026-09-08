import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import {
  EFFORT_TONE, keyOf, pctOf, spanLabel, tierOf,
} from "./pairs/pairModel";
import { bpmTag, keyRel } from "../theme";

// "What does this track go with." The same rows the dock ranks, drawn compact
// because here you already know one side of every pair — it is the track you
// are looking at.

export function PartnersRail({ candidates, role, ratings, judgedCount,
                               onPlay, playingKey, onStudio }) {
  const other = role === "instrumental" ? "vocals" : "beds";
  return (
    <aside className="partners">
      <div className="partners-head">
        <span className="partners-title">Best partners</span>
        <span className="mono partners-sub">
          {candidates.length
            ? `this ${role === "instrumental" ? "bed" : "vocal"} over `
              + `${new Set(candidates.map((c) => (role === "instrumental"
                ? c.vocal_song_id : c.inst_song_id))).size} ${other}`
            : "none scored yet"}
        </span>
      </div>

      <div className="partners-list">
        {candidates.length === 0 && (
          <div className="pd-msg">
            Nothing scored against this track.
            <span className="hint">
              It needs stems and structure before the matcher can place it, and
              then a “Score library” run.
            </span>
          </div>
        )}
        {candidates.map((c) => {
          const k = keyOf(c);
          const partnerIsBed = role !== "instrumental";
          const songId = partnerIsBed ? c.inst_song_id : c.vocal_song_id;
          const title = partnerIsBed ? c.inst_title : c.vocal_title;
          const span = partnerIsBed
            ? spanLabel(c.inst_section_label, c.inst_section_start, c.inst_section_end)
            : spanLabel(c.vocal_section_label, c.vocal_section_start, c.vocal_section_end);
          const tier = tierOf(c);
          const rel = keyRel(c.vocal_camelot, c.inst_camelot);
          return (
            <div key={k} className={`partner${playingKey === k ? " playing" : ""}`}
              onClick={() => onPlay(c)}
              onDoubleClick={() => onStudio(c)}
              title="Click to loop this pairing · double-click to open it in Studio">
              <div className="partner-top">
                <div className="partner-score mono" style={{ color: tier.color }}>
                  {pctOf(c)}
                </div>
                <TrackArt id={songId} className="pc-art" />
                <div className="pc-sidetext">
                  <div className="pc-title">{title}</div>
                  <div className="pc-span mono">{span}</div>
                </div>
                {c.effort_label && (
                  <span className={`pc-effort mono ${EFFORT_TONE[c.effort_label] || ""}`}
                    title={c.effort_reason || ""}>{c.effort_label}</span>
                )}
              </div>
              <div className="partner-tags">
                <span className="pc-tag mono"
                  style={{ background: rel.tagBg, color: rel.tagColor }}
                  title={rel.text}>{rel.tag}</span>
                <span className="pc-tag mono neutral">
                  {bpmTag(c.vocal_bpm, c.inst_bpm)}
                </span>
                <span style={{ marginLeft: "auto" }}
                  onClick={(e) => e.stopPropagation()}>
                  <StarRating value={ratings.ratingOf(c)}
                    onRate={(n) => ratings.rate(c, n)} size={13} />
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="partners-foot">
        Ranked by the scorer this library is using. Rating a pair 1–5 feeds the
        learned scorer; it has {judgedCount} judgement{judgedCount === 1 ? "" : "s"} so far.
      </div>
    </aside>
  );
}
