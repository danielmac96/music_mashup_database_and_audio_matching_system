import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import {
  camelotColor, fmtDur, fmtPlays, fmtYear, pipelineDots, playsColor, yearColor,
} from "../theme";

// The library table.
//
// Energy and section count are deliberately NOT columns. They cost the title
// column width it cannot spare at this frame size, and both are one click away
// on the track's own screen — where there is room to show them properly.

const COLS = "26px 30px minmax(170px,1fr) 56px 40px 52px 52px 44px 56px 70px 44px";

const HEADS = ["", "", "TITLE / ARTIST", "GENRE", "YEAR", "PLAYS", "BPM",
               "KEY", "PIPE", "RATING", "TIME"];

// Four stages, in the order they run: downloaded, analysed, sections, stems.
// There are no per-stage columns on `songs` — the truth is assembled from the
// stems, features and section-count the list already returns.
const DOT_ORDER = ["dl", "analyse", "structure", "stems"];
const DOT_TITLE = {
  dl: "downloaded", analyse: "analysed", structure: "sections detected",
  stems: "stems separated",
};

export function TrackTable({ tracks, selectedId, onSelect, onOpen, onPlay,
                             runningKind = () => null, playingId = null,
                             menuId = null, onMenu = () => {},
                             renderMenu = () => null }) {
  return (
    <div className="track-table">
      <div className="tt-head" style={{ gridTemplateColumns: COLS }}>
        {HEADS.map((h, i) => (
          <div key={i} className={i === HEADS.length - 1 ? "right" : ""}>{h}</div>
        ))}
      </div>
      <div className="tt-body">
        {tracks.map((t) => (
          <TrackRow key={t.id} t={t}
            selected={selectedId === t.id}
            playing={playingId === t.id}
            running={runningKind(t)}
            menuOpen={menuId === t.id}
            onMenu={onMenu} renderMenu={renderMenu}
            onSelect={onSelect} onOpen={onOpen} onPlay={onPlay} />
        ))}
        {tracks.length === 0 && (
          <div className="tt-empty">
            Nothing matches these filters. <span className="faint">
              The bar above says how many of the library are shown.</span>
          </div>
        )}
      </div>
    </div>
  );
}

function TrackRow({ t, selected, playing, running, menuOpen, onMenu,
                    renderMenu, onSelect, onOpen, onPlay }) {
  const f = t.features?.full || {};
  const dots = pipelineDots(t, running);
  const cam = f.camelot;

  return (
    <div
      className={`tt-row${selected ? " selected" : ""}${playing ? " playing" : ""}`}
      style={{ gridTemplateColumns: COLS }}
      onClick={() => onSelect(t.id)}
      onDoubleClick={() => onOpen(t.id)}
      title="Click to scope the pair list">
      <button className="tt-play" title={playing ? "Stop" : "Play"}
        onClick={(e) => { e.stopPropagation(); onPlay(t); }}>
        {playing ? "❚❚" : "▶"}
      </button>

      <TrackArt id={t.id} thumbnail={t.thumbnail} className="tt-art" />

      {/* The title is the link into the track's own screen. The row itself
          still belongs to the dock — clicking it scopes the pair list — so
          navigation needs a surface of its own, and it has to look like one.
          stopPropagation is what stops one click doing both. */}
      <button className="tt-name" title={`Open ${t.title}`}
        onClick={(e) => { e.stopPropagation(); onOpen(t.id); }}>
        <span className="tt-title">
          {/* The chevron sits OUTSIDE the ellipsised span. Inside it, a title
              long enough to truncate — which is most of them at this column
              width — eats the only affordance the row has. */}
          <span className="tt-text">{t.title}</span>
          <span className="tt-go">›</span>
        </span>
        <span className="tt-artist">{t.artist || "—"}</span>
      </button>

      <div className="tt-cell">
        {t.genre
          ? <span className="tt-genre mono" title={t.genre}>{t.genre}</span>
          : <span className="mono" style={{ color: "var(--faint-2)" }}>—</span>}
      </div>

      <div className="tt-num mono" style={{ color: yearColor(t.release_year) }}>
        {fmtYear(t.release_year)}
      </div>

      <div className="tt-num mono" style={{ color: playsColor(t.plays) }}
        title={t.plays ? `${t.plays.toLocaleString()} plays` : "no play count"}>
        {fmtPlays(t.plays)}
      </div>

      <div className="tt-num mono">
        {f.bpm != null ? Math.round(f.bpm) : <span className="faint">—</span>}
      </div>

      <div className="tt-cell">
        {cam
          ? <span className="tt-key mono"
              style={{ background: camelotColor(cam) }}
              title={`${f.key || ""} ${f.mode || ""}`.trim()}>{cam}</span>
          : <span className="mono" style={{ color: "var(--faint-2)" }}>—</span>}
      </div>

      <div className="tt-dots-cell">
        <button className={`tt-dots${menuOpen ? " open" : ""}`}
          title="Pipeline — click to run a stage, correct or delete"
          onClick={(e) => { e.stopPropagation(); onMenu(menuOpen ? null : t.id); }}>
          {DOT_ORDER.map((k) => (
            <span key={k} style={{ background: dots[k] }} title={DOT_TITLE[k]} />
          ))}
        </button>
        {menuOpen && renderMenu(t)}
      </div>

      <div className="tt-cell">
        <StarRating value={t.rating} />
      </div>

      <div className="tt-num mono right">
        {fmtDur(t.duration_secs)}
      </div>
    </div>
  );
}
