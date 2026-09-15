import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import { SortHead } from "./SortHead";
import { audioSubstitution } from "../sources";
import {
  camelotColor, fmtDur, fmtPlays, fmtYear, pipelineDots, playsColor, yearColor,
} from "../theme";

// The library table.
//
// Energy and section count are deliberately NOT columns. They cost the title
// column width it cannot spare at this frame size, and both are one click away
// on the track's own screen — where there is room to show them properly.

const COLS = "26px 30px minmax(170px,1fr) 56px 40px 52px 52px 44px 56px 70px 44px";

// The sort keys are useLibraryFilters' own — the state already existed and was
// reachable only through the dropdown in the filter bar, which still shares it.
// PIPE has no key: it is four assembled booleans, not a value to order by.
const HEADS = [
  { label: "" },
  { label: "" },
  { label: "TITLE", key: "title", also: { label: "ARTIST", key: "artist" } },
  { label: "GENRE", key: "genre" },
  { label: "YEAR", key: "year", numeric: true },
  { label: "PLAYS", key: "plays", numeric: true },
  { label: "BPM", key: "bpm", numeric: true },
  { label: "KEY", key: "key" },
  { label: "PIPE" },
  { label: "RATING", key: "rating", numeric: true },
  { label: "TIME", key: "duration", numeric: true, right: true },
];

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
                             sort = null, onSort = null,
                             menuId = null, onMenu = () => {},
                             renderMenu = () => null }) {
  // A header click sets the PRIMARY key. The filter bar's two-level sort keeps
  // its secondary, which is what makes "artist, then year" reachable there and
  // still one click away here.
  const sortable = Boolean(onSort);
  const key = sort?.primary ?? "";
  const dir = sort?.primaryDir ?? "desc";
  const setKey = (p) => onSort?.({ ...(sort || {}), primary: p.sort,
                                   primaryDir: p.dir });

  return (
    <div className="track-table">
      <div className="tt-head" style={{ gridTemplateColumns: COLS }}>
        {HEADS.map((h, i) => (
          <div key={i} className={`tt-h${h.right ? " right" : ""}`}>
            <SortHead label={h.label} sortKey={sortable ? h.key : null}
              sort={key} dir={dir} onSort={setKey} numeric={h.numeric} />
            {h.also && (
              <>
                <span className="sh-sep">/</span>
                <SortHead label={h.also.label} sortKey={sortable ? h.also.key : null}
                  sort={key} dir={dir} onSort={setKey} />
              </>
            )}
          </div>
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
  // Audio that is not the link it was imported from (a YouTube substitute, or
  // your own pick) is marked on the row — the stems are cut from that file.
  const sub = audioSubstitution(t);

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
          {sub && (
            <span className={`tt-src mono ${sub.kind}`} title={`${sub.label}. ${sub.title}`}>
              {sub.chip}
            </span>
          )}
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
