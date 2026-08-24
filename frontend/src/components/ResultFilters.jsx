import {
  SORTS, cratesIn, countriesIn, genresIn, kindOf,
} from "../hooks/useResultFilters";

// The filter/sort bar above a Discovery listing.
//
// Facets are kind-aware: sorting artists by play count or tracks by follower
// count would both be meaningless, so the controls follow the rows present.
// Everything here operates on rows already loaded — no control in this bar
// triggers a fetch.

export function ResultFilters({ items, filters, onChange, visibleCount, crateOf }) {
  const rows = items || [];
  if (!rows.length) return null;

  const kind = kindOf(rows);
  const sorts = SORTS[kind] || {};
  const set = (patch) => onChange({ ...filters, ...patch });

  const genres = kind === "user" ? [] : genresIn(rows);
  const crates = kind === "track" ? cratesIn(rows, crateOf) : [];
  const countries = kind === "user" ? countriesIn(rows) : [];

  const narrowed = visibleCount !== rows.length;

  return (
    <div className="sc-filterbar">
      <select className="mini-select" value={filters.sort}
        onChange={(e) => set({ sort: e.target.value })} title="Sort these results">
        {/* Unsorted is the default: SoundCloud's relevance order is meaningful. */}
        <option value="">Sort: relevance</option>
        {Object.entries(sorts).map(([key, s]) => (
          <option key={key} value={key}>{s.label}</option>
        ))}
      </select>

      {filters.sort && (
        <button className="mini-btn" title="Reverse the sort"
          onClick={() => set({ dir: filters.dir === "asc" ? "desc" : "asc" })}>
          {filters.dir === "asc" ? "↑ asc" : "↓ desc"}
        </button>
      )}

      {genres.length > 1 && (
        <select className="mini-select" value={filters.genre}
          onChange={(e) => set({ genre: e.target.value })} title="Filter by genre">
          <option value="">All genres</option>
          {genres.map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
      )}

      {kind === "track" && (
        <>
          <select className="mini-select" value={filters.inLibrary}
            onChange={(e) => set({ inLibrary: e.target.value })}
            title="Filter by whether it is already in your library">
            <option value="any">Library: any</option>
            <option value="no">Not in library</option>
            <option value="yes">In library</option>
          </select>

          <select className="mini-select" value={filters.inCrate}
            onChange={(e) => set({ inCrate: e.target.value })}
            title="Filter by crate membership">
            <option value="any">Crate: any</option>
            <option value="none">In no crate</option>
            {crates.map((c) => (
              <option key={c.crate_id} value={c.crate_id}>In “{c.name}”</option>
            ))}
          </select>

          <label className="sc-filter-check" title="Hide Go+ tracks, which only stream a ~30s preview">
            <input type="checkbox" checked={filters.hideSnip}
              onChange={(e) => set({ hideSnip: e.target.checked })} />
            Hide Go+
          </label>

          <span className="sc-filter-range" title="Duration in minutes">
            <input className="mini-num" type="number" min="0" step="0.5" placeholder="min"
              value={filters.minMin} onChange={(e) => set({ minMin: e.target.value })} />
            <span className="faint">–</span>
            <input className="mini-num" type="number" min="0" step="0.5" placeholder="max"
              value={filters.maxMin} onChange={(e) => set({ maxMin: e.target.value })} />
            <span className="faint">min</span>
          </span>
        </>
      )}

      {kind === "playlist" && (
        <select className="mini-select" value={filters.album}
          onChange={(e) => set({ album: e.target.value })} title="Albums or playlists">
          <option value="any">Sets: any</option>
          <option value="album">Albums only</option>
          <option value="playlist">Playlists only</option>
        </select>
      )}

      {kind === "user" && (
        <>
          <select className="mini-select" value={filters.verified}
            onChange={(e) => set({ verified: e.target.value })} title="Verified artists only">
            <option value="any">Verified: any</option>
            <option value="yes">Verified only</option>
          </select>
          {countries.length > 1 && (
            <select className="mini-select" value={filters.country}
              onChange={(e) => set({ country: e.target.value })} title="Filter by country">
              <option value="">All countries</option>
              {countries.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          )}
        </>
      )}

      <span style={{ flex: 1 }} />

      {/* Say plainly what the numbers mean. These are the rows fetched so far,
          not everything SoundCloud holds — Load more appends into the filter. */}
      <span className="faint sc-filter-count">
        {narrowed
          ? `showing ${visibleCount} of ${rows.length} loaded`
          : `${rows.length} loaded`}
      </span>
    </div>
  );
}
