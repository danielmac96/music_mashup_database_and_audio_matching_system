// A clickable column header, shared by the library table and Discover's results.
//
// Both tables already HELD a sort — Discover in `useResultFilters`, the library
// in `useLibraryFilters` — and both hid it in a dropdown while the column
// headings sat there as inert <div>s. This is the affordance, not a new engine.
//
// THE CYCLE IS THREE-STATE, and the third state is the point: unsorted is a
// real choice in both tables. Discover's default is SoundCloud's own relevance
// order, and the library's is import order; a header that could only toggle
// asc/desc would make either impossible to get back to without hunting for the
// dropdown.
//
// Numbers open DESCENDING (most plays first is what anyone means by clicking
// PLAYS) and text opens ascending.

export function SortHead({ label, sortKey, sort, dir, onSort, numeric = false,
                           className = "", title }) {
  const active = Boolean(sortKey) && sort === sortKey;

  const next = () => {
    const first = numeric ? "desc" : "asc";
    const second = numeric ? "asc" : "desc";
    if (!active) return { sort: sortKey, dir: first };
    if (dir === first) return { sort: sortKey, dir: second };
    return { sort: "", dir: "asc" };          // back to the natural order
  };

  if (!sortKey) return <div className={className}>{label}</div>;

  return (
    <button type="button"
      className={`sort-head${active ? " on" : ""} ${className}`.trim()}
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
      title={title || (active
        ? `Sorted by ${label.toLowerCase()} — click to ${dir === (numeric ? "desc" : "asc") ? "reverse" : "clear"}`
        : `Sort by ${label.toLowerCase()}`)}
      onClick={(e) => { e.stopPropagation(); onSort(next()); }}>
      <span className="sh-label">{label}</span>
      <span className="sh-arrow" aria-hidden="true">
        {active ? (dir === "asc" ? "▲" : "▼") : ""}
      </span>
    </button>
  );
}
