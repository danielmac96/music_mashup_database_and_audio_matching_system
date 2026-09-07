// ★★★★☆. Stars are the verdict: 1-5, they filter the library and they train
// the ranking.
//
// An unrated thing shows five hollow stars in the disabled ink rather than
// nothing, so the column keeps its width and a rated row reads as a difference
// in colour rather than a difference in layout.
export function StarRating({ value, onRate = null, size = 12, title }) {
  const stars = Math.max(0, Math.min(5, Math.round(value || 0)));
  const interactive = typeof onRate === "function";
  const glyphs = "★★★★★".slice(0, stars) + "☆☆☆☆☆".slice(0, 5 - stars);

  if (!interactive) {
    return (
      <span className="stars mono"
        style={{ fontSize: size, color: stars ? "var(--amber)" : "var(--faint-2)" }}
        title={title || (stars ? `${stars} of 5` : "not rated")}>
        {glyphs}
      </span>
    );
  }

  return (
    <span className="stars interactive mono" style={{ fontSize: size }}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} type="button"
          className={n <= stars ? "on" : ""}
          title={`Rate ${n} of 5`}
          onClick={(e) => { e.stopPropagation(); onRate(n); }}>
          {n <= stars ? "★" : "☆"}
        </button>
      ))}
    </span>
  );
}
