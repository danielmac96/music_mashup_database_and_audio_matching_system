// What is done to a pair to build it, and what to listen out for.
//
// Read straight from the server's recipe (matcher/recipe.pair_recipe), never
// re-derived here: the card, the dock's loop, Studio and the FL export all play
// the numbers this strip prints. Each chip's tooltip says why; its tone is the
// effort chip's free / light / heavy. A pair with nothing to do says so — that
// is the best thing a card can tell you.

export function RecipeStrip({ recipe, compact = false }) {
  if (!recipe) return null;
  const adj = recipe.adjustments || [];
  const warn = recipe.warnings || [];
  return (
    <div className={`pc-recipe${compact ? " compact" : ""}`}>
      <div className="pc-do">
        <span className="pc-recipe-label mono" title="What to do to build this pair — the loop, Studio and the FL export already do it">DO</span>
        {adj.length ? adj.map((a) => (
          <span key={a.key} className={`pc-adj mono ${a.level}`} title={a.why}>{a.text}</span>
        )) : (
          <span className="pc-adj mono free" title="Same tempo, same key, on the grid">
            nothing — drop both in
          </span>
        )}
      </div>
      {warn.length > 0 && (
        <div className="pc-watch">
          <span className="pc-recipe-label mono" title="What the measurements cannot promise — check by ear">WATCH</span>
          {warn.map((w) => <span key={w.key} className="pc-warn">{w.text}</span>)}
        </div>
      )}
    </div>
  );
}
