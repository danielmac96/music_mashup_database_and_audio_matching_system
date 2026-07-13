import { artGradient } from "../theme";

// Track artwork square: the real thumbnail when we have one, else a
// deterministic per-track gradient so every row keeps a stable "cover".
// Rendered as a background-image div (all the .art/.card-art/.pair-art CSS
// sizes assume that), with optional children (e.g. ♪) centered on top.
export function TrackArt({ id, thumbnail, className = "", children = null, style = {} }) {
  const background = thumbnail
    ? `url(${JSON.stringify(thumbnail)}) center/cover no-repeat, ${artGradient(id)}`
    : artGradient(id);
  return (
    <div className={className} style={{ background, ...style }} aria-hidden="true">
      {thumbnail ? null : children}
    </div>
  );
}
