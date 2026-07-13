// Tiny "where did this track come from" badge (SC / YT). Renders nothing for
// unknown/empty sources so old rows without a source stay clean.
const META = {
  soundcloud: { label: "SC", cls: "src-sc", title: "SoundCloud" },
  youtube: { label: "YT", cls: "src-yt", title: "YouTube" },
};

export function SourceBadge({ source }) {
  const m = META[(source || "").toLowerCase()];
  if (!m) return null;
  return (
    <span className={`source-badge ${m.cls}`} title={m.title}>
      {m.label}
    </span>
  );
}
