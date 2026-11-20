import { useEffect, useMemo, useRef, useState } from "react";
import { SegmentTable } from "../segments/SegmentTable";
import { SegmentFilters } from "../segments/SegmentFilters";
import {
  buildSideRows, cycleSegSort, EMPTY_SEG_FILTERS, LABELS, SEG_SORTS, segKey, STEM_OF,
} from "../segments/segmentModel";
import { toast } from "../../toast";
import { useModeView } from "../../hooks/useModeView";

// The Library in Parts mode (master plan 3A — what the Segments tab was):
// every vocal section, or every bed section, of the library as one
// virtualised table, filtered in memory with the Library's search language
// (useLibraryFilters.parseQuery: 8A~, 124-128, house, plus chorus / 16bars).
// Selecting a part turns the pair dock into its live matches (PartsDock).
//
// `segs` is useSegments() — owned by App so the table and the dock read one
// copy. `seed` ({songId, sectionIndex, role, at}) is Track detail's
// "Find beds ▸" / ⌘K: it selects that section (or the track's best one).

export function PartsView({ role, segs, selected, onSelect, onOpenTrack, player, tracks,
                            seed = null }) {
  const [filters, setFilters] = useState({ vocal: { ...EMPTY_SEG_FILTERS },
                                           bed: { ...EMPTY_SEG_FILTERS } });
  const [sorts, setSorts] = useState({ vocal: { id: null, dir: null }, bed: { id: null, dir: null } });
  // Saved per mode (useModeView): each side's filters and sort come back on
  // the next visit, as the Tracks view's do.
  const saved = useMemo(() => ({ filters, sorts }), [filters, sorts]);
  useModeView("parts", saved, (v) => {
    const side = (x, empty) => (x && typeof x === "object" ? { ...empty, ...x } : empty);
    if (v.filters) {
      setFilters({ vocal: side(v.filters.vocal, { ...EMPTY_SEG_FILTERS }),
                   bed: side(v.filters.bed, { ...EMPTY_SEG_FILTERS }) });
    }
    if (v.sorts) {
      setSorts({ vocal: side(v.sorts.vocal, { id: null, dir: null }),
                 bed: side(v.sorts.bed, { id: null, dir: null }) });
    }
  });

  const genres = useMemo(() => {
    const g = new Set();
    for (const r of [...segs.vocal, ...segs.bed]) if (r.genre) g.add(r.genre);
    return [...g].sort();
  }, [segs.vocal, segs.bed]);

  const view = useMemo(() => buildSideRows(segs[role] || [], null, filters[role], sorts[role], genres),
    [segs, role, filters, sorts, genres]);

  // A section handed over from elsewhere: select it, or the track's
  // most-wanted part for the role (the scorer's label order, then energy).
  const seeded = useRef(null);
  useEffect(() => {
    if (!seed || segs.loading || seeded.current === seed.at) return;
    seeded.current = seed.at;
    const mine = (segs[role] || []).filter((r) => r.song_id === seed.songId);
    const hit = seed.sectionIndex != null
      ? mine.find((r) => r.section_index === seed.sectionIndex)
      : mine.filter((r) => r.usable)
        .sort((a, b) => (SEG_SORTS.label(a) - SEG_SORTS.label(b))
          || ((b.energy_rank ?? -1) - (a.energy_rank ?? -1)))[0];
    if (!hit) {
      toast(`That section is not offered as a ${role === "vocal" ? "vocal" : "bed"} here`);
      return;
    }
    setFilters((f) => ({ ...f, [role]: { ...EMPTY_SEG_FILTERS } }));
    onSelect(hit);
  }, [seed, segs, role]); // eslint-disable-line react-hooks/exhaustive-deps

  const stemsOf = useMemo(() => new Map((tracks || []).map((t) => [t.id, t.stems])), [tracks]);
  const play = (r) => player.toggle({
    kind: "track", songId: r.song_id, stem: STEM_OF[role],
    loop: { start: r.start_sec, end: r.end_sec },
    title: `${r.title} — ${r.label || "section"}`, subtitle: r.artist || "",
    duration: r.duration_secs, stems: stemsOf.get(r.song_id) || null,
  });
  const playingKey = useMemo(() => {
    const s = player.source;
    if (s?.kind !== "track" || !s.loop || !player.playing) return null;
    const hit = (segs[role] || []).find((r) => r.song_id === s.songId
      && Math.abs(r.start_sec - s.loop.start) < 0.01 && Math.abs(r.end_sec - s.loop.end) < 0.01);
    return hit ? segKey(hit) : null;
  }, [player.source, player.playing, segs, role]);

  const asked = filters[role].labels || [];
  const empty = segs.loading ? "Loading the parts…"
    : segs.error ? segs.error
      : (segs[role] || []).length === 0
        ? `No ${role} parts yet — they appear once tracks are separated and analysed.`
        : role === "vocal" && asked.some((l) => l === "drop" || l === "breakdown" || l === "build")
          ? "Drops, builds and breakdowns rarely carry a vocal — look for them under Bed parts."
          : "No parts match these filters.";

  return (
    <div className={`parts-view ${role}`}>
      <SegmentFilters role={role} filters={filters[role]} genres={genres}
        onChange={(f) => setFilters((all) => ({ ...all, [role]: f }))}
        shown={view.rows.length} total={view.total} matchMode={false} labels={LABELS} />
      <SegmentTable role={role} rows={view.rows} matchMode={false}
        sort={sorts[role]}
        onSort={(id) => setSorts((s) => ({ ...s, [role]: cycleSegSort(s[role], id) }))}
        selectedKey={selected ? segKey(selected) : null}
        onSelect={onSelect}
        onOpen={onOpenTrack}
        onPlay={play}
        playingKey={playingKey}
        onSwitch={null}
        onTabOther={() => document.querySelector(".pair-dock [data-parts-list]")?.focus()}
        emptyText={empty}
        label={role === "vocal" ? "Vocal parts" : "Bed parts"} />
    </div>
  );
}
