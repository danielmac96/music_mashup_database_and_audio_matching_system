import { useEffect, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { toast } from "../toast";

// Backfill bar for the Library.
//
// Phases D and E added features that only exist on tracks analysed since:
// band occupancy and stem quality, per-section chroma and the measured
// transpose. An existing library keeps working, but the new chips and filters
// stay empty until those tracks are re-processed — and nobody is going to press
// ⟳ nine hundred times. This says what is missing, what it costs, and does it.
//
// It also reports SUSPECT AUDIO: tracks whose YouTube download fallback ran
// before substitutes were verified, so the file may be a remix or another cut
// of the record the track links to. That is a correctness problem rather than a
// missing feature, so it gets its own bar and goes first.
//
// And MISSING METADATA: tracks whose per-track SoundCloud fetch was throttled at
// import. Their audio, stems and analysis are fine — only genre, year and play
// count are blank — so nothing else in the app ever complains, and the columns
// just sit empty. This is the only bar whose button IS the work: it finishes
// when the fetches finish, rather than handing tracks to the pipeline queue.
//
// It renders nothing when there is nothing to do, so a current library is not
// nagged.

const WHAT_IS_MISSING = [
  ["missing_section_chroma", "measured harmony", "the ♪ shift and bass-clash chips"],
  ["missing_section_grid", "section tempo + grid", "bar counts, downbeats and phrase alignment"],
  ["missing_section_stem_measures", "section stem measures", "vocal activity, per-stem bands and sung range per section"],
  ["missing_band_energy", "band occupancy", "spectral collision scoring"],
  ["missing_stem_quality", "stem quality", "filtering out unusable acapellas"],
  ["missing_sections", "structure", "section-level pairing at all"],
];

export function BulkReprocess({ onQueued }) {
  const [stale, setStale] = useState(null);
  const [job, setJob] = useState(null);   // { id, action }
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [dismissed, setDismissed] = useState(false);

  const load = () => api.getStaleness().then(setStale).catch(() => setStale(null));
  useEffect(() => { load(); }, []);

  if (!stale || dismissed) return null;

  const needsAnalysis = stale.needs_analysis || 0;
  const needsSeparate = stale.missing_four_stems || 0;
  const suspect = stale.suspect_audio || 0;
  const noMeta = stale.missing_metadata || 0;
  // allin1 (readme §5.5): owed in the allin1 engine; offered either way so
  // the Engines comparison can have both analyses.
  const noAllin1 = stale.missing_allin1 || 0;
  const allin1On = stale.segmentation_engine === "allin1";
  const wrongPipeline = stale.wrong_pipeline || 0;
  if (!needsAnalysis && !needsSeparate && !suspect && !noMeta && !noAllin1
      && !wrongPipeline) return null;

  const run = async (action, scope) => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.bulkReprocess({ action, scope });
      setJob({ id: out.job_id, action });
      // "metadata" does its own work rather than filling the pipeline queue,
      // so "queued" would be a lie about what is happening next.
      toast(action === "metadata"
        ? `Fetching metadata for ${out.count} track${out.count === 1 ? "" : "s"}…`
        : `Queued ${out.count} track${out.count === 1 ? "" : "s"}`);
      onQueued?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const badge = (action) => job?.action === action && (
    <JobBadge jobId={job.id} onComplete={(done) => {
      setJob(null);
      load();
      onQueued?.();
      if (done.status === "completed") {
        if (done.result?.summary) toast(done.result.summary);
        if (done.result?.reasons?.length) setError(done.result.reasons.join(" · "));
      } else if (done.status === "failed") {
        setError(done.message || "Bulk job failed");
      }
    }} />
  );

  const missing = WHAT_IS_MISSING
    .filter(([key]) => (stale[key] || 0) > 0)
    .map(([key, what, why]) => `${stale[key]} missing ${what} (${why})`);

  return (
    <>
      <div className="drawer-section">
        <h3>Re-process the library</h3>
        <span className="hint">
          Bulk re-analysis and stem separation. It belongs here rather
          than on the Library screen because it acts on everything at
          once — the per-track stages are on each row's pipeline dots.
        </span>
      </div>
      <div className="drawer-body">
      {suspect > 0 && (
        <div className="bulk-bar">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              {suspect} track{suspect === 1 ? "" : "s"} may have the wrong audio
            </div>
            <div className="faint" style={{ fontSize: 11, lineHeight: 1.45 }}>
              SoundCloud would not serve {suspect === 1 ? "it" : "them"}, and the
              YouTube upload downloaded instead was never checked against the
              linked record — it can be a remix or a different cut. Re-downloading
              points each one back at its SoundCloud link, restores the credited
              artist, and only accepts an upload of the same record. Stems and
              analysis are redone; your pair verdicts are kept.
            </div>
          </div>
          {badge("redownload_suspect") || (
            <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
              <button className="btn" disabled={busy || !!job}
                title="Re-fetch each track's SoundCloud link and length, then download again through the verified fallback"
                onClick={() => run("redownload_suspect", "stale")}>
                ⟳ Re-download {suspect}
              </button>
            </div>
          )}
        </div>
      )}

      {wrongPipeline > 0 && (
        <div className="bulk-bar">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              {wrongPipeline} track{wrongPipeline === 1 ? " is" : "s are"} not on
              the {stale.separation_pipeline} pipeline yet
            </div>
            <div className="faint" style={{ fontSize: 11, lineHeight: 1.45 }}>
              Tracks whose stems for it already exist are swapped in place — no
              separation, no analysis, nothing deleted. The rest are queued for
              separation. Switching back later is the same swap.
            </div>
          </div>
          {badge("switch_pipeline") || (
            <button className="btn" disabled={busy || !!job}
              title="Swap each track's stems, features and sections to the configured pipeline, queueing only those never separated that way"
              onClick={() => run("switch_pipeline", "stale")}>
              ⇄ Switch {wrongPipeline}
            </button>
          )}
        </div>
      )}

      {noAllin1 > 0 && (
        <div className="bulk-bar">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              {noAllin1} track{noAllin1 === 1 ? " has" : "s have"} no allin1 analysis
            </div>
            <div className="faint" style={{ fontSize: 11, lineHeight: 1.45 }}>
              {allin1On
                ? "The allin1 engine is on: until a track has run, its pairs use its Essentia sections (marked essentia_fallback). "
                : "Only needed for the allin1 engine, or to compare the two engines in Analysis › Engines. "}
              allin1 finds downbeats and labelled sections — about a minute or
              more a track on CPU, in the background, never beside Demucs.
            </div>
          </div>
          {badge("allin1") || (
            <button className="btn" disabled={busy || !!job}
              title="Queue allin1 for every analysed track without a current allin1 analysis"
              onClick={() => run("allin1", "stale")}>
              ⟳ allin1 {noAllin1}
            </button>
          )}
        </div>
      )}

      {noMeta > 0 && (
        <div className="bulk-bar">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              {noMeta} track{noMeta === 1 ? "" : "s"} {noMeta === 1 ? "is" : "are"} missing
              {" "}genre, year and play count
            </div>
            <div className="faint" style={{ fontSize: 11, lineHeight: 1.45 }}>
              SoundCloud throttled the metadata fetch while{" "}
              {noMeta === 1 ? "it was" : "they were"} imported, so the library
              columns are blank. The audio, stems and analysis are fine and are
              not touched — this only re-fetches the description. A few seconds
              a track, and some uploads genuinely carry no genre.
            </div>
          </div>
          {badge("metadata") || (
            <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
              <button className="btn" disabled={busy || !!job}
                title="Re-fetch genre, year, play count, likes and tags from the link each track was imported from. Nothing is re-downloaded or re-analysed."
                onClick={() => run("metadata", "stale")}>
                ⟳ Refresh metadata {noMeta}
              </button>
            </div>
          )}
        </div>
      )}

      {(needsAnalysis > 0 || needsSeparate > 0) && (
        <div className="bulk-bar">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              {needsAnalysis > 0
                ? `${needsAnalysis} of ${stale.total_analysed} tracks were analysed before the latest features`
                : `${needsSeparate} tracks still have two stems`}
            </div>
            <div className="faint" style={{ fontSize: 11, lineHeight: 1.45 }}>
              {missing.length > 0 && <>{missing.join(" · ")}. </>}
              Re-analysing keeps your stems and takes roughly a minute a track.
              Nothing is lost either way — the pairs you have keep working.
            </div>
          </div>

          {badge("analyze") || badge("separate") || (
            <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
              {needsAnalysis > 0 && (
                <button className="btn" disabled={busy || !!job}
                  title="Re-run analysis + structure on just the tracks missing something. Stems are kept."
                  onClick={() => run("analyze", "stale")}>
                  ⟳ Re-analyse {needsAnalysis}
                </button>
              )}
              {needsSeparate > 0 && (
                <button className="btn" disabled={busy || !!job}
                  title="Re-separate into four stems (drums / bass / other / vocals), then re-analyse. This is the slow one — hours for a large library."
                  onClick={() => run("separate", "stale")}>
                  ⟳ Re-separate {needsSeparate}
                </button>
              )}
              <button className="mini-btn" disabled={busy}
                title="Hide until the next reload"
                onClick={() => setDismissed(true)}>
                later
              </button>
            </div>
          )}
        </div>
      )}
      {error && <div className="error-text" style={{ fontSize: 11 }}>{error}</div>}
      </div>
    </>
  );
}
