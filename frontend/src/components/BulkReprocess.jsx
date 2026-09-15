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
// It renders nothing when there is nothing to do, so a current library is not
// nagged.

const WHAT_IS_MISSING = [
  ["missing_section_chroma", "measured harmony", "the ♪ shift and bass-clash chips"],
  ["missing_section_grid", "section tempo + grid", "bar counts, downbeats and phrase alignment"],
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
  if (!needsAnalysis && !needsSeparate && !suspect) return null;

  const run = async (action, scope) => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.bulkReprocess({ action, scope });
      setJob({ id: out.job_id, action });
      toast(`Queued ${out.count} track${out.count === 1 ? "" : "s"}`);
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
    </>
  );
}
