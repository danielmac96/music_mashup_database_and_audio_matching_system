import { useEffect, useState } from "react";
import { api } from "../api";
import { audioSubstitution, classifyUrl } from "../sources";
import { fmtDur } from "../theme";
import { toast } from "../toast";

// Where a track's audio came from, and a way to fix it when it is wrong.
//
// SoundCloud serves a lot of label catalogue DRM-protected, so the downloader
// substitutes a YouTube upload of the same record. It verifies that upload — no
// remix, the credited artist, the linked record's length — but "Drake - Massive"
// once came back as an OCTANE remix before it did, and nothing on screen said
// the audio was not the link. This says so, and the picker lists the candidates
// with the SAME verdict the downloader applied, so you can see why one was
// chosen and choose another.

const SOURCE_NAME = { soundcloud: "SoundCloud", youtube: "YouTube" };

function fmtDelta(d) {
  if (d == null || !Number.isFinite(d)) return null;
  const r = Math.round(d);
  return `Δ ${r > 0 ? "+" : r < 0 ? "−" : "±"}${Math.abs(r)}s`;
}

// One line under the track's title: the link, what the audio is, and the way in.
export function AudioSource({ track, onPick, onChanged }) {
  const sub = audioSubstitution(track);
  const [confirming, setConfirming] = useState(false);
  const origin = track.origin_url;

  // "I listened and this is the record": YT? becomes YT, and the track leaves
  // the suspect-audio count. Offered only once there is audio to have listened to.
  const confirm = async () => {
    setConfirming(true);
    try {
      await api.confirmAudio(track.id);
      toast(`Confirmed the audio for "${track.title}"`);
      onChanged?.();
    } catch (e) {
      toast(`Could not confirm: ${e.message}`);
    } finally {
      setConfirming(false);
    }
  };
  const canConfirm = sub && !sub.confirmed && (track.stems?.full || track.raw_path);
  const originName = origin ? SOURCE_NAME[classifyUrl(origin).source] || "link" : null;
  return (
    <div className="audio-src">
      {origin && (
        <a href={origin} target="_blank" rel="noreferrer" title={origin}>
          Linked: {originName} ↗
        </a>
      )}
      {sub && (
        <span className={`audio-src-sub ${sub.kind}`} title={sub.title}>
          {sub.url
            ? <a href={sub.url} target="_blank" rel="noreferrer">{sub.label}</a>
            : sub.label}
        </span>
      )}
      {canConfirm && (
        <button className="mini-btn" disabled={confirming} onClick={confirm}
          title="You listened and this is the record — stop flagging it">
          ✓ Sounds right
        </button>
      )}
      <button className="mini-btn" onClick={onPick}
        title="Search YouTube for uploads of this record and pick the right one">
        Wrong audio?
      </button>
    </div>
  );
}

export function AudioSourcePicker({ track, onClose, onChanged }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    api.audioCandidates(track.id)
      .then((d) => { if (live) setData(d); })
      .catch((e) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [track.id]);

  const use = async (c) => {
    setBusy(c.url);
    setError(null);
    try {
      await api.updateTrackUrl(track.id, c.url, {
        title: c.title, uploader: c.uploader, duration_secs: c.duration_secs,
      });
      toast(`Re-downloading "${track.title}" from ${c.uploader || "YouTube"}`);
      onChanged?.();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  const expected = data?.expected_duration;
  return (
    <div className="asp" onClick={(e) => e.stopPropagation()}>
      <div className="asp-head mono">
        <span>WRONG AUDIO?</span>
        <button className="mini-btn" onClick={onClose}>close</button>
      </div>
      <div className="asp-note faint">
        {expected
          ? `The linked record is ${fmtDur(expected)}. `
          : "The linked record's length is unknown, so length is not checked. "}
        Uploads that fail the check say why — you can still choose one.
        Choosing re-downloads and reprocesses the track.
      </div>

      {!data && !error && <div className="hint">Searching YouTube…</div>}
      {error && <div className="error-text">{error}</div>}
      {data && data.candidates.length === 0 && (
        <div className="hint">YouTube returned nothing for “{data.artist} {data.title}”.</div>
      )}

      {(data?.candidates || []).map((c) => (
        <div key={c.url} className={`asp-row${c.passes ? " ok" : ""}`}>
          <div className="asp-text">
            <a className="asp-title" href={c.url} target="_blank" rel="noreferrer"
              title={c.title}>{c.title}</a>
            <span className="asp-meta mono">
              {[c.uploader, fmtDur(c.duration_secs), fmtDelta(c.duration_delta)]
                .filter(Boolean).join(" · ")}
            </span>
          </div>
          <span className={`asp-verdict${c.passes ? " ok" : ""}`}
            title={c.passes ? "Matches the linked record" : c.reason}>
            {c.passes ? "✓ matches" : c.reason}
          </span>
          <button className="mini-btn" disabled={!!busy || c.in_use}
            onClick={() => use(c)}>
            {c.in_use ? "in use" : busy === c.url ? "…" : "Use this"}
          </button>
        </div>
      ))}
    </div>
  );
}
