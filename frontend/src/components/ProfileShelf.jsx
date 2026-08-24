import { useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { TrackArt } from "./TrackArt";

// Your own shelves, so Discovery opens somewhere instead of on a blank search
// box. "Connect" IDENTIFIES a public profile — it is not a login: soundcloud_oauth
// is dormant because registering an app requires an Artist Pro subscription.
// That limit is stated on screen rather than discovered as an empty Likes tab.

const FEEDS = [
  ["playlists", "Your sets"],
  ["likes", "Your likes"],
  ["tracks", "Your uploads"],
];

export function ProfileShelf({ onOpenFeed }) {
  const [profile, setProfile] = useState(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.discoveryProfile()
      .then((b) => setProfile(b.profile))
      .catch(() => setProfile(null));
  }, []);

  const connect = async () => {
    const clean = url.trim();
    if (!clean) return;
    setBusy(true);
    setError("");
    try {
      const body = await api.discoverySetProfile(clean);
      setProfile(body.profile);
      setUrl("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    try {
      await api.discoveryDisconnect();
      setProfile(null);
    } catch (e) {
      toast(e.message);
    }
  };

  if (!profile) {
    return (
      <div className="sc-shelf connect">
        <div className="sc-shelf-head">
          <strong>Connect your SoundCloud</strong>
          <span className="hint">
            Paste your profile link and Discover opens on your own sets, likes
            and uploads. This only reads public pages — there is no login to
            offer, so private sets and private likes stay invisible.
          </span>
        </div>
        <div className="import-input-row">
          <div className="import-input">
            <input value={url} placeholder="https://soundcloud.com/your-name"
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") connect(); }} />
          </div>
          <button className="btn" onClick={connect} disabled={busy || !url.trim()}>
            {busy ? "Checking…" : "Connect"}
          </button>
        </div>
        {error && <div className="error-text">{error}</div>}
      </div>
    );
  }

  return (
    <div className="sc-shelf">
      <TrackArt id={profile.user_id} thumbnail={profile.avatar_url} className="sc-art" />
      <div className="mix-info">
        <div className="mix-title">
          {profile.username}
          {profile.verified && <span className="faint" title="Verified"> ✓</span>}
        </div>
        <div className="mix-url">
          <span className="faint">{profile.track_count} tracks · public pages only</span>
        </div>
      </div>
      <div className="seg sc-feed">
        {FEEDS.map(([id, label]) => (
          <button key={id}
            onClick={() => onOpenFeed(profile.user_id, profile.username, id)}>
            {label}
          </button>
        ))}
      </div>
      <button className="mini-btn" onClick={disconnect} title="Forget this profile">
        disconnect
      </button>
    </div>
  );
}
