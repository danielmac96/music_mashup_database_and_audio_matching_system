// useScWidget — play a SoundCloud search result without leaving the app.
//
// Discover's ▶ used to be an <a target="_blank">: "this app never streams
// external audio". Auditioning a record you are considering importing meant a
// new tab and a lost place in the list, so it now plays here.
//
// It plays through SOUNDCLOUD'S OWN WIDGET rather than a stream we resolve
// ourselves, and that is the load-bearing decision. Measured on a live search,
// only 3 of 10 results expose a `progressive` mp3; the rest are HLS and most
// also carry DRM-encrypted variants. Resolving those ourselves would need
// hls.js AND one extra api-v2 request per play against the scraped client_id
// that the FROZEN mixes resolver shares — the one rate limit this repo refuses
// to spend. The widget costs zero api-v2 requests and plays everything.
//
// Nothing in here may ever learn about client_id, api-v2 or transcodings. A
// test asserts that.

const API_JS = "https://w.soundcloud.com/player/api.js";
const PLAYER = "https://w.soundcloud.com/player/";

// The iframe is off-screen: this bar draws its own transport, and a second set
// of controls inside it would be the very inconsistency this change removes.
// Flip to true to show SoundCloud's own 20px strip inside the bar instead —
// the whole widget keeps working either way.
export const WIDGET_VISIBLE = false;

let scriptPromise = null;

/** Load the Widget API once, on FIRST external play — never at module scope, so
 *  the app still starts with no network. */
function loadApi() {
  if (scriptPromise) return scriptPromise;
  scriptPromise = new Promise((resolve, reject) => {
    if (window.SC?.Widget) { resolve(window.SC.Widget); return; }
    const el = document.createElement("script");
    el.src = API_JS;
    el.async = true;
    el.onload = () => (window.SC?.Widget
      ? resolve(window.SC.Widget)
      : reject(new Error("SoundCloud's player API loaded but exposed nothing")));
    el.onerror = () => {
      scriptPromise = null;                 // let a later play retry
      reject(new Error("Could not reach SoundCloud's player"));
    };
    document.head.appendChild(el);
  });
  return scriptPromise;
}

export function widgetSrc(permalink) {
  const opts = [
    `url=${encodeURIComponent(permalink)}`,
    "auto_play=true", "visual=false", "single_active=true",
    "hide_related=true", "show_comments=false", "show_teaser=false",
    "show_artwork=false", "buying=false", "download=false", "sharing=false",
  ];
  return `${PLAYER}?${opts.join("&")}`;
}

/**
 * One reusable widget, created imperatively so it survives every route change.
 *
 * `on` receives {playing, position, duration, ended, error} patches — the same
 * shape the element and engine backends report, so the bar never branches on
 * which one is sounding.
 */
export function createScPlayer(on) {
  let frame = null;
  let widget = null;
  let ready = null;             // Promise<widget> for the CURRENT url
  let currentUrl = "";
  let token = 0;                // a late READY from a row you left must not win
  let watchdog = 0;

  // SoundCloud's widget can load a track, report its duration, emit PLAY — and
  // then never make a sound. Its own media endpoint 404s on part of the
  // major-label catalogue (measured: two of the first six results for "drake"),
  // and it emits no ERROR when that happens; it just goes quiet. So the bar
  // would sit there showing 0:00 / 3:57 next to a button that does nothing,
  // which is the exact silent failure this whole change exists to remove.
  //
  // The watchdog therefore waits for the POSITION TO MOVE, not for a PLAY
  // event. PLAY is the widget's intention; a rising currentPosition is the only
  // evidence that audio is actually coming out.
  const PLAY_TIMEOUT_MS = 8000;

  const armWatchdog = (mine) => {
    clearTimeout(watchdog);
    watchdog = setTimeout(() => {
      if (token !== mine) return;
      on({
        playing: false,
        error: "SoundCloud won't stream this one here — open it there instead",
      });
    }, PLAY_TIMEOUT_MS);
  };
  const disarmWatchdog = () => clearTimeout(watchdog);

  // How often the widget is asked where it is. PLAY_PROGRESS is push-only and
  // stops entirely when paused or after a paused seek, which is the whole of
  // "skipping around works but the bottom bar does not track it".
  const POLL_MS = 250;
  let poll = 0;
  let lastSecs = -1;
  let lastDuration = 0;

  const stopPoll = () => { clearInterval(poll); poll = 0; };

  // The one path that means "audio is actually coming out", fed by both the
  // widget's own event and the poll. The watchdog's contract is unchanged: it is
  // disarmed by the position MOVING, never by a PLAY event.
  const advanced = (secs) => {
    if (secs > 0 && secs !== lastSecs) { disarmWatchdog(); on({ error: null }); }
    lastSecs = secs;
    on({ position: secs });
  };

  const startPoll = (mine) => {
    stopPoll();
    poll = setInterval(() => {
      if (token !== mine || !widget) return;
      widget.getPosition((ms) => {
        if (token === mine) advanced((ms || 0) / 1000);
      });
      // The widget is the authority on whether it is paused. Trusting only the
      // PAUSE event means one missed message leaves a ❚❚ on a silent bar.
      widget.isPaused((paused) => {
        if (token === mine) on({ playing: !paused });
      });
    }, POLL_MS);
  };

  const unmount = () => {
    if (frame) { frame.remove(); frame = null; }
  };

  const mount = (src) => {
    frame = document.createElement("iframe");
    // encrypted-media is not optional: most SoundCloud tracks are served as
    // cbc/ctr-encrypted HLS, and without it the widget logs a permissions-policy
    // violation and falls back — or, on a track with no clear rendition, plays
    // nothing at all. Chrome reports it only in the console, so the failure is
    // silent from the bar's point of view.
    frame.allow = "autoplay; encrypted-media";
    frame.title = "SoundCloud preview";
    frame.setAttribute("scrolling", "no");
    frame.frameBorder = "no";
    frame.width = "100%";
    // Positioned inline rather than by class: the stylesheet must not carry a
    // rule for something that is normally off-screen and usually not emitted.
    // Off-screen rather than display:none — a hidden iframe is allowed to stop
    // loading, and 0x0 makes some browsers skip the media entirely.
    frame.height = "20";
    frame.style.cssText = WIDGET_VISIBLE
      ? "position:fixed;right:18px;bottom:14px;width:320px;height:20px;border:0;z-index:6;"
      : "position:fixed;left:-9999px;top:0;width:400px;height:20px;border:0;";
    document.body.appendChild(frame);
    frame.src = src;
    return frame;
  };

  const bind = (Widget, mine) => new Promise((resolve, reject) => {
    const w = Widget(frame);
    const E = Widget.Events;
    const timer = setTimeout(
      () => reject(new Error("SoundCloud's player did not start")), 15000);
    w.bind(E.READY, () => {
      clearTimeout(timer);
      if (token !== mine) return;
      widget = w;
      w.bind(E.PLAY, () => {
        if (token !== mine) return;
        on({ playing: true });
      });
      w.bind(E.PAUSE, () => token === mine && on({ playing: false }));
      w.bind(E.FINISH, () => {
        if (token !== mine) return;
        disarmWatchdog();
        stopPoll();
        // FINISH arriving nowhere near the end is the same documented failure the
        // watchdog exists for, wearing a different hat: SoundCloud's media
        // endpoint gives up on part of the major-label catalogue and says
        // nothing useful about it. Treating that as "the track ended" silently
        // closes the bar a few seconds in, tells the user nothing, and takes the
        // "open it there instead" link away with it.
        if (lastDuration > 0 && lastSecs >= 0 && lastSecs < lastDuration - 5) {
          on({ playing: false,
               error: "SoundCloud stopped this one early — open it there instead" });
          return;
        }
        on({ playing: false, ended: true });
      });
      w.bind(E.PLAY_PROGRESS, (d) => {
        if (token !== mine) return;
        advanced((d?.currentPosition || 0) / 1000);
      });
      w.bind(E.ERROR, () => {
        if (token !== mine) return;
        disarmWatchdog();
        stopPoll();
        on({ playing: false, error: "SoundCloud would not play this track" });
      });
      // Duration is only known once the widget has the track.
      w.getDuration((ms) => {
        if (token !== mine) return;
        lastDuration = (ms || 0) / 1000;
        on({ duration: lastDuration });
      });
      resolve(w);
    });
  });

  // Transport commands have to survive being issued before READY. The iframe
  // carries auto_play=true and the bar appears the instant a row is clicked, so
  // there is a window in which audio is already sounding — and every control
  // used to be a silent no-op through all of it, because `widget` was still null.
  const live = async () => {
    if (widget) return widget;
    if (!ready) return null;
    try { return await ready; } catch { return null; }
  };

  return {
    /** Point a FRESH widget at `permalink` and start it. */
    async play(permalink) {
      const mine = ++token;
      const Widget = await loadApi();
      if (token !== mine) return;
      // A new iframe every time, not a new src on the old one, and NO reuse of
      // a previous bind. SC's Widget(frame) hands back the SAME wrapper for an
      // element it has already seen, so re-pointing the src left the previous
      // track's handlers registered — bound against the previous token, whose
      // guard then discarded every PLAY_PROGRESS and PAUSE that arrived. The
      // first row worked perfectly and every row after it showed a frozen clock
      // and a dead pause button while the audio really was playing.
      //
      // Skipping this when the url is unchanged looks like a free optimisation
      // and is the same bug again: `ready` would then hold a bind made under an
      // older token, which the ++token above has already invalidated. `toggle`
      // routes a repeat click on the live row to resume() and never reaches
      // here, so the optimisation bought nothing.
      stopPoll();
      unmount();
      widget = null;
      lastSecs = -1;
      lastDuration = 0;
      currentUrl = permalink;
      mount(widgetSrc(permalink));
      ready = bind(Widget, mine);
      let w;
      try {
        w = await ready;
      } catch (e) {
        stopPoll();
        throw e;
      }
      if (token !== mine) { w.pause(); return; }
      armWatchdog(mine);
      startPoll(mine);
      w.play();
    },
    async pause() {
      disarmWatchdog();
      stopPoll();
      (await live())?.pause();
    },
    async resume() {
      const mine = token;
      armWatchdog(mine);
      const w = await live();
      if (token !== mine) return;
      startPoll(mine);
      w?.play();
    },
    async seek(secs) {
      const w = await live();
      if (!w) return;
      const ms = Math.max(0, secs) * 1000;
      w.seekTo(ms);
      // seekTo emits no PLAY_PROGRESS while paused, so report the move itself or
      // the readout sits on the old time until playback resumes.
      on({ position: ms / 1000 });
      lastSecs = ms / 1000;
    },
    stop() {
      token += 1;
      disarmWatchdog();
      stopPoll();
      widget?.pause();
      widget = null;
      ready = null;
      currentUrl = "";
      lastSecs = -1;
      lastDuration = 0;
      unmount();
    },
    dispose() {
      token += 1;
      disarmWatchdog();
      stopPoll();
      widget = null;
      ready = null;
      currentUrl = "";
      unmount();
    },
  };
}
