/* Grab a tracklist out of the page you are looking at.
 *
 * WHY THIS EXISTS. 1001tracklists is behind Cloudflare Turnstile. Firecrawl's
 * stealth proxy rendered it until 2026-09-19 and has been refused since; your
 * own browser is never refused, because you are a person. So the capture runs
 * where the page already is, and hands the app the SAME markdown Firecrawl
 * produced — one parser (ingest/firecrawl_scrape.parse_markdown_tracklist)
 * serves the scrape, the capture and the plain paste.
 *
 * OUTPUT, one line per track:
 *   1. [0:00] Artist \- Title[open track page](https://www.1001tracklists.com/track/ID/)
 *   w/ [0:40] Artist \- Title[open track page](https://www.1001tracklists.com/track/ID/)
 *
 * SELECTORS. Anchors only — never class names. 1001tracklists' classes are
 * obfuscated and change; the one durable fact, proved by every page Firecrawl
 * ever returned, is that each track row contains a link to "/track/<id>/".
 * Everything else here is a guess until a real page proves otherwise, which is
 * exactly why this shows you what it found instead of importing it blind, and
 * why DEBUG exists.
 *
 * This file is the SOURCE. It is minified into a javascript: URL at build time
 * by frontend/scripts/buildBookmarklet.mjs — it cannot be loaded from the app
 * at runtime, because an https:// page refuses a script from http://localhost.
 */
(function grabTracklist() {
  var TRACK_RE = /\/track\/([^/?#]+)/;
  var CUE_RE = /\b(\d{1,2}:\d{2}(?::\d{2})?)\b/;
  var OVERLAY_RE = /(^|\s)w\/(\s|$)/i;

  function text(el) {
    return (el.textContent || "").replace(/\s+/g, " ").trim();
  }

  // The row is the nearest ancestor that holds this anchor and NO other track
  // anchor. Walking up by tag or class would need markup knowledge we do not
  // have; "stop before you swallow the next track" needs none.
  function rowFor(a) {
    var el = a;
    var best = a;
    for (var i = 0; i < 8 && el.parentElement; i++) {
      el = el.parentElement;
      if (el.querySelectorAll("a[href*='/track/']").length > 1) break;
      best = el;
    }
    return best;
  }

  var anchors = [].slice.call(document.querySelectorAll("a[href*='/track/']"));
  var seen = {};
  var rows = [];

  anchors.forEach(function (a) {
    var m = TRACK_RE.exec(a.getAttribute("href") || "");
    if (!m) return;
    var id = m[1];
    if (seen[id]) return;            // the title and its artwork both link out
    seen[id] = true;

    var row = rowFor(a);
    var whole = text(row);
    var label = text(a);
    // Prefer the row's own text; fall back to the link text when the row adds
    // nothing (some layouts put the whole name inside the anchor).
    var body = whole || label;

    var overlay = OVERLAY_RE.test(body);
    if (overlay) body = body.replace(OVERLAY_RE, "$1").trim();

    var cue = "";
    var c = CUE_RE.exec(body);
    if (c) {
      cue = c[1];
      body = body.replace(c[0], " ").trim();
    }

    // Trim leading position numbers and the site's own furniture, and drop the
    // trailing "[open track page]"-ish link label if the row repeated it.
    body = body.replace(/^[\s.|\-–—]*\d{1,3}[.)]\s*/, "").trim();
    body = body.replace(/\s*\[?open track page\]?\s*$/i, "").trim();
    if (!body) return;

    rows.push({ id: id, body: body, cue: cue, overlay: overlay,
                url: a.href.split("#")[0], html: row.outerHTML });
  });

  // The markdown the app parses. A backslash before the hyphen is Firecrawl's
  // own escaping; the parser strips it either way, so it stays for fidelity.
  var bed = 0;
  var md = rows.map(function (r) {
    var lead;
    if (r.overlay) {
      lead = "w/ ";
    } else {
      bed += 1;
      lead = bed + ". ";
    }
    var cue = r.cue ? "[" + r.cue + "] " : "";
    var name = r.body.replace(/ - /, " \\- ");
    return lead + cue + name + "[open track page](" + r.url + ")";
  }).join("\n");

  var overlays = rows.filter(function (r) { return r.overlay; }).length;

  /* ── the review panel ──────────────────────────────────────────────────── */
  // Nothing is sent anywhere. You read what was captured, then copy it. The
  // panel is also the only debugger we get: when a selector guess is wrong,
  // "show the HTML" is what tells us which one.
  var old = document.getElementById("mashup-grab-panel");
  if (old) old.remove();

  var wrap = document.createElement("div");
  wrap.id = "mashup-grab-panel";
  wrap.setAttribute("style", [
    "position:fixed", "z-index:2147483647", "top:16px", "right:16px",
    "width:560px", "max-width:calc(100vw - 32px)", "background:#11151d",
    "color:#e6ebf5", "border:1px solid #2b3446", "border-radius:10px",
    "padding:12px", "box-shadow:0 10px 40px rgba(0,0,0,.5)",
    "font:13px/1.45 ui-sans-serif,system-ui,sans-serif",
  ].join(";"));

  var head = document.createElement("div");
  head.setAttribute("style", "display:flex;align-items:center;gap:8px;margin-bottom:8px");
  head.innerHTML =
    "<b style='font-size:13px'>Mashup Engine</b>" +
    "<span style='color:#8b97ad;font-size:12px'>captured " + rows.length +
    " tracks, " + overlays + " overlays</span>";

  var close = document.createElement("button");
  close.textContent = "close";
  close.setAttribute("style", "margin-left:auto;cursor:pointer;background:none;border:0;color:#8b97ad;font:12px inherit");
  close.onclick = function () { wrap.remove(); };
  head.appendChild(close);

  var area = document.createElement("textarea");
  area.value = md;
  area.setAttribute("style", [
    "width:100%", "height:260px", "box-sizing:border-box", "resize:vertical",
    "background:#0b0e14", "color:#e6ebf5", "border:1px solid #2b3446",
    "border-radius:7px", "padding:8px",
    "font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace",
  ].join(";"));

  var bar = document.createElement("div");
  bar.setAttribute("style", "display:flex;gap:8px;align-items:center;margin-top:8px");

  var copy = document.createElement("button");
  copy.textContent = "Copy";
  copy.setAttribute("style", "cursor:pointer;border:0;border-radius:7px;padding:7px 16px;font:600 12px inherit;color:#07090d;background:#5b8cff");
  // A real click is what earns clipboard permission; execCommand is the fallback
  // for browsers that still refuse it from a bookmarklet's context.
  copy.onclick = function () {
    area.select();
    var done = function () { copy.textContent = "Copied ✓"; };
    try {
      navigator.clipboard.writeText(area.value).then(done, function () {
        document.execCommand("copy"); done();
      });
    } catch (e) {
      document.execCommand("copy"); done();
    }
  };

  var dbg = document.createElement("button");
  dbg.textContent = "show the HTML";
  dbg.setAttribute("style", "cursor:pointer;background:none;border:0;color:#8b97ad;font:12px inherit;text-decoration:underline");
  var showingMd = true;
  dbg.onclick = function () {
    showingMd = !showingMd;
    area.value = showingMd
      ? md
      : rows.slice(0, 3).map(function (r) { return r.html; }).join("\n\n---\n\n");
    dbg.textContent = showingMd ? "show the HTML" : "show the tracklist";
  };

  var hint = document.createElement("span");
  hint.setAttribute("style", "color:#8b97ad;font-size:11px;margin-left:auto");
  hint.textContent = rows.length ? "paste into the Mixes tab" : "no tracks found";

  bar.appendChild(copy);
  bar.appendChild(dbg);
  bar.appendChild(hint);
  wrap.appendChild(head);
  wrap.appendChild(area);
  wrap.appendChild(bar);
  document.body.appendChild(wrap);
  area.focus();
  area.select();
})();
