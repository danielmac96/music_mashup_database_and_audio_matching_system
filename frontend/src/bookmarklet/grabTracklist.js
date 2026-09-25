/* Grab a tracklist out of the page you are looking at.
 *
 * WHY THIS EXISTS. 1001tracklists is behind Cloudflare Turnstile and, since
 * 2026-09-24, its own image captcha; Firecrawl is refused by both. Your own
 * browser is not, because you are a person. So the capture runs where the page
 * already is, and hands the app the SAME markdown Firecrawl produced — one
 * parser (ingest/firecrawl_scrape.parse_markdown_tracklist) serves both.
 *
 * OUTPUT, one line per track, in set order:
 *   1. [0:00] Artist \- Title[open track page](https://www.1001tracklists.com/track/ID/)
 *   w/ [0:40] Artist \- Title[open track page](https://www.1001tracklists.com/track/ID/)
 *   2. [3:10] Artist \- Title[no track page]
 *
 * FINDING ROWS. No class names are hard-coded — 1001tracklists' change. Rows
 * are found from what every page proves:
 *   1. a track the site knows links to "/track/<id>/"; its row is the highest
 *      ancestor holding no OTHER track's link (rowFor);
 *   2. a track the site does not know is printed as plain text with no link at
 *      all — very often the opener and the closer of a set, which is how
 *      selecting on anchors alone lost the first and last tracks. Its row is
 *      built like its neighbours: same tag, carrying every class token all the
 *      linked rows share. Each such element inside the tracklist is a row.
 *
 * NAMING A ROW. A row's text is the name plus the label, a vote count, the user
 * who IDed it and a "Save" button — none of it the record, all of it poison to
 * a SoundCloud/YouTube search. So the name comes from, in order: schema.org
 * microdata (meta[itemprop=name], an SEO contract rather than styling); else
 * the smallest element holding "Artist - Title", with label links/elements
 * removed; else the row text. The server strips whatever furniture survives
 * (ingest/tracklist_parse.strip_row_furniture), so this may be imperfect.
 *
 * This file is the SOURCE. It is minified into a javascript: URL at build time
 * by frontend/scripts/buildBookmarklet.mjs — it cannot be loaded from the app
 * at runtime, because an https:// page refuses a script from http://localhost.
 */
(function grabTracklist() {
  var LINK = "a[href*='/track/']";
  var TRACK_RE = /\/track\/([^/?#]+)/;
  var CUE_RE = /\b(\d{1,2}:\d{2}(?::\d{2})?)\b/;
  var OVERLAY_RE = /(^|\s)w\/(\s|$)/i;
  var SUB = "data-mashup-sub";
  var DROP = "script,style,button,input,select,textarea,img,svg,[" + SUB + "]," +
             "a[href*='/label/'],[class*='label' i]";

  // Text nodes joined by a space. textContent glues neighbouring elements
  // together — the cue "01:58" and the marker "w/" become "01:58w/", and
  // neither can be read back out.
  function text(el) {
    var out = [];
    var walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    while (walk.nextNode()) out.push(walk.currentNode.nodeValue);
    return out.join(" ").replace(/\s+/g, " ").replace(/\s+([),.])/g, "$1")
      .replace(/([(])\s+/g, "$1").trim();
  }
  function classes(el) {
    return (el.getAttribute("class") || "").split(/\s+/).filter(Boolean);
  }
  function trackId(a) {
    return (TRACK_RE.exec(a.getAttribute("href") || "") || [])[1];
  }

  // The row is the highest ancestor that holds this track's anchors and NO
  // other track's. Walking up by tag or class would need markup knowledge we
  // do not have; "stop before you swallow the next track" needs none.
  function rowFor(a) {
    var id = trackId(a);
    var el = a;
    var best = a;
    for (var i = 0; i < 8 && el.parentElement; i++) {
      el = el.parentElement;
      var other = [].some.call(el.querySelectorAll(LINK), function (b) {
        return trackId(b) !== id;
      });
      if (other) break;
      best = el;
    }
    return best;
  }

  var linked = [];
  [].forEach.call(document.querySelectorAll(LINK), function (a) {
    if (!trackId(a)) return;
    var r = rowFor(a);
    if (linked.indexOf(r) < 0) linked.push(r);
  });

  // The row "shape": the commonest tag among linked rows, and the class tokens
  // at least 80% of them carry. A link that does not fit the shape — a "most
  // liked" sidebar, a player widget — is not a tracklist row and is dropped.
  function commonest(list) {
    var n = {};
    list.forEach(function (k) { n[k] = (n[k] || 0) + 1; });
    return Object.keys(n).sort(function (a, b) { return n[b] - n[a]; })[0] || "";
  }
  var tag = commonest(linked.map(function (r) { return r.tagName; }));
  var tokens = {};
  linked.forEach(function (r) {
    classes(r).forEach(function (c) { tokens[c] = (tokens[c] || 0) + 1; });
  });
  var shared = Object.keys(tokens).filter(function (c) {
    return tokens[c] >= Math.max(2, 0.8 * linked.length);
  });
  function fits(el) {
    var cs = classes(el);
    return el.tagName === tag &&
      shared.every(function (c) { return cs.indexOf(c) >= 0; });
  }
  if (linked.length > 2) linked = linked.filter(fits);

  // The tracklist is the lowest element holding every linked row.
  var root = linked.length ? linked[0].parentElement : null;
  while (root && !linked.every(function (r) { return root.contains(r); })) {
    root = root.parentElement;
  }
  var plain = !root ? [] : [].filter.call(root.querySelectorAll(tag), function (el) {
    if (linked.some(function (r) { return r === el || el.contains(r) || r.contains(el); })) {
      return false;
    }
    var alike = shared.length
      ? fits(el)
      : linked.some(function (r) { return r.parentElement === el.parentElement; });
    // ...and it reads like a track, not a heading that happens to match.
    return alike && /\s[-–—]\s|^\s*ID\s*$/.test(text(el));
  });

  var all = linked.concat(plain);
  all.sort(function (a, b) {
    return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
  });

  function nameOf(row) {
    // A row nested inside this one (an overlay inside its bed) is not its name.
    all.forEach(function (o) { if (o !== row && row.contains(o)) o.setAttribute(SUB, ""); });
    var clone = row.cloneNode(true);
    all.forEach(function (o) { o.removeAttribute(SUB); });

    var meta = [].filter.call(clone.querySelectorAll("meta[itemprop='name']"), function (m) {
      return / - /.test(m.getAttribute("content") || "");
    })[0];
    var bare = clone.cloneNode(true);
    [].forEach.call(clone.querySelectorAll(DROP), function (el) { el.remove(); });
    // If the label guess took the name with it, keep the label: the server
    // strips a label it can recognise, and a row with no name is lost.
    if (!/\s[-–—]\s/.test(text(clone))) clone = bare;
    var best = null;
    [].forEach.call(clone.querySelectorAll("*"), function (el) {
      var t = text(el);
      if (/\s[-–—]\s/.test(t) && (!best || t.length < text(best).length)) best = el;
    });
    var whole = text(clone);
    var shown = best ? text(best) : whole;
    var at = whole.indexOf(shown);
    return {
      name: meta ? meta.getAttribute("content").replace(/\s+/g, " ").trim() : shown,
      // What precedes the name in the row: the cue, the number, the "w/".
      lead: at > 0 ? whole.slice(0, at) : "",
    };
  }

  var rows = [];
  all.forEach(function (row) {
    var a = row.matches(LINK) ? row : row.querySelector(LINK);
    var n = nameOf(row);
    var body = n.name;
    var overlay = OVERLAY_RE.test(n.lead) || /^\s*w\//i.test(body);
    var c = CUE_RE.exec(n.lead);
    var cue = c ? c[1] : "";
    // The fallback path names the row by its whole text; peel what leads it.
    body = body.replace(/^\s*w\/\s*/i, "").replace(/^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*/, "");
    body = body.replace(/^[\s.|\-–—]*\d{1,3}[.)]?\s+(?=\S)/, "").trim();
    body = body.replace(/\s*\[?open track page\]?\s*$/i, "").replace(/\s[–—]\s/, " - ").trim();
    if (!body) return;
    rows.push({ body: body, cue: cue, overlay: overlay,
                url: a ? a.href.split("#")[0] : "", html: row.outerHTML });
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
    return lead + cue + name +
      (r.url ? "[open track page](" + r.url + ")" : "[no track page]");
  }).join("\n");

  var overlays = rows.filter(function (r) { return r.overlay; }).length;
  var unlinked = rows.filter(function (r) { return !r.url; }).length;

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
    " tracks, " + overlays + " overlays" +
    (unlinked ? ", " + unlinked + " without a track page" : "") + "</span>";

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
    // First, second and last rows: the edges are where a capture goes wrong.
    var pick = rows.length > 3 ? [rows[0], rows[1], rows[rows.length - 1]] : rows;
    area.value = showingMd
      ? md
      : pick.map(function (r) { return r.html; }).join("\n\n---\n\n");
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
