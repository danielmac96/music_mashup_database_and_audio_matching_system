// The refresh's no-truncation check (master plan 3C, criterion 3.6), and the
// PR's screenshots. Walks every screen of a running app with Playwright at
// 1280×800 and 1920×1080 and reports each visible element whose TEXT is cut
// off — scrollWidth > clientWidth with an ellipsis or hidden / clipped
// overflow — unless it is in EXEMPT, each of which names its fallback.
//
//   node tests/ui/ui_check.cjs http://localhost:8000 [screenshot-dir]
//
// Prints one JSON object: { screens: [...], problems: [{viewport, screen,
// selector, text}] }. tests/test_ui_overflow.py runs it against a synthetic
// library; with a folder it also writes a deviceScaleFactor-2 PNG per screen.
const { chromium } = require("playwright");

// Clipped on purpose, each with the way to read the whole text.
const EXEMPT = [
  // Library name cell at a narrow column: the .tt-fullname card on hover and
  // keyboard focus, plus the native tooltip.
  ".tt-title", ".tt-artist",
  // Canvas-adjacent waveform / structure labels: the section table lists them.
  ".ss-lbl", ".strip-label",
  // Native <select> and <input> values: the control's own dropdown / caret.
  "select", "input", "textarea", "option",
];

const SCREENS = [
  ["library", async (p) => { await rail(p, "Library"); }],
  ["library-dock-expanded", async (p) => {
    await rail(p, "Library");
    const card = p.locator(".pair-card").first();
    if (await card.count()) { await card.click(); await p.waitForTimeout(400); }
  }],
  ["tune-drawer", async (p) => {
    await rail(p, "Library");
    const b = p.getByRole("button", { name: /^Tune/ });
    if (await b.count()) { await b.first().click(); await p.waitForTimeout(200); }
  }],
  ["vocal-parts", async (p) => {
    await rail(p, "Library");
    await p.getByRole("button", { name: "Vocal parts" }).click();
    await p.waitForTimeout(1200);
    const row = p.locator("[data-seg-table] .seg-row, [data-seg-table] [role=row]").nth(1);
    if (await row.count()) { await row.click(); await p.waitForTimeout(1500); }
  }],
  ["columns-attributes", async (p) => {
    await rail(p, "Library");
    await p.getByRole("button", { name: "Tracks" }).click();
    await p.getByRole("button", { name: /Columns & attributes/ }).click();
    await p.waitForTimeout(300);
  }],
  ["track-detail", async (p) => {
    await rail(p, "Library");
    await p.getByRole("button", { name: "Tracks" }).click().catch(() => {});
    const t = p.locator(".tt-title").first();
    if (await t.count()) { await t.click(); await p.waitForTimeout(900); }
  }],
  ["compare", async (p) => {
    const b = p.getByRole("button", { name: /Compare with/ });
    if (await b.count()) { await b.first().click(); await p.waitForTimeout(500); }
  }],
  ["discover", async (p) => { await rail(p, "Discover"); }],
  ["studio", async (p) => { await rail(p, "Studio"); }],
  ["mixes", async (p) => { await rail(p, "Mixes"); }],
  ["queue", async (p) => { await rail(p, "Queue"); }],
  ["settings", async (p) => {
    await rail(p, "Library");
    await p.getByRole("button", { name: "Settings" }).click();
    await p.waitForTimeout(800);
  }],
];

async function rail(p, label) {
  await p.keyboard.press("Escape").catch(() => {});
  const drawerX = p.locator(".drawer-x");
  if (await drawerX.count()) await drawerX.first().click().catch(() => {});
  await p.locator(".rail").getByRole("button", { name: label, exact: true }).first().click();
  await p.waitForTimeout(700);
}

function audit(exempt) {
  const bad = [];
  const sel = (el) => {
    const c = typeof el.className === "string" && el.className.trim()
      ? `.${el.className.trim().split(/\s+/).join(".")}` : "";
    return `${el.tagName.toLowerCase()}${c}`;
  };
  for (const el of document.querySelectorAll("body *")) {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height || r.bottom < 0 || r.top > innerHeight) continue;
    if (el.scrollWidth <= el.clientWidth + 1) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden") continue;
    const ownText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
    if (!ownText) continue;
    const clips = cs.textOverflow === "ellipsis" || cs.overflowX === "hidden" || cs.overflowX === "clip";
    if (!clips) continue;
    if (exempt.some((x) => el.matches(x) || el.closest(x))) continue;
    bad.push({ selector: sel(el), text: el.textContent.trim().slice(0, 70) });
  }
  return bad;
}

(async () => {
  const [base, shots] = process.argv.slice(2);
  const browser = await chromium.launch();
  const problems = [];
  const screens = [];
  for (const [w, h] of [[1280, 800], [1920, 1080]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h },
      deviceScaleFactor: shots && w === 1920 ? 2 : 1 });
    await page.goto(base, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    for (const [name, go] of SCREENS) {
      try { await go(page); } catch (e) { problems.push({ viewport: `${w}x${h}`, screen: name, selector: "(navigation)", text: String(e).slice(0, 120) }); continue; }
      await page.waitForTimeout(300);
      for (const b of await page.evaluate(audit, EXEMPT)) {
        problems.push({ viewport: `${w}x${h}`, screen: name, ...b });
      }
      if (shots && w === 1920) await page.screenshot({ path: `${shots}/${name}.png` });
      if (w === 1280) screens.push(name);
    }
    await page.close();
  }
  await browser.close();
  console.log(JSON.stringify({ screens, exempt: EXEMPT, problems }));
})();
