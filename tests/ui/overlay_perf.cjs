// Track detail's overlay in a browser (master plan Phase 4, criteria 4.1 and
// 4.2): the canvas's draw time while the vocal is dragged across the bed, and
// how long after space the audition is sounding. Needs a running app whose
// stems exist on disk (a real library, or a synthetic one with stub audio).
//
//   node tests/ui/overlay_perf.cjs http://localhost:8000 [partners=3]
//
// Prints one JSON object: { draw_ms: {p50, p95, max, n}, drag_frames: {n,
// over16}, audition_ms: [...] }.
const { chromium } = require("playwright");

const pct = (xs, q) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.round(q * (s.length - 1)))] : null;
};

(async () => {
  const [base, nArg] = process.argv.slice(2);
  const partners = Number(nArg || 3);
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  await page.locator(".rail").getByRole("button", { name: "Library", exact: true }).first().click();
  await page.getByRole("button", { name: "Tracks" }).click().catch(() => {});
  await page.locator(".tt-title").first().click();
  const canvas = page.locator(".ov-canvas");
  await canvas.waitFor({ timeout: 15000 });
  await page.locator(".ov-panel").scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);

  // 4.1 — drag the incoming vocal across the bed and back.
  await page.evaluate(() => { window.__overlayDrawMs = []; window.__frames = [];
    let last = performance.now();
    const tick = (t) => { window.__frames.push(t - last); last = t;
      if (window.__frames.length < 2000) requestAnimationFrame(tick); };
    requestAnimationFrame(tick); });
  const box = await canvas.boundingBox();
  const y = box.y + 110;
  const x0 = box.x + box.width * 0.45;
  await page.mouse.move(x0, y);
  await page.mouse.down();
  for (let k = 0; k <= 60; k += 1) await page.mouse.move(x0 + (k * box.width * 0.4) / 60, y);
  for (let k = 60; k >= -30; k -= 1) await page.mouse.move(x0 + (k * box.width * 0.4) / 60, y);
  await page.mouse.up();
  await page.waitForTimeout(200);
  const draws = await page.evaluate(() => window.__overlayDrawMs || []);
  const frames = await page.evaluate(() => window.__frames || []);

  // 4.2 — space with the panel focused, until the transport says it plays.
  const audition = [];
  for (let i = 0; i < partners; i += 1) {
    await page.locator(".ov-panel").focus();
    if (i > 0) {
      await page.keyboard.press("ArrowRight");
      await page.waitForResponse((r) => r.url().includes("/overlay"), { timeout: 15000 }).catch(() => {});
      await page.waitForTimeout(400);
    }
    const t0 = Date.now();
    await page.keyboard.press("Space");
    await page.locator(".ov-controls .btn-primary", { hasText: "Pause" })
      .waitFor({ timeout: 15000 }).then(() => audition.push(Date.now() - t0))
      .catch(() => audition.push(null));
    await page.waitForTimeout(500);
    await page.keyboard.press("Space");
    await page.waitForTimeout(300);
  }
  await browser.close();
  console.log(JSON.stringify({
    draw_ms: { p50: pct(draws, 0.5), p95: pct(draws, 0.95), max: Math.max(...draws), n: draws.length },
    drag_frames: { n: frames.length, over16: frames.filter((f) => f > 16.7 * 1.5).length,
                   p95: pct(frames, 0.95) },
    audition_ms: audition,
  }));
})();
