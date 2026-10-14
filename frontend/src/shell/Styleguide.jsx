import { camelotColor, cssVar, TIERS } from "../theme";
import {
  BpmChip, CompatMeter, Disclosure, GenreTag, KeyChip, MiniWave, Sparkline,
  StatusPill, TierBadge,
} from "../components/ui/primitives";

// ?styleguide — dev builds only. The tokens from styles.css :root and the
// shared primitives, drawn by the same code the screens use, so a token change
// shows up here first. Swatches read the live variable (theme.cssVar); nothing
// here repeats a hex.

const SWATCHES = [
  ["Background", "--bg", "App ground"], ["Zebra", "--zebra", "Alternate rows"],
  ["Sunk", "--bg-sunk", "Docks and side panels"], ["Well", "--well", "Inputs and wells"],
  ["Bar", "--bar", "Rail, sticky headers, player"], ["Card", "--card", "Cards"],
  ["Panel", "--panel", "Buttons and panels"], ["Raised", "--raised", "Popovers, hover"],
  ["Line", "--line", "Dividers"], ["Line 2", "--line-2", "Control borders"],
  ["Text", "--fg", "Primary text", true], ["Text 2", "--fg-2", "Secondary text", true],
  ["Muted", "--muted", "Labels, units", true], ["Faint", "--faint", "Hints", true],
  ["Vocal", "--vocal", "Every vocal chip, lane, wave", true],
  ["Bed", "--bed", "Every instrumental chip, lane, wave", true],
  ["Accent", "--accent", "Focus, selection, one primary action", true],
  ["OK", "--ok", "Status: done", true], ["Warn", "--warn", "Status: attention", true],
  ["Fail", "--fail", "Status: failed", true], ["Running", "--run", "Status: in progress", true],
  ["Stars", "--star", "Ratings", true], ["Playhead", "--playhead", "Transport position", true],
];

const TYPE = [
  ["22 / 600", "sg-t22", "Track title"],
  ["16 / 600", "sg-t16", "Screen title"],
  ["13 / 500", "sg-t13", "Base UI text, row titles"],
  ["12 / 400", "sg-t12", "Secondary, artist"],
  ["15 mono", "sg-tm15 mono", "124.0 BPM · −8.4 LUFS"],
  ["10 mono caps", "caps", "COLUMN · LABEL"],
];

const SPACE = [4, 8, 12, 16, 24, 32, 48];
const SECTIONS = [["Intro/Outro", "--sec-intro"], ["Verse", "--sec-verse"],
  ["Pre/Build", "--sec-build"], ["Chorus/Drop", "--sec-chorus"], ["Break/Bridge", "--sec-break"]];

// Deterministic sample bars so the page never shifts between renders.
const wave = (n, k) => Array.from({ length: n },
  (_, i) => 0.2 + 0.8 * Math.abs(Math.sin(i * k)) * (0.6 + 0.4 * Math.abs(Math.cos(i * 1.7))));

export function Styleguide() {
  return (
    <div className="styleguide">
      <header className="sg-head">
        <span className="caps">Revamp step 0 · ?styleguide</span>
        <h1>Styleguide</h1>
        <p className="muted">Tokens from styles.css and the shared primitives. Every screen
          is built only from these.</p>
      </header>

      <section>
        <h2>Colour</h2>
        <div className="sg-swatches">
          {SWATCHES.map(([name, v, use, light]) => (
            <div key={v} className="sg-swatch">
              <div className={`chip${light ? " light" : ""}`} style={{ background: `var(${v})` }}>
                Aa 124.0
              </div>
              <div className="row"><span>{name}</span>
                <span className="mono muted">{v} {cssVar(v)}</span></div>
              <span className="muted use">{use}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>Camelot wheel · one 24-colour set everywhere</h2>
        <p className="muted">Mixed In Key-style hues. A (minor) is the deeper tone, B (major)
          the lighter one. The chip always carries its label, so colour is never the only cue.</p>
        {["A", "B"].map((l) => (
          <div key={l} className="sg-keys">
            {Array.from({ length: 12 }, (_, i) => `${i + 1}${l}`).map((k) => (
              <span key={k} className="key-chip sg-key" style={{ background: camelotColor(k) }}>{k}</span>
            ))}
          </div>
        ))}
        <div className="sg-keys">
          {SECTIONS.map(([n, v]) => (
            <span key={v} className="sg-sec" style={{ background: `var(${v})` }}>{n}</span>
          ))}
        </div>
      </section>

      <section className="sg-cols">
        <div>
          <h2>Type · Space Grotesk + IBM Plex Mono</h2>
          {TYPE.map(([spec, cls, sample]) => (
            <div key={spec} className="sg-type">
              <span className="mono muted spec">{spec}</span>
              <span className={cls}>{sample}</span>
            </div>
          ))}
        </div>
        <div>
          <h2>Space, radius, elevation, motion</h2>
          <div className="sg-space">
            {SPACE.map((n) => (
              <div key={n}><span style={{ width: n, height: n }} /><span className="mono muted">{n}</span></div>
            ))}
          </div>
          <div className="sg-radii">
            <div className="r3">r 3</div><div className="r6">r 6</div>
            <div className="pop-demo">pop</div><div className="modal">modal</div>
          </div>
          <p className="muted">Motion: 120 ms ease for hover, width and colour. Nothing animates
            longer than 150 ms. Running state pulses at 1.4 s.</p>
        </div>
      </section>

      <section>
        <h2>Primitives</h2>
        <div className="sg-prims">
          <div className="sg-prim">
            <span className="mono muted">KeyChip · BpmChip · GenreTag</span>
            <div className="sg-row">
              <KeyChip camelot="8A" /><KeyChip camelot={null} />
              <BpmChip bpm={124} confidence={0.92} />
              <BpmChip bpm={72} fix="×2?" onFix={() => {}} />
              <GenreTag genre="house" /><GenreTag genre="R&B" differs />
            </div>
          </div>
          <div className="sg-prim">
            <span className="mono muted">TierBadge · CompatMeter</span>
            <div className="sg-row">{TIERS.map((t) => <TierBadge key={t.id} tier={t.id} />)}</div>
            <CompatMeter name="VOI" value={0.88} />
            <CompatMeter name="HARM" value={0.67} />
            <CompatMeter name="RHY" value={0.42} />
            <CompatMeter name="ENGY" value={null} title="Not measured" />
          </div>
          <div className="sg-prim">
            <span className="mono muted">StatusPill · ok / warn / fail / running</span>
            <div className="sg-row">
              {["analysed", "downloaded", "stemmed", "queued", "error_download"].map((s) => (
                <StatusPill key={s} status={s} />
              ))}
              <span className="status-pill" style={{ color: "var(--star)", background: "var(--warn-bg)" }}>
                ⚠ Low key confidence</span>
            </div>
          </div>
          <div className="sg-prim">
            <span className="mono muted">MiniWave · Sparkline</span>
            <MiniWave peaks={wave(40, 0.7)} role="vocal" />
            <MiniWave peaks={wave(40, 0.35)} role="bed" />
            <Sparkline values={wave(24, 0.4)} />
          </div>
          <div className="sg-prim">
            <span className="mono muted">Disclosure · toast · status readout</span>
            <Disclosure title="Stems & quality" meta="Demucs 2-stem · bleed 4%">
              <span className="muted">Disclosure body.</span>
            </Disclosure>
            <div className="toast sg-static"><span className="dot">●</span>Pair hidden
              <button className="toast-action">Undo</button></div>
            <div className="rail-pipe sg-static"><span className="running-dot" />
              <span className="rail-pipe-txt">3 running · 41 waiting · ~18 min</span></div>
          </div>
        </div>
      </section>
    </div>
  );
}
