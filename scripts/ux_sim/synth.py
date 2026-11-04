"""Synthesise a library for the UX simulation: per track a full mix, a vocal
stem and an instrumental stem, plus meta.json with the ground truth (tempo,
key, and every section's label and span). usage: synth.py OUT_DIR N_TRACKS [--pickups]

--pickups (the allin1 simulation, readme §8): the voice sings three bars of
every four, each phrase entering at a per-track pickup (−1 beat, −½ beat, on
the downbeat or +80 ms late); a few tracks drift 2% in tempo or run in 3/4,
for the timing gate. meta.json then also carries every beat, downbeat, phrase
start and the pickup. Without the flag the library is the one it always was."""
import json, sys, random
from pathlib import Path
import numpy as np, soundfile as sf

SR = 22050
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(7)
PICKUPS = "--pickups" in sys.argv
rng2 = random.Random(11)   # only drawn from with --pickups, so the default library is unchanged
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
GEN = [("House",124,128),("Tech House",124,127),("Pop",100,124),("Hip-Hop",86,96),
       ("Dubstep",140,150),("Future Bass",145,155),("Drum & Bass",170,176),("Indie Dance",112,120),("Disco",115,122)]
ADJ = "Midnight Golden Neon Velvet Electric Hollow Paper Crystal Silver Wild Broken Lost Endless Fading Sunset Ocean Static Burning".split()
NOUN = "Signal Hearts Skyline Echoes Motion Fever Lights Gravity Horizon Paradise Rhythm Shadows Waves Fire Dreams Satellite Stranger Avenue".split()
ART = "Kade Monroe|Lumen Park|Ivy Sterling|North Atlas|DJ Rook|Sola Vee|The Arcades|Mira Kane|Bassline Jack|Echo Ridge|Nova Hart|Coastal Kid|Rhea Lux|Yung Prism|Atlas & Wren".split('|')

def env(n, a=0.005, d=0.15):
    t = np.arange(n)/SR; return np.minimum(t/a, 1)*np.exp(-t/d)

def track(i):
    g, lo, hi = rng.choice(GEN); bpm = round(rng.uniform(lo, hi), 1)
    tonic = rng.randrange(12); minor = rng.random() < .6
    beat = 60/bpm; bar = 4*beat
    meter, drift, pickup = 4, 0.0, 0.0
    if PICKUPS:
        pickup = rng2.choice([-1.0, -0.5, 0.0, 0.0, None])   # in beats; None = +80 ms
        pickup = 0.08 if pickup is None else pickup * beat
        odd = rng2.random()
        meter = 3 if odd < 0.08 else 4
        drift = 0.02 if 0.08 <= odd < 0.18 else 0.0
    plan = [("intro",8,0,1,0),("verse",16,1,1,1),("chorus",16,1,1,1),("breakdown",8,1,0,1),("drop",16,0,1,1),("outro",8,0,1,0)]
    # scale time: keep tracks ~90-150 s
    total_bars = sum(p[1] for p in plan)
    # Every bar's start and beat length (a drifting track speeds up linearly).
    bar_t, beat_len, t = [], [], 0.0
    for b in range(total_bars):
        bl = 60/(bpm*(1 + drift*b/max(1, total_bars-1)))
        bar_t.append(t); beat_len.append(bl); t += meter*bl
    dur = t
    n = int(dur*SR)+SR
    voc = np.zeros(n); ins = np.zeros(n)
    scale = [0,2,3,5,7,8,10] if minor else [0,2,4,5,7,9,11]
    prog = [0,5,3,4] if minor else [0,4,5,3]
    melody = [rng.choice(scale) for _ in range(8)]
    pos = 0.0
    truth = []
    gb = 0                      # global bar index
    beats_all, phrase_starts = [], []
    for name, bars, has_voc, drums, chords in plan:
        end_t = bar_t[gb+bars] if gb+bars < total_bars else dur
        truth.append(dict(label=name, start=round(bar_t[gb], 3), end=round(end_t, 3), sung=bool(has_voc)))
        loud = 1.0 if name in ("chorus","drop") else 0.6
        for b in range(bars):
            t0 = bar_t[gb]; s0 = int(t0*SR); beat = beat_len[gb]; bar = meter*beat
            beats_all += [round(t0 + k*beat, 4) for k in range(meter)]
            gb += 1
            if drums:
                for k in range(meter):
                    s = int((t0+k*beat)*SR); L = int(.25*SR)
                    tt = np.arange(L)/SR; f = 50+120*np.exp(-tt*30)
                    ins[s:s+L] += .9*loud*np.sin(2*np.pi*np.cumsum(f)/SR)*env(L,.001,.12)
                    sh = int((t0+k*beat+beat/2)*SR); Lh=int(.05*SR)
                    ins[sh:sh+Lh] += .15*np.random.randn(Lh)*env(Lh,.001,.02)
            deg = prog[b % 4]; root = tonic + scale[deg % 7]
            # bass
            Lb = int(bar*SR); tt = np.arange(Lb)/SR; fb = 32.703*2**(((root)%12)/12)
            ins[s0:s0+Lb] += .35*loud*np.sign(np.sin(2*np.pi*fb*tt))*0.5*(1-np.exp(-tt*20))
            if chords:
                for iv in (0, 3 if minor else 4, 7):
                    fc = 130.813*2**(((root+iv)%12)/12)
                    ins[s0:s0+Lb] += .07*np.sin(2*np.pi*fc*tt)*(0.5+0.5*np.sin(np.pi*tt/bar))
            if has_voc and PICKUPS and b % 4 == 3:
                pass            # one bar of rest closes every 4-bar phrase
            elif has_voc:
                if PICKUPS and b % 4 == 0 and (not phrase_starts or phrase_starts[-1]["bar"] != gb-1):
                    phrase_starts.append(dict(bar=gb-1, downbeat=round(t0, 4), onset=round(t0+pickup, 4)))
                for k in range(2):
                    m = melody[(b*2+k) % 8]; f0 = 261.626*2**(((tonic+m)%12)/12)
                    s = int((t0+k*2*beat+(pickup if PICKUPS else 0))*SR); L = int(1.8*beat*SR); tt = np.arange(L)/SR
                    s = max(0, s)
                    vib = 1+0.006*np.sin(2*np.pi*5.5*tt)
                    ph = 2*np.pi*np.cumsum(f0*vib)/SR
                    w = sum(np.sin(h*ph)/h for h in range(1,6))*np.minimum(tt/.03,1)*np.minimum((L/SR-tt)/.05,1)
                    voc[s:s+L] += .35*loud*w
    full = voc + ins
    pk = max(np.abs(full).max(), 1e-6)/0.9
    title = f"{rng.choice(ADJ)} {rng.choice(NOUN)}"
    artist = rng.choice(ART)
    stem = OUT/f"t{i:02d}"; stem.mkdir(exist_ok=True)
    for nm, x in (("full",full),("vocals",voc),("instrumental",ins)):
        sf.write(stem/f"{nm}.wav", (x/pk).astype(np.float32), SR)
    return dict(i=i, title=title, artist=artist, genre=g, bpm=bpm, key=NOTES[tonic]+("m" if minor else ""),
                dur=round(len(full)/SR,1), dir=str(stem.resolve()), sections=truth, year=rng.randint(2008,2026),
                plays=int(10**rng.uniform(3,7)),
                **(dict(meter=meter, drift=drift, pickup_ms=round(pickup*1000, 1),
                        beats=beats_all, downbeats=[round(x, 4) for x in bar_t],
                        phrase_starts=phrase_starts) if PICKUPS else {}))

meta = [track(i) for i in range(int(sys.argv[2]))]
(OUT/"meta.json").write_text(json.dumps(meta, indent=1))
print(len(meta), "tracks")
