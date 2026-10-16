"""Synthesise a library for the UX simulation: per track a full mix, a vocal
stem and an instrumental stem, plus meta.json with the ground truth (tempo,
key, and every section's label and span). usage: synth.py OUT_DIR N_TRACKS"""
import json, sys, random
from pathlib import Path
import numpy as np, soundfile as sf

SR = 22050
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(7)
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
    plan = [("intro",8,0,1,0),("verse",16,1,1,1),("chorus",16,1,1,1),("breakdown",8,1,0,1),("drop",16,0,1,1),("outro",8,0,1,0)]
    # scale time: keep tracks ~90-150 s
    total_bars = sum(p[1] for p in plan); dur = total_bars*bar
    n = int(dur*SR)+SR
    voc = np.zeros(n); ins = np.zeros(n)
    scale = [0,2,3,5,7,8,10] if minor else [0,2,4,5,7,9,11]
    prog = [0,5,3,4] if minor else [0,4,5,3]
    melody = [rng.choice(scale) for _ in range(8)]
    pos = 0.0
    truth = []
    for name, bars, has_voc, drums, chords in plan:
        truth.append(dict(label=name, start=round(pos, 3), end=round(pos + bars*bar, 3), sung=bool(has_voc)))
        loud = 1.0 if name in ("chorus","drop") else 0.6
        for b in range(bars):
            t0 = pos + b*bar; s0 = int(t0*SR)
            if drums:
                for k in range(4):
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
            if has_voc:
                for k in range(2):
                    m = melody[(b*2+k) % 8]; f0 = 261.626*2**(((tonic+m)%12)/12)
                    s = int((t0+k*2*beat)*SR); L = int(1.8*beat*SR); tt = np.arange(L)/SR
                    vib = 1+0.006*np.sin(2*np.pi*5.5*tt)
                    ph = 2*np.pi*np.cumsum(f0*vib)/SR
                    w = sum(np.sin(h*ph)/h for h in range(1,6))*np.minimum(tt/.03,1)*np.minimum((L/SR-tt)/.05,1)
                    voc[s:s+L] += .35*loud*w
        pos += bars*bar
    full = voc + ins
    pk = max(np.abs(full).max(), 1e-6)/0.9
    title = f"{rng.choice(ADJ)} {rng.choice(NOUN)}"
    artist = rng.choice(ART)
    stem = OUT/f"t{i:02d}"; stem.mkdir(exist_ok=True)
    for nm, x in (("full",full),("vocals",voc),("instrumental",ins)):
        sf.write(stem/f"{nm}.wav", (x/pk).astype(np.float32), SR)
    return dict(i=i, title=title, artist=artist, genre=g, bpm=bpm, key=NOTES[tonic]+("m" if minor else ""),
                dur=round(len(full)/SR,1), dir=str(stem.resolve()), sections=truth, year=rng.randint(2008,2026),
                plays=int(10**rng.uniform(3,7)))

meta = [track(i) for i in range(int(sys.argv[2]))]
(OUT/"meta.json").write_text(json.dumps(meta, indent=1))
print(len(meta), "tracks")
