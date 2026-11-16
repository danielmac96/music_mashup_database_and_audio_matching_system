#!/usr/bin/env python
"""scripts/synth_library.py — a synthetic library, for measuring at scale.

Writes a FRESH database (its own folder; never the live one) holding N tracks
as the pipeline would have left them — `songs`, `stems`, `features`,
`feature_arrays`, `sections` (the Essentia cut), `sections_allin1`,
`allin1_analysis`, `vocal_phrases`, `analysis_runs` — and no audio. Every value
comes from a seeded generator, so the same seed and size give the same library
on any machine, and track i is the same track at any size (each track draws
from its own stream).

    python scripts/synth_library.py --tracks 10000 --seed 7 --out /tmp/synth10k
    export MASHUP_DB_PATH=/tmp/synth10k/mashup.db \\
           MASHUP_AUDIO_ROOT=/tmp/synth10k/audio MASHUP_DATA_DIR=/tmp/synth10k/data

What the library looks like:
  * genres with their own tempo bands, keys spread unevenly round the Camelot
    wheel, timbre clustered by genre;
  * three kinds of track — full songs (a vocal AND a bed), instrumentals, and
    vocal-forward records — cut into bar-aligned sections with per-section
    chroma, band occupancy, sung range and bar lines;
  * most tracks cut by allin1 on a steady grid; a share allin1 never cut (the
    Essentia fallback) and a share with a wandering tempo (the timing gate has
    something to reject);
  * **planted drop-ins**: pairs built to be the easy win the engine exists to
    find — the bed is in the vocal's key or one Camelot step away, within 2 %
    of its tempo, and has a chorus *it originally sang over*, the same length
    as the vocal's chorus, with its own phrases starting on the same bars. They
    are listed in `<out>/synth_manifest.json` so a later change can measure
    how many it ranks first.

`--stub-audio` also writes a tiny silent WAV for every mix, vocal and
instrumental path, for code that only checks a file exists.

Not written: frame sidecars, embeddings, hook clips, candidates (run Score
library, or scripts/bench_scale.py, against the result).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIPELINE = "unified_4stem"
SEPARATOR = "demucs:htdemucs:4:unified"
ALLIN1_SECTION_BASE = 1000          # models.ALLIN1_SECTION_BASE (asserted below)
N_WAVEFORM = 360                    # analysis.essentia_groups.N_WAVEFORM
MANIFEST = "synth_manifest.json"

# name, share, BPM low, BPM high, the Discogs style its tags carry
GENRES = (
    ("House", 0.28, 120, 128, "Electronic---House"),
    ("Tech House", 0.12, 124, 130, "Electronic---Tech House"),
    ("Techno", 0.10, 128, 140, "Electronic---Techno"),
    ("Pop", 0.18, 95, 128, "Pop---Dance-pop"),
    ("Hip Hop", 0.10, 80, 100, "Hip Hop---Pop Rap"),
    ("Drum & Bass", 0.08, 170, 176, "Electronic---Drum n Bass"),
    ("Trance", 0.07, 132, 140, "Electronic---Trance"),
    ("Dubstep", 0.07, 138, 142, "Electronic---Dubstep"),
)
KINDS = (("song", 0.70), ("instrumental", 0.20), ("vocal_forward", 0.10))
ALLIN1_SHARE = 0.85                 # the rest keep only the Essentia cut
WANDERING_SHARE = 0.08              # tempo drifts: fails the timing gate
PLANTED_RELATIONS = ("same", "up", "down", "relative")

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl–Kessler profiles: how much of each pitch class a key holds.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PLANS = {
    "song": (
        (("intro", 8), ("verse", 16), ("chorus", 16), ("verse", 16), ("chorus", 16),
         ("breakdown", 8), ("chorus", 16), ("outro", 8)),
        (("intro", 16), ("verse", 16), ("build", 8), ("chorus", 16), ("verse", 16),
         ("build", 8), ("chorus", 16), ("bridge", 8), ("chorus", 16), ("outro", 16)),
        (("intro", 8), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8),
         ("bridge", 8), ("chorus", 16), ("outro", 8)),
    ),
    "instrumental": (
        (("intro", 16), ("build", 8), ("drop", 16), ("breakdown", 16), ("build", 8),
         ("drop", 16), ("outro", 16)),
        (("intro", 32), ("drop", 32), ("breakdown", 16), ("drop", 32), ("outro", 16)),
    ),
    "vocal_forward": (
        (("intro", 4), ("verse", 16), ("chorus", 8), ("verse", 16), ("chorus", 8),
         ("bridge", 8), ("chorus", 16), ("outro", 4)),
    ),
}
SUNG = {"verse", "chorus", "bridge"}
_ENERGY = {"intro": (0.30, 0.50), "outro": (0.25, 0.45), "verse": (0.50, 0.70),
           "chorus": (0.75, 1.00), "drop": (0.80, 1.00), "build": (0.55, 0.80),
           "breakdown": (0.30, 0.50), "bridge": (0.45, 0.65)}
# Typical stage times (ms per audio second) for `analysis_runs`, from the
# readme's measurements — what the queue's ETAs and a simulated import read.
_STAGE_MS_PER_SEC = {"download": 25.0, "stems": 620.0, "analysis": 54.0,
                     "structure": 14.0, "allin1": 300.0}


def _rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(index)])


def _pick(rng, table):
    r, acc = rng.random(), 0.0
    for row in table:
        acc += row[1]
        if r < acc:
            return row
    return table[-1]


def _camelot(tonic: int, minor: bool) -> str:
    from database.models import camelot_for
    return camelot_for(KEY_NAMES[tonic], "minor" if minor else "major")


def _key_from_camelot(code: str) -> Tuple[int, bool]:
    for tonic in range(12):
        for minor in (False, True):
            if _camelot(tonic, minor) == code:
                return tonic, minor
    raise ValueError(code)


def camelot_neighbour(code: str, relation: str) -> str:
    """The Camelot code one `relation` away: same, up / down a wheel step
    (8A→9A / 7A), or the relative major/minor (8A↔8B)."""
    num, letter = int(code[:-1]), code[-1]
    if relation == "up":
        num = num % 12 + 1
    elif relation == "down":
        num = (num - 2) % 12 + 1
    elif relation == "relative":
        letter = "B" if letter == "A" else "A"
    return f"{num}{letter}"


def _chroma(rng, tonic: int, minor: bool, noise: float, triad: float = 0.0) -> List[float]:
    base = (_MINOR if minor else _MAJOR).copy()
    if triad:                        # a voice leans on the tonic triad
        for step in (0, 3 if minor else 4, 7):
            base[step] += triad
    v = np.roll(base, tonic) + rng.normal(0.0, noise, 12)
    v = np.clip(v, 0.05, None)
    return [round(float(x), 4) for x in v / v.max()]


def _bands(rng, centre: float) -> List[float]:
    x = np.arange(8)
    v = np.exp(-0.5 * ((x - centre) / 1.6) ** 2) + rng.uniform(0.02, 0.10, 8)
    return [round(float(a), 4) for a in v / v.sum()]


# ── One track ─────────────────────────────────────────────────────────────────

def track_spec(seed: int, index: int, plant: Optional[dict] = None) -> dict:
    """Everything about track `index` (1-based id), from its own stream.

    `plant` turns it into the bed of a planted drop-in: {"vocal": the vocal
    track's spec, "relation": one of PLANTED_RELATIONS}.
    """
    rng = _rng(seed, index)
    genre, _share, lo, hi, style = _pick(rng, GENRES)
    kind = _pick(rng, KINDS)[0]
    minor = rng.random() < 0.62
    # Keys are not uniform: a few tonics hold most dance records.
    tonic = int(rng.choice(12, p=np.array([9, 5, 8, 4, 7, 9, 5, 9, 6, 11, 5, 6]) / 84.0))
    bpm = float(rng.integers(lo, hi + 1)) + float(rng.normal(0.0, 0.03))
    allin1 = rng.random() < ALLIN1_SHARE
    wandering = rng.random() < WANDERING_SHARE
    plan = PLANS[kind][int(rng.integers(len(PLANS[kind])))]
    quality_v = float(rng.beta(6, 2))
    quality_i = float(rng.beta(7, 2))
    register = float(rng.uniform(55, 72))            # sung median, MIDI

    if plant:
        vocal = plant["vocal"]
        kind, allin1, wandering = "song", True, False
        bpm = vocal["bpm"] * (1.0 + float(rng.uniform(-0.02, 0.02)))
        tonic, minor = _key_from_camelot(
            camelot_neighbour(vocal["camelot"], plant["relation"]))
        bars = vocal["sections"][vocal["planted_section"]]["bar_count"]
        plan = (("intro", 8), ("verse", 16), ("chorus", bars), ("verse", 16),
                ("chorus", bars), ("outro", 8))
        quality_v, quality_i = max(quality_v, 0.8), max(quality_i, 0.85)
        register = vocal["register"] + float(rng.uniform(-2.0, 2.0))
        genre, style = vocal["genre"], vocal["style"]

    bar = 240.0 / bpm
    t = float(rng.uniform(0.02, 0.45))
    sections, phrases = [], []
    wobble_phase = float(rng.uniform(0, 6.28))
    bar_no = 0
    for si, (label, bars) in enumerate(plan):
        downbeats = []
        for _ in range(int(bars)):
            downbeats.append(t)
            step = bar
            if wandering:
                step *= 1.0 + 0.025 * np.sin(bar_no / 3.0 + wobble_phase)
            t += step + float(rng.normal(0.0, 0.0015))
            bar_no += 1
        start, end = downbeats[0], t
        beats = []
        for k, d in enumerate(downbeats):
            nxt = downbeats[k + 1] if k + 1 < len(downbeats) else end
            beats += [d + (nxt - d) * q / 4.0 for q in range(4)]
        sung = label in SUNG and kind != "instrumental"
        lo_e, hi_e = _ENERGY[label]
        energy = float(rng.uniform(lo_e, hi_e))
        presence = float(rng.uniform(0.62, 0.96)) if sung else float(rng.uniform(0.01, 0.12))
        if kind == "vocal_forward" and sung:
            presence = max(presence, 0.85)
        activity = round(presence * float(rng.uniform(0.80, 0.97)), 4)
        sec = {
            "start_sec": round(start, 4), "end_sec": round(end, 4), "label": label,
            "engine_label": {"breakdown": "break", "drop": "inst", "build": "inst"}.get(label, label),
            "energy": round(energy, 4), "vocal_presence": round(presence, 4),
            "repetition": sum(1 for lab, _b in plan if lab == label),
            "confidence": round(float(rng.uniform(0.6, 0.97)), 3),
            "downbeat_conf": round(float(rng.uniform(0.5, 0.95)), 3),
            "chroma": _chroma(rng, tonic, minor, 0.5),
            "bass_chroma": _chroma(rng, tonic, minor, 0.7, triad=3.0),
            "chroma_bed": _chroma(rng, tonic, minor, 0.5, triad=1.0),
            "key": KEY_NAMES[tonic], "mode": "minor" if minor else "major",
            "camelot": _camelot(tonic, minor),
            "key_confidence": round(float(rng.uniform(0.3, 0.9)), 3),
            "bpm": round(bpm, 2), "bpm_source": "section",
            "bpm_confidence": round(float(rng.uniform(0.6, 0.98)), 3),
            "energy_absolute": round(energy * 0.2, 4),
            "energy_slope": round(float(rng.normal(0, 0.05)), 4),
            "energy_trend": "rising" if label == "build" else "flat",
            "beat_times": [round(b, 3) for b in beats],
            "downbeats": [round(d, 4) for d in downbeats],
            "beat_count": len(beats), "bar_count": float(bars), "beats_per_bar": 4,
            "phrase_length_bars": 8.0 if bars >= 8 else float(bars),
            "section_class": ("vocal" if presence >= 0.8 else "mixed") if sung else "instrumental",
            "vocal_activity": activity,
            "band_energy_bed": _bands(rng, 2.6),
        }
        if sung:
            sec["chroma_vocal"] = _chroma(rng, tonic, minor, 0.4, triad=3.0)
            sec["band_energy_vocal"] = _bands(rng, 4.4)
            spread = float(rng.uniform(3.0, 7.0))
            sec["f0"] = {"median_midi": round(register, 2),
                         "p10_midi": round(register - spread, 2),
                         "p90_midi": round(register + spread, 2),
                         "range_st": round(2 * spread, 2), "voiced": round(activity, 4)}
            # A phrase every 4 or 8 bars, each with its own pickup.
            every = 8 if bars >= 16 and rng.random() < 0.5 else 4
            for b0 in range(0, int(bars), every):
                r = rng.random()
                pickup = (0.0 if r < 0.5 else -bar / 8 if r < 0.75
                          else -bar / 4 if r < 0.9 else bar / 16)
                pickup += float(rng.normal(0.0, 0.004))
                d = downbeats[b0]
                last = min(b0 + every, int(bars)) - 1
                phrases.append({
                    "section": si, "bar": b0, "onset_sec": round(d + pickup, 4),
                    "downbeat_sec": round(d, 4), "pickup_sec": round(pickup, 4),
                    "end_sec": round(downbeats[last] + bar * 0.85, 4),
                    "bars": min(every, int(bars) - b0),
                    "confidence": round(float(rng.uniform(0.5, 0.95)), 3)})
        sections.append(sec)

    planted_section = None
    choruses = [i for i, s in enumerate(sections)
                if s["label"] == "chorus" and "chroma_vocal" in s]
    if choruses:
        planted_section = choruses[0]

    if plant:
        # The bed's chorus is where ITS OWN vocal sang, on the same bars the
        # incoming vocal's phrases start — the vacated slot.
        vocal = plant["vocal"]
        v_sec = vocal["planted_section"]
        want = [p for p in vocal["phrases"] if p["section"] == v_sec]
        phrases = [p for p in phrases if p["section"] != planted_section]
        sec = sections[planted_section]
        sec["vocal_presence"] = max(sec["vocal_presence"], 0.85)
        sec["vocal_activity"] = round(max(sec["vocal_activity"], 0.78), 4)
        sec["energy"] = max(sec["energy"], 0.85)
        for p in want:
            d = sec["downbeats"][p["bar"]]
            pickup = p["pickup_sec"] * (vocal["bpm"] / bpm) + float(rng.normal(0.0, 0.003))
            phrases.append({**p, "section": planted_section,
                            "onset_sec": round(d + pickup, 4), "downbeat_sec": d,
                            "pickup_sec": round(pickup, 4),
                            "end_sec": round(d + p["bars"] * bar * 0.9, 4)})
        phrases.sort(key=lambda p: p["onset_sec"])

    duration = sections[-1]["end_sec"] + float(rng.uniform(0.5, 3.0))
    mfcc_centre = np.random.default_rng([int(seed), 10_000_019, GENRES.index(
        next(g for g in GENRES if g[0] == genre))]).normal(0.0, 14.0, 13)
    mfcc_centre[0] = -60.0

    def _mfcc(shift: float) -> List[float]:
        v = mfcc_centre + rng.normal(0.0, 7.0, 13)
        v[0] += shift + float(rng.normal(0.0, 25.0))
        return [round(float(x), 3) for x in v]

    plays = int(rng.lognormal(10.0, 2.0))
    return {
        "id": index, "seed": seed, "kind": kind, "genre": genre, "style": style,
        "title": f"Synth {index:05d} ({kind.replace('_', ' ')})",
        "artist": f"Artist {int(rng.integers(1, 2500)):04d}",
        "bpm": round(bpm, 2), "tonic": tonic, "minor": minor,
        "camelot": _camelot(tonic, minor), "allin1": allin1, "wandering": wandering,
        "duration": round(duration, 2), "sections": sections, "phrases": phrases,
        "planted_section": planted_section, "register": register,
        "quality": {"vocals": round(quality_v if kind != "instrumental"
                                    else quality_v * 0.4, 4),
                    "instrumental": round(quality_i, 4)},
        "bpm_confidence": round(float(rng.beta(8, 2)) * (0.6 if wandering else 1.0), 4),
        "key_confidence": round(float(rng.beta(4, 3)), 4),
        "loudness": {"full": round(float(rng.uniform(0.10, 0.25)), 5),
                     "vocals": round(float(rng.uniform(0.03, 0.12))
                                     * (0.15 if kind == "instrumental" else 1.0), 5),
                     "instrumental": round(float(rng.uniform(0.08, 0.20)), 5)},
        "mfcc": {"full": _mfcc(0.0), "vocals": _mfcc(-80.0), "instrumental": _mfcc(-15.0)},
        "bands": {"full": _bands(rng, 3.2), "vocals": _bands(rng, 4.4),
                  "instrumental": _bands(rng, 2.6)},
        "residual_vocal": round(float(rng.beta(2, 9)), 4),
        "lufs": round(float(rng.normal(-11.0, 2.5)), 2),
        "plays": plays, "likes": int(plays * float(rng.uniform(0.005, 0.04))),
        "year": int(rng.integers(1998, 2027)),
        "waveform": [round(float(x), 4) for x in
                     np.clip(np.abs(rng.normal(0.55, 0.2, N_WAVEFORM)), 0.0, 1.0)],
    }


def plan_planted(seed: int, n_tracks: int, planted: int) -> Dict[int, Tuple[int, str]]:
    """{bed id: (vocal id, relation)} — which tracks are rebuilt as the bed of
    a drop-in, and for which vocal. Vocals are tracks whose own stream makes
    them a steady, allin1-cut record with a sung chorus; beds are any other
    track, taken from the far end of the library so the two never overlap."""
    rng = np.random.default_rng([int(seed), 20_000_003])
    order = [int(i) + 1 for i in rng.permutation(n_tracks)]
    vocals: List[int] = []
    for i in order:
        if len(vocals) >= planted:
            break
        s = track_spec(seed, i)
        if s["kind"] != "instrumental" and s["allin1"] and not s["wandering"] \
                and s["planted_section"] is not None:
            vocals.append(i)
    taken = set(vocals)
    beds = [i for i in reversed(order) if i not in taken][:len(vocals)]
    return {bed: (vocal, PLANTED_RELATIONS[k % len(PLANTED_RELATIONS)])
            for k, (vocal, bed) in enumerate(zip(vocals, beds))}


# ── Writing ───────────────────────────────────────────────────────────────────

def _stub_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)


def _write_track(conn: sqlite3.Connection, s: dict, audio: Path, stub: bool,
                 section_cols, section_values) -> None:
    sid = s["id"]
    raw = audio / "full_song" / f"synth_{sid:05d}.mp3"
    conn.execute(
        """INSERT INTO songs (id, title, artist, source_url, source, duration_secs,
               genre, raw_path, status, artist_id, track_id, duration_str,
               upload_date, likes, plays, tags, release_year, quick_state,
               pairs_input_at, separation_pipeline, allin1_state, origin_url,
               origin_duration_secs)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),?,?,?,?)""",
        (sid, s["title"], s["artist"], f"https://soundcloud.com/synth/track-{sid:05d}",
         "soundcloud", s["duration"], s["genre"], str(raw), "analysed",
         f"a{sid % 2500}", f"9{sid:08d}",
         f"{int(s['duration'] // 60)}:{int(s['duration'] % 60):02d}",
         f"{s['year']}0615", s["likes"], s["plays"], json.dumps([s["genre"].lower()]),
         s["year"], "done", PIPELINE, "done" if s["allin1"] else "failed",
         f"https://soundcloud.com/synth/track-{sid:05d}", s["duration"]))
    if stub:
        _stub_wav(raw.with_suffix(".wav"))

    stem_dir = audio / "stems" / f"synth_{sid:05d}"
    for stem in ("vocals", "drums", "bass", "other", "instrumental"):
        q = s["quality"].get(stem)
        path = stem_dir / f"{stem}.flac"
        conn.execute(
            "INSERT INTO stems (song_id, stem_type, file_path, separator, quality, "
            "bleed, hf_loss, noise_floor, content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, stem, str(path), SEPARATOR, q,
             None if q is None else round(1.0 - q, 4), None if q is None else 0.1,
             None if q is None else 0.02, f"synth-{s['seed']}-{sid}-{stem}"))
        if stub and stem in ("vocals", "instrumental"):
            _stub_wav(path.with_suffix(".wav"))

    hook = next((x for x in s["sections"] if x["label"] in ("chorus", "drop")),
                s["sections"][0])
    for stem in ("full", "vocals", "instrumental"):
        conn.execute(
            """INSERT INTO features (song_id, stem_type, bpm, bpm_confidence, key,
                   mode, camelot, loudness_rms, energy, mfcc_json, spectral_centroid,
                   spectral_rolloff, zero_crossing_rate, band_energy_json,
                   residual_vocal_ratio, key_confidence, beat_phase, hook_start,
                   hook_end, hook_role, analyzer, tuning_hz, lufs, tags_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, stem, s["bpm"], s["bpm_confidence"], KEY_NAMES[s["tonic"]],
             "minor" if s["minor"] else "major", s["camelot"], s["loudness"][stem],
             round(s["loudness"][stem] * 4.0, 4), json.dumps(s["mfcc"][stem]),
             1800.0 + 40.0 * s["mfcc"][stem][1], 5200.0, 0.07,
             json.dumps(s["bands"][stem]),
             s["residual_vocal"] if stem == "instrumental" else None,
             s["key_confidence"], 0, hook["start_sec"],
             hook["start_sec"] + 16 * 240.0 / s["bpm"],
             "vocal" if stem == "vocals" else "bed", "essentia", 440.0,
             s["lufs"] - (6.0 if stem == "vocals" else 0.0),
             json.dumps({"genre": [{"label": s["style"], "p": 0.6}]})
             if stem == "full" else None))
        conn.execute(
            "INSERT INTO feature_arrays (song_id, stem_type, name, dtype, dim, data) "
            "VALUES (?,?,?,?,?,?)",
            (sid, stem, "waveform_rms", "float32", N_WAVEFORM,
             np.asarray(s["waveform"], dtype=np.float32).tobytes()))
    beats = np.asarray([b for x in s["sections"] for b in x["beat_times"]], dtype=np.float32)
    downbeats = np.asarray([d for x in s["sections"] for d in x["downbeats"]], dtype=np.float32)
    conn.execute(
        "INSERT INTO feature_arrays (song_id, stem_type, name, dtype, dim, data) "
        "VALUES (?,?,?,?,?,?)",
        (sid, "full", "beat_times", "float32", int(beats.size), beats.tobytes()))

    ph = ", ".join("?" * len(section_cols))
    conn.executemany(
        f"INSERT INTO sections (song_id, section_index, {', '.join(section_cols)}) "
        f"VALUES (?,?,{ph})",
        [(sid, i, *section_values(x)) for i, x in enumerate(s["sections"])])
    grid = "allin1" if s["allin1"] else "essentia"
    if s["allin1"]:
        conn.executemany(
            f"INSERT INTO sections_allin1 (song_id, pipeline, section_index, "
            f"engine_label, downbeat_conf, {', '.join(section_cols)}) "
            f"VALUES (?,?,?,?,?,{ph})",
            [(sid, PIPELINE, ALLIN1_SECTION_BASE + i, x["engine_label"],
              x["downbeat_conf"], *section_values(x))
             for i, x in enumerate(s["sections"])])
        conn.execute(
            """INSERT INTO allin1_analysis (song_id, pipeline, content_hash, model,
                   version, pooling, demixed_by, bpm, beats, downbeats,
                   beat_positions, segments_json, conf_mean, conf_min, ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, PIPELINE, f"synth-{s['seed']}-{sid}", "harmonix-fold0", "1.1.0",
             None, "pipeline", round(s["bpm"] * 1.0004, 2), beats.tobytes(),
             downbeats.tobytes(),
             np.tile(np.array([1, 2, 3, 4], dtype=np.int8), downbeats.size).tobytes(),
             json.dumps([[x["start_sec"], x["end_sec"], x["engine_label"]]
                         for x in s["sections"]]),
             0.7, 0.3, s["duration"] * _STAGE_MS_PER_SEC["allin1"]))
    conn.executemany(
        """INSERT INTO vocal_phrases (song_id, pipeline, grid_source, phrase_index,
               onset_sec, downbeat_sec, pickup_sec, end_sec, bars, confidence,
               content_hash, version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(sid, PIPELINE, grid, i, p["onset_sec"], p["downbeat_sec"], p["pickup_sec"],
          p["end_sec"], p["bars"], p["confidence"], f"synth-{sid}", 1)
         for i, p in enumerate(s["phrases"])])
    conn.executemany(
        "INSERT INTO analysis_runs (song_id, stem_type, grp, tier, analyzer, ms, "
        "audio_secs, ok) VALUES (?,?,?,?,?,?,?,1)",
        [(sid, None, grp, None, "synth", s["duration"] * per_sec, s["duration"])
         for grp, per_sec in _STAGE_MS_PER_SEC.items()
         if grp != "allin1" or s["allin1"]])


def generate(out_dir, n_tracks: int, seed: int = 7, planted: Optional[int] = None,
             stub_audio: bool = False, force: bool = False, progress=None) -> dict:
    """Write the library under `out_dir` and return its manifest (also saved
    as synth_manifest.json): the paths, the counts and the planted pairs."""
    from database import schema
    from database import models
    assert models.ALLIN1_SECTION_BASE == ALLIN1_SECTION_BASE

    out = Path(out_dir)
    db, audio, data = out / "mashup.db", out / "audio", out / "data"
    if db.exists():
        if not force:
            raise FileExistsError(f"{db} exists — pass force=True / --force to replace it")
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
    for folder in (out, audio, data):
        folder.mkdir(parents=True, exist_ok=True)
    if planted is None:
        planted = max(1, n_tracks // 50)
    planted = max(0, min(int(planted), n_tracks // 2))
    plants = plan_planted(seed, n_tracks, planted) if planted else {}
    vocal_specs = {v: track_spec(seed, v) for v, _rel in plants.values()}

    started = time.perf_counter()
    conn = sqlite3.connect(db)
    pairs, kinds = [], {}
    try:
        schema.apply(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("BEGIN")
        for i in range(1, n_tracks + 1):
            plant = None
            if i in plants:
                vocal_id, relation = plants[i]
                plant = {"vocal": vocal_specs[vocal_id], "relation": relation}
            spec = vocal_specs.get(i) or track_spec(seed, i, plant)
            _write_track(conn, spec, audio, stub_audio,
                         models.SECTION_INSERT_COLUMNS, models.section_values)
            kinds[spec["kind"]] = kinds.get(spec["kind"], 0) + 1
            if plant:
                v = plant["vocal"]
                pairs.append({
                    "vocal_song_id": v["id"], "inst_song_id": i,
                    "vocal_section": ALLIN1_SECTION_BASE + v["planted_section"],
                    "inst_section": ALLIN1_SECTION_BASE + spec["planted_section"],
                    "relation": plant["relation"],
                    "vocal_camelot": v["camelot"], "inst_camelot": spec["camelot"],
                    "vocal_bpm": v["bpm"], "inst_bpm": spec["bpm"],
                    "bars": spec["sections"][spec["planted_section"]]["bar_count"]})
            if progress and i % 500 == 0:
                progress(i, n_tracks)
        conn.execute("COMMIT")
    finally:
        conn.close()

    manifest = {
        "tracks": n_tracks, "seed": seed, "kinds": kinds,
        "db_path": str(db), "audio_root": str(audio), "data_dir": str(data),
        "stub_audio": bool(stub_audio), "planted": pairs,
        "seconds": round(time.perf_counter() - started, 2),
        "db_bytes": db.stat().st_size,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="folder for the library (created)")
    ap.add_argument("--planted", type=int, default=None,
                    help="planted drop-in pairs (default: tracks / 50)")
    ap.add_argument("--stub-audio", action="store_true",
                    help="write a tiny silent WAV for each mix / vocal / instrumental")
    ap.add_argument("--force", action="store_true", help="replace an existing library")
    args = ap.parse_args(argv)

    def _progress(done, total):
        print(f"  {done}/{total} tracks", file=sys.stderr)

    m = generate(args.out, args.tracks, args.seed, args.planted, args.stub_audio,
                 args.force, _progress)
    print(f"{m['tracks']} tracks ({m['kinds']}), {len(m['planted'])} planted drop-ins, "
          f"{m['db_bytes'] / 1e6:.1f} MB in {m['seconds']} s")
    print(f"export MASHUP_DB_PATH={m['db_path']} MASHUP_AUDIO_ROOT={m['audio_root']} "
          f"MASHUP_DATA_DIR={m['data_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
