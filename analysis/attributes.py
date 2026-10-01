"""
analysis/attributes.py — every attribute a track carries, defined once.

The Analysis panel lists these, the Library shows the ones toggled on as
columns, and Track detail as a card; all three read this catalogue and
`extract`, so a label, a unit or where a value comes from cannot differ
between them. Show/hide only: nothing here computes a feature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

CATEGORIES = ("Tempo & grid", "Key & harmony", "Loudness", "Timbre",
              "Genre & tags", "Mood", "Vocals & stems")
HIST_BINS = 12
TOP_CATEGORIES = 8


@dataclass(frozen=True)
class Attr:
    id: str
    label: str
    short: str
    category: str
    source: str
    kind: str            # number | category | top
    unit: str
    decimals: int
    description: str
    get: Callable[[dict, Optional[dict], Optional[dict]], Any]


def _col(name):
    return lambda f, v, i: (f or {}).get(name)


def _json(row: Optional[dict], col: str) -> Optional[dict]:
    raw = (row or {}).get(col)
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None


def _tag(*path):
    def get(f, v, i):
        node = _json(f, "tags_json")
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
        return node
    return get


def _style_name(label: str) -> str:
    return label.split("---", 1)[-1]


def _styles(f, v, i):
    g = _tag("genre")(f, v, i)
    return [{"label": _style_name(x["label"]), "p": x["p"]} for x in g] if g else None


def _style(f, v, i):
    s = _styles(f, v, i)
    return s[0]["label"] if s else None


def _melody(key):
    return lambda f, v, i: (_json(v, "melody_json") or {}).get(key)


def _stem(row_index, col):
    return lambda f, v, i: ((v, i)[row_index] or {}).get(col)


def _a(id, label, short, category, source, kind, unit, decimals, description, get):
    return Attr(id, label, short, category, source, kind, unit, decimals, description, get)


E_R, E_T, E_L, E_S = "essentia.rhythm", "essentia.tonal", "essentia.loudness", "essentia.spectral"
EFF, MEL, PIPE = "essentia.effnet", "essentia.melody", "pipeline"
MOODS = ("happy", "sad", "aggressive", "relaxed", "party", "acoustic", "electronic")

ATTRIBUTES = (
    _a("bpm", "BPM", "BPM", "Tempo & grid", E_R, "number", "BPM", 1,
       "Tempo of the full mix.", _col("bpm")),
    _a("bpm_confidence", "Grid confidence", "GRID", "Tempo & grid", E_R, "number", "", 2,
       "How steady and salient the beat grid is, 0-1.", _col("bpm_confidence")),
    _a("onset_rate", "Onset rate", "ONSETS", "Tempo & grid", E_R, "number", "/s", 1,
       "Note/percussion onsets per second.", _col("onset_rate")),
    _a("dfa", "Danceability (DFA)", "DFA", "Tempo & grid", E_R, "number", "", 2,
       "Essentia's detrended-fluctuation danceability, roughly 0-3.", _col("danceability")),
    _a("key", "Key (Camelot)", "KEY", "Key & harmony", E_T, "category", "", 0,
       "Key of the full mix as a Camelot code.", _col("camelot")),
    _a("key_confidence", "Key confidence", "KEYCF", "Key & harmony", E_T, "number", "", 2,
       "Key strength × how many profiles agree, 0-1.", _col("key_confidence")),
    _a("key_strength", "Key strength", "KEYST", "Key & harmony", E_T, "number", "", 2,
       "How well the pitch profile fits the key, 0-1.", _col("key_strength")),
    _a("tuning_hz", "Tuning", "TUNE", "Key & harmony", E_T, "number", "Hz", 1,
       "Reference pitch of A4.", _col("tuning_hz")),
    _a("dissonance", "Dissonance", "DISS", "Key & harmony", E_S, "number", "", 2,
       "Sensory roughness of the spectrum, 0-1.", _col("dissonance")),
    _a("tonal", "Tonal", "TONAL", "Key & harmony", EFF, "number", "", 2,
       "Probability the track is tonal rather than atonal.", _tag("tonal")),
    _a("lufs", "Loudness (LUFS)", "LUFS", "Loudness", E_L, "number", "LUFS", 1,
       "EBU R128 integrated loudness.", _col("lufs")),
    _a("lra", "Loudness range", "LRA", "Loudness", E_L, "number", "LU", 1,
       "EBU R128 loudness range.", _col("lra")),
    _a("true_peak", "True peak", "TPEAK", "Loudness", E_L, "number", "dBTP", 1,
       "Highest inter-sample peak.", _col("true_peak")),
    _a("replay_gain", "ReplayGain", "RGAIN", "Loudness", E_L, "number", "dB", 1,
       "Gain to reach the ReplayGain reference.", _col("replay_gain")),
    _a("dynamic_complexity", "Dynamic complexity", "DYN", "Loudness", E_L, "number", "", 2,
       "How much the loudness moves around.", _col("dynamic_complexity")),
    _a("loudness_rms", "RMS loudness", "RMS", "Loudness", E_L, "number", "", 3,
       "Mean RMS level (the matcher's energy term).", _col("loudness_rms")),
    _a("energy", "Energy", "ENERGY", "Loudness", E_S, "number", "", 3,
       "Mean spectral energy.", _col("energy")),
    _a("spectral_centroid", "Spectral centroid", "CENTR", "Timbre", E_S, "number", "Hz", 0,
       "Where the spectrum's weight sits — higher is brighter.", _col("spectral_centroid")),
    _a("spectral_rolloff", "Spectral rolloff", "ROLL", "Timbre", E_S, "number", "Hz", 0,
       "Frequency below which most of the energy lies.", _col("spectral_rolloff")),
    _a("zero_crossing_rate", "Zero-crossing rate", "ZCR", "Timbre", E_S, "number", "", 3,
       "Noisiness / percussiveness.", _col("zero_crossing_rate")),
    _a("bright", "Bright", "BRIGHT", "Timbre", EFF, "number", "", 2,
       "Probability the timbre is bright rather than dark.", _tag("bright")),
    _a("style", "Style (Discogs)", "STYLE", "Genre & tags", EFF, "category", "", 0,
       "Most likely Discogs style (Essentia). Not SoundCloud's GENRE.", _style),
    _a("genre_parent", "Parent genre (Discogs)", "PARENT", "Genre & tags", EFF, "category", "", 0,
       "Discogs parent genre with the most summed style probability.", _tag("genre_parent")),
    _a("styles", "Top styles", "STYLES", "Genre & tags", EFF, "top", "", 2,
       "The five most likely Discogs styles.", _styles),
    _a("moodtheme", "Mood / theme", "THEME", "Genre & tags", EFF, "top", "", 2,
       "Top MTG-Jamendo moods and themes.", _tag("moodtheme")),
    _a("instruments", "Instruments", "INSTR", "Genre & tags", EFF, "top", "", 2,
       "Top MTG-Jamendo instruments.", _tag("instruments")),
    _a("danceable", "Danceable", "DANCE", "Mood", EFF, "number", "", 2,
       "Probability the track is danceable.", _tag("danceable")),
    *(_a(m, m.capitalize(), m[:6].upper(), "Mood", EFF, "number", "", 2,
         f"Probability the track sounds {m}.", _tag("mood", m)) for m in MOODS),
    _a("voice", "Voice", "VOICE", "Vocals & stems", EFF, "number", "", 2,
       "Probability the track has vocals.", _tag("voice")),
    _a("female", "Female vocal", "FEMALE", "Vocals & stems", EFF, "number", "", 2,
       "Probability the voice is female (only when the track sings).", _tag("female")),
    _a("sung_range", "Sung range", "RANGE", "Vocals & stems", MEL, "number", "st", 1,
       "10th-90th percentile of the sung pitch, in semitones.", _melody("range_st")),
    _a("sung_centre", "Sung centre", "CENTRE", "Vocals & stems", MEL, "number", "MIDI", 1,
       "Median sung pitch (MIDI note number).", _melody("median_midi")),
    _a("vocal_quality", "Vocal stem quality", "VQUAL", "Vocals & stems", PIPE, "number", "", 2,
       "Separation quality of the vocal stem, 0-1.", _stem(0, "stem_quality")),
    _a("bed_quality", "Bed stem quality", "BQUAL", "Vocals & stems", PIPE, "number", "", 2,
       "Separation quality of the instrumental, 0-1.", _stem(1, "stem_quality")),
    _a("residual_vocal", "Residual vocal", "RESID", "Vocals & stems", PIPE, "number", "", 2,
       "How much voice the instrumental still carries.", _stem(1, "residual_vocal_ratio")),
)
BY_ID = {a.id: a for a in ATTRIBUTES}


def extract(full: Optional[dict], vocals: Optional[dict], inst: Optional[dict]) -> dict:
    out = {}
    for a in ATTRIBUTES:
        try:
            val = a.get(full, vocals, inst)
        except (TypeError, ValueError, KeyError, AttributeError):
            val = None
        if val is not None and val != []:
            out[a.id] = val
    return out


def distribution(attr: Attr, values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    if attr.kind == "number":
        x = np.asarray(vals, dtype=float)
        lo, hi = float(x.min()), float(x.max())
        hist, edges = np.histogram(x, bins=HIST_BINS, range=(lo, hi if hi > lo else lo + 1))
        return {"min": round(lo, 4), "median": round(float(np.median(x)), 4),
                "max": round(hi, 4), "hist": [int(n) for n in hist],
                "edges": [round(float(e), 4) for e in edges]}
    counts: dict = {}
    for v in vals:
        label = v[0]["label"] if attr.kind == "top" and v else v
        counts[label] = counts.get(label, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:TOP_CATEGORIES]
    return {"top": [{"label": k, "count": n} for k, n in top]}


def describe() -> list:
    return [{"id": a.id, "label": a.label, "short": a.short, "category": a.category,
             "source": a.source, "kind": a.kind, "unit": a.unit, "decimals": a.decimals,
             "description": a.description} for a in ATTRIBUTES]


# Shown until you choose: nothing extra in the Library (its columns are
# already full), a useful handful on Track detail.
DEFAULT_VISIBILITY = {"library": [],
                      "detail": ["style", "genre_parent", "styles", "lufs", "voice",
                                 "party", "danceable", "sung_range"]}


def clean_visibility(v: Optional[dict]) -> dict:
    """A stored visibility with ids that no longer exist dropped."""
    v = v or {}
    return {k: [i for i in (v.get(k) or []) if i in BY_ID] for k in ("library", "detail")}
