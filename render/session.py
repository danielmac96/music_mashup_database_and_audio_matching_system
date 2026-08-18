"""render/session.py — export a mashup as an FL Studio session folder (B.1/B.2).

The Studio's only output was a summed WAV, which is a bounce: you cannot mix a
bounce. `prep_fl_session` in matcher/match.py did more, but it shipped untouched
source stems, so the actual FL workflow was still "read the recipe → import raw
stems → set tempo → stretch by hand → pitch by hand → find the chorus → nudge it
onto the grid". Every one of those numbers is already computed here.

What this writes instead, per mashup:

    01_{vocal}_over_{inst}/
      vocals.wav          conformed to the target tempo + key, grid-aligned
      instrumental.wav    same
      click.wav           bar/beat click at the target tempo, same length
      README.txt          the build_mashup_plan recipe
      session.json        the arrangement, in build_mixdown's clip shape

"Grid-aligned" is the part that removes the manual work: each stem is trimmed to
its chosen section, conformed, then padded so the section's first downbeat lands
at sample 0. Dropped into the playlist at 0:00 with the project at the stated
BPM, they sit on the grid with no nudging.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from config import PREVIEWS_DIR
from render.dsp import (
    MAX_RENDER_SECS, RENDER_SR, STEM_TYPES, AudioStackMissing, ProgressCb,
    clamp_gain, clamp_rate, clamp_semitones, conform, is_valid_token,
    load_segment, require_audio_stack, resolve_stem_path,
)

log = logging.getLogger(__name__)

BEATS_PER_BAR = 4

# Sessions per batch export. Each one is two phase-vocoder passes over a full
# section, so this bounds a "export top N" click to a few minutes of CPU rather
# than an afternoon.
MAX_SESSIONS = 16

CLICK_BARS_LEAD_IN = 0          # click starts with the audio, no count-in
CLICK_FREQ_DOWNBEAT = 1600.0    # Hz
CLICK_FREQ_BEAT = 800.0
CLICK_LEN_SECS = 0.02

# The bed's constituent stems, written alongside the summed instrumental when
# the track was separated in four-stem mode. This is the export that makes the
# advice the engine already gives actionable: `matcher/harmony.py` tells you to
# high-pass the bed when its bass root fights the vocal's tonic, and until now
# the only thing in the folder was a summed instrumental you would have to EQ
# rather than simply mute. Same for the standard move of keeping the bed's drums
# and dropping everything else.
BED_COMPONENT_STEMS = ("drums", "bass", "other")

# How far apart the two conformed stems' onsets may sit before the export is
# not actually on the grid. 10 ms is around where a doubled transient starts to
# read as flam rather than as one hit.
LOCK_TOLERANCE_MS = 10.0

# Widest offset the lock check will look for. Beyond a second the answer is not
# "a small residual error" but "these are aligned to different bars", which is a
# different problem and not one a nudge fixes.
LOCK_MAX_SHIFT_SECS = 1.0

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def session_dir(token: str) -> Optional[Path]:
    """Folder for a session export by job token. None when malformed."""
    if not is_valid_token(token):
        return None
    return PREVIEWS_DIR / f"fl_session_{token}"


def session_archive_path(token: str) -> Optional[Path]:
    d = session_dir(token)
    return None if d is None else d.with_suffix(".zip")


def _safe(name: str, limit: int = 40) -> str:
    """Filesystem-safe fragment of a title/artist for a folder name."""
    cleaned = _SAFE_NAME_RE.sub("_", (name or "").strip()).strip("_")
    return (cleaned or "untitled")[:limit]


def _bpm_key_tag(plan: Optional[dict]) -> str:
    """`128_8A` for a session folder name, or "" when either is unknown.

    The target tempo and the vocal's Camelot code are what the session IS — the
    two numbers you match on when picking the next thing to open.
    """
    if not plan:
        return ""
    bpm = plan.get("target_bpm") or 0
    cam = ((plan.get("vocal") or {}).get("camelot") or "").strip()
    parts = []
    if bpm:
        parts.append(f"{int(round(float(bpm)))}")
    if cam and cam != "?":
        parts.append(_SAFE_NAME_RE.sub("", cam))
    return "_".join(parts)


def _downbeats(feat: dict) -> list[float]:
    """Absolute bar-line times, honouring the detected beat phase.

    Mirrors analysis.hooks._downbeats — beat_times says where the beats are, not
    which of them starts a bar, and librosa's tracker is just as happy to latch
    onto beat 3."""
    beats = (feat or {}).get("beat_times") or []
    if not beats:
        return []
    try:
        phase = int((feat or {}).get("beat_phase") or 0) % BEATS_PER_BAR
    except (TypeError, ValueError):
        phase = 0
    return [t for i, t in enumerate(beats) if i % BEATS_PER_BAR == phase]


def first_downbeat_in(feat: dict, start_sec: float, end_sec: float) -> Optional[float]:
    """The first bar line at or after `start_sec` and before `end_sec`.

    None when the track has no beat grid or the window contains no bar line —
    the caller then treats `start_sec` itself as the alignment point rather than
    inventing an offset."""
    downs = _downbeats(feat)
    if not downs:
        return None
    for d in downs:
        if d >= start_sec - 1e-9:
            return d if d < end_sec else None
    return None


def measure_lock(vocal, bed, sr: int = RENDER_SR,
                 hop: int = 512) -> Optional[float]:
    """Residual timing offset between two conformed stems, in milliseconds.

    Positive means the bed sits EARLY and should move later by that much.
    None when there is not enough onset information on one side to tell.

    Everything upstream of here is an estimate: the beat tracker's grid, the
    detected phase, the phrase snap, the section boundary. Each is individually
    reasonable and they compose into an export that can still be half a bar out.
    Cross-correlating the two rendered onset envelopes is the one check that
    looks at what was actually written, and it costs a fraction of a second next
    to the two phase-vocoder passes that produced the files.

    The point is to find out in the README rather than in FL.
    """
    np, librosa, _sf = require_audio_stack()
    if vocal is None or bed is None or len(vocal) < sr or len(bed) < sr:
        return None
    ev = librosa.onset.onset_strength(y=vocal, sr=sr, hop_length=hop)
    eb = librosa.onset.onset_strength(y=bed, sr=sr, hop_length=hop)
    n = min(len(ev), len(eb))
    if n < 16:
        return None
    a = ev[:n] - ev[:n].mean()
    b = eb[:n] - eb[:n].mean()
    if float(np.linalg.norm(a)) < 1e-9 or float(np.linalg.norm(b)) < 1e-9:
        return None

    # correlate(a, b)[lag] peaks where b shifted LATER by `lag` matches a.
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(n - 1), n)
    max_lag = max(1, int(LOCK_MAX_SHIFT_SECS * sr / hop))
    keep = np.abs(lags) <= max_lag
    if not keep.any():
        return None
    best = int(lags[keep][int(np.argmax(corr[keep]))])
    return float(best * hop / sr * 1000.0)


def render_click(duration_secs: float, bpm: float, sr: int = RENDER_SR):
    """A bar/beat click at `bpm` for `duration_secs`.

    Downbeats are pitched higher than the other beats, so dropping this into the
    DAW next to the stems makes a misaligned bar audible immediately."""
    np, _librosa, _sf = require_audio_stack()
    n = max(1, int(round(max(0.0, duration_secs) * sr)))
    out = np.zeros(n, dtype="float32")
    if bpm <= 0:
        return out
    beat_secs = 60.0 / bpm
    click_n = max(1, int(round(CLICK_LEN_SECS * sr)))
    env = np.exp(-np.linspace(0.0, 6.0, click_n)).astype("float32")
    t = np.arange(click_n, dtype="float32") / sr

    beat = 0
    pos = 0.0
    while pos < duration_secs:
        freq = CLICK_FREQ_DOWNBEAT if beat % BEATS_PER_BAR == 0 else CLICK_FREQ_BEAT
        tone = (np.sin(2.0 * np.pi * freq * t) * env * 0.5).astype("float32")
        start = int(round(pos * sr))
        end = min(n, start + click_n)
        if end > start:
            out[start:end] += tone[: end - start]
        beat += 1
        pos += beat_secs
    return out


def conform_stem(song_id: int, stem: str, *, start_sec: Optional[float],
                 end_sec: Optional[float], rate: float, semitones: int,
                 feat: Optional[dict] = None, sr: int = RENDER_SR,
                 on_progress: ProgressCb = None, label: str = "",
                 db_path=None):
    """One stem, trimmed to its section, conformed, and grid-aligned.

    Returns (samples, info) or (None, reason). `info` carries the alignment that
    was applied so the caller can describe it in the README.

    The trim starts at the section's first downbeat rather than its boundary:
    phrase snapping puts boundaries on the 8-bar grid, but a section that could
    not be snapped still starts mid-phrase, and a stem that begins three beats
    into a bar is exactly the nudging this export exists to remove.
    """
    if stem not in STEM_TYPES:
        return None, f"unknown stem '{stem}'"
    path = resolve_stem_path(song_id, stem, db_path=db_path)
    if path is None:
        return None, (f"no {stem} audio for song {song_id} — "
                      "separate/download it first")

    align = None
    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        align = first_downbeat_in(feat or {}, start_sec, end_sec)
        read_start = align if align is not None else start_sec
        read_end = end_sec
    else:
        read_start, read_end = None, None

    if on_progress:
        on_progress(None, f"{label}loading {path.name}…")
    y = load_segment(path, sr, start_sec=read_start, end_sec=read_end,
                     max_secs=MAX_RENDER_SECS, rate=rate)
    y = conform(y, sr, rate, semitones, on_progress=on_progress, label=label)
    return y, {
        "source": str(path),
        "section_start": read_start,
        "section_end": read_end,
        "snapped_to_downbeat": align is not None,
        "rate": rate,
        "semitones": semitones,
        "duration_secs": round(len(y) / sr, 3),
    }


def _write_tags(path: Path, bpm: Optional[float], key: Optional[str]) -> None:
    """ID3 TBPM/TKEY so the folder is also usable from Serato/rekordbox.

    Never fatal: mutagen is optional, and a missing tag is cosmetic next to the
    audio being right."""
    if not bpm and not key:
        return
    try:
        from mutagen.id3 import ID3, TBPM, TKEY, ID3NoHeaderError
    except ImportError:
        log.info("mutagen not installed — skipping ID3 tags on %s", path.name)
        return
    try:
        try:
            tags = ID3(str(path))
        except ID3NoHeaderError:
            tags = ID3()
        if bpm:
            tags.add(TBPM(encoding=3, text=[str(int(round(bpm)))]))
        if key:
            tags.add(TKEY(encoding=3, text=[str(key)]))
        tags.save(str(path))
    except Exception:  # noqa: BLE001
        log.warning("could not write ID3 tags to %s", path.name, exc_info=True)


def build_session(token: str, vocal_song_id: int, inst_song_id: int, *,
                  on_progress: ProgressCb = None,
                  vocal_section_idx: Optional[int] = None,
                  inst_section_idx: Optional[int] = None,
                  harmonic_shift: Optional[int] = None,
                  db_path=None) -> Optional[Path]:
    """Write one mashup's FL session folder. Returns the folder, or None on a
    caller-fixable problem (reported through on_progress).

    `vocal_section_idx` / `inst_section_idx` / `harmonic_shift` come off the
    candidate row the user actually chose. Without them this re-derived both the
    section pairing and the transpose from scratch, so the folder you opened in
    FL was frequently not the mashup you auditioned — a different chorus over a
    different drop, pitched by a Camelot estimate instead of the measured shift.
    """
    def _tick(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    out = session_dir(token)
    if out is None:
        _tick(None, "Invalid session token")
        return None

    try:
        np, _librosa, sf = require_audio_stack()
    except AudioStackMissing as exc:
        log.error("session export needs librosa + soundfile: %s", exc)
        _tick(None, str(exc))
        return None

    from matcher.plan import build_mashup_plan

    _tick(5, "Building the plan…")
    plan = build_mashup_plan(vocal_song_id, inst_song_id, db_path=db_path,
                             vocal_section_idx=vocal_section_idx,
                             inst_section_idx=inst_section_idx,
                             harmonic_shift=harmonic_shift)
    if not plan:
        _tick(None, f"No such pair: {vocal_song_id} over {inst_song_id}")
        return None

    target_bpm = plan.get("target_bpm") or 0.0
    stretch = plan.get("stretch_factor") or 1.0
    shift = plan.get("semitone_shift") or 0

    # The bed is played at `stretch` to reach the vocal's tempo; the vocal plays
    # native. This is the same convention Studio's SYNC and the audition use.
    rate_vocal = clamp_rate(1.0)
    rate_inst = clamp_rate(stretch)
    semis_inst = clamp_semitones(shift)

    pairing = (plan.get("pairings") or [None])[0]
    v_start = pairing.get("vocal_start") if pairing else None
    v_end = pairing.get("vocal_end") if pairing else None
    i_start = pairing.get("inst_start") if pairing else None
    i_end = pairing.get("inst_end") if pairing else None

    from database.models import get_features_for_song
    v_feat = (get_features_for_song(vocal_song_id, "vocals", db_path=db_path)
              or get_features_for_song(vocal_song_id, "full", db_path=db_path) or {})
    i_feat = (get_features_for_song(inst_song_id, "instrumental", db_path=db_path)
              or get_features_for_song(inst_song_id, "full", db_path=db_path) or {})

    _tick(20, "Conforming the vocal…")
    v_y, v_info = conform_stem(vocal_song_id, "vocals",
                               start_sec=v_start, end_sec=v_end,
                               rate=rate_vocal, semitones=0, feat=v_feat,
                               on_progress=on_progress, label="Vocal: ",
                               db_path=db_path)
    if v_y is None:
        _tick(None, f"Vocal: {v_info}")
        return None

    _tick(50, "Conforming the instrumental…")
    i_y, i_info = conform_stem(inst_song_id, "instrumental",
                               start_sec=i_start, end_sec=i_end,
                               rate=rate_inst, semitones=semis_inst, feat=i_feat,
                               on_progress=on_progress, label="Bed: ",
                               db_path=db_path)
    if i_y is None:
        _tick(None, f"Instrumental: {i_info}")
        return None

    # The bed's own stems, conformed identically, when four-stem separation ran.
    # resolve_stem_path returns None on a two-stem library, so this is a no-op
    # there rather than a failure.
    _tick(65, "Conforming the bed's stems…")
    components: dict = {}
    for name in BED_COMPONENT_STEMS:
        c_y, c_info = conform_stem(inst_song_id, name,
                                   start_sec=i_start, end_sec=i_end,
                                   rate=rate_inst, semitones=semis_inst,
                                   feat=i_feat, on_progress=on_progress,
                                   label=f"{name.title()}: ", db_path=db_path)
        if c_y is not None:
            components[name] = (c_y, c_info)

    # Both start at bar 1 already; pad the shorter to the longer so the two
    # files line up in the playlist and the click covers both.
    _tick(80, "Aligning to the bar grid…")
    length = max([len(v_y), len(i_y)] + [len(c) for c, _ in components.values()])
    v_y = np.pad(v_y, (0, length - len(v_y)))
    i_y = np.pad(i_y, (0, length - len(i_y)))
    components = {k: (np.pad(c, (0, length - len(c))), info)
                  for k, (c, info) in components.items()}
    click = render_click(length / RENDER_SR, target_bpm)
    click = np.pad(click, (0, max(0, length - len(click))))[:length]

    # Verify what was actually rendered, not what the estimates promised.
    _tick(84, "Checking the grid lock…")
    lock_ms = measure_lock(v_y, i_y)

    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    _tick(88, "Writing files…")
    sf.write(str(out / "vocals.wav"), v_y.astype("float32"), RENDER_SR)
    sf.write(str(out / "instrumental.wav"), i_y.astype("float32"), RENDER_SR)
    sf.write(str(out / "click.wav"), click.astype("float32"), RENDER_SR)
    for name, (c_y, _info) in components.items():
        sf.write(str(out / f"bed_{name}.wav"), c_y.astype("float32"), RENDER_SR)

    v_side, i_side = plan.get("vocal") or {}, plan.get("inst") or {}
    target_key = f"{v_side.get('key') or ''}{'m' if v_side.get('mode') == 'minor' else ''}"
    for fname in (["vocals.wav", "instrumental.wav", "click.wav"]
                  + [f"bed_{n}.wav" for n in components]):
        _write_tags(out / fname, target_bpm, target_key or None)

    # session.json uses build_mixdown's exact clip shape, so an exported session
    # round-trips back into Studio.
    (out / "session.json").write_text(json.dumps({
        "target_bpm": target_bpm,
        "target_key": target_key or None,
        "semitone_shift": semis_inst,
        "stretch_factor": rate_inst,
        "vocal": {**v_side, "conformed": v_info},
        "inst": {**i_side, "conformed": i_info},
        "bed_stems": {n: info for n, (_y, info) in components.items()},
        "lock_offset_ms": None if lock_ms is None else round(lock_ms, 1),
        "clips": [
            {"song_id": vocal_song_id, "stem": "vocals", "offset_sec": 0.0,
             "rate": rate_vocal, "semitones": 0, "gain": 0.8},
            {"song_id": inst_song_id, "stem": "instrumental", "offset_sec": 0.0,
             "rate": rate_inst, "semitones": semis_inst, "gain": 0.8},
        ],
    }, indent=2), encoding="utf-8")

    (out / "README.txt").write_text(
        _readme(plan, v_info, i_info, target_bpm, sorted(components), lock_ms),
        encoding="utf-8")

    _tick(100, f"Session ready: {out.name}")
    log.info("FL session exported: %s", out)
    return out


def _readme(plan: dict, v_info: dict, i_info: dict,
            target_bpm: float, component_stems: Optional[list] = None,
            lock_ms: Optional[float] = None) -> str:
    """The recipe, from build_mashup_plan, plus what this export already did."""
    v, i = plan.get("vocal") or {}, plan.get("inst") or {}
    component_stems = component_stems or []
    lines = [
        f"{v.get('title')} ({v.get('artist')})  —  vocal",
        f"over",
        f"{i.get('title')} ({i.get('artist')})  —  instrumental",
        "",
        "=" * 68,
        "ALREADY DONE FOR YOU",
        "=" * 68,
        f"Both WAVs are rendered at {target_bpm:.1f} BPM and in the vocal's key.",
        "They are trimmed to the sections below and padded so BAR 1 IS AT 0:00.",
        "",
        f"  1. Set the FL project tempo to {target_bpm:.1f} BPM.",
        "  2. Drag vocals.wav and instrumental.wav into the playlist at 0:00.",
        "  3. That's it — they are on the grid. click.wav is there to check.",
        "",
        "Do NOT re-stretch or re-pitch these files; that work is baked in.",
        "",
        _lock_note(lock_ms),
        "",
    ]
    if component_stems:
        lines += [
            "=" * 68,
            "THE BED, IN PARTS",
            "=" * 68,
            "instrumental.wav is the sum of these; they are conformed and",
            "grid-aligned identically, so use EITHER the sum OR the parts.",
            "",
        ] + [f"  bed_{n}.wav" for n in component_stems] + [
            "",
            "  Bass clashing with the vocal? Mute bed_bass.wav instead of",
            "  EQ-ing the sum. Want the groove only? Keep bed_drums.wav.",
            "",
        ]
    lines += [
        "=" * 68,
        "WHAT WAS APPLIED",
        "=" * 68,
        f"  vocal        : native tempo, no transpose",
        f"                 section {_fmt(v_info.get('section_start'))}"
        f"–{_fmt(v_info.get('section_end'))}"
        f"{'  (snapped to a downbeat)' if v_info.get('snapped_to_downbeat') else ''}",
        f"  instrumental : ×{i_info.get('rate'):.4f} time-stretch, "
        f"{i_info.get('semitones'):+d} semitones",
        f"                 section {_fmt(i_info.get('section_start'))}"
        f"–{_fmt(i_info.get('section_end'))}"
        f"{'  (snapped to a downbeat)' if i_info.get('snapped_to_downbeat') else ''}",
        f"  key relation : {plan.get('key_relation')}",
        "",
        "=" * 68,
        "THE FULL RECIPE",
        "=" * 68,
    ]
    lines += [f"  {s}" for s in (plan.get("steps") or [])]
    lines += [
        "",
        "=" * 68,
        "SOURCE FILES",
        "=" * 68,
        f"  vocal stem        : {v_info.get('source')}",
        f"  instrumental stem : {i_info.get('source')}",
        "",
    ]
    return "\n".join(lines)


def _fmt(secs) -> str:
    if secs is None:
        return "whole track"
    s = int(round(float(secs)))
    return f"{s // 60}:{s % 60:02d}"


def _lock_note(lock_ms: Optional[float]) -> str:
    """What the grid-lock check found, in words the reader can act on."""
    if lock_ms is None:
        return ("GRID CHECK: not enough onset detail to verify — trust the "
                "click and your ears.")
    if abs(lock_ms) <= LOCK_TOLERANCE_MS:
        return f"GRID CHECK: locked (residual {lock_ms:+.0f} ms). Nothing to nudge."
    direction = "later" if lock_ms > 0 else "earlier"
    return (
        f"GRID CHECK: ⚠ the bed reads {abs(lock_ms):.0f} ms {'early' if lock_ms > 0 else 'late'} "
        f"against the vocal.\n"
        f"  Nudge instrumental.wav {abs(lock_ms):.0f} ms {direction} — or check whether the two\n"
        f"  sections were snapped to different bars, which a nudge will not fix."
    )


# ── A.4: export the arrangement you actually built ───────────────────────────
#
# Studio lets you place N lanes, set each one's rate, pitch, gain and offset,
# and nudge until it sits right. Its export then threw all of that away and
# re-planned the pair server-side, so there was no way to export what you had
# just built — the only paths out were a bounced mixdown (which you cannot mix)
# and a session rebuilt from the engine's own opinion.
#
# Same clip shape as /studio/mixdown, so what you hear in the browser, what the
# mixdown renders and what lands in FL are three views of one arrangement.

def build_session_from_clips(token: str, clips: list[dict], *,
                             target_bpm: Optional[float] = None,
                             on_progress: ProgressCb = None,
                             db_path=None) -> Optional[Path]:
    """Write an FL session folder from an explicit Studio arrangement.

    Each clip is conformed exactly as the mixdown would conform it, then padded
    at the head by its own offset, so every WAV starts at the arrangement's zero
    and they all drop into the playlist at 0:00 together — the same guarantee
    the pair export gives, extended to N lanes.
    """
    def _tick(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    out = session_dir(token)
    if out is None:
        _tick(None, "Invalid session token")
        return None
    if not clips:
        _tick(None, "No clips to export")
        return None

    try:
        np, _librosa, sf = require_audio_stack()
    except AudioStackMissing as exc:
        log.error("session export needs librosa + soundfile: %s", exc)
        _tick(None, str(exc))
        return None

    from database.models import get_features_for_song, get_song

    # Relative to the earliest clip, so a lane dragged left of zero still
    # renders instead of being silently clipped away (mirrors build_mixdown).
    base = min(0.0, *(float(c.get("offset_sec") or 0.0) for c in clips))

    rendered: list[tuple[dict, "np.ndarray", dict]] = []
    n = len(clips)
    for idx, c in enumerate(clips):
        song_id = int(c["song_id"])
        stem = str(c.get("stem") or "full")
        if stem not in STEM_TYPES:
            _tick(None, f"Clip {idx + 1}: unknown stem '{stem}'")
            return None
        path = resolve_stem_path(song_id, stem, db_path=db_path)
        if path is None:
            _tick(None, f"Clip {idx + 1}: no {stem} audio for song {song_id} — "
                        "separate/download it first")
            return None

        rate = clamp_rate(c.get("rate"))
        semitones = clamp_semitones(c.get("semitones"))
        gain = clamp_gain(c.get("gain"))
        offset = float(c.get("offset_sec") or 0.0)

        label = f"Clip {idx + 1}/{n}: "
        _tick(int(5 + 75 * idx / n), f"{label}conforming {path.name}…")
        y = load_segment(path, RENDER_SR, max_secs=MAX_RENDER_SECS, rate=rate)
        y = conform(y, RENDER_SR, rate, semitones,
                    on_progress=on_progress, label=label)
        # Head padding IS the offset: the point of a session folder is that
        # every file starts at the same zero, so the placement has to be baked
        # into the audio rather than left as an instruction.
        pad = max(0, int(round((offset - base) * RENDER_SR)))
        y = np.pad(y, (pad, 0))

        song = get_song(song_id, db_path=db_path) or {}
        rendered.append((c, y, {
            "song_id": song_id, "stem": stem, "title": song.get("title"),
            "artist": song.get("artist"), "source": str(path),
            "offset_sec": round(offset - base, 3), "rate": rate,
            "semitones": semitones, "gain": gain,
            "duration_secs": round(len(y) / RENDER_SR, 3),
        }))

    _tick(84, "Aligning to the bar grid…")
    length = max(len(y) for _c, y, _i in rendered)
    rendered = [(c, np.pad(y, (0, length - len(y))), info)
                for c, y, info in rendered]

    # The project tempo. Prefer what Studio says, since that is what the user
    # conformed everything to; fall back to the first lane's own tempo.
    if not target_bpm:
        first = rendered[0][2]
        feat = (get_features_for_song(first["song_id"], first["stem"],
                                      db_path=db_path)
                or get_features_for_song(first["song_id"], "full",
                                         db_path=db_path) or {})
        target_bpm = (feat.get("bpm") or 0.0) * first["rate"]

    click = render_click(length / RENDER_SR, target_bpm or 0.0)
    click = np.pad(click, (0, max(0, length - len(click))))[:length]

    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    _tick(90, "Writing files…")
    used: set = set()
    for idx, (_c, y, info) in enumerate(rendered, start=1):
        name = f"{idx:02d}_{_safe(info['title'], 28)}_{info['stem']}.wav"
        while name in used:                     # two lanes on the same stem
            name = f"{idx:02d}_{_safe(info['title'], 24)}_{info['stem']}_b.wav"
        used.add(name)
        info["file"] = name
        sf.write(str(out / name), y.astype("float32"), RENDER_SR)
        _write_tags(out / name, target_bpm, None)
    sf.write(str(out / "click.wav"), click.astype("float32"), RENDER_SR)
    _write_tags(out / "click.wav", target_bpm, None)

    (out / "session.json").write_text(json.dumps({
        "source": "studio",
        "target_bpm": target_bpm,
        "lanes": [info for _c, _y, info in rendered],
        # build_mixdown's clip shape, offsets rebased to zero because the
        # placement is now baked into each file's head padding.
        "clips": [{"song_id": info["song_id"], "stem": info["stem"],
                   "offset_sec": 0.0, "rate": info["rate"],
                   "semitones": info["semitones"], "gain": info["gain"]}
                  for _c, _y, info in rendered],
    }, indent=2), encoding="utf-8")

    (out / "README.txt").write_text(
        _clips_readme(rendered, target_bpm), encoding="utf-8")

    _tick(100, f"Session ready: {out.name}")
    log.info("FL session exported from Studio: %s (%d lanes)", out, len(rendered))
    return out


def _clips_readme(rendered: list, target_bpm: Optional[float]) -> str:
    lines = [
        "Exported from Studio — this is the arrangement you built,",
        "not a re-plan of the pair.",
        "",
        "=" * 68,
        "ALREADY DONE FOR YOU",
        "=" * 68,
        f"Every WAV is rendered at {target_bpm:.1f} BPM."
        if target_bpm else "Project tempo unknown — check against click.wav.",
        "Each lane's placement is baked into its head padding, so they all",
        "START AT 0:00 TOGETHER and stay in the arrangement you heard.",
        "",
        f"  1. Set the FL project tempo to {target_bpm:.1f} BPM."
        if target_bpm else "  1. Set the FL project tempo by ear.",
        "  2. Drag every WAV into the playlist at 0:00.",
        "  3. That's it. click.wav is there to check.",
        "",
        "Do NOT re-stretch or re-pitch these files; that work is baked in.",
        "Gain is NOT baked in — it is listed below so you can dial it in.",
        "",
        "=" * 68,
        "LANES",
        "=" * 68,
    ]
    for info in (i for _c, _y, i in rendered):
        lines += [
            f"  {info['file']}",
            f"      {info.get('title') or '?'} — {info.get('artist') or '?'}"
            f"  [{info['stem']}]",
            f"      placed at {_fmt(info['offset_sec'])}, "
            f"x{info['rate']:.4f} time-stretch, {info['semitones']:+d} semitones, "
            f"gain {info['gain']:.2f}",
            f"      source: {info['source']}",
            "",
        ]
    return "\n".join(lines)


def build_session_batch(token: str, pairs: list[dict],
                        on_progress: ProgressCb = None,
                        db_path=None,
                        on_exported=None) -> Optional[Path]:
    """Export several mashups into one parent folder, then zip it.

    `pairs`: [{vocal_song_id, inst_song_id, vocal_section_idx?,
    inst_section_idx?, harmonic_shift?}, …]. The optional keys pin each export
    to the exact section pair and transpose of the candidate row it came from
    (A.1); omitted, the plan re-chooses them. A pair that cannot be rendered is
    skipped with a note in the folder rather than failing the batch — one
    un-separated track should not cost the other nine exports.

    `on_exported(vocal_song_id, inst_song_id)` fires per pair that actually
    rendered, so the caller can record it. Only the successes: a pair that was
    skipped for a missing stem is not evidence of anything.
    """
    def _tick(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    parent = session_dir(token)
    if parent is None:
        _tick(None, "Invalid session token")
        return None
    if not pairs:
        _tick(None, "No pairs to export")
        return None
    pairs = pairs[:MAX_SESSIONS]

    if parent.exists():
        shutil.rmtree(parent, ignore_errors=True)
    parent.mkdir(parents=True, exist_ok=True)

    from matcher.plan import build_mashup_plan

    skipped: list[str] = []
    made = 0
    for idx, p in enumerate(pairs, start=1):
        v_id, i_id = int(p["vocal_song_id"]), int(p["inst_song_id"])
        lo = int(5 + 90 * (idx - 1) / len(pairs))
        _tick(lo, f"Pair {idx}/{len(pairs)}…")

        # Render into a per-pair token folder, then move it under the parent
        # with a readable name.
        sub_token = f"{token}{idx:02x}"
        pin = {
            "vocal_section_idx": p.get("vocal_section_idx"),
            "inst_section_idx": p.get("inst_section_idx"),
            "harmonic_shift": p.get("harmonic_shift"),
        }
        # Same pin for the naming plan and the render, so the folder's BPM/key
        # tag describes the take inside it.
        plan = build_mashup_plan(v_id, i_id, db_path=db_path, **pin)
        made_path = build_session(
            sub_token, v_id, i_id,
            on_progress=lambda pct, msg, _lo=lo: _tick(_lo, msg),
            db_path=db_path, **pin)
        if made_path is None:
            skipped.append(f"{v_id} over {i_id}")
            continue
        v = (plan or {}).get("vocal") or {}
        i = (plan or {}).get("inst") or {}
        # Tempo and key in the folder name: the file browser is where you
        # actually choose what to open next, and "128_8A" is what you are
        # choosing on. Sorting the folder then groups everything you could mix
        # together, which the rank prefix alone never did.
        name = "_".join(x for x in (
            f"{idx:02d}",
            _bpm_key_tag(plan),
            _safe(v.get("title"), 32),
            "over",
            _safe(i.get("title"), 32),
        ) if x)
        shutil.move(str(made_path), str(parent / name))
        made += 1
        if on_exported is not None:
            try:
                on_exported(v_id, i_id)
            except Exception:  # noqa: BLE001 — never fail an export over a label
                log.warning("on_exported callback raised", exc_info=True)

    if skipped:
        (parent / "SKIPPED.txt").write_text(
            "These pairs could not be exported (usually a missing stem —\n"
            "download and separate the track, then export again):\n\n"
            + "\n".join(f"  {s}" for s in skipped) + "\n", encoding="utf-8")

    if not made:
        _tick(None, "Nothing could be exported — check the tracks have stems")
        return None

    _tick(96, "Zipping…")
    archive = session_archive_path(token)
    if archive and archive.exists():
        archive.unlink()
    shutil.make_archive(str(parent), "zip", root_dir=str(parent))

    _tick(100, f"{made} session{'s' if made != 1 else ''} ready")
    log.info("FL session batch exported: %s (%d pairs, %d skipped)",
             parent, made, len(skipped))
    return parent
