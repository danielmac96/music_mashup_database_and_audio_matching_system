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
    clamp_rate, clamp_semitones, conform, is_valid_token, load_segment,
    require_audio_stack, resolve_stem_path,
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
                  db_path=None) -> Optional[Path]:
    """Write one mashup's FL session folder. Returns the folder, or None on a
    caller-fixable problem (reported through on_progress)."""
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
    plan = build_mashup_plan(vocal_song_id, inst_song_id, db_path=db_path)
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

    # Both start at bar 1 already; pad the shorter to the longer so the two
    # files line up in the playlist and the click covers both.
    _tick(80, "Aligning to the bar grid…")
    length = max(len(v_y), len(i_y))
    v_y = np.pad(v_y, (0, length - len(v_y)))
    i_y = np.pad(i_y, (0, length - len(i_y)))
    click = render_click(length / RENDER_SR, target_bpm)
    click = np.pad(click, (0, max(0, length - len(click))))[:length]

    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    _tick(88, "Writing files…")
    sf.write(str(out / "vocals.wav"), v_y.astype("float32"), RENDER_SR)
    sf.write(str(out / "instrumental.wav"), i_y.astype("float32"), RENDER_SR)
    sf.write(str(out / "click.wav"), click.astype("float32"), RENDER_SR)

    v_side, i_side = plan.get("vocal") or {}, plan.get("inst") or {}
    target_key = f"{v_side.get('key') or ''}{'m' if v_side.get('mode') == 'minor' else ''}"
    for fname in ("vocals.wav", "instrumental.wav", "click.wav"):
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
        "clips": [
            {"song_id": vocal_song_id, "stem": "vocals", "offset_sec": 0.0,
             "rate": rate_vocal, "semitones": 0, "gain": 0.8},
            {"song_id": inst_song_id, "stem": "instrumental", "offset_sec": 0.0,
             "rate": rate_inst, "semitones": semis_inst, "gain": 0.8},
        ],
    }, indent=2), encoding="utf-8")

    (out / "README.txt").write_text(_readme(plan, v_info, i_info, target_bpm),
                                    encoding="utf-8")

    _tick(100, f"Session ready: {out.name}")
    log.info("FL session exported: %s", out)
    return out


def _readme(plan: dict, v_info: dict, i_info: dict,
            target_bpm: float) -> str:
    """The recipe, from build_mashup_plan, plus what this export already did."""
    v, i = plan.get("vocal") or {}, plan.get("inst") or {}
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


def build_session_batch(token: str, pairs: list[dict],
                        on_progress: ProgressCb = None,
                        db_path=None) -> Optional[Path]:
    """Export several mashups into one parent folder, then zip it.

    `pairs`: [{vocal_song_id, inst_song_id}, …]. A pair that cannot be rendered
    is skipped with a note in the folder rather than failing the batch — one
    un-separated track should not cost the other nine exports."""
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
        plan = build_mashup_plan(v_id, i_id, db_path=db_path)
        made_path = build_session(
            sub_token, v_id, i_id,
            on_progress=lambda pct, msg, _lo=lo: _tick(_lo, msg),
            db_path=db_path)
        if made_path is None:
            skipped.append(f"{v_id} over {i_id}")
            continue
        v = (plan or {}).get("vocal") or {}
        i = (plan or {}).get("inst") or {}
        name = f"{idx:02d}_{_safe(v.get('title'))}_over_{_safe(i.get('title'))}"
        shutil.move(str(made_path), str(parent / name))
        made += 1

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
