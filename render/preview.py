"""
render/preview.py — Render an audible preview of a vocal-over-instrumental
mashup so a producer can judge a pair by ear before opening a DAW.

What it does, following the plan from matcher.plan.build_mashup_plan:
  * time-stretches the instrumental to the vocal's tempo (halftime/doubletime
    aware via the plan's stretch_factor)
  * pitch-shifts the instrumental to the vocal's key (semitone_shift)
  * lays the vocal on top, aligned on the best section pairing (e.g. vocal
    chorus over instrumental drop)
  * mixes the two and writes a WAV

Stretch and pitch are decoupled here (librosa's phase vocoder), which is the
whole point — native browser playback can only change both together.

librosa + soundfile are imported lazily so the rest of the app (and the mock
test pipeline) keeps working without the audio stack installed.
"""
from pathlib import Path
from typing import Callable, Optional
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PREVIEWS_DIR
from matcher.plan import build_mashup_plan

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[Optional[int], str], None]]

PREVIEW_SR = 44100
MAX_PREVIEW_SECS = 30.0   # cap so a render stays fast and replayable


def preview_path(vocal_song_id: int, inst_song_id: int) -> Path:
    return PREVIEWS_DIR / f"preview_{vocal_song_id}_over_{inst_song_id}.wav"


def adjusted_path(stem_song_id: int, ref_song_id: int) -> Path:
    return PREVIEWS_DIR / f"adjusted_{stem_song_id}_to_{ref_song_id}.wav"


def build_adjusted_stem(vocal_song_id: int, inst_song_id: int, anchor: str,
                        db_path=None, on_progress: ProgressCb = None,
                        force: bool = False, stretch_override: Optional[float] = None,
                        shift_override: Optional[int] = None) -> Optional[Path]:
    """Render (and cache) a full-length, tempo/key-matched stem so the Audition
    Studio can scrub and replay freely without re-running the DSP each time.

    anchor='instrumental': stretch+pitch-shift the FULL instrumental stem to
      the vocal's tempo/key (the plan's stretch_factor / semitone_shift).
    anchor='vocal': stretch+pitch-shift the FULL vocal stem to the
      instrumental's tempo/key (the inverse: 1/stretch_factor, -semitone_shift).

    stretch_override / shift_override: when supplied, used in place of the
    plan-suggested values (e.g. the user edited the suggested numbers in the
    Audition Studio before applying)."""
    def _tick(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    plan = build_mashup_plan(vocal_song_id, inst_song_id, db_path=db_path)
    if not plan:
        log.warning("adjust: no plan (song missing) for %s/%s",
                    vocal_song_id, inst_song_id)
        return None

    stretch = plan.get("stretch_factor") or 1.0
    shift = int(plan.get("semitone_shift") or 0)

    if anchor == "instrumental":
        src_path = plan["files"].get("instrumental")
        out = adjusted_path(inst_song_id, vocal_song_id)
        suggested_stretch, suggested_shift = stretch, shift
    elif anchor == "vocal":
        src_path = plan["files"].get("vocals")
        out = adjusted_path(vocal_song_id, inst_song_id)
        suggested_stretch, suggested_shift = (1.0 / stretch if stretch else 1.0), -shift
    else:
        raise ValueError("anchor must be 'vocal' or 'instrumental'")

    eff_stretch = float(stretch_override) if stretch_override is not None else suggested_stretch
    eff_shift = int(shift_override) if shift_override is not None else suggested_shift

    if out.exists() and not force:
        _tick(100, "Already adjusted")
        return out

    if not src_path or not Path(src_path).exists():
        log.warning("adjust: missing stem file (anchor=%s, path=%s)", anchor, src_path)
        return None

    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        log.error("adjust render needs librosa + soundfile: %s", exc)
        return None

    _tick(10, "Loading stem…")
    y, _ = librosa.load(src_path, sr=PREVIEW_SR, mono=True)

    if abs(eff_stretch - 1.0) > 1e-3:
        _tick(40, f"Time-stretching ×{eff_stretch:.3f}…")
        y = librosa.effects.time_stretch(y, rate=float(eff_stretch), n_fft=1024)

    if eff_shift:
        _tick(70, f"Pitch-shifting {eff_shift:+d} st…")
        y = librosa.effects.pitch_shift(y, sr=PREVIEW_SR, n_steps=eff_shift, n_fft=1024)

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y.astype("float32"), PREVIEW_SR)
    _tick(100, "Adjustment ready")
    log.info("adjusted stem rendered: %s", out.name)
    return out


def export_path(vocal_song_id: int, inst_song_id: int) -> Path:
    return PREVIEWS_DIR / f"export_{vocal_song_id}_x_{inst_song_id}.wav"


MAX_EXPORT_SECS = 600.0  # safety cap so a runaway render can't fill the disk


def build_mashup_export(vocal_song_id: int, inst_song_id: int, anchor: str,
                        stretch: float, shift: int,
                        vocal_offset: float, inst_offset: float,
                        db_path=None, on_progress: ProgressCb = None,
                        force: bool = True) -> Optional[Path]:
    """Render the full mashup exactly as aligned in the Audition Studio:
    the anchor stem is time-stretched (×stretch) and pitch-shifted (shift st)
    with the two effects decoupled, then both stems are laid on one timeline at
    their drag offsets (in display-seconds) and mixed.

    This is the non-destructive boundary: the source stems are never modified;
    only this mixed WAV is written, and only when the user clicks Export."""
    def _tick(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    if anchor not in ("vocal", "instrumental"):
        raise ValueError("anchor must be 'vocal' or 'instrumental'")

    plan = build_mashup_plan(vocal_song_id, inst_song_id, db_path=db_path)
    if not plan:
        log.warning("export: no plan (song missing) for %s/%s",
                    vocal_song_id, inst_song_id)
        return None

    v_path = plan["files"].get("vocals")
    i_path = plan["files"].get("instrumental")
    if not v_path or not i_path or not Path(v_path).exists() or not Path(i_path).exists():
        log.warning("export: missing stem files (vocals=%s, inst=%s)", v_path, i_path)
        return None

    out = export_path(vocal_song_id, inst_song_id)
    if out.exists() and not force:
        _tick(100, "Export already rendered")
        return out

    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except ImportError as exc:
        log.error("export render needs librosa + soundfile: %s", exc)
        return None

    _tick(10, "Loading vocal…")
    v_y, _ = librosa.load(v_path, sr=PREVIEW_SR, mono=True)
    _tick(25, "Loading instrumental…")
    i_y, _ = librosa.load(i_path, sr=PREVIEW_SR, mono=True)

    rate = float(stretch) if stretch and stretch > 0 else 1.0
    n_steps = int(shift or 0)

    # Apply the decoupled stretch + pitch to whichever side is anchored. After
    # this, both arrays live on the shared "display" timeline (samples / SR).
    if anchor == "vocal":
        if abs(rate - 1.0) > 1e-3:
            _tick(45, f"Time-stretching vocal ×{rate:.3f}…")
            v_y = librosa.effects.time_stretch(v_y, rate=rate, n_fft=1024)
        if n_steps:
            _tick(60, f"Pitch-shifting vocal {n_steps:+d} st…")
            v_y = librosa.effects.pitch_shift(v_y, sr=PREVIEW_SR, n_steps=n_steps, n_fft=1024)
    else:
        if abs(rate - 1.0) > 1e-3:
            _tick(45, f"Time-stretching instrumental ×{rate:.3f}…")
            i_y = librosa.effects.time_stretch(i_y, rate=rate, n_fft=1024)
        if n_steps:
            _tick(60, f"Pitch-shifting instrumental {n_steps:+d} st…")
            i_y = librosa.effects.pitch_shift(i_y, sr=PREVIEW_SR, n_steps=n_steps, n_fft=1024)

    _tick(80, "Aligning + mixing…")
    # Place each stem on the global timeline at its drag offset (display secs).
    base = min(vocal_offset, inst_offset, 0.0)
    v_at = int(round((vocal_offset - base) * PREVIEW_SR))
    i_at = int(round((inst_offset - base) * PREVIEW_SR))
    total = max(v_at + len(v_y), i_at + len(i_y))
    total = min(total, int(MAX_EXPORT_SECS * PREVIEW_SR))
    if total <= 0:
        log.warning("export: empty timeline")
        return None

    mix = np.zeros(total, dtype="float32")
    v_end = min(total, v_at + len(v_y))
    if v_at < total and v_end > v_at:
        mix[v_at:v_end] += v_y[: v_end - v_at] * 0.95
    i_end = min(total, i_at + len(i_y))
    if i_at < total and i_end > i_at:
        mix[i_at:i_end] += i_y[: i_end - i_at] * 0.8

    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 1.0:
        mix = mix / peak

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix.astype("float32"), PREVIEW_SR)
    _tick(100, "Export ready")
    log.info("mashup export rendered: %s", out.name)
    return out


def build_preview(vocal_song_id: int, inst_song_id: int, db_path=None,
                  on_progress: ProgressCb = None, force: bool = False,
                  vocal_start: Optional[float] = None,
                  inst_start: Optional[float] = None) -> Optional[Path]:
    """Render (and cache) a mashup preview WAV. Returns the path, or None when
    stems/features are missing or the audio stack is unavailable.

    vocal_start / inst_start: when supplied (from the Audition Studio marker),
    override the auto-detected pairing start times."""
    def _tick(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    out = preview_path(vocal_song_id, inst_song_id)
    if out.exists() and not force:
        _tick(100, "Preview already rendered")
        return out

    plan = build_mashup_plan(vocal_song_id, inst_song_id, db_path=db_path)
    if not plan:
        log.warning("preview: no plan (song missing) for %s/%s",
                    vocal_song_id, inst_song_id)
        return None

    v_path = plan["files"].get("vocals")
    i_path = plan["files"].get("instrumental")
    if not v_path or not i_path or not Path(v_path).exists() or not Path(i_path).exists():
        log.warning("preview: missing stem files (vocals=%s, inst=%s)", v_path, i_path)
        return None

    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except ImportError as exc:
        log.error("preview render needs librosa + soundfile: %s", exc)
        return None

    stretch = plan.get("stretch_factor") or 1.0
    shift = int(plan.get("semitone_shift") or 0)
    pairings = plan.get("pairings") or []

    # Alignment: use caller-supplied marker times when available; otherwise
    # fall back to the auto-detected best pairing from the plan.
    if pairings:
        p = pairings[0]
        auto_v_start = float(p["vocal_start"])
        auto_v_end   = float(p["vocal_end"])
        auto_i_start = float(p["inst_start"])
    else:
        auto_v_start, auto_v_end, auto_i_start = 0.0, MAX_PREVIEW_SECS, 0.0

    v_start = float(vocal_start) if vocal_start is not None else auto_v_start
    i_start = float(inst_start)  if inst_start  is not None else auto_i_start
    # Custom marker: render a full MAX_PREVIEW_SECS clip; auto: clip to section boundary
    v_end   = auto_v_end if vocal_start is None else v_start + MAX_PREVIEW_SECS

    v_dur = min(max(v_end - v_start, 1.0), MAX_PREVIEW_SECS)

    _tick(10, "Loading vocal…")
    v_y, _ = librosa.load(v_path, sr=PREVIEW_SR, mono=True,
                          offset=max(0.0, v_start), duration=v_dur)

    # Load enough instrumental that, after stretching by `stretch`, it still
    # covers the vocal segment (stretched duration = raw / stretch).
    i_load_dur = v_dur * stretch + 2.0
    _tick(30, "Loading instrumental…")
    i_y, _ = librosa.load(i_path, sr=PREVIEW_SR, mono=True,
                          offset=max(0.0, i_start), duration=i_load_dur)

    if abs(stretch - 1.0) > 1e-3:
        _tick(45, f"Time-stretching instrumental ×{stretch:.3f}…")
        i_y = librosa.effects.time_stretch(i_y, rate=float(stretch), n_fft=1024)

    if shift:
        _tick(65, f"Pitch-shifting instrumental {shift:+d} st…")
        i_y = librosa.effects.pitch_shift(i_y, sr=PREVIEW_SR, n_steps=shift, n_fft=1024)

    _tick(85, "Mixing…")
    n = min(len(v_y), len(i_y))
    if n <= 0:
        log.warning("preview: empty audio after processing")
        return None
    v_y, i_y = v_y[:n], i_y[:n]

    # Mix at natural stem levels — no per-stem normalization so the vocal and
    # instrumental volumes match what you hear when playing each stem alone.
    # Light gain on vocal to sit above the bed, then peak-limit to prevent clip.
    mix = v_y * 0.9 + i_y * 0.7
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 1.0:
        mix = mix / peak

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix.astype("float32"), PREVIEW_SR)
    _tick(100, "Preview ready")
    log.info("preview rendered: %s", out.name)
    return out
