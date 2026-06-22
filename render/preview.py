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
