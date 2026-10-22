"""Hear and see whether a pair's bar lines line up (readme §5.7, §8).

For one stored pair (or one section pair), render:

  <data_dir>/bench/align/<v>_<b>/overlay.wav    stereo clicks — the vocal's bar
                                                lines left, the bed's right, as
                                                placed; one click = a grid that
                                                holds, a flam = drift
  <data_dir>/bench/align/<v>_<b>/mix_clicks.wav the two conformed stems summed
                                                with the clicks on top
  <data_dir>/bench/align/<v>_<b>/grids.png      both grids on one time axis
                                                (needs matplotlib)

    docker compose exec app python scripts/align_debug.py 12 40
    python scripts/align_debug.py 12 40 --vocal-section 1002 --inst-section 1003

The placement is matcher/phrase_align.place under the configured engine's
sections (the allin1 grid in allin1 mode), at the pair's own stretch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SR = 44100


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vocal_song_id", type=int)
    ap.add_argument("inst_song_id", type=int)
    ap.add_argument("--vocal-section", type=int, default=None)
    ap.add_argument("--inst-section", type=int, default=None)
    args = ap.parse_args(argv)
    import soundfile as sf
    from config import DATA_DIR, current_float
    from database import alignment, models
    from matcher.match import compute_stretch_factor
    from matcher.phrase_align import overlay_clicks, place, scale_grid
    from render.dsp import conform, load_segment, resolve_stem_path

    v_id, b_id = args.vocal_song_id, args.inst_song_id
    vs, bs = args.vocal_section, args.inst_section
    if vs is None or bs is None:
        conn = models.get_conn()
        r = conn.execute("SELECT vocal_section_idx, inst_section_idx FROM mashup_candidates "
                         "WHERE vocal_song_id=? AND inst_song_id=? AND vocal_section_idx "
                         "IS NOT NULL ORDER BY score_total DESC LIMIT 1", (v_id, b_id)).fetchone()
        conn.close()
        if not r:
            ap.error("no stored pairing for that pair — pass --vocal-section/--inst-section")
        vs, bs = r[0], r[1]
    v_sec = next((s for s in models.get_sections(v_id) if s["section_index"] == vs), None)
    b_sec = next((s for s in models.get_sections(b_id) if s["section_index"] == bs), None)
    if not v_sec or not b_sec:
        ap.error("section not found under the configured engine")
    v_feat = models.get_features_for_song(v_id, "full") or {}
    b_feat = models.get_features_for_song(b_id, "full") or {}
    stretch = compute_stretch_factor(v_feat.get("bpm"), b_feat.get("bpm")) or 1.0
    p = place(v_sec, b_sec, stretch, phrases=alignment.get_vocal_phrases(v_id),
              tol_ms=current_float("timing_tol_ms"))
    print({k: p[k] for k in p})
    if p["bed_landing_sec"] is None:
        print("no grid to place on", file=sys.stderr)
        return 1

    d_v, land = p["alignment_downbeat"], p["bed_landing_sec"]
    lead = max(0.0, -(p["vocal_pickup_ms"] or 0) / 1000.0)
    length = min(v_sec["end_sec"] - d_v, (b_sec["end_sec"] - land) / stretch) + lead
    v_dbs = [t - d_v + lead for t in v_sec.get("downbeats") or [] if t >= d_v - 1e-3]
    b_dbs = list(scale_grid([t for t in b_sec.get("downbeats") or [] if t >= land - 1e-3],
                            stretch, land) + lead)
    out = DATA_DIR / "bench" / "align" / f"{v_id}_{b_id}"
    out.mkdir(parents=True, exist_ok=True)
    clicks = overlay_clicks(v_dbs, b_dbs, 0.0, length, sr=SR)
    sf.write(str(out / "overlay.wav"), clicks, SR)

    vp, bp = resolve_stem_path(v_id, "vocals"), resolve_stem_path(b_id, "instrumental")
    if vp and bp:
        v = conform(load_segment(vp, SR, start_sec=d_v - lead, end_sec=d_v + length), SR, 1.0, 0)
        b = conform(load_segment(bp, SR, start_sec=land - lead * stretch,
                                 end_sec=land + length * stretch, rate=stretch), SR, stretch, 0)
        n = min(len(v), len(b), len(clicks))
        mix = 0.5 * (v[:n] + b[:n])
        sf.write(str(out / "mix_clicks.wav"),
                 np.stack([mix + clicks[:n, 0], mix + clicks[:n, 1]], axis=1), SR)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 2.4))
        ax.vlines(v_dbs, 0.5, 1.0, color="tab:pink", label="vocal bar lines")
        ax.vlines(b_dbs, 0.0, 0.5, color="tab:cyan", label="bed bar lines (conformed)")
        ax.set_xlim(0, length)
        ax.set_yticks([])
        ax.set_xlabel("seconds from the start of the overlap")
        ax.set_title(f"timing_ok={p['timing_ok']} mean {p['timing_err_mean_ms']} ms, "
                     f"worst {p['timing_err_max_ms']} ms, pickup {p['vocal_pickup_ms']} ms")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "grids.png", dpi=120)
    except ImportError:
        pass
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
