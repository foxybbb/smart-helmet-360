"""Lens model experiment: Kannala-Brandt (fx, fy, cx, cy, k1..k4) vs + decentering p1, p2.

Re-solves the best stands on the already rendered frames of run_study (helmet samples seed 1 and 2) with the
extended model and writes Output/calib_study/tangential.json (used by summarize).

  python -m calib_lab.lens_model [7x141 ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import evaluate as E
from . import methods as M
from .run_study import OUT, test_room


def main(keys=None, seeds=(1, 2)):
    keys = keys or ["7x141"]
    out = {}
    M.TANGENTIAL = True
    for key in keys:
        for seed in seeds:
            ctx = M.make_ctx(key, seed, OUT)
            ctx.verbose = False
            runs = [("charuco", lambda: M.run_charuco(ctx)),
                    ("cube_panels", lambda: M.run_cube(ctx, trust_build=False)),
                    ("projector_scaled", lambda: M.run_projector(ctx, scale_bars=True)),
                    ("markers", lambda: M.run_markers(ctx))]
            for name, fn in runs:
                rig, info = fn()
                rig.save(ctx.out / "rigs" / f"{name}_tangential.json")
                al = E.align(rig, ctx.true)
                agg, _ = E.camera_errors(al, ctx.true)
                pi = E.pano_distortion(al, ctx.true, np.eye(3), np.zeros(3))
                td = E.pano_distortion(al, ctx.true, np.eye(3), np.zeros(3), scene=test_room(), W=1024, depth="true")
                out.setdefault(f"{key}_s{seed}", {})[name] = {"rms_px": info["rms_px"], "calib": agg,
                                                              "pano_infinity": pi, "pano_room_truedepth": td}
                print(f"{key}_s{seed} {name}: rot {agg['rot_err_deg']:.3f} deg, ray {agg['ray_rms_arcmin']:.2f}', "
                      f"pano {pi['geom_err_arcmin']['mean']:.2f}', seam {pi['seam_break_arcmin']['mean']:.2f}', "
                      f"true-depth seam {td['seam_break_arcmin']['mean']:.2f}'", flush=True)
                (OUT / "tangential.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
