"""Field drift: the helmet was calibrated in the lab, then the mounts moved (knock, transport, temperature).

Lab truth -> lab calibrations (handheld ChArUco, calibration cube with panel poses) from run_study.
Field truth = lab truth + extra mounting rotation (sigma `drift_deg`) and lens-centre shift (sigma 1 mm);
lenses themselves do not change.  Compared on field frames:
  stale lab calibration | production self-calibration in the field | hybrid = lab lens + field rotations.

  python -m calib_lab.drift [--variant 7x141] [--seed 1] [--drift 1.0]
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from smart_helmet_pano.rig import Rig, rot_ue
from . import methods as M
from .run_study import OUT, score, test_frames, gt_pano


def field_truth(true: Rig, drift_deg, seed):
    rng = np.random.default_rng(seed + 555)
    cams = []
    for c in true.cameras:
        d = copy.deepcopy(c)
        d.rotation_matrix = (np.asarray(c.R_rig_from_cam_ue) @ rot_ue(*rng.normal(0, drift_deg, 3))).tolist()
        d.position_m = (np.asarray(c.position_m) + rng.normal(0, 0.001, 3)).tolist()
        cams.append(d)
    return Rig(true.name + "_field", cams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="7x141")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drift", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    lab = M.make_ctx(a.variant, a.seed, OUT, a.workers)
    field = M.Ctx(lab.spec, a.seed, lab.nominal, field_truth(lab.true, a.drift, a.seed),
                  OUT / f"{a.variant}_s{a.seed}_drift{a.drift:g}", a.workers)
    field.out.mkdir(parents=True, exist_ok=True)
    (field.out / "rigs").mkdir(exist_ok=True)
    imgs, gt = test_frames(field), gt_pano()
    lab_rigs = {m: Rig.load(lab.out / "rigs" / f"{m}.json") for m in ("charuco", "cube_panels")}
    _, frames = M.render_natural(field, 8)
    res = {"variant": a.variant, "seed": a.seed, "drift_deg": a.drift, "methods": {}}
    runs = [("nominal", lambda: (lab.nominal, {})),
            ("lab_charuco_stale", lambda: (lab_rigs["charuco"], {})),
            ("lab_cube_stale", lambda: (lab_rigs["cube_panels"], {})),
            ("field_selfcal", lambda: M.run_selfcal(field)),
            ("hybrid_charuco_lens", lambda: M.run_hybrid(field, lab_rigs["charuco"], frames=frames)),
            ("hybrid_cube_lens", lambda: M.run_hybrid(field, lab_rigs["cube_panels"], frames=frames)),
            ("oracle", lambda: (field.true, {}))]
    for name, fn in runs:
        t0 = time.perf_counter()
        rig, info = fn()
        rig.save(field.out / "rigs" / f"{name}.json")
        sc = score(field, rig, imgs, gt, save_pano=None)
        res["methods"][name] = {"info": {k: v for k, v in info.items() if k != "per_camera"}, **sc,
                                "seconds": round(time.perf_counter() - t0, 1)}
        c, pi = sc["calib"], sc["pano_infinity"]
        print(f"  {name:20s} rot {c['rot_err_deg']:.3f} deg  ray {c['ray_rms_arcmin']:6.1f}'  "
              f"pano {pi['geom_err_arcmin']['mean']:6.1f}'  seam {pi['seam_break_arcmin']['mean']:6.1f}'", flush=True)
        (field.out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
