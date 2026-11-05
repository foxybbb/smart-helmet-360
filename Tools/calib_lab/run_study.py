"""Calibration study: every calibration stand x every overlap variant, scored against the truth.

  python -m calib_lab.run_study --variants 7x141 6x141 8x141 6x160 7x160 --seeds 1 [--methods ...]

Per variant and seed: Output/calib_study/<variant>_s<seed>/{results.json, rigs/<method>.json, frames}
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from smart_helmet_pano.rig import Rig
from smart_helmet_pano.stitch import equirect_dirs
from . import evaluate as E
from . import methods as M
from . import scenes as S
from .render import render_jobs, scene_key
from .truth import overlap_stats, describe_truth

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Output" / "calib_study"
W_IMG = 2048          # stitched test panorama width
W_GEO = 1024          # analytic distortion map width

METHODS = ["nominal", "selfcal", "charuco", "markers", "markers_surveyed", "projector", "projector_scaled",
           "cube_design", "cube_panels", "hybrid", "oracle"]

_TEST = {}


def test_room():
    if "scene" not in _TEST:
        _TEST["scene"] = S.TestRoom()
    return _TEST["scene"]


def gt_pano():
    p = OUT / f"gt_test_{W_IMG}.png"
    if not p.exists():
        d = equirect_dirs(W_IMG, W_IMG // 2).astype(np.float64)
        rad = test_room().shade(np.zeros(3), d.reshape(-1, 3)).reshape(W_IMG // 2, W_IMG, 3)
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.clip(rad * 255 + 0.5, 0, 255).astype(np.uint8))
    return cv2.imread(str(p))


def test_frames(ctx):
    jobs = {ci: [{"file": f"t00_{c.name}.jpg", "R_w": np.eye(3).tolist(), "t_w": [0.0, 0.0, 0.0], "seed": 9000 + ci}]
            for ci, c in enumerate(ctx.true.cameras)}
    render_jobs(scene_key("TestRoom"), ctx.true, jobs, ctx.out / "test", workers=ctx.workers)
    return [cv2.imread(str(ctx.out / "test" / f"t00_{c.name}.jpg")) for c in ctx.true.cameras]


def calibrate(ctx, m):
    if m == "nominal":
        return ctx.nominal, {"method": m}
    if m == "oracle":
        return ctx.true, {"method": m}
    if m == "selfcal":
        return M.run_selfcal(ctx)
    if m == "charuco":
        return M.run_charuco(ctx)
    if m == "markers":
        return M.run_markers(ctx)
    if m == "markers_surveyed":
        return M.run_markers(ctx, surveyed=True)
    if m == "projector":
        return M.run_projector(ctx)
    if m == "projector_scaled":
        return M.run_projector(ctx, scale_bars=True)
    if m == "cube_design":
        return M.run_cube(ctx, trust_build=True)
    if m == "cube_panels":
        return M.run_cube(ctx, trust_build=False)
    if m == "hybrid":
        lens = Rig.load(ctx.out / "rigs" / "charuco.json")
        return M.run_hybrid(ctx, lens)
    raise KeyError(m)


def score(ctx, rig, imgs, gt, save_pano=None):
    al = E.align(rig, ctx.true) if rig is not ctx.true else rig
    agg, rows = E.camera_errors(al, ctx.true)
    inf = E.pano_distortion(al, ctx.true, np.eye(3), np.zeros(3), scene=None, W=W_GEO)
    room = E.pano_distortion(al, ctx.true, np.eye(3), np.zeros(3), scene=test_room(), W=W_GEO, proxy_depth_m=3.0)
    pano, img = E.image_metrics(al, imgs, np.eye(3), gt, W_IMG, flow=False)
    _, img_flow = E.image_metrics(al, imgs, np.eye(3), gt, W_IMG, flow=True)
    if save_pano:
        cv2.imwrite(str(save_pano), pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"calib": agg, "cameras": rows, "pano_infinity": inf, "pano_room": room, "image": img,
            "image_flow": img_flow, **score_true_depth(ctx, al, imgs, gt, save_pano)}


def depth_map(W):
    p = OUT / f"depth_test_{W}.npy"
    if not p.exists():
        np.save(p, E.true_depth(test_room(), np.zeros(3), W))
    return np.load(p)


def score_true_depth(ctx, al, imgs, gt, save_pano=None):
    """Ideal depth mode: every pixel projected at its true distance -> only the calibration error is left."""
    room = E.pano_distortion(al, ctx.true, np.eye(3), np.zeros(3), scene=test_room(), W=W_GEO, depth="true")
    pano, img = E.image_metrics(al, imgs, np.eye(3), gt, W_IMG, flow=False, proxy_depth_m=depth_map(W_IMG))
    if save_pano:
        cv2.imwrite(str(save_pano).replace(".jpg", "_truedepth.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"pano_room_truedepth": room, "image_truedepth": img}


def run_variant(key, seed, methods, workers=7):
    ctx = M.make_ctx(key, seed, OUT, workers)
    res_p = ctx.out / "results.json"
    res = json.loads(res_p.read_text(encoding="utf-8")) if res_p.exists() else {}
    res["variant"] = key
    res["seed"] = seed
    res["overlap"] = overlap_stats(ctx.nominal)
    res["truth"] = describe_truth(ctx.true, ctx.nominal)
    res.setdefault("methods", {})
    (ctx.out / "rigs").mkdir(exist_ok=True)
    (ctx.out / "panos").mkdir(exist_ok=True)
    imgs = test_frames(ctx)
    gt = gt_pano()
    print(f"== {key} seed {seed}: overlap {res['overlap']}")
    for m in methods:
        rp = ctx.out / "rigs" / f"{m}.json"
        if m in res["methods"] and "error" not in res["methods"][m] and (rp.exists() or m in ("nominal", "oracle")):
            continue
        t0 = time.perf_counter()
        try:
            rig, info = calibrate(ctx, m)
            if m not in ("oracle",):
                rig.save(rp)
            sc = score(ctx, rig, imgs, gt, save_pano=ctx.out / "panos" / f"{m}.jpg")
            res["methods"][m] = {"info": info, **sc, "seconds_total": round(time.perf_counter() - t0, 1)}
            c, pi, pr = sc["calib"], sc["pano_infinity"], sc["pano_room"]
            print(f"  {m:17s} rot {c['rot_err_deg']:.3f} deg (max {c['rot_err_max_deg']:.2f})  ray {c['ray_rms_arcmin']:6.1f}'"
                  f"  pano inf {pi['geom_err_arcmin'].get('mean', -1):6.1f}' seam {pi['seam_break_arcmin'].get('mean', -1):6.1f}'"
                  f"  room seam {pr['seam_break_arcmin'].get('mean', -1):6.1f}'  PSNR {sc['image']['PSNR']:.2f}"
                  f"  art {sc['image']['seam_artifacts_%']:.2f}%  [{time.perf_counter() - t0:.0f}s]", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            res["methods"][m] = {"error": repr(e)}
        res_p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["7x141", "6x141", "8x141", "6x160", "7x160"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    for s in a.seeds:
        for v in a.variants:
            run_variant(v, s, a.methods, a.workers)


if __name__ == "__main__":
    main()
