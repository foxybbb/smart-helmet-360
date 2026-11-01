"""Benchmark of the 5 research directions (Docs/optical_flow_review.md) on the same frames.

Variants (each a subclass / option of the production PanoStitcher):
  base       – production pipeline: DIS flow per ring pair in equirect, symmetric half-warp
  N5_rot     – + per-frame re-estimation of every camera rotation (absorbs head motion during the
               unsynchronised trigger) before stitching
  N1_spline  – global regularised spline displacement field per camera, solved jointly from the
               flow of ALL overlapping pairs incl. the top camera (Google "Seamless Street View")
  N3_tangent – flow computed in a gnomonic (tangent) view centred on every overlap instead of the
               distorted equirect (fixes the top/ring seam near the pole); fed into N1's solver
  N2_raft    – deep flow (torchvision RAFT-large, GPU) instead of DIS, inside N3+N1
  N4_depth   – no warping: per-pixel depth (triangulated from rig flow, optionally completed with
               Depth Anything V2) replaces the 6 m proxy sphere -> parallax-correct reprojection
Metrics: seam artifacts % (owner-boundary band, criterion < 10 %), seam MAD (mean abs grey
difference of the two cameras inside seam bands; lower = less ghosting), runtime; for the
synthetic room also PSNR against the ground-truth panorama.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_helmet_pano.calibrate import _Cal, _features, _match, _ransac_rotation, _CV2UE, _FLIP  # noqa: E402
from smart_helmet_pano.rig import Rig, Camera  # noqa: E402
from smart_helmet_pano.stitch import StitchOptions, PanoStitcher, equirect_dirs  # noqa: E402
from smart_helmet_pano.quality import seam_artifacts, psnr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TUNING = str(ROOT / "Data" / "imx219_160d.json")


from smart_helmet_pano.advanced import (DisFlow, RaftFlow, confidence, dir_to_eq, tangent_grid,  # noqa: E402,F401
    pair_correspondences, solve_global_warp, VariantStitcher, triangulated_depth, MonoDepth)


# ===================================================================== N5: per-frame rotation refinement
def refine_rotations(rig: Rig, imgs, verbose=False):
    names = [c.name for c in rig.cameras]
    cal = _Cal([imgs], names, rig, rig.cameras[0].fx)
    cal.f, cal.k = rig.cameras[0].fx, np.array(rig.cameras[0].k, float)
    cal.f_ref = cal.f
    cal.R = [_FLIP @ c.R_rig_from_cam_ue @ _CV2UE for c in rig.cameras]
    feats = [_features(im) for im in imgs]
    obs = []
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            m = _match(feats[i][1], feats[j][1])
            if len(m) < 12:
                continue
            pa, pb = feats[i][0][m[:, 0]], feats[j][0][m[:, 1]]
            ra = cal.rays(i, pa) @ cal.R[i].T
            rb = cal.rays(j, pb) @ cal.R[j].T
            R, inl = _ransac_rotation(ra, rb, thresh_deg=2.5)
            if inl.sum() >= 15 and math.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])) < 8:
                obs.append((i, j, pa[inl], pb[inl]))
    cal.obs = obs
    e0 = np.median(cal.errors(cal.f, cal.k, cal.R))
    cal.bundle_adjust(fix_lens=True, passes=2, verbose=verbose)
    e1 = np.median(cal.errors(cal.f, cal.k, cal.R))
    cams = []
    for c, R in zip(rig.cameras, cal.R):
        R_ue = _FLIP @ R @ _CV2UE.T
        d = dict(c.__dict__)
        d["rotation_matrix"] = R_ue.tolist()
        cams.append(Camera(**d))
    return Rig(rig.name + "_perframe", cams), {"median_px_before": round(float(e0), 2), "median_px_after": round(float(e1), 2),
                                                "pairs": len(obs)}


# ===================================================================== evaluation
def ssim(a, b, mask):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    ma, mb = mu(a), mu(b)
    va, vb, cab = mu(a * a) - ma * ma, mu(b * b) - mb * mb, mu(a * b) - ma * mb
    c1, c2 = 6.5025, 58.5225
    m = ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(m[mask].mean())

def seam_mad(warped, weights):
    Wt = np.stack(weights)
    owner = np.where(Wt.max(0) > 0, np.argmax(Wt, 0), -1)
    H, W = owner.shape
    edge = np.zeros((H, W), bool)
    edge[:, :-1] |= owner[:, :-1] != owner[:, 1:]
    edge[:-1, :] |= owner[:-1, :] != owner[1:, :]
    band = cv2.dilate(edge.astype(np.uint8), np.ones((2 * max(2, W // 256) + 1,) * 2, np.uint8)) > 0
    g = [cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(np.float32) for w in warped]
    diffs = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            m = band & (weights[i] > 0.05) & (weights[j] > 0.05) & ((owner == i) | (owner == j))
            m &= (g[i] < 248) & (g[j] < 248)
            if m.sum() > 100:
                diffs.append(np.abs(g[i][m] - g[j][m]))
    return float(np.mean(np.concatenate(diffs))) if diffs else float("nan")


def opts_for(real, width):
    if real:
        return StitchOptions(width=width, level="none", proxy_depth_m=6, tuning=TUNING, theta_max_deg=75,
                             theta_ramp_deg=75, max_flow_frac=0.005, flow_sigma_frac=0.008)
    return StitchOptions(width=width, level="tilt", proxy_depth_m=4)


def run_variant(name, rig, imgs, meta, real, width, flows, mono=None, gt=None, opts=None, gt_mask=None):
    o = opts or opts_for(real, width)
    info = {}
    t = time.perf_counter()
    r = rig
    if name.startswith("N5") or "+N5" in name:
        r, info["rot"] = refine_rotations(rig, imgs)
    if name.startswith("N4"):
        # 1st pass (no warp) -> correspondences -> depth -> 2nd pass with depth proxy, no flow warp
        o1 = StitchOptions(**{**o.__dict__, "flow": False})
        s1 = VariantStitcher(r, o1)
        s1.stitch(imgs, meta)
        corr = pair_correspondences(s1, s1.last_warped, s1.last_weights, flows["DIS"], mode="tangent")
        from smart_helmet_pano.stitch import leveling_matrix
        maps = s1.build_maps(leveling_matrix(meta, o.level, r))
        mono_map = mono.pano(s1, imgs, maps, s1.last_weights) if (mono is not None and "mono" in name) else None
        depth, info["depth"] = triangulated_depth(s1, corr, maps, s1.W, s1.H, mono=mono_map)
        o2 = StitchOptions(**{**o.__dict__, "flow": "flow" in name})
        st = VariantStitcher(r, o2, align="global", corr_mode="tangent", flow=flows["DIS"], depth_map=depth)
    elif name.startswith("base"):
        st = VariantStitcher(r, o, align="base")
    elif name.startswith("N1"):
        st = VariantStitcher(r, o, align="global", corr_mode="equirect", flow=flows["DIS"])
    elif name.startswith("N3"):
        st = VariantStitcher(r, o, align="global", corr_mode="tangent", flow=flows["DIS"])
    elif name.startswith("N2"):
        st = VariantStitcher(r, o, align="global", corr_mode="tangent", flow=flows["RAFT"])
    elif name.startswith("N5"):
        st = VariantStitcher(r, o, align="base")
    else:
        raise ValueError(name)
    pano = st.stitch(imgs, meta)
    dt = time.perf_counter() - t
    q, _ = seam_artifacts(st.last_warped, st.last_weights, st.W, st.H)
    res = {"seam_artifacts_%": q["artifact_%_of_central"], "seam_MAD": round(seam_mad(st.last_warped, st.last_weights), 2),
           "seconds": round(dt, 1), **{k: v for k, v in {**info, **st.info}.items()}}
    if gt is not None:
        cov = np.sum(st.last_weights, 0) > 0
        if gt_mask is not None:
            cov &= gt_mask
        # global per-channel gain to the reference exposure (exposure normalisation is not an error)
        pf, gf = pano.astype(np.float32), gt.astype(np.float32)
        g = np.array([np.median(gf[..., c][cov] + 1) / np.median(pf[..., c][cov] + 1) for c in range(3)])
        pano_n = np.clip(pf * g, 0, 255).astype(np.uint8)
        res["PSNR_vs_GT"] = round(psnr(pano_n, gt, cov), 2)
        res["SSIM_vs_GT"] = round(ssim(pano_n, gt, cov), 4)
    return pano, res
