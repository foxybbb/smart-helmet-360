"""Advanced stitching mode "depth": parallax-correct reprojection for real helmet photos.

Winner of the research-direction benchmark (Docs/directions_results.md):
  1. first pass (sphere proxy, no warp) -> flow correspondences between all overlapping cameras,
     computed in gnomonic views centred on every overlap (low distortion)
  2. triangulate the correspondences with the real lens centres (metric depth at the seams)
  3. Depth Anything V2 (monocular, GPU) gives dense relative depth for every camera; it is affine-fitted
     to the triangulated depth and fills the panorama everywhere
  4. second pass: every pixel is projected at its own depth instead of a 6 m sphere, then a global
     regularised spline warp (Google "Seamless Street View" idea) removes the remaining seam error.
Real helmet, 6 frames: seam artifacts 1.65 % -> 0.80 %, seam MAD 26 -> 18 (Output/real_eval).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .rig import Rig
from .stitch import PanoStitcher, StitchOptions, leveling_matrix

# ===================================================================== flow back-ends
class DisFlow:
    name = "DIS"

    def __init__(self):
        self.d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def __call__(self, ga, gb):
        return self.d.calc(np.ascontiguousarray(ga), np.ascontiguousarray(gb), None)


class RaftFlow:
    """torchvision RAFT-large (C+T+S+K+H weights) on CUDA; grey -> 3ch, sizes padded to /8."""
    name = "RAFT"

    def __init__(self):
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(self.dev)

    def __call__(self, ga, gb):
        t = self.torch
        h, w = ga.shape
        H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8
        def prep(g):
            g = cv2.copyMakeBorder(g, 0, H - h, 0, W - w, cv2.BORDER_REPLICATE)
            x = t.from_numpy(np.repeat(g[None, None].astype(np.float32), 3, 1)) / 127.5 - 1
            return x.to(self.dev)
        with t.no_grad():
            f = self.model(prep(ga), prep(gb), num_flow_updates=12)[-1]
        return f[0].permute(1, 2, 0).cpu().numpy()[:h, :w].astype(np.float32)


def confidence(ga, gb, fab, fba, fb_thresh=2.0):
    gx = cv2.Sobel(ga, cv2.CV_32F, 1, 0); gy = cv2.Sobel(ga, cv2.CV_32F, 0, 1)
    tex = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2) > 12
    unsat = (ga < 245) & (gb < 245) & (ga > 8)
    h, w = ga.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    back = cv2.remap(fba, xx + fab[..., 0], yy + fab[..., 1], cv2.INTER_LINEAR)
    return tex & unsat & (np.linalg.norm(fab + back, axis=2) < fb_thresh)


# ===================================================================== geometry helpers
def dir_to_eq(d, W, H):
    lon = np.arctan2(d[..., 1], d[..., 0])
    lat = np.arcsin(np.clip(d[..., 2], -1, 1))
    return (lon / (2 * np.pi) + 0.5) * W - 0.5, (0.5 - lat / np.pi) * H - 0.5


def tangent_grid(center, size, half_fov_deg):
    c = center / np.linalg.norm(center)
    up = np.array([0, 0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0, 0])
    ex = np.cross(up, c); ex /= np.linalg.norm(ex)
    ey = np.cross(c, ex)
    f = (size / 2) / math.tan(math.radians(half_fov_deg))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    a, b = (xx - size / 2 + 0.5) / f, (yy - size / 2 + 0.5) / f
    d = c + a[..., None] * ex - b[..., None] * ey
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# ===================================================================== correspondences between warped images
def pair_correspondences(st, warped, weights, flow, mode="equirect", stride=3):
    """List of (i, j, xa[N,2], f[N,2]) in panorama pixels: content at xa in cam i is at xa+f in cam j."""
    W, H = st.W, st.H
    n = len(warped)
    grays = [cv2.cvtColor(w, cv2.COLOR_BGR2GRAY) for w in warped]
    out = []
    dirs = st._dirs
    for i in range(n):
        for j in range(i + 1, n):
            ov = (weights[i] > 0.05) & (weights[j] > 0.05)
            if ov.sum() < 800:
                continue
            if mode == "equirect":
                ys, xs = np.nonzero(ov)
                shift = W // 2 if xs.max() - xs.min() > W / 2 else 0
                xs2 = (xs + shift) % W
                x0, x1 = max(xs2.min() - 16, 0), min(xs2.max() + 16, W)
                y0, y1 = max(ys.min() - 16, 0), min(ys.max() + 16, H)
                crop = lambda im: np.roll(im, shift, axis=1)[y0:y1, x0:x1]
                ga, gb, m = crop(grays[i]), crop(grays[j]), crop(ov)
                fab, fba = flow(ga, gb), flow(gb, ga)
                ok = confidence(ga, gb, fab, fba) & m
                yy, xx = np.nonzero(ok[::stride, ::stride])
                yy, xx = yy * stride, xx * stride
                xa = np.stack([(xx + x0 - shift) % W, yy + y0], 1).astype(np.float32)
                out.append((i, j, xa, fab[yy, xx]))
            else:  # tangent / gnomonic view centred on the overlap
                c = dirs[ov].mean(0)
                c /= np.linalg.norm(c)
                ang = np.degrees(np.arccos(np.clip(dirs[ov][::50] @ c, -1, 1)))
                half = float(np.clip(np.percentile(ang, 98) + 3, 15, 70))
                S = 640 if W <= 2048 else 1024
                D = tangent_grid(c, S, half)
                mx, my = dir_to_eq(D, W, H)
                mx, my = mx.astype(np.float32), my.astype(np.float32)
                rem = lambda im, interp=cv2.INTER_LINEAR: cv2.remap(im, mx, my, interp, borderMode=cv2.BORDER_WRAP)
                ga, gb = rem(grays[i]), rem(grays[j])
                m = rem(ov.astype(np.uint8), cv2.INTER_NEAREST) > 0
                fab, fba = flow(ga, gb), flow(gb, ga)
                ok = confidence(ga, gb, fab, fba) & m
                yy, xx = np.nonzero(ok[::stride, ::stride])
                yy, xx = yy * stride, xx * stride
                if len(yy) == 0:
                    continue
                f = 2 * math.tan(math.radians(half)) / S
                # tangent pixel -> direction -> equirect, for both ends of the flow vector
                def t2eq(px, py):
                    ex_ = (px - S / 2 + 0.5) * f
                    ey_ = (py - S / 2 + 0.5) * f
                    up = np.array([0, 0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0, 0])
                    e1 = np.cross(up, c); e1 /= np.linalg.norm(e1); e2 = np.cross(c, e1)
                    d = c + ex_[:, None] * e1 - ey_[:, None] * e2
                    d /= np.linalg.norm(d, axis=1, keepdims=True)
                    return np.stack(dir_to_eq(d, W, H), 1)
                pa = t2eq(xx.astype(float), yy.astype(float))
                pb = t2eq(xx + fab[yy, xx, 0].astype(float), yy + fab[yy, xx, 1].astype(float))
                fe = pb - pa
                fe[:, 0] = (fe[:, 0] + W / 2) % W - W / 2
                out.append((i, j, pa.astype(np.float32), fe.astype(np.float32)))
    return out


# ===================================================================== N1: global spline displacement
def solve_global_warp(corr, n, W, H, grid=(96, 48), lam=4.0, mu=0.05, max_disp_frac=0.01, iters=3,
                      max_per_pair=2500):
    """Per-camera displacement u_c on a bilinear control grid (wraps in x).  Model: a point at xa in
    cam i and xa+f in cam j land on the same output pixel  =>  u_i(xa) - u_j(xa) = f."""
    rng = np.random.default_rng(0)
    corr = [(i, j, xa[s], f[s]) for (i, j, xa, f) in corr
            for s in [rng.choice(len(xa), min(len(xa), max_per_pair), replace=False) if len(xa) else slice(0, 0)]]
    gx, gy = grid
    ncell = gx * gy
    nvar = n * ncell                     # per component (x and y solved separately, same matrix)
    rows, cols, vals, rhs_x, rhs_y, wts = [], [], [], [], [], []
    r = 0
    eq_pts = []
    for (i, j, xa, f) in corr:
        if len(xa) == 0:
            continue
        u = xa[:, 0] / W * gx
        v = np.clip(xa[:, 1] / H * (gy - 1), 0, gy - 1.001)
        u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
        du, dv = u - u0, v - v0
        for k in range(len(xa)):
            for (cu, cv, w_) in ((u0[k], v0[k], (1 - du[k]) * (1 - dv[k])), (u0[k] + 1, v0[k], du[k] * (1 - dv[k])),
                                 (u0[k], v0[k] + 1, (1 - du[k]) * dv[k]), (u0[k] + 1, v0[k] + 1, du[k] * dv[k])):
                cell = (cv * gx + cu % gx)
                rows += [r, r]; cols += [i * ncell + cell, j * ncell + cell]; vals += [w_, -w_]
            rhs_x.append(f[k, 0]); rhs_y.append(f[k, 1]); r += 1
        eq_pts.append(len(xa))
    n_data = r
    # smoothness (first differences, wrap in x) + damping
    for c in range(n):
        base = c * ncell
        for yy in range(gy):
            for xx in range(gx):
                a = base + yy * gx + xx
                b = base + yy * gx + (xx + 1) % gx
                rows += [r, r]; cols += [a, b]; vals += [lam, -lam]; rhs_x.append(0); rhs_y.append(0); r += 1
                if yy + 1 < gy:
                    b = base + (yy + 1) * gx + xx
                    rows += [r, r]; cols += [a, b]; vals += [lam, -lam]; rhs_x.append(0); rhs_y.append(0); r += 1
                rows.append(r); cols.append(a); vals.append(mu); rhs_x.append(0); rhs_y.append(0); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    bx, by = np.array(rhs_x, float), np.array(rhs_y, float)
    w = np.ones(r)
    for _ in range(iters):  # IRLS (Huber) against bad flow
        Aw = sp.diags(w) @ A
        sx = lsqr(Aw, w * bx, atol=1e-6, btol=1e-6, iter_lim=3000)[0]
        sy = lsqr(Aw, w * by, atol=1e-6, btol=1e-6, iter_lim=3000)[0]
        res = np.hypot(A[:n_data] @ sx - bx[:n_data], A[:n_data] @ sy - by[:n_data])
        k = 3.0
        w[:n_data] = np.where(res < k, 1.0, k / np.maximum(res, 1e-6))
    lim = max_disp_frac * W
    fields = []
    for c in range(n):
        ux = np.clip(sx[c * ncell:(c + 1) * ncell].reshape(gy, gx), -lim, lim)
        uy = np.clip(sy[c * ncell:(c + 1) * ncell].reshape(gy, gx), -lim, lim)
        # bilinear up-sampling with x wrap
        ux = np.hstack([ux, ux[:, :1]]); uy = np.hstack([uy, uy[:, :1]])
        fx = cv2.resize(ux.astype(np.float32), (W + W // gx, H), interpolation=cv2.INTER_LINEAR)[:, :W]
        fy = cv2.resize(uy.astype(np.float32), (W + W // gx, H), interpolation=cv2.INTER_LINEAR)[:, :W]
        fields.append(np.stack([fx, fy], -1))
    return fields, {"equations": n_data, "residual_px_median": round(float(np.median(res)), 2)}


class VariantStitcher(PanoStitcher):
    """PanoStitcher with pluggable alignment (base | global) and correspondence mode / flow back-end."""

    def __init__(self, rig, opts, align="base", corr_mode="equirect", flow=None, depth_map=None):
        super().__init__(rig, opts)
        self.align, self.corr_mode = align, corr_mode
        self.flow_fn = flow or DisFlow()
        self.depth_map = depth_map
        self.info = {}

    def build_maps(self, M):
        if self.depth_map is None:
            return super().build_maps(M)
        # per-pixel proxy depth instead of a constant sphere
        keep = self._dirs
        self._dirs = keep * self.depth_map[..., None].astype(np.float32) / self.o.proxy_depth_m
        try:
            maps = super().build_maps(M)
        finally:
            self._dirs = keep
        return maps

    def _flow_align(self, warped, weights):
        if self.align == "base":
            if isinstance(self.flow_fn, DisFlow):
                return super()._flow_align(warped, weights)
        corr = pair_correspondences(self, warped, weights, self.flow_fn, mode=self.corr_mode)
        self.last_corr = corr
        fields, info = solve_global_warp(corr, len(warped), self.W, self.H)
        self.info.update(info)
        gx, gy = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        return [cv2.remap(img, gx - u[..., 0], gy - u[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                for img, u in zip(warped, fields)]



# ===================================================================== N4: depth from rig flow (+ optional mono prior)
def triangulated_depth(st, corr, maps, W, H, default=6.0, mono=None):
    """Triangulate every correspondence with the real lens centres; splat depth into the panorama."""
    cams = st.rig.cameras
    pts, dep = [], []
    for (i, j, xa, f) in corr:
        if len(xa) == 0:
            continue
        xb = xa + f
        xb[:, 0] %= W
        def cam_px(c, p):
            mx, my, _ = maps[c]
            x = np.clip(p[:, 0], 0, W - 1).astype(np.float32).reshape(-1, 1)
            y = np.clip(p[:, 1], 0, H - 1).astype(np.float32).reshape(-1, 1)
            pad = (-len(x)) % 1024
            x = np.pad(x, ((0, pad), (0, 0))).reshape(-1, 1024); y = np.pad(y, ((0, pad), (0, 0))).reshape(-1, 1024)
            u = cv2.remap(mx, x, y, cv2.INTER_LINEAR).ravel()[:len(p)]
            v = cv2.remap(my, x, y, cv2.INTER_LINEAR).ravel()[:len(p)]
            return u, v
        ua, va = cam_px(i, xa)
        ub, vb = cam_px(j, xb)
        a = cams[i].unproject(ua, va); b = cams[j].unproject(ub, vb)
        ci, cj = np.asarray(cams[i].position_m), np.asarray(cams[j].position_m)
        w0 = ci - cj
        B = np.sum(a * b, 1); D_ = a @ w0; E = b @ w0
        den = 1 - B * B
        s = (B * E - D_) / np.maximum(den, 1e-9)
        t = (E - B * D_) / np.maximum(den, 1e-9)
        X = 0.5 * (ci + s[:, None] * a + cj + t[:, None] * b)
        r = np.linalg.norm(X, axis=1)
        good = (den > 1e-6) & (s > 0.2) & (t > 0.2) & (r > 0.3) & (r < 40)
        pts.append(xa[good]); dep.append(r[good])
    if not pts:
        return np.full((H, W), default, np.float32), {"points": 0}
    P = np.concatenate(pts); Dp = np.concatenate(dep)
    # robust splat in inverse depth (normalized convolution), default where unknown
    inv = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    xi, yi = P[:, 0].astype(int) % W, np.clip(P[:, 1].astype(int), 0, H - 1)
    np.add.at(inv, (yi, xi), 1.0 / Dp); np.add.at(cnt, (yi, xi), 1.0)
    sig = W / 160
    pad = int(4 * sig)
    wrap_blur = lambda a: cv2.GaussianBlur(np.hstack([a[:, -pad:], a, a[:, :pad]]), (0, 0), sig)[:, pad:pad + W]
    num, den = wrap_blur(inv), wrap_blur(cnt)
    have = den > 0.05
    invd = np.where(have, num / np.maximum(den, 1e-6), 1.0 / default)
    info = {"points": int(len(Dp)), "depth_m_p10_p50_p90": [round(float(x), 2) for x in np.percentile(Dp, [10, 50, 90])]}
    if mono is not None:  # affine-fit mono inverse depth to triangulated inverse depth, use it where no points
        m = mono[yi, xi]
        A = np.stack([m, np.ones_like(m)], 1)
        sol, *_ = np.linalg.lstsq(A, 1.0 / Dp, rcond=None)
        mono_inv = np.clip(mono * sol[0] + sol[1], 1 / 40, 1 / 0.3)
        blend = np.clip(den / 0.5, 0, 1)
        invd = blend * invd + (1 - blend) * mono_inv
        info["mono_fit"] = [round(float(sol[0]), 4), round(float(sol[1]), 4)]
    depth = np.clip(1.0 / np.maximum(invd, 1e-3), 0.4, 40).astype(np.float32)
    return depth, info


class MonoDepth:
    """Depth Anything V2 (Small) via HF transformers on the GPU; returns relative inverse depth."""

    def __init__(self):
        import torch
        from transformers import pipeline
        self.p = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                          device=0 if torch.cuda.is_available() else -1)

    def pano(self, st, imgs, maps, weights):
        from PIL import Image
        acc = np.zeros((st.H, st.W), np.float32); wsum = np.zeros_like(acc)
        for im, (mx, my, w), ww in zip(imgs, maps, weights):
            out = self.p(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))["predicted_depth"]
            d = out.squeeze().cpu().numpy().astype(np.float32)
            d = cv2.resize(d, (im.shape[1], im.shape[0]))
            d = (d - np.percentile(d, 2)) / max(np.percentile(d, 98) - np.percentile(d, 2), 1e-6)  # rel. inverse depth
            dw = cv2.remap(d, mx, my, cv2.INTER_LINEAR)
            acc += dw * ww; wsum += ww
        return acc / np.maximum(wsum, 1e-6)



# ===================================================================== production entry point
_MONO = None


def mono_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"torch/transformers missing ({e})"
    import torch
    if not torch.cuda.is_available():
        return False, "no CUDA GPU"
    return True, "ok"


def get_mono():
    global _MONO
    if _MONO is None:
        _MONO = MonoDepth()
    return _MONO


def stitch_depth(rig: Rig, imgs, meta, opts: StitchOptions, flow=None, use_mono: bool = True):
    """Two-pass depth-aware stitch. Returns (panorama, stitcher, info)."""
    t0 = time.perf_counter()
    flow = flow or DisFlow()
    o1 = StitchOptions(**{**opts.__dict__, "flow": False})
    s1 = VariantStitcher(rig, o1)
    s1.stitch(imgs, meta)
    corr = pair_correspondences(s1, s1.last_warped, s1.last_weights, flow, mode="tangent")
    maps = s1.build_maps(leveling_matrix(meta, opts.level, rig))
    mono_map = get_mono().pano(s1, imgs, maps, s1.last_weights) if use_mono else None
    depth, dinfo = triangulated_depth(s1, corr, maps, s1.W, s1.H, mono=mono_map)
    o2 = StitchOptions(**{**opts.__dict__, "flow": True})
    st = VariantStitcher(rig, o2, align="global", corr_mode="tangent", flow=flow, depth_map=depth)
    pano = st.stitch(imgs, meta)
    info = {"mode": "depth" + ("+mono" if use_mono else ""), "depth": dinfo, **st.info,
            "seconds": round(time.perf_counter() - t0, 1)}
    st.timings["total"] = time.perf_counter() - t0
    st.depth_map = depth
    return pano, st, info
