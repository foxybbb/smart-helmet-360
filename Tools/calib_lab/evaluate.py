"""How good is a calibration?  Errors against the truth and the distortion of the resulting panorama.

* camera errors  – mounting rotation, lens centre, focal, principal point, and the RAY error over the whole
                   image (angle between the estimated and the true viewing ray of every pixel; lens-only
                   part in the camera frame, total part in the rig frame)
* panorama distortion (exact, analytic) – for every equirect pixel the stitcher picks a camera pixel with
                   the ESTIMATED rig; we trace the TRUE ray of that pixel into the scene and measure where it
                   really looks:  geometric error = angle(output direction, true direction from the helmet
                   centre); seam break = angle between the two cameras that meet at a seam line.
                   Scene = None -> points at infinity (pure calibration error, no parallax).
* image metrics  – stitched test room vs ground-truth panorama: PSNR, SSIM, seam artifacts (test programme).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from smart_helmet_pano.rig import Camera, Rig
from smart_helmet_pano.stitch import PanoStitcher, StitchOptions, equirect_dirs

ARCMIN = 180 * 60 / math.pi


def _rot(c):
    return np.asarray(c.R_rig_from_cam_ue, float)


def align(est: Rig, true: Rig) -> Rig:
    """Best global rotation (a common rotation of the whole rig is invisible in the seams; levelling is the
    IMU's job), applied to the estimated rig."""
    A = sum(_rot(t) @ _rot(e).T for e, t in zip(est.cameras, true.cameras))
    U, _, Vt = np.linalg.svd(A)
    G = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    cams = []
    for e in est.cameras:
        d = {k: getattr(e, k) for k in Camera.__dataclass_fields__}
        d["rotation_matrix"] = (G @ _rot(e)).tolist()
        d["position_m"] = (G @ np.asarray(e.position_m, float)).tolist()
        cams.append(Camera(**d))
    return Rig(est.name, cams)


def _ang(a, b):
    return np.arccos(np.clip(np.sum(a * b, -1), -1, 1))


THETA_BINS = [0, 15, 30, 45, 60, 75, 90]


def camera_errors(est: Rig, true: Rig, step=12, theta_max_deg=88.0):
    rows = []
    by_theta = []
    for e, t in zip(est.cameras, true.cameras):
        vv, uu = np.mgrid[0:t.height:step, 0:t.width:step].astype(np.float64)
        rt = t.unproject(uu, vv)
        rt_cam = rt @ t.R_cv_from_rig.T
        ok = np.degrees(np.arccos(np.clip(rt_cam[..., 2], -1, 1))) < theta_max_deg
        re = e.unproject(uu, vv)
        tot = _ang(re, rt)[ok] * ARCMIN
        lens = _ang(re @ e.R_cv_from_rig.T, rt_cam)[ok] * ARCMIN
        th = np.degrees(np.arccos(np.clip(rt_cam[..., 2], -1, 1)))[ok]
        by_theta.append([float(np.sqrt(np.mean(tot[(th >= a) & (th < b)] ** 2))) if ((th >= a) & (th < b)).any()
                         else np.nan for a, b in zip(THETA_BINS[:-1], THETA_BINS[1:])])
        dR = _rot(e).T @ _rot(t)
        rows.append({"name": t.name,
                     "rot_err_deg": round(math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1))), 3),
                     "pos_err_mm": round(1000 * float(np.linalg.norm(np.subtract(e.position_m, t.position_m))), 1),
                     "f_err_px": round(e.fx - t.fx, 2), "pp_err_px": round(float(np.hypot(e.cx - t.cx, e.cy - t.cy)), 2),
                     "ray_rms_arcmin": round(float(np.sqrt(np.mean(tot ** 2))), 2),
                     "ray_p95_arcmin": round(float(np.percentile(tot, 95)), 2),
                     "ray_max_arcmin": round(float(tot.max()), 2),
                     "lens_rms_arcmin": round(float(np.sqrt(np.mean(lens ** 2))), 2),
                     "ray_rms_px": round(float(np.sqrt(np.mean(tot ** 2))) / ARCMIN * t.fx, 3)})
    agg = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in
           ("rot_err_deg", "pos_err_mm", "ray_rms_arcmin", "ray_p95_arcmin", "lens_rms_arcmin", "ray_rms_px")}
    agg["ray_max_arcmin"] = round(float(max(r["ray_max_arcmin"] for r in rows)), 2)
    agg["rot_err_max_deg"] = round(float(max(r["rot_err_deg"] for r in rows)), 3)
    bt = np.nanmean(np.array(by_theta), 0)
    agg["ray_rms_by_theta_arcmin"] = {f"{a}-{b}": round(float(v), 2) for a, b, v in
                                      zip(THETA_BINS[:-1], THETA_BINS[1:], bt)}
    return agg, rows


def stitch_options(width, proxy_depth_m=3.0, flow=False):
    return StitchOptions(width=width, level="full", proxy_depth_m=proxy_depth_m, theta_max_deg=75,
                         theta_ramp_deg=75, flow=flow, exposure=True, max_flow_frac=0.006)


def true_depth(scene, t_w, W):
    """Distance from the helmet centre to the scene along every panorama direction (ideal depth mode)."""
    d = equirect_dirs(W, W // 2).astype(np.float64).reshape(-1, 3)
    X, _ = scene.hit_points(np.asarray(t_w, float), d)
    return np.linalg.norm(X - np.asarray(t_w, float), axis=1).reshape(W // 2, W).astype(np.float32)


def pano_distortion(est: Rig, true: Rig, R_w, t_w, scene=None, W=1024, proxy_depth_m=3.0, central_deg=45.0,
                    return_maps=False, depth="proxy"):
    """depth="proxy": sphere of proxy_depth_m (parallax errors included);  depth="true": every pixel projected at
    its true distance (ideal depth mode, only the calibration error is left)."""
    R_w, t_w = np.asarray(R_w, float), np.asarray(t_w, float)
    H = W // 2
    if scene is None:                      # points at infinity -> the stitcher must project at infinity too
        proxy_depth_m = float("inf")
    elif depth == "true":
        proxy_depth_m = true_depth(scene, t_w, W)
    st = PanoStitcher(est, stitch_options(W, proxy_depth_m))
    maps = st.build_maps(R_w.T)
    d_out = equirect_dirs(W, H).astype(np.float64)
    wts = np.stack([m[2] for m in maps])
    owner = np.where(wts.max(0) > 0, np.argmax(wts, 0), -1)
    E = np.zeros((len(maps), H, W, 3))
    for c, ((u, v, w), tc) in enumerate(zip(maps, true.cameras)):
        m = w > 0
        if not m.any():
            continue
        r = tc.unproject(u[m].astype(np.float64), v[m].astype(np.float64)) @ R_w.T
        if scene is None:
            e = r
        else:
            o = R_w @ np.asarray(tc.position_m) + t_w
            X, _ = scene.hit_points(o, r)
            e = X - t_w
            e /= np.linalg.norm(e, axis=1, keepdims=True)
        E[c][m] = e
    cov = owner >= 0
    e_own = np.zeros((H, W, 3))
    for c in range(len(maps)):
        e_own[owner == c] = E[c][owner == c]
    err = np.where(cov, _ang(e_own, d_out), np.nan) * ARCMIN
    # seam lines: owner changes between horizontal / vertical neighbours
    brk = np.full((H, W), np.nan)
    for axis in (0, 1):
        a_own = owner
        b_own = np.roll(owner, -1, axis=axis)
        seam = (a_own >= 0) & (b_own >= 0) & (a_own != b_own)
        if axis == 0:
            seam[-1] = False
        ys, xs = np.nonzero(seam)
        if not len(ys):
            continue
        ca, cb = a_own[ys, xs], b_own[ys, xs]
        # both cameras evaluated at the same pixel; if cb does not cover it, use its neighbour pixel
        ea = E[ca, ys, xs]
        eb = E[cb, ys, xs]
        okb = wts[cb, ys, xs] > 0
        y2, x2 = (ys + 1, xs) if axis == 0 else (ys, (xs + 1) % W)
        eb[~okb] = E[cb[~okb], y2[~okb], x2[~okb]]
        ea2 = np.where((wts[ca, y2, x2] > 0)[:, None], E[ca, y2, x2], ea)
        b = _ang(np.where(okb[:, None], ea, ea2), eb) * ARCMIN
        brk[ys, xs] = np.fmax(brk[ys, xs], b)
    lat = 90 - (np.arange(H) + 0.5) / H * 180
    cw = np.repeat(np.cos(np.radians(lat))[:, None], W, 1)
    central = np.repeat((np.abs(lat) <= central_deg)[:, None], W, 1)

    def stats(x, mask):
        m = mask & np.isfinite(x)
        v, w_ = x[m], cw[m]
        if not len(v):
            return {}
        o = np.argsort(v)
        cs = np.cumsum(w_[o]) / w_.sum()
        p95 = float(v[o][np.searchsorted(cs, 0.95)])
        return {"mean": round(float((v * w_).sum() / w_.sum()), 2), "p95": round(p95, 2), "max": round(float(v.max()), 1)}
    px4096 = 360 * 60 / 4096
    res = {"geom_err_arcmin": stats(err, cov), "geom_err_central_arcmin": stats(err, cov & central),
           "seam_break_arcmin": stats(brk, np.isfinite(brk)),
           "seam_break_central_arcmin": stats(brk, np.isfinite(brk) & central)}
    sb = res["seam_break_arcmin"]
    if sb:
        res["seam_break_px4096"] = {"mean": round(sb["mean"] / px4096, 2), "p95": round(sb["p95"] / px4096, 2)}
    if return_maps:
        return res, err, brk, owner
    return res


def ssim(a, b, mask):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    C1, C2 = 6.5025, 58.5225
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return float(m[mask].mean())


def image_metrics(est: Rig, imgs, R_w, gt, W, flow=False, proxy_depth_m=3.0):
    from smart_helmet_pano.quality import seam_artifacts
    st = PanoStitcher(est, stitch_options(W, proxy_depth_m, flow))
    pano = st.stitch(imgs, {"rig_to_world": np.asarray(R_w).tolist()})
    H = W // 2
    lat = 90 - (np.arange(H) + 0.5) / H * 180
    cover = np.sum(st.last_weights, axis=0) > 0
    mask = cover & np.repeat((lat > -55)[:, None], W, 1)
    a = pano.astype(np.float32); b = gt.astype(np.float32)
    g = (a[mask] * b[mask]).sum(0) / np.maximum((a[mask] ** 2).sum(0), 1)
    a = np.clip(a * g, 0, 255).astype(np.uint8)
    mse = float(((a.astype(np.float32) - b) ** 2)[mask].mean())
    q, _ = seam_artifacts(st.last_warped, st.last_weights, st.W, st.H)
    return pano, {"PSNR": round(10 * math.log10(255 ** 2 / max(mse, 1e-6)), 2), "SSIM": round(ssim(a, gt, mask), 4),
                  "seam_artifacts_%": q["artifact_%_of_central"]}
