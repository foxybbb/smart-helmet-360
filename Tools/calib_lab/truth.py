"""Rig layouts with different camera overlap, and "true" cameras with the faults of real hardware.

The nominal rig is what the CAD drawing and the lens datasheet say: equidistant fisheye, centred principal
point, exact mounting angles.  The true rig, drawn from a seed, is what was actually assembled:

* lens design differs from the ideal equidistant curve (k_base), every lens has its own k1..k4 spread;
* focal length spread, principal point offset (M12 holder tolerance), slight pixel aspect;
* decentering (tangential) distortion that the Kannala-Brandt model used by the pipeline cannot express;
* mounting error of every camera: rotation (deg) and lens-centre offset (mm);
* photometry: exposure / white balance gain, vignetting, sensor noise, lens blur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from smart_helmet_pano.rig import Camera, Rig, rot_ue, coverage_report


# ---------------------------------------------------------------- true camera model
@dataclass
class TrueCamera(Camera):
    """Kannala-Brandt + decentering distortion (p1, p2 act on the distorted normalised coordinates)."""
    p: list = field(default_factory=lambda: [0.0, 0.0])
    gain_bgr: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    vignette: float = 0.0          # cos^4 fall-off strength (0 = none, 1 = full cos^4)
    noise: float = 2.0
    blur: float = 0.7              # lens MTF, Gaussian sigma in px

    def project(self, P_rig):
        p = (P_rig - np.asarray(self.position_m)) @ self.R_cv_from_rig.T
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        rxy = np.sqrt(x * x + y * y)
        theta = np.arctan2(rxy, z)
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        td = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        s = np.where(rxy > 1e-12, td / np.maximum(rxy, 1e-12), 0.0)
        xd, yd = x * s, y * s
        p1, p2 = self.p
        r2 = xd * xd + yd * yd
        xt = xd + 2 * p1 * xd * yd + p2 * (r2 + 2 * xd * xd)
        yt = yd + p1 * (r2 + 2 * yd * yd) + 2 * p2 * xd * yd
        u = self.fx * xt + self.cx
        v = self.fy * yt + self.cy
        valid = (theta < math.radians(self.max_theta_deg)) & (u >= 0) & (u <= self.width - 1) \
            & (v >= 0) & (v <= self.height - 1)
        return u, v, valid

    def unproject(self, u, v):
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        p1, p2 = self.p
        xd, yd = mx.copy(), my.copy()
        if p1 or p2:
            for _ in range(6):
                r2 = xd * xd + yd * yd
                xd = mx - (2 * p1 * xd * yd + p2 * (r2 + 2 * xd * xd))
                yd = my - (p1 * (r2 + 2 * yd * yd) + 2 * p2 * xd * yd)
        td = np.sqrt(xd * xd + yd * yd)
        th = td.copy()
        k1, k2, k3, k4 = self.k
        for _ in range(12):
            t2 = th * th
            f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - td
            df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + 9 * k4 * t2)))
            th = np.clip(th - f / np.maximum(df, 0.2), 0, 3.0)
        s = np.where(td > 1e-12, np.sin(th) / np.maximum(td, 1e-12), 0.0)
        ray_cv = np.stack([xd * s, yd * s, np.cos(th)], axis=-1)
        return ray_cv @ self.R_cv_from_rig

    def as_camera(self) -> Camera:
        d = {k: v for k, v in asdict(self).items() if k in Camera.__dataclass_fields__}
        return Camera(**d)


# ---------------------------------------------------------------- rig layouts (overlap variants)
@dataclass
class RigSpec:
    key: str
    title: str
    ring: int = 7
    vfov_deg: float = 141.4        # lens FOV along the long (vertical) side of the portrait sensor
    pitch_deg: float = 9.3
    radius_m: float = 0.105
    top_height_m: float = 0.10
    width: int = 1080
    height: int = 1920


VARIANTS = {
    "6x141": RigSpec("6x141", "6 камер × 141°", ring=6, vfov_deg=141.4),
    "7x141": RigSpec("7x141", "7 камер × 141° (схема)", ring=7, vfov_deg=141.4),
    "8x141": RigSpec("8x141", "8 камер × 141°", ring=8, vfov_deg=141.4),
    "6x160": RigSpec("6x160", "6 камер × 160°", ring=6, vfov_deg=160.0),
    "7x160": RigSpec("7x160", "7 камер × 160°", ring=7, vfov_deg=160.0),
}


def nominal_rig(spec: RigSpec) -> Rig:
    f = (spec.height / 2) / math.radians(spec.vfov_deg / 2)
    W, H = spec.width, spec.height
    cams = []
    for i in range(spec.ring):
        yaw = i * 360.0 / spec.ring
        pos = [spec.radius_m * math.cos(math.radians(yaw)), spec.radius_m * math.sin(math.radians(yaw)), 0.0]
        cams.append(Camera(f"cam{i}", W, H, f, f, (W - 1) / 2, (H - 1) / 2, position_m=pos,
                           rotation_pyr_deg=[spec.pitch_deg, yaw, 0.0], max_theta_deg=95.0))
    cams.append(Camera(f"top{spec.ring}", W, H, f, f, (W - 1) / 2, (H - 1) / 2,
                       position_m=[0.0, 0.0, spec.top_height_m], rotation_pyr_deg=[90.0, 0.0, 0.0],
                       max_theta_deg=95.0))
    for c in cams:
        c.rotation_matrix = rot_ue(*c.rotation_pyr_deg).tolist()
    return Rig(f"helmet_{spec.key}", cams)


def overlap_stats(rig: Rig, w: int = 1440) -> dict:
    """Overlap of neighbouring ring cameras at the horizon, ring-top overlap, sphere coverage."""
    rep = coverage_report(rig, w, w // 2)
    cnt = rep["count_map"]
    h = w // 2
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    eq = cnt[np.argmin(np.abs(lat))]
    ring_n = len(rig.ring)
    horizon_overlap = float((eq >= 2).sum() / w * 360 / ring_n)
    # ring-top: latitude band seen by both the top camera and some ring camera, along longitude 0
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    top = rig.cameras[-1]
    band = []
    for j in range(0, w, w // 36):
        d = np.stack([np.cos(np.radians(lat)) * np.cos(lon[j]), np.cos(np.radians(lat)) * np.sin(lon[j]),
                      np.sin(np.radians(lat))], -1) * 1e4
        seen_top = top.project(d)[2]
        seen_ring = np.zeros(h, bool)
        for c in rig.ring:
            seen_ring |= c.project(d)[2]
        both = seen_top & seen_ring
        band.append(float(both.sum() * 180 / h))
    f = rig.cameras[0].fx
    return {"cameras": len(rig.cameras), "ring": ring_n,
            "hfov_deg": round(math.degrees(rig.cameras[0].width / f), 1),
            "vfov_deg": round(math.degrees(rig.cameras[0].height / f), 1),
            "px_per_deg": round(f * math.pi / 180, 2),
            "horizon_overlap_deg": round(horizon_overlap, 1),
            "ring_top_overlap_deg": round(float(np.mean(band)), 1),
            "covered_%": round(rep["covered_%"], 1), "double_covered_%": round(rep["overlap_%"], 1),
            "lowest_lat_deg": round(rep["lowest_covered_lat_deg"], 1)}


# ---------------------------------------------------------------- faults
@dataclass
class FaultSpec:
    mount_rot_std_deg: float = 3.0
    mount_pos_std_mm: float = 2.0
    f_std_rel: float = 0.015
    pp_std_px: float = 15.0
    aspect_std: float = 0.001
    k_base: tuple = (-0.035, 0.004, -0.0005, 0.0)      # lens design: between equidistant and equisolid
    k_std: tuple = (0.010, 0.004, 0.0015, 0.0005)
    tangential_std: float = 4e-4
    gain_std_stops: float = 0.4
    wb_std: float = 0.06
    vignette: tuple = (0.3, 0.6)
    noise: float = 2.0
    blur: float = 0.7


def _theta_d(theta, k):
    t2 = theta * theta
    return theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def sample_truth(nominal: Rig, faults: FaultSpec = FaultSpec(), seed: int = 1) -> Rig:
    """Draw the as-built rig. The lens focal is chosen so that the *datasheet FOV* (long side) still holds
    for the design curve k_base, i.e. the nominal equidistant model is right at the edge, wrong inside."""
    rng = np.random.default_rng(seed)
    cams = []
    for c in nominal.cameras:
        half = (c.height / 2) / c.fx                          # nominal half-FOV (rad) of the long side
        f_design = (c.height / 2) / _theta_d(half, faults.k_base)
        f = f_design * (1 + rng.normal(0, faults.f_std_rel))
        fy = f * (1 + rng.normal(0, faults.aspect_std))
        k = [float(b + rng.normal(0, s)) for b, s in zip(faults.k_base, faults.k_std)]
        p = [float(v) for v in rng.normal(0, faults.tangential_std, 2)]
        cx = (c.width - 1) / 2 + rng.normal(0, faults.pp_std_px)
        cy = (c.height - 1) / 2 + rng.normal(0, faults.pp_std_px)
        err = rng.normal(0, faults.mount_rot_std_deg, 3)
        R = np.asarray(c.R_rig_from_cam_ue) @ rot_ue(*err)
        pos = np.asarray(c.position_m) + rng.normal(0, faults.mount_pos_std_mm / 1000, 3)
        g = 2 ** rng.normal(0, faults.gain_std_stops)
        wb = 1 + rng.normal(0, faults.wb_std, 3)
        wb /= wb[1]
        cams.append(TrueCamera(c.name, c.width, c.height, float(f), float(fy), float(cx), float(cy), k=k,
                               position_m=pos.tolist(), rotation_pyr_deg=list(c.rotation_pyr_deg),
                               max_theta_deg=95.0, rotation_matrix=R.tolist(), p=p,
                               gain_bgr=(g * wb).tolist(), vignette=float(rng.uniform(*faults.vignette)),
                               noise=faults.noise, blur=faults.blur))
    return Rig(nominal.name + f"_true_s{seed}", cams)


def describe_truth(true: Rig, nominal: Rig) -> list:
    rows = []
    for t, n in zip(true.cameras, nominal.cameras):
        dR = np.asarray(n.R_rig_from_cam_ue).T @ np.asarray(t.R_rig_from_cam_ue)
        ang = math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        rows.append({"name": t.name, "mount_err_deg": round(ang, 2),
                     "pos_err_mm": round(1000 * float(np.linalg.norm(np.subtract(t.position_m, n.position_m))), 1),
                     "f": round(t.fx, 1), "f_nominal": round(n.fx, 1),
                     "pp_offset_px": [round(t.cx - n.cx, 1), round(t.cy - n.cy, 1)],
                     "k": [round(v, 4) for v in t.k], "p": [round(v, 5) for v in t.p]})
    return rows
