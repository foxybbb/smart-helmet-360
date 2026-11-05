"""Calibrated-rig panorama stitcher (Street-View style) for the Smart Helmet.

Pipeline (see README for the rationale):
 1. LUT projection   – every equirect pixel -> ray -> point on a proxy sphere
                       (radius = proxy_depth) -> each camera's fisheye model.
                       Optional IMU levelling is folded into the same LUT, so
                       pixels are interpolated only once (report sec. 4.2.1).
 2. Exposure comp.   – OpenCV BlocksGainCompensator on the warped images.
 3. Flow seam fix    – DIS optical flow inside every ring-pair overlap; each
                       image is warped part-way towards its neighbour
                       (Google "Seamless Street View" / Surround360 idea).
 4. Seams + blending – weighted-Voronoi seams + MultiBandBlender (or feather).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .rig import Rig, quat_to_matrix, rot_ue


@dataclass
class StitchOptions:
    width: int = 4096
    proxy_depth_m: float = 4.0          # zero-parallax distance; float('inf') = pure rotation; or an (H, W) depth map
    level: str = "tilt"                  # none | tilt (pitch+roll only) | full (yaw too, north-up)
    exposure: bool = True
    flow: bool = True
    blend: str = "multiband"            # multiband | feather
    bands: int = 5
    feather_deg: float = 6.0             # blend ramp width at image borders
    max_flow_frac: float = 0.008         # flow clamp, fraction of pano width (~3 deg = max parallax at 1.5 m)
    tuning: str | None = None            # libcamera tuning json (ISP gamma + ALSC) for real cameras; None = sRGB
    devignette: bool = True              # complete the ALSC lens-shading correction (needs tuning)
    theta_max_deg: float | None = None   # angle-from-axis weighting: prefer pixels near each optical axis
    theta_ramp_deg: float = 20.0         #   (Street-View-like seams midway between cameras; avoids dark rims)
    gain_theta_deg: float = 55.0         # exposure/WB ratios only from the central part of both lenses
    flow_sigma_frac: float = 0.004       # flow smoothing, fraction of pano width
    nadir_fill: tuple = (40, 40, 40)


# ---------------------------------------------------------------- helpers
def equirect_dirs(w: int, h: int) -> np.ndarray:
    lon = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(h, dtype=np.float32) + 0.5) / h) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def leveling_matrix(meta: dict | None, mode: str, rig: Rig) -> np.ndarray:
    """Matrix M so that d_rig = M @ d_out (d_out = direction in the output panorama frame)."""
    if meta is None or mode == "none":
        return np.eye(3)
    R = None
    if "rig_to_world" in meta:
        R = np.array(meta["rig_to_world"], float)
    elif "imu" in meta and "quaternion" in meta["imu"]:
        R = quat_to_matrix(*meta["imu"]["quaternion"]) @ rot_ue(*rig.imu_from_rig_pyr_deg).T
    if R is None:
        return np.eye(3)
    if mode == "full":
        return R.T
    # keep heading: output frame = world frame rotated by the rig's yaw
    fwd = R[:, 0]
    yaw = math.atan2(fwd[1], fwd[0])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return R.T @ Rz


def _border_weight(cam, u, v, feather_px):
    du = np.minimum(u, cam.width - 1 - u)
    dv = np.minimum(v, cam.height - 1 - v)
    # radial limit for very wide lenses
    w = np.clip(np.minimum(du, dv) / feather_px, 0, 1)
    return w * w * (3 - 2 * w)  # smoothstep


# ---------------------------------------------------------------- core
class PanoStitcher:
    def __init__(self, rig: Rig, opts: StitchOptions = StitchOptions()):
        self.rig, self.o = rig, opts
        self.W, self.H = opts.width, opts.width // 2
        self._dirs = equirect_dirs(self.W, self.H)
        self.timings = {}

    def build_maps(self, M: np.ndarray):
        d = self._dirs @ M.T  # rig-frame rays
        depth = self.o.proxy_depth_m
        if np.ndim(depth):        # per-pixel depth map (H, W) along each panorama direction, metres
            P = d * np.asarray(depth, np.float32)[..., None]
        else:
            P = d * (1e6 if not np.isfinite(depth) else depth)
        maps = []
        self.thetas = []
        for cam in self.rig.cameras:
            u, v, valid = cam.project(P)
            pc = (P - np.asarray(cam.position_m)) @ cam.R_cv_from_rig.T
            theta = np.degrees(np.arctan2(np.hypot(pc[..., 0], pc[..., 1]), pc[..., 2])).astype(np.float32)
            fp = self.o.feather_deg * math.pi / 180 * cam.fx
            w = np.where(valid, _border_weight(cam, u, v, fp), 0)
            if self.o.theta_max_deg:
                t = np.clip((self.o.theta_max_deg - theta) / self.o.theta_ramp_deg, 0, 1)
                w = w * (t * t * (3 - 2 * t))
            maps.append((u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)))
            self.thetas.append(np.where(valid, theta, 180).astype(np.float32))
        return maps

    # ---- step 3: optical-flow seam alignment between ring neighbours
    def _flow_align(self, warped, weights):
        ring = [i for i, c in enumerate(self.rig.cameras) if not c.name.startswith("top")]
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        disp = [np.zeros((self.H, self.W, 2), np.float32) for _ in warped]
        for k, a in enumerate(ring):
            b = ring[(k + 1) % len(ring)]
            ov = (weights[a] > 0) & (weights[b] > 0)
            if ov.sum() < 500:
                continue
            ys, xs = np.nonzero(ov)
            # unwrap the seam around the +-180 deg longitude
            shift = 0
            if xs.max() - xs.min() > self.W / 2:
                shift = self.W // 2
                xs = (xs + shift) % self.W
            x0, x1 = max(xs.min() - 16, 0), min(xs.max() + 16, self.W)
            y0, y1 = max(ys.min() - 16, 0), min(ys.max() + 16, self.H)
            roll = lambda im: np.roll(im, shift, axis=1)[y0:y1, x0:x1]
            ga = cv2.cvtColor(roll(warped[a]), cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(roll(warped[b]), cv2.COLOR_BGR2GRAY)
            m = roll(ov.astype(np.uint8))
            f_ab = dis.calc(ga, gb, None)
            f_ba = dis.calc(gb, ga, None)
            conf = self._flow_confidence(ga, gb, f_ab, f_ba) * m
            max_flow = self.o.max_flow_frac * self.W
            for f in (f_ab, f_ba):
                # normalized convolution: trust only textured, unsaturated, fwd/bwd-consistent pixels
                sig = self.o.flow_sigma_frac * self.W
                num = cv2.GaussianBlur(f * conf[..., None], (0, 0), sig)
                den = cv2.GaussianBlur(conf, (0, 0), sig)[..., None]
                f[:] = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0) * np.minimum(den / 0.2, 1)
                mag = np.linalg.norm(f, axis=2, keepdims=True)
                f *= np.minimum(1.0, max_flow / np.maximum(mag, 1e-6))
                f *= m[..., None]
            wa, wb = roll(weights[a]), roll(weights[b])
            t = (wb / np.maximum(wa + wb, 1e-6)) * m  # 0 on a's side -> 1 on b's side
            # a(x) ~ b(x + f_ab)  =>  a warped towards b:  a'(x) = a(x - t * f_ab)
            for idx, f, tt in ((a, f_ab, t), (b, f_ba, (1 - t) * m)):
                full = np.zeros((self.H, self.W, 2), np.float32)
                full[y0:y1, x0:x1] = -f * tt[..., None]
                disp[idx] += np.roll(full, -shift, axis=1)
        out = []
        gx, gy = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        for img, dd in zip(warped, disp):
            if not dd.any():
                out.append(img)
                continue
            out.append(cv2.remap(img, gx + dd[..., 0], gy + dd[..., 1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP))
        return out

    @staticmethod
    def _flow_confidence(ga, gb, f_ab, f_ba):
        h, w = ga.shape
        gx = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2) > 12
        unsat = (ga < 245) & (gb < 245) & (ga > 8)
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        back = cv2.remap(f_ba, xx + f_ab[..., 0], yy + f_ab[..., 1], cv2.INTER_LINEAR)
        consistent = np.linalg.norm(f_ab + back, axis=2) < 2.0
        return (texture & unsat & consistent).astype(np.float32)

    def stitch(self, images: list, meta: dict | None = None):
        t0 = time.perf_counter()
        M = leveling_matrix(meta, self.o.level, self.rig)
        maps = self.build_maps(M)
        self.timings["maps"] = time.perf_counter() - t0

        from .photometric import load_tuning
        tun = load_tuning(self.o.tuning)
        warped, weights = [], []
        for img, (mx, my, w) in zip(images, maps):
            lin = tun.linear(img)                                  # float32 linear light
            if self.o.devignette and self.o.tuning:
                h_, w_ = img.shape[:2]
                rot = cv2.ROTATE_90_COUNTERCLOCKWISE if h_ > w_ else None
                vg = tun.residual_vignette(w_, h_, rotate=rot)
                if vg is not None:
                    lin = lin * vg[..., None]
            warped.append(cv2.remap(lin, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
            weights.append(w)
        self.timings["warp"] = time.perf_counter() - t0

        if self.o.exposure:
            warped = self._compensate_linear(warped, weights)
        warped = [tun.display(np.clip(x, 0, 1)) for x in warped]
        self.timings["exposure"] = time.perf_counter() - t0
        if self.o.flow:
            warped = self._flow_align(warped, weights)
        self.timings["flow"] = time.perf_counter() - t0

        pano = self._blend(warped, weights)
        cover = np.sum(weights, axis=0) > 0
        pano[~cover] = self.o.nadir_fill
        self.timings["total"] = time.perf_counter() - t0
        self.last_warped, self.last_weights = warped, weights
        return pano

    def _compensate_linear(self, warped, weights):
        """Per-camera, per-channel gains in linear light (exposure, ISO and white balance are all
        multiplicative there). Robust: median log-ratio over every overlap, least squares with
        sum(log g) = 0, so the panorama keeps the average exposure."""
        n = len(warped)
        s = 4
        small = [cv2.resize(w, (self.W // s, self.H // s), interpolation=cv2.INTER_AREA) for w in warped]
        gt = self.o.gain_theta_deg
        msk = [cv2.resize(((w > 0.05) & (th < gt)).astype(np.uint8), (self.W // s, self.H // s),
                          interpolation=cv2.INTER_NEAREST) > 0 for w, th in zip(weights, self.thetas)]
        gains = np.ones((n, 3))
        for ch in range(3):
            rows, rhs, wts = [], [], []
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = small[i][..., ch], small[j][..., ch]
                    ov = msk[i] & msk[j] & (a > 0.01) & (b > 0.01) & (a < 0.95) & (b < 0.95)
                    N = int(ov.sum())
                    if N < 200:
                        continue
                    r = float(np.median(np.log(b[ov]) - np.log(a[ov])))   # g_i / g_j = I_j / I_i
                    row = np.zeros(n); row[i], row[j] = 1, -1
                    rows.append(row); rhs.append(r); wts.append(np.sqrt(N))
            if not rows:
                continue
            A = np.array(rows) * np.array(wts)[:, None]
            y = np.array(rhs) * np.array(wts)
            A = np.vstack([A, np.ones(n) * max(wts) * 10]); y = np.append(y, 0.0)
            gains[:, ch] = np.exp(np.linalg.lstsq(A, y, rcond=None)[0])
        self.gains = gains
        return [w * g[None, None, :].astype(np.float32) for w, g in zip(warped, gains)]

    def _compensate(self, warped, weights, sigma_n=10.0, sigma_g=0.1):
        """Per-camera, per-channel gain (Brown & Lowe / OpenCV GainCompensator objective),
        solved on 1/4-scale images over all overlap zones."""
        n = len(warped)
        s = 4
        small = [cv2.resize(w, (self.W // s, self.H // s), interpolation=cv2.INTER_AREA).astype(np.float64)
                 for w in warped]
        msk = [cv2.resize((w > 0.05).astype(np.uint8), (self.W // s, self.H // s),
                          interpolation=cv2.INTER_NEAREST) > 0 for w in weights]
        gains = np.ones((n, 3))
        for ch in range(3):
            A = np.zeros((n, n))
            b = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ov = msk[i] & msk[j]
                    N = ov.sum()
                    if N < 50:
                        continue
                    Ii, Ij = small[i][..., ch][ov].mean(), small[j][..., ch][ov].mean()
                    A[i, i] += N * (Ii * Ii / sigma_n ** 2 + 1 / sigma_g ** 2)
                    A[i, j] -= N * Ii * Ij / sigma_n ** 2
                    b[i] += N / sigma_g ** 2
            ok = np.diag(A) > 0
            if ok.any():
                gains[ok, ch] = np.linalg.solve(A[np.ix_(ok, ok)], b[ok])
        self.gains = gains
        return [np.clip(img.astype(np.float32) * g[None, None, :].astype(np.float32), 0, 255).astype(np.uint8)
                for img, g in zip(warped, gains)]

    def _blend(self, warped, weights):
        W = np.stack(weights)
        if self.o.blend == "feather":
            s = W.sum(0, keepdims=True)
            Wn = np.where(s > 0, W / np.maximum(s, 1e-6), 0)
            acc = np.zeros((self.H, self.W, 3), np.float32)
            for img, w in zip(warped, Wn):
                acc += img.astype(np.float32) * w[..., None]
            return np.clip(acc, 0, 255).astype(np.uint8)
        # seams: each pixel goes to the camera with the highest weight (weighted Voronoi)
        owner = np.argmax(W, axis=0)
        blender = cv2.detail_MultiBandBlender(0, self.o.bands)
        blender.prepare((0, 0, self.W, self.H))
        for i, img in enumerate(warped):
            m = ((owner == i) & (W[i] > 0)).astype(np.uint8) * 255
            if m.any():
                blender.feed(img.astype(np.int16), m, (0, 0))
        res, _ = blender.blend(None, None)
        return np.clip(res, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- frame IO
def load_frame(frame_dir: Path, rig: Rig):
    frame_dir = Path(frame_dir)
    imgs = []
    for cam in rig.cameras:
        cands = [frame_dir / f"{cam.name}{ext}" for ext in (".png", ".jpg", ".jpeg")]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            raise FileNotFoundError(f"{cam.name} image missing in {frame_dir}")
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im.shape[1] != cam.width or im.shape[0] != cam.height:
            raise ValueError(f"{p.name}: {im.shape[1]}x{im.shape[0]} != rig {cam.width}x{cam.height}")
        imgs.append(im)
    meta_p = frame_dir / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else None
    return imgs, meta


def find_rig(frame_dir: Path) -> Path:
    for p in [Path(frame_dir), *Path(frame_dir).parents][:4]:
        if (p / "rig.json").exists():
            return p / "rig.json"
    raise FileNotFoundError("rig.json not found next to the frame or in its parent folders")


def stitch_frame(frame_dir, out_path=None, opts: StitchOptions = StitchOptions(), rig_path=None):
    frame_dir = Path(frame_dir)
    rig = Rig.load(rig_path or find_rig(frame_dir))
    imgs, meta = load_frame(frame_dir, rig)
    st = PanoStitcher(rig, opts)
    pano = st.stitch(imgs, meta)
    out_path = Path(out_path or frame_dir / "panorama.jpg")
    cv2.imwrite(str(out_path), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path, pano, st
