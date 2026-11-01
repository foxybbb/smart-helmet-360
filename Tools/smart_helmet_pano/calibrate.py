"""Self-calibration of the real helmet from its own images – no checkerboard needed.

Same idea as PTGui / Hugin "optimise lens + positions", and as Google re-calibrating
the Street View rosette from imagery:

 1. SIFT features (after CLAHE, so different exposure / white balance still match).
 2. Every camera pair is matched and verified with RANSAC on *rays*
    (rotation-only model, Kabsch) using the nominal fisheye lens.
 3. Initial rotations: maximum-inlier spanning tree from cam0.
 4. Bundle adjustment (scipy least_squares, Huber loss) of shared focal f,
    shared Kannala-Brandt k1..k4 and per-camera rotations.
 5. Top camera: its overlap with the ring is at the extreme lens edge, so it is
    matched in a common "zenith" (azimuthal) view instead of raw images.
 6. Levelling without IMU: the ring optical axes form a cone around the helmet's
    vertical axis -> its normal is "up".
Output: calibrated rig.json + report (lens FOV, residuals, each camera's mounting error).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from .rig import Camera, Rig, default_rig, rot_ue, RING_RADIUS_M, TOP_HEIGHT_M

_CV2UE = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], float)  # v_ue = M @ v_cv
# UE is left-handed; internally we use a right-handed frame (X fwd, Y LEFT, Z up)
_FLIP = np.diag([1.0, -1.0, 1.0])


# ------------------------------------------------------------------ features
def _prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)


def _features(img, scale=0.6, mask=None):
    g = _prep(img)
    if scale != 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015)
    kp, des = sift.detectAndCompute(g, mask)
    pts = np.array([k.pt for k in kp], np.float64).reshape(-1, 2) / scale
    return pts, des


def _match(d1, d2, ratio=0.8):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m12 = bf.knnMatch(d1, d2, k=2)
    m21 = bf.knnMatch(d2, d1, k=2)
    best21 = {m[0].queryIdx: m[0].trainIdx for m in m21 if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    out = [(m[0].queryIdx, m[0].trainIdx) for m in m12
           if len(m) == 2 and m[0].distance < ratio * m[1].distance and best21.get(m[0].trainIdx) == m[0].queryIdx]
    return np.array(out, int).reshape(-1, 2)


def _kabsch(a, b):
    """R minimising |R a - b| for unit rays (N,3)."""
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def _ransac_rotation(ra, rb, thresh_deg=3.0, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ra)
    best, best_in = None, np.zeros(n, bool)
    ct = math.cos(math.radians(thresh_deg))
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R = _kabsch(ra[idx], rb[idx])
        inl = np.sum((ra @ R.T) * rb, 1) > ct
        if inl.sum() > best_in.sum():
            best, best_in = R, inl
    if best is not None and best_in.sum() >= 3:
        best = _kabsch(ra[best_in], rb[best_in])
        best_in = np.sum((ra @ best.T) * rb, 1) > ct
    return best, best_in


# ------------------------------------------------------------------ lens model (OpenCV camera frame)
def _unproject(u, v, f, cx, cy, k):
    mx, my = (u - cx) / f, (v - cy) / f
    td = np.sqrt(mx * mx + my * my)
    th = td.copy()
    k1, k2, k3, k4 = k
    for _ in range(8):
        t2 = th * th
        fn = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - td
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + 9 * k4 * t2)))
        th = np.clip(th - fn / np.maximum(df, 0.1), 0, 3.0)
    s = np.where(td > 1e-12, np.sin(th) / np.maximum(td, 1e-12), 0.0)
    return np.stack([mx * s, my * s, np.cos(th)], -1)


def _project(rays_cv, f, cx, cy, k):
    x, y, z = rays_cv[..., 0], rays_cv[..., 1], rays_cv[..., 2]
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(r, z)
    t2 = th * th
    td = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
    s = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 0)
    return f * x * s + cx, f * y * s + cy, th


def _theta_of_radius(r_px, f, k):
    td = r_px / f
    th = td
    for _ in range(30):
        t2 = th * th
        fn = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - td
        df = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + 9 * k[3] * t2)))
        th -= fn / df
    return th


def _pyr(R):
    """UE FRotator (pitch, yaw, roll) of a rig<-cam matrix (columns fwd, right, up)."""
    fwd, right, up = R[:, 0], R[:, 1], R[:, 2]
    pitch = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    R0 = rot_ue(pitch, yaw, 0)
    roll = math.degrees(math.atan2(R0[:, 2] @ right, R0[:, 2] @ up))
    return pitch, yaw, roll


# ------------------------------------------------------------------ calibration
class _Cal:
    def __init__(self, frames, names, base, f_init):
        self.frames, self.names = frames, names          # frames: list of [img per camera]
        self.images = frames[0]
        self.n = len(names)
        self.sizes = [(im.shape[1], im.shape[0]) for im in frames[0]]
        self.base = base
        self.f, self.k = f_init, np.zeros(4)
        self.f_ref = f_init               # fixed residual scale (residual * f would let f collapse to 0)
        self.R = [None] * self.n          # rig(UE-ish frame) <- camera(OpenCV)
        self.obs = []                     # (i, j, pts_i, pts_j)

    def c(self, i):
        w, h = self.sizes[i]
        return (w - 1) / 2, (h - 1) / 2

    def rays(self, i, pts, f=None, k=None):
        """f / k may be shared (scalar, (4,)) or per camera ((n,), (n,4))."""
        cx, cy = self.c(i)
        f = self.f if f is None else f
        k = self.k if k is None else k
        fi = f[i] if np.ndim(f) == 1 else f
        ki = k[i] if np.ndim(k) == 2 else k
        return _unproject(pts[:, 0], pts[:, 1], fi, cx, cy, ki)

    # ---- step 2/3
    def match_pairs(self, verbose, max_per_pair=400):
        """Match every camera pair in every frame; the rig is rigid, so all frames share rotations."""
        pairs = {}
        for imgs in self.frames:
            feats = [_features(im) for im in imgs]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    m = _match(feats[i][1], feats[j][1])
                    if len(m) < 12:
                        continue
                    R, inl = _ransac_rotation(self.rays(i, feats[i][0][m[:, 0]]), self.rays(j, feats[j][0][m[:, 1]]))
                    if inl.sum() < 12:
                        continue
                    pa, pb = feats[i][0][m[inl, 0]], feats[j][0][m[inl, 1]]
                    if (i, j) in pairs:
                        R0, qa, qb, best = pairs[(i, j)]
                        if math.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R).as_rotvec())) > 4:
                            continue  # inconsistent with the rig geometry seen in other frames -> false match
                        pairs[(i, j)] = (R0 if best >= len(pa) else R, np.vstack([qa, pa]), np.vstack([qb, pb]), max(best, len(pa)))
                    else:
                        pairs[(i, j)] = (R, pa, pb, len(pa))
        rng = np.random.default_rng(0)
        for key, (R, pa, pb, best) in list(pairs.items()):
            if len(pa) < 25:
                del pairs[key]; continue
            if len(pa) > max_per_pair:
                sel = rng.choice(len(pa), max_per_pair, replace=False)
                pa, pb = pa[sel], pb[sel]
            pairs[key] = (R, pa, pb)
        if verbose:
            print("verified pairs:", {f"{self.names[a]}-{self.names[b]}": len(v[1]) for (a, b), v in pairs.items()})
        self.R[0] = _FLIP @ self.base.cameras[0].R_rig_from_cam_ue @ _CV2UE
        edges = sorted(pairs.items(), key=lambda kv: -len(kv[1][1]))
        changed = True
        while changed:
            changed = False
            for (i, j), (Rij, _, _) in edges:   # x_j = Rij x_i  ->  R_j = R_i Rij^T
                if self.R[i] is not None and self.R[j] is None:
                    self.R[j] = self.R[i] @ Rij.T; changed = True
                elif self.R[j] is not None and self.R[i] is None:
                    self.R[i] = self.R[j] @ Rij; changed = True
        self.obs = [(i, j, pa, pb) for (i, j), (_, pa, pb) in pairs.items()
                    if self.R[i] is not None and self.R[j] is not None]

    # ---- step 4
    def bundle_adjust(self, fix_lens=False, passes=2, verbose=True, per_camera_lens=False,
                      w_dlogf=200.0, w_dk=15.0):
        """per_camera_lens: every camera gets its own f and k1..k4 (lens-to-lens spread), softly tied to
        the shared mean (w_dlogf, w_dk are prior weights in residual pixels per unit)."""
        active = [c for c in range(self.n) if self.R[c] is not None]
        free = [c for c in active if c != 0]
        n = self.n
        f0 = float(np.exp(np.mean(np.log(self.f)))) if np.ndim(self.f) else self.f
        k0 = np.mean(self.k, 0) if np.ndim(self.k) == 2 else np.asarray(self.k)
        nl = 5 + (5 * n if per_camera_lens else 0)

        def unpack(x):
            if fix_lens:
                f, k, o = self.f, self.k, 0
            elif per_camera_lens:
                lf, kk = x[0], x[1:5]
                dlf, dk = x[5:5 + n], x[5 + n:5 + 5 * n].reshape(n, 4)
                f, k, o = np.exp(lf + dlf), kk[None, :] + dk, nl
            else:
                f, k, o = math.exp(x[0]), x[1:5], 5
            Rs = list(self.R)
            for t, c in enumerate(free):
                Rs[c] = Rot.from_rotvec(x[o + 3 * t: o + 3 * t + 3]).as_matrix()
            return f, k, Rs

        def resid(x):
            f, k, Rs = unpack(x)
            out = [r.ravel() for r in self._pair_residuals(f, k, Rs)]
            if not fix_lens:
                out.append(np.asarray(x[1:5]) * 20.0)
                if per_camera_lens:
                    out.append(x[5:5 + n] * w_dlogf)
                    out.append(x[5 + n:5 + 5 * n] * w_dk)
            return np.concatenate(out)

        if fix_lens:
            lens = []
        elif per_camera_lens:
            dlf0 = (np.log(self.f) - math.log(f0)) if np.ndim(self.f) else np.zeros(n)
            dk0 = (self.k - k0).ravel() if np.ndim(self.k) == 2 else np.zeros(4 * n)
            lens = [[math.log(f0)], k0, dlf0, dk0]
        else:
            lens = [[math.log(f0)], k0]
        x = np.concatenate(lens + [Rot.from_matrix(self.R[c]).as_rotvec() for c in free])
        e0 = self.errors(*unpack(x))
        for p in range(passes):
            sol = least_squares(resid, x, loss="huber", f_scale=2.0, x_scale="jac", max_nfev=400)
            x = sol.x
            f, k, Rs = unpack(x)
            if p < passes - 1:  # drop gross outliers
                keep_obs = []
                for (i, j, pa, pb), e in zip(self.obs, self.errors(f, k, Rs, per_obs=True)):
                    keep = e < max(6.0, 3 * np.median(e))
                    if keep.sum() >= 6:
                        keep_obs.append((i, j, pa[keep], pb[keep]))
                self.obs = keep_obs
        self.f, self.k, self.R = f, np.asarray(k), Rs
        e1 = self.errors(f, k, Rs)
        if verbose:
            print(f"BA: rms {np.sqrt(np.mean(e0 ** 2)):.2f}px -> {np.sqrt(np.mean(e1 ** 2)):.2f}px, f={np.round(f, 1)}, k={np.round(k, 4)}")
        return e0, e1

    def centers(self, Rs):
        """Lens centres (m) in the internal frame: ring on a circle along each camera's heading, top above."""
        out = []
        for c, R in enumerate(Rs):
            if R is None:
                out.append(np.zeros(3)); continue
            if self.names[c].startswith("top"):
                out.append(np.array([0, 0, TOP_HEIGHT_M])); continue
            fw = R @ np.array([0, 0, 1.0])
            h = np.array([fw[0], fw[1], 0.0])
            out.append(RING_RADIUS_M * h / max(np.linalg.norm(h), 1e-9))
        return out

    def _pair_residuals(self, f, k, Rs):
        """Angular residuals (in nominal pixels) after triangulating every match from the real lens centres,
        so near objects (parallax between lenses ~10 cm apart) do not bias the lens / rotation estimate."""
        C = self.centers(Rs)
        out = []
        for i, j, pa, pb in self.obs:
            a = self.rays(i, pa, f, k) @ Rs[i].T
            b = self.rays(j, pb, f, k) @ Rs[j].T
            w0 = C[i] - C[j]
            B = np.sum(a * b, 1)
            D, E = a @ w0, b @ w0
            den = np.maximum(1 - B * B, 1e-9)
            s = np.clip((B * E - D) / den, 0.3, 1000)
            t = np.clip((E - B * D) / den, 0.3, 1000)
            X = 0.5 * (C[i] + s[:, None] * a + C[j] + t[:, None] * b)
            da = X - C[i]; da /= np.linalg.norm(da, axis=1, keepdims=True)
            db = X - C[j]; db /= np.linalg.norm(db, axis=1, keepdims=True)
            out.append(np.concatenate([a - da, b - db], 1) * self.f_ref)
        return out

    def errors(self, f, k, Rs, per_obs=False):
        out = [np.linalg.norm(r.reshape(-1, 2, 3), axis=2).mean(1) for r in self._pair_residuals(f, k, Rs)]
        return out if per_obs else (np.concatenate(out) if out else np.zeros(1))

    # ---- step 5: zenith view for the top camera
    def attach_top(self, top, verbose=True, size=1400, max_zenith_deg=55.0):
        ring = [c for c in range(self.n) if self.R[c] is not None and c != top]
        # current "up" estimate from the ring cone
        up = self.up_vector(ring)
        ex = np.cross(up, [1.0, 0, 0]); ex /= np.linalg.norm(ex)
        ey = np.cross(up, ex)
        # azimuthal equidistant grid around the zenith
        rmax = math.radians(max_zenith_deg)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        px, py = (xx - size / 2) / (size / 2) * rmax, (yy - size / 2) / (size / 2) * rmax
        th = np.hypot(px, py)
        ph = np.arctan2(py, px)
        D = (np.cos(th)[..., None] * up + (np.sin(th) * np.cos(ph))[..., None] * ex
             + (np.sin(th) * np.sin(ph))[..., None] * ey)
        inside = th <= rmax

        def render(c, R):
            rc = D @ R            # rig -> camera(cv): R^T d  (row-wise: d @ R)
            cx, cy = self.c(c)
            u, v, t = _project(rc, self.f, cx, cy, self.k)
            w, h = self.sizes[c]
            ok = inside & (t < math.radians(88)) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            img = cv2.remap(self.images[c], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
            img[~ok] = 0
            return img, ok, u, v

        comp = np.zeros((size, size, 3), np.uint8)
        owner = -np.ones((size, size), int)
        maps = {}
        for c in ring:
            img, ok, u, v = render(c, self.R[c])
            maps[c] = (u, v)
            sel = ok & (owner < 0)
            comp[sel] = img[sel]
            owner[sel] = c
        # top camera with a nominal "looking up" rotation; its roll about the zenith is unknown.
        cam_up = np.stack([ex, ey, up], 1)       # columns: cv x, cv y, cv z(optical axis = up)
        top_img, top_ok, tu, tv = render(top, cam_up)
        ctr = (size / 2 - 0.5, size / 2 - 0.5)

        def grad(img):
            g = _prep(img).astype(np.float32)
            return cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 3)

        gc, gt = grad(comp), grad(top_img)
        mc = (owner >= 0).astype(np.float32)

        def score(ang):
            M = cv2.getRotationMatrix2D(ctr, ang, 1.0)
            g = cv2.warpAffine(gt, M, (size, size))
            m = cv2.warpAffine(top_ok.astype(np.float32), M, (size, size)) * mc > 0.5
            if m.sum() < 1000:
                return -1
            a, b = gc[m] - gc[m].mean(), g[m] - g[m].mean()
            return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

        coarse = [(score(a), a) for a in np.arange(0, 360, 3.0)]
        best = max(coarse)[1]
        best = max((score(a), a) for a in np.arange(best - 3, best + 3.01, 0.25))[1]
        M = cv2.getRotationMatrix2D(ctr, best, 1.0)
        top_rot = cv2.warpAffine(top_img, M, (size, size))
        ok_rot = cv2.warpAffine(top_ok.astype(np.uint8) * 255, M, (size, size), flags=cv2.INTER_NEAREST)
        pa, da = _features(comp, scale=1.0, mask=(owner >= 0).astype(np.uint8) * 255)
        pb, db = _features(top_rot, scale=1.0, mask=ok_rot)
        # spatially gated matching: after the roll search, true matches lie within a few % of each other
        bf = cv2.BFMatcher(cv2.NORM_L2)
        gate = 0.04 * size
        cand = []
        for mm in bf.knnMatch(da, db, k=8):
            near = [x for x in mm if np.linalg.norm(pa[x.queryIdx] - pb[x.trainIdx]) < gate]
            if len(near) >= 1 and (len(near) == 1 or near[0].distance < 0.9 * near[1].distance):
                cand.append((near[0].queryIdx, near[0].trainIdx))
        m = np.array(cand, int).reshape(-1, 2)
        if len(m) < 10:
            raise RuntimeError("top camera: not enough matches in the zenith view")
        Minv = cv2.invertAffineTransform(M)
        pb_orig = np.c_[pb, np.ones(len(pb))] @ Minv.T
        if verbose:
            print(f"top camera: roll about zenith {best:.2f} deg (ncc {max(coarse)[0]:.3f}), {len(m)} gated matches")
        cv2.imwrite(str(Path(__file__).resolve().parents[2] / "Output" / "debug_zenith.jpg"),
                    np.hstack([comp, top_rot]))
        P_comp, P_top = pa[m[:, 0]], pb_orig[m[:, 1]]
        ix, iy = np.clip(P_comp[:, 0].round().astype(int), 0, size - 1), np.clip(P_comp[:, 1].round().astype(int), 0, size - 1)
        jx, jy = np.clip(P_top[:, 0].round().astype(int), 0, size - 1), np.clip(P_top[:, 1].round().astype(int), 0, size - 1)
        own = owner[iy, ix]
        new = []
        for c in ring:
            sel = own == c
            if sel.sum() < 6:
                continue
            u, v = maps[c]
            pr = np.stack([u[iy[sel], ix[sel]], v[iy[sel], ix[sel]]], 1)
            pt = np.stack([tu[jy[sel], jx[sel]], tv[jy[sel], jx[sel]]], 1)
            new.append((c, top, pr, pt))
        return best, new

    def init_top_rotation(self, top):
        ra = np.concatenate([self.rays(top, pt) for (c, t, pr, pt) in self.obs if t == top])
        rb = np.concatenate([self.rays(c, pr) @ self.R[c].T for (c, t, pr, pt) in self.obs if t == top])
        R, inl = _ransac_rotation(ra, rb, thresh_deg=3.0)
        self.R[top] = R
        keep = []
        for (c, t, pr, pt) in self.obs:
            if t != top:
                keep.append((c, t, pr, pt)); continue
            a = self.rays(top, pt) @ R.T
            b = self.rays(c, pr) @ self.R[c].T
            ok = np.sum(a * b, 1) > math.cos(math.radians(3.0))
            if ok.sum() >= 6:
                keep.append((c, t, pr[ok], pt[ok]))
        self.obs = keep
        return int(inl.sum())

    def up_vector(self, ring):
        fwd = np.array([self.R[c] @ np.array([0, 0, 1.0]) for c in ring])
        w, V = np.linalg.eigh(np.cov((fwd - fwd.mean(0)).T))
        up = V[:, 0]
        return up * np.sign(up @ fwd.mean(0)) if abs(up @ fwd.mean(0)) > 1e-3 else up * np.sign(up[2])


def calibrate(frames: list, names: list, base: Rig | None = None, f_init: float | None = None, verbose=True,
              top_frames: int = 8, per_camera_lens: bool = False):
    """frames: list of [image per camera] (one synchronous frame = one list), or a single list of images."""
    if frames and isinstance(frames[0], np.ndarray):
        frames = [frames]
    base = base or default_rig()
    cal = _Cal(frames, names, base, f_init or base.cameras[0].fx)
    cal.match_pairs(verbose)
    e_first, _ = cal.bundle_adjust(verbose=verbose)
    tops = [i for i, nm in enumerate(names) if nm.startswith("top") and cal.R[i] is None]
    for t in tops:
        results = []
        for fi in np.linspace(0, len(frames) - 1, min(top_frames, len(frames))).astype(int):
            cal.images = frames[fi]
            try:
                results.append(cal.attach_top(t, verbose=verbose))
            except RuntimeError as e:
                if verbose:
                    print("  frame", fi, e)
        if not results:
            raise RuntimeError("top camera could not be attached in any frame")
        rolls = np.array([r for r, _ in results])
        med = math.degrees(math.atan2(np.median(np.sin(np.radians(rolls))), np.median(np.cos(np.radians(rolls)))))
        for r, new in results:
            if abs(((r - med + 180) % 360) - 180) < 6:
                cal.obs += new
        n_in = cal.init_top_rotation(t)
        if verbose:
            print(f"top camera: median roll {med:.1f} deg over {len(results)} frames, {n_in} ray inliers")
    if tops:
        cal.bundle_adjust(verbose=verbose)
    if per_camera_lens:
        cal.bundle_adjust(verbose=verbose, per_camera_lens=True)
    missing = [names[i] for i in range(cal.n) if cal.R[i] is None]
    if missing:
        raise RuntimeError(f"cameras not connected by matches: {missing}")

    # ---- levelling: helmet frame with Z = ring-cone axis, X = first ring camera heading
    ring = [i for i, nm in enumerate(names) if not nm.startswith("top")]
    up = cal.up_vector(ring)
    f0 = cal.R[ring[0]] @ np.array([0, 0, 1.0])
    x_ax = f0 - up * (f0 @ up); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(up, x_ax)                 # right-handed internal frame
    L = np.stack([x_ax, y_ax, up])

    errs = cal.errors(cal.f, cal.k, cal.R, per_obs=True)
    entries = []
    for c, nm in enumerate(names):
        w, h = cal.sizes[c]
        R_ue = _FLIP @ (L @ cal.R[c]) @ _CV2UE.T   # rig <- camera, back to UE axes
        p, y, r = _pyr(R_ue)
        if nm.startswith("top"):
            pos = [0.0, 0.0, TOP_HEIGHT_M]
        else:
            pos = [RING_RADIUS_M * math.cos(math.radians(y)), RING_RADIUS_M * math.sin(math.radians(y)), 0.0]
        fc = float(cal.f[c]) if np.ndim(cal.f) else float(cal.f)
        kc = cal.k[c] if np.ndim(cal.k) == 2 else cal.k
        cam = Camera(nm, w, h, fc, fc, (w - 1) / 2, (h - 1) / 2, k=[float(v) for v in kc],
                     position_m=pos, rotation_pyr_deg=[p, y, r], rotation_matrix=R_ue.tolist())
        e_c = np.concatenate([e for (i, j, _, _), e in zip(cal.obs, errs) if c in (i, j)] or [np.zeros(1)])
        entries.append((cam, {"name": nm, "f": round(fc, 1), "k": [round(float(v), 4) for v in kc],
                              "yaw": round(y, 2), "pitch": round(p, 2), "roll": round(r, 2),
                              "matches": int(len(e_c)), "rms_px": round(float(np.sqrt(np.mean(e_c ** 2))), 2)}))
    # ring order by yaw (UE yaw grows to the right); errors vs. the scheme: 360/7 steps, +9.3 deg pitch
    ring_e = sorted([e for e in entries if not e[0].name.startswith("top")], key=lambda e: e[1]["yaw"] % 360)
    step = 360.0 / len(ring_e)
    for idx, (cam, d) in enumerate(ring_e):
        d["ring_index"] = idx
        d["yaw_error_deg"] = round(((d["yaw"] - idx * step + 180) % 360) - 180, 2)
        d["pitch_error_deg"] = round(d["pitch"] - 9.3, 2)
        d["roll_error_deg"] = d["roll"]
    ordered = ring_e + [e for e in entries if e[0].name.startswith("top")]

    e_all = cal.errors(cal.f, cal.k, cal.R)
    f_m = float(np.exp(np.mean(np.log(cal.f)))) if np.ndim(cal.f) else float(cal.f)
    k_m = np.mean(cal.k, 0) if np.ndim(cal.k) == 2 else cal.k
    th = lambda rpx: math.degrees(_theta_of_radius(rpx, f_m, k_m))
    W, H = cal.sizes[ring[0]]
    report = {
        "frames": len(frames), "matches": int(len(e_all)),
        "rms_px_nominal_lens": round(float(np.sqrt(np.mean(e_first ** 2))), 2),
        "rms_px_calibrated": round(float(np.sqrt(np.mean(e_all ** 2))), 2),
        "median_px_calibrated": round(float(np.median(e_all)), 2),
        "per_camera_lens": per_camera_lens,
        "focal_px": round(f_m, 1), "k_kannala_brandt": [round(float(v), 5) for v in k_m],
        "fov_deg": {"long_side": round(2 * th(max(W, H) / 2), 1), "short_side": round(2 * th(min(W, H) / 2), 1),
                    "diagonal": round(2 * th(math.hypot(W, H) / 2), 1)},
        "lens_distortion_vs_equidistant_at_edge_%": round(100 * (th(max(W, H) / 2) / math.degrees(max(W, H) / 2 / f_m) - 1), 1),
        "cameras": [d for _, d in ordered],
    }
    return Rig("SmartHelmet_real_selfcal", [c for c, _ in ordered]), report


def calibrate_session(session_dir, names=None, n_frames=16, f_init=680.0, out_rig=None, verbose=True,
                      per_camera_lens=False):
    """Calibrate from several synchronous frames of a session (frames/<id>/<cam>.jpg)."""
    session_dir = Path(session_dir)
    fdirs = sorted(p for p in (session_dir / "frames").iterdir() if p.is_dir())
    if names is None:
        names = sorted(p.stem for p in fdirs[0].glob("*.jpg"))
        names = [n for n in names if not n.startswith("top")] + [n for n in names if n.startswith("top")]
    pick = [fdirs[i] for i in np.linspace(0, len(fdirs) - 1, min(n_frames, len(fdirs))).astype(int)]
    frames = [[cv2.imread(str(next(d.glob(nm + ".*")))) for nm in names] for d in pick]
    rig, report = calibrate(frames, names, f_init=f_init, verbose=verbose, per_camera_lens=per_camera_lens)
    out_rig = Path(out_rig or session_dir / "rig.json")
    rig.save(out_rig)
    (out_rig.parent / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rig, report


def calibrate_frame(frame_dir, names=None, out_rig=None, f_init=680.0, verbose=True):
    frame_dir = Path(frame_dir)
    names = names or [f"cam{i}" for i in range(7)] + ["top7"]
    imgs = [cv2.imread(str(next(frame_dir.glob(nm + ".*")))) for nm in names]
    rig, report = calibrate([imgs], names, f_init=f_init, verbose=verbose)
    out_rig = Path(out_rig or frame_dir.parent.parent / "rig.json")
    rig.save(out_rig)
    (out_rig.parent / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rig, report
