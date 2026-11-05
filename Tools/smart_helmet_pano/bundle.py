"""Generic multi-camera bundle adjustment for target-based calibration of the helmet.

One solver for every calibration stand:
  * fixed points      – surveyed 3D coordinates (calibration cube / room)
  * free points       – unknown 3D points (marker stickers, projected corners, natural features),
                        optional survey prior, marker-size (distance) prior, planarity prior
  * board points      – rigid board with known layout and unknown pose per placement (handheld ChArUco)
Unknowns: per-camera intrinsics (Kannala-Brandt: fx, fy, cx, cy, k1..k4), per-camera extrinsics
(rotation + lens centre in the rig frame, soft CAD prior), helmet pose of every capture.

Frames follow rig.py: rig/world = UE axes (X fwd, Y right, Z up); camera rotation R_c = rig <- camera
(columns fwd, right, up); OpenCV camera coordinates (x right, y down, z optical axis).
Residuals are in units of the detection sigma; Huber loss; sparse finite-difference Jacobian.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.spatial.transform import Rotation as Rot

from .rig import Camera, Rig

FIXED, FREE, BOARD = 0, 1, 2


def kb_project(Xc_cv, f, fy, cx, cy, k, p=None):
    """Vectorised Kannala-Brandt projection (+ optional decentering p1, p2); all intrinsics per row."""
    x, y, z = Xc_cv[:, 0], Xc_cv[:, 1], Xc_cv[:, 2]
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(r, z)
    t2 = th * th
    td = th * (1 + t2 * (k[:, 0] + t2 * (k[:, 1] + t2 * (k[:, 2] + t2 * k[:, 3]))))
    s = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 0.0)
    xd, yd = x * s, y * s
    if p is not None:
        p1, p2 = p[:, 0], p[:, 1]
        r2 = xd * xd + yd * yd
        xd, yd = xd + 2 * p1 * xd * yd + p2 * (r2 + 2 * xd * xd), yd + p1 * (r2 + 2 * yd * yd) + 2 * p2 * xd * yd
    return np.stack([f * xd + cx, fy * yd + cy], 1)


def ue_to_cv(Xu):
    return np.stack([Xu[:, 1], -Xu[:, 2], Xu[:, 0]], 1)


def cv_to_ue(Xc):
    return np.stack([Xc[..., 2], Xc[..., 0], -Xc[..., 1]], -1)


@dataclass
class Problem:
    rig0: Rig
    obs_cam: np.ndarray
    obs_cap: np.ndarray
    obs_pt: np.ndarray
    obs_uv: np.ndarray
    cap_R: list                      # initial world <- rig rotations
    cap_t: list
    pt_kind: np.ndarray              # FIXED / FREE / BOARD
    pt_X: np.ndarray                 # world coords (FIXED / FREE init) or board-local coords (BOARD)
    pt_board: np.ndarray             # board index for BOARD points, else -1
    board_R: list = field(default_factory=list)
    board_t: list = field(default_factory=list)
    obs_sigma: np.ndarray | None = None
    cap_fixed: np.ndarray | None = None
    pt_prior_sigma: np.ndarray | None = None   # (P,) metres, inf = no prior (FREE points only)
    pt_prior_X: np.ndarray | None = None
    dist_pairs: np.ndarray | None = None       # (M, 2) point indices
    dist_value: np.ndarray | None = None
    dist_sigma: float = 0.002
    planes: list = field(default_factory=list)  # list of point-index arrays
    plane_sigma: float = 0.003
    # what is estimated
    est_f: bool = True
    est_aspect: bool = False
    est_pp: bool = True
    est_k: int = 4
    est_tangential: bool = False
    shared_lens: bool = False
    est_rot: bool = True
    est_t: bool = True
    t_prior_m: float = 0.003
    rot_gauge_deg: float = 30.0
    # weak prior k_i ~ N(0, sigma_i): keeps the lens polynomial sane where no target reached the image
    k_prior_sigma: tuple | None = None


class BundleAdjuster:
    def __init__(self, pb: Problem):
        self.pb = pb
        self.nc = len(pb.rig0.cameras)
        cams = pb.rig0.cameras
        self.I0 = np.array([[c.fx, c.fy, c.cx, c.cy, *c.k, *(getattr(c, "p", None) or [0.0, 0.0])] for c in cams], float)
        self.R0 = np.array([np.asarray(c.R_rig_from_cam_ue, float) for c in cams])
        self.T0 = np.array([np.asarray(c.position_m, float) for c in cams])
        self.Tprior = self.T0.copy()        # CAD lens centres: the prior stays here even if T0 is re-seeded
        self.capR0 = np.array(pb.cap_R, float).reshape(-1, 3, 3)
        self.capT0 = np.array(pb.cap_t, float).reshape(-1, 3)
        self.bR0 = np.array(pb.board_R, float).reshape(-1, 3, 3)
        self.bT0 = np.array(pb.board_t, float).reshape(-1, 3)
        self.X = np.array(pb.pt_X, float)
        nobs = len(pb.obs_cam)
        self.sig = pb.obs_sigma if pb.obs_sigma is not None else np.full(nobs, 0.5)
        self.cap_fixed = pb.cap_fixed if pb.cap_fixed is not None else np.zeros(len(self.capR0), bool)
        self.active = np.ones(nobs, bool)
        self._layout()

    # ------------------------------------------------------------ parameter layout
    def _layout(self):
        pb, nc = self.pb, self.nc
        o = 0
        # intrinsic index map: (nc, 8) -> x index or -1
        imap = -np.ones((nc, 10), int)
        free_cols = []
        if pb.est_f:
            free_cols.append(0)
        if pb.est_aspect:
            free_cols.append(1)
        if pb.est_pp:
            free_cols += [2, 3]
        free_cols += list(range(4, 4 + pb.est_k))
        if pb.est_tangential:
            free_cols += [8, 9]
        shared = {0, 4, 5, 6, 7} if pb.shared_lens else set()
        shared_idx = {}
        for c in range(nc):
            for col in free_cols:
                if col in shared:
                    if col not in shared_idx:
                        shared_idx[col] = o; o += 1
                    imap[c, col] = shared_idx[col]
                else:
                    imap[c, col] = o; o += 1
        self.imap = imap
        self.o_rot = o
        o += 3 * nc if pb.est_rot else 0
        self.o_t = o
        o += 3 * nc if pb.est_t else 0
        self.free_caps = np.flatnonzero(~self.cap_fixed)
        self.cap_idx = -np.ones(len(self.capR0), int)
        self.cap_idx[self.free_caps] = np.arange(len(self.free_caps))
        self.o_cap = o
        o += 6 * len(self.free_caps)
        self.free_pts = np.flatnonzero(pb.pt_kind == FREE)
        self.pt_idx = -np.ones(len(pb.pt_kind), int)
        self.pt_idx[self.free_pts] = np.arange(len(self.free_pts))
        self.o_pt = o
        o += 3 * len(self.free_pts)
        self.o_board = o
        o += 6 * len(self.bR0)
        self.o_plane = o
        o += 3 * len(pb.planes)
        self.n = o

    def _k_prior_cols(self):
        """x indices of the estimated k1..k4 (unique, shared lens counted once) and their k order."""
        cols, js = [], []
        for col in range(4, 8):
            for ix in np.unique(self.imap[:, col]):
                if ix >= 0:
                    cols.append(ix); js.append(col - 4)
        self._k_prior_j = np.array(js, int)
        return np.array(cols, int)

    def x0(self):
        x = np.zeros(self.n)
        for c in range(self.nc):
            for col in range(10):
                if self.imap[c, col] >= 0:
                    x[self.imap[c, col]] = self.I0[c, col]
        for i, pts in enumerate(self.pb.planes):
            P = self.X[pts]
            ctr = P.mean(0)
            nrm = np.linalg.svd(P - ctr)[2][2]
            d = nrm @ ctr
            if d < 0:
                nrm, d = -nrm, -d
            x[self.o_plane + 3 * i: self.o_plane + 3 * i + 3] = nrm * max(d, 0.5)
        return x

    def unpack(self, x):
        I = self.I0.copy()
        m = self.imap >= 0
        I[m] = x[self.imap[m]]
        if not self.pb.est_aspect:
            I[:, 1] = I[:, 0] * (self.I0[:, 1] / self.I0[:, 0])
        R = self.R0.copy()
        if self.pb.est_rot:
            R = Rot.from_rotvec(x[self.o_rot:self.o_rot + 3 * self.nc].reshape(-1, 3)).as_matrix() @ self.R0
        T = self.T0.copy()
        if self.pb.est_t:
            T = self.T0 + x[self.o_t:self.o_t + 3 * self.nc].reshape(-1, 3)
        cR, cT = self.capR0.copy(), self.capT0.copy()
        if len(self.free_caps):
            v = x[self.o_cap:self.o_cap + 6 * len(self.free_caps)].reshape(-1, 6)
            cR[self.free_caps] = Rot.from_rotvec(v[:, :3]).as_matrix() @ self.capR0[self.free_caps]
            cT[self.free_caps] = self.capT0[self.free_caps] + v[:, 3:]
        X = self.X.copy()
        if len(self.free_pts):
            X[self.free_pts] = self.X[self.free_pts] + x[self.o_pt:self.o_pt + 3 * len(self.free_pts)].reshape(-1, 3)
        bR, bT = self.bR0.copy(), self.bT0.copy()
        if len(self.bR0):
            v = x[self.o_board:self.o_board + 6 * len(self.bR0)].reshape(-1, 6)
            bR = Rot.from_rotvec(v[:, :3]).as_matrix() @ self.bR0
            bT = self.bT0 + v[:, 3:]
        return I, R, T, cR, cT, X, bR, bT

    def world_points(self, X, bR, bT, idx):
        pb = self.pb
        out = X[idx].copy()
        kb = pb.pt_kind[idx] == BOARD
        if kb.any():
            b = pb.pt_board[idx[kb]]
            out[kb] = np.einsum("nij,nj->ni", bR[b], self.X[idx[kb]]) + bT[b]
        return out

    def predict(self, x, sel=None):
        pb = self.pb
        sel = np.flatnonzero(self.active) if sel is None else sel
        I, R, T, cR, cT, X, bR, bT = self.unpack(x)
        c, k, p = pb.obs_cam[sel], pb.obs_cap[sel], pb.obs_pt[sel]
        Xw = self.world_points(X, bR, bT, p)
        Xr = np.einsum("nji,nj->ni", cR[k], Xw - cT[k])
        Xc = np.einsum("nji,nj->ni", R[c], Xr - T[c])
        Ii = I[c]
        return kb_project(ue_to_cv(Xc), Ii[:, 0], Ii[:, 1], Ii[:, 2], Ii[:, 3], Ii[:, 4:8], Ii[:, 8:10]), Xc

    def residuals(self, x):
        pb = self.pb
        sel = np.flatnonzero(self.active)
        uv, Xc = self.predict(x, sel)
        r = ((uv - pb.obs_uv[sel]) / self.sig[sel, None])
        behind = Xc[:, 0] < -0.02
        r[behind] *= 3                                       # keep points in front of the lens
        out = [r.ravel()]
        I, R, T, cR, cT, X, bR, bT = self.unpack(x)
        if pb.est_t:
            out.append(((T - self.Tprior).ravel() / pb.t_prior_m))
        if pb.est_rot:
            out.append(x[self.o_rot:self.o_rot + 3 * self.nc] / math.radians(pb.rot_gauge_deg))
        if pb.k_prior_sigma is not None:
            ks = self._k_prior_cols()
            out.append(x[ks] / np.asarray(pb.k_prior_sigma)[self._k_prior_j])
        if pb.pt_prior_sigma is not None and len(self.free_pts):
            s = pb.pt_prior_sigma[self.free_pts]
            ok = np.isfinite(s)
            if ok.any():
                out.append(((X[self.free_pts[ok]] - pb.pt_prior_X[self.free_pts[ok]]) / s[ok, None]).ravel())
        if pb.dist_pairs is not None and len(pb.dist_pairs):
            a, b = pb.dist_pairs[:, 0], pb.dist_pairs[:, 1]
            Pa = self.world_points(X, bR, bT, a)
            Pb = self.world_points(X, bR, bT, b)
            out.append((np.linalg.norm(Pa - Pb, axis=1) - pb.dist_value) / pb.dist_sigma)
        for i, pts in enumerate(pb.planes):
            q = x[self.o_plane + 3 * i: self.o_plane + 3 * i + 3]
            d = np.linalg.norm(q)
            out.append((X[pts] @ (q / d) - d) / pb.plane_sigma)
        return np.concatenate(out)

    def sparsity(self):
        pb = self.pb
        sel = np.flatnonzero(self.active)
        rows, cols = [], []
        n = len(sel)
        c, k, p = pb.obs_cam[sel], pb.obs_cap[sel], pb.obs_pt[sel]
        r2 = np.arange(n) * 2

        def add(rr, cc):
            rows.append(rr); cols.append(cc)
        for col in range(10):
            ix = self.imap[c, col]
            ok = ix >= 0
            for d in (0, 1):
                add(r2[ok] + d, ix[ok])
        for base, on in ((self.o_rot, pb.est_rot), (self.o_t, pb.est_t)):
            if on:
                for j in range(3):
                    for d in (0, 1):
                        add(r2 + d, base + 3 * c + j)
        ci = self.cap_idx[k]
        ok = ci >= 0
        for j in range(6):
            for d in (0, 1):
                add(r2[ok] + d, self.o_cap + 6 * ci[ok] + j)
        pi = self.pt_idx[p]
        ok = pi >= 0
        for j in range(3):
            for d in (0, 1):
                add(r2[ok] + d, self.o_pt + 3 * pi[ok] + j)
        kb = pb.pt_kind[p] == BOARD
        bi = pb.pt_board[p]
        for j in range(6):
            for d in (0, 1):
                add(r2[kb] + d, self.o_board + 6 * bi[kb] + j)
        row = 2 * n
        if pb.est_t:
            add(row + np.arange(3 * self.nc), self.o_t + np.arange(3 * self.nc)); row += 3 * self.nc
        if pb.est_rot:
            add(row + np.arange(3 * self.nc), self.o_rot + np.arange(3 * self.nc)); row += 3 * self.nc
        if pb.k_prior_sigma is not None:
            ks = self._k_prior_cols()
            add(row + np.arange(len(ks)), ks); row += len(ks)
        if pb.pt_prior_sigma is not None and len(self.free_pts):
            s = pb.pt_prior_sigma[self.free_pts]
            okp = np.flatnonzero(np.isfinite(s))
            for j in range(3):
                add(row + 3 * np.arange(len(okp)) + j, self.o_pt + 3 * okp + j)
            row += 3 * len(okp)
        if pb.dist_pairs is not None and len(pb.dist_pairs):
            m = len(pb.dist_pairs)
            for e in (0, 1):
                q = pb.dist_pairs[:, e]
                pi = self.pt_idx[q]
                ok = pi >= 0
                for j in range(3):
                    add(row + np.flatnonzero(ok), self.o_pt + 3 * pi[ok] + j)
                kb = pb.pt_kind[q] == BOARD
                for j in range(6):
                    add(row + np.flatnonzero(kb), self.o_board + 6 * pb.pt_board[q[kb]] + j)
            row += m
        for i, pts in enumerate(pb.planes):
            m = len(pts)
            pi = self.pt_idx[pts]
            for j in range(3):
                add(row + np.arange(m), self.o_pt + 3 * pi + j)
                add(row + np.arange(m), np.full(m, self.o_plane + 3 * i + j))
            row += m
        rr, cc = np.concatenate(rows), np.concatenate(cols)
        return sp.coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(row, self.n)).tocsr()

    # ------------------------------------------------------------ solve
    def solve(self, x=None, passes=3, max_nfev=60, verbose=True, label="", huber=3.0):
        x = self.x0() if x is None else x
        t0 = time.perf_counter()
        for p in range(passes):
            S = self.sparsity()
            x = levenberg_marquardt(self.residuals, x, S, 2 * int(self.active.sum()), huber=huber,
                                    max_iter=max_nfev)
            e = self.obs_errors(x)
            if p < passes - 1:
                med = np.median(e[self.active])
                thr = max(4 * 1.4826 * med, 2.5 * np.median(self.sig))
                self.active &= e < thr
            if verbose:
                print(f"  BA{label} pass {p}: rms {np.sqrt(np.mean(e[self.active] ** 2)):.3f}px "
                      f"median {np.median(e[self.active]):.3f}px, obs {self.active.sum()}/{len(self.active)}, "
                      f"{time.perf_counter() - t0:.1f}s")
        self.x = x
        return x

    def obs_errors(self, x):
        uv, _ = self.predict(x, np.arange(len(self.pb.obs_cam)))
        return np.linalg.norm(uv - self.pb.obs_uv, axis=1)

    def rig(self, x=None, name="bundle") -> Rig:
        x = self.x if x is None else x
        I, R, T, *_ = self.unpack(x)
        cams = []
        for c, cam in enumerate(self.pb.rig0.cameras):
            cams.append(Camera(cam.name, cam.width, cam.height, float(I[c, 0]), float(I[c, 1]), float(I[c, 2]),
                               float(I[c, 3]), k=[float(v) for v in I[c, 4:8]], position_m=T[c].tolist(),
                               p=[float(v) for v in I[c, 8:10]],
                               rotation_pyr_deg=list(cam.rotation_pyr_deg), max_theta_deg=cam.max_theta_deg,
                               rotation_matrix=R[c].tolist()))
        return Rig(name, cams)

    def report(self, x=None):
        x = self.x if x is None else x
        e = self.obs_errors(x)
        a = self.active
        per = {}
        for c, cam in enumerate(self.pb.rig0.cameras):
            m = a & (self.pb.obs_cam == c)
            per[cam.name] = {"obs": int(m.sum()), "rms_px": round(float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else -1, 3)}
        return {"obs_used": int(a.sum()), "obs_total": int(len(a)), "rms_px": round(float(np.sqrt(np.mean(e[a] ** 2))), 3),
                "median_px": round(float(np.median(e[a])), 3), "per_camera": per}


# ---------------------------------------------------------------- solver
def levenberg_marquardt(fun, x, S, n_robust, huber=3.0, max_iter=60, tol=1e-7):
    """Sparse LM: finite-difference Jacobian with column grouping, Huber IRLS weights on the first
    `n_robust` residuals (reprojection), normal equations solved directly (SuperLU)."""
    groups = group_columns(S)
    r = fun(x)

    def weights(r):
        w = np.ones_like(r)
        a = np.abs(r[:n_robust])
        w[:n_robust] = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        return w

    def cost(r):
        a = np.abs(r[:n_robust])
        c = np.where(a <= huber, 0.5 * a * a, huber * (a - 0.5 * huber)).sum()
        return c + 0.5 * float(r[n_robust:] @ r[n_robust:])
    c0 = cost(r)
    lam = 1e-3
    for it in range(max_iter):
        J = approx_derivative(fun, x, method="2-point", f0=r, sparsity=(S, groups))
        J = sp.csr_matrix(J)
        w = weights(r)
        Jw = sp.diags(np.sqrt(w)) @ J
        A = (Jw.T @ Jw).tocsc()
        g = Jw.T @ (np.sqrt(w) * r)
        dA = A.diagonal()
        improved = False
        for _ in range(8):
            M = A + sp.diags(lam * np.maximum(dA, 1e-9))
            try:
                dx = -spla.spsolve(M, g)
            except Exception:  # noqa: BLE001
                lam *= 10; continue
            if not np.all(np.isfinite(dx)):
                lam *= 10; continue
            r1 = fun(x + dx)
            c1 = cost(r1)
            if c1 < c0:
                rel = (c0 - c1) / max(c0, 1e-12)
                x, r, c0 = x + dx, r1, c1
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 4
        if not improved or rel < tol:
            break
    return x


# ---------------------------------------------------------------- helpers
def camera_rays(cam: Camera, uv):
    """Unit rays in the rig frame + lens centre for pixel observations of one camera."""
    return cam.unproject(uv[:, 0], uv[:, 1]), np.asarray(cam.position_m, float)


def triangulate(rays_w, origins_w):
    """Least-squares closest point to several rays (n,3)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for d, o in zip(rays_w, origins_w):
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ o
    return np.linalg.lstsq(A, b, rcond=None)[0], np.linalg.cond(A)


def init_free_points(rig: Rig, obs_cam, obs_cap, obs_pt, obs_uv, cap_R, cap_t, n_pts, default_depth=2.5):
    """Triangulate every free point from its observations with the initial rig; fallback: depth prior."""
    X = np.zeros((n_pts, 3))
    ok = np.zeros(n_pts, bool)
    rays = np.zeros((len(obs_cam), 3)); orig = np.zeros((len(obs_cam), 3))
    for c, cam in enumerate(rig.cameras):
        m = obs_cam == c
        if m.any():
            r, o = camera_rays(cam, obs_uv[m])
            rays[m], orig[m] = r, o
    capR, capT = np.asarray(cap_R).reshape(-1, 3, 3), np.asarray(cap_t).reshape(-1, 3)
    rw = np.einsum("nij,nj->ni", capR[obs_cap], rays)
    ow = np.einsum("nij,nj->ni", capR[obs_cap], orig) + capT[obs_cap]
    order = np.argsort(obs_pt, kind="stable")
    bounds = np.searchsorted(obs_pt[order], np.arange(n_pts + 1))
    for p in range(n_pts):
        idx = order[bounds[p]:bounds[p + 1]]
        if len(idx) == 0:
            continue
        d = rw[idx]
        if len(idx) >= 2:
            P, cond = triangulate(d, ow[idx])
            depth = (P - ow[idx]) @ d.T
            if cond < 1e6 and np.all(np.diag(depth) > 0.3) and np.all(np.diag(depth) < 20):
                X[p] = P; ok[p] = True; continue
        md = d.mean(0); md /= np.linalg.norm(md)
        X[p] = ow[idx].mean(0) + md * default_depth
        ok[p] = len(idx) >= 2
    return X, ok
