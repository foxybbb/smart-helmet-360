"""Target-based calibration of the real helmet with a handheld ChArUco board (lab / factory calibration).

Capture: the helmet stands still, the board is moved around it.  Every board placement = one synchronous
frame folder  frames/<id>/<cam>.jpg  (all cameras shoot, the board may be visible in one or several).
Cover every image including its edges (~15 placements per camera) and put the board into every overlap of
neighbouring cameras (~5 placements per pair, including ring <-> top).  Hold the board still while the
Raspberry Pis shoot: without a hardware trigger the cameras are not synchronous.

Solved jointly (bundle.py): per-camera fx, fy, cx, cy, k1..k4, decentering p1 p2, mounting rotation and lens
centre (soft CAD prior), and the pose of every board placement.  Simulation study: Docs/CALIBRATION.md.

  python -m smart_helmet_pano board --out board.png            # printable board (default 11 x 8 x 50 mm)
  python -m smart_helmet_pano calibrate-board <session> [--rig nominal_rig.json]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .bundle import BOARD, BundleAdjuster, Problem
from .rig import Rig, default_rig
from . import targets as TG

ARUCO = cv2.aruco
_CV2UE = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], float)


def make_board(squares=(11, 8), square_m=0.05, marker_ratio=0.75, dictionary=ARUCO.DICT_5X5_1000):
    d = ARUCO.getPredefinedDictionary(dictionary)
    n = (squares[0] * squares[1]) // 2
    return ARUCO.CharucoBoard(squares, square_m, square_m * marker_ratio, d, np.arange(n, dtype=np.int32))


def save_board_image(path, board, dpi=300, margin_mm=10):
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    px = lambda m: int(round(m / 0.0254 * dpi))
    img = board.generateImage((px(sx * sq), px(sy * sq)), marginSize=0, borderBits=1)
    m = px(margin_mm / 1000)
    img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(str(path), img)
    return img.shape[1], img.shape[0]


def solve_board(rig0: Rig, detections, board, sigma_px=0.3, verbose=True, min_corners=6, tangential=True):
    """detections: list of (placement id, camera index, corner ids (n,), corners (n, 2)).
    tangential: also estimate the lens decentering p1, p2 (simulation: seams 2.0 -> 0.45 arcmin)."""
    loc = board.getChessboardCorners().astype(np.float64)
    nco = len(loc)
    per = {}
    for k, ci, ids, cc in detections:
        if len(ids) >= min_corners:
            per.setdefault(k, []).append((ci, np.asarray(ids, int), np.asarray(cc, float)))
    used = sorted(per)
    bidx = {k: j for j, k in enumerate(used)}
    bR, bT, obs = [], [], []
    for k in used:
        ci, ids, cc = max(per[k], key=lambda e: len(e[1]))      # board pose from the best view, nominal lens
        c = rig0.cameras[ci]
        und = cv2.fisheye.undistortPoints(cc.reshape(-1, 1, 2), c.K, c.D).reshape(-1, 2)
        _, rv, tv = cv2.solvePnP(loc[ids], und, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        Rc = np.asarray(c.R_rig_from_cam_ue)
        bR.append(Rc @ _CV2UE @ cv2.Rodrigues(rv)[0])
        bT.append(Rc @ _CV2UE @ tv.ravel() + np.asarray(c.position_m))
        for ci2, ids2, cc2 in per[k]:
            for i, (u, v) in zip(ids2, cc2):
                obs.append((ci2, bidx[k] * nco + i, u, v))
    a = np.array(obs, float)
    cam, pt, uv = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:4]
    P = len(used) * nco
    common = dict(board_R=bR, board_t=bT, cap_fixed=np.array([True]), obs_sigma=np.full(len(cam), sigma_px))
    args = (cam, np.zeros(len(cam), int), pt, uv, [np.eye(3)], [np.zeros(3)], np.full(P, BOARD),
            np.tile(loc, (len(used), 1)), np.repeat(np.arange(len(used)), nco))
    # stage 1: datasheet lens, only mounting + board poses;  stage 2: everything
    pb = Problem(rig0, *args, est_f=False, est_pp=False, est_k=0, **common)
    ba1 = BundleAdjuster(pb)
    x1 = ba1.solve(passes=1, max_nfev=40, verbose=verbose, label=" board/init")
    _, R, T, cR, cT, X, bR1, bT1 = ba1.unpack(x1)
    pb = Problem(rig0, *args, est_tangential=tangential, **common)
    ba = BundleAdjuster(pb)
    ba.R0, ba.T0, ba.bR0, ba.bT0 = R, T, bR1, bT1
    x = ba.solve(passes=3, max_nfev=80, verbose=verbose, label=" board")
    rep = ba.report(x)
    rep["coverage"] = coverage(rig0, cam, uv)
    rep["placements"] = len(used)
    rep["multi_camera_placements"] = sum(len({e[0] for e in per[k]}) > 1 for k in used)
    return ba.rig(x, rig0.name + "_charuco"), rep


def coverage(rig0: Rig, cam, uv, grid=(5, 8)):
    """Share of image cells (grid columns x rows) that contain at least one corner, per camera.
    The lens polynomial is only trustworthy where there were corners: aim for 100 %, at least the
    cells the stitcher uses (theta < 75 deg)."""
    out = {}
    for ci, c in enumerate(rig0.cameras):
        m = cam == ci
        occ = np.zeros(grid[::-1], bool)
        if m.any():
            gx = np.clip((uv[m, 0] / c.width * grid[0]).astype(int), 0, grid[0] - 1)
            gy = np.clip((uv[m, 1] / c.height * grid[1]).astype(int), 0, grid[1] - 1)
            occ[gy, gx] = True
        out[c.name] = {"cells_%": round(100 * float(occ.mean()), 1),
                       "map": ["".join("#" if v else "." for v in row) for row in occ]}
    return out


def calibrate_board_session(session, rig0: Rig | None = None, board=None, f_init=None, out_rig=None, verbose=True,
                            tangential=True):
    session = Path(session)
    board = board or make_board()
    if rig0 is None:
        rig0 = Rig.load(session / "rig.json") if (session / "rig.json").exists() else default_rig()
    if f_init:
        for c in rig0.cameras:
            c.fx = c.fy = float(f_init)
    fdirs = sorted(p for p in (session / "frames").iterdir() if p.is_dir())
    det = TG.aruco_detector(ARUCO.DICT_5X5_1000)
    detections = []
    t0 = time.perf_counter()
    for k, fd in enumerate(fdirs):
        for ci, c in enumerate(rig0.cameras):
            f = next((p for p in (fd / f"{c.name}{e}" for e in (".jpg", ".png", ".jpeg")) if p.exists()), None)
            if f is None:
                continue
            for _, ids, cc in TG.detect_charuco(cv2.imread(str(f)), [board], det):
                detections.append((k, ci, ids, cc))
    if verbose:
        print(f"detected {len(detections)} board views in {len(fdirs)} frames ({time.perf_counter() - t0:.0f}s)")
    rig, rep = solve_board(rig0, detections, board, verbose=verbose, tangential=tangential)
    if verbose:
        for nm, cv in rep["coverage"].items():
            warn = "   <- add board placements in the empty cells" if cv["cells_%"] < 90 else ""
            print(f"  {nm}: corners in {cv['cells_%']}% of the image cells {'|'.join(cv['map'])}{warn}")
    for c, c0 in zip(rig.cameras, rig0.cameras):
        c.max_theta_deg = c0.max_theta_deg
    out_rig = Path(out_rig or session / "rig.json")
    rig.save(out_rig)
    rep["cameras"] = [{"name": c.name, "f": round(c.fx, 1), "fy": round(c.fy, 1), "cx": round(c.cx, 1),
                       "cy": round(c.cy, 1), "k": [round(v, 5) for v in c.k], "p": [round(v, 6) for v in c.p],
                       "position_mm": [round(1000 * v, 1) for v in c.position_m], **rep["per_camera"][c.name],
                       "coverage_%": rep["coverage"][c.name]["cells_%"]}
                      for c in rig.cameras]
    rep.pop("per_camera", None)
    (out_rig.parent / "board_calibration_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rig, rep


# ---------------------------------------------------------------- field re-calibration (hybrid)
def refine_mounts(frames, lens_rig: Rig, max_per_pair=40, verbose=True):
    """Lens (fx, fy, cx, cy, k) fixed from the lab calibration; mounting rotations and lens centres
    re-estimated from SIFT matches of ordinary frames (after knocks / transport).  Simulation: the lab
    accuracy is restored (seams 2.8 arcmin vs 77 for the stale lab rig, Docs/CALIBRATION.md "Field drift").
    frames: list of [image per camera, same order as lens_rig.cameras]."""
    from .bundle import FREE, init_free_points
    from .calibrate import _features, _match, _ransac_rotation
    rng = np.random.default_rng(0)
    obs, pid = [], 0
    for fi, imgs in enumerate(frames):
        feats = [_features(im) for im in imgs]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                m = _match(feats[i][1], feats[j][1])
                if len(m) < 15:
                    continue
                pa, pb = feats[i][0][m[:, 0]], feats[j][0][m[:, 1]]
                ra = lens_rig.cameras[i].unproject(pa[:, 0], pa[:, 1])
                rb = lens_rig.cameras[j].unproject(pb[:, 0], pb[:, 1])
                _, inl = _ransac_rotation(ra, rb, thresh_deg=1.5)
                if inl.sum() < 15:
                    continue
                sel = np.flatnonzero(inl)
                if len(sel) > max_per_pair:
                    sel = rng.choice(sel, max_per_pair, replace=False)
                for s in sel:
                    obs.append((i, fi, pid, pa[s, 0], pa[s, 1]))
                    obs.append((j, fi, pid, pb[s, 0], pb[s, 1]))
                    pid += 1
    if not obs:
        raise RuntimeError("no verified matches between cameras")
    a = np.array(obs, float)
    cam, cap, pt, uv = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int), a[:, 3:5]
    nf = len(frames)
    eye, zero = [np.eye(3)] * nf, [np.zeros(3)] * nf
    X0, _ = init_free_points(lens_rig, cam, cap, pt, uv, eye, zero, pid, default_depth=2.5)
    pb = Problem(lens_rig, cam, cap, pt, uv, eye, zero, np.full(pid, FREE), X0, -np.ones(pid, int),
                 cap_fixed=np.ones(nf, bool), est_f=False, est_pp=False, est_k=0,
                 obs_sigma=np.full(len(cam), 0.8), t_prior_m=0.002)
    ba = BundleAdjuster(pb)
    x = ba.solve(passes=3, max_nfev=80, verbose=verbose, label=" mounts")
    rig = ba.rig(x, lens_rig.name + "_field")
    for c, c0 in zip(rig.cameras, lens_rig.cameras):
        c.max_theta_deg = c0.max_theta_deg
    rep = ba.report(x)
    rep["frames"], rep["matches"] = nf, pid
    # change of every mount relative to the lab rig, without the common rotation of the whole helmet
    Rn = [np.asarray(c.R_rig_from_cam_ue) for c in rig.cameras]
    Ro = [np.asarray(c.R_rig_from_cam_ue) for c in lens_rig.cameras]
    U, _, Vt = np.linalg.svd(sum(o @ n.T for n, o in zip(Rn, Ro)))
    G = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    rep["rotation_change_deg"] = {c.name: round(float(np.degrees(np.arccos(np.clip(
        (np.trace(o.T @ G @ n) - 1) / 2, -1, 1)))), 3) for c, n, o in zip(rig.cameras, Rn, Ro)}
    return rig, rep


def refine_mounts_session(session, lens_rig_path, n_frames=8, out_rig=None, verbose=True):
    """Pick n_frames evenly from frames/<id>/<cam>.jpg (static moments, varied places) and refine the mounts."""
    session = Path(session)
    lens = Rig.load(lens_rig_path)
    fdirs = sorted(p for p in (session / "frames").iterdir() if p.is_dir())
    pick = [fdirs[i] for i in np.linspace(0, len(fdirs) - 1, min(n_frames, len(fdirs))).astype(int)]
    frames = []
    for d in pick:
        imgs = []
        for c in lens.cameras:
            f = next((p for p in (d / f"{c.name}{e}" for e in (".jpg", ".png", ".jpeg")) if p.exists()), None)
            if f is None:
                raise FileNotFoundError(f"{c.name} missing in {d}")
            imgs.append(cv2.imread(str(f)))
        frames.append(imgs)
    rig, rep = refine_mounts(frames, lens, verbose=verbose)
    out_rig = Path(out_rig or session / "rig.json")
    rig.save(out_rig)
    rep.pop("per_camera", None)
    (out_rig.parent / "mount_refinement_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rig, rep
