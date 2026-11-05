"""Calibration methods = capture plan on a stand + detection on the rendered frames + bundle adjustment.

Every method only uses what a real operator would have: the nominal rig (CAD + datasheet), commanded tripod
poses, the pattern layout (and, for the surveyed stands, the measured coordinates).  Ground truth is used to
render the frames and, for the handheld board, to emulate an operator who watches the live preview.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from smart_helmet_pano.bundle import BundleAdjuster, Problem, FIXED, FREE, BOARD, init_free_points, cv_to_ue
from smart_helmet_pano.rig import Rig, rot_ue
from smart_helmet_pano import targets as TG
from . import scenes as S
from .render import render_jobs, scene_key
from .truth import RigSpec, VARIANTS, nominal_rig, sample_truth

ARUCO = cv2.aruco
_CV2UE = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], float)   # v_ue_cam = M @ v_cv


@dataclass
class Ctx:
    spec: RigSpec
    seed: int
    nominal: Rig
    true: Rig
    out: Path
    workers: int = 7
    verbose: bool = True


def make_ctx(key, seed, out_root, workers=7, faults=None):
    spec = VARIANTS[key]
    nom = nominal_rig(spec)
    tru = sample_truth(nom, faults, seed) if faults else sample_truth(nom, seed=seed)
    out = Path(out_root) / f"{key}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    return Ctx(spec, seed, nom, tru, out, workers)


# ---------------------------------------------------------------- capture plans
PITCH = [0, 18, -18, 9, -9, 24, -14, 4, 12, -6, 20, -20]
ROLL = [0, 8, -8, 12, -4, -10, 6, 0, 3, -3, 10, -6]


def tripod_poses(ctx: Ctx, n, seed, rot_err_deg=1.0, pos_err_m=0.005):
    """Helmet on a pan-tilt tripod: commanded poses sweep two camera steps in yaw, pitch and roll vary.
    The true pose differs from the commanded one (tripod marks, helmet not exactly on the axis)."""
    rng = np.random.default_rng(seed)
    step = 360.0 / ctx.spec.ring
    poses = []
    for i in range(n):
        cmd = rot_ue(PITCH[i % len(PITCH)], i * step / 4, ROLL[i % len(ROLL)])
        R = cmd @ rot_ue(*rng.normal(0, rot_err_deg, 3))
        t = rng.normal(0, pos_err_m, 3)
        poses.append({"cmd_R": cmd, "cmd_t": np.zeros(3), "R_w": R, "t_w": t})
    return poses


def roaming_poses(n, seed, area_m=0.5):
    """Operator walking in a room (self-calibration frames)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        R = rot_ue(rng.uniform(-10, 10), rng.uniform(0, 360), rng.uniform(-6, 6))
        t = np.array([rng.uniform(-area_m, area_m), rng.uniform(-area_m, area_m), rng.uniform(-0.1, 0.1)])
        out.append({"cmd_R": R, "cmd_t": t, "R_w": R, "t_w": t})
    return out


def _jobs(ctx, poses, prefix, seed0):
    return {ci: [{"file": f"{prefix}{pi:02d}_{c.name}.jpg", "R_w": p["R_w"].tolist(), "t_w": list(map(float, p["t_w"])),
                  "seed": seed0 + 100 * pi + ci} for pi, p in enumerate(poses)]
            for ci, c in enumerate(ctx.true.cameras)}


def _load(ctx, sub, file):
    return cv2.imread(str(ctx.out / sub / file))


K_PRIOR = (0.05, 0.02, 0.01, 0.005)   # weak lens prior (k1..k4 ~ N(0, sigma)) for the "improved" projector stand
TANGENTIAL = False      # also estimate decentering p1, p2 (lens model KB + p1 p2), see Docs/CALIBRATION.md


def _finish(ctx, pb: Problem, name, t0, extra=None, passes=3, stage1=True):
    if TANGENTIAL and pb.est_k:
        pb.est_tangential = True
    ba = BundleAdjuster(pb)
    x = None
    if stage1:      # first rotations / poses / points with the datasheet lens, then everything
        e_f, e_pp, e_k, e_t = pb.est_f, pb.est_pp, pb.est_k, getattr(pb, "est_tangential", False)
        pb.est_f, pb.est_pp, pb.est_k, pb.est_tangential = False, False, 0, False
        ba1 = BundleAdjuster(pb)
        x1 = ba1.solve(passes=1, max_nfev=40, verbose=ctx.verbose, label=f" {name}/init")
        _, R, T, cR, cT, X, bR, bT = ba1.unpack(x1)
        pb.est_f, pb.est_pp, pb.est_k, pb.est_tangential = e_f, e_pp, e_k, e_t
        ba = BundleAdjuster(pb)
        ba.R0, ba.T0 = R, T
        ba.capR0, ba.capT0, ba.X, ba.bR0, ba.bT0 = cR, cT, X, bR, bT
    x = ba.solve(passes=passes, max_nfev=80, verbose=ctx.verbose, label=f" {name}")
    rig = ba.rig(x, name)
    info = {"method": name, "seconds": round(time.perf_counter() - t0, 1), **ba.report(x)}
    if extra:
        info.update(extra)
    return rig, info


def _compact(obs):
    """obs list (cam, cap, pid, u, v) -> arrays with compact point indices; returns map old->new."""
    a = np.array(obs, float)
    cam, cap, pid, uv = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int), a[:, 3:5]
    uniq, inv = np.unique(pid, return_inverse=True)
    return cam, cap, inv, uv, uniq


def _drop_single(cam, cap, pt, uv, kind_free):
    """Free points need >= 2 observations from different camera centres."""
    key = cam * 1000 + cap
    keep = np.ones(len(pt), bool)
    order = np.argsort(pt)
    bounds = np.searchsorted(pt[order], np.arange(pt.max() + 2))
    for p in range(pt.max() + 1):
        idx = order[bounds[p]:bounds[p + 1]]
        if kind_free[p] and len(np.unique(key[idx])) < 2:
            keep[idx] = False
    return keep


# ---------------------------------------------------------------- 1. marker points (ArUco stickers)
def run_markers(ctx: Ctx, surveyed=False, n_poses=8, refine=ARUCO.CORNER_REFINE_APRILTAG, sigma=0.7):
    """ArUco corner refinement: SUBPIX is precise but biased inwards (~0.3 px with lens blur), the AprilTag
    edge-fit refinement is unbiased (noisier); the bias does not average out, so AprilTag is the default."""
    t0 = time.perf_counter()
    key = scene_key("MarkerRoom")
    poses = tripod_poses(ctx, n_poses, ctx.seed * 10 + 1)
    render_jobs(key, ctx.true, _jobs(ctx, poses, "p", 1000), ctx.out / "markers", workers=ctx.workers)
    det = TG.aruco_detector(ARUCO.DICT_4X4_1000, refine=refine)
    obs, n_det = [], 0
    for pi in range(n_poses):
        for ci, c in enumerate(ctx.true.cameras):
            m = TG.detect_markers(_load(ctx, "markers", f"p{pi:02d}_{c.name}.jpg"), det)
            n_det += len(m)
            for mid, cc in m.items():
                for k in range(4):
                    obs.append((ci, pi, mid * 4 + k, cc[k, 0], cc[k, 1]))
    if surveyed:            # markers missing from the survey (false decodes) are ignored
        sc = S.MarkerRoom()
        obs = [o for o in obs if o[2] // 4 in sc.marker_truth]
    cam, cap, pt, uv, uniq = _compact(obs)
    P = len(uniq)
    if surveyed:
        rng = np.random.default_rng(ctx.seed + 7)
        Xs = np.array([sc.marker_truth[u // 4][u % 4] for u in uniq]) + rng.normal(0, 0.002, (P, 3))
        X0, sig = Xs, np.full(P, 0.002)
        cap_fixed = np.zeros(n_poses, bool)
        name = "markers_surveyed"
    else:
        keep = _drop_single(cam, cap, pt, uv, np.ones(P, bool))
        cam, cap, pt, uv = cam[keep], cap[keep], pt[keep], uv[keep]
        X0, _ = init_free_points(ctx.nominal, cam, cap, pt, uv, [p["cmd_R"] for p in poses],
                                 [p["cmd_t"] for p in poses], P)
        sig = None
        cap_fixed = np.zeros(n_poses, bool); cap_fixed[0] = True
        name = "markers"
    # marker size prior: 4 edges of every marker whose 4 corners are all known
    pairs = []
    lut = {u: i for i, u in enumerate(uniq)}
    for u in uniq[uniq % 4 == 0]:
        m = u // 4
        ids = [lut.get(m * 4 + k) for k in range(4)]
        if all(i is not None for i in ids):
            pairs += [(ids[k], ids[(k + 1) % 4]) for k in range(4)]
    pairs = np.array(pairs, int).reshape(-1, 2)
    pb = Problem(ctx.nominal, cam, cap, pt, uv, [p["cmd_R"] for p in poses], [p["cmd_t"] for p in poses],
                 np.full(P, FREE), X0, -np.ones(P, int), cap_fixed=cap_fixed,
                 pt_prior_sigma=sig, pt_prior_X=X0.copy() if surveyed else None,
                 dist_pairs=pairs, dist_value=np.full(len(pairs), 0.15), dist_sigma=0.002,
                 obs_sigma=np.full(len(cam), sigma))
    return _finish(ctx, pb, name, t0, {"captures": n_poses, "markers_detected": n_det, "points": P})


# ---------------------------------------------------------------- 2. chessboards projected on the walls
def run_projector(ctx: Ctx, n_poses=10, scale_bars=False):
    """scale_bars: the width of every projected board (first and last corner row) is measured with a tape
    (sigma 3 mm) - the only metric information, otherwise the scale rests on the CAD lens centres; together
    with a weak lens prior (K_PRIOR) that keeps the polynomial sane in image corners no board reached."""
    t0 = time.perf_counter()
    key = scene_key("ProjectorRoom")
    sc = S.ProjectorRoom()
    poses = tripod_poses(ctx, n_poses, ctx.seed * 10 + 2)
    render_jobs(key, ctx.true, _jobs(ctx, poses, "p", 2000), ctx.out / "projector", workers=ctx.workers)
    rng = np.random.default_rng(ctx.seed + 3)
    rough = {b: P.mean(0) + rng.normal(0, 0.25, 3) for b, P in sc.board_truth.items()}   # "the board is about there"
    inner = (sc.squares[0] - 1, sc.squares[1] - 1)
    npb = inner[0] * inner[1]
    obs, n_det, n_unid = [], 0, 0
    for pi, p in enumerate(poses):
        for ci, c in enumerate(ctx.nominal.cameras):
            img = _load(ctx, "projector", f"p{pi:02d}_{c.name}.jpg")
            for corners in TG.detect_chessboards(img, inner, 3):
                ray = c.unproject(*corners.mean(0)[:, None])[0]
                dw = p["cmd_R"] @ ray
                best, ang = None, 1e9
                for b, X in rough.items():
                    v = X - p["cmd_t"]; v /= np.linalg.norm(v)
                    a = math.degrees(math.acos(np.clip(v @ dw, -1, 1)))
                    if a < ang:
                        best, ang = b, a
                if ang > 20:
                    n_unid += 1; continue
                n_det += 1
                for k, (u, v) in enumerate(corners):
                    obs.append((ci, pi, best * npb + k, u, v))
    cam, cap, pt, uv, uniq = _compact(obs)
    P = len(uniq)
    keep = _drop_single(cam, cap, pt, uv, np.ones(P, bool))
    cam, cap, pt, uv = cam[keep], cap[keep], pt[keep], uv[keep]
    X0, _ = init_free_points(ctx.nominal, cam, cap, pt, uv, [p["cmd_R"] for p in poses], [p["cmd_t"] for p in poses], P)
    planes = [np.flatnonzero(uniq // npb == b) for b in range(len(sc.projectors))]
    planes = [q for q in planes if len(q) >= 6]
    cap_fixed = np.zeros(n_poses, bool); cap_fixed[0] = True
    pairs, dist = [], []
    if scale_bars:
        lut = {int(u): i for i, u in enumerate(uniq)}
        tape = np.random.default_rng(ctx.seed + 11)
        for b, X in sc.board_truth.items():
            for a, c in ((0, inner[0] - 1), (npb - inner[0], npb - 1)):
                if b * npb + a in lut and b * npb + c in lut:
                    pairs.append((lut[b * npb + a], lut[b * npb + c]))
                    dist.append(np.linalg.norm(X[a] - X[c]) + tape.normal(0, 0.003))
    pb = Problem(ctx.nominal, cam, cap, pt, uv, [p["cmd_R"] for p in poses], [p["cmd_t"] for p in poses],
                 np.full(P, FREE), X0, -np.ones(P, int), cap_fixed=cap_fixed, planes=planes, plane_sigma=0.003,
                 obs_sigma=np.full(len(cam), 0.4),
                 dist_pairs=np.array(pairs, int).reshape(-1, 2) if pairs else None,
                 dist_value=np.array(dist) if dist else None, dist_sigma=0.003,
                 k_prior_sigma=K_PRIOR if scale_bars else None)
    return _finish(ctx, pb, "projector_scaled" if scale_bars else "projector", t0,
                   {"captures": n_poses, "boards_detected": n_det, "boards_unidentified": n_unid, "points": P,
                    "scale_bars": len(pairs)})


# ---------------------------------------------------------------- 3. calibration cube (surveyed ChArUco room)
def run_cube(ctx: Ctx, n_poses=4, trust_build=True):
    """trust_build=True: corners taken from the drawing of the cube (survey = design, build errors ignored);
    False: every face is a rigid board with a precisely printed layout but unknown pose (estimated)."""
    t0 = time.perf_counter()
    key = scene_key("CubeRoom")
    sc = S.CubeRoom()
    poses = tripod_poses(ctx, n_poses, ctx.seed * 10 + 3)
    render_jobs(key, ctx.true, _jobs(ctx, poses, "p", 3000), ctx.out / "cube", workers=ctx.workers)
    boards = [b for _, b in sc.boards]
    faces = [k for k, _ in sc.boards]
    det = TG.aruco_detector(ARUCO.DICT_5X5_1000)
    ncorner = len(boards[0].getChessboardCorners())
    obs = []
    for pi in range(n_poses):
        for ci, c in enumerate(ctx.nominal.cameras):
            for bi, ids, cc in TG.detect_charuco(_load(ctx, "cube", f"p{pi:02d}_{c.name}.jpg"), boards, det):
                for i, (u, v) in zip(ids, cc):
                    obs.append((ci, pi, bi * ncorner + i, u, v))
    cam, cap, pt, uv, uniq = _compact(obs)
    P = len(uniq)
    if trust_build:
        X = np.array([sc.nominal_pts[faces[u // ncorner]][u % ncorner] for u in uniq])
        pb = Problem(ctx.nominal, cam, cap, pt, uv, [p["cmd_R"] for p in poses], [p["cmd_t"] for p in poses],
                     np.full(P, FIXED), X, -np.ones(P, int), cap_fixed=np.zeros(n_poses, bool),
                     obs_sigma=np.full(len(cam), 0.3))
        name = "cube_design"
    else:
        loc = boards[0].getChessboardCorners().astype(np.float64)
        q = {qq.name: qq for qq in S.box_quads(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5,
                                               lambda k, w, h: np.zeros((2, 2, 3), np.uint8))}
        bR = [np.stack([q[f].e1, q[f].e2, q[f].n], 1) for f in faces]
        bT = [q[f].O for f in faces]
        cap_fixed = np.zeros(n_poses, bool); cap_fixed[0] = True
        pb = Problem(ctx.nominal, cam, cap, pt, uv, [p["cmd_R"] for p in poses], [p["cmd_t"] for p in poses],
                     np.full(P, BOARD), loc[uniq % ncorner], uniq // ncorner, board_R=bR, board_t=bT,
                     cap_fixed=cap_fixed, obs_sigma=np.full(len(cam), 0.3))
        name = "cube_panels"
    return _finish(ctx, pb, name, t0, {"captures": n_poses, "corners": int(len(cam)), "points": P})


# ---------------------------------------------------------------- 4. handheld ChArUco board (classic)
BOARD_ARGS = [(11, 8), 0.05, 0.75, ARUCO.DICT_5X5_1000, 0]


def _board_pose(fwd, up_hint):
    q = S.facing_quad(np.zeros(3), fwd, up_hint, 1, 1, np.zeros((2, 2, 3), np.uint8))
    return np.stack([q.e1, q.e2, q.n], 1)


def plan_board(ctx: Ctx, singles=14, pairs=5, seed=0):
    """Operator holds the board in front of the static helmet while watching the live view:
    `singles` placements covering each image (edges included), `pairs` in every neighbour overlap."""
    rng = np.random.default_rng(seed)
    board = S.charuco_board(*BOARD_ARGS)
    loc = board.getChessboardCorners()
    sx, sy = board.getChessboardSize(); sq = board.getSquareLength()
    ctr_local = np.array([sx * sq / 2, sy * sq / 2, 0])
    tru = ctx.true.cameras

    def visible(R, t):
        X = loc @ R.T + t
        n = R[:, 2]
        out = {}
        for ci, c in enumerate(tru):
            u, v, ok = c.project(X)
            to_cam = np.asarray(c.position_m) - (R @ ctr_local + t)
            if np.dot(n, to_cam) > 0 and ok.sum() >= 10:   # printed side faces this camera (n points to the viewer)
                out[ci] = int(ok.sum())
        return out

    def tilt(d, max_deg):
        a = rng.normal(0, max_deg / 2, 3)
        return rot_ue(*a) @ d

    placements = []
    ring = [i for i, c in enumerate(tru) if not c.name.startswith("top")]
    top = len(tru) - 1
    for ci, c in enumerate(tru):
        got, tries = 0, 0
        while got < singles and tries < 400:
            tries += 1
            u = rng.uniform(0.06, 0.94) * c.width
            v = rng.uniform(0.04, 0.96) * c.height
            d = c.unproject(np.array([u]), np.array([v]))[0]
            D = rng.uniform(0.45, 1.0)
            C = np.asarray(c.position_m) + D * d
            up = np.asarray(c.R_rig_from_cam_ue)[:, 2]
            R = _board_pose(tilt(d, 50), rot_ue(*rng.normal(0, 25, 3)) @ up)
            t = C - R @ ctr_local
            vis = visible(R, t)
            if vis.get(ci, 0) >= 20:
                placements.append({"R": R, "t": t, "vis": vis, "target": [ci]})
                got += 1
    nb = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))] + [(r, top) for r in ring]
    dirs = rng.normal(0, 1, (20000, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for a, b in nb:
        ok = tru[a].project(dirs * 50)[2] & tru[b].project(dirs * 50)[2]
        cand = dirs[ok]
        got, tries = 0, 0
        while got < pairs and tries < 400 and len(cand):
            tries += 1
            d = cand[rng.integers(len(cand))]
            D = rng.uniform(0.7, 1.4)
            R = _board_pose(tilt(d, 30), rot_ue(*rng.normal(0, 25, 3)) @ np.array([0, 0, 1.0]) if b != top else
                            rot_ue(*rng.normal(0, 25, 3)) @ np.array([1.0, 0, 0]))
            t = D * d - R @ ctr_local
            vis = visible(R, t)
            if vis.get(a, 0) >= 12 and vis.get(b, 0) >= 12:
                placements.append({"R": R, "t": t, "vis": vis, "target": [a, b]})
                got += 1
    return placements


def run_charuco(ctx: Ctx, singles=14, pairs=5):
    t0 = time.perf_counter()
    pl = plan_board(ctx, singles, pairs, seed=ctx.seed * 10 + 4)
    jobs = {ci: [] for ci in range(len(ctx.true.cameras))}
    for k, p in enumerate(pl):
        for ci in p["vis"]:
            jobs[ci].append({"file": f"b{k:03d}_{ctx.true.cameras[ci].name}.jpg", "R_w": np.eye(3).tolist(),
                             "t_w": [0.0, 0.0, 0.0], "seed": 4000 + 10 * k + ci,
                             "board_R": p["R"].tolist(), "board_t": p["t"].tolist()})
    spec = {"args": BOARD_ARGS, "px": [2200, 1600]}
    render_jobs(scene_key("NaturalRoom"), ctx.true, jobs, ctx.out / "charuco", board_spec=spec, workers=ctx.workers)
    board = S.charuco_board(*BOARD_ARGS)
    loc = board.getChessboardCorners().astype(np.float64)
    nco = len(loc)
    det = TG.aruco_detector(ARUCO.DICT_5X5_1000)
    obs = []
    per_pl = {}
    for ci, items in jobs.items():
        for it in items:
            k = int(it["file"][1:4])
            for _, ids, cc in TG.detect_charuco(_load(ctx, "charuco", it["file"]), [board], det):
                if len(ids) < 6:
                    continue
                per_pl.setdefault(k, []).append((ci, ids, cc))
                for i, (u, v) in zip(ids, cc):
                    obs.append((ci, 0, k * nco + i, u, v))
    # board pose init: PnP in the camera with most corners, nominal lens, nominal mounting
    used = sorted(per_pl)
    bidx = {k: j for j, k in enumerate(used)}
    bR, bT = [], []
    for k in used:
        ci, ids, cc = max(per_pl[k], key=lambda e: len(e[1]))
        c = ctx.nominal.cameras[ci]
        und = cv2.fisheye.undistortPoints(cc.reshape(-1, 1, 2), c.K, c.D).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(loc[ids], und, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        Rp = cv2.Rodrigues(rv)[0]
        Rc = np.asarray(c.R_rig_from_cam_ue)
        bR.append(Rc @ _CV2UE @ Rp)
        bT.append(Rc @ _CV2UE @ tv.ravel() + np.asarray(c.position_m))
    a = np.array(obs, float)
    cam, cap = a[:, 0].astype(int), a[:, 1].astype(int)
    k_of = (a[:, 2].astype(int) // nco)
    i_of = (a[:, 2].astype(int) % nco)
    pt = np.array([bidx[k] for k in k_of]) * nco + i_of
    uv = a[:, 3:5]
    P = len(used) * nco
    pt_X = np.tile(loc, (len(used), 1))
    pt_board = np.repeat(np.arange(len(used)), nco)
    pb = Problem(ctx.nominal, cam, cap, pt, uv, [np.eye(3)], [np.zeros(3)], np.full(P, BOARD), pt_X, pt_board,
                 board_R=bR, board_t=bT, cap_fixed=np.array([True]), obs_sigma=np.full(len(cam), 0.3))
    n_pair = sum(len(p["target"]) == 2 for p in pl)
    return _finish(ctx, pb, "charuco", t0, {"placements": len(pl), "pair_placements": n_pair,
                                            "images": sum(len(v) for v in jobs.values()), "corners": int(len(cam))})


# ---------------------------------------------------------------- 5. self-calibration on natural features (production)
def render_natural(ctx: Ctx, n=8):
    poses = roaming_poses(n, ctx.seed * 10 + 5)
    render_jobs(scene_key("NaturalRoom"), ctx.true, _jobs(ctx, poses, "f", 5000), ctx.out / "natural",
                workers=ctx.workers)
    names = [c.name for c in ctx.true.cameras]
    return poses, [[_load(ctx, "natural", f"f{pi:02d}_{nm}.jpg") for nm in names] for pi in range(n)]


def run_selfcal(ctx: Ctx, n=8):
    from smart_helmet_pano.calibrate import calibrate
    t0 = time.perf_counter()
    _, frames = render_natural(ctx, n)
    names = [c.name for c in ctx.true.cameras]
    rig, rep = calibrate(frames, names, base=ctx.nominal, f_init=ctx.nominal.cameras[0].fx, verbose=False,
                         per_camera_lens=True)
    for c, cn in zip(rig.cameras, ctx.nominal.cameras):
        c.max_theta_deg = cn.max_theta_deg
    return rig, {"method": "selfcal", "seconds": round(time.perf_counter() - t0, 1), "frames": n,
                 "rms_px": rep["rms_px_calibrated"], "median_px": rep["median_px_calibrated"],
                 "matches": rep["matches"]}


# ---------------------------------------------------------------- 6. hybrid: factory lens + field rotations
def run_hybrid(ctx: Ctx, lens_rig: Rig, n=8, max_per_pair=40, frames=None, name="hybrid"):
    """Intrinsics fixed from a lab calibration (e.g. handheld ChArUco); mounting rotations / centres
    re-estimated from natural features of ordinary frames (field re-calibration after knocks)."""
    from smart_helmet_pano.calibrate import _features, _match, _ransac_rotation
    t0 = time.perf_counter()
    if frames is None:
        _, frames = render_natural(ctx, n)
    base = Rig("lens", [c for c in lens_rig.cameras])
    rng = np.random.default_rng(0)
    obs, pid = [], 0
    for fi, imgs in enumerate(frames):
        feats = [_features(im) for im in imgs]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                m = _match(feats[i][1], feats[j][1])
                if len(m) < 15:
                    continue
                pa, pb_ = feats[i][0][m[:, 0]], feats[j][0][m[:, 1]]
                ra = base.cameras[i].unproject(pa[:, 0], pa[:, 1])
                rb = base.cameras[j].unproject(pb_[:, 0], pb_[:, 1])
                R, inl = _ransac_rotation(ra, rb, thresh_deg=1.5)
                if inl.sum() < 15:
                    continue
                sel = np.flatnonzero(inl)
                if len(sel) > max_per_pair:
                    sel = rng.choice(sel, max_per_pair, replace=False)
                for s in sel:
                    obs.append((i, fi, pid, pa[s, 0], pa[s, 1]))
                    obs.append((j, fi, pid, pb_[s, 0], pb_[s, 1]))
                    pid += 1
    a = np.array(obs, float)
    cam, cap, pt, uv = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int), a[:, 3:5]
    nf = len(frames)
    X0, _ = init_free_points(base, cam, cap, pt, uv, [np.eye(3)] * nf, [np.zeros(3)] * nf, pid, default_depth=2.5)
    pb = Problem(base, cam, cap, pt, uv, [np.eye(3)] * nf, [np.zeros(3)] * nf, np.full(pid, FREE), X0,
                 -np.ones(pid, int), cap_fixed=np.ones(nf, bool), est_f=False, est_pp=False, est_k=0,
                 obs_sigma=np.full(len(cam), 0.8), t_prior_m=0.002)
    return _finish(ctx, pb, name, t0, {"frames": nf, "matches": pid}, stage1=False)
