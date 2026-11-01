"""Evaluate calibration + the 5 research directions on a UE imperfection session against ground truth.

  python experiments/ue_eval.py <session_dir> [width] [variants...]
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from directions import DisFlow, RaftFlow, MonoDepth, run_variant  # noqa: E402
from smart_helmet_pano.cli import frame_dirs  # noqa: E402
from smart_helmet_pano.rig import Rig, Camera, rot_ue  # noqa: E402
from smart_helmet_pano.stitch import StitchOptions, load_frame, equirect_dirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def ref_equirect(frame_dir, W):
    """World-aligned ground-truth panorama from the 6 reference cube faces."""
    faces = [cv2.imread(str(Path(frame_dir) / f"ref_face{f}.png")) for f in range(6)]
    N = faces[0].shape[0]
    rots = [(0, 0, 0), (0, 90, 0), (0, -90, 0), (90, 0, 0), (-90, 0, 0), (0, 180, 0)]
    d = equirect_dirs(W, W // 2).astype(np.float64)
    out = np.zeros((W // 2, W, 3), np.uint8)
    dots = np.stack([d @ rot_ue(*r)[:, 0] for r in rots])
    best = np.argmax(dots, 0)
    for f, r in enumerate(rots):
        M = rot_ue(*r)
        sel = best == f
        if not sel.any():
            continue
        dd = d[sel]
        fw = dd @ M[:, 0]
        a = dd @ M[:, 1] / fw
        b = dd @ M[:, 2] / fw
        u = ((a + 1) * 0.5 * N - 0.5).astype(np.float32)
        v = ((1 - b) * 0.5 * N - 0.5).astype(np.float32)
        pad = (-len(u)) % 1024
        uu = np.pad(u, (0, pad)).reshape(-1, 1024); vv = np.pad(v, (0, pad)).reshape(-1, 1024)
        out[sel] = cv2.remap(faces[f], uu, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)[:len(u)]
    return out


def oracle_rig(session):
    d = json.loads((Path(session) / "rig.json").read_text())
    cams = []
    for c in d["cameras"]:
        cams.append(Camera(c["name"], c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"], k=c["true_k"],
                           position_m=c["position_m"], rotation_pyr_deg=c["true_rotation_pyr_deg"]))
    return Rig("oracle", cams), d


def load_selfcal(session, nominal, oracle, percam, out):
    """Self-calibrated rig (shared or per-camera lens), rotated into the oracle frame (a global rig rotation
    does not affect seams, but matters for the comparison with the world-aligned ground truth)."""
    from smart_helmet_pano.calibrate import calibrate_session
    p = session / ("rig_selfcal_percam.json" if percam else "rig_selfcal.json")
    if not p.exists():
        names = [c.name for c in nominal.cameras]
        t = time.perf_counter()
        _, rep = calibrate_session(session, names=names, n_frames=8, f_init=nominal.cameras[0].fx, out_rig=p,
                                   verbose=False, per_camera_lens=percam)
        rep["seconds"] = round(time.perf_counter() - t, 1)
        (out / f"selfcal{'_percam' if percam else ''}_report.json").write_text(json.dumps(rep, indent=1))
    sc = Rig.load(p)
    by = {c.name: c for c in sc.cameras}
    A = sum(c.R_rig_from_cam_ue @ by[c.name].R_rig_from_cam_ue.T for c in oracle.cameras)
    U, _, Vt = np.linalg.svd(A)
    G = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    cams = []
    for c in oracle.cameras:
        d = dict(by[c.name].__dict__)
        d["rotation_matrix"] = (G @ by[c.name].R_rig_from_cam_ue).tolist()
        d["position_m"] = c.position_m
        cams.append(Camera(**d))
    return Rig("selfcal_percam" if percam else "selfcal", cams)


def rot_err_deg(Ra, Rb):
    return math.degrees(np.linalg.norm(cv2.Rodrigues(Ra.T @ Rb)[0]))


def main():
    session = Path(sys.argv[1])
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    variants = sys.argv[3:] or ["base_nominal", "base_oracle", "base_selfcal", "N5_rot", "N1_spline", "N3_tangent",
                                "N2_raft", "N4_depth", "N4_depth_mono"]
    out = ROOT / "Output" / "ue_eval"
    out.mkdir(parents=True, exist_ok=True)
    nominal = Rig.load(session / "rig.json")
    oracle, raw = oracle_rig(session)

    selfcal = load_selfcal(session, nominal, oracle, percam=False, out=out)
    selfcal_pc = load_selfcal(session, nominal, oracle, percam=True, out=out)
    calib = {c.name: {"nominal_err_deg": round(rot_err_deg(n.R_rig_from_cam_ue, c.R_rig_from_cam_ue), 2),
                      "selfcal_err_deg": round(rot_err_deg(s.R_rig_from_cam_ue, c.R_rig_from_cam_ue), 2),
                      "selfcal_percam_err_deg": round(rot_err_deg(sp.R_rig_from_cam_ue, c.R_rig_from_cam_ue), 2),
                      "true_k1": c.k[0], "selfcal_k1": round(s.k[0], 4)}
             for n, s, sp, c in zip(nominal.cameras, selfcal.cameras, selfcal_pc.cameras, oracle.cameras)}
    print("calibration vs truth:", json.dumps(calib))

    flows = {"DIS": DisFlow()}
    if any("raft" in v for v in variants):
        flows["RAFT"] = RaftFlow()
    mono = MonoDepth() if any("mono" in v for v in variants) else None

    o = StitchOptions(width=W, level="full", proxy_depth_m=4.0, theta_max_deg=75, theta_ramp_deg=75)
    lat = (0.5 - (np.arange(W // 2) + 0.5) / (W // 2)) * 180
    gt_mask = np.repeat((lat > -50)[:, None], W, 1)          # ignore the operator's body / nadir
    rows = {v: [] for v in variants}
    for fd in frame_dirs(session):
        gt = ref_equirect(fd, W)
        cv2.imwrite(str(out / f"{fd.name}_GT.jpg"), gt)
        for v in variants:
            rig = {"nominal": nominal, "oracle": oracle, "pc": selfcal_pc}.get(v.split("_")[-1], selfcal)
            imgs, meta = load_frame(fd, rig)
            name = "base" if v.startswith("base") else v.replace("_pc", "")
            try:
                pano, res = run_variant(name, rig, imgs, meta, False, W, flows, mono=mono, gt=gt, opts=o, gt_mask=gt_mask)
            except Exception as e:  # noqa: BLE001
                print(v, fd.name, "FAILED", e)
                continue
            cv2.imwrite(str(out / f"{fd.name}_{v}.jpg"), pano)
            rows[v].append(res)
            print(fd.name, v, json.dumps({k: res[k] for k in ("PSNR_vs_GT", "SSIM_vs_GT", "seam_artifacts_%", "seam_MAD", "seconds")}))
    summary = {}
    for v, rs in rows.items():
        if rs:
            summary[v] = {k: round(float(np.mean([r[k] for r in rs])), 3)
                          for k in ("PSNR_vs_GT", "SSIM_vs_GT", "seam_artifacts_%", "seam_MAD", "seconds")}
    (out / "summary.json").write_text(json.dumps({"calibration": calib, "variants": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
