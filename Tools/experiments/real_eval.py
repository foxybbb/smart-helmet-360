"""All research directions on REAL helmet photos (no ground truth -> seam metrics + visual sheets).

  python experiments/real_eval.py [session] [n_frames] [width] [variants...]
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from directions import DisFlow, RaftFlow, MonoDepth, run_variant  # noqa: E402
from smart_helmet_pano.cli import frame_dirs  # noqa: E402
from smart_helmet_pano.rig import Rig  # noqa: E402
from smart_helmet_pano.stitch import load_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    session = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Output" / "real_helmet"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    variants = sys.argv[4:] or ["base", "base_pc", "N5_rot_pc", "N1_spline_pc", "N3_tangent_pc", "N2_raft_pc",
                                "N4_depth_mono_pc", "N4_depth_mono_flow_pc"]
    out = ROOT / "Output" / "real_eval"
    out.mkdir(parents=True, exist_ok=True)
    rig = Rig.load(session / "rig.json")
    pc_path = session / "rig_percam.json"
    if not pc_path.exists():
        from smart_helmet_pano.calibrate import calibrate_session
        names = [c.name for c in rig.cameras]
        t = time.perf_counter()
        _, rep = calibrate_session(session, names=names, n_frames=16, f_init=680.0, out_rig=pc_path, verbose=True,
                                   per_camera_lens=True)
        rep["seconds"] = round(time.perf_counter() - t, 1)
        (out / "calibration_percam.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print("per-camera calibration:", {k: rep[k] for k in ("rms_px_calibrated", "median_px_calibrated", "focal_px")})
    rig_pc = Rig.load(pc_path)

    flows = {"DIS": DisFlow()}
    if any("raft" in v for v in variants):
        flows["RAFT"] = RaftFlow()
    mono = MonoDepth() if any("mono" in v for v in variants) else None
    fds = frame_dirs(session)
    fds = [fds[i] for i in np.linspace(0, len(fds) - 1, min(n, len(fds))).astype(int)]
    rows = {v: [] for v in variants}
    for fd in fds:
        for v in variants:
            r = rig_pc if v.endswith("_pc") else rig
            name = "base" if v.startswith("base") else v.replace("_pc", "")
            imgs, meta = load_frame(fd, r)
            try:
                pano, res = run_variant(name, r, imgs, meta, True, W, flows, mono=mono)
            except Exception as e:  # noqa: BLE001
                print(fd.name, v, "FAILED", repr(e)); continue
            cv2.imwrite(str(out / f"{fd.name}_{v}.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows[v].append(res)
            print(fd.name, v, json.dumps({k: res[k] for k in ("seam_artifacts_%", "seam_MAD", "seconds")}), flush=True)
    summary = {v: {k: round(float(np.mean([r[k] for r in rs])), 3) for k in ("seam_artifacts_%", "seam_MAD", "seconds")}
               for v, rs in rows.items() if rs}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
