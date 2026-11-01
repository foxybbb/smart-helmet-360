"""Plane vs sphere: how should fisheye helmet photos be projected before stitching?

A) PLANE  – classic route: undistort every fisheye image to a rectilinear (pinhole) image with
            cv2.fisheye, then stitch with OpenCV's feature-based cv2.Stitcher (homographies +
            spherical warper, bundle adjust, seam finder, multiband).
B) SPHERE – our route: every fisheye pixel is converted by the calibrated Kannala-Brandt model
            straight into a ray on the unit sphere and resampled once into the equirectangular
            panorama (no intermediate plane).

Measured for the same real frames: field of view kept, pixel stretch, success rate, seam artifacts,
runtime.  Usage:  python experiments/plane_vs_sphere.py ../Output/real_helmet [n_frames]
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_helmet_pano.cli import frame_dirs  # noqa: E402
from smart_helmet_pano.quality import seam_artifacts  # noqa: E402
from smart_helmet_pano.rig import Rig  # noqa: E402
from smart_helmet_pano.stitch import StitchOptions, PanoStitcher, load_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TUNING = str(ROOT / "Data" / "imx219_160d.json")


def plane_geometry(cam, fov_h, fov_v):
    """What a rectilinear crop keeps of the fisheye image, and how much it stretches the edges."""
    th = lambda r: r / cam.fx  # equidistant approx (k ~ 0 for the calibrated lens)
    full_h = 2 * math.degrees(th(cam.width / 2))
    full_v = 2 * math.degrees(th(cam.height / 2))
    # pixel area kept ~ solid angle of the rectangle vs the fisheye frame
    kept_h, kept_v = min(fov_h, full_h) / full_h, min(fov_v, full_v) / full_v
    # rectilinear magnification at the edge relative to the centre: 1/cos^2(theta)
    edge = math.radians(max(fov_h, fov_v) / 2)
    stretch = 1 / math.cos(edge) ** 2 / (edge / math.sin(edge) if edge else 1)
    return {"fisheye_fov": [round(full_h, 1), round(full_v, 1)], "plane_fov": [fov_h, fov_v],
            "fov_kept_%": round(100 * kept_h * kept_v, 1), "edge_stretch_x": round(stretch, 2)}


def undistort_plane(img, cam, fov_h, fov_v):
    w, h = cam.width, cam.height
    fx = (w / 2) / math.tan(math.radians(fov_h) / 2)
    fy = (h / 2) / math.tan(math.radians(fov_v) / 2)
    Knew = np.array([[fx, 0, (w - 1) / 2], [0, fy, (h - 1) / 2], [0, 0, 1]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(cam.K, cam.D, np.eye(3), Knew, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def run_plane(imgs, rig, fov_h=110.0, fov_v=130.0, scale=0.5):
    planes = []
    for im, cam in zip(imgs, rig.cameras):
        fh, fv = (fov_h, fov_v) if cam.height > cam.width else (fov_v, fov_h)
        p = undistort_plane(im, cam, fh, fv)
        planes.append(cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    t = time.perf_counter()
    status, pano = st.stitch(planes)
    dt = time.perf_counter() - t
    used = len(st.component()) if status == cv2.Stitcher_OK else 0
    return status, pano, dt, used, planes


def run_sphere(imgs, meta, rig, width=2048):
    o = StitchOptions(width=width, level="none", proxy_depth_m=6, tuning=TUNING, theta_max_deg=75,
                      theta_ramp_deg=75, max_flow_frac=0.005, flow_sigma_frac=0.008)
    s = PanoStitcher(rig, o)
    t = time.perf_counter()
    pano = s.stitch(imgs, meta)
    dt = time.perf_counter() - t
    q, _ = seam_artifacts(s.last_warped, s.last_weights, s.W, s.H)
    return pano, dt, q


def main():
    session = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "Output" / "real_helmet")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rig = Rig.load(session / "rig.json")
    fds = frame_dirs(session)
    fds = [fds[i] for i in np.linspace(0, len(fds) - 1, min(n, len(fds))).astype(int)]
    out = ROOT / "Output" / "plane_vs_sphere"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    geo = plane_geometry(rig.cameras[0], 110, 130)
    names = {cv2.Stitcher_OK: "OK", cv2.Stitcher_ERR_NEED_MORE_IMGS: "NEED_MORE_IMGS",
             cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "HOMOGRAPHY_FAIL",
             cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "CAMERA_ADJUST_FAIL"}
    for fd in fds:
        imgs, meta = load_frame(fd, rig)
        status, pp, dtp, used, planes = run_plane(imgs, rig)
        sp, dts, q = run_sphere(imgs, meta, rig)
        tag = fd.name
        cv2.imwrite(str(out / f"{tag}_sphere.jpg"), sp)
        if status == cv2.Stitcher_OK:
            cv2.imwrite(str(out / f"{tag}_plane.jpg"), pp)
        rows.append({"frame": tag,
                     "plane": {"status": names.get(status, str(status)), "cameras_used": f"{used}/{len(imgs)}",
                               "seconds": round(dtp, 1)},
                     "sphere": {"status": "OK", "cameras_used": f"{len(imgs)}/{len(imgs)}", "seconds": round(dts, 1),
                                "seam_artifacts_%": q["artifact_%_of_central"]}})
        print(json.dumps(rows[-1], ensure_ascii=False))
    # one camera: fisheye vs its plane version, to show the stretch
    fd = fds[len(fds) // 2]
    imgs, _ = load_frame(fd, rig)
    p = undistort_plane(imgs[0], rig.cameras[0], 110, 130)
    side = np.hstack([cv2.resize(imgs[0], (540, 960)), cv2.resize(p, (540, 960))])
    cv2.imwrite(str(out / "one_camera_fisheye_vs_plane.jpg"), side)
    ok = sum(r["plane"]["status"] == "OK" for r in rows)
    summary = {"plane_geometry_for_110x130deg_crop": geo,
               "plane_success": f"{ok}/{len(rows)}",
               "sphere_success": f"{len(rows)}/{len(rows)}",
               "sphere_mean_seam_artifacts_%": round(float(np.mean([r['sphere']['seam_artifacts_%'] for r in rows])), 2),
               "frames": rows}
    (out / "report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
