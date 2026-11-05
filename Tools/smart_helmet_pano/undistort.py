"""Fisheye undistortion – test programme sec. 3.2, steps 1-2.

Converts one camera image to a rectilinear (pinhole) view with cv2.fisheye.
A 141 deg lens cannot be fully rectified, so `fov_h_deg`/`fov_v_deg` choose the
rectilinear crop.  `sweep_balance` evaluates the straight-line error for a set
of crops, which is how the optimal parameter set is chosen in step 2.
"""
import math

import cv2
import numpy as np

from .rig import Camera


def rectilinear_K(fov_h_deg, fov_v_deg, w, h):
    fx = (w / 2) / math.tan(math.radians(fov_h_deg) / 2)
    fy = (h / 2) / math.tan(math.radians(fov_v_deg) / 2)
    return np.array([[fx, 0, (w - 1) / 2], [0, fy, (h - 1) / 2], [0, 0, 1]], float)


def undistort(img, cam: Camera, fov_h_deg=75.0, fov_v_deg=110.0, out_size=None):
    w, h = out_size or (cam.width, cam.height)
    Knew = rectilinear_K(fov_h_deg, fov_v_deg, w, h)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(cam.K, cam.D, np.eye(3), Knew, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), Knew


def line_straightness_error(img) -> float:
    """Mean deviation (px) of detected long edges from straight lines; lower = better."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(g, 60, 160)
    n, lab = cv2.connectedComponents(edges)
    errs = []
    for i in range(1, min(n, 400)):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 80:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        errs.append(float(np.median(d)))
    return float(np.mean(errs)) if errs else float("nan")


def sweep_fov(img, cam: Camera, fovs=((60, 90), (70, 100), (75, 110), (80, 120), (85, 130))):
    rows = []
    for fh, fv in fovs:
        und, _ = undistort(img, cam, fh, fv)
        valid = (und.sum(2) > 0).mean() if und.ndim == 3 else (und > 0).mean()
        rows.append({"fov_h": fh, "fov_v": fv, "line_err_px": line_straightness_error(und),
                     "filled_%": round(100 * valid, 1)})
    return rows
