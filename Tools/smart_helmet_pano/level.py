"""Horizon levelling of an already stitched equirectangular panorama.

Direct implementation of the algorithm in the "3D Engineering" stage-2 report,
sec. 4.2.1 (inverse ray tracing):  for every pixel of the ideal output grid
  lambda = (u - 0.5) * 2pi,  phi = (0.5 - v) * pi
  P = (cos phi cos lambda, cos phi sin lambda, sin phi)
  P_src = R(q) P                     (q = BNO055 orientation quaternion)
  u_src = atan2(y', x') / 2pi + 0.5,  v_src = 0.5 - asin(z') / pi
The stitcher (stitch.py) folds the same rotation into its projection LUT, which
avoids a second interpolation; this module is for panoramas made elsewhere.
"""
import numpy as np
import cv2

from .rig import quat_to_matrix


def rotate_equirect(pano: np.ndarray, R: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = pano.shape[:2]
    u = (np.arange(w, dtype=np.float32) + 0.5) / w
    v = (np.arange(h, dtype=np.float32) + 0.5) / h
    u, v = np.meshgrid(u, v)
    lam = (u - 0.5) * 2 * np.pi
    phi = (0.5 - v) * np.pi
    P = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], -1)
    Ps = P @ R.T
    lam_s = np.arctan2(Ps[..., 1], Ps[..., 0])
    phi_s = np.arcsin(np.clip(Ps[..., 2], -1, 1))
    map_x = ((lam_s / (2 * np.pi) + 0.5) * w - 0.5).astype(np.float32)
    map_y = ((0.5 - phi_s / np.pi) * h - 0.5).astype(np.float32)
    return cv2.remap(pano, map_x, map_y, interp, borderMode=cv2.BORDER_WRAP)


def level_with_quaternion(pano: np.ndarray, q_wxyz, keep_yaw: bool = True) -> np.ndarray:
    """q = rig->world orientation. Output = world-aligned panorama (heading kept if keep_yaw)."""
    R = quat_to_matrix(*q_wxyz)               # world <- rig
    if keep_yaw:
        fwd = R[:, 0]
        yaw = np.arctan2(fwd[1], fwd[0])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        M = R.T @ Rz                          # rig <- output
    else:
        M = R.T
    return rotate_equirect(pano, M)
