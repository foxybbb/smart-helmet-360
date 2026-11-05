"""Calibration target detection on raw fisheye frames: ArUco marker points, ChArUco boards, chessboards.

Works on the distorted images directly (no prior undistortion), so it can be used on real helmet photos.
"""
from __future__ import annotations

import cv2
import numpy as np

ARUCO = cv2.aruco


def aruco_detector(dictionary=ARUCO.DICT_4X4_1000, refine=ARUCO.CORNER_REFINE_SUBPIX):
    p = ARUCO.DetectorParameters()
    p.cornerRefinementMethod = refine
    p.cornerRefinementWinSize = 4
    p.adaptiveThreshWinSizeMax = 53
    return ARUCO.ArucoDetector(ARUCO.getPredefinedDictionary(dictionary), p)


def detect_markers(img, det) -> dict:
    """{marker id: (4, 2) corners, OpenCV order TL, TR, BR, BL}"""
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.ravel())}


def detect_charuco(img, boards, det):
    """boards: list of CharucoBoard sharing one dictionary (disjoint ids).
    Returns [(board index, corner ids (n,), corners (n, 2))]."""
    mc, mids, _ = det.detectMarkers(img)
    out = []
    if mids is None:
        return out
    for bi, b in enumerate(boards):
        ids_b = set(int(v) for v in np.asarray(b.getIds()).ravel())
        sel = [k for k, v in enumerate(mids.ravel()) if int(v) in ids_b]
        if len(sel) < 2:
            continue
        cd = ARUCO.CharucoDetector(b)
        ch, chid, _, _ = cd.detectBoard(img, markerCorners=tuple(mc[k] for k in sel),
                                        markerIds=mids[sel].reshape(-1, 1))
        if chid is None or len(chid) < 4:
            continue
        out.append((bi, chid.ravel().astype(int), ch.reshape(-1, 2).astype(np.float64)))
    return out


def _orient_chessboard(gray, c, inner):
    """Make the corner order un-mirrored and put pattern square (0, 0) (black) first.
    Board with (nx+1) x (ny+1) squares, nx odd / ny even: corner squares TL, BL black, TR, BR white."""
    nx, ny = inner
    g = c.reshape(ny, nx, 2)
    a = g[0, 1] - g[0, 0]
    b = g[1, 0] - g[0, 0]
    if a[0] * b[1] - a[1] * b[0] < 0:          # mirrored labelling -> reverse every row
        g = g[:, ::-1]
    h, w = gray.shape

    def outer(r, q, dr, dq):
        """centre of the corner square diagonal-outside of corner (r, q)"""
        p = g[r, q]
        return p + 0.5 * (p - g[r + dr, q]) + 0.5 * (p - g[r, q + dq])

    def val(p):
        x, y = int(round(p[0])), int(round(p[1]))
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            return np.nan
        return float(np.median(gray[y - 2:y + 3, x - 2:x + 3]))
    left = [val(outer(0, 0, 1, 1)), val(outer(ny - 1, 0, -1, 1))]
    right = [val(outer(0, nx - 1, 1, -1)), val(outer(ny - 1, nx - 1, -1, -1))]
    if np.nansum(left) > np.nansum(right):     # left corner squares are bright -> 180 deg flip
        g = g[::-1, ::-1]
    return g.reshape(-1, 2)


def detect_chessboards(img, inner=(9, 6), max_boards=3):
    """Several identical chessboards per image (one SB detection at a time, found boards are masked)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    work = gray.copy()
    out = []
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.005)
    for _ in range(max_boards):
        ok, c = cv2.findChessboardCornersSB(work, inner, flags)
        if not ok:   # SB misses low-contrast (projected) boards; the classic detector + sub-pixel refine finds them
            ok, c = cv2.findChessboardCorners(work, inner, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                break
            c = cv2.cornerSubPix(gray, c.astype(np.float32), (5, 5), (-1, -1), crit)
        c = c.reshape(-1, 2).astype(np.float64)
        out.append(_orient_chessboard(gray, c, inner))
        # mask the found board (convex hull grown by ~1.5 squares) and look again
        g = c.reshape(inner[1], inner[0], 2)
        sq = np.median(np.linalg.norm(np.diff(g, axis=1), axis=2))
        ctr = c.mean(0)
        hull = cv2.convexHull(c.astype(np.float32)).reshape(-1, 2)
        grown = ctr + (hull - ctr) * (1 + 1.8 * sq / np.maximum(np.linalg.norm(hull - ctr, axis=1, keepdims=True), 1))
        cv2.fillPoly(work, [grown.astype(np.int32)], int(np.median(gray)))
    return out
