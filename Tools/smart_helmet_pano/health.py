"""Per-photo health check (test programme 3.1 / 3.3): flags images that cannot be stitched.

  noise    – sensor noise dominates (dark scene at max analogue gain, covered/defective camera)
  over     – too many clipped highlights (AE failure / looking into a lamp or sun)
  dark     – too dark
  cast     – strong colour cast (AWB failure), e.g. the green frames of cam8
  blur     – motion blur (low sharpness)
"""
from __future__ import annotations

import cv2
import numpy as np


def photo_stats(img) -> dict:
    if img is None or img.size == 0:
        return {"corrupt": True, "mean": 0, "noise": 0, "sharp": 0, "clipped_%": 0, "cast": 0}
    small = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(g.mean())
    # noise: pixel-level variation vs. 3x3 median, on a full-resolution crop (downscaling averages noise away)
    h, w = img.shape[:2]
    crop = img[h // 2 - 256: h // 2 + 256, w // 2 - 256: w // 2 + 256]
    med = cv2.medianBlur(crop, 3).astype(np.float32)
    noise = float(np.median(np.abs(crop.astype(np.float32) - med)))
    sharp = float(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F).var())
    clipped = float((g > 250).mean() * 100)
    b, gr, r = [float(small[..., c].mean()) + 1 for c in range(3)]
    cast = float(max(abs(np.log(gr / r)), abs(np.log(gr / b)), abs(np.log(r / b))))
    return {"mean": round(mean, 1), "noise": round(noise, 2), "sharp": round(sharp, 1),
            "clipped_%": round(clipped, 1), "cast": round(cast, 3)}


def verdict(s: dict) -> list:
    if s.get("corrupt"):
        return ["corrupt"]
    bad = []
    if s["noise"] > 6:
        bad.append("noise")
    if s["clipped_%"] > 35:
        bad.append("over")
    if s["mean"] < 25:
        bad.append("dark")
    if s["cast"] > 0.35:
        bad.append("cast")
    if s["sharp"] < 6 and "noise" not in bad:
        bad.append("blur")
    return bad
