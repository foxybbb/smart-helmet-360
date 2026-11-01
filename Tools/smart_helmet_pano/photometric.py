"""Photometric normalisation for real IMX219 helmet cameras, driven by the libcamera tuning file.

* ISP gamma (rpi.contrast.gamma_curve) -> exact inverse LUT, so gains are applied in *linear* light
  (exposure / ISO / white-balance differences between cameras are multiplicative only there).
* Residual vignetting: ALSC corrects only `luminance_strength` (0.7) of the lens shading, so the
  corners stay ~15 % darker; the remaining part of `luminance_lut` is applied here.
* Per-camera, per-channel gains are solved in the stitcher from overlap ratios (log domain).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def srgb_to_linear_lut():
    v = np.arange(256) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


class Tuning:
    def __init__(self, path: str | Path | None):
        self.alg = {}
        if path and Path(path).exists():
            self.alg = {k: v for a in json.loads(Path(path).read_text(encoding="utf-8"))["algorithms"] for k, v in a.items()}
        self.to_linear = self._gamma_inverse()
        self.to_display = self._gamma_forward()

    def _curve(self):
        g = self.alg.get("rpi.contrast", {}).get("gamma_curve")
        if not g:
            return None
        g = np.array(g, float).reshape(-1, 2) / 65535.0
        return g[:, 0], g[:, 1]           # linear in -> display out

    def _gamma_inverse(self):
        c = self._curve()
        if c is None:
            return srgb_to_linear_lut()
        x, y = c
        return np.interp(np.arange(256) / 255.0, y, x).astype(np.float32)

    def _gamma_forward(self):
        c = self._curve()
        lin = np.linspace(0, 1, 4096)
        if c is None:
            out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        else:
            out = np.interp(lin, c[0], c[1])
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    def linear(self, img_u8):
        return self.to_linear[img_u8]

    def display(self, lin):
        idx = np.clip(lin * 4095, 0, 4095).astype(np.int32)
        return self.to_display[idx]

    def residual_vignette(self, w, h, rotate=None, sensor=(3280, 2464)):
        """Gain map (>=1) that completes the ALSC lens-shading correction for a WxH output image.
        Assumes the 1080p mode scales the full sensor width and crops rows symmetrically."""
        a = self.alg.get("rpi.alsc")
        if not a or "luminance_lut" not in a:
            return None
        L = np.array(a["luminance_lut"], np.float32).reshape(12, 16)
        s = float(a.get("luminance_strength", 1.0))
        resid = L / (1 + s * (L - 1))
        sw, sh = sensor
        full = cv2.resize(resid, (sw // 8, sh // 8), interpolation=cv2.INTER_CUBIC)
        out_w, out_h = (h, w) if rotate in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) else (w, h)
        rows = int(round(sh / 8 * (sw / out_w) * out_h / sw))  # rows of the sensor used by the mode
        rows = min(rows, full.shape[0])
        r0 = (full.shape[0] - rows) // 2
        m = cv2.resize(full[r0:r0 + rows], (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if rotate is not None:
            m = cv2.rotate(m, rotate)
        return m


@lru_cache(maxsize=4)
def load_tuning(path: str | None):
    return Tuning(path)
