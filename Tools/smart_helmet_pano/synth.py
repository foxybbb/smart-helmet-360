"""Synthetic test frames: a textured box room rendered by the 8 helmet cameras.

Rays are traced from each camera's *real* position, so the frames contain the
same parallax as the physical helmet.  Also produces the ground-truth panorama
seen from the helmet centre.  Used to validate the stitcher without UE/hardware.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .rig import Rig, default_rig, rot_ue

TEX = 1024


def _texture(seed: int, label: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.full((TEX, TEX, 3), rng.integers(90, 200, 3), np.uint8)
    for _ in range(60):  # posters / boxes
        x, y = rng.integers(0, TEX - 60, 2)
        w, h = rng.integers(30, 260, 2)
        cv2.rectangle(t, (int(x), int(y)), (int(x + w), int(y + h)), rng.integers(0, 255, 3).tolist(), -1)
    for _ in range(40):
        c = rng.integers(0, TEX, 2)
        cv2.circle(t, (int(c[0]), int(c[1])), int(rng.integers(8, 60)), rng.integers(0, 255, 3).tolist(), 3)
    for i in range(0, TEX, 128):  # grid = straight lines for distortion checks
        cv2.line(t, (i, 0), (i, TEX), (20, 20, 20), 2)
        cv2.line(t, (0, i), (TEX, i), (20, 20, 20), 2)
    cv2.putText(t, label, (40, TEX // 2), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 14)
    return t


class BoxRoom:
    """Axis-aligned room [-hx,hx] x [-hy,hy] x [z0, z1] (metres, UE axes)."""

    def __init__(self, hx=2.5, hy=2.0, z0=-1.7, z1=1.3, seed=1):
        self.b = np.array([[-hx, -hy, z0], [hx, hy, z1]], float)
        names = ["+X FRONT", "-X BACK", "+Y RIGHT", "-Y LEFT", "+Z CEIL", "-Z FLOOR"]
        self.tex = [_texture(seed + i, n) for i, n in enumerate(names)]

    def shade(self, origin: np.ndarray, rays: np.ndarray) -> np.ndarray:
        """rays (...,3) unit, origin (3,) -> BGR image (...,3)."""
        shp = rays.shape[:-1]
        r = rays.reshape(-1, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = (self.b[1] - origin) / r
            t_lo = (self.b[0] - origin) / r
        t_exit = np.where(r > 0, t_hi, t_lo)
        t_exit[~np.isfinite(t_exit)] = np.inf
        axis = np.argmin(t_exit, 1)
        t = t_exit[np.arange(len(r)), axis]
        P = origin + r * t[:, None]
        sign = np.sign(r[np.arange(len(r)), axis])
        face = axis * 2 + (sign < 0)
        out = np.zeros((len(r), 3), np.uint8)
        ext = self.b[1] - self.b[0]
        for f in range(6):
            sel = face == f
            if not sel.any():
                continue
            a = f // 2
            u_ax, v_ax = [(1, 2), (1, 2), (0, 2), (0, 2), (0, 1), (0, 1)][f]
            s = (P[sel, u_ax] - self.b[0, u_ax]) / ext[u_ax]
            tt = 1 - (P[sel, v_ax] - self.b[0, v_ax]) / ext[v_ax]
            n = int(sel.sum())
            pad = (-n) % 4096
            mx = np.pad((s * (TEX - 1)).astype(np.float32), (0, pad)).reshape(-1, 4096)
            my = np.pad((tt * (TEX - 1)).astype(np.float32), (0, pad)).reshape(-1, 4096)
            col = cv2.remap(self.tex[f], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[sel] = col.reshape(-1, 3)[:n]
        return out.reshape(*shp, 3)


def render_frame(out_dir, rig: Rig | None = None, helmet_pyr=(0.0, 0.0, 0.0), room: BoxRoom | None = None,
                 gains=None, noise=2.0, mount_error_deg=0.0, seed=0):
    """Write cam*.png + meta.json (+ rig.json one level up) like the UE plugin does."""
    rig = rig or default_rig()
    room = room or BoxRoom()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rig.save(out_dir.parent / "rig.json")
    R_world = rot_ue(*helmet_pyr)
    rng = np.random.default_rng(seed)
    gains = gains if gains is not None else np.ones(len(rig.cameras))
    for cam, g in zip(rig.cameras, gains):
        uu, vv = np.meshgrid(np.arange(cam.width, dtype=np.float64), np.arange(cam.height, dtype=np.float64))
        rays = cam.unproject(uu, vv)
        if mount_error_deg:
            e = rng.normal(0, mount_error_deg, 3)
            rays = rays @ rot_ue(*e).T
        rays = rays @ R_world.T
        origin = R_world @ np.asarray(cam.position_m)
        img = room.shade(origin, rays).astype(np.float32) * g
        img += rng.normal(0, noise, img.shape)
        cv2.imwrite(str(out_dir / f"{cam.name}.png"), np.clip(img, 0, 255).astype(np.uint8))
    q = _matrix_to_quat(R_world)
    meta = {"source": "synthetic", "helmet_pyr_deg": list(helmet_pyr), "rig_to_world": R_world.tolist(),
            "imu": {"quaternion": q}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def ground_truth(room: BoxRoom, w=2048) -> np.ndarray:
    from .stitch import equirect_dirs
    return room.shade(np.zeros(3), equirect_dirs(w, w // 2).astype(np.float64))


def _matrix_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [float(w), float(x), float(y), float(z)]
