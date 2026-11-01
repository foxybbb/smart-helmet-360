"""Smart Helmet rig geometry and fisheye camera model.

Coordinate frames
-----------------
* Rig / helmet frame = Unreal Engine axes: X forward, Y right, Z up (metres).
* Camera frame = OpenCV axes: x right, y down, z = optical axis.
* Equirectangular panorama: longitude 0 = rig forward (+X), positive to the
  right (+Y); latitude +90 = up (+Z).  u = lon/2pi + 0.5,  v = 0.5 - lat/pi.
  This is the same mapping as in the "3D Engineering" stage-2 report (sec. 4.2.1).

Default rig = "Scheme_Helmet_8_Cameras":
  * 7 ring cameras, yaw step 360/7 = 51.43 deg, optical axis pitched +9.3 deg,
    portrait IMX462 (1080 x 1920), FOV 81.7 deg (H) x 141.4 deg (V),
    vertical coverage -61.4 .. +80 deg, horizontal overlap 30.27 deg.
  * 1 top camera pointing at the zenith, FOV 81.7 deg -> covers 49.15 .. 90 deg,
    overlap with the ring 30.85 deg.
Lens model: OpenCV fisheye / Kannala-Brandt
  r = f * theta_d, theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
With k = 0 this is the ideal equidistant fisheye used by the UE simulation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# Sensor + lens (Sony IMX462 2MP, M12 141 deg lens, mounted in portrait)
SENSOR_W, SENSOR_H = 1080, 1920
VFOV_DEG = 141.4
RING_PITCH_DEG = 9.3
RING_COUNT = 7
RING_RADIUS_M = 0.105      # lens distance from helmet centre (horizontal)
TOP_HEIGHT_M = 0.10        # top lens height above the ring plane


def rot_ue(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation matrix of an Unreal FRotator (columns = forward, right, up in rig coords).

    Matches FRotationMatrix(FRotator(P, Y, R)) applied as  world = M @ local.
    """
    p, y, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return np.stack([fwd, right, up], axis=1)


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """R(q) exactly as written in the stage-2 report (sec. 4.2.1)."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# camera axes (OpenCV) expressed in the UE component frame (X fwd, Y right, Z up)
_CV_FROM_UE = np.array([[0, 1, 0],    # x_cv = right
                        [0, 0, -1],   # y_cv = -up
                        [1, 0, 0]],   # z_cv = forward
                       dtype=float)


@dataclass
class Camera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    position_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_pyr_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # UE pitch, yaw, roll
    max_theta_deg: float = 89.0
    # optional calibrated rotation (rig <- camera, UE axes); overrides rotation_pyr_deg when set
    rotation_matrix: list | None = None

    # --- derived -------------------------------------------------------
    @property
    def R_rig_from_cam_ue(self) -> np.ndarray:
        if self.rotation_matrix is not None:
            return np.asarray(self.rotation_matrix, float)
        return rot_ue(*self.rotation_pyr_deg)

    @property
    def R_cv_from_rig(self) -> np.ndarray:
        """Rotate a rig-frame vector into OpenCV camera coordinates."""
        return _CV_FROM_UE @ self.R_rig_from_cam_ue.T

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)

    @property
    def D(self) -> np.ndarray:
        return np.array(self.k, float).reshape(4, 1)

    def project(self, P_rig: np.ndarray):
        """Project rig-frame points (..., 3) to pixel coords. Returns (u, v, valid)."""
        p = (P_rig - np.asarray(self.position_m)) @ self.R_cv_from_rig.T
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        rxy = np.sqrt(x * x + y * y)
        theta = np.arctan2(rxy, z)
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        scale = np.where(rxy > 1e-12, theta_d / np.maximum(rxy, 1e-12), 0.0)
        u = self.fx * x * scale + self.cx
        v = self.fy * y * scale + self.cy
        valid = (theta < math.radians(self.max_theta_deg)) & (u >= 0) & (u <= self.width - 1) \
            & (v >= 0) & (v <= self.height - 1)
        return u, v, valid

    def unproject(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Pixel -> unit ray in rig frame (equidistant inverse with Newton for k != 0)."""
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        theta_d = np.sqrt(mx * mx + my * my)
        theta = theta_d.copy()
        k1, k2, k3, k4 = self.k
        if any(self.k):
            for _ in range(10):
                t2 = theta * theta
                f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
                df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + 9 * k4 * t2)))
                theta = theta - f / df
        s = np.where(theta_d > 1e-12, np.sin(theta) / np.maximum(theta_d, 1e-12), 0.0)
        ray_cv = np.stack([mx * s, my * s, np.cos(theta)], axis=-1)
        return ray_cv @ self.R_cv_from_rig  # (R^T applied row-wise)


@dataclass
class Rig:
    name: str
    cameras: list
    imu_from_rig_pyr_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def to_json(self) -> dict:
        return {"name": self.name, "imu_from_rig_pyr_deg": self.imu_from_rig_pyr_deg,
                "cameras": [asdict(c) for c in self.cameras]}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "Rig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cams = [Camera(**{k: v for k, v in c.items() if k in Camera.__dataclass_fields__})
                for c in d["cameras"]]
        return Rig(d.get("name", "rig"), cams, d.get("imu_from_rig_pyr_deg", [0, 0, 0]))

    @property
    def ring(self):
        return [c for c in self.cameras if not c.name.startswith("top")]


def default_rig(width: int = SENSOR_W, height: int = SENSOR_H, ring_radius: float = RING_RADIUS_M,
                top_height: float = TOP_HEIGHT_M) -> Rig:
    # equidistant focal so that the long (vertical) side spans exactly VFOV
    f = (height / 2) / math.radians(VFOV_DEG / 2)
    cams = []
    for i in range(RING_COUNT):
        yaw = i * 360.0 / RING_COUNT
        pos = [ring_radius * math.cos(math.radians(yaw)), ring_radius * math.sin(math.radians(yaw)), 0.0]
        cams.append(Camera(f"cam{i}", width, height, f, f, (width - 1) / 2, (height - 1) / 2,
                           position_m=pos, rotation_pyr_deg=[RING_PITCH_DEG, yaw, 0.0]))
    # top camera: same sensor, optical axis = +Z; image "up" points to rig forward
    cams.append(Camera("top7", width, height, f, f, (width - 1) / 2, (height - 1) / 2,
                       position_m=[0.0, 0.0, top_height], rotation_pyr_deg=[90.0, 0.0, 0.0]))
    return Rig("SmartHelmet_8cam_IMX462", cams)


def coverage_report(rig: Rig, w: int = 720, h: int = 360) -> dict:
    """How many cameras see each direction (at infinity)."""
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    lon, lat = np.meshgrid(lon, lat)
    d = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1) * 1e4
    count = np.zeros((h, w), int)
    for c in rig.cameras:
        count += c.project(d)[2]
    area = np.cos(lat)
    tot = area.sum()
    return {"covered_%": float(100 * area[count > 0].sum() / tot),
            "overlap_%": float(100 * area[count > 1].sum() / tot),
            "lowest_covered_lat_deg": float(np.degrees(lat[count > 0].min())),
            "count_map": count}
