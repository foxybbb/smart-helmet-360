"""Render helmet captures of a scene with the TRUE cameras (faults + sensor photometry), in parallel.

Task = one camera, several helmet poses (rays of the camera are cached per worker, poses only rotate them).
Scenes are rebuilt deterministically inside every worker from (factory name, kwargs).
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from smart_helmet_pano.rig import Rig
from . import scenes as S
from .truth import TrueCamera

_SCENES: dict = {}
_RAYS: dict = {}


def load_true_rig(d: dict) -> Rig:
    cams = [TrueCamera(**c) for c in d["cameras"]]
    return Rig(d["name"], cams)


def rig_dict(rig: Rig) -> dict:
    return rig.to_json()


def _scene(key):
    if key not in _SCENES:
        name, kw = key[0], dict(key[1])
        _SCENES.clear()                     # one scene per worker at a time (memory)
        _SCENES[key] = getattr(S, name)(**kw)
    return _SCENES[key]


def _rays(rig_name, cam: TrueCamera):
    k = (rig_name, cam.name)
    if k not in _RAYS:
        if len(_RAYS) > 3:
            _RAYS.pop(next(iter(_RAYS)))
        uu, vv = np.meshgrid(np.arange(cam.width, dtype=np.float64), np.arange(cam.height, dtype=np.float64))
        rays = cam.unproject(uu, vv)
        ray_cv = rays @ cam.R_cv_from_rig.T
        theta = np.arccos(np.clip(ray_cv[..., 2], -1, 1))
        vig = (1 - cam.vignette * (1 - np.cos(np.minimum(theta, 1.45)) ** 4)).astype(np.float32)
        _RAYS[k] = (rays.reshape(-1, 3), vig)
    return _RAYS[k]


def sensor(rad, cam: TrueCamera, vig, rng):
    img = rad * np.asarray(cam.gain_bgr, np.float32)[None, None, :] * vig[..., None]
    img = np.clip(img, 0, 1) ** (1 / 1.0)
    img = img * 255
    if cam.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), cam.blur)
    img += rng.normal(0, cam.noise, img.shape).astype(np.float32)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def render_camera(scene, rig: Rig, ci: int, R_w, t_w, seed=0, board=None):
    cam = rig.cameras[ci]
    rays, vig = _rays(rig.name, cam)
    R_w = np.asarray(R_w, float)
    origin = R_w @ np.asarray(cam.position_m) + np.asarray(t_w, float)
    sc = scene
    if board is not None:              # handheld board placed in front of the background scene
        b, tex, bR, bt = board
        sc = S.BoardScene(b, tex, np.asarray(bR), np.asarray(bt), scene)
    rad = sc.shade(origin, rays @ R_w.T).reshape(cam.height, cam.width, 3)
    return sensor(rad, cam, vig, np.random.default_rng(seed))


def _task(args):
    scene_key, rig_d, ci, items, out_dir, board_spec = args
    rig = load_true_rig(rig_d)
    scene = _scene(scene_key)
    board = None
    if board_spec is not None:
        b = S.charuco_board(*board_spec["args"])
        tex = b.generateImage(tuple(board_spec["px"]), marginSize=0, borderBits=1)
        tex = cv2.GaussianBlur(np.repeat(tex[..., None], 3, 2), (0, 0), 0.6)
    paths = []
    for it in items:
        p = Path(out_dir) / it["file"]
        if p.exists():
            paths.append(str(p)); continue
        bd = (b, tex, it["board_R"], it["board_t"]) if board_spec is not None else None
        img = render_camera(scene, rig, ci, it["R_w"], it["t_w"], seed=it.get("seed", 0), board=bd)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


def render_jobs(scene_key, rig: Rig, jobs: dict, out_dir, board_spec=None, workers=None):
    """jobs: {camera index: [ {file, R_w, t_w, seed, (board_R, board_t)} ]}"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = rig_dict(rig)
    tasks = []
    for ci, items in jobs.items():
        todo = [it for it in items if not (out_dir / it["file"]).exists()]
        for s in range(0, len(todo), 6):     # small chunks keep all workers busy
            tasks.append((scene_key, rd, ci, todo[s:s + 6], str(out_dir), board_spec))
    if not tasks:
        return
    workers = workers or max(1, min(8, (os.cpu_count() or 4) // 2))
    with ProcessPoolExecutor(workers) as ex:
        for _ in ex.map(_task, tasks):
            pass


def scene_key(name, **kw):
    return (name, tuple(sorted(kw.items())))
