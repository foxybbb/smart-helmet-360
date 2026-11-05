"""Final panoramas of the test room after calibration, stitched like production (4096 x 2048).

Rigs: no calibration, self-calibration, handheld ChArUco + decentering p1 p2 (best), true parameters.
Modes: "depth" = production depth mode (Depth Anything V2 + triangulation + global seam warp, GPU),
       "ideal" = every pixel at its true distance (upper bound of a perfect depth mode).
Writes Output/calib_study/final/*.jpg, a comparison sheet with seam close-ups and metrics.json.

  python -m calib_lab.final_panos [--variant 7x141] [--seed 1] [--width 4096]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np

from smart_helmet_pano.quality import seam_artifacts
from smart_helmet_pano.rig import Rig
from smart_helmet_pano.stitch import PanoStitcher, equirect_dirs
from . import evaluate as E
from . import methods as M
from .run_study import OUT, test_frames, test_room

RIGS = [("nominal", "Без калибровки"), ("selfcal", "Самокалибровка по сцене"),
        ("charuco_tangential", "ChArUco + p1, p2 (лучшая)"), ("oracle", "Истинные параметры")]
LABEL_EN = {"nominal": "no calibration", "selfcal": "self-calibration (natural features)",
            "charuco_tangential": "handheld ChArUco + lens decentering (best)", "oracle": "true parameters",
            "gt": "ground truth (ideal camera in the helmet centre)"}


def gt(W):
    p = OUT / f"gt_test_{W}.png"
    if not p.exists():
        d = equirect_dirs(W, W // 2).astype(np.float64).reshape(-1, 3)
        rad = test_room().shade(np.zeros(3), d).reshape(W // 2, W, 3)
        cv2.imwrite(str(p), np.clip(rad * 255 + 0.5, 0, 255).astype(np.uint8))
    return cv2.imread(str(p))


def metrics(pano, ref, weights, warped):
    H, W = pano.shape[:2]
    lat = 90 - (np.arange(H) + 0.5) / H * 180
    mask = (np.sum(weights, axis=0) > 0) & np.repeat((lat > -55)[:, None], W, 1)
    a, b = pano.astype(np.float32), ref.astype(np.float32)
    g = (a[mask] * b[mask]).sum(0) / np.maximum((a[mask] ** 2).sum(0), 1)
    a = np.clip(a * g, 0, 255).astype(np.uint8)
    mse = float(((a.astype(np.float32) - b) ** 2)[mask].mean())
    q, _ = seam_artifacts(warped, weights, W, H)
    return {"PSNR": round(10 * math.log10(255 ** 2 / mse), 2), "SSIM": round(E.ssim(a, ref, mask), 4),
            "seam_artifacts_%": q["artifact_%_of_central"]}


def label(img, text, scale=1.1):
    out = img.copy()
    h = int(34 * scale) + 12
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="7x141")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=4096)
    a = ap.parse_args()
    W = a.width
    ctx = M.make_ctx(a.variant, a.seed, OUT)
    out = OUT / "final"
    out.mkdir(parents=True, exist_ok=True)
    imgs = test_frames(ctx)
    ref = gt(W)
    cv2.imwrite(str(out / f"{a.variant}_gt_{W}.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])
    from smart_helmet_pano.advanced import mono_available, stitch_depth
    ok, why = mono_available()
    meta = {"rig_to_world": np.eye(3).tolist()}
    res, panos = {}, {}
    for key, _ in RIGS:
        rig = ctx.true if key == "oracle" else E.align(Rig.load(ctx.out / "rigs" / f"{key}.json"), ctx.true) \
            if key != "nominal" else E.align(ctx.nominal, ctx.true)
        opts = E.stitch_options(W, 3.0, flow=True)
        t0 = time.perf_counter()
        if ok:
            pano, st, info = stitch_depth(rig, imgs, meta, opts)
            mode = "depth"
        else:                                   # no GPU: production base mode (sphere + DIS flow)
            st = PanoStitcher(rig, opts)
            pano = st.stitch(imgs, meta)
            mode = "base"
        r = {"mode": mode, "seconds": round(time.perf_counter() - t0, 1),
             **metrics(pano, ref, st.last_weights, st.last_warped)}
        cv2.imwrite(str(out / f"{a.variant}_{key}_{mode}_{W}.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
        panos[key] = pano
        # ideal depth (true distance of every pixel) for the same rig, no flow warping
        o2 = E.stitch_options(W, E.true_depth(test_room(), np.zeros(3), W), flow=False)
        s2 = PanoStitcher(rig, o2)
        p2 = s2.stitch(imgs, meta)
        r["ideal_depth"] = metrics(p2, ref, s2.last_weights, s2.last_warped)
        cv2.imwrite(str(out / f"{a.variant}_{key}_ideal_{W}.jpg"), p2, [cv2.IMWRITE_JPEG_QUALITY, 92])
        res[key] = r
        print(key, json.dumps(r, ensure_ascii=False), flush=True)
    (out / f"{a.variant}_metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    # comparison sheet: same three windows in every panorama (ring seam, ring-top seam, near furniture)
    step = 360 / ctx.spec.ring
    H = W // 2
    wins = [(0.5 + step / 2 / 360, 0.47, "ring seam cam0 | cam1"),
            (0.5 + (2.5 * step) / 360, 0.2, "ring | top seam"),
            (0.5 - 0.115, 0.55, "column 1.1 m away")]
    cw, ch = int(W * 0.11), int(H * 0.16)
    rows = []
    for key in [k for k, _ in RIGS] + ["gt"]:
        im = ref if key == "gt" else panos[key]
        tiles = []
        for u, v, _ in wins:
            x, y = int(u * W) - cw // 2, int(v * H) - ch // 2
            c = im[max(y, 0):y + ch, max(x, 0):x + cw]
            tiles.append(cv2.resize(c, (720, int(720 * ch / cw)), interpolation=cv2.INTER_CUBIC))
        row = np.hstack(tiles)
        m = "" if key == "gt" else f"   PSNR {res[key]['PSNR']} dB"
        rows.append(label(row, LABEL_EN[key] + m, 0.9))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out / f"{a.variant}_seam_closeups_{W}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # overview: full panoramas stacked (scaled to 2048 wide)
    ov = [label(cv2.resize(panos[k], (2048, 1024), interpolation=cv2.INTER_AREA), LABEL_EN[k]) for k, _ in RIGS]
    ov.append(label(cv2.resize(ref, (2048, 1024), interpolation=cv2.INTER_AREA), LABEL_EN["gt"]))
    cv2.imwrite(str(out / f"{a.variant}_overview.jpg"), np.vstack(ov), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
