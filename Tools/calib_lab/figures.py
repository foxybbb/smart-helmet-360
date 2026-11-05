"""Figures for Docs/CALIBRATION.md (OpenCV drawing only, no plotting libraries needed).

  python -m calib_lab.figures
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from smart_helmet_pano import targets as TG
from smart_helmet_pano.rig import Rig, coverage_report
from . import evaluate as E
from . import methods as M
from . import scenes as S
from .run_study import OUT, test_room
from .truth import VARIANTS, nominal_rig

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "Docs" / "media"
ARUCO = cv2.aruco
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, scale=0.9, y=34):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], y + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (10, y), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _put(img, text, org, scale=0.7, color=(255, 255, 255), th=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, th, cv2.LINE_AA)


# ---------------------------------------------------------------- 1. stands with detections
def stands(variant="7x141", seed=1):
    d = OUT / f"{variant}_s{seed}"
    tiles = []
    # marker points
    im = cv2.imread(str(d / "markers" / "p03_cam0.jpg"))
    for mid, c in TG.detect_markers(im, TG.aruco_detector(ARUCO.DICT_4X4_1000)).items():
        cv2.polylines(im, [c.astype(np.int32)], True, (0, 255, 0), 3)
    tiles.append((im, "ArUco marker points"))
    # projected chessboards
    im = cv2.imread(str(d / "projector" / "p02_cam6.jpg"))
    for c in TG.detect_chessboards(im, (9, 6), 3):
        cv2.drawChessboardCorners(im, (9, 6), c.astype(np.float32).reshape(-1, 1, 2), True)
    tiles.append((im, "Projected chessboards"))
    # cube
    sc = S.CubeRoom()
    im = cv2.imread(str(d / "cube" / "p01_cam2.jpg"))
    for bi, ids, cc in TG.detect_charuco(im, [b for _, b in sc.boards], TG.aruco_detector(ARUCO.DICT_5X5_1000)):
        for p in cc:
            cv2.circle(im, tuple(int(v) for v in p), 7, (0, 0, 255), 3)
    tiles.append((im, "Calibration cube (ChArUco)"))
    # handheld board
    files = sorted((d / "charuco").glob("b*_cam3.jpg"))
    im = cv2.imread(str(files[len(files) // 3]))
    b = S.charuco_board(*M.BOARD_ARGS)
    for _, ids, cc in TG.detect_charuco(im, [b], TG.aruco_detector(ARUCO.DICT_5X5_1000)):
        for p in cc:
            cv2.circle(im, tuple(int(v) for v in p), 7, (0, 0, 255), 3)
    tiles.append((im, "Handheld ChArUco board"))
    im = cv2.imread(str(d / "natural" / "f02_cam1.jpg"))
    tiles.append((im, "Natural room (self-cal)"))
    tiles = [label(cv2.resize(t, (432, 768), interpolation=cv2.INTER_AREA), n, 0.62, 26) for t, n in tiles]
    out = np.hstack(tiles)
    cv2.imwrite(str(MEDIA / "15_calibration_stands.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- 2. overlap variants
def overlap_maps(w=720):
    colors = np.array([[40, 40, 40], [120, 160, 90], [60, 160, 230], [40, 80, 220], [150, 40, 200]], np.uint8)
    rows = []
    for key, spec in VARIANTS.items():
        rig = nominal_rig(spec)
        cnt = np.clip(coverage_report(rig, w, w // 2)["count_map"], 0, 4)
        img = colors[cnt]
        res = json.loads((OUT / f"{key}_s1" / "results.json").read_text(encoding="utf-8"))["overlap"] \
            if (OUT / f"{key}_s1" / "results.json").exists() else None
        t = spec.title.replace('×', 'x')
        if res:
            t += f": horizon overlap {res['horizon_overlap_deg']} deg, {res['px_per_deg']} px/deg"
        img = label(img, t, 0.6, 24)
        rows.append(img)
    leg = np.full((40, w, 3), 255, np.uint8)
    for i, txt in enumerate(["0", "1 camera", "2", "3", "4+"]):
        cv2.rectangle(leg, (10 + i * 140, 8), (40 + i * 140, 32), colors[i].tolist(), -1)
        cv2.putText(leg, txt, (46 + i * 140, 28), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    grid = np.vstack([np.hstack(rows[0:2]), np.hstack([rows[2], rows[3]]),
                      np.hstack([rows[4], np.full_like(rows[4], 255)])])
    legw = np.full((40, grid.shape[1], 3), 255, np.uint8)
    legw[:, :w] = leg
    cv2.imwrite(str(MEDIA / "16_overlap_variants.jpg"), np.vstack([grid, legw]), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------- 3. analytic distortion maps
def heat(err, vmax):
    v = np.clip(np.nan_to_num(err, nan=0) / vmax, 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img[~np.isfinite(err)] = 60
    return img


def distortion_maps(variant="7x141", seed=1, methods=("nominal", "selfcal", "charuco", "cube_panels"), vmax=60.0):
    ctx = M.make_ctx(variant, seed, OUT)
    rows = []
    names = {"nominal": "no calibration (CAD + datasheet)", "selfcal": "self-calibration (natural features)",
             "charuco": "handheld ChArUco", "cube_panels": "calibration cube, panel poses estimated",
             "markers": "ArUco marker points", "projector": "projected chessboards"}
    for m in methods:
        rig = ctx.nominal if m == "nominal" else Rig.load(ctx.out / "rigs" / f"{m}.json")
        res, err, brk, owner = E.pano_distortion(E.align(rig, ctx.true), ctx.true, np.eye(3), np.zeros(3),
                                                 scene=None, W=1024, return_maps=True)
        img = heat(err, vmax)
        seam = np.isfinite(brk)
        img[cv2.dilate(seam.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (255, 255, 255)
        g = res["geom_err_arcmin"]; s = res["seam_break_arcmin"]
        rows.append(label(img, f"{names.get(m, m)}: error {g['mean']:.1f}' mean / {g['p95']:.1f}' p95, "
                               f"seam break {s['mean']:.1f}'", 0.55, 22))
    bar = np.zeros((36, 1024, 3), np.uint8)
    ramp = heat(np.tile(np.linspace(0, vmax, 700), (16, 1)), vmax)
    bar[4:20, 160:860] = ramp
    _put(bar, "0'", (128, 18), 0.5); _put(bar, f"{vmax:.0f}'", (868, 18), 0.5)
    _put(bar, "white = seams", (10, 18), 0.45)
    cv2.imwrite(str(MEDIA / "17_distortion_maps.jpg"), np.vstack(rows + [bar]), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------- 4. seam close-ups of the stitched test room
def seam_closeups(variant="7x141", seed=1, methods=("nominal", "selfcal", "charuco", "cube_panels", "oracle"),
                  suffix="_truedepth"):
    """Stitched test room at a ring seam; suffix "_truedepth" = panoramas stitched at the true depth of every
    pixel, so only the calibration error is visible (no parallax of the proxy sphere)."""
    d = OUT / f"{variant}_s{seed}" / "panos"
    gt = cv2.imread(str(OUT / "gt_test_2048.png"))
    ctx = M.make_ctx(variant, seed, OUT)
    # pick the seam between cam0 and cam1 near the horizon: longitude = half a ring step
    step = 360 / ctx.spec.ring
    W = gt.shape[1]
    cx = int((0.5 + step / 2 / 360) * W)
    cy = int(0.5 * gt.shape[0]) - 40
    crops = []
    titles = {"nominal": "no calibration", "selfcal": "self-cal", "charuco": "handheld ChArUco",
              "cube_panels": "cube (panels)", "oracle": "true parameters", "gt": "ground truth"}
    for m in list(methods) + ["gt"]:
        im = gt if m == "gt" else cv2.imread(str(d / f"{m}{suffix}.jpg"))
        c = im[cy - 150:cy + 150, cx - 200:cx + 200]
        c = cv2.resize(c, (600, 450), interpolation=cv2.INTER_CUBIC)
        crops.append(label(c, titles.get(m, m), 0.8, 30))
    out = np.vstack([np.hstack(crops[:3]), np.hstack(crops[3:6])])
    cv2.imwrite(str(MEDIA / "18_seam_closeups.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------- 5. bar chart (seam break per method and variant)
def chart(results: dict, metric=("pano_infinity", "seam_break_arcmin", "mean"), fname="19_seam_break_chart.jpg",
          title="Seam break at infinity, arcmin (lower is better)"):
    variants = [v for v in VARIANTS if v in results]
    methods = ["nominal", "selfcal", "charuco", "markers", "markers_surveyed", "projector", "projector_scaled",
               "cube_design", "cube_panels", "hybrid", "oracle"]
    methods = [m for m in methods if any(m in results[v] for v in variants)]
    vals = {(v, m): results[v][m][metric[0]][metric[1]].get(metric[2]) for v in variants for m in methods
            if m in results[v] and metric[1] in results[v][m].get(metric[0], {})}
    vmax = max(x for x in vals.values() if x is not None and np.isfinite(x))
    cap = min(vmax, 3 * np.median([x for x in vals.values() if x]))
    step = 2 if cap <= 16 else 5
    cap = step * math.ceil(cap / step)
    Wd, Hd = 1500, 640
    img = np.full((Hd, Wd, 3), 255, np.uint8)
    pal = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148)]
    x0, y0, y1 = 90, 70, Hd - 120
    gw = (Wd - x0 - 20) / len(methods)
    bw = gw * 0.8 / len(variants)
    for t in np.arange(0, cap + 1e-9, step):
        y = int(y1 - (y1 - y0) * t / cap)
        cv2.line(img, (x0, y), (Wd - 20, y), (225, 225, 225), 1)
        cv2.putText(img, f"{t:.0f}", (x0 - 50, y + 5), FONT, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    for mi, m in enumerate(methods):
        grp = [vals.get((v, m)) for v in variants if vals.get((v, m)) is not None]
        all_over = grp and min(grp) > cap
        if all_over:          # one label for the whole group: the range of its values
            cv2.putText(img, f"{min(grp):.0f}-{max(grp):.0f}", (int(x0 + mi * gw + gw * 0.15), y0 - 6), FONT, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        for vi, v in enumerate(variants):
            x = vals.get((v, m))
            if x is None:
                continue
            xa = int(x0 + mi * gw + gw * 0.1 + vi * bw)
            h = int((y1 - y0) * min(x, cap) / cap)
            cv2.rectangle(img, (xa, y1 - h), (int(xa + bw - 2), y1), pal[vi % len(pal)], -1)
            if x > cap and not all_over:
                cv2.putText(img, f"{x:.0f}", (xa - 4, y0 - 4), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        lbl = m.replace("_", " ")
        cv2.putText(img, lbl, (int(x0 + mi * gw + 4), y1 + 24), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (x0, y1), (Wd - 20, y1), (0, 0, 0), 1)
    cv2.putText(img, title, (x0, 40), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for vi, v in enumerate(variants):
        cv2.rectangle(img, (x0 + vi * 200, Hd - 50), (x0 + vi * 200 + 24, Hd - 30), pal[vi % len(pal)], -1)
        cv2.putText(img, v.replace("x", " cams x ") + " deg", (x0 + vi * 200 + 30, Hd - 34), FONT, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(MEDIA / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def load_results(seed=1):
    out = {}
    for key in VARIANTS:
        p = OUT / f"{key}_s{seed}" / "results.json"
        if p.exists():
            out[key] = {m: r for m, r in json.loads(p.read_text(encoding="utf-8"))["methods"].items() if "error" not in r}
    return out


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    stands()
    overlap_maps()
    distortion_maps()
    seam_closeups()
    r = load_results()
    chart(r)
    chart(r, ("pano_room_truedepth", "seam_break_arcmin", "mean"), "20_seam_break_truedepth_chart.jpg",
          "Seam break in the test room, true depth per pixel, arcmin (lower is better)")


if __name__ == "__main__":
    main()
