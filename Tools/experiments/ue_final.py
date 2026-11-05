"""Final panoramas of the Unreal Engine 5 captures, stitched like production (depth mode, GPU).

Calibration as recommended in Docs/CALIBRATION.md: calibrate once on a static, synchronous session
(dataset A: no trigger jitter -> per-camera self-calibration, 0.15-0.55 deg) and reuse that rig for every
later session of the same helmet (dataset B: Pi trigger jitter up to 0.5 s, where its own self-calibration
breaks down to 10 deg).  Ground truth = the 360 reference cube rendered from the helmet centre at the trigger.

  python experiments/ue_final.py [--width 4096] [--compare-frame 4]
Writes Output/ue_final/{A,B}/<frame>_<rig>.jpg, comparison sheets and metrics.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ue_eval import oracle_rig, ref_equirect  # noqa: E402
from calib_lab import evaluate as E  # noqa: E402
from smart_helmet_pano.advanced import mono_available, stitch_depth  # noqa: E402
from smart_helmet_pano.cli import frame_dirs  # noqa: E402
from smart_helmet_pano.quality import seam_artifacts  # noqa: E402
from smart_helmet_pano.rig import Rig  # noqa: E402
from smart_helmet_pano.stitch import PanoStitcher, StitchOptions, load_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UE = ROOT / "MyProject" / "Saved" / "SmartHelmet"
SETS = {"A": ("20261101_201049_PIE", "synchronous capture"), "B": ("20261101_200454_PIE", "desynchronised up to 0.5 s")}
CAL = UE / SETS["A"][0] / "rig_selfcal_percam.json"       # calibrated once, on the static synchronous set
LABEL = {"nominal": "no calibration", "own_selfcal": "self-calibration on this session",
         "calibrated": "calibrated once on set A (per-camera lens)", "oracle": "true parameters",
         "gt": "ground truth (360 reference from the helmet centre)"}


def by_names(rig: Rig, ref: Rig) -> Rig:
    """Same camera order as the reference rig (self-calibration stores the ring sorted by azimuth)."""
    d = {c.name: c for c in rig.cameras}
    return Rig(rig.name, [d[c.name] for c in ref.cameras])


def metrics(pano, ref, st):
    H, W = pano.shape[:2]
    lat = 90 - (np.arange(H) + 0.5) / H * 180
    mask = (np.sum(st.last_weights, axis=0) > 0) & np.repeat((lat > -50)[:, None], W, 1)   # skip operator body
    a, b = pano.astype(np.float32), ref.astype(np.float32)
    g = (a[mask] * b[mask]).sum(0) / np.maximum((a[mask] ** 2).sum(0), 1)
    a = np.clip(a * g, 0, 255).astype(np.uint8)
    mse = float(((a.astype(np.float32) - b) ** 2)[mask].mean())
    q, _ = seam_artifacts(st.last_warped, st.last_weights, W, H)
    return {"PSNR": round(10 * math.log10(255 ** 2 / mse), 2), "SSIM": round(E.ssim(a, ref, mask), 4),
            "seam_artifacts_%": q["artifact_%_of_central"]}


def label(img, text, scale=1.0):
    out = img.copy()
    h = int(32 * scale) + 12
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stitch(rig, fd, W, depth_mode):
    imgs, meta = load_frame(fd, rig)
    opts = StitchOptions(width=W, level="full", proxy_depth_m=4.0, theta_max_deg=75, theta_ramp_deg=75)
    if depth_mode:
        pano, st, _ = stitch_depth(rig, imgs, meta, opts)
    else:
        st = PanoStitcher(rig, opts)
        pano = st.stitch(imgs, meta)
    return pano, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--compare-frame", type=int, default=4, help="frame number for the rig comparison")
    a = ap.parse_args()
    W = a.width
    depth_mode, why = mono_available()
    out = ROOT / "Output" / "ue_final"
    res = {"mode": "depth" if depth_mode else f"base ({why})", "width": W, "calibration": str(CAL.relative_to(ROOT)),
           "sets": {}}
    for key, (name, title) in SETS.items():
        session = UE / name
        oracle, _ = oracle_rig(session)
        rigs = {"calibrated": E.align(by_names(Rig.load(CAL), oracle), oracle)}
        rigs_cmp = {"nominal": E.align(by_names(Rig.load(session / "rig.json"), oracle), oracle),
                    "own_selfcal": E.align(by_names(Rig.load(session / "rig_selfcal_percam.json"), oracle), oracle),
                    "oracle": oracle}
        (out / key).mkdir(parents=True, exist_ok=True)
        rows = {}
        for fd in frame_dirs(session):
            ref = ref_equirect(fd, W)
            cv2.imwrite(str(out / key / f"{fd.name}_gt.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 90])
            todo = dict(rigs)
            if int(fd.name) == a.compare_frame:
                todo.update(rigs_cmp)
            panos = {}
            for rn, rig in todo.items():
                t0 = time.perf_counter()
                pano, st = stitch(rig, fd, W, depth_mode)
                m = {**metrics(pano, ref, st), "seconds": round(time.perf_counter() - t0, 1)}
                rows.setdefault(fd.name, {})[rn] = m
                cv2.imwrite(str(out / key / f"{fd.name}_{rn}.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
                panos[rn] = pano
                print(key, fd.name, rn, json.dumps(m), flush=True)
            if len(panos) > 1:          # comparison sheet for this frame
                order = ["nominal", "own_selfcal", "calibrated", "oracle"]
                ov = [label(cv2.resize(panos[r], (2048, 1024), interpolation=cv2.INTER_AREA),
                            f"{LABEL[r]}   PSNR {rows[fd.name][r]['PSNR']} dB") for r in order]
                ov.append(label(cv2.resize(ref, (2048, 1024), interpolation=cv2.INTER_AREA), LABEL["gt"]))
                cv2.imwrite(str(out / f"{key}_{fd.name}_comparison.jpg"), np.vstack(ov), [cv2.IMWRITE_JPEG_QUALITY, 85])
            res["sets"][key] = {"session": name, "title": title, "frames": rows}
            (out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        cal = [r["calibrated"] for r in rows.values()]
        res["sets"][key]["mean_calibrated"] = {k: round(float(np.mean([c[k] for c in cal])), 3)
                                               for k in ("PSNR", "SSIM", "seam_artifacts_%")}
        # contact sheet of all frames of the set (calibrated rig)
        sheet = [label(cv2.resize(cv2.imread(str(out / key / f"{f}_calibrated.jpg")), (1024, 512),
                                  interpolation=cv2.INTER_AREA), f"set {key}, frame {f}", 0.7) for f in sorted(rows)]
        grid = [np.hstack(sheet[i:i + 2]) if i + 1 < len(sheet) else np.hstack([sheet[i], np.zeros_like(sheet[i])])
                for i in range(0, len(sheet), 2)]
        cv2.imwrite(str(out / f"{key}_all_frames.jpg"), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 85])
        (out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v.get("mean_calibrated") for k, v in res["sets"].items()}, indent=1))


if __name__ == "__main__":
    main()
