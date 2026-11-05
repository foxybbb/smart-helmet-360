"""Add the ideal-depth metrics to results that were scored before they existed.

  python -m calib_lab.rescore [--seeds 1]
"""
from __future__ import annotations

import argparse
import json

from smart_helmet_pano.rig import Rig
from . import evaluate as E
from . import methods as M
from .run_study import OUT, gt_pano, score_true_depth, test_frames
from .truth import VARIANTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    a = ap.parse_args()
    gt = gt_pano()
    for seed in a.seeds:
        for key in VARIANTS:
            p = OUT / f"{key}_s{seed}" / "results.json"
            if not p.exists():
                continue
            res = json.loads(p.read_text(encoding="utf-8"))
            ctx = M.make_ctx(key, seed, OUT)
            imgs = test_frames(ctx)
            for m, r in res["methods"].items():
                if "error" in r or "pano_room_truedepth" in r:
                    continue
                rig = ctx.true if m == "oracle" else (ctx.nominal if m == "nominal" else
                                                      Rig.load(ctx.out / "rigs" / f"{m}.json"))
                al = rig if m == "oracle" else E.align(rig, ctx.true)
                r.update(score_true_depth(ctx, al, imgs, gt, ctx.out / "panos" / f"{m}.jpg"))
                x = r["pano_room_truedepth"]["seam_break_arcmin"].get("mean")
                print(f"{key} {m:17s} true-depth seam {x}'  PSNR {r['image_truedepth']['PSNR']}", flush=True)
                p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
