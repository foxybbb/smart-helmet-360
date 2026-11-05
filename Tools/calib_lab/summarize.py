"""Tables (markdown) + compact JSON of the calibration study.

  python -m calib_lab.summarize      -> Output/calib_study/tables.md, Docs/results/calibration_study.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .run_study import OUT, METHODS
from .truth import VARIANTS

ROOT = Path(__file__).resolve().parents[2]

NAMES = {
    "nominal": "No calibration (drawing + lens datasheet)",
    "selfcal": "Self-calibration on the scene (current pipeline)",
    "charuco": "Handheld ChArUco board",
    "markers": "ArUco marker points, positions unknown",
    "markers_surveyed": "Marker points, positions surveyed",
    "projector": "Chessboards projected onto the walls",
    "projector_scaled": "Projected chessboards + tape (width of every board)",
    "cube_design": "Square room (cube), geometry from the drawing",
    "cube_panels": "Square room (cube), panel poses estimated",
    "hybrid": "ChArUco lens + rotations from the scene",
    "oracle": "True parameters (limit)",
}


def load(seed=1, suffix=""):
    out = {}
    for key in VARIANTS:
        p = OUT / f"{key}_s{seed}{suffix}" / "results.json"
        if p.exists():
            out[key] = json.loads(p.read_text(encoding="utf-8"))
    return out


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f(x, n=1):
    return "—" if x is None else f"{x:.{n}f}"


SHOTS = {"markers": "{captures} tripod poses", "markers_surveyed": "{captures} tripod poses + total-station survey",
         "projector": "{captures} tripod poses × 14 projector aims",
         "projector_scaled": "same + 28 tape measurements", "cube_design": "{captures} poses in the cube",
         "cube_panels": "{captures} poses in the cube", "charuco": "{placements} board placements ({images} images)",
         "selfcal": "{frames} ordinary frames", "hybrid": "{frames} ordinary frames + lens from ChArUco"}


def main_table(r):
    rows = ["| Method | Camera rotation, ° (mean / max) | Lens centre, mm | Ray error, ′ | "
            "Panorama distortion ∞, ′ (mean / p95) | Seam break ∞, ′ (mean / p95) | "
            "Seams at true depth, ′ | PSNR at true depth, dB | Capture |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        x = r["methods"].get(m)
        if not x or "error" in x:
            continue
        c, pi, info = x["calib"], x["pano_infinity"], x["info"]
        td, itd = x.get("pano_room_truedepth", {}), x.get("image_truedepth", {})
        try:
            shots = SHOTS[m].format(**info)
        except (KeyError, IndexError):
            shots = "—"
        rows.append(f"| {NAMES[m]} | {f(c['rot_err_deg'], 2)} / {f(c['rot_err_max_deg'], 2)} | {f(c['pos_err_mm'])} | "
                    f"{f(c['ray_rms_arcmin'])} | "
                    f"{f(g(pi, 'geom_err_arcmin', 'mean'))} / {f(g(pi, 'geom_err_arcmin', 'p95'))} | "
                    f"{f(g(pi, 'seam_break_arcmin', 'mean'))} / {f(g(pi, 'seam_break_arcmin', 'p95'))} | "
                    f"{f(g(td, 'seam_break_arcmin', 'mean'))} | {f(itd.get('PSNR'), 2)} | {shots} |")
    return "\n".join(rows)


def proxy_table(r):
    rows = ["| Method | Seam break, ′ | PSNR, dB | SSIM | Artifacts without flow, % | Artifacts with flow, % |",
            "|---|---|---|---|---|---|"]
    for m in METHODS:
        x = r["methods"].get(m)
        if not x or "error" in x:
            continue
        im, imf = x["image"], x.get("image_flow", {})
        rows.append(f"| {NAMES[m]} | {f(g(x, 'pano_room', 'seam_break_arcmin', 'mean'))} | {f(im['PSNR'], 2)} | "
                    f"{f(im['SSIM'], 3)} | {f(im['seam_artifacts_%'], 2)} | {f(imf.get('seam_artifacts_%'), 2)} |")
    return "\n".join(rows)


def overlap_table(res):
    rows = ["| Variant | Cameras | Image FOV, ° (H × V) | px/° | Neighbour overlap at the horizon, ° | "
            "Ring–top overlap, ° | Sphere covered, % | Double coverage, % |",
            "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        o = r["overlap"]
        rows.append(f"| {VARIANTS[k].title} | {o['cameras']} | {o['hfov_deg']} × {o['vfov_deg']} | {o['px_per_deg']} | "
                    f"{o['horizon_overlap_deg']} | {o['ring_top_overlap_deg']} | {o['covered_%']} | {o['double_covered_%']} |")
    return "\n".join(rows)


def matrix(res, path, title, n=1, higher_better=False):
    keys = list(res)
    rows = [f"| {title} | " + " | ".join(VARIANTS[k].title for k in keys) + " |",
            "|---|" + "---|" * len(keys)]
    for m in METHODS:
        vals = [g(res[k], "methods", m, *path) for k in keys]
        if all(v is None for v in vals):
            continue
        pick = max if higher_better else min
        best = pick((v for v in vals if v is not None), default=None)
        cells = [("**" + f(v, n) + "**") if v is not None and v == best and len(keys) > 1 else f(v, n) for v in vals]
        rows.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def theta_table(r):
    bins = list(g(r, "methods", "charuco", "calib", "ray_rms_by_theta_arcmin", default={}).keys())
    rows = ["| Method | " + " | ".join(f"{b}°" for b in bins) + " |", "|---|" + "---|" * len(bins)]
    for m in METHODS:
        bt = g(r, "methods", m, "calib", "ray_rms_by_theta_arcmin")
        if not bt or m == "oracle":
            continue
        rows.append(f"| {NAMES[m]} | " + " | ".join(f(bt.get(b)) for b in bins) + " |")
    return "\n".join(rows)


def drift_table(seed=1, variant="7x141", drift=1.0):
    p = OUT / f"{variant}_s{seed}_drift{drift:g}" / "results.json"
    if not p.exists():
        return ""
    r = json.loads(p.read_text(encoding="utf-8"))
    names = {"nominal": "No calibration", "lab_charuco_stale": "Lab ChArUco, not updated",
             "lab_cube_stale": "Lab cube, not updated", "field_selfcal": "Self-calibration in the field",
             "hybrid_charuco_lens": "Hybrid: lens from ChArUco + rotations from the scene",
             "hybrid_cube_lens": "Hybrid: lens from the cube + rotations from the scene", "oracle": "True parameters"}
    rows = ["| Variant | Camera rotation, ° (mean / max) | Ray error, ′ | Distortion ∞, ′ | Seam break ∞, ′ | "
            "Seams at true depth, ′ | PSNR at true depth, dB |",
            "|---|---|---|---|---|---|---|"]
    for m, x in r["methods"].items():
        rows.append(f"| {names.get(m, m)} | {f(x['calib']['rot_err_deg'], 2)} / {f(x['calib']['rot_err_max_deg'], 2)} | "
                    f"{f(x['calib']['ray_rms_arcmin'])} | "
                    f"{f(g(x, 'pano_infinity', 'geom_err_arcmin', 'mean'))} | {f(g(x, 'pano_infinity', 'seam_break_arcmin', 'mean'))} | "
                    f"{f(g(x, 'pano_room_truedepth', 'seam_break_arcmin', 'mean'))} | {f(g(x, 'image_truedepth', 'PSNR'), 2)} |")
    return "\n".join(rows)


def tangential_table():
    """Lens model Kannala-Brandt (8 parameters) vs + decentering p1, p2: before -> after."""
    p = OUT / "tangential.json"
    if not p.exists():
        return ""
    t = json.loads(p.read_text(encoding="utf-8"))
    rows = ["| Method | Sample | Ray error, ′ | Distortion ∞, ′ | Seam break ∞, ′ | Seams at true depth, ′ |",
            "|---|---|---|---|---|---|"]
    for key, methods in t.items():
        variant, seed = key.rsplit("_s", 1)
        base = load(int(seed)).get(variant, {}).get("methods", {})
        for m, x in methods.items():
            b = base.get(m, {})
            def pair(path, n=1):
                return f"{f(g(b, *path), n)} → **{f(g(x, *path), n)}**"
            rows.append(f"| {NAMES[m]} | seed {seed} | {pair(('calib', 'ray_rms_arcmin'))} | "
                        f"{pair(('pano_infinity', 'geom_err_arcmin', 'mean'))} | "
                        f"{pair(('pano_infinity', 'seam_break_arcmin', 'mean'))} | "
                        f"{pair(('pano_room_truedepth', 'seam_break_arcmin', 'mean'))} |")
    return "\n".join(rows)


def compact(res):
    keep = {}
    for k, r in res.items():
        keep[k] = {"overlap": r["overlap"], "truth": r["truth"], "methods": {}}
        for m, x in r["methods"].items():
            if "error" in x:
                keep[k]["methods"][m] = x; continue
            keep[k]["methods"][m] = {"info": {a: b for a, b in x["info"].items() if a != "per_camera"},
                                     "calib": x["calib"], "pano_infinity": x["pano_infinity"],
                                     "pano_room": x["pano_room"], "image": x["image"],
                                     "image_flow": x.get("image_flow"),
                                     "pano_room_truedepth": x.get("pano_room_truedepth"),
                                     "image_truedepth": x.get("image_truedepth")}
    return keep


def main():
    res = load(1)
    parts = ["## Overlap variants\n", overlap_table(res)]
    if "7x141" in res:
        parts += ["\n## Scheme 7 × 141°: all methods\n", main_table(res["7x141"]),
                  "\n## Ray error by angle from the axis, ′ (7 × 141°)\n", theta_table(res["7x141"])]
    mx = lambda path, n=1, hb=False: matrix(res, path, "Method \\ variant", n, hb)
    parts += ["\n## Seam break at infinity, ′ (mean)\n", mx(("pano_infinity", "seam_break_arcmin", "mean")),
              "\n## Panorama distortion at infinity, ′ (mean)\n", mx(("pano_infinity", "geom_err_arcmin", "mean")),
              "\n## Seam break in the room at true depth, ′ (mean)\n",
              mx(("pano_room_truedepth", "seam_break_arcmin", "mean")),
              "\n## PSNR of the room at true depth, dB\n", mx(("image_truedepth", "PSNR"), 2, True),
              "\n## Seam break in the room, 3 m proxy sphere, ′ (mean)\n", mx(("pano_room", "seam_break_arcmin", "mean")),
              "\n## Camera rotation error, ° (mean)\n", mx(("calib", "rot_err_deg"), 3)]
    if "7x141" in res:
        parts += ["\n## Test room, 3 m proxy sphere (base mode), 7 × 141°\n", proxy_table(res["7x141"])]
    s2 = load(2)
    if "7x141" in s2:
        parts += ["\n## Repeat on another helmet sample (seed 2), 7 × 141°\n", main_table(s2["7x141"])]
    tt = tangential_table()
    if tt:
        parts += ["\n## Lens model: Kannala-Brandt vs + decentering p1, p2, 7 × 141°\n", tt]
    d = drift_table()
    if d:
        parts += ["\n## Mount drift in the field (σ 1°), 7 × 141°\n", d]
    (OUT / "tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    out = {"seed1": compact(res)}
    if s2:
        out["seed2"] = compact(s2)
    dp = OUT / "7x141_s1_drift1" / "results.json"
    if dp.exists():
        out["drift"] = compact({"7x141": json.loads(dp.read_text(encoding="utf-8")) | {"overlap": {}, "truth": []}})
    (ROOT / "Docs" / "results" / "calibration_study.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                     encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print((OUT / "tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
