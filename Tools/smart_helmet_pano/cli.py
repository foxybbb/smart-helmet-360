"""Command line for the Smart Helmet panorama pipeline.

  python -m smart_helmet_pano stitch  <frame_dir | session_dir> [--width 4096] [--no-flow] ...
  python -m smart_helmet_pano synth   <out_session_dir> [--frames 3]
  python -m smart_helmet_pano undistort <image> --cam cam0 [--rig rig.json]
  python -m smart_helmet_pano rig     [--out rig.json]       # default rig + coverage
  python -m smart_helmet_pano board   --out board.png        # printable ChArUco board
  python -m smart_helmet_pano calibrate-board <session>      # lab calibration with the handheld board
  python -m smart_helmet_pano refine-mounts <session> --lens lab_rig.json   # field re-calibration of the mounts
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def frame_dirs(path: Path):
    path = Path(path)
    is_frame = lambda p: (p / "meta.json").exists() or any(p.glob("cam*.png")) or any(p.glob("cam*.jpg"))
    if is_frame(path):
        return [path]
    root = path / "frames" if (path / "frames").is_dir() else path
    return sorted(p for p in root.iterdir() if p.is_dir() and is_frame(p))


def cmd_stitch(a):
    from .stitch import StitchOptions, stitch_frame
    from .quality import seam_artifacts
    kw = dict(width=a.width, proxy_depth_m=a.depth, level=a.level, exposure=not a.no_exposure,
              flow=not a.no_flow, blend=a.blend, tuning=a.tuning, theta_max_deg=a.theta_max)
    if a.real:  # preset for real helmet photos (self-calibrated rig, independent AE/AWB per camera)
        kw.update(level="none" if a.level == "tilt" else a.level, proxy_depth_m=6.0 if a.depth == 4.0 else a.depth,
                  theta_max_deg=a.theta_max or 75.0, theta_ramp_deg=a.theta_max or 75.0, max_flow_frac=0.005, flow_sigma_frac=0.008,
                  tuning=a.tuning or str(Path(__file__).resolve().parents[2] / "Data" / "imx219_160d.json"))
    opts = StitchOptions(**kw)
    mode = getattr(a, "mode", "auto") or "auto"
    if mode == "auto":
        mode = "depth" if a.real else "base"
    if mode == "depth":
        from .advanced import mono_available
        ok, why = mono_available()
        if not ok:
            print(f"[stitch] depth mode unavailable ({why}) -> falling back to base mode", file=sys.stderr)
            mode = "base"
    results = []
    fds = frame_dirs(Path(a.path))
    if a.limit:
        fds = [fds[i] for i in np.linspace(0, len(fds) - 1, min(a.limit, len(fds))).astype(int)]
    for fd in fds:
        if mode == "depth":
            from .advanced import stitch_depth
            from .rig import Rig
            from .stitch import find_rig, load_frame
            rig = Rig.load(find_rig(fd))
            imgs, meta = load_frame(fd, rig)
            pano, st, info = stitch_depth(rig, imgs, meta, opts)
            out = fd / "panorama.jpg"
            cv2.imwrite(str(out), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            out, pano, st = stitch_frame(fd, fd / "panorama.jpg", opts)
        q, art = seam_artifacts(st.last_warped, st.last_weights, st.W, st.H)
        if a.debug:
            vis = pano.copy()
            vis[art] = (0, 0, 255)
            cv2.imwrite(str(fd / "panorama_artifacts.jpg"), vis)
        r = {"frame": str(fd), "panorama": str(out), "mode": mode, "seconds": round(st.timings["total"], 2), **q}
        (fd / "quality.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
    return results


def cmd_synth(a):
    from .synth import BoxRoom, render_frame
    room = BoxRoom(seed=a.seed)
    rng = np.random.default_rng(a.seed)
    for i in range(a.frames):
        pyr = (float(rng.normal(0, 6)), float(rng.uniform(-180, 180)), float(rng.normal(0, 5)))
        gains = rng.uniform(0.85, 1.15, 8)
        d = render_frame(Path(a.out) / "frames" / f"{i + 1:06d}", helmet_pyr=pyr, room=room, gains=gains,
                         mount_error_deg=a.mount_error, seed=i)
        print("rendered", d, "helmet pyr", [round(x, 1) for x in pyr])


def cmd_undistort(a):
    from .rig import Rig, default_rig
    from .undistort import undistort, sweep_fov
    rig = Rig.load(a.rig) if a.rig else default_rig()
    cam = next(c for c in rig.cameras if c.name == a.cam)
    img = cv2.imread(a.image)
    for row in sweep_fov(img, cam):
        print(row)
    und, _ = undistort(img, cam, a.fov_h, a.fov_v)
    out = Path(a.image).with_name(Path(a.image).stem + "_undistorted.png")
    cv2.imwrite(str(out), und)
    print("saved", out)


def cmd_ingest(a):
    from .ingest import export
    return export(a.raw_root, a.out, top=a.top)


def cmd_calibrate(a):
    from .calibrate import calibrate_session
    rig, rep = calibrate_session(a.session, n_frames=a.frames, f_init=a.f_init)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def cmd_board(a):
    from .target_calib import make_board, save_board_image
    b = make_board((a.squares_x, a.squares_y), a.square_mm / 1000)
    w, h = save_board_image(a.out, b, dpi=a.dpi)
    print(f"saved {a.out}: {w}x{h} px at {a.dpi} dpi = {w / a.dpi * 25.4:.0f} x {h / a.dpi * 25.4:.0f} mm "
          f"(print at 100 %, check one square = {a.square_mm} mm with a ruler)")


def cmd_calibrate_board(a):
    from .rig import Rig
    from .target_calib import make_board, calibrate_board_session
    b = make_board((a.squares_x, a.squares_y), a.square_mm / 1000)
    rig0 = Rig.load(a.rig) if a.rig else None
    rig, rep = calibrate_board_session(a.session, rig0, b, f_init=a.f_init)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def cmd_refine_mounts(a):
    from .target_calib import refine_mounts_session
    rig, rep = refine_mounts_session(a.session, a.lens, n_frames=a.frames)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def cmd_health(a):
    import collections
    import glob
    from concurrent.futures import ThreadPoolExecutor
    from .health import photo_stats, verdict
    files = [f for f in glob.glob(str(Path(a.raw_root) / "cam*" / "session_*" / "*.jpg")) if "(1)" not in f]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda f: (f, verdict(photo_stats(cv2.imread(f)))), files))
    by = collections.defaultdict(list)
    for f, v in res:
        by[Path(f).parts[-3]].append(v)
    out = {}
    for c in sorted(by):
        L = by[c]
        out[c] = {"photos": len(L), "usable_%": round(100 * sum(1 for v in L if not v) / len(L)),
                  "problems": dict(collections.Counter(x for v in L for x in v))}
        print(c, out[c])
    return out


def cmd_rig(a):
    from .rig import default_rig, coverage_report
    rig = default_rig()
    rep = coverage_report(rig)
    print({k: round(v, 2) for k, v in rep.items() if k != "count_map"})
    if a.out:
        rig.save(a.out)
        print("saved", a.out)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser("smart_helmet_pano")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stitch")
    s.add_argument("path")
    s.add_argument("--width", type=int, default=4096)
    s.add_argument("--depth", type=float, default=4.0, help="proxy sphere radius, m (inf = rotation-only)")
    s.add_argument("--level", choices=["none", "tilt", "full"], default="tilt")
    s.add_argument("--blend", choices=["multiband", "feather"], default="multiband")
    s.add_argument("--no-flow", action="store_true")
    s.add_argument("--no-exposure", action="store_true")
    s.add_argument("--debug", action="store_true", help="also save artifact overlay")
    s.add_argument("--tuning", help="libcamera tuning json (ISP gamma + ALSC), e.g. Data/imx219_160d.json")
    s.add_argument("--theta-max", type=float, default=None, help="prefer pixels within this angle of each lens axis")
    s.add_argument("--real", action="store_true", help="preset for real helmet photos")
    s.add_argument("--limit", type=int, default=0, help="stitch only N evenly spaced frames")
    s.add_argument("--mode", choices=["auto", "base", "depth"], default="auto",
                   help="depth = Depth Anything + triangulated depth + global seam warp (GPU); auto = depth for --real")
    s.set_defaults(fn=cmd_stitch)
    s = sp.add_parser("ingest", help="raw Pi folders (cam1..cam8) -> synchronous frames + health filter")
    s.add_argument("raw_root")
    s.add_argument("out")
    s.add_argument("--top", default="cam1")
    s.set_defaults(fn=cmd_ingest)
    s = sp.add_parser("calibrate", help="self-calibrate lens + camera rotations from a session")
    s.add_argument("session")
    s.add_argument("--frames", type=int, default=16)
    s.add_argument("--f-init", type=float, default=680.0)
    s.set_defaults(fn=cmd_calibrate)
    for name, fn, hlp in (("board", cmd_board, "printable ChArUco board for the lab calibration"),
                          ("calibrate-board", cmd_calibrate_board,
                           "lab calibration with a handheld ChArUco board: per-camera lens + mounting")):
        s = sp.add_parser(name, help=hlp)
        if name == "board":
            s.add_argument("--out", default="charuco_board.png")
            s.add_argument("--dpi", type=int, default=300)
        else:
            s.add_argument("session", help="folder with frames/<id>/<cam>.jpg, one board placement per frame")
            s.add_argument("--rig", help="nominal rig.json (default: session rig.json or the scheme)")
            s.add_argument("--f-init", type=float, default=None)
        s.add_argument("--squares-x", type=int, default=11)
        s.add_argument("--squares-y", type=int, default=8)
        s.add_argument("--square-mm", type=float, default=50.0)
        s.set_defaults(fn=fn)
    s = sp.add_parser("refine-mounts", help="field re-calibration: lab lens kept, mounting re-estimated from ordinary frames")
    s.add_argument("session")
    s.add_argument("--lens", required=True, help="rig.json from the lab calibration (calibrate-board)")
    s.add_argument("--frames", type=int, default=8)
    s.set_defaults(fn=cmd_refine_mounts)
    s = sp.add_parser("health", help="per-camera photo quality report for raw Pi folders")
    s.add_argument("raw_root")
    s.set_defaults(fn=cmd_health)
    s = sp.add_parser("synth")
    s.add_argument("out")
    s.add_argument("--frames", type=int, default=3)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--mount-error", type=float, default=0.0, help="camera mounting error std, deg")
    s.set_defaults(fn=cmd_synth)
    s = sp.add_parser("undistort")
    s.add_argument("image")
    s.add_argument("--cam", default="cam0")
    s.add_argument("--rig")
    s.add_argument("--fov-h", type=float, default=75)
    s.add_argument("--fov-v", type=float, default=110)
    s.set_defaults(fn=cmd_undistort)
    s = sp.add_parser("rig")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_rig)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
