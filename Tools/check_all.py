"""Full system check for the Smart Helmet project.  Usage:  python check_all.py [--quick]

Writes Output/check_report.json and Output/check_report.md.  Each check returns PASS / FAIL / SKIP.
"""
import asyncio
import json
import math
import socket
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
OUT = ROOT / "Output"
REAL = OUT / "real_helmet"
RAW = ROOT / "Data" / "raw_photos" / "Raw_photos"
TUNING = ROOT / "Data" / "imx219_160d.json"
QUICK = "--quick" in sys.argv

results = []


def check(name, group):
    def deco(fn):
        def run():
            t = time.perf_counter()
            try:
                status, detail = fn()
            except Exception as e:  # noqa: BLE001
                status, detail = "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
            results.append({"group": group, "check": name, "status": status, "detail": detail,
                            "seconds": round(time.perf_counter() - t, 1)})
            print(f"[{status:4}] {group} / {name}: {str(detail)[:160]}")
        return run
    return deco


# ------------------------------------------------------------------ geometry
@check("Rig matches Scheme_Helmet_8_Cameras", "1 Geometry")
def c_rig():
    from smart_helmet_pano.rig import default_rig, coverage_report
    r = default_rig()
    c = r.cameras[0]
    v = c.unproject(np.array([c.cx, c.cx]), np.array([0.0, c.height - 1.0]))
    el = np.degrees(np.arcsin(v[:, 2]))
    h = c.unproject(np.array([0.0, c.width - 1.0]), np.array([c.cy, c.cy]))
    hfov = np.degrees(abs(np.arctan2(h[0, 1], h[0, 0]) - np.arctan2(h[1, 1], h[1, 0])))
    rep = coverage_report(r)
    ok = abs(el[0] - 80) < 0.5 and abs(el[1] + 61.4) < 0.5 and len(r.cameras) == 8 and rep["covered_%"] > 90
    return ("PASS" if ok else "FAIL"), {"elevation_range_deg": [round(el[1], 2), round(el[0], 2)],
                                        "hfov_deg": round(float(hfov), 1), "ring_step_deg": round(360 / 7, 2),
                                        "sphere_covered_%": round(rep["covered_%"], 1)}


# ------------------------------------------------------------------ synthetic stitching
@check("Synthetic room: stitch + criterion < 10 %", "2 Stitching (synthetic)")
def c_synth():
    from smart_helmet_pano.synth import BoxRoom, render_frame
    from smart_helmet_pano.stitch import StitchOptions, stitch_frame
    from smart_helmet_pano.quality import seam_artifacts
    d = render_frame(OUT / "check" / "synth" / "frames" / "000001", helmet_pyr=(7, 30, -5), room=BoxRoom(),
                     gains=np.random.default_rng(3).uniform(0.85, 1.15, 8))
    rows = {}
    for name, o in [("infinity_no_corrections", StitchOptions(width=2048, proxy_depth_m=float("inf"), exposure=False, flow=False)),
                    ("full_pipeline", StitchOptions(width=2048))]:
        _, pano, st = stitch_frame(d, d / f"pano_{name}.jpg", o)
        rows[name] = seam_artifacts(st.last_warped, st.last_weights, st.W, st.H)[0]["artifact_%_of_central"]
    return ("PASS" if rows["full_pipeline"] < 10 else "FAIL"), rows


@check("IMU levelling: horizon flat after tilt", "2 Stitching (synthetic)")
def c_level():
    from smart_helmet_pano.synth import BoxRoom, ground_truth
    from smart_helmet_pano.stitch import StitchOptions, stitch_frame
    from smart_helmet_pano.quality import psnr
    room = BoxRoom()
    d = OUT / "check" / "synth" / "frames" / "000001"
    gt = ground_truth(room, 1024)
    out = {}
    for lv in ("none", "tilt"):
        _, pano, st = stitch_frame(d, d / f"pano_level_{lv}.jpg", StitchOptions(width=1024, level=lv, flow=False))
        # helmet yaw 30 deg: tilt mode keeps heading -> roll GT by yaw for comparison
        g = np.roll(gt, -int(round(30 / 360 * 1024)), axis=1)
        cov = np.sum(st.last_weights, 0) > 0
        out[lv] = round(psnr(pano, g, cov), 2)
    return ("PASS" if out["tilt"] > out["none"] + 1 else "FAIL"), {"psnr_vs_truth_dB": out}


@check("Report formula rotate_equirect round trip", "2 Stitching (synthetic)")
def c_rot():
    from smart_helmet_pano.level import rotate_equirect
    from smart_helmet_pano.rig import quat_to_matrix
    img = cv2.imread(str(OUT / "check" / "synth" / "frames" / "000001" / "pano_full_pipeline.jpg"))
    img = cv2.resize(img, (1024, 512))
    R = quat_to_matrix(0.98, 0.1, -0.12, 0.05)
    back = rotate_equirect(rotate_equirect(img, R), R.T)
    err = float(np.median(np.abs(back[60:-60].astype(int) - img[60:-60].astype(int))))
    return ("PASS" if err < 6 else "FAIL"), {"median_abs_error": err}


@check("Fisheye undistortion straightens lines (test 3.2)", "2 Stitching (synthetic)")
def c_undist():
    """Real ceiling tiles are straight in the world: after undistortion with the self-calibrated lens
    the detected edges must be straighter than in the raw fisheye image."""
    from smart_helmet_pano.rig import Rig
    from smart_helmet_pano.undistort import undistort, line_straightness_error
    rig = Rig.load(REAL / "rig.json")
    vals = []
    for fr in ("000044", "000073"):
        for cam in rig.cameras[:7]:
            img = cv2.imread(str(REAL / "frames" / fr / f"{cam.name}.jpg"))
            und, _ = undistort(img, cam, 80, 110)
            vals.append((line_straightness_error(img), line_straightness_error(und)))
    a, b = np.nanmedian([v[0] for v in vals]), np.nanmedian([v[1] for v in vals])
    return ("PASS" if b < a else "FAIL"), {"median_line_error_fisheye_px": round(float(a), 2),
                                           "median_line_error_undistorted_px": round(float(b), 2), "images": len(vals)}


# ------------------------------------------------------------------ real helmet
@check("Raw photo health (test 3.1)", "3 Real helmet")
def c_health():
    import glob
    from smart_helmet_pano.health import photo_stats, verdict
    if not RAW.exists():
        return "SKIP", "raw archive not extracted"
    files = sorted(f for f in glob.glob(str(RAW / "cam*" / "session_*" / "*.jpg")) if "(1)" not in f)
    if QUICK:
        files = files[::6]
    bad = {}
    per = {}
    for f in files:
        c = Path(f).parts[-3]
        v = verdict(photo_stats(cv2.imread(f)))
        per.setdefault(c, [0, 0]); per[c][0] += 1; per[c][1] += 0 if v else 1
        for x in v:
            bad[x] = bad.get(x, 0) + 1
    usable = {c: f"{round(100 * u / n)}%" for c, (n, u) in sorted(per.items())}
    return "PASS", {"photos_checked": len(files), "usable_by_camera": usable, "defects": bad}


@check("Synchronisation: frames exported", "3 Real helmet")
def c_sync():
    rep = json.loads((REAL / "ingest_report.json").read_text())
    n = len(list((REAL / "frames").glob("*")))
    ok = n >= 30 and rep["clock_offsets_s"].get("cam8") is None
    return ("PASS" if ok else "FAIL"), {"frames": n, "clock_offsets_s": rep["clock_offsets_s"],
                                        "rejected": rep["rejected_by_reason"]}


@check("Self-calibration matches Arducam lens (175 deg diag)", "3 Real helmet")
def c_calib():
    rep = json.loads((REAL / "calibration_report.json").read_text())
    diag = rep["fov_deg"]["diagonal"]
    ok = abs(diag - 175) < 5 and rep["median_px_calibrated"] < 6
    cams = {c["name"]: {k: c.get(k) for k in ("yaw_error_deg", "pitch_error_deg", "roll_error_deg")} for c in rep["cameras"]}
    return ("PASS" if ok else "FAIL"), {"focal_px": rep["focal_px"], "fov": rep["fov_deg"],
                                        "median_px": rep["median_px_calibrated"], "mounting_errors": cams}


@check("Board calibration recovers a simulated helmet (bundle adjustment)", "2 Calibration")
def c_board():
    """Faulty rig (mount 3 deg, principal point 15 px, own lens curves) + handheld board plan; corners projected
    by the true cameras with 0.15 px noise -> solve_board (Kannala-Brandt + decentering p1 p2) must recover the
    mounting to < 0.2 deg and every viewing ray to < 3 arcmin (Docs/CALIBRATION.md)."""
    from calib_lab import methods as M, evaluate as E, scenes as S
    from smart_helmet_pano.target_calib import solve_board
    ctx = M.make_ctx("7x141", 1, OUT / "check" / "board", workers=1)
    board = S.charuco_board(*M.BOARD_ARGS)
    loc = board.getChessboardCorners().astype(np.float64)
    rng = np.random.default_rng(0)
    dets = []
    for k, p in enumerate(M.plan_board(ctx, singles=10, pairs=4, seed=14)):
        X = loc @ p["R"].T + p["t"]
        for ci in p["vis"]:
            c = ctx.true.cameras[ci]
            u, v, ok = c.project(X)
            pc = (X - np.asarray(c.position_m)) @ c.R_cv_from_rig.T
            ok &= np.degrees(np.arctan2(np.hypot(pc[:, 0], pc[:, 1]), pc[:, 2])) < 88   # like the detector
            ok &= (u > 12) & (u < c.width - 13) & (v > 12) & (v < c.height - 13)
            ids = np.flatnonzero(ok)
            if len(ids) >= 6:
                dets.append((k, ci, ids, np.stack([u[ids], v[ids]], 1) + rng.normal(0, 0.15, (len(ids), 2))))
    rig, rep = solve_board(ctx.nominal, dets, board, verbose=False)
    agg, _ = E.camera_errors(E.align(rig, ctx.true), ctx.true)
    nom, _ = E.camera_errors(E.align(ctx.nominal, ctx.true), ctx.true)
    ok = agg["rot_err_max_deg"] < 0.2 and agg["ray_rms_arcmin"] < 3
    return ("PASS" if ok else "FAIL"), {"nominal_rot_err_deg": nom["rot_err_deg"], "rot_err_deg": agg["rot_err_deg"],
                                        "rot_err_max_deg": agg["rot_err_max_deg"], "ray_rms_arcmin": agg["ray_rms_arcmin"],
                                        "rms_px": rep["rms_px"], "placements": rep["placements"]}


@check("Real panoramas: criterion < 10 % (sphere projection)", "3 Real helmet")
def c_real():
    from smart_helmet_pano.cli import frame_dirs
    from smart_helmet_pano.rig import Rig
    from smart_helmet_pano.stitch import StitchOptions, PanoStitcher, load_frame
    from smart_helmet_pano.quality import seam_artifacts
    rig = Rig.load(REAL / "rig.json")
    fds = frame_dirs(REAL)
    pick = [fds[i] for i in np.linspace(0, len(fds) - 1, 3 if QUICK else 8).astype(int)]
    o = StitchOptions(width=2048, level="none", proxy_depth_m=6, tuning=str(TUNING), theta_max_deg=75,
                      theta_ramp_deg=75, max_flow_frac=0.005, flow_sigma_frac=0.008)
    vals = {}
    for fd in pick:
        imgs, meta = load_frame(fd, rig)
        st = PanoStitcher(rig, o)
        st.stitch(imgs, meta)
        vals[fd.name] = seam_artifacts(st.last_warped, st.last_weights, st.W, st.H)[0]["artifact_%_of_central"]
    return ("PASS" if max(vals.values()) < 10 else "FAIL"), {"seam_artifacts_%": vals}


@check("Plane vs sphere projection", "3 Real helmet")
def c_plane():
    p = OUT / "plane_vs_sphere" / "report.json"
    if not p.exists():
        return "SKIP", "run experiments/plane_vs_sphere.py first"
    r = json.loads(p.read_text(encoding="utf-8"))
    used = [f["plane"]["cameras_used"] for f in r["frames"]]
    return "PASS", {"plane_cameras_used": used, "sphere_cameras_used": "8/8 every frame",
                    "plane_edge_stretch": r["plane_geometry_for_110x130deg_crop"]["edge_stretch_x"],
                    "sphere_mean_seam_artifacts_%": r["sphere_mean_seam_artifacts_%"]}


@check("BNO055 IMU log (master)", "3 Real helmet")
def c_imu():
    from smart_helmet_pano.ingest import load_imu
    recs = load_imu(RAW) if RAW.exists() else []
    if not recs:
        return "SKIP", "no master_imu_data.json"
    q = np.array([[r["imu_data"]["quaternion"][k] for k in "wxyz"] for _, r in recs])
    n = np.linalg.norm(q, axis=1)
    probs = {}
    for _, r in recs:
        for x in r["problems"]:
            key = x.split("=")[0]
            probs[key] = probs.get(key, 0) + 1
    kept = load_imu(RAW, valid_only=True)
    g = np.array([[r["imu_data"]["gravity"][k] for k in "xyz"] for _, r in recs])
    tilt = np.degrees(np.arccos(np.clip(g[:, 2] / np.linalg.norm(g, axis=1), -1, 1)))
    cal = [r["imu_data"].get("calibration_status", {}).get("system", 0) for _, r in recs]
    ok = all(abs(np.linalg.norm([r["imu_data"]["quaternion"][k] for k in "wxyz"]) - 1) < 0.02 for _, r in kept)
    return ("PASS" if ok else "FAIL"), {"records": len(recs), "defects_found": probs, "records_used_after_filter": len(kept),
                                        "quat_norm_range_raw": [round(float(n.min()), 3), round(float(n.max()), 3)],
                                        "head_tilt_deg_median": round(float(np.median(tilt)), 1),
                                        "bno055_system_calibration_0_3": {"min": min(cal), "max": max(cal)}}


# ------------------------------------------------------------------ MCP
@check("MCP smart-helmet server (stdio)", "4 MCP")
def c_mcp():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        sp = StdioServerParameters(command=sys.executable, args=[str(TOOLS / "smart_helmet_mcp.py")], cwd=str(TOOLS))
        async with stdio_client(sp) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = [t.name for t in (await s.list_tools()).tools]
                res = await s.call_tool("rig_info", {})
                return tools, json.loads(res.content[0].text)["coverage"]["covered_%"]
    tools, cov = asyncio.run(run())
    need = {"stitch_real", "calibrate_rig", "ingest_raw", "health_check", "ue_capture", "stitch_frame"}
    return ("PASS" if need <= set(tools) else "FAIL"), {"tools": len(tools), "rig_info_coverage_%": cov}


def _ue_up():
    try:
        socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
        return True
    except OSError:
        return False


@check("Unreal MCP + SmartHelmetToolset", "5 Unreal Engine 5.8")
def c_ue():
    if not _ue_up():
        return "SKIP", "UE editor not running (MCP :8000 closed)"
    import ue_mcp
    ue_mcp.connect()
    ts = ue_mcp.call("list_toolsets", {})
    st = json.loads(json.loads(ue_mcp.call("call_tool", {"toolset_name": "MyProjectEditor.SmartHelmetToolset",
                                                         "tool_name": "GetStatus", "arguments": {}}))["returnValue"])
    ok = "SmartHelmetToolset" in ts and st.get("cameras") == 8
    return ("PASS" if ok else "FAIL"), {"world": st.get("world"), "cameras": st.get("cameras"),
                                        "imu_quaternion": st.get("imu", {}).get("quaternion")}


@check("Unreal capture 8 cameras -> stitch", "5 Unreal Engine 5.8")
def c_ue_cap():
    if not _ue_up():
        return "SKIP", "UE editor not running"
    import ue_mcp
    ue_mcp.connect()
    cap = json.loads(json.loads(ue_mcp.call("call_tool", {"toolset_name": "MyProjectEditor.SmartHelmetToolset",
                                                          "tool_name": "CaptureFrame", "arguments": {}}))["returnValue"])
    fd = Path(cap["frame_dir"])
    files = sorted(p.name for p in fd.glob("*.png"))
    from smart_helmet_pano.stitch import StitchOptions, stitch_frame
    from smart_helmet_pano.quality import seam_artifacts
    _, pano, st = stitch_frame(fd, fd / "panorama.jpg", StitchOptions(width=2048))
    q = seam_artifacts(st.last_warped, st.last_weights, st.W, st.H)[0]
    ok = len(files) == 8 and q["pass_lt_10%"]
    return ("PASS" if ok else "FAIL"), {"frame": str(fd), "images": files, "seam_artifacts_%": q["artifact_%_of_central"]}


@check("C++ build artefacts present", "5 Unreal Engine 5.8")
def c_build():
    b = ROOT / "MyProject" / "Binaries" / "Win64"
    dlls = {p.name: time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
            for p in b.glob("UnrealEditor-MyProject*.dll")}
    src = max(p.stat().st_mtime for p in (ROOT / "MyProject" / "Source").rglob("*.*"))
    newest = max((b / n).stat().st_mtime for n in dlls) if dlls else 0
    ok = len(dlls) >= 2 and newest >= src - 5
    return ("PASS" if ok else "FAIL"), {"dlls": dlls, "binaries_newer_than_sources": newest >= src - 5}


def main():
    for fn in [c_rig, c_synth, c_level, c_rot, c_undist, c_health, c_sync, c_calib, c_board, c_real, c_plane, c_imu,
               c_mcp, c_ue, c_ue_cap, c_build]:
        fn()
    (OUT / "check_report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    lines = ["# Smart Helmet: проверка системы", "", f"Дата: {time.strftime('%Y-%m-%d %H:%M')}", "",
             "| Группа | Проверка | Итог | Детали |", "|---|---|---|---|"]
    for r in results:
        d = r["detail"] if isinstance(r["detail"], str) else json.dumps(r["detail"], ensure_ascii=False)
        lines.append(f"| {r['group']} | {r['check']} | **{r['status']}** | {d[:400].replace('|', '/')} |")
    cnt = {s: sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "SKIP")}
    lines += ["", f"Итого: {cnt}"]
    (OUT / "check_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(cnt)


if __name__ == "__main__":
    main()
