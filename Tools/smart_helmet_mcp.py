"""MCP server "smart-helmet" – the dock-station side of the system.

Tools for Claude (or any MCP client):
  * list_sessions / list_frames      – browse captured data (UE sim or real helmet)
  * stitch_frame / stitch_session    – OpenCV panorama + quality (artifact %) report
  * level_panorama                   – horizon levelling with a BNO055 quaternion
  * undistort_camera                 – fisheye undistortion sweep (test programme 3.2)
  * rig_info                         – rig geometry and coverage
  * synth_session                    – synthetic test frames with real parallax
  * ue_capture / ue_call             – trigger the helmet in Unreal via UE's own MCP server

Register:  claude mcp add smart-helmet -- "<venv>/python.exe" "<this file>"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
try:  # mcp SDK 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP, Image
except ImportError:  # mcp SDK 1.x
    from mcp.server.fastmcp import FastMCP, Image

sys.path.insert(0, str(Path(__file__).parent))
from smart_helmet_pano.cli import frame_dirs  # noqa: E402
from smart_helmet_pano.rig import Rig, default_rig, coverage_report  # noqa: E402
from smart_helmet_pano.stitch import StitchOptions, stitch_frame  # noqa: E402
from smart_helmet_pano.quality import seam_artifacts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [ROOT / "MyProject" / "Saved" / "SmartHelmet", ROOT / "Output"]
UE_MCP_URL = os.environ.get("UE_MCP_URL", "http://127.0.0.1:8000/mcp")

mcp = FastMCP("smart-helmet")


import contextlib  # noqa: E402


def _quiet():
    """Library code prints progress; stdout is the MCP channel, so send it to stderr."""
    return contextlib.redirect_stdout(sys.stderr)


def _sessions():
    out = []
    for base in DATA_DIRS:
        if base.is_dir():
            out += [p for p in base.iterdir() if p.is_dir() and ((p / "rig.json").exists() or (p / "frames").is_dir())]
    return sorted(set(out), key=lambda p: p.stat().st_mtime, reverse=True)


@mcp.tool()
def list_sessions() -> str:
    """List capture sessions (newest first) with their frame counts."""
    rows = [{"session": str(s), "frames": len(frame_dirs(s))} for s in _sessions()]
    return json.dumps(rows, ensure_ascii=False, indent=1)


@mcp.tool()
def list_frames(session: str) -> str:
    """List frame folders of a session and whether a panorama exists."""
    rows = [{"frame": str(f), "panorama": (f / "panorama.jpg").exists()} for f in frame_dirs(Path(session))]
    return json.dumps(rows, ensure_ascii=False, indent=1)


@mcp.tool(name="stitch_frame")
def stitch_frame_tool(frame_dir: str, width: int = 4096, proxy_depth_m: float = 4.0, level: str = "tilt",
                      flow: bool = True, exposure: bool = True, blend: str = "multiband") -> str:
    """Stitch the 8 helmet images of one frame into an equirectangular panorama.

    level: none | tilt (IMU pitch/roll levelling, keep heading) | full.
    Returns paths + artifact metric (test programme 3.2: pass if < 10 % of central area)."""
    opts = StitchOptions(width=width, proxy_depth_m=proxy_depth_m, level=level, flow=flow,
                         exposure=exposure, blend=blend)
    fd = Path(frame_dir)
    out, pano, st = stitch_frame(fd, fd / "panorama.jpg", opts)
    q, _ = seam_artifacts(st.last_warped, st.last_weights, st.W, st.H)
    r = {"panorama": str(out), "seconds": round(st.timings["total"], 2), **q}
    (fd / "quality.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return json.dumps(r, ensure_ascii=False)


@mcp.tool()
def stitch_session(session: str, width: int = 2048, only_missing: bool = True) -> str:
    """Stitch every frame of a session (dock-station batch job). Returns a summary."""
    res = []
    for fd in frame_dirs(Path(session)):
        if only_missing and (fd / "panorama.jpg").exists():
            continue
        res.append(json.loads(stitch_frame_tool(str(fd), width=width)))
    ok = sum(r["pass_lt_10%"] for r in res)
    return json.dumps({"stitched": len(res), "passed": ok, "frames": res}, ensure_ascii=False, indent=1)


@mcp.tool()
def preview_panorama(path: str, max_width: int = 1600) -> Image:
    """Return a downscaled JPEG of a panorama (or any image) so it can be viewed."""
    p = Path(path)
    if p.is_dir():
        p = p / "panorama.jpg"
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    if img.shape[1] > max_width:
        img = cv2.resize(img, (max_width, int(img.shape[0] * max_width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Image(data=buf.tobytes(), format="jpeg")


@mcp.tool()
def level_panorama(path: str, quaternion_wxyz: list[float], keep_yaw: bool = True) -> str:
    """Level an existing equirectangular panorama with a BNO055 quaternion [w,x,y,z] (report sec. 4.2.1)."""
    from smart_helmet_pano.level import level_with_quaternion
    img = cv2.imread(path)
    out = Path(path).with_name(Path(path).stem + "_leveled.jpg")
    cv2.imwrite(str(out), level_with_quaternion(img, quaternion_wxyz, keep_yaw))
    return str(out)


@mcp.tool()
def undistort_camera(image: str, cam: str = "cam0", fov_h_deg: float = 75, fov_v_deg: float = 110) -> str:
    """Fisheye undistortion of one camera image + straight-line error sweep over crops (test 3.2 steps 1-2)."""
    from smart_helmet_pano.undistort import undistort, sweep_fov
    try:
        from smart_helmet_pano.stitch import find_rig
        rig = Rig.load(find_rig(Path(image).parent))
    except FileNotFoundError:
        rig = default_rig()
    c = next(x for x in rig.cameras if x.name == cam)
    img = cv2.imread(image)
    und, _ = undistort(img, c, fov_h_deg, fov_v_deg)
    out = Path(image).with_name(Path(image).stem + "_undistorted.png")
    cv2.imwrite(str(out), und)
    return json.dumps({"output": str(out), "sweep": sweep_fov(img, c)}, indent=1)


@mcp.tool()
def rig_info(rig_json: str = "") -> str:
    """Camera layout of the helmet (default = Scheme_Helmet_8_Cameras) and sphere coverage."""
    rig = Rig.load(rig_json) if rig_json else default_rig()
    rep = {k: round(v, 2) for k, v in coverage_report(rig).items() if k != "count_map"}
    return json.dumps({"coverage": rep, "rig": rig.to_json()}, indent=1)


@mcp.tool()
def synth_session(out_dir: str = str(ROOT / "Output" / "synth_mcp"), frames: int = 2,
                  mount_error_deg: float = 0.0) -> str:
    """Render synthetic helmet frames (textured room, true parallax, random head tilt and camera gains)."""
    from smart_helmet_pano.cli import cmd_synth
    import argparse
    with _quiet():
        cmd_synth(argparse.Namespace(out=out_dir, frames=frames, seed=1, mount_error=mount_error_deg))
    return json.dumps({"session": out_dir, "frames": [str(f) for f in frame_dirs(Path(out_dir))]})


# ------------------------------------------------------------------ real helmet (dock station)
REAL_TUNING = str(ROOT / "Data" / "imx219_160d.json")


@mcp.tool()
def health_check(raw_root: str) -> str:
    """Per-camera quality of raw Pi photos (cam1..cam8 folders): empty files, noise, clipping, colour cast, blur."""
    import argparse
    from smart_helmet_pano.cli import cmd_health
    with _quiet():
        return json.dumps(cmd_health(argparse.Namespace(raw_root=raw_root)), ensure_ascii=False, indent=1)


@mcp.tool()
def ingest_raw(raw_root: str, out_dir: str, top_camera: str = "cam1") -> str:
    """Raw Pi folders -> synchronous frames. Fixes per-Pi clock offsets and fake-hwclock session overlaps
    (matching by trigger number + image content), rejects unusable photos, rotates portrait ring cameras."""
    from smart_helmet_pano.ingest import export
    with _quiet():
        return json.dumps(export(raw_root, out_dir, top=top_camera, verbose=False), ensure_ascii=False, indent=1)


@mcp.tool()
def calibrate_rig(session: str, frames: int = 16, f_init: float = 680.0) -> str:
    """Self-calibrate the real helmet (no checkerboard): shared fisheye lens (f, k1..k4) + every camera's
    rotation from feature matches across many frames; writes <session>/rig.json + calibration_report.json."""
    from smart_helmet_pano.calibrate import calibrate_session
    with _quiet():
        _, rep = calibrate_session(session, n_frames=frames, f_init=f_init, verbose=False)
    return json.dumps(rep, ensure_ascii=False, indent=1)


@mcp.tool()
def calibrate_board(session: str, nominal_rig: str = "", squares_x: int = 11, squares_y: int = 8,
                    square_mm: float = 50.0) -> str:
    """Lab calibration with a handheld ChArUco board (best seams in the simulation study, 2 arcmin):
    session/frames/<id>/<cam>.jpg, one board placement per frame, helmet static. Estimates every camera's
    lens (fx, fy, cx, cy, k1..k4), mounting rotation and lens centre; writes <session>/rig.json and
    board_calibration_report.json (incl. per-camera image coverage - aim for > 90 %)."""
    from smart_helmet_pano.rig import Rig
    from smart_helmet_pano.target_calib import calibrate_board_session, make_board
    with _quiet():
        _, rep = calibrate_board_session(session, Rig.load(nominal_rig) if nominal_rig else None,
                                         make_board((squares_x, squares_y), square_mm / 1000), verbose=False)
    return json.dumps(rep, ensure_ascii=False, indent=1)


@mcp.tool()
def refine_mounts(session: str, lab_rig: str, frames: int = 8) -> str:
    """Field re-calibration after knocks / transport: the lens from the lab rig.json is kept, mounting rotations
    and lens centres are re-estimated from ordinary frames (simulation: 1 deg drift, seams 77 -> 2.8 arcmin).
    Writes <session>/rig.json + mount_refinement_report.json (rotation change of every camera)."""
    from smart_helmet_pano.target_calib import refine_mounts_session
    with _quiet():
        _, rep = refine_mounts_session(session, lab_rig, n_frames=frames, verbose=False)
    return json.dumps(rep, ensure_ascii=False, indent=1)


@mcp.tool()
def stitch_real(frame_or_session: str, width: int = 4096, limit: int = 0, mode: str = "depth") -> str:
    """Stitch real helmet photos with the self-calibrated rig: ISP-gamma linearisation + ALSC devignetting
    (imx219_160d.json), per-camera exposure/WB gains, nearest-axis seams.
    mode="depth" (default, GPU): Depth Anything V2 + triangulated depth instead of a 6 m sphere, then a global
    spline seam warp (best in tests: seam artifacts 1.65 % -> 0.80 %). mode="base": sphere proxy + DIS flow (CPU)."""
    import argparse
    from smart_helmet_pano.cli import cmd_stitch
    a = argparse.Namespace(path=frame_or_session, width=width, depth=4.0, level="tilt", blend="multiband",
                           no_flow=False, no_exposure=False, debug=False, tuning=REAL_TUNING, theta_max=None,
                           real=True, limit=limit, mode=mode)
    with _quiet():
        return json.dumps(cmd_stitch(a), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ UE bridge
def _ue_rpc(tool: str, args: dict):
    import ue_mcp
    ue_mcp.URL = UE_MCP_URL
    ue_mcp.connect()
    return ue_mcp.call(tool, args)


@mcp.tool()
def ue_call(toolset: str, tool: str, arguments_json: str = "{}") -> str:
    """Call any toolset tool of the Unreal Editor MCP server (e.g. toolset 'MyProjectEditor.SmartHelmetToolset')."""
    return str(_ue_rpc("call_tool", {"toolset_name": toolset, "tool_name": tool,
                                     "arguments": json.loads(arguments_json)}))


@mcp.tool()
def ue_capture(stitch: bool = True, width: int = 2048) -> str:
    """Trigger a synchronous 8-camera capture on the helmet in the running UE editor/PIE, then stitch it."""
    res = _ue_rpc("call_tool", {"toolset_name": "MyProjectEditor.SmartHelmetToolset",
                                "tool_name": "CaptureFrame", "arguments": {}})
    try:
        frame = json.loads(json.loads(res)["returnValue"])["frame_dir"]
    except Exception:
        return f"UE capture failed: {res}"
    if not stitch:
        return frame
    return stitch_frame_tool(frame, width=width)


if __name__ == "__main__":
    mcp.run()
