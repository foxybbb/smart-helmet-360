"""Generate a UE5 test set with ALL real-hardware imperfections + ground truth, via UE's MCP server.

  python experiments/ue_dataset.py [frames] [seed]

Per camera (random, reproducible): mounting error, Kannala-Brandt k spread, trigger delay (operator
walks and turns in PIE), auto-exposure / white-balance spread, cos^4 vignetting, sensor noise.
Each frame also stores an ideal world-aligned 360 cube from the helmet centre (ref_face*.png).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ue_mcp  # noqa: E402

TS = "MyProjectEditor.SmartHelmetToolset"
APP = "EditorToolset.EditorAppToolset"

PROFILE = dict(Seed=1, MountErrorStdDeg=3.0, LensKStd=0.03, TriggerJitterMaxS=0.5, ExposureStdStops=0.8,
               WhiteBalanceStd=0.08, VignetteStrength=0.5, NoiseStd=2.0, bCaptureReference=True)


def call(toolset, tool, **args):
    r = ue_mcp.call("call_tool", {"toolset_name": toolset, "tool_name": tool, "arguments": args})
    try:
        v = json.loads(r)["returnValue"]
        return json.loads(v) if isinstance(v, str) and v.strip().startswith("{") else v
    except Exception:
        return r


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    jitter = float(sys.argv[3]) if len(sys.argv) > 3 else PROFILE["TriggerJitterMaxS"]
    ue_mcp.connect()
    if call(APP, "IsPIERunning") is not True:
        call(APP, "StartPIE", options={"bSimulate": False})
        time.sleep(6)
    prof = dict(PROFILE, Seed=seed, TriggerJitterMaxS=jitter)
    rig = call(TS, "ConfigureImperfections", **prof)
    print("imperfections:", prof)
    call(TS, "ConfigureCapture", FaceResolution=1024, ExposureEV100=12.5, MountErrorStdDeg=prof["MountErrorStdDeg"])
    call(TS, "ConfigureImperfections", **prof)   # ConfigureCapture changed std -> redraw with same seed
    print(call(TS, "SetOperatorMotion", bEnabled=True, WalkInput=0.5, TurnDegPerSec=25.0))
    time.sleep(3)
    frames = []
    for i in range(n):
        r = call(TS, "CaptureFrame")
        fd = Path(r["frame_dir"])
        t0 = time.time()
        while not (fd / "meta.json").exists() and time.time() - t0 < 30:
            time.sleep(0.3)
        frames.append(str(fd))
        print("frame", fd.name, "complete" if (fd / "meta.json").exists() else "TIMEOUT")
        time.sleep(2.5)
    call(TS, "SetOperatorMotion", bEnabled=False, WalkInput=0.0, TurnDegPerSec=0.0)
    session = str(Path(frames[0]).parents[1])
    rigj = call(TS, "GetRigCalibration")                     # rig.json with the drawn truth
    Path(session, "rig.json").write_text(json.dumps(rigj, indent=1) if isinstance(rigj, dict) else str(rigj))
    print(json.dumps({"session": session, "frames": frames}))
    call(APP, "StopPIE")


if __name__ == "__main__":
    main()
