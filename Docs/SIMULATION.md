# How the simulation was done: Unreal Engine 5.8 + MCP

## Tools
| What | With |
|---|---|
| Engine | Unreal Engine 5.8, C++ modules `MyProject` (runtime) and `MyProjectEditor` (editor) |
| Driving the editor from Claude | The MCP server built into UE 5.8 by Epic: plugins `ModelContextProtocol`, `ToolsetRegistry`, `EditorToolset`. Address `http://localhost:8000/mcp`, streamable HTTP transport |
| Our own helmet MCP tools | `USmartHelmetToolset` (`Source/MyProjectEditor`) registers in ToolsetRegistry and shows up as `MyProjectEditor.SmartHelmetToolset` |
| Bridge and automation | `Tools/ue_mcp.py` (mini MCP client), `Tools/ue_build_scene.py` (level build), `Tools/experiments/ue_dataset.py` (dataset generation) |
| Stitching and evaluation | Python 3.14, OpenCV 5, NumPy / SciPy, PyTorch 2.11 + CUDA (RAFT, Depth Anything V2), MCP SDK 2.x (`Tools/smart_helmet_mcp.py`) |
| Assets | UE 5.8 templates: Manny / Quinn mannequin and animations, Enhanced Input, LevelPrototyping. Sky and sun from the SimBlank level. The helmet is built from BasicShapes |

## Helmet model in UE (`SmartHelmetComponent`)
- 7 ring cameras with a 51.43° step, pitched up by 9.3°, radius 10.5 cm; 1 camera pointing at the zenith. All as in `Scheme_Helmet_8_Cameras`.
- Every camera renders a cube map (5 faces), then a lookup table (LUT) turns it into a 1080×1920 fisheye with the Kannala-Brandt model. This is the same model the OpenCV stitcher uses.
- The helmet is attached to the mannequin's `head` bone, so at the moment of a shot the IMU (logged in BNO055 format) sees the real head sway.
- Capture: key `C`, auto-capture `T` every 5 s (as in the test programme), MCP command `CaptureFrame`.
- Output: `rig.json` (nominal calibration), `frames/NNNNNN/cam*.png`, `meta.json` (helmet pose at the trigger and IMU), `imu.jsonl`.

## Faults of real hardware (`ConfigureImperfections`)
Per-camera spread, reproducible from a seed:

| Parameter | What it models | In the tests |
|---|---|---|
| `MountErrorStdDeg` | camera mounting error | σ 3° (up to 10° drawn) |
| `LensKStd` | different distortion coefficients k1..k4 | σ k1 = 0.03 |
| `TriggerJitterMaxS` | unsynchronised Raspberry Pis | up to 0.5 s; the operator walks and turns at 25°/s |
| `ExposureStdStops`, `WhiteBalanceStd` | independent AE and AWB | 0.8 stops, 0.08 |
| `VignetteStrength`, `NoiseStd` | lens fall-off and sensor noise | 0.5, σ 2 |
| `bCaptureReference` | ground truth: an ideal 360° cube from the helmet centre at the trigger | on |

`rig.json` stores the nominal values, as on the real helmet; the truth goes into separate `true_*` fields, for evaluation only.

## Course of work (key points)
1. **Documents.** Geometry and FOV (−61.4°…+80°, 30° overlap) come from the scheme. The BNO055 format and the horizon-levelling formulas from the quaternion come from the stage-2 report. The 5 s interval and the criterion "artifacts < 10 % of the central area" come from the test programme.
2. **Street View analysis.** Calibrated projection instead of feature search on every frame, projection onto a proxy depth, seam correction by flow, exposure equalisation. All of this became the basis of the pipeline.
3. **MCP in UE 5.8.** The built-in server was enabled and our own toolset written. The `HelmetTest` level and the operator were built with MCP commands.
4. **First stitch.** A synthetic room with real parallax gave 3.4 % artifacts, frames from UE 1.2–2.4 %. Sphere vs plane: projecting onto a plane followed by `cv2.Stitcher` links 2–4 cameras out of 8 or falls apart, the spherical projection links 8 out of 8.
5. **Real helmet** (1257 photos). Found and handled:
   - Pi clocks off by 4 h and sessions overlapping because of fake-hwclock;
   - 44 empty files;
   - camera exposure differing by up to 7×;
   - 2 broken BNO055 quaternions.
   Lens self-calibration gave a diagonal FOV of 174.8° against the specified 175°.
6. **Literature review of optical flow**: `Docs/optical_flow_review.md`.
7. **UE test bench with all faults** (datasets A and B) and **the same test on real frames**: `Docs/RESULTS.md`.
8. **The winner went into the pipeline**: the depth mode (Depth Anything V2 + triangulation from inter-camera flow + global spline seam fit). On real frames seam artifacts went down from 1.65 % to 0.80 %.
9. **Calibration study** (stands with targets, overlap variants, field drift, lens model): `Docs/CALIBRATION.md`.

## How to reproduce
```bash
# 1. UE: open MyProject/MyProject.uproject (template assets needed, see README), MCP starts by itself
python Tools/ue_build_scene.py                     # HelmetTest level via MCP
python Tools/experiments/ue_dataset.py 8 1 0       # dataset A: faults without desynchronisation
python Tools/experiments/ue_dataset.py 8 1 0.5     # dataset B: plus desynchronisation up to 0.5 s
python Tools/experiments/ue_eval.py <session_dir> 2048
python Tools/experiments/ue_final.py --width 4096  # final 4096 panoramas of both datasets (depth mode)
```
