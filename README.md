# Smart Helmet 360: UE5 + MCP simulation and OpenCV panorama stitching

A digital twin of a helmet with 8 cameras (7 in a ring + 1 at the zenith, Raspberry Pi, ~175° fisheye lenses, BNO055 IMU).
It consists of:
- **Unreal Engine 5.8.** An operator wearing the helmet, synchronous capture by the 8 cameras, a model of the faults of real hardware and a 360° ground-truth capture. Everything is driven through **MCP** (the server built into UE 5.8 plus our own toolset).
- **A "dock station" pipeline in Python / OpenCV.** Ingest of Raspberry Pi sessions with synchronisation repair, photo quality control, lab calibration with a ChArUco board and field re-calibration, self-calibration without a target, photometry from the libcamera tuning file, stitching into an equirectangular panorama. There is an MCP server `smart-helmet`.
- **Research.** How Google Street View captures and stitches; a literature review of optical flow; 5 research directions tested on UE (with ground truth) and on real frames; 9 calibration methods compared on 5 camera-overlap variants.

![Real panorama](Docs/media/11_real_panorama_atrium_4096_depth.jpg)
*Real helmet, atrium: final depth mode, 4096×2048.*

## Main results
| What | Result |
|---|---|
| Geometry per the scheme | The ring sees from −61.4° to +80°, neighbouring cameras overlap by 30°, 93.7 % of the sphere is covered (the nadir is not seen) |
| Projection | **A sphere, not a plane.** A fisheye pixel turns straight into a ray, with no intermediate plane. Plane + `cv2.Stitcher` links 2–4 cameras out of 8 or falls apart ([07](Docs/media/07_plane_vs_sphere_projection.jpg)) |
| Self-calibration of the real helmet | Diagonal FOV **174.8°** against the specified 175° (Arducam IMX219 M12). Measured mounting errors: azimuth −5…+11°, tilt +1…+5.6° |
| Self-calibration in UE (truth known) | Camera rotation error 3–10° (nominal) → **0.15–0.55°** |
| Real data, 1257 photos | Pi clocks off by 4 h, fake-hwclock overlapped 5 sessions, 44 empty files, camera exposure differs by up to 7×. All handled, 73 synchronous frames obtained |
| Test programme criterion (artifacts < 10 % of the central area) | UE 0.4–2.7 %; real helmet **0.27–1.0 %** in the final mode |
| Target-based calibration (stands with known truth, 5 overlap variants) | Seam break 5.5° without calibration, 30–47′ with self-calibration, 2–9′ with target stands. **Best — handheld ChArUco board with the lens decentering model p1, p2: 0.5′ (0.1 px at 4096).** Overlap of 19–39° barely affects calibration. Details: [Docs/CALIBRATION.md](Docs/CALIBRATION.md) |
| Final panorama after calibration | Depth mode, 4096×2048: PSNR **17.1 dB** against 17.2 dB for the true helmet parameters (self-calibration 13.0, no calibration 9.7) — [panorama](Docs/media/21_final_panorama_charuco_4096.jpg), [seam close-ups](Docs/media/23_final_seam_closeups.jpg) |
| Field maintenance | Mounts moved by ~1° → stale lab calibration 77′ at the seams; **lab lens + rotations from ordinary frames** bring it back to 1.5–3′ |
| UE captures, final panoramas | Calibrated once on the synchronous dataset: PSNR **19.9 dB** vs ground truth (true parameters 21.4 dB on frame 4); with camera desynchronisation up to 0.5 s — 12.9 dB, limited by sync, not by calibration ([RESULTS.md](Docs/RESULTS.md)) |
| Best seam method | **Depth (Depth Anything V2 + triangulation from flow) + global spline fit.** On real frames seams 1.65 % → 0.80 %. Details: [Docs/RESULTS.md](Docs/RESULTS.md) |

| UE: operator and 8 cameras | UE: ground truth vs methods |
|---|---|
| ![](Docs/media/01_ue_operator_with_helmet.jpg) ![](Docs/media/02_ue_8_fisheye_cameras.jpg) | ![](Docs/media/05_ue_eval_gt_vs_methods.jpg) |

| Real helmet: base vs final mode | Photometry: before and after |
|---|---|
| ![](Docs/media/10_real_base_vs_depth.jpg) | ![](Docs/media/13_real_no_photometric_correction.jpg) ![](Docs/media/14_real_with_photometric_correction.jpg) |

## Documents
- [Docs/SIMULATION.md](Docs/SIMULATION.md) — how the simulation was done, tools, fault model, course of work.
- [Docs/RESULTS.md](Docs/RESULTS.md) — directions 1–5 compared on UE (datasets A/B with ground truth) and on real frames; final UE panoramas.
- [Docs/CALIBRATION.md](Docs/CALIBRATION.md) — helmet calibration: camera fault model, stands (ChArUco board, marker points, projected chessboards, calibration cube, self-calibration), camera overlap, lens model, field drift.
- [Docs/optical_flow_review.md](Docs/optical_flow_review.md) — literature review of optical flow for 360° stitching.
- [Docs/results/](Docs/results/) — raw metrics, calibration reports, system check.
- [Docs/media/](Docs/media/) — all illustrations.

## Layout
```
MyProject/                 UE 5.8 (Source, Config, .uproject only; assets are copied from the engine templates)
  Source/MyProject/SmartHelmet/   SmartHelmetComponent (8 cameras, fisheye LUT, faults, IMU), Character, GameMode, TestRange
  Source/MyProjectEditor/         SmartHelmetToolset: helmet MCP tools
Tools/
  smart_helmet_pano/       rig, stitch, advanced (depth mode), calibrate (self-calibration), bundle (bundle adjustment),
                           targets (ArUco / ChArUco / chessboards), target_calib (board + field re-calibration),
                           photometric, ingest, health, quality, level, undistort, synth, cli
  calib_lab/               calibration study: fault model, stands, methods, evaluation, report, final panoramas
  smart_helmet_mcp.py      "dock station" MCP server (17 tools)
  ue_mcp.py, ue_build_scene.py, setup_ue_assets.py, check_all.py
  experiments/             directions.py, ue_dataset.py, ue_eval.py, ue_final.py, real_eval.py, plane_vs_sphere.py
Data/imx219_160d.json      libcamera tuning file (ISP gamma, ALSC)
```

## Quick start
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # torch with CUDA — see requirements.txt
cd Tools
python -m smart_helmet_pano synth ../Output/synth --frames 2              # synthetic frames with parallax
python -m smart_helmet_pano stitch ../Output/synth --width 2048
# real helmet: folders cam1..cam8 from the Raspberry Pis
python -m smart_helmet_pano health    <raw_root>
python -m smart_helmet_pano ingest    <raw_root> ../Output/real_helmet
python -m smart_helmet_pano board --out charuco_board.png                # A2 board for the lab calibration
python -m smart_helmet_pano calibrate-board <board_session> --rig <nominal rig.json>   # lens + mounting of every camera
python -m smart_helmet_pano refine-mounts ../Output/real_helmet --lens <lab rig.json>  # field re-calibration of the mounts
python -m smart_helmet_pano calibrate ../Output/real_helmet --frames 16   # self-calibration if there is no lab calibration
python -m smart_helmet_pano stitch    ../Output/real_helmet --real --width 4096   # depth mode on a GPU, base mode without one
python check_all.py                                                     # full system check
```

**Unreal.** Run `python Tools/setup_ue_assets.py`, open `MyProject/MyProject.uproject`, then `python Tools/ue_build_scene.py`. Keys in Play: `C` — capture, `T` — auto-capture every 5 s, `V` — helmet view.

**MCP in Claude Code:**
```bash
claude mcp add --transport http unreal http://localhost:8000/mcp
claude mcp add smart-helmet -- <venv>/python.exe Tools/smart_helmet_mcp.py
```

## Hardware recommendations (from the results)
1. A hardware trigger, or NTP plus an RTC on every Pi. Desynchronisation of up to 1 s is the main source of errors and spoils self-calibration.
2. Fixed exposure and white balance broadcast by the master Pi (`AeEnable=False`, `AwbEnable=False`).
3. A lab calibration of every helmet with a handheld ChArUco board (`calibrate-board`) or in a calibration cube: the lenses have different distortion and an offset principal point, which cannot be separated from a scene. In the field, keep the lens and update only the mounts (`refine-mounts`). Details: [Docs/CALIBRATION.md](Docs/CALIBRATION.md).
4. Verify writing of the photos (`fsync`, file-size check) and validate the BNO055 quaternions.
