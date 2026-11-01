# Smart Helmet: проверка системы

Дата: 2026-11-01 19:54

| Группа | Проверка | Итог | Детали |
|---|---|---|---|
| 1 Geometry | Rig matches Scheme_Helmet_8_Cameras | **PASS** | {"elevation_range_deg": [-61.36, 79.96], "hfov_deg": 80.2, "ring_step_deg": 51.43, "sphere_covered_%": 93.7} |
| 2 Stitching (synthetic) | Synthetic room: stitch + criterion < 10 % | **PASS** | {"infinity_no_corrections": 1.92, "full_pipeline": 0.2} |
| 2 Stitching (synthetic) | IMU levelling: horizon flat after tilt | **PASS** | {"psnr_vs_truth_dB": {"none": 9.85, "tilt": 13.91}} |
| 2 Stitching (synthetic) | Report formula rotate_equirect round trip | **PASS** | {"median_abs_error": 1.0} |
| 2 Stitching (synthetic) | Fisheye undistortion straightens lines (test 3.2) | **FAIL** | {"line_error_fisheye_px": 23.28, "line_error_undistorted_px": 24.21} |
| 3 Real helmet | Raw photo health (test 3.1) | **PASS** | {"photos_checked": 1257, "usable_by_camera": {"cam1": "77%", "cam2": "82%", "cam3": "78%", "cam4": "88%", "cam5": "81%", "cam6": "76%", "cam7": "72%", "cam8": "86%"}, "defects": {"corrupt": 44, "blur": 136, "cast": 68, "over": 35, "noise": 30, "dark": 4}} |
| 3 Real helmet | Synchronisation: frames exported | **PASS** | {"frames": 73, "clock_offsets_s": {"cam5": 0.0, "cam1": 1.0, "cam2": 0.0, "cam3": 0.0, "cam4": 0.0, "cam6": 0.0, "cam7": 0.0, "cam8": null}, "rejected": {"missing_camera": 25, "corrupt": 37, "over": 10, "blur": 82, "cast": 27, "noise": 9, "noise+cast": 18, "cast+blur": 13, "over+blur": 22, "over+cast+blur": 1, "dark+cast+blur": 2, "dark+cast": 1, "dark+blur": 1}} |
| 3 Real helmet | Self-calibration matches Arducam lens (175 deg diag) | **PASS** | {"focal_px": 664.1, "fov": {"long_side": 152.9, "short_side": 90.9, "diagonal": 174.8}, "median_px": 4.08, "mounting_errors": {"cam2": {"yaw_error_deg": 0.0, "pitch_error_deg": 5.06, "roll_error_deg": 9.29}, "cam3": {"yaw_error_deg": 9.66, "pitch_error_deg": 4.7, "roll_error_deg": -1.05}, "cam4": {"yaw_error_deg": 4.43, "pitch_error_deg": 1.18, "roll_error_deg": -0.63}, "cam5": {"yaw_error_deg": 1 |
| 3 Real helmet | Real panoramas: criterion < 10 % (sphere projection) | **PASS** | {"seam_artifacts_%": {"000001": 2.92, "000011": 2.37, "000021": 1.28, "000031": 1.1, "000042": 2.11, "000052": 1.48, "000062": 1.16, "000073": 0.91}} |
| 3 Real helmet | Plane vs sphere projection | **PASS** | {"plane_cameras_used": ["0/8", "2/8", "2/8", "4/8", "2/8", "2/8"], "sphere_cameras_used": "8/8 every frame", "plane_edge_stretch": 4.47, "sphere_mean_seam_artifacts_%": 1.65} |
| 3 Real helmet | BNO055 IMU log (master) | **PASS** | {"records": 156, "defects_found": {"temperature": 108, "system_uncalibrated": 4, "quaternion_norm": 2}, "records_used_after_filter": 154, "quat_norm_range_raw": [0.993, 2.223], "head_tilt_deg_median": 7.8, "bno055_system_calibration_0_3": {"min": 0, "max": 3}} |
| 4 MCP | MCP smart-helmet server (stdio) | **PASS** | {"tools": 15, "rig_info_coverage_%": 93.75} |
| 5 Unreal Engine 5.8 | Unreal MCP + SmartHelmetToolset | **PASS** | {"world": "HelmetTest", "cameras": 8, "imu_quaternion": [0.9923389, -0.0182155, 0.103461, 0.06502]} |
| 5 Unreal Engine 5.8 | Unreal capture 8 cameras -> stitch | **PASS** | {"frame": "D:\\Games\\Smart Helmet\\MyProject\\Saved\\SmartHelmet\\20261101_190702_Editor\\frames\\000003", "images": ["cam0.png", "cam1.png", "cam2.png", "cam3.png", "cam4.png", "cam5.png", "cam6.png", "top7.png"], "seam_artifacts_%": 0.33} |
| 5 Unreal Engine 5.8 | C++ build artefacts present | **PASS** | {"dlls": {"UnrealEditor-MyProject.dll": "2026-11-01 19:06", "UnrealEditor-MyProjectEditor.dll": "2026-11-01 19:06"}, "binaries_newer_than_sources": true} |

Итого: {'PASS': 14, 'FAIL': 1, 'SKIP': 0}