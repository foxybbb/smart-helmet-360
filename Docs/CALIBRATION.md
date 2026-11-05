# Helmet calibration: stands, methods and camera overlap

Code: [`Tools/calib_lab/`](../Tools/calib_lab) (camera model, stands, methods, evaluation), shared bundle adjustment [`Tools/smart_helmet_pano/bundle.py`](../Tools/smart_helmet_pano/bundle.py), target detectors [`targets.py`](../Tools/smart_helmet_pano/targets.py), production board calibration [`target_calib.py`](../Tools/smart_helmet_pano/target_calib.py).
Raw numbers: [`Docs/results/calibration_study.json`](results/calibration_study.json).

## Summary

- **9 calibration methods** were compared on **5 camera-overlap variants** of the helmet (6–8 cameras, 141° and 160° lenses, neighbour overlap 19–39°). The cameras have the faults of real hardware: mounting error up to 7.5°, principal point offset up to 32 px, every lens with its own curve, lens decentering.
- **Without calibration** the seam break is 5.5°, with the **current self-calibration on the scene** 30–47′ (6–9 px at 4096). **Target stands** give 2–9′, mean camera rotation error 0.04–0.2°. The exceptions are a cube built "per drawing" and a projector without a tape measure.
- **The most practical method is a handheld ChArUco board:** seam break 2.0′, and with the lens model p1, p2 **0.5′, i.e. 0.1 px**. **A calibration cube** is the fastest to capture (4 poses) with comparable accuracy, but the pose of its panels must be estimated: a cube "per drawing" is 2–3× worse. **Marker points** do not need surveyed coordinates. **Projected chessboards** need one tape measure; without it they give 15–18′ on 6 cameras.
- **Overlap of 19–39° barely affects calibration accuracy.** It matters for a real scene: 7–8 cameras have less parallax and occlusion than 6. **The least panorama distortion** comes from 7–8 cameras × 141° or 7 × 160° together with target-based calibration and depth-mode stitching. The current scheme 7 × 141° is already in that zone.
- **Indoors, parallax** with sphere stitching (57′) is 15–30× larger than the error of a good calibration. Accurate calibration only pays off together with the depth mode.
- **After the mounts move by ~1°** the lab calibration goes stale (77′ at the seams). The lab lens plus rotations from ordinary frames bring it back to 1.5–3′.
- **The final panorama** after ChArUco + p1, p2 in the depth mode almost matches the one stitched with the true parameters: PSNR 17.1 vs 17.2 dB ([section](#final-panoramas-after-calibration)).
- New in the pipeline: commands `board`, `calibrate-board`, `refine-mounts`, the same MCP tools, a lens model with decentering, and a check in `check_all.py`.

## How it was done

The stands are ray-traced in Python, without Unreal (the editor with MCP was not running for this work). For calibration this is even more accurate: the geometry is analytic, the truth is known to a micron, every ray starts at the real centre of its lens (10.5 cm from the helmet centre), so the parallax between the cameras is real. Every image goes through the "true" camera with all its faults, then through the same OpenCV detector that will run on real photos.

### Camera faults (like real hardware)

Nominal is what the pipeline knows: the helmet drawing and the lens datasheet (equidistant fisheye, principal point in the centre, mounting angles per the scheme). The "true" helmet is drawn from a seed:

| What | Model | Spread |
|---|---|---|
| Lens curve | the design differs from equidistant (between equidistant and equisolid), the datasheet FOV holds at the edge | k_base = (−0.035, 0.004, −0.0005, 0) |
| Lens to lens | own k1..k4 for every lens | σ k1 = 0.010, k2 = 0.004, k3 = 0.0015, k4 = 0.0005 |
| Focal length | assembly spread | σ 1.5 % |
| Principal point | M12 holder tolerance | σ 15 px per axis |
| Decentering (tangential distortion) | missing from the pipeline's original Kannala-Brandt model; this work adds p1, p2 | σ p1, p2 = 4·10⁻⁴ (≈1 px at the edge) |
| Pixel | aspect ratio | σ 0.1 % |
| Camera mounting | rotation | σ 3° per axis |
| Lens centre | offset | σ 2 mm per axis |
| Photometry | exposure, white balance, vignetting, noise, lens blur | σ 0.4 stops, 6 %, cos⁴ 0.3–0.6, σ 2, σ 0.7 px |

## Stands and methods

![Stands](media/15_calibration_stands.jpg)
*View from one camera on every stand; detected targets in green and red.*

| Method | What is needed | Capture | What is estimated |
|---|---|---|---|
| **Handheld ChArUco board** | board of 11×8 squares of 50 mm (A2, 570×420 mm), `python -m smart_helmet_pano board` | helmet static; the board 14 times in front of every camera (image edges included) and 5 times in every overlap of neighbours (ring–ring and ring–top) | lens of every camera (fx, fy, cx, cy, k1..k4, p1, p2), mounting, pose of every board placement |
| **ArUco marker points** | ~150 markers of 15 cm stuck anywhere on the walls and ceiling of a 6×5×3 m room | helmet on a tripod, 8 poses (azimuth sweep of almost 2 ring steps, tilt up to ±24°, roll up to ±12°) | the same + coordinates of all markers + helmet poses; scale from the marker size |
| **Marker points, positions surveyed** | the same + a total-station survey (σ 2 mm) | the same | the same, marker coordinates held near the survey |
| **Chessboards projected onto the walls** | a projector on a stand, 14 aims: 2 per wall, 4 on the ceiling, 2 on the floor; boards of 10×7 squares 1–1.6 m, keystoned by the oblique projection | 10 tripod poses; the operator only knows "roughly which wall the board is on" | the same + board corners (unknown), every board lies in the plane of its wall |
| **Square room (cube), per drawing** | 3×3×3 m cube, ChArUco of 12×12 squares of 25 cm on every face; built with errors of ±3 mm, ±0.15°, print ±0.07 % | 4 tripod poses in the centre | lens, mounting, poses; corner coordinates taken from the drawing |
| **Square room (cube), panels estimated** | the same cube | the same images | the same, but the pose of every face is estimated (precise print, imprecise assembly) |
| **Self-calibration on the scene** | nothing, an ordinary room | 8 ordinary frames from different places | the current `calibrate.py`: shared lens + per camera, rotations; principal point in the centre |
| **Hybrid** | lens from the ChArUco calibration | 8 ordinary frames | lens fixed, rotations and centres from SIFT matches |

Detection runs on the raw fisheye frames: ArUco (AprilTag refinement), ChArUco (saddle points), chessboards (`findChessboardCornersSB`, and for faint projected boards the classic detector with sub-pixel refinement; orientation from the colour of the four corner squares). Everything is solved by one bundle adjustment (`bundle.py`): sparse Levenberg–Marquardt, Huber loss, first with the datasheet lens, then everything together.

## How it was evaluated

- **Calibration error.** Rotation of every camera and lens centre against the truth; **ray error** — the angle between the true and the estimated ray of every pixel over the whole image (after the common rotation of the helmet, which does not affect the seams). The unit is the arcminute: 1′ = 1/60°; in a 4096 px panorama one pixel = 5.3′.
- **Panorama distortion, exact.** For every panorama pixel the stitcher picks a camera pixel using the estimated calibration; we trace the **true** ray of that pixel into the scene and see where it really lands. *Distortion* is the angle between where the panorama pixel should look and where it looks. *Seam break* is the angle between the two cameras at a seam line: the visible "kink" of straight lines.
  - **∞** — points at infinity: calibration only, no parallax;
  - **room, 3 m proxy sphere** — a 6×5×3 m test room with furniture at 0.8–2 m, as in the base stitching mode;
  - **room, true depth** — every pixel is placed at its true distance (an ideal depth mode), only the calibration error at real distances remains.
- **Image.** PSNR and SSIM of the stitched test room against a ground-truth panorama from the helmet centre; artifacts per the test programme criterion (< 10 % of the central area).

## Results: scheme 7 × 141°

| Method | Camera rotation, ° (mean / max) | Lens centre, mm | Ray error, ′ | Panorama distortion ∞, ′ (mean / p95) | Seam break ∞, ′ (mean / p95) | Seams at true depth, ′ | PSNR at true depth, dB | Capture |
|---|---|---|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | 4.02 / 7.56 | 4.4 | 208.9 | 200.9 / 445.0 | 331.8 / 707.6 | 319.7 | 10.20 | — |
| Self-calibration on the scene (current pipeline) | 1.29 / 2.57 | 5.9 | 57.2 | 46.7 / 75.1 | 34.6 / 69.5 | 37.2 | 12.48 | 8 ordinary frames |
| Handheld ChArUco board | 0.08 / 0.17 | 1.3 | 5.4 | 5.2 / 8.1 | 2.0 / 4.2 | 1.6 | 17.41 | 182 board placements (544 images) |
| ArUco marker points, positions unknown | 0.06 / 0.08 | 2.7 | 6.1 | 6.2 / 8.8 | 4.4 / 9.0 | 2.5 | 17.99 | 8 tripod poses |
| Marker points, positions surveyed | 0.06 / 0.09 | 2.9 | 6.2 | 6.4 / 9.9 | 3.7 / 7.6 | 2.5 | 17.92 | 8 tripod poses + total-station survey |
| Chessboards projected onto the walls | 0.07 / 0.12 | 3.3 | 5.5 | 4.6 / 7.7 | 4.3 / 8.9 | 2.1 | 18.09 | 10 tripod poses × 14 projector aims |
| Projected chessboards + tape (width of every board) | 0.08 / 0.12 | 3.3 | 4.8 | 4.5 / 8.1 | 4.0 / 7.8 | 1.7 | 18.08 | same + 28 tape measurements |
| Square room (cube), geometry from the drawing | 0.32 / 0.77 | 6.0 | 14.9 | 8.8 / 20.4 | 10.9 / 19.3 | 4.2 | 16.24 | 4 poses in the cube |
| Square room (cube), panel poses estimated | 0.04 / 0.06 | 1.9 | 4.3 | 4.5 / 6.5 | 3.5 / 5.9 | 1.4 | 18.21 | 4 poses in the cube |
| ChArUco lens + rotations from the scene | 0.09 / 0.20 | 1.7 | 5.9 | 5.5 / 9.9 | 2.3 / 4.5 | 1.9 | 17.35 | 8 ordinary frames + lens from ChArUco |
| True parameters (limit) | 0.00 / 0.00 | 0.0 | 0.0 | 0.3 / 1.0 | 0.0 / 0.0 | 0.1 | 18.81 | — |

*"Seams at true depth" — the seam break in the test room when every pixel is placed at its distance: this is how an ideal depth mode works, only the calibration error remains. 1′ ≈ 0.19 px of a 4096 panorama.*

![Distortion maps](media/17_distortion_maps.jpg)
*Panorama distortion at infinity over the whole sphere (0–60′), white lines are seams. Top to bottom: no calibration, self-calibration, handheld ChArUco board, cube with estimated panels.*

![Seam close-ups](media/18_seam_closeups.jpg)
*Seam between cam0 and cam1, test room, stitched at true depth. Without calibration and after self-calibration the grid lines break at the seam; after target-based calibration they do not.*

**What it shows:**
1. **Without calibration the panorama is unusable:** 3.3° of distortion, seam breaks of 5.5° (≈ 63 px at 4096). Mounting errors of 2.6–7.5° and a focal length 3–6 % off the datasheet are not forgiven.
2. **Self-calibration on the scene (current pipeline) gives 35–47′ (≈ 7–9 px at a seam).** It cannot estimate the principal point (the M12 holder offsets it by up to 32 px), and the different lens curves cannot be separated from one scene. This agrees with the conclusion of [RESULTS.md](RESULTS.md) for the real helmet.
3. **Target stands (except the cube "per drawing") give 2–5′ at the seams (0.4–0.9 px) and 4–6′ of absolute distortion**, an order of magnitude better than self-calibration. Camera rotations 0.04–0.09°, lens centres 1–3 mm.
4. **The best seams come from the handheld ChArUco board (2.0′) and the hybrid (2.3′).** A board placed in the overlap of two cameras ties them directly together. **The best absolute distortion comes from the cube with estimated panels (4.3′ ray error, 1.4′ at the seams at true depth)** with the fastest capture (4 poses).
5. **The cube "per drawing" is 2–3× worse (10.9′ at the seams).** An assembly error of ±3 mm at 1.5 m is already 1–2 px, and the bundle adjustment smears it over the focal length and the rotations. The stand geometry must be estimated, not taken from the drawing.
6. **Marker points without surveyed coordinates (6.2′/4.4′) are almost as good as those surveyed with a total station (6.4′/3.7′).** A total station is not needed: the bundle adjustment finds the marker positions itself.
7. **Projected chessboards work on the scheme (4.6′/4.3′),** but they have no metric scale. On 6 cameras this gave 15–18′ at the seams. One tape measure (width of every board, 28 measurements) plus a weak lens prior fixes this on all variants: 4.0–5.1′.

### Ray error across the image

| Method | 0-15° | 15-30° | 30-45° | 45-60° | 60-75° | 75-90° |
|---|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | 192.4 | 205.4 | 211.4 | 201.0 | 197.0 | 234.7 |
| Self-calibration on the scene (current pipeline) | 49.2 | 49.1 | 49.1 | 48.1 | 51.6 | 138.4 |
| Handheld ChArUco board | 5.8 | 5.8 | 5.6 | 5.2 | 4.9 | 4.9 |
| ArUco marker points, positions unknown | 6.2 | 6.2 | 6.2 | 6.0 | 5.8 | 6.8 |
| Marker points, positions surveyed | 6.6 | 6.5 | 6.4 | 6.0 | 5.6 | 6.3 |
| Chessboards projected onto the walls | 4.4 | 4.5 | 4.7 | 4.9 | 5.8 | 11.7 |
| Projected chessboards + tape (width of every board) | 4.4 | 4.5 | 4.5 | 4.7 | 5.2 | 6.5 |
| Square room (cube), geometry from the drawing | 9.3 | 10.2 | 11.0 | 12.1 | 14.1 | 51.0 |
| Square room (cube), panel poses estimated | 5.0 | 4.8 | 4.5 | 4.0 | 3.3 | 4.0 |
| ChArUco lens + rotations from the scene | 6.3 | 6.2 | 6.0 | 5.6 | 5.2 | 5.3 |

For all target methods the error is almost the same from the centre to the edge of the image (4–7′ ≈ 1–1.5 px). This is **neither noise nor lack of data but the limit of the lens model:** real lenses have decentering (tangential distortion), and the pipeline's original Kannala-Brandt model does not describe it. Self-calibration and the cube "per drawing" get much worse towards the edge of the image (51–138′ at 75–90°).

### Lens model: decentering

The 4–7′ limit of all stands is lens decentering, which the Kannala-Brandt model lacks. The pipeline's model was extended with two coefficients p1, p2 (`rig.py`, `bundle.py`). Old `rig.json` files load as before, with p = 0. The same images were re-solved with the new model:

| Method | Sample | Ray error, ′ | Distortion ∞, ′ | Seam break ∞, ′ | Seams at true depth, ′ |
|---|---|---|---|---|---|
| Handheld ChArUco board | seed 1 | 5.4 → **1.7** | 5.2 → **1.5** | 2.0 → **0.5** | 1.6 → **0.5** |
| Square room (cube), panel poses estimated | seed 1 | 4.3 → **1.5** | 4.5 → **1.3** | 3.5 → **1.8** | 1.4 → **1.0** |
| Projected chessboards + tape (width of every board) | seed 1 | 4.8 → **4.3** | 4.5 → **3.2** | 4.0 → **3.0** | 1.7 → **1.2** |
| ArUco marker points, positions unknown | seed 1 | 6.1 → **4.9** | 6.2 → **4.6** | 4.4 → **4.6** | 2.5 → **2.3** |
| Handheld ChArUco board | seed 2 | 6.1 → **3.1** | 6.5 → **2.9** | 6.5 → **1.1** | 4.1 → **1.0** |
| Square room (cube), panel poses estimated | seed 2 | 7.4 → **2.7** | 7.8 → **2.5** | 8.8 → **3.1** | 2.9 → **1.6** |
| Projected chessboards + tape (width of every board) | seed 2 | 17.3 → **3.7** | 7.5 → **3.4** | 8.6 → **4.1** | 3.8 → **2.1** |
| ArUco marker points, positions unknown | seed 2 | 7.1 → **4.3** | 6.8 → **3.6** | 6.8 → **5.0** | 4.3 → **2.7** |

- **The ChArUco board gains the most:** seams 0.5–1.1′ (0.1–0.2 px at 4096), distortion 1.5–2.9′. The board is close, its corners are precise saddle points, and there are many of them all over the image, so there is enough data to separate the decentering from the other parameters.
- The cube and the projector improve by 1.3–3.5×.
- The model barely helps ArUco markers: they are limited by the accuracy of the marker corners (0.3–0.6 px vs 0.03–0.15 px for chessboard corners).
- `calibrate-board` estimates p1, p2 by default. `refine-mounts` with such a lens gives 1.5′ at the seams after mount drift, against 2.8′ with the old model.

### With sphere stitching, parallax matters more than calibration

| Method | Seam break, ′ | PSNR, dB | SSIM | Artifacts without flow, % | Artifacts with flow, % |
|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | 339.5 | 10.16 | 0.253 | 1.34 | 1.14 |
| Self-calibration on the scene (current pipeline) | 56.0 | 12.36 | 0.364 | 0.34 | 0.28 |
| Handheld ChArUco board | 57.6 | 14.05 | 0.550 | 0.40 | 0.35 |
| ArUco marker points, positions unknown | 56.6 | 14.15 | 0.560 | 0.39 | 0.36 |
| Marker points, positions surveyed | 57.6 | 14.10 | 0.556 | 0.40 | 0.36 |
| Chessboards projected onto the walls | 58.3 | 14.15 | 0.566 | 0.40 | 0.35 |
| Projected chessboards + tape (width of every board) | 57.5 | 14.16 | 0.567 | 0.40 | 0.35 |
| Square room (cube), geometry from the drawing | 57.3 | 14.03 | 0.538 | 0.39 | 0.33 |
| Square room (cube), panel poses estimated | 57.5 | 14.17 | 0.568 | 0.39 | 0.35 |
| ChArUco lens + rotations from the scene | 57.6 | 14.05 | 0.551 | 0.40 | 0.35 |
| True parameters (limit) | 57.6 | 14.29 | 0.583 | 0.40 | 0.36 |

In a room with furniture at 0.8–2 m, base stitching on a 3 m sphere gives **a 57′ break for every method, even for the true parameters.** This is the parallax between lenses 10.5 cm from the centre, and it is 15–30× larger than the error of a good calibration. So accurate calibration only pays off together with the depth mode ([RESULTS.md](RESULTS.md)): at true depth the seams are 1.4–2.5′ against 37′ for self-calibration. The artifacts per the test programme criterion (< 10 %) are below 0.5 % for every method: the criterion does not trigger on such breaks, it only catches gross errors.

## Overlap: which variant gives the least distortion

| Variant | Cameras | Image FOV, ° (H × V) | px/° | Neighbour overlap at the horizon, ° | Ring–top overlap, ° | Sphere covered, % | Double coverage, % |
|---|---|---|---|---|---|---|---|
| 6 cams × 141° | 7 | 79.5 × 141.4 | 13.58 | 19.0 | 47.8 | 93.7 | 52.7 |
| 7 cams × 141° (scheme) | 8 | 79.5 × 141.4 | 13.58 | 27.7 | 47.9 | 93.7 | 70.0 |
| 8 cams × 141° | 9 | 79.5 × 141.4 | 13.58 | 34.0 | 48.0 | 93.8 | 83.0 |
| 6 cams × 160° | 7 | 90.0 × 160.0 | 12.0 | 29.5 | 65.1 | 97.0 | 71.6 |
| 7 cams × 160° | 8 | 90.0 × 160.0 | 12.0 | 38.9 | 65.1 | 97.1 | 87.5 |

![Overlap variants](media/16_overlap_variants.jpg)
*How many cameras see every direction (equirectangular projection, the bottom is the nadir). Green — one camera, orange — two.*

**Seam break at infinity, ′** (calibration only; bold — the best overlap variant for every method):

| Method \ variant | 6 cams × 141° | 7 cams × 141° (scheme) | 8 cams × 141° | 6 cams × 160° | 7 cams × 160° |
|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | **291.3** | 331.8 | 307.0 | 306.5 | 347.2 |
| Self-calibration on the scene (current pipeline) | 42.3 | 34.6 | 49.8 | **22.7** | 34.6 |
| Handheld ChArUco board | 2.1 | **2.0** | 2.5 | 2.9 | 2.5 |
| ArUco marker points, positions unknown | 6.9 | 4.4 | 4.7 | 6.0 | **4.2** |
| Marker points, positions surveyed | 4.3 | **3.7** | 4.1 | 4.8 | 4.2 |
| Chessboards projected onto the walls | 14.7 | **4.3** | 6.5 | 18.3 | 5.4 |
| Projected chessboards + tape (width of every board) | 4.2 | **4.0** | 4.5 | 4.1 | 5.1 |
| Square room (cube), geometry from the drawing | 11.5 | 10.9 | 9.9 | 9.8 | **9.1** |
| Square room (cube), panel poses estimated | 3.6 | **3.5** | 4.0 | 4.1 | 3.9 |
| ChArUco lens + rotations from the scene | 3.0 | **2.3** | 2.6 | 2.8 | 2.5 |
| True parameters (limit) | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |

![Seam break at infinity](media/19_seam_break_chart.jpg)

**Seam break in the test room at true depth, ′** (calibration + occlusion behind near objects):

| Method \ variant | 6 cams × 141° | 7 cams × 141° (scheme) | 8 cams × 141° | 6 cams × 160° | 7 cams × 160° |
|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | **281.8** | 319.7 | 294.2 | 293.8 | 336.1 |
| Self-calibration on the scene (current pipeline) | 40.5 | 37.2 | 47.1 | **22.7** | 37.4 |
| Handheld ChArUco board | 2.2 | **1.6** | 1.8 | 2.6 | 1.8 |
| ArUco marker points, positions unknown | 3.2 | 2.5 | **2.4** | 3.3 | 2.6 |
| Marker points, positions surveyed | 2.9 | 2.5 | **2.4** | 3.1 | 2.7 |
| Chessboards projected onto the walls | 17.2 | **2.1** | 3.5 | 192.2 | 3.3 |
| Projected chessboards + tape (width of every board) | 2.4 | 1.7 | 2.0 | 2.0 | **1.7** |
| Square room (cube), geometry from the drawing | 4.8 | 4.2 | 4.1 | 4.0 | **3.6** |
| Square room (cube), panel poses estimated | 1.9 | **1.4** | 1.4 | 2.1 | 1.5 |
| ChArUco lens + rotations from the scene | 2.8 | 1.9 | **1.8** | 2.7 | 2.0 |
| True parameters (limit) | 0.6 | 0.1 | **0.1** | 0.7 | **0.1** |

![Seam break at true depth](media/20_seam_break_truedepth_chart.jpg)

**Seam break in the test room with stitching on a 3 m sphere, ′** (base mode, parallax):

| Method \ variant | 6 cams × 141° | 7 cams × 141° (scheme) | 8 cams × 141° | 6 cams × 160° | 7 cams × 160° |
|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | **302.4** | 339.5 | 310.7 | 322.9 | 358.0 |
| Self-calibration on the scene (current pipeline) | 45.2 | 56.0 | **37.5** | 59.6 | 61.5 |
| Handheld ChArUco board | 57.0 | 57.6 | **46.4** | 55.9 | 57.1 |
| ArUco marker points, positions unknown | 55.5 | 56.6 | **45.4** | 54.7 | 56.3 |
| Marker points, positions surveyed | 57.0 | 57.6 | **46.3** | 56.1 | 57.2 |
| Chessboards projected onto the walls | 66.5 | 58.3 | **44.6** | 193.9 | 58.6 |
| Projected chessboards + tape (width of every board) | 56.8 | 57.5 | **45.9** | 55.6 | 56.7 |
| Square room (cube), geometry from the drawing | 54.7 | 57.3 | **45.8** | 53.4 | 56.5 |
| Square room (cube), panel poses estimated | 56.7 | 57.5 | **46.4** | 55.8 | 57.0 |
| ChArUco lens + rotations from the scene | 56.6 | 57.6 | **46.2** | 56.0 | 56.8 |
| True parameters (limit) | 56.6 | 57.6 | **46.1** | 55.6 | 57.0 |

**Panorama distortion at infinity, ′:**

| Method \ variant | 6 cams × 141° | 7 cams × 141° (scheme) | 8 cams × 141° | 6 cams × 160° | 7 cams × 160° |
|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | **182.2** | 200.9 | 201.1 | 190.4 | 210.8 |
| Self-calibration on the scene (current pipeline) | 30.2 | 46.7 | 39.5 | **28.2** | 49.1 |
| Handheld ChArUco board | **4.1** | 5.2 | 6.2 | 4.6 | 5.4 |
| ArUco marker points, positions unknown | 6.4 | **6.2** | 6.2 | 6.9 | 6.2 |
| Marker points, positions surveyed | 6.7 | **6.4** | 6.7 | 7.1 | 6.7 |
| Chessboards projected onto the walls | 7.7 | **4.6** | 6.0 | 9.4 | 6.1 |
| Projected chessboards + tape (width of every board) | 4.7 | **4.5** | 5.5 | 4.8 | 6.0 |
| Square room (cube), geometry from the drawing | 10.6 | 8.8 | 10.5 | **7.7** | 7.9 |
| Square room (cube), panel poses estimated | **4.0** | 4.5 | 5.3 | 4.6 | 5.1 |
| ChArUco lens + rotations from the scene | **4.4** | 5.5 | 6.7 | 4.8 | 5.4 |
| True parameters (limit) | **0.3** | **0.3** | **0.3** | **0.3** | **0.3** |

**Mean camera rotation error, °:**

| Method \ variant | 6 cams × 141° | 7 cams × 141° (scheme) | 8 cams × 141° | 6 cams × 160° | 7 cams × 160° |
|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | 4.103 | 4.019 | **3.967** | 4.103 | 4.019 |
| Self-calibration on the scene (current pipeline) | **1.056** | 1.294 | 1.294 | 1.147 | 1.449 |
| Handheld ChArUco board | 0.086 | 0.079 | 0.082 | 0.072 | **0.071** |
| ArUco marker points, positions unknown | 0.060 | **0.058** | **0.058** | 0.061 | **0.058** |
| Marker points, positions surveyed | 0.062 | **0.056** | 0.060 | 0.062 | 0.057 |
| Chessboards projected onto the walls | 0.074 | 0.075 | 0.085 | 0.086 | **0.071** |
| Projected chessboards + tape (width of every board) | 0.073 | 0.078 | 0.091 | 0.078 | **0.071** |
| Square room (cube), geometry from the drawing | 0.310 | 0.318 | 0.365 | 0.290 | **0.260** |
| Square room (cube), panel poses estimated | **0.038** | 0.039 | 0.040 | 0.047 | 0.043 |
| ChArUco lens + rotations from the scene | 0.102 | 0.085 | 0.084 | 0.094 | **0.082** |
| True parameters (limit) | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |

**Conclusions on overlap:**
1. **For target-based calibration an overlap of 19–39° hardly matters.** The seam break from calibration is 2–5′ on every variant. Targets tie the cameras together even at 19° of overlap.
2. **Overlap matters for the seams in a real scene.** At true depth 6 cameras give 1.9–3.3′, 7–8 cameras 1.4–2.7′ (for every method 6 cameras are 0.4–0.9′ worse), and even with the true parameters 6 cameras keep 0.6′ against 0.1′. This is occlusion behind near objects: the baseline between neighbouring lenses is 10.5 cm with 6 cameras, 9.1 cm with 7, 8.0 cm with 8. In the base mode on a sphere, 8 cameras give 46′ against 55–57′ for 6–7 cameras.
3. **160° lenses cover more of the sphere** (97 % vs 93.7 %, down to −70.6° instead of −61°) and give a large overlap with the top camera (65° vs 48°). The price is resolution: 12.0 px/° vs 13.6 px/° (−12 %). The seams of 6 × 160° are not better than those of 7 × 141° at the same ~29° overlap.
4. **Self-calibration reacts to overlap erratically** (23–50′): it is limited not by the number of matches but by the fact that the principal point and the lens curve cannot be determined from a scene.
5. **The least panorama distortion:** 7–8 cameras × 141° (overlap 28–34°) or 7 × 160° (39°, if the bottom of the sphere is needed), target-based calibration and depth-mode stitching. The current scheme 7 × 141° is already in that zone; moving to 8 cameras only gains in parallax (−20 % in the base mode).

## Field drift of the mounts

Scenario: the helmet was calibrated in the lab (ChArUco and cube), then the mounts moved after a knock or transport. Every camera rotated by another σ 1° per axis (from 0.25° to 2.4° for different cameras), the lens centre moved by σ 1 mm; the lenses did not change. Everything is compared with the new truth.

| Variant | Camera rotation, ° (mean / max) | Ray error, ′ | Distortion ∞, ′ | Seam break ∞, ′ | Seams at true depth, ′ | PSNR at true depth, dB |
|---|---|---|---|---|---|---|
| No calibration | 4.00 / 6.34 | 201.4 | 196.0 | 335.2 | 323.1 | 10.18 |
| Lab ChArUco, not updated | 1.22 / 2.51 | 54.9 | 50.8 | 76.6 | 74.0 | 12.57 |
| Lab cube, not updated | 1.20 / 2.46 | 54.7 | 50.5 | 76.9 | 73.8 | 12.62 |
| Self-calibration in the field | 1.27 / 2.56 | 70.7 | 48.6 | 36.2 | 38.0 | 12.40 |
| Hybrid: lens from ChArUco + rotations from the scene | 0.08 / 0.21 | 6.2 | 6.0 | 2.8 | 2.1 | 17.31 |
| Hybrid: lens from the cube + rotations from the scene | 0.07 / 0.10 | 4.7 | 4.7 | 3.0 | 2.2 | 18.02 |
| True parameters | 0.00 / 0.00 | 0.0 | 0.3 | 0.0 | 0.1 | 18.81 |

- **After such a move the stale lab calibration is worse than self-calibration:** 77′ against 36′ at the seams. Its lens is right, but its rotations no longer are.
- **The hybrid restores lab accuracy.** The lens comes from the lab, rotations and lens centres from 8 ordinary frames, about 2 minutes of computation. Seams 2.8–3.0′, rotations 0.07–0.08°, the move of every mount recovered to within 0.03°. The principal point and the lens curve do not change from knocks, so they need to be determined only once.
- In the pipeline this is the command `python -m smart_helmet_pano refine-mounts <session> --lens <lab rig.json>` and the MCP tool `refine_mounts`.

## Second helmet sample

The same stands on another draw of faults (seed 2). This sample has twice the decentering: 2 px at the image edge on average against 1 px.

| Method | Camera rotation, ° (mean / max) | Lens centre, mm | Ray error, ′ | Panorama distortion ∞, ′ (mean / p95) | Seam break ∞, ′ (mean / p95) | Seams at true depth, ′ | PSNR at true depth, dB | Capture |
|---|---|---|---|---|---|---|---|---|
| No calibration (drawing + lens datasheet) | 2.95 / 5.79 | 4.3 | 164.8 | 151.5 / 353.4 | 253.9 / 560.1 | 240.2 | 10.79 | — |
| Self-calibration on the scene (current pipeline) | 1.28 / 2.10 | 7.0 | 40.8 | 33.2 / 74.3 | 29.6 / 66.8 | 25.1 | 13.48 | 8 ordinary frames |
| Handheld ChArUco board | 0.19 / 0.43 | 1.3 | 6.1 | 6.5 / 9.3 | 6.5 / 13.8 | 4.1 | 17.75 | 182 board placements (525 images) |
| ArUco marker points, positions unknown | 0.11 / 0.16 | 5.2 | 7.1 | 6.8 / 11.0 | 6.8 / 12.2 | 4.3 | 18.06 | 8 tripod poses |
| Marker points, positions surveyed | 0.12 / 0.18 | 4.9 | 6.8 | 6.5 / 11.2 | 6.6 / 11.5 | 4.2 | 18.11 | 8 tripod poses + total-station survey |
| Chessboards projected onto the walls | 0.21 / 0.32 | 5.3 | 13.7 | 8.1 / 16.4 | 11.1 / 20.5 | 9.9 | 17.54 | 10 tripod poses × 14 projector aims |
| Projected chessboards + tape (width of every board) | 0.21 / 0.30 | 5.5 | 17.3 | 7.5 / 13.2 | 8.6 / 18.2 | 3.8 | 18.27 | same + 28 tape measurements |
| Square room (cube), geometry from the drawing | 0.33 / 0.77 | 7.4 | 13.0 | 10.1 / 21.6 | 14.4 / 30.5 | 5.2 | 17.01 | 4 poses in the cube |
| Square room (cube), panel poses estimated | 0.09 / 0.12 | 3.6 | 7.4 | 7.8 / 13.1 | 8.8 / 15.3 | 2.9 | 18.15 | 4 poses in the cube |
| ChArUco lens + rotations from the scene | 0.21 / 0.46 | 2.7 | 5.5 | 5.1 / 9.8 | 5.4 / 9.2 | 4.7 | 18.16 | 8 ordinary frames + lens from ChArUco |
| True parameters (limit) | 0.00 / 0.00 | 0.0 | 0.0 | 0.3 / 1.0 | 0.0 / 0.0 | 0.2 | 18.94 | — |

The order "no calibration ≫ self-calibration ≫ targets" repeats, as do the conclusions about the cube "per drawing" and the projector without a tape measure. Within the group of target stands the leader changes (here the hybrid and the cube), and the overall level is higher (5–9′ at the seams) because of the decentering. With the p1, p2 model the difference almost disappears: ChArUco gives 1.1′ at the seams.

## Final panoramas after calibration

Test room 6×5×3 m with furniture at 0.8–2 m, scheme 7 × 141°, stitched as in the production pipeline: depth mode (Depth Anything V2 + triangulation + global seam fit), 4096 × 2048. The images are the same; only the calibration changes.

![Final panorama](media/21_final_panorama_charuco_4096.jpg)
*Final panorama after calibration with a handheld ChArUco board and the lens model p1, p2 (4096 × 2048). The dark band at the bottom is the nadir, which the camera ring does not see.*

| Calibration | PSNR vs ground truth, dB | SSIM | Seam artifacts, % | PSNR / SSIM at true depth |
|---|---|---|---|---|
| No calibration | 9.7 | 0.39 | 1.02 | 10.2 / 0.40 |
| Self-calibration on the scene | 13.0 | 0.51 | 0.31 | 12.5 / 0.49 |
| **ChArUco + p1, p2** | **17.1** | **0.81** | **0.20** | **18.7 / 0.91** |
| True parameters (limit) | 17.2 | 0.82 | 0.17 | 18.9 / 0.92 |

- **After ChArUco + p1, p2 calibration the panorama practically matches the one stitched with the true helmet parameters:** PSNR 17.1 vs 17.2 dB, SSIM 0.81 vs 0.82. Calibration no longer limits the quality.
- The remaining gap to the ground truth (an ideal camera in the helmet centre) is the exposure, vignetting and noise of the cameras, and, with depth estimated by Depth Anything, its errors. At true depth the same calibration gives 18.7 dB and SSIM 0.91.
- Self-calibration lifts the panorama from 9.7 to 13.0 dB, but straight lines still kink at the seams. Without calibration every seam tears.

![All variants](media/22_final_panoramas_overview.jpg)
*Top to bottom: no calibration, self-calibration, ChArUco + p1, p2, true parameters, ground truth. The ground truth is brighter because it has no camera exposure or vignetting.*

![Close-ups](media/23_final_seam_closeups.jpg)
*Three places of every panorama: ring seam cam0 | cam1, ring–top seam, a column 1.1 m from the helmet. Without calibration the grid lines and the ceiling arcs tear, after self-calibration the kinks are smaller, after ChArUco + p1, p2 the seams are invisible.*

Metrics: [`Docs/results/final_panoramas_7x141.json`](results/final_panoramas_7x141.json).

## Recommendations for the real helmet

1. **A lab calibration of every helmet with targets is mandatory.** Self-calibration leaves 30–47′ (6–9 px at a seam), targets 2–9′, a ChArUco board with the p1, p2 model 0.5–1.1′.
2. **The main method is a handheld ChArUco board** (best seams, nothing to build; 0.5′ at the seams with p1, p2):
   ```bash
   python -m smart_helmet_pano board --out charuco_board.png          # print A2 at 100 %, check 50 mm with a ruler
   python -m smart_helmet_pano calibrate-board <session> --rig <nominal rig.json>
   ```
   The helmet is static. The board — about 14 placements in front of every camera (image edges are a must) and about 5 in every overlap of neighbours, ring–top included. Every placement is one synchronous frame `frames/<id>/<cam>.jpg`. Without a hardware trigger, hold the board still until all Raspberry Pis have fired. The report shows the share of image cells (5 × 8) that contain board corners: below 90 % coverage, add placements in the empty places.
3. **For series production — a 3 × 3 × 3 m calibration cube with ChArUco on its faces:** 4 poses and about 10 s of computation, the best absolute distortion. Estimate the panel poses; do not trust the drawing.
4. **For checks and adjustment on site — ArUco marker points:** stickers in any room, 8 tripod poses, no total station needed. Projected chessboards are fine if the width of every projected board is measured with a tape.
5. **After knocks and transport** the factory calibration goes stale fastest (see "Field drift"). Keep the lab lens and update the mounts from ordinary frames: `python -m smart_helmet_pano refine-mounts <session> --lens <lab rig.json>`.
6. **The lens model with decentering (p1, p2)** is already part of `calibrate-board`: seams 2.0′ → 0.5′. Without it every stand hits 4–7′.
7. **Stitch in the depth mode.** On a sphere the indoor parallax (57′) outweighs any calibration.

## How to reproduce

```bash
cd Tools
python -m calib_lab.run_study --variants 7x141 6x141 8x141 6x160 7x160 --seeds 1   # ≈ 2 h on 8 cores, cached
python -m calib_lab.drift                                                         # field drift of the mounts
python -m calib_lab.run_study --variants 7x141 --seeds 2                          # second helmet sample
python -m calib_lab.lens_model 7x141                                              # lens model + p1, p2 (both samples)
python -m calib_lab.summarize                                                     # tables + Docs/results/calibration_study.json
python -m calib_lab.figures                                                       # Docs/media/15–20
python -m calib_lab.final_panos --variant 7x141 --width 4096                      # final panoramas (depth mode, GPU)
python check_all.py                                                               # includes the board calibration check
```
Images and intermediate results go to `Output/calib_study/<variant>_s<seed>/` (≈ 4 thousand images).

## Limitations

- The stands are ray-traced in Python, not in Unreal: geometry and parallax are exact, lighting is simplified, there is no rolling shutter, defocus or flare. The images are JPEG with noise, blur and vignetting; detection uses the same OpenCV functions as for real photos.
- The capture is synchronous. The effect of Raspberry Pi desynchronisation is studied separately in [RESULTS.md](RESULTS.md); calibration needs a static scene.
- The main run uses one draw of faults (seed 1) and is repeated on a second one for the 7 × 141° scheme. The order of the methods within the target group depends on the sample; the conclusions about self-calibration, the cube "per drawing", the projector without a tape measure, the drift and the lens model repeat.
- Housing deformation from temperature and wear was not modelled; the mount drift is only a rotation and a shift of the lens.
