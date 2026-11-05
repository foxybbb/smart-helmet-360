# Optical flow for 360° stitching from a multi-camera helmet: literature review (November 2026)

[unverified] marks entries whose arXiv ID or venue was not confirmed by a search.

## 1. Classical methods (used in the stitcher today)
- **Brox et al., ECCV 2004.** Variational flow. The gradient-constancy term makes it robust to brightness changes.
- **DIS: Kroeger et al., ECCV 2016, arXiv:1603.03590.** Our current method (`cv2.DISOpticalFlow`). Hundreds of Hz on a CPU. Weak on large displacements and plain ceilings.
- **DeepFlow (ICCV 2013) and EpicFlow (CVPR 2015, arXiv:1501.02565).** Sparse matches with edge-aware interpolation. Fill textureless areas well.
- **Shum & Szeliski, IJCV 2000.** Deghosting: every image is warped halfway towards its neighbour. This is exactly our symmetric half-warp.
- **Google "Seamless Street View Panoramas", blog 2017.** A global spline warp field fitted to confident flow points (Ceres solver). Explicitly compensates parallax, camera desynchronisation and calibration errors.
- **Jump (Anderson et al., SIGGRAPH Asia 2016) and Facebook Surround360.** Flow between neighbouring cameras is used to synthesise intermediate views. The top and bottom fisheye cameras are stitched separately.
- **Megastereo (Richardt et al., CVPR 2013).** Ray densification from flow and vertical-parallax correction.

## 2. Neural flow
| Model | Reference | For us |
|---|---|---|
| RAFT | Teed & Deng, ECCV 2020, arXiv:2003.12039 | ONNX available in the OpenCV Zoo. Reliable on large displacements. Seconds per frame on a CPU |
| GMFlow / UniMatch | Xu et al., CVPR 2022 / TPAMI 2023 | One model gives both flow and stereo depth |
| **SEA-RAFT** | Wang, Lipson, Deng, ECCV 2024, arXiv:2405.14793 | 2.3× faster than RAFT. Gives **per-pixel uncertainty**. Best candidate for the GPU |
| DPFlow | CVPR 2025, arXiv:2503.14880 | Generalises to high resolutions, up to 8K |
| FlowSeek | ICCV 2025, arXiv:2509.05297 | Built on a depth foundation model. 10–15 % more accurate than SEA-RAFT |
| WAFT | ICLR 2026 [ID unverified] | Warping instead of a cost volume. Leads the Spring, Sintel and KITTI benchmarks |
| NeuFlow v2 | 2024 [unverified] | Runs on edge devices and ARM. The only realistic option for a CPU |

## 3. Parallax-tolerant stitching
- **Mesh warps:** APAP (CVPR 2013), SPHP (CVPR 2014), Parallax-tolerant (Zhang & Liu, CVPR 2014). The last one chooses the seam so that it avoids near objects.
- **UDIS (TIP 2021) and UDIS++ (ICCV 2023, arXiv:2302.08207).** Thin-plate-spline (TPS) warp and a learned seam.
- **StabStitch (ECCV 2024) and StabStitch++ (TPAMI 2025).** Bidirectional warp towards a virtual middle plane.
- **DSFN (NeurIPS 2025, arXiv:2510.21396) and PIS3R (2025).** Stitching guided by depth and 3D reconstruction. **Trend of 2025–26: with large parallax, 3D / depth replaces 2D warps.**

## 4. Flow on spherical and fisheye images
- **OmniFlowNet** (ICPR 2020), **LiteFlowNet360 / FLOW360** (ICPR 2020) [ID unverified], **PanoFlow** (T-ITS 2023, arXiv:2202.13388).
- **Tangent images** (Eder et al., CVPR 2020) and SphereNet: convolutions aware of the projection distortion.
- **Takeaway for us.** The seam between the top camera and the ring lies near the pole of the equirectangular projection, where the distortion is largest. That is why ceiling lines break. Flow for this seam should be computed in a gnomonic (tangent) projection and the result mapped back.

## 5. Desynchronisation and different exposure
- **Brightness-invariant terms:** census transform, census / ternary loss (UnFlow, AAAI 2018), gradient constancy.
- **Exposure normalisation before computing the flow.** Gain compensation (Brown & Lowe, IJCV 2007), HDR flow (Kalantari & Ramamoorthi, SIGGRAPH 2017).
- **Desynchronisation of up to 1 s is mostly head rotation.** For every frame, first estimate the 3-DoF rotation of every camera and leave only the remainder to the flow. This is an engineering trick: no paper about unsynchronised helmets specifically was found.

## 6. Depth instead of a proxy sphere
- **Casual 3D Photography** (Hedman et al., 2017).
- **OmniMVS** (ICCV 2019). A wide-baseline fisheye rig and a spherical sweep, the closest analogue of our helmet.
- **MatryODShka** (ECCV 2020), **OmniPhotos** (2020).
- **Monocular depth as an initial guess:** Depth Anything V2 (NeurIPS 2024), Depth Anywhere (2024).
- **Gaussian Splatting for 360°:** ODGS (NeurIPS 2024), OmniGS, SC-OmniGS (self-calibration), ODGS-SLAM (CVPR 2026). Needs a GPU and minutes of optimisation per scene.

## What to implement, by priority
1. **Per-frame rotation of every camera plus exposure normalisation before the flow.** Removes the effect of desynchronisation and exposure differences. High gain, low effort, runs on a CPU.
2. **Top-camera seam in a tangent projection and a global spline warp, as in Street View.** Straightens ceiling lines. Medium effort.
3. **Flow → depth → reprojection by depth instead of a 6 m sphere.** The main solution for objects closer than 1 m. Medium to high effort.
4. **On the dock-station GPU: SEA-RAFT instead of DIS.** Its uncertainty replaces the forward–backward check.
5. **Seam selection around near objects:** graph cut weighted by flow confidence or by depth.

## Metrics and data for evaluation
- **Metrics:** PSNR / SSIM in the overlap zones, residual flow after warping, share of pixels failing the forward–backward check, **straightness of lines** (LSD) across seams, gradient energy along the seam, NIQE / BRISQUE, time per frame.
- **Data:** synthetic frames with our exact geometry (UE5 is already there, plus ground-truth depth and flow, controlled desynchronisation and exposure differences), UDIS-D, FlowScape (PanoFlow), OmniThings / OmniHouse, our own set of frames with near objects and ceilings.
