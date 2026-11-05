# Directions 1–5: test results (UE5 with ground truth + real helmet)

Scripts: `Tools/experiments/directions.py` (all variants), `ue_dataset.py` (generation in UE), `ue_eval.py`, `real_eval.py`.
Raw numbers: `Docs/results/*.json`. Illustrations: `Docs/media/`.

## Unreal Engine test bench (per-camera faults, seed 1)
- mounting error σ = 3° (up to 10° drawn);
- distortion spread k1 σ = 0.03 (from −0.085 to +0.056), k2..k4 proportionally;
- exposure σ = 0.8 stops (gain 0.56…1.55), white balance σ = 0.08;
- cos⁴ vignetting (strength 0.5), noise σ = 2;
- trigger desynchronisation up to 0.5 s, the operator walks in a circle and turns at 25°/s (dataset B only);
- ground truth — an ideal 360° cube from the helmet centre at the trigger.

## Calibration (UE, truth known)
| Camera | Nominal, ° | Self-calibration, shared lens, ° | Self-calibration, lens per camera, ° |
|---|---|---|---|
| cam0 | 8.3 | 0.28 | 0.19 |
| cam1 | 3.2 | 0.92 | 0.38 |
| cam2 | 4.9 | 0.50 | 0.34 |
| cam3 | 6.2 | 0.29 | 0.15 |
| cam4 | 8.1 | 0.44 | 0.24 |
| cam5 | 4.8 | 1.05 | 0.55 |
| cam6 | 10.2 | 1.11 | 0.33 |
| top7 | 4.9 | 0.61 | 0.19 |

- Alignment residual: 4.2 px (shared lens) → 0.98 px (lens per camera).
- Lens curve "angle → pixel" at the image edge: 37–138 px (shared lens) → 19–43 px (per camera). Absolute f and k1 cannot be separated from one scene, so an accurate edge curve needs a one-off ChArUco calibration of every camera (done in `Docs/CALIBRATION.md`).
- **Desynchronisation breaks self-calibration:** on dataset B the error of some cameras grows to 7–10°. Calibrate on frames where the operator stands still, or synchronise the cameras in hardware.

## UE, dataset A — all static faults, no desynchronisation (8 frames)
| Variant | PSNR, dB | SSIM | Seams, % | Seam disagreement |
|---|---|---|---|---|
| Nominal calibration | 14.9 | 0.623 | 2.04 | 22.4 |
| Oracle (true parameters) | 20.2 | 0.763 | 0.41 | 10.9 |
| Self-calibration | 17.0 | 0.664 | 1.37 | 19.1 |
| D5 Per-frame rotation | 16.5 | 0.656 | 1.27 | 18.6 |
| D1 Global spline | 16.8 | 0.676 | 0.53 | 13.7 |
| D3 Tangent projection | 16.6 | 0.667 | 0.76 | 14.7 |
| D2 RAFT | 16.9 | 0.676 | **0.43** | **13.2** |
| D4 Depth from flow | 16.7 | 0.656 | 1.70 | 20.8 |
| D4 Depth + Depth Anything | **18.5** | **0.700** | 0.90 | 14.9 |

PSNR is computed after fitting the overall brightness to the ground truth. "Seam disagreement" is the mean brightness difference of two cameras in the seam zone, in 0–255 units.

## UE, dataset B — the same plus desynchronisation up to 0.5 s
| Variant | Seams, % | Seam disagreement |
|---|---|---|
| Nominal calibration | 2.71 | 28.5 |
| Oracle | 1.58 | 20.7 |
| Self-calibration | 2.40 | 26.7 |
| D5 Per-frame rotation | 2.79 | 38.1 |
| D1 Global spline | 1.70 | 22.6 |
| D3 Tangent projection | 1.75 | 22.9 |
| **D2 RAFT** | **1.43** | **21.1** |
| D4 Depth + Depth Anything | 1.63 | 21.7 |

## Real helmet (6 control frames, 2048×1024)
| Variant | Seams, % | Seam disagreement | Time, s |
|---|---|---|---|
| Base (current) | 1.65 | 26.0 | 6 |
| Base + lens per camera | 1.79 | 26.8 | 6 |
| D5 Per-frame rotation | 2.12 | 32.0 | 53 |
| D1 Global spline | 1.43 | 24.1 | 9 |
| D3 Tangent projection | 1.45 | 23.9 | 10 |
| D2 RAFT | 1.30 | 23.8 | 17 |
| D4 Depth + Depth Anything | 1.55 | 25.1 | 15 |
| **D4 Depth + Depth Anything + flow** | **0.80** | **18.2** | 22 |

On the frame with people and a hand 30–60 cm from the cameras (frame 1), only the depth variant assembles the people without tears.

## UE, final panoramas (depth mode, 4096×2048)
`Tools/experiments/ue_final.py`: both datasets stitched with the production depth mode. The helmet is calibrated once on the static, synchronous dataset A (per-camera self-calibration, mounting error 0.30° mean, 0.55° max), and that rig is reused for dataset B of the same helmet — its own self-calibration breaks down to 10° because of the desynchronisation.

| | Dataset A (synchronous) | Dataset B (desynchronised up to 0.5 s) |
|---|---|---|
| PSNR vs ground truth, mean of 8 frames | **19.9 dB** (18.5–21.1) | 12.9 dB (11.8–14.1) |
| SSIM | 0.77 | 0.67 |
| Seam artifacts (criterion < 10 %) | 0.26 % | 1.4 % |

Frame 4 of dataset A: no calibration 13.5 dB, calibrated 21.1 dB, true parameters 21.4 dB. On dataset B even the true parameters give only 13.9 dB: the cameras fired at different moments while the operator was walking, so the panorama is assembled from different instants. Calibration cannot fix this; hardware synchronisation is needed.

## Conclusions
1. **Self-calibration is mandatory:** +2.1 dB over nominal, rotations better than 1°. But calibrate on static frames: desynchronisation spoils it. Target-based calibration is better still, see `Docs/CALIBRATION.md`.
2. **Best final variant — D4 + flow:** depth (Depth Anything V2, scaled to metres by triangulation between cameras) instead of a 6 m sphere, then a global seam fit. On real frames seams 1.65 % → 0.80 %, disagreement 26 → 18. Cost: a GPU and ~22 s per 2048 panorama.
3. **D2 RAFT** is the best flow-only variant and the most robust to desynchronisation (B: 1.43 %). It is worth using as the flow inside variant 2.
4. **D1 global spline** is the best variant without a GPU: 9 s, seams −15…60 %.
5. **D3** helps only on real ceilings near the pole, and only a little. In UE it is worse than D1.
6. **D5** in its current form is harmful: a rotation cannot compensate the operator's displacement during the delay, and dropping pairs with a relative rotation above 8° loses data. The right solution is hardware synchronisation.
7. A separate lens per camera helps in UE (synchronous frames), but on real desynchronised data it overfits: f comes out between 598 and 721 px. Real cameras need a one-off ChArUco calibration.
