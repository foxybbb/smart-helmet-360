"""Panorama quality metrics for the test programme (sec. 3.2, criterion 4):
"artifacts cover less than 10 % of the central area".

Artifact pixel = inside a camera-overlap zone, the two aligned contributions
disagree by more than `thresh` (after gain compensation) on a textured pixel.
Central area = |latitude| <= central_lat_deg.
"""
import numpy as np
import cv2


def seam_artifacts(warped, weights, W, H, central_lat_deg=45.0, thresh=30):
    lat = (0.5 - (np.arange(H) + 0.5) / H) * 180
    central = np.abs(lat)[:, None] <= central_lat_deg
    central = np.repeat(central, W, axis=1)
    art = np.zeros((H, W), bool)
    seam_art = np.zeros((H, W), bool)
    ov_any = np.zeros((H, W), bool)
    seam_any = np.zeros((H, W), bool)
    n = len(warped)
    Wt = np.stack(weights)
    # visible seams = boundaries of the per-pixel owner (argmax weight), widened by the multiband blend width
    owner = np.where(Wt.max(0) > 0, np.argmax(Wt, 0), -1)
    edge = np.zeros((H, W), bool)
    edge[:, :-1] |= owner[:, :-1] != owner[:, 1:]
    edge[:-1, :] |= owner[:-1, :] != owner[1:, :]
    edge &= owner >= 0
    rad = max(2, W // 256)
    band = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))) > 0
    grays = [cv2.GaussianBlur(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16) for w in warped]
    for i in range(n):
        for j in range(i + 1, n):
            ov = (weights[i] > 0.05) & (weights[j] > 0.05)
            if not ov.any():
                continue
            ov_any |= ov
            sat = (grays[i] > 248) & (grays[j] > 248)
            d = np.abs(grays[i] - grays[j])
            bad = ov & (d > thresh) & ~sat
            art |= bad
            seam = band & ((owner == i) | (owner == j))           # seam band touching these two cameras
            seam_any |= seam
            seam_art |= bad & seam
    k = np.ones((3, 3), np.uint8)
    art = cv2.morphologyEx(art.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    seam_art = cv2.morphologyEx(seam_art.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    cw = np.cos(np.radians(lat))[:, None] * np.ones((1, W))  # solid-angle weight
    c_area = (cw * central).sum()
    pct = lambda m: round(float(100 * (cw * (m & central)).sum() / c_area), 2)
    seam_pct = pct(seam_art)
    return {"artifact_%_of_central": seam_pct,                  # visible: disagreement where cameras are blended
            "overlap_disagreement_%": pct(art),                 # strict: anywhere two cameras overlap
            "seam_zone_%_of_central": pct(seam_any),
            "overlap_%_of_central": pct(ov_any),
            "pass_lt_10%": bool(seam_pct < 10.0)}, seam_art


def psnr(a, b, mask=None):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        d = d[mask]
    mse = float(d.mean())
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
