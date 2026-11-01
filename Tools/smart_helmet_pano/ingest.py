"""Ingest raw Raspberry-Pi helmet sessions (as unloaded by the dock station).

Layout:  <root>/cam1..cam8/session_YYYYMMDD/camN_[YYYYMMDD_]HHMMSS_<n>.jpg
         cam1 (master, top camera) also has master_imu_data.json (BNO055 record per trigger).

Real-hardware problems handled here (found in the 2025-08-20 sessions):
  * Pi clocks without NTP/RTC: per-camera clock offset is estimated from capture bursts.
  * fake-hwclock: after every reboot a Pi may restart at the same saved time, so several
    sessions overlap in time (cam8). Such photos are matched by trigger number and then by
    image content (the candidate that best matches the other cameras of that instant).
  * empty (0-byte) / noisy / clipped / colour-cast photos are rejected (health.py).
Portrait-mounted ring cameras are rotated upright and the standard frame layout is written:
    <out>/rig_frames/000001/<cam>.jpg + meta.json (IMU quaternion)
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .health import photo_stats, verdict

_T = re.compile(r"_(\d{6})_(\d+)\.jpg$")


def scan(root: Path):
    """{cam: [(t, counter, path)]} sorted by time; skips duplicate '(1)' copies."""
    cams = {}
    for cdir in sorted(Path(root).glob("cam*")):
        for sdir in sorted(cdir.glob("session_*")):
            date = sdir.name.split("_")[-1]
            for p in sorted(sdir.glob("*.jpg")):
                m = _T.search(p.name)
                if not m:
                    continue
                t = datetime.strptime(date + m.group(1), "%Y%m%d%H%M%S").timestamp()
                cams.setdefault(cdir.name, []).append((t, int(m.group(2)), p))
    for c in cams:
        cams[c].sort(key=lambda x: x[0])
    return cams


def imu_valid(d: dict) -> list:
    """Problems of one BNO055 record (empty list = valid). Seen in real logs: torn I2C reads give
    quaternions with |q| up to 2.2, and the temperature register returns -97 / -100 C."""
    bad = []
    q = d.get("quaternion", {})
    n = math.sqrt(sum(float(q.get(k, 0)) ** 2 for k in "wxyz"))
    if abs(n - 1) > 0.02:
        bad.append(f"quaternion_norm={n:.3f}")
    t = d.get("temperature")
    if t is not None and not (-20 <= t <= 85):
        bad.append(f"temperature={t}")
    if d.get("calibration_status", {}).get("system", 3) == 0:
        bad.append("system_uncalibrated")
    return bad


def load_imu(root: Path, valid_only: bool = False):
    """[(t, record)] sorted by time; each record gets 'problems'. valid_only drops bad quaternions."""
    recs = []
    for p in Path(root).glob("cam1/session_*/master_imu_data.json"):
        for r in json.loads(p.read_text(encoding="utf-8")):
            d = r.get("imu_data", {})
            if not d.get("available"):
                continue
            r["problems"] = imu_valid(d)
            if valid_only and any(x.startswith("quaternion") for x in r["problems"]):
                continue
            recs.append((datetime.fromisoformat(r["timestamp"]).timestamp(), r))
    return sorted(recs, key=lambda x: x[0])


def clock_offsets(cams: dict, ref: str, tol=1.5):
    """Per-camera clock offset (s) vs the reference camera (most aligned capture times)."""
    r = np.array([t for t, _, _ in cams[ref]])
    out = {ref: 0.0}
    for c, v in cams.items():
        if c == ref:
            continue
        # interleaved sessions (counter jumps back within seconds) => clock restarted (fake-hwclock)
        resets = sum(1 for a, b in zip(v, v[1:]) if b[1] < a[1] and b[0] - a[0] < 30)
        if resets > 3:
            out[c] = None
            continue
        ts = np.array([t for t, _, _ in v])
        cand = {round(b - a) for a in ts[::2] for b in r}
        score = {off: int((np.abs((ts + off)[:, None] - r[None, :]).min(1) <= tol).sum()) for off in cand}
        best = max(score, key=score.get)
        out[c] = float(best) if score[best] >= 0.6 * len(ts) else None   # None = clock unusable
    return out


def _small_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if im is None:
        return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(3.0, (8, 8)).apply(g)


_sift = cv2.SIFT_create(nfeatures=1500)
_bf = cv2.BFMatcher(cv2.NORM_L2)


def _desc(path, cache):
    if path not in cache:
        g = _small_gray(path)
        cache[path] = None if g is None else _sift.detectAndCompute(g, None)[1]
    return cache[path]


def _nmatch(d1, d2):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return 0
    return sum(1 for m in _bf.knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < 0.7 * m[1].distance)


def build_frames(root, ref="cam5", tol=2.0, verbose=True):
    """Synchronous frames: list of (t_ref, {cam: path}, {cam: health})."""
    cams = scan(root)
    offs = clock_offsets(cams, ref)
    if verbose:
        print("clock offsets vs", ref, offs)
    timed = [c for c, o in offs.items() if o is not None and c != ref]
    untimed = [c for c, o in offs.items() if o is None]
    cache = {}
    frames = []
    for t, k, p in cams[ref]:
        fr = {ref: p}
        for c in timed:
            best = min(cams[c], key=lambda x: abs(x[0] + offs[c] - t))
            if abs(best[0] + offs[c] - t) <= tol:
                fr[c] = best[2]
        for c in untimed:  # same trigger number, choose by content
            cands = [x[2] for x in cams[c] if x[1] == k]
            if len(cands) == 1:
                fr[c] = cands[0]
            elif cands:
                others = [_desc(fr[o], cache) for o in fr]
                fr[c] = max(cands, key=lambda q: sum(_nmatch(_desc(q, cache), d) for d in others))
        frames.append((t, fr))
    return frames, offs


def export(root, out, top="cam1", ring_rotate=cv2.ROTATE_90_COUNTERCLOCKWISE, ref="cam5",
           require_all=True, verbose=True):
    """Write usable synchronous frames: ring cameras rotated upright, top camera renamed 'top1'."""
    frames, offs = build_frames(root, ref=ref, verbose=verbose)
    imu = load_imu(root, valid_only=True)
    all_cams = sorted({c for _, fr in frames for c in fr})
    out = Path(out)
    n, rejected = 0, {}
    for t, fr in frames:
        if require_all and len(fr) < len(all_cams):
            rejected["missing_camera"] = rejected.get("missing_camera", 0) + 1
            continue
        imgs, bad = {}, []
        for c, p in fr.items():
            im = cv2.imread(str(p))
            v = verdict(photo_stats(im))
            if v:
                bad.append(f"{c}:{'+'.join(v)}")
            imgs[c] = im
        if bad:
            for b in bad:
                key = b.split(":")[1]
                rejected[key] = rejected.get(key, 0) + 1
            continue
        n += 1
        d = out / "frames" / f"{n:06d}"
        d.mkdir(parents=True, exist_ok=True)
        for c, im in imgs.items():
            name = "top" + c[3:] if c == top else c
            if c != top and ring_rotate is not None:
                im = cv2.rotate(im, ring_rotate)
            cv2.imwrite(str(d / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta = {"frame": n, "timestamp": datetime.fromtimestamp(t).isoformat(),
                "sources": {c: str(p) for c, p in fr.items()}}
        if imu:
            tt, rec = min(imu, key=lambda x: abs(x[0] - t))
            if abs(tt - t) < 3:
                q = rec["imu_data"]["quaternion"]
                meta["imu"] = {"quaternion": [q["w"], q["x"], q["y"], q["z"]], "raw": rec["imu_data"],
                               "problems": rec["problems"]}
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    summary = {"frames_total": len(frames), "frames_exported": n, "rejected_by_reason": rejected,
               "clock_offsets_s": offs}
    (out / "ingest_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if verbose:
        print(summary)
    return summary
