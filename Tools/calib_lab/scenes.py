"""Ray-traced calibration stands (exact geometry, true parallax between the lenses).

A scene is a set of textured planar quads (walls, boards, boxes) plus optional projectors that light the
walls with a pattern.  Coordinates: UE axes (X forward, Y right, Z up), metres, same as the rig.

Stands
  * MarkerRoom     – office room with ArUco "marker points" stuck at unknown places on walls / ceiling
  * ProjectorRoom  – plain room, projectors throw chessboards onto 4 walls + ceiling (keystoned, unknown)
  * CubeRoom       – 3 x 3 x 3 m calibration cube, every face is a surveyed ChArUco board (built with errors)
  * BoardScene     – handheld ChArUco board in front of the helmet (classic per-camera calibration)
  * NaturalRoom    – ordinary textured room (self-calibration on natural features)
  * TestRoom       – evaluation room with furniture at 0.8..2 m (parallax) and straight lines
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from smart_helmet_pano.rig import rot_ue

ARUCO = cv2.aruco
FACE_ID = {"front": 0, "back": 1, "right": 2, "left": 3, "ceiling": 4, "floor": 5}


def _u8(t):
    """Textures are kept as uint8 (4x less memory per render worker)."""
    return t if t.dtype == np.uint8 else np.clip(t * 255 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- geometry
class Quad:
    """Rectangle O + a*e1 + b*e2, a in [0, w], b in [0, h].  Texture column ~ a, row ~ b."""

    def __init__(self, origin, e1, e2, w, h, texture, name=""):
        self.O = np.asarray(origin, float)
        self.e1 = np.asarray(e1, float) / np.linalg.norm(e1)
        self.e2 = np.asarray(e2, float) / np.linalg.norm(e2)
        self.n = np.cross(self.e1, self.e2)
        self.w, self.h = float(w), float(h)
        self.tex = _u8(texture)
        self.name = name

    def to_world(self, a, b):
        return self.O + np.multiply.outer(a, self.e1) + np.multiply.outer(b, self.e2)


def facing_quad(center, view_dir, up_hint, w, h, texture, name=""):
    """Quad seen un-mirrored by a viewer looking along view_dir (UE axes: right = up x forward)."""
    fwd = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    up = np.asarray(up_hint, float)
    up = up - fwd * (up @ fwd)
    up /= np.linalg.norm(up)
    right = np.cross(up, fwd)
    e1, e2 = right, -up
    O = np.asarray(center, float) - e1 * w / 2 - e2 * h / 2
    return Quad(O, e1, e2, w, h, texture, name)


class Projector:
    """Pinhole projector (+ small radial distortion) throwing `pattern` (grey, 0..1) into the room."""

    def __init__(self, position, pyr_deg, pattern, hfov_deg=50.0, k1=0.01, power=1.0):
        self.C = np.asarray(position, float)
        self.R = rot_ue(*pyr_deg)                    # columns fwd, right, up
        self.pattern = pattern.astype(np.float32)
        h, w = pattern.shape[:2]
        self.f = (w / 2) / math.tan(math.radians(hfov_deg / 2))
        self.cx, self.cy = (w - 1) / 2, (h - 1) / 2
        self.k1, self.power = k1, power

    def project(self, P):
        q = (P - self.C) @ self.R                    # (fwd, right, up)
        z = q[:, 0]
        x, y = q[:, 1] / np.maximum(z, 1e-6), -q[:, 2] / np.maximum(z, 1e-6)
        r2 = x * x + y * y
        s = 1 + self.k1 * r2
        return self.f * x * s + self.cx, self.f * y * s + self.cy, z > 0.05

    def light(self, P):
        u, v, ok = self.project(P)
        h, w = self.pattern.shape[:2]
        ok &= (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        out = np.zeros(len(P), np.float32)
        if ok.any():
            out[ok] = _sample(self.pattern, u[ok], v[ok])[:, 0] if self.pattern.ndim == 3 else \
                _sample(self.pattern[..., None], u[ok], v[ok])[:, 0]
        d2 = np.sum((P - self.C) ** 2, 1)
        return out * self.power * (4.0 / np.maximum(d2, 0.5))       # inverse-square, 1.0 at 2 m


def _sample(tex, u, v):
    """Bilinear lookup of many points (any count) in a texture (H, W, C)."""
    n = len(u)
    pad = (-n) % 4096
    mx = np.pad(u.astype(np.float32), (0, pad)).reshape(-1, 4096)
    my = np.pad(v.astype(np.float32), (0, pad)).reshape(-1, 4096)
    out = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, tex.shape[2] if tex.ndim == 3 else 1)[:n]


class Scene:
    def __init__(self, quads, projectors=(), ambient=1.0, name="scene"):
        self.quads = list(quads)
        self.projectors = list(projectors)
        self.ambient = ambient
        self.name = name

    def intersect(self, origin, rays):
        """Nearest hit: returns t, quad index (-1 = miss), a, b."""
        N = len(rays)
        best = np.full(N, np.inf)
        idx = -np.ones(N, int)
        A = np.zeros(N)
        B = np.zeros(N)
        for qi, q in enumerate(self.quads):
            den = rays @ q.n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((q.O - origin) @ q.n) / den
            ok = np.isfinite(t) & (t > 1e-4) & (t < best)
            if not ok.any():
                continue
            P = origin + rays[ok] * t[ok, None]
            a = (P - q.O) @ q.e1
            b = (P - q.O) @ q.e2
            inside = (a >= 0) & (a <= q.w) & (b >= 0) & (b <= q.h)
            sel = np.flatnonzero(ok)[inside]
            best[sel] = t[sel]
            idx[sel] = qi
            A[sel], B[sel] = a[inside], b[inside]
        return best, idx, A, B

    def hit_points(self, origin, rays):
        t, idx, _, _ = self.intersect(origin, rays)
        return origin + rays * np.where(np.isfinite(t), t, 1e3)[:, None], idx

    def shade(self, origin, rays):
        """origin (3,), rays (..., 3) unit -> float32 BGR radiance (..., 3), 0..~1."""
        shp = rays.shape[:-1]
        r = rays.reshape(-1, 3)
        t, idx, A, B = self.intersect(origin, r)
        out = np.zeros((len(r), 3), np.float32)
        for qi, q in enumerate(self.quads):
            sel = idx == qi
            if not sel.any():
                continue
            th, tw = q.tex.shape[:2]
            u = A[sel] / q.w * tw - 0.5          # a in [0, w] <-> texture edge coords [0, tw]
            v = B[sel] / q.h * th - 0.5
            out[sel] = _sample(q.tex, u, v)[:, :3].astype(np.float32) / 255
        if self.projectors:
            P = origin + r * np.where(np.isfinite(t), t, 0)[:, None]
            L = np.full(len(r), self.ambient, np.float32)
            for pj in self.projectors:
                L += pj.light(P)
            out *= L[:, None]
        else:
            out *= self.ambient
        return out.reshape(*shp, 3)


# ---------------------------------------------------------------- textures
def _noise_tex(h, w, base, amp, rng, blur=3.0):
    t = np.empty((h, w, 3), np.float32)
    t[:] = np.asarray(base, np.float32)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), blur)
    n /= max(n.std(), 1e-6)
    t += (amp * n)[..., None]
    return np.clip(t, 0, 1)


def natural_texture(h, w, seed, label=""):
    """Posters, boxes, circles, lines (like the synth room, but at ~3 mm/texel)."""
    rng = np.random.default_rng(seed)
    t = (_noise_tex(h, w, rng.uniform(0.35, 0.75, 3), 0.04, rng, blur=6) * 255).astype(np.uint8)
    s = w / 1024
    for _ in range(int(90 * s * h / w)):
        x, y = rng.integers(0, w - 30, 1)[0], rng.integers(0, h - 30, 1)[0]
        ww, hh = rng.integers(int(20 * s), int(220 * s), 2)
        cv2.rectangle(t, (int(x), int(y)), (int(x + ww), int(y + hh)), rng.integers(0, 255, 3).tolist(), -1)
    for _ in range(int(60 * s * h / w)):
        c = rng.integers(0, max(w, h), 2)
        cv2.circle(t, (int(c[0] % w), int(c[1] % h)), int(rng.integers(6, int(60 * s))),
                   rng.integers(0, 255, 3).tolist(), max(2, int(3 * s)))
    for _ in range(int(40 * s)):
        p0 = rng.integers(0, [w, h]); p1 = p0 + rng.integers(-300, 300, 2) * s
        cv2.line(t, tuple(int(v) for v in p0), tuple(int(v) for v in p1), rng.integers(0, 255, 3).tolist(),
                 max(2, int(2 * s)))
    if label:
        cv2.putText(t, label, (int(40 * s), h // 2), cv2.FONT_HERSHEY_SIMPLEX, 5 * s, (255, 255, 255), int(12 * s))
    t = cv2.GaussianBlur(t, (0, 0), 0.8)
    return t.astype(np.float32) / 255


def grid_texture(h, w, seed, step_px=128, label=""):
    t = natural_texture(h, w, seed, label)
    for i in range(0, w, step_px):
        cv2.line(t, (i, 0), (i, h), (0.08, 0.08, 0.08), 3)
    for i in range(0, h, step_px):
        cv2.line(t, (0, i), (w, i), (0.08, 0.08, 0.08), 3)
    return t


# ---------------------------------------------------------------- rooms
def box_quads(hx0, hx1, hy0, hy1, z0, z1, textures):
    """6 inner faces of an axis-aligned box, un-mirrored from inside. textures: dict face -> (fn(w_m, h_m))"""
    cx, cy, cz = (hx0 + hx1) / 2, (hy0 + hy1) / 2, (z0 + z1) / 2
    X, Y, Z = hx1 - hx0, hy1 - hy0, z1 - z0
    up, fwd = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    spec = {  # name: (centre, view dir, up hint, w, h)
        "front": ([hx1, cy, cz], [1, 0, 0], up, Y, Z),
        "back": ([hx0, cy, cz], [-1, 0, 0], up, Y, Z),
        "right": ([cx, hy1, cz], [0, 1, 0], up, X, Z),
        "left": ([cx, hy0, cz], [0, -1, 0], up, X, Z),
        "ceiling": ([cx, cy, z1], [0, 0, 1], -fwd, Y, X),
        "floor": ([cx, cy, z0], [0, 0, -1], fwd, Y, X),
    }
    return [facing_quad(c, v, u, w, h, textures(k, w, h), k) for k, (c, v, u, w, h) in spec.items()]


def NaturalRoom(seed=11, bounds=(-2.6, 3.0, -2.2, 2.4, -1.55, 1.45), px_per_m=400):
    def tex(k, w, h):
        return natural_texture(int(h * px_per_m), int(w * px_per_m), seed + FACE_ID[k] * 7, k.upper())
    return Scene(box_quads(*bounds, tex), name="natural")


def TestRoom(seed=101, px_per_m=420):
    """Room 6 x 5 x 3 m with walls of straight lines plus near furniture (boxes) for parallax."""
    b = (-2.8, 3.2, -2.3, 2.7, -1.6, 1.4)

    def tex(k, w, h):
        return grid_texture(int(h * px_per_m), int(w * px_per_m), seed + FACE_ID[k] * 7, step_px=int(0.5 * px_per_m),
                            label=k.upper())
    quads = box_quads(*b, tex)
    rng = np.random.default_rng(seed)
    # furniture: boxes at 0.8..2 m from the helmet (cupboard, table edge, a column)
    boxes = [((1.3, -0.9, -1.6), (0.5, 0.5, 1.9)), ((-1.0, 1.1, -1.6), (0.6, 0.5, 1.0)),
             ((0.2, 1.7, -1.6), (0.35, 0.35, 3.0)), ((-1.5, -1.6, -0.2), (0.7, 0.4, 0.5)),
             ((0.9, 0.8, 0.6), (0.4, 0.4, 0.3))]
    for (x0, y0, z0), (sx, sy, sz) in boxes:
        def btex(k, w, h, s=int(rng.integers(1000))):
            return grid_texture(max(32, int(h * px_per_m)), max(32, int(w * px_per_m)), s, step_px=int(0.1 * px_per_m))
        for q in box_quads(x0, x0 + sx, y0, y0 + sy, z0, z0 + sz, btex):
            # box faces must face outwards: flip the inner-face normal by mirroring e1
            q2 = Quad(q.O + q.e1 * q.w, -q.e1, q.e2, q.w, q.h, q.tex, "box_" + q.name)
            quads.append(q2)
    return Scene(quads, name="test")


# ---------------------------------------------------------------- calibration patterns
def plain_wall(h, w, seed, base=(0.78, 0.78, 0.76)):
    return _noise_tex(h, w, base, 0.015, np.random.default_rng(seed), blur=4)


def MarkerRoom(seed=21, bounds=(-2.7, 3.3, -2.4, 2.6, -1.5, 1.5), marker_m=0.15, per_m2=1.6, px_per_m=500,
               dictionary=ARUCO.DICT_4X4_1000):
    """Unknown-position ArUco marker points.  Returns scene + ground truth corners {id: (4,3)}."""
    d = ARUCO.getPredefinedDictionary(dictionary)
    rng = np.random.default_rng(seed)
    truth = {}
    next_id = [0]

    def tex(k, w, h):
        H, W = int(h * px_per_m), int(w * px_per_m)
        t = plain_wall(H, W, seed + FACE_ID[k] * 7)
        n = int(per_m2 * w * h) if k != "floor" else 0
        ms = int(marker_m * px_per_m)
        border = ms // 6 + 4
        taken = []
        for _ in range(n * 30):
            if len(taken) >= n:
                break
            x = int(rng.integers(border, W - ms - border)); y = int(rng.integers(border, H - ms - border))
            if any(abs(x - a) < ms + 2 * border and abs(y - b) < ms + 2 * border for a, b in taken):
                continue
            taken.append((x, y))
            mid = next_id[0]; next_id[0] += 1
            img = ARUCO.generateImageMarker(d, mid, ms, borderBits=1).astype(np.float32) / 255
            cv2.rectangle(t, (x - border, y - border), (x + ms + border, y + ms + border), (0.97, 0.97, 0.97), -1)
            t[y:y + ms, x:x + ms] = img[..., None]
            truth[(k, mid)] = np.array([[x, y], [x + ms, y], [x + ms, y + ms], [x, y + ms]], float) * [w / W, h / H]
        return cv2.GaussianBlur(t, (0, 0), 0.7)
    quads = box_quads(*bounds, tex)
    by_face = {q.name: q for q in quads}
    pts = {mid: by_face[k].to_world(c[:, 0], c[:, 1]) for (k, mid), c in truth.items()}
    sc = Scene(quads, name="markers")
    sc.marker_truth = pts
    sc.marker_m = marker_m
    sc.dictionary = dictionary
    return sc


def chessboard_pattern(squares=(10, 7), res=(1920, 1080), fill=0.82, margin_sq=0.8):
    """Projector frame: white-bordered chessboard on black (no light outside the board)."""
    W, H = res
    sq = int(min(W * fill / (squares[0] + 2 * margin_sq), H * fill / (squares[1] + 2 * margin_sq)))
    bw, bh = squares[0] * sq, squares[1] * sq
    m = int(margin_sq * sq)
    img = np.zeros((H, W), np.float32)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    img[y0 - m:y0 + bh + m, x0 - m:x0 + bw + m] = 1.0
    for j in range(squares[1]):
        for i in range(squares[0]):
            if (i + j) % 2 == 0:
                img[y0 + j * sq:y0 + (j + 1) * sq, x0 + i * sq:x0 + (i + 1) * sq] = 0.0
    # inner corners in projector pixels, row-major (cols-1) x (rows-1)
    cs = np.array([[x0 + i * sq - 0.5, y0 + j * sq - 0.5] for j in range(1, squares[1]) for i in range(1, squares[0])])
    return cv2.GaussianBlur(img, (0, 0), 0.6), cs


PROJ_TARGETS = [(3.3, -1.0, 0.3), (3.3, 1.2, -0.6), (-2.7, 0.8, 0.4), (-2.7, -1.2, -0.5),
                (0.8, 2.6, 0.2), (-1.4, 2.6, -0.4), (-0.6, -2.4, 0.3), (1.5, -2.4, -0.5),
                (0.9, -0.6, 1.5), (-0.8, 0.9, 1.5), (2.0, 1.5, 1.5), (-1.7, -1.4, 1.5),
                (1.6, 0.7, -1.5), (-1.2, -1.3, -1.5)]


def ProjectorRoom(seed=31, bounds=(-2.7, 3.3, -2.4, 2.6, -1.5, 1.5), squares=(10, 7), px_per_m=300):
    """One projector on a stand, re-aimed 14 times (4 walls x 2, ceiling x 4, floor x 2); the captures of one
    helmet pose are merged, so all 14 keystoned chessboards are rendered at once.  Board corners are unknown
    to the calibration (only "which wall, roughly where")."""
    def tex(k, w, h):
        return plain_wall(int(h * px_per_m), int(w * px_per_m), seed + FACE_ID[k] * 7)
    quads = box_quads(*bounds, tex)
    pat, corners = chessboard_pattern(squares)
    ys, xs = np.nonzero(pat > 0.5)
    outline = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], float)
    rng = np.random.default_rng(seed)
    sc = Scene(quads, [], ambient=0.28, name="projector")
    projs, truth, regions = [], {}, []
    for T in PROJ_TARGETS:
        T = np.asarray(T, float)
        C = -0.25 * T + rng.uniform(-0.3, 0.3, 3) * [1, 1, 0.5]
        d = T - C
        pyr = (math.degrees(math.asin(d[2] / np.linalg.norm(d))), math.degrees(math.atan2(d[1], d[0])),
               float(rng.uniform(-6, 6)))
        for hfov in np.arange(46, 20, -2.0):          # shrink until the board lands on a single wall
            pj = Projector(C, pyr, pat, hfov_deg=float(hfov), k1=float(rng.uniform(0.005, 0.02)), power=2.2)
            P, qi = sc.hit_points(pj.C, _projector_rays(pj, np.vstack([corners, outline])))
            if len(set(qi.tolist())) == 1:
                break
        poly = P[len(corners):]
        clash = any(np.linalg.norm(poly.mean(0) - r.mean(0)) < 0.5 * (np.ptp(poly, 0).max() + np.ptp(r, 0).max())
                    and qr == qi[0] for r, qr in regions)
        if clash:
            continue
        regions.append((poly, qi[0]))
        pj.wall = quads[qi[0]].name
        truth[len(projs)] = P[:len(corners)]
        projs.append(pj)
    sc.projectors = projs
    sc.squares = squares
    sc.board_truth = truth          # ground truth (evaluation only): projected corners on the walls
    return sc


def _projector_rays(pj, uv):
    x = (uv[:, 0] - pj.cx) / pj.f
    y = (uv[:, 1] - pj.cy) / pj.f
    for _ in range(10):                                  # undo k1
        r2 = x * x + y * y
        x, y = ((uv[:, 0] - pj.cx) / pj.f) / (1 + pj.k1 * r2), ((uv[:, 1] - pj.cy) / pj.f) / (1 + pj.k1 * r2)
    d = pj.R @ np.stack([np.ones_like(x), x, -y])        # fwd + x*right - y*up
    d = d.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def charuco_board(squares, square_m, marker_ratio=0.75, dictionary=ARUCO.DICT_5X5_1000, id0=0):
    d = ARUCO.getPredefinedDictionary(dictionary)
    n = (squares[0] * squares[1]) // 2
    ids = np.arange(id0, id0 + n, dtype=np.int32)
    return ARUCO.CharucoBoard(squares, square_m, square_m * marker_ratio, d, ids)


def CubeRoom(seed=41, size_m=3.0, squares=12, build_err_mm=3.0, build_err_deg=0.15, px_per_m=1000):
    """Calibration cube: 6 ChArUco faces with surveyed (nominal) corners; the build is not perfect."""
    rng = np.random.default_rng(seed)
    sq = size_m / squares
    h = size_m / 2
    boards, nominal_pts, true_pts = [], {}, {}
    faces = ["front", "back", "right", "left", "ceiling", "floor"]

    def tex(k, w, hh):
        fi = faces.index(k)
        b = charuco_board((squares, squares), sq, id0=fi * (squares * squares // 2))
        boards.append((k, b))
        img = b.generateImage((int(w * px_per_m), int(hh * px_per_m)), marginSize=0, borderBits=1)
        return cv2.GaussianBlur(np.repeat(img[..., None], 3, 2).astype(np.float32) / 255, (0, 0), 0.7)
    ideal = box_quads(-h, h, -h, h, -h, h, tex)
    built = []
    for q in ideal:
        # build error: every face panel shifted / rotated a little, printed at a slightly wrong scale
        dR = rot_ue(*rng.normal(0, build_err_deg, 3))
        dt = rng.normal(0, build_err_mm / 1000, 3)
        s = 1 + rng.normal(0, 0.0007)
        c = q.O + q.e1 * q.w / 2 + q.e2 * q.h / 2
        e1, e2 = dR @ q.e1, dR @ q.e2
        O = c + dt - e1 * q.w * s / 2 - e2 * q.h * s / 2
        built.append(Quad(O, e1, e2, q.w * s, q.h * s, q.tex, q.name))
        b = dict(boards)[q.name]
        cc = b.getChessboardCorners()                    # (N,3) board frame, metres (x right, y down)
        nominal_pts[q.name] = q.to_world(cc[:, 0], cc[:, 1])
        true_pts[q.name] = built[-1].to_world(cc[:, 0] * s, cc[:, 1] * s)
    sc = Scene(built, name="cube")
    sc.boards = boards
    sc.nominal_pts, sc.true_pts = nominal_pts, true_pts
    return sc


def BoardScene(board, board_tex, pose_R, pose_t, background: Scene):
    """Handheld board (quad with the ChArUco texture) in front of a background room."""
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    w, h = sx * sq, sy * sq
    q = Quad(pose_t, pose_R[:, 0], pose_R[:, 1], w, h, board_tex, "board")
    return Scene([q] + background.quads, name="board")
