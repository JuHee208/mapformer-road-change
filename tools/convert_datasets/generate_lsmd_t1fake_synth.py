#!/usr/bin/env python3
"""Generate synthetic LSMD train tiles by editing T1 map and recomputing change GT.

This script keeps original data untouched and writes only synthetic samples to out-root:
  out_root/
    images/t2/<region>/<tile_syn>.tif          (linked/copied from source)
    labels/t1/<region>/<tile_syn>.tif          (generated T1_fake)
    labels/t2/<region>/<tile_syn>.tif          (linked/copied from source)
    labels/change/<region>/<tile_syn>.tif      (recomputed from T1_fake, T2)
    splits/<out_split_name>.txt                (synthetic IDs only)
"""

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import rasterio


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--src-split-dir", type=Path, required=True)
    p.add_argument("--train-split-name", type=str, default="train.txt")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--out-split-name", type=str, default="train_synth_v1.txt")
    p.add_argument("--synth-ratio", type=float, default=0.30,
                   help="num_synth = round(num_train * synth_ratio), ignored if --num-synth is set")
    p.add_argument("--num-synth", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode-probs", type=str, default="0.50,0.50,0.0",
                   help="remove,add,both probabilities")
    p.add_argument("--balance-c12", action="store_true", default=True,
                   help="Dynamically bias mode selection to keep cls1/cls2 pixels balanced.")
    p.add_argument("--balance-ratio-high", type=float, default=1.05,
                   help="Upper bound for cls1/cls2 pixel ratio when --balance-c12 is enabled. "
                        "If exceeded, next mode is biased toward the weaker class.")
    p.add_argument("--min-change-pixels", type=int, default=32)
    p.add_argument("--max-change-ratio", type=float, default=0.20)
    p.add_argument("--min-change-component-pixels", type=int, default=120,
                   help="Minimum connected-component area for final cls1/cls2 in change GT.")
    p.add_argument("--min-edit-component-pixels", type=int, default=24,
                   help="Minimum connected-component area for synthetic add/remove edits.")
    p.add_argument("--add-branch-prob", type=float, default=0.50,
                   help="Probability of branch-style add edits.")
    p.add_argument("--add-thickness-mult", type=float, default=1.35,
                   help="Global thickness multiplier for add edits (cls2 bias).")
    p.add_argument("--add-width-scale-range", type=str, default="0.55,1.15",
                   help="Relative width scale range for strip-style add edits, e.g., 0.35,0.85")
    p.add_argument("--add-branch-width-scale-range", type=str, default="0.55,1.15",
                   help="Relative width scale range for branch-style add edits, e.g., 0.35,0.85")
    p.add_argument("--min-add-width-px", type=int, default=8,
                   help="Minimum width in pixels for add edits.")
    p.add_argument("--add-rect-strip-prob", type=float, default=0.35,
                   help="Probability of straight rectangular strip add edits (attached).")
    p.add_argument("--remove-bite-prob", type=float, default=0.45,
                   help="Probability of bite-style remove edits.")
    p.add_argument("--curve-edit-prob", type=float, default=0.80,
                   help="Probability of using curved geometry for add/remove edits.")
    p.add_argument("--large-edit-prob", type=float, default=0.90,
                   help="Probability of generating larger add/remove edits.")
    p.add_argument("--mega-edit-prob", type=float, default=0.20,
                   help="Probability of generating very large add/remove edits.")
    p.add_argument("--mega-width-mult-range", type=str, default="1.8,2.8",
                   help="Width multiplier range for mega edits, e.g., 1.8,2.8")
    p.add_argument("--mega-length-mult-range", type=str, default="1.8,3.2",
                   help="Length multiplier range for mega edits, e.g., 1.8,3.2")
    p.add_argument("--min-target-pixels", type=int, default=500,
                   help="Minimum target-class pixels per synthetic tile. "
                        "For remove mode target is cls1, for add mode target is cls2.")
    p.add_argument("--min-target-ratio", type=float, default=0.0020,
                   help="Minimum target-class ratio in valid pixels.")
    p.add_argument("--max-tries-per-sample", type=int, default=20)
    p.add_argument("--target-road-width-m", type=float, default=7.0,
                   help="Target synthetic road width in meters.")
    p.add_argument("--pixel-size-m", type=float, default=0.12,
                   help="Pixel size (meter/pixel).")
    p.add_argument("--width-scale-range", type=str, default="0.7,1.2",
                   help="Relative width scale range around estimated width, e.g., 0.7,1.2")
    p.add_argument("--thin-edit-prob", type=float, default=0.25,
                   help="Probability of generating thin add/remove edits per stroke.")
    p.add_argument("--thin-width-scale-range", type=str, default="0.15,0.45",
                   help="Relative width scale for thin edits, e.g., 0.15,0.45")
    p.add_argument("--min-draw-thickness-px", type=int, default=6)
    p.add_argument("--min-thin-draw-thickness-px", type=int, default=2,
                   help="Minimum line thickness for thin edits.")
    p.add_argument("--max-synth-per-source", type=int, default=1,
                   help="Maximum number of synthetic samples generated from one source tile.")
    p.add_argument("--min-t2-road-pixels", type=int, default=1,
                   help="Skip candidate tiles with fewer T2-road pixels than this value. "
                        "Set 1 to exclude zero-road tiles.")
    p.add_argument("--min-road-union-pixels", type=int, default=1,
                   help="Skip candidate tiles with fewer (T1 or T2) road pixels than this value. "
                        "Set 1 to exclude pure-background tiles.")
    p.add_argument("--max-no-road-source-ratio", type=float, default=0.05,
                   help="Maximum ratio of selected synthetic sources from pure no-road tiles (road_union==0). "
                        "Use <1.0 to prevent cls2-only dominance.")
    p.add_argument("--exclude-black-t2-image", action="store_true", default=True,
                   help="Exclude candidates where T2 image pixels are all zero across all bands.")
    p.add_argument("--quad-erase-prob", type=float, default=0.35,
                   help="Probability to apply quadrant keep/erase on T1 before edits.")
    p.add_argument("--affine-jitter-prob", type=float, default=0.35,
                   help="Probability to apply affine jitter to T1 before edits.")
    p.add_argument("--affine-max-rotate-deg", type=float, default=20.0,
                   help="Max absolute rotation in degrees for affine jitter.")
    p.add_argument("--affine-max-shift-px", type=float, default=24.0,
                   help="Max absolute translation in pixels for affine jitter.")
    p.add_argument("--affine-scale-range", type=str, default="0.92,1.08",
                   help="Scale range for affine jitter, e.g., 0.92,1.08")
    p.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    p.add_argument("--compress", type=str, default="LZW")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@dataclass
class TileRef:
    region: str
    tile: str

    @property
    def id(self):
        return f"{self.region}/{self.tile}"


def _read_ids(split_path: Path) -> List[TileRef]:
    ids = []
    for ln in split_path.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        if "/" not in s:
            raise ValueError(f"Invalid split line: {s}")
        r, t = s.split("/", 1)
        ids.append(TileRef(r, t))
    return ids


def _read_u8(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    return arr.astype(np.uint8), profile


def _is_black_t2_image(path: Path) -> bool:
    with rasterio.open(path) as ds:
        for band_idx in range(1, ds.count + 1):
            band = ds.read(band_idx)
            if np.any(band != 0):
                return False
    return True


def _save_u8(path: Path, arr: np.ndarray, profile: dict, compress: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(
        count=1,
        dtype="uint8",
        height=int(arr.shape[0]),
        width=int(arr.shape[1]),
        compress=compress,
    )
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.uint8), 1)


def _copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _random_road_pixel(mask: np.ndarray, rng: np.random.Generator):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    i = int(rng.integers(0, ys.size))
    return int(ys[i]), int(xs[i])


def _parse_pair(text: str, name: str) -> Tuple[float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 2:
        raise ValueError(f"{name} must contain 2 comma-separated values")
    if vals[0] <= 0 or vals[1] <= 0 or vals[0] > vals[1]:
        raise ValueError(f"{name} must satisfy 0 < low <= high")
    return vals[0], vals[1]


def _estimate_road_width_px(mask: np.ndarray, fallback_width_px: float) -> float:
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) < 64:
        return fallback_width_px
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    local_max = (dist == cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    center = local_max & (m > 0) & (dist > 1.0)
    vals = (2.0 * dist[center]).astype(np.float32)
    if vals.size < 16:
        return fallback_width_px
    return float(np.clip(np.median(vals), 2.0, 256.0))


def _sample_thickness(
    rng: np.random.Generator,
    nominal_width_px: float,
    low_scale: float,
    high_scale: float,
    min_px: int,
    thin_prob: float = 0.0,
    thin_low_scale: float = 0.15,
    thin_high_scale: float = 0.45,
    min_thin_px: int = 2,
) -> int:
    if thin_prob > 0 and rng.random() < thin_prob:
        low_scale = thin_low_scale
        high_scale = thin_high_scale
        min_px = min(min_px, min_thin_px)
    t = nominal_width_px * float(rng.uniform(low_scale, high_scale))
    return int(max(min_px, round(t)))


def _clip_pt(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    xx = int(np.clip(round(x), 0, w - 1))
    yy = int(np.clip(round(y), 0, h - 1))
    return xx, yy


def _pick_boundary_anchor(mask: np.ndarray, rng: np.random.Generator):
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    margin = 20
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contours = [c for c in contours if len(c) >= 8] or contours
    for _ in range(40):
        c = contours[int(rng.integers(0, len(contours)))]
        n = len(c)
        if n < 4:
            continue
        idx = int(rng.integers(0, n))
        p0 = c[(idx - 3) % n, 0]
        p1 = c[idx, 0]
        p2 = c[(idx + 3) % n, 0]
        # Avoid anchors near tile border; clipping creates sharp wedge artifacts.
        if not (margin <= int(p1[0]) < (w - margin) and margin <= int(p1[1]) < (h - margin)):
            continue

        v_prev = np.array([float(p1[0] - p0[0]), float(p1[1] - p0[1])], dtype=np.float32)
        v_next = np.array([float(p2[0] - p1[0]), float(p2[1] - p1[1])], dtype=np.float32)
        n_prev = float(np.linalg.norm(v_prev))
        n_next = float(np.linalg.norm(v_next))
        if n_prev < 1e-6 or n_next < 1e-6:
            continue
        # Skip sharp contour corners (spike-prone edit anchors).
        corner_cos = float(np.dot(v_prev, v_next) / (n_prev * n_next))
        if corner_cos < 0.45:
            continue

        tx = float(p2[0] - p0[0])
        ty = float(p2[1] - p0[1])
        norm = math.hypot(tx, ty)
        if norm < 1e-6:
            continue
        tx, ty = tx / norm, ty / norm
        nx, ny = -ty, tx
        # decide outward normal by checking road occupancy on both sides
        s = 3.0
        x1, y1 = _clip_pt(float(p1[0]) + nx * s, float(p1[1]) + ny * s, w, h)
        x2, y2 = _clip_pt(float(p1[0]) - nx * s, float(p1[1]) - ny * s, w, h)
        v1 = int(m[y1, x1])
        v2 = int(m[y2, x2])
        if v1 == 0 and v2 == 1:
            out_nx, out_ny = nx, ny
        elif v2 == 0 and v1 == 1:
            out_nx, out_ny = -nx, -ny
        else:
            # fallback: choose direction with lower local road occupancy
            r = 2
            y1a, y1b = max(0, y1 - r), min(h, y1 + r + 1)
            x1a, x1b = max(0, x1 - r), min(w, x1 + r + 1)
            y2a, y2b = max(0, y2 - r), min(h, y2 + r + 1)
            x2a, x2b = max(0, x2 - r), min(w, x2 + r + 1)
            occ1 = float(m[y1a:y1b, x1a:x1b].mean())
            occ2 = float(m[y2a:y2b, x2a:x2b].mean())
            if occ1 <= occ2:
                out_nx, out_ny = nx, ny
            else:
                out_nx, out_ny = -nx, -ny
        return float(p1[0]), float(p1[1]), tx, ty, out_nx, out_ny, c[:, 0, :].astype(np.float32), idx
    return None


def _draw_oriented_rect(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    tx: float,
    ty: float,
    nx: float,
    ny: float,
    length: float,
    width: float,
):
    hl = 0.5 * float(length)
    hw = 0.5 * float(width)
    pts = np.array([
        [cx - tx * hl - nx * hw, cy - ty * hl - ny * hw],
        [cx + tx * hl - nx * hw, cy + ty * hl - ny * hw],
        [cx + tx * hl + nx * hw, cy + ty * hl + ny * hw],
        [cx - tx * hl + nx * hw, cy - ty * hl + ny * hw],
    ], dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    cv2.fillConvexPoly(canvas, pts, 1)


def _wrap_contour_idx(idx: int, n: int) -> int:
    return int((idx % n + n) % n)


def _contour_tangent(contour: np.ndarray, idx: int) -> Tuple[float, float]:
    n = contour.shape[0]
    i0 = _wrap_contour_idx(idx - 2, n)
    i1 = _wrap_contour_idx(idx + 2, n)
    dx = float(contour[i1, 0] - contour[i0, 0])
    dy = float(contour[i1, 1] - contour[i0, 1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 1.0, 0.0
    return dx / norm, dy / norm


def _draw_contour_strip(
    canvas: np.ndarray,
    contour: np.ndarray,
    idx_center: int,
    out_nx_ref: float,
    out_ny_ref: float,
    half_len_px: int,
    strip_width_px: float,
    inward: bool = False,
):
    n = contour.shape[0]
    if n < 8:
        return
    half_len_px = int(max(4, half_len_px))
    strip_width_px = float(max(1.5, strip_width_px))

    idxs = [_wrap_contour_idx(idx_center + s, n) for s in range(-half_len_px, half_len_px + 1)]
    boundary_pts = contour[idxs]  # (m,2)

    offset_pts = []
    for ii in idxs:
        tx, ty = _contour_tangent(contour, ii)
        nx, ny = -ty, tx
        if nx * out_nx_ref + ny * out_ny_ref < 0:
            nx, ny = -nx, -ny
        if inward:
            nx, ny = -nx, -ny
        p = contour[ii]
        offset_pts.append([p[0] + nx * strip_width_px, p[1] + ny * strip_width_px])
    offset_pts = np.array(offset_pts, dtype=np.float32)

    poly = np.concatenate([boundary_pts, offset_pts[::-1]], axis=0)
    poly = np.round(poly).astype(np.int32)
    cv2.fillPoly(canvas, [poly], 1)
    # Round strip caps to avoid sharp triangular tips.
    cap_r = int(max(2, round(strip_width_px * 0.55)))
    c0 = np.round((boundary_pts[0] + offset_pts[0]) * 0.5).astype(np.int32)
    c1 = np.round((boundary_pts[-1] + offset_pts[-1]) * 0.5).astype(np.int32)
    cv2.circle(canvas, (int(c0[0]), int(c0[1])), cap_r, 1, thickness=-1, lineType=cv2.LINE_8)
    cv2.circle(canvas, (int(c1[0]), int(c1[1])), cap_r, 1, thickness=-1, lineType=cv2.LINE_8)


def _smooth_binary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    if radius <= 0:
        return out
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return (out > 0).astype(np.uint8)


def _draw_quadratic_curve(
    canvas: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    thickness: float,
    n_points: int = 9,
):
    ts = np.linspace(0.0, 1.0, int(max(5, n_points)), dtype=np.float32)
    pts = []
    for t in ts:
        a = (1.0 - t) * (1.0 - t)
        b = 2.0 * (1.0 - t) * t
        c = t * t
        x = a * p0[0] + b * p1[0] + c * p2[0]
        y = a * p0[1] + b * p1[1] + c * p2[1]
        pts.append([x, y])
    pts = np.round(np.array(pts, dtype=np.float32)).astype(np.int32)
    thick = int(max(2, round(thickness)))
    cv2.polylines(canvas, [pts], isClosed=False, color=1, thickness=thick, lineType=cv2.LINE_8)


def _sample_branch_axis(
    tx: float,
    ty: float,
    nx: float,
    ny: float,
    rng: np.random.Generator,
):
    # Branch/stub direction: mostly outward normal with tangent mix.
    a = float(rng.uniform(-0.8, 0.8))
    dx = nx + a * tx
    dy = ny + a * ty
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return nx, ny
    return dx / n, dy / n


def _draw_connected_stub(
    canvas: np.ndarray,
    x: float,
    y: float,
    tx: float,
    ty: float,
    nx: float,
    ny: float,
    width: float,
    rng: np.random.Generator,
    curve_prob: float,
    large_prob: float,
    max_width_px: float,
):
    # Connected protrusion: starts inside/at boundary and grows outward.
    if rng.random() < large_prob:
        width *= float(rng.uniform(1.4, 1.9))
    width = float(min(width, max_width_px))
    main_len = float(min(140.0, max(12.0, width * float(rng.uniform(2.0, 5.5)))))
    tan_shift = float(width * float(rng.uniform(-0.5, 0.5)))
    bend = float(width * float(rng.uniform(-0.35, 0.35)))

    sx, sy = x - nx * (0.35 * width), y - ny * (0.35 * width)  # ensure overlap with road
    mx, my = x + nx * (0.55 * main_len) + tx * bend, y + ny * (0.55 * main_len) + ty * bend
    ex, ey = x + nx * main_len + tx * tan_shift, y + ny * main_len + ty * tan_shift

    if rng.random() < curve_prob:
        _draw_quadratic_curve(canvas, (sx, sy), (mx, my), (ex, ey), thickness=width, n_points=11)
    else:
        pts = np.array([[sx, sy], [mx, my], [ex, ey]], dtype=np.float32)
        pts = np.round(pts).astype(np.int32)
        thick = int(min(max(2, int(round(max_width_px))), max(2, round(width))))
        cv2.polylines(canvas, [pts], isClosed=False, color=1, thickness=thick, lineType=cv2.LINE_8)


def _draw_free_add_when_no_road(
    canvas: np.ndarray,
    rng: np.random.Generator,
    nominal_width_px: float,
    min_add_width_px: int,
    curve_prob: float,
    large_prob: float,
    mega_edit_prob: float,
    mega_width_mult_low: float,
    mega_width_mult_high: float,
    mega_length_mult_low: float,
    mega_length_mult_high: float,
):
    h, w = canvas.shape
    n_ops = int(rng.integers(1, 3))
    base_w = float(max(min_add_width_px, nominal_width_px * float(rng.uniform(0.65, 1.15))))
    for _ in range(n_ops):
        side = int(rng.integers(0, 4))
        if side == 0:  # top -> down
            x0, y0 = float(rng.uniform(0, w - 1)), 0.0
            ang = float(rng.uniform(65, 115))
        elif side == 1:  # bottom -> up
            x0, y0 = float(rng.uniform(0, w - 1)), float(h - 1)
            ang = float(rng.uniform(245, 295))
        elif side == 2:  # left -> right
            x0, y0 = 0.0, float(rng.uniform(0, h - 1))
            ang = float(rng.uniform(-25, 25))
        else:  # right -> left
            x0, y0 = float(w - 1), float(rng.uniform(0, h - 1))
            ang = float(rng.uniform(155, 205))

        theta = math.radians(ang)
        dx, dy = math.cos(theta), math.sin(theta)

        width = float(min(44.0, base_w))
        length = float(max(72.0, nominal_width_px * float(rng.uniform(3.0, 6.2))))
        if rng.random() < large_prob:
            width = float(min(54.0, width * float(rng.uniform(1.15, 1.6))))
            length = float(min(380.0, length * float(rng.uniform(1.2, 2.0))))
        if mega_edit_prob > 0 and rng.random() < mega_edit_prob:
            width = float(min(64.0, width * float(rng.uniform(mega_width_mult_low, mega_width_mult_high))))
            length = float(min(460.0, length * float(rng.uniform(mega_length_mult_low, mega_length_mult_high))))

        x1 = x0 + dx * length
        y1 = y0 + dy * length
        if rng.random() < curve_prob:
            px, py = -dy, dx
            bend = float(rng.uniform(-0.35, 0.35) * length)
            cx = x0 + dx * (0.55 * length) + px * bend
            cy = y0 + dy * (0.55 * length) + py * bend
            _draw_quadratic_curve(canvas, (x0, y0), (cx, cy), (x1, y1), thickness=width, n_points=11)
        else:
            p0 = np.array([[x0, y0], [x1, y1]], dtype=np.float32)
            p0 = np.round(p0).astype(np.int32)
            cv2.polylines(
                canvas,
                [p0],
                isClosed=False,
                color=1,
                thickness=max(2, int(round(width))),
                lineType=cv2.LINE_8,
            )


def _apply_remove(
    mask: np.ndarray,
    rng: np.random.Generator,
    nominal_width_px: float,
    low_scale: float,
    high_scale: float,
    min_draw_px: int,
    thin_prob: float,
    thin_low_scale: float,
    thin_high_scale: float,
    min_thin_px: int,
    min_component_px: int,
    bite_prob: float,
    curve_prob: float,
    large_prob: float,
    mega_edit_prob: float,
    mega_width_mult_low: float,
    mega_width_mult_high: float,
    mega_length_mult_low: float,
    mega_length_mult_high: float,
) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    n_ops = int(rng.integers(1, 3))
    erase = np.zeros_like(out, dtype=np.uint8)
    h, w = out.shape
    for _ in range(n_ops):
        anchor = _pick_boundary_anchor(out, rng)
        if anchor is None:
            break
        x, y, tx, ty, out_nx, out_ny, contour, contour_idx = anchor
        in_nx, in_ny = -out_nx, -out_ny
        thickness = _sample_thickness(
            rng,
            nominal_width_px,
            low_scale,
            high_scale,
            min_draw_px,
            thin_prob=thin_prob,
            thin_low_scale=thin_low_scale,
            thin_high_scale=thin_high_scale,
            min_thin_px=min_thin_px,
        )
        use_branch = bool(rng.random() < bite_prob)
        if use_branch:
            # Boundary-connected bite toward interior.
            bx, by = _sample_branch_axis(tx, ty, in_nx, in_ny, rng)
            cut_wid = float(min(20.0, max(3.0, thickness * float(rng.uniform(0.30, 0.75)))))
            cut_len = float(min(120.0, max(12.0, cut_wid * float(rng.uniform(1.8, 4.5)))))
            if rng.random() < large_prob:
                cut_wid = float(min(28.0, cut_wid * float(rng.uniform(1.25, 1.8))))
                cut_len = float(min(210.0, cut_len * float(rng.uniform(1.3, 1.9))))
            if mega_edit_prob > 0 and rng.random() < mega_edit_prob:
                cut_wid = float(min(72.0, cut_wid * float(rng.uniform(mega_width_mult_low, mega_width_mult_high))))
                cut_len = float(min(460.0, cut_len * float(rng.uniform(mega_length_mult_low, mega_length_mult_high))))
            if rng.random() < curve_prob:
                # Slightly curved inward bite.
                sx, sy = x - bx * (0.25 * cut_wid), y - by * (0.25 * cut_wid)
                ex, ey = x + bx * cut_len, y + by * cut_len
                bend = float(rng.uniform(-0.8, 0.8) * cut_len)
                cx, cy = x + bx * (0.55 * cut_len) + tx * bend, y + by * (0.55 * cut_len) + ty * bend
                _draw_quadratic_curve(erase, (sx, sy), (cx, cy), (ex, ey), thickness=cut_wid, n_points=11)
            else:
                px, py = -by, bx
                cx = x + bx * (0.5 * cut_len)
                cy = y + by * (0.5 * cut_len)
                _draw_oriented_rect(erase, cx, cy, bx, by, px, py, cut_len, cut_wid)
        else:
            depth = float(min(20.0, max(3.0, thickness * float(rng.uniform(0.35, 0.85)))))
            length = float(min(130.0, max(12.0, depth * float(rng.uniform(2.4, 5.2)))))
            if rng.random() < large_prob:
                depth = float(min(28.0, depth * float(rng.uniform(1.25, 1.8))))
                length = float(min(220.0, length * float(rng.uniform(1.3, 2.0))))
            if mega_edit_prob > 0 and rng.random() < mega_edit_prob:
                depth = float(min(72.0, depth * float(rng.uniform(mega_width_mult_low, mega_width_mult_high))))
                length = float(min(460.0, length * float(rng.uniform(mega_length_mult_low, mega_length_mult_high))))
            # contour-following inward strip (more natural than free rectangle)
            _draw_contour_strip(
                erase,
                contour,
                contour_idx,
                out_nx,
                out_ny,
                half_len_px=int(round(0.5 * length)),
                strip_width_px=depth,
                inward=True,
            )

    if erase.any():
        erase = ((erase > 0) & (out > 0)).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        road_boundary = ((out > 0) & (cv2.erode(out, k, iterations=1) == 0)).astype(np.uint8)
        # keep edge-connected, sufficiently large components only
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(erase, connectivity=8)
        if n_cc > 1:
            keep = np.zeros_like(erase, dtype=np.uint8)
            for cid in range(1, n_cc):
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area < min_component_px:
                    continue
                comp = (cc == cid)
                if np.any(road_boundary[comp]):
                    keep[comp] = 1
            erase = keep
        erase = _smooth_binary(erase, radius=1)
        erase = cv2.morphologyEx(erase, cv2.MORPH_CLOSE, k, iterations=1)
        out[erase > 0] = 0
    return (out > 0).astype(np.uint8)


def _apply_add(
    mask: np.ndarray,
    rng: np.random.Generator,
    nominal_width_px: float,
    low_scale: float,
    high_scale: float,
    min_draw_px: int,
    thin_prob: float,
    thin_low_scale: float,
    thin_high_scale: float,
    min_thin_px: int,
    min_component_px: int,
    branch_prob: float,
    add_thickness_mult: float,
    add_width_low: float,
    add_width_high: float,
    add_branch_width_low: float,
    add_branch_width_high: float,
    min_add_width_px: int,
    rect_strip_prob: float,
    curve_prob: float,
    large_prob: float,
    mega_edit_prob: float,
    mega_width_mult_low: float,
    mega_width_mult_high: float,
    mega_length_mult_low: float,
    mega_length_mult_high: float,
) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    road_empty = int(out.sum()) == 0
    n_ops = int(rng.integers(1, 3))
    add = np.zeros_like(out, dtype=np.uint8)
    for _ in range(n_ops):
        anchor = _pick_boundary_anchor(out, rng)
        if anchor is None:
            if road_empty:
                _draw_free_add_when_no_road(
                    add,
                    rng,
                    nominal_width_px=nominal_width_px,
                    min_add_width_px=min_add_width_px,
                    curve_prob=curve_prob,
                    large_prob=large_prob,
                    mega_edit_prob=mega_edit_prob,
                    mega_width_mult_low=mega_width_mult_low,
                    mega_width_mult_high=mega_width_mult_high,
                    mega_length_mult_low=mega_length_mult_low,
                    mega_length_mult_high=mega_length_mult_high,
                )
            continue
        x, y, tx, ty, out_nx, out_ny, contour, contour_idx = anchor
        thickness = _sample_thickness(
            rng,
            nominal_width_px,
            low_scale,
            high_scale,
            min_draw_px,
            thin_prob=thin_prob,
            thin_low_scale=thin_low_scale,
            thin_high_scale=thin_high_scale,
            min_thin_px=min_thin_px,
        )
        thickness = float(max(min_draw_px, round(thickness * add_thickness_mult)))
        # Allow synthetic additions to match local road scale (avoid tiny blue slivers).
        base_max_w = float(max(28.0, nominal_width_px * 1.15))
        large_max_w = float(max(36.0, nominal_width_px * 1.45))
        use_branch = bool(rng.random() < branch_prob)
        if use_branch:
            # Connected side-branch (non-rectangular) from road boundary.
            bx, by = _sample_branch_axis(tx, ty, out_nx, out_ny, rng)
            branch_w = max(float(min_add_width_px), thickness * float(rng.uniform(add_branch_width_low, add_branch_width_high)))
            branch_w = min(base_max_w, branch_w)
            if mega_edit_prob > 0 and rng.random() < mega_edit_prob:
                branch_w = min(large_max_w, branch_w * float(rng.uniform(mega_width_mult_low, mega_width_mult_high)))
            _draw_connected_stub(
                add, x, y, tx, ty, bx, by, branch_w, rng, curve_prob, large_prob, large_max_w
            )
        else:
            # Edge-parallel strip (your second example style).
            width = float(min(base_max_w, max(float(min_add_width_px), thickness * float(rng.uniform(add_width_low, add_width_high)))))
            length = float(min(150.0, max(16.0, width * float(rng.uniform(2.8, 6.8)))))
            if rng.random() < large_prob:
                width = float(min(large_max_w, width * float(rng.uniform(1.2, 1.9))))
                length = float(min(260.0, length * float(rng.uniform(1.3, 2.1))))
            if mega_edit_prob > 0 and rng.random() < mega_edit_prob:
                width = float(min(large_max_w, width * float(rng.uniform(mega_width_mult_low, mega_width_mult_high))))
                length = float(min(460.0, length * float(rng.uniform(mega_length_mult_low, mega_length_mult_high))))
            if rng.random() < rect_strip_prob:
                # Straight attached strip (a small subset by design).
                mix = float(rng.uniform(-0.45, 0.45))
                ax = tx + mix * out_nx
                ay = ty + mix * out_ny
                an = math.hypot(ax, ay)
                if an > 1e-6:
                    ax, ay = ax / an, ay / an
                else:
                    ax, ay = tx, ty
                bx, by = -ay, ax
                if bx * out_nx + by * out_ny < 0:
                    bx, by = -bx, -by
                offset = 0.20 * width
                cx = x + bx * (offset + 0.5 * width)
                cy = y + by * (offset + 0.5 * width)
                _draw_oriented_rect(add, cx, cy, ax, ay, bx, by, length, width)
            else:
                # contour-following outward strip captures local road curvature
                _draw_contour_strip(
                    add,
                    contour,
                    contour_idx,
                    out_nx,
                    out_ny,
                    half_len_px=int(round(0.5 * length)),
                    strip_width_px=width,
                    inward=False,
                )
            # short connector so strip is definitely connected to road pixels.
            conn_len = float(max(5.0, width * float(rng.uniform(0.9, 1.6))))
            _draw_oriented_rect(add, x, y, out_nx, out_ny, tx, ty, conn_len, width * 0.9)
    k = np.ones((3, 3), np.uint8)
    add = cv2.morphologyEx((add > 0).astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1)

    # keep only sufficiently large components.
    # If road exists, enforce connectivity to existing roads.
    # If road is absent (pure non-road tile), allow unattached additions.
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(add, connectivity=8)
    if n_cc > 1:
        keep = np.zeros_like(add, dtype=np.uint8)
        road_anchor = cv2.dilate((out > 0).astype(np.uint8), k, iterations=2) if not road_empty else None
        for cid in range(1, n_cc):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < min_component_px:
                continue
            comp = (cc == cid)
            if road_empty or np.any(road_anchor[comp]):
                keep[comp] = 1
        add = keep

    add = _enforce_add_attachment(add, out)
    add = _smooth_binary(add, radius=1)
    add = _enforce_add_attachment(add, out)
    out[(add > 0) & (out == 0)] = 1
    return (out > 0).astype(np.uint8)


def _recompute_change(t1_fake: np.ndarray, t2: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    chg = np.zeros_like(t2, dtype=np.uint8)
    chg[(t1_fake == 0) & (t2 == 1)] = 1  # new
    chg[(t1_fake == 1) & (t2 == 0)] = 2  # removed
    chg[(t1_fake == 1) & (t2 == 1)] = 3  # unchanged
    chg[ignore_mask] = 255
    return chg


def _apply_map_noise(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Conservative smoothing only: fill pinholes, avoid creating gaps/islands.
    out = mask.copy().astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return (out > 0).astype(np.uint8)


def _apply_quadrant_erase(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    h, w = mask.shape
    h2, w2 = h // 2, w // 2
    quads = [
        (slice(0, h2), slice(0, w2)),
        (slice(0, h2), slice(w2, w)),
        (slice(h2, h), slice(0, w2)),
        (slice(h2, h), slice(w2, w)),
    ]
    q = int(rng.integers(0, 4))
    ys, xs = quads[q]
    out[ys, xs] = (mask[ys, xs] > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return (out > 0).astype(np.uint8)


def _apply_affine_jitter(
    mask: np.ndarray,
    rng: np.random.Generator,
    rotate_deg: float,
    shift_px: float,
    scale_low: float,
    scale_high: float,
) -> np.ndarray:
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return (mask > 0).astype(np.uint8)
    angle = float(rng.uniform(-rotate_deg, rotate_deg))
    scale = float(rng.uniform(scale_low, scale_high))
    tx = float(rng.uniform(-shift_px, shift_px))
    ty = float(rng.uniform(-shift_px, shift_px))
    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(
        (mask > 0).astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Reject overly destructive warps.
    if int(warped.sum()) < max(8, int(0.15 * int((mask > 0).sum()))):
        return (mask > 0).astype(np.uint8)
    return (warped > 0).astype(np.uint8)


def _choose_mode(
    rng: np.random.Generator,
    mode_names: np.ndarray,
    mode_probs: np.ndarray,
    balance_c12: bool,
    balance_ratio_high: float,
    cls1_pixels: int,
    cls2_pixels: int,
) -> str:
    if not balance_c12:
        return str(rng.choice(mode_names, p=mode_probs))

    # bootstrap: force missing side first
    if cls1_pixels == 0 and cls2_pixels > 0:
        return "remove"
    if cls2_pixels == 0 and cls1_pixels > 0:
        return "add"
    if cls1_pixels == 0 and cls2_pixels == 0:
        return str(rng.choice(mode_names, p=mode_probs))

    ratio = cls1_pixels / float(max(cls2_pixels, 1))
    if ratio > balance_ratio_high:
        return "add"
    if ratio < (1.0 / balance_ratio_high):
        return "remove"
    return str(rng.choice(mode_names, p=mode_probs))


def _quality_ok(chg: np.ndarray, min_change_pixels: int, max_change_ratio: float,
                mode: str, min_target_pixels: int, min_target_ratio: float) -> bool:
    valid = (chg != 255)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return False
    c1 = int((chg == 1).sum())
    c2 = int((chg == 2).sum())
    n_change = c1 + c2
    if n_change < min_change_pixels:
        return False
    if n_change / float(n_valid) > max_change_ratio:
        return False
    if mode == "remove":
        target = c1
    elif mode == "add":
        target = c2
    else:
        target = min(c1, c2)
    if target < min_target_pixels:
        return False
    if target / float(n_valid) < min_target_ratio:
        return False
    return True


def _component_touches_class(chg: np.ndarray, comp_mask: np.ndarray, cls_value: int) -> bool:
    k = np.ones((3, 3), np.uint8)
    d = cv2.dilate(comp_mask.astype(np.uint8), k, iterations=1)
    return bool(np.any((d > 0) & (chg == cls_value)))


def _component_hole_area(comp_mask: np.ndarray) -> int:
    comp = (comp_mask > 0).astype(np.uint8)
    if comp.max() == 0:
        return 0
    ff = comp.copy()
    h, w = ff.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood, seedPoint=(0, 0), newVal=2)
    holes = (ff == 0) & (comp == 0)
    return int(np.sum(holes))


def _nearest_point_bridge(add_mask: np.ndarray, comp_mask: np.ndarray, road_mask: np.ndarray, thickness: int):
    comp_pts = np.argwhere(comp_mask > 0)  # y,x
    road_pts = np.argwhere(road_mask > 0)  # y,x
    if comp_pts.size == 0 or road_pts.size == 0:
        return

    # Subsample for speed if needed.
    rng = np.random.default_rng(123)
    if comp_pts.shape[0] > 300:
        comp_pts = comp_pts[rng.choice(comp_pts.shape[0], size=300, replace=False)]
    if road_pts.shape[0] > 600:
        road_pts = road_pts[rng.choice(road_pts.shape[0], size=600, replace=False)]

    d2 = np.sum((comp_pts[:, None, :] - road_pts[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    y1, x1 = comp_pts[i]
    y2, x2 = road_pts[j]
    cv2.line(add_mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=max(1, int(thickness)))


def _enforce_add_attachment(add_mask: np.ndarray, road_mask: np.ndarray):
    out = (add_mask > 0).astype(np.uint8)
    road = (road_mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)

    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n_cc <= 1:
        return out

    road_adj = cv2.dilate(road, k, iterations=1)  # direct adjacency allowed
    for cid in range(1, n_cc):
        comp = (cc == cid).astype(np.uint8)
        if np.any((comp > 0) & (road_adj > 0)):
            continue

        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        width_est = float(np.max(dist[comp > 0]) * 2.0) if np.any(comp > 0) else 2.0
        bridge_th = int(max(2, round(width_est * 0.6)))
        _nearest_point_bridge(out, comp, road, thickness=bridge_th)

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return (out > 0).astype(np.uint8)


def _suppress_small_change_components(
    t1_fake: np.ndarray,
    t2_bin: np.ndarray,
    ignore_mask: np.ndarray,
    min_change_component_pixels: int,
):
    """Remove tiny cls1/cls2 components by reverting T1_fake locally."""
    out = t1_fake.copy().astype(np.uint8)
    chg = _recompute_change(out, t2_bin, ignore_mask)
    if min_change_component_pixels <= 1:
        return out, chg, 0, 0

    has_bg_cls0 = bool(np.any(chg == 0))
    has_unch_cls3 = bool(np.any(chg == 3))

    removed_comp = 0
    removed_pix = 0
    removed_no_bg_touch = 0
    removed_no_unch_touch = 0
    removed_holey = 0
    min_keep_area = int(min_change_component_pixels)
    if not has_unch_cls3:
        # For pure non-road scenes (no cls3), keep smaller valid changes.
        min_keep_area = max(64, int(min_change_component_pixels // 4))
    for cls, revert_t1_value in ((1, 1), (2, 0)):
        m = (chg == cls).astype(np.uint8)
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n_cc <= 1:
            continue
        for cid in range(1, n_cc):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            comp = (cc == cid)
            touch_bg = _component_touches_class(chg, comp, cls_value=0)
            touch_unch = _component_touches_class(chg, comp, cls_value=3)
            touch_border = bool(
                np.any(comp[0, :]) or np.any(comp[-1, :]) or np.any(comp[:, 0]) or np.any(comp[:, -1])
            )
            hole_area = _component_hole_area(comp.astype(np.uint8))
            is_holey = hole_area > max(8, int(0.03 * area))
            # Reject all border-touching components in synth to avoid clipped spikes.
            border_bad = touch_border
            bg_ok = (not has_bg_cls0) or touch_bg
            unch_ok = (not has_unch_cls3) or touch_unch
            if area >= min_keep_area and bg_ok and unch_ok and (not is_holey) and (not border_bad):
                continue
            out[comp] = np.uint8(revert_t1_value)
            removed_comp += 1
            removed_pix += area
            if not touch_bg:
                removed_no_bg_touch += 1
            if not touch_unch:
                removed_no_unch_touch += 1
            if is_holey:
                removed_holey += 1

    if removed_comp > 0:
        chg = _recompute_change(out, t2_bin, ignore_mask)
    return out, chg, removed_comp, removed_pix, removed_no_bg_touch, removed_no_unch_touch, removed_holey


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    width_low, width_high = _parse_pair(args.width_scale_range, "--width-scale-range")
    thin_width_low, thin_width_high = _parse_pair(args.thin_width_scale_range, "--thin-width-scale-range")
    add_width_low, add_width_high = _parse_pair(args.add_width_scale_range, "--add-width-scale-range")
    add_branch_width_low, add_branch_width_high = _parse_pair(
        args.add_branch_width_scale_range, "--add-branch-width-scale-range")
    mega_width_mult_low, mega_width_mult_high = _parse_pair(
        args.mega_width_mult_range, "--mega-width-mult-range")
    mega_length_mult_low, mega_length_mult_high = _parse_pair(
        args.mega_length_mult_range, "--mega-length-mult-range")
    affine_scale_low, affine_scale_high = _parse_pair(args.affine_scale_range, "--affine-scale-range")
    if args.pixel_size_m <= 0:
        raise ValueError("--pixel-size-m must be > 0")
    if args.thin_edit_prob < 0 or args.thin_edit_prob > 1:
        raise ValueError("--thin-edit-prob must be in [0,1]")
    if args.min_thin_draw_thickness_px <= 0:
        raise ValueError("--min-thin-draw-thickness-px must be > 0")
    if args.min_t2_road_pixels < 0:
        raise ValueError("--min-t2-road-pixels must be >= 0")
    if args.min_road_union_pixels < 0:
        raise ValueError("--min-road-union-pixels must be >= 0")
    if not (0.0 <= args.max_no_road_source_ratio <= 1.0):
        raise ValueError("--max-no-road-source-ratio must be in [0,1]")
    if args.min_edit_component_pixels <= 0:
        raise ValueError("--min-edit-component-pixels must be > 0")
    if args.min_change_component_pixels <= 0:
        raise ValueError("--min-change-component-pixels must be > 0")
    if args.balance_ratio_high <= 1.0:
        raise ValueError("--balance-ratio-high must be > 1.0")
    if not (0.0 <= args.add_branch_prob <= 1.0):
        raise ValueError("--add-branch-prob must be in [0,1]")
    if args.add_thickness_mult <= 0:
        raise ValueError("--add-thickness-mult must be > 0")
    if args.min_add_width_px <= 0:
        raise ValueError("--min-add-width-px must be > 0")
    if not (0.0 <= args.add_rect_strip_prob <= 1.0):
        raise ValueError("--add-rect-strip-prob must be in [0,1]")
    if not (0.0 <= args.remove_bite_prob <= 1.0):
        raise ValueError("--remove-bite-prob must be in [0,1]")
    if not (0.0 <= args.curve_edit_prob <= 1.0):
        raise ValueError("--curve-edit-prob must be in [0,1]")
    if not (0.0 <= args.large_edit_prob <= 1.0):
        raise ValueError("--large-edit-prob must be in [0,1]")
    if not (0.0 <= args.mega_edit_prob <= 1.0):
        raise ValueError("--mega-edit-prob must be in [0,1]")
    if not (0.0 <= args.quad_erase_prob <= 1.0):
        raise ValueError("--quad-erase-prob must be in [0,1]")
    if not (0.0 <= args.affine_jitter_prob <= 1.0):
        raise ValueError("--affine-jitter-prob must be in [0,1]")
    if args.affine_max_rotate_deg < 0:
        raise ValueError("--affine-max-rotate-deg must be >= 0")
    if args.affine_max_shift_px < 0:
        raise ValueError("--affine-max-shift-px must be >= 0")
    fallback_width_px = args.target_road_width_m / args.pixel_size_m

    src_train_split = args.src_split_dir / args.train_split_name
    if not src_train_split.exists():
        raise FileNotFoundError(src_train_split)

    mode_probs = [float(x.strip()) for x in args.mode_probs.split(",")]
    if len(mode_probs) != 3 or any(x < 0 for x in mode_probs):
        raise ValueError("--mode-probs must be three non-negative values")
    mode_probs = np.array(mode_probs, dtype=np.float64)
    mode_probs = mode_probs / mode_probs.sum()
    mode_names = np.array(["remove", "add", "both"])

    train_ids = _read_ids(src_train_split)
    num_train = len(train_ids)
    num_synth = int(round(num_train * args.synth_ratio)) if args.num_synth is None else int(args.num_synth)
    if num_synth <= 0:
        raise ValueError("num_synth must be > 0")

    # Candidate pool: no existing change(1|2), optional road constraints, optional T2 black-image exclusion.
    candidates: List[TileRef] = []
    candidate_no_road_flags: List[bool] = []
    for tid in train_ids:
        cp = args.src_root / "labels" / "change" / tid.region / f"{tid.tile}.tif"
        c, _ = _read_u8(cp)
        valid = (c != 255)
        has_change = np.isin(c[valid], [1, 2]).any()
        t1p = args.src_root / "labels" / "t1" / tid.region / f"{tid.tile}.tif"
        t2p = args.src_root / "labels" / "t2" / tid.region / f"{tid.tile}.tif"
        t2ip = args.src_root / "images" / "t2" / tid.region / f"{tid.tile}.tif"
        t1, _ = _read_u8(t1p)
        t2, _ = _read_u8(t2p)
        road_px = int((t2 == 1).sum())
        road_union_px = int(((t1 == 1) | (t2 == 1)).sum())
        black_t2_img = _is_black_t2_image(t2ip) if args.exclude_black_t2_image else False
        if (
            (not has_change)
            and (road_px >= args.min_t2_road_pixels)
            and (road_union_px >= args.min_road_union_pixels)
            and (not black_t2_img)
        ):
            candidates.append(tid)
            candidate_no_road_flags.append(road_union_px == 0)
    if not candidates:
        raise RuntimeError("No no-change candidates found in train split.")
    if args.max_synth_per_source <= 0:
        raise ValueError("--max-synth-per-source must be >= 1")
    if num_synth > len(candidates) * args.max_synth_per_source:
        raise ValueError(
            f"Requested num_synth={num_synth} exceeds candidate capacity "
            f"{len(candidates) * args.max_synth_per_source} "
            f"(candidates={len(candidates)}, max_synth_per_source={args.max_synth_per_source})."
        )

    # Pick source tiles while respecting max_synth_per_source.
    max_no_road_sources = int(round(num_synth * args.max_no_road_source_ratio))
    if args.max_synth_per_source == 1:
        order = rng.permutation(len(candidates))
        selected_sources = []
        selected_no_road = 0
        for i in order:
            idx = int(i)
            is_no_road = bool(candidate_no_road_flags[idx])
            if is_no_road and selected_no_road >= max_no_road_sources:
                continue
            selected_sources.append(candidates[idx])
            if is_no_road:
                selected_no_road += 1
            if len(selected_sources) >= num_synth:
                break
    else:
        src_counts = np.zeros((len(candidates),), dtype=np.int32)
        selected_sources = []
        selected_no_road = 0
        guard = 0
        while len(selected_sources) < num_synth:
            guard += 1
            if guard > len(candidates) * max(20, args.max_synth_per_source * 10):
                break
            idx = int(rng.integers(0, len(candidates)))
            if src_counts[idx] >= args.max_synth_per_source:
                continue
            is_no_road = bool(candidate_no_road_flags[idx])
            if is_no_road and selected_no_road >= max_no_road_sources:
                continue
            src_counts[idx] += 1
            selected_sources.append(candidates[idx])
            if is_no_road:
                selected_no_road += 1

    if len(selected_sources) < num_synth:
        raise ValueError(
            f"Not enough candidates after applying --max-no-road-source-ratio={args.max_no_road_source_ratio}. "
            f"selected={len(selected_sources)}, target={num_synth}. "
            "Lower ratio constraint or increase candidate pool."
        )

    out_split_dir = args.out_root / "splits"
    out_split_path = out_split_dir / args.out_split_name
    out_summary_path = out_split_dir / (Path(args.out_split_name).stem + "_summary.json")

    stats = {
        "num_train": num_train,
        "num_candidates_no_change": len(candidates),
        "num_synth_target": num_synth,
        "num_synth_written": 0,
        "target_road_width_m": args.target_road_width_m,
        "pixel_size_m": args.pixel_size_m,
        "fallback_width_px": fallback_width_px,
        "max_synth_per_source": args.max_synth_per_source,
        "max_no_road_source_ratio": args.max_no_road_source_ratio,
        "selected_no_road_sources": int(selected_no_road),
        "min_t2_road_pixels": args.min_t2_road_pixels,
        "min_road_union_pixels": args.min_road_union_pixels,
        "thin_edit_prob": args.thin_edit_prob,
        "thin_width_scale_range": [thin_width_low, thin_width_high],
        "min_thin_draw_thickness_px": args.min_thin_draw_thickness_px,
        "min_edit_component_pixels": args.min_edit_component_pixels,
        "add_branch_prob": args.add_branch_prob,
        "add_thickness_mult": args.add_thickness_mult,
        "add_width_scale_range": [add_width_low, add_width_high],
        "add_branch_width_scale_range": [add_branch_width_low, add_branch_width_high],
        "min_add_width_px": args.min_add_width_px,
        "add_rect_strip_prob": args.add_rect_strip_prob,
        "remove_bite_prob": args.remove_bite_prob,
        "curve_edit_prob": args.curve_edit_prob,
        "large_edit_prob": args.large_edit_prob,
        "mega_edit_prob": args.mega_edit_prob,
        "mega_width_mult_range": [mega_width_mult_low, mega_width_mult_high],
        "mega_length_mult_range": [mega_length_mult_low, mega_length_mult_high],
        "exclude_black_t2_image": bool(args.exclude_black_t2_image),
        "quad_erase_prob": args.quad_erase_prob,
        "affine_jitter_prob": args.affine_jitter_prob,
        "affine_max_rotate_deg": args.affine_max_rotate_deg,
        "affine_max_shift_px": args.affine_max_shift_px,
        "affine_scale_range": [affine_scale_low, affine_scale_high],
        "style_schedule_slots": 6,
        "style_schedule_enabled": True,
        "min_change_component_pixels": args.min_change_component_pixels,
        "mode_counts": {"remove": 0, "add": 0, "both": 0},
        "class_pixels": {"cls1_new": 0, "cls2_removed": 0},
        "small_change_components_removed": 0,
        "small_change_pixels_removed": 0,
        "small_change_no_bg_touch_removed": 0,
        "small_change_no_unch_touch_removed": 0,
        "small_change_holey_removed": 0,
        "estimated_width_px_sum": 0.0,
    }

    synth_lines = []
    if not args.dry_run:
        out_split_dir.mkdir(parents=True, exist_ok=True)

    for i, src_tid in enumerate(selected_sources):
        t2_img_path = args.src_root / "images" / "t2" / src_tid.region / f"{src_tid.tile}.tif"
        t2_lbl_path = args.src_root / "labels" / "t2" / src_tid.region / f"{src_tid.tile}.tif"
        t1_src_path = args.src_root / "labels" / "t1" / src_tid.region / f"{src_tid.tile}.tif"
        chg_src_path = args.src_root / "labels" / "change" / src_tid.region / f"{src_tid.tile}.tif"

        t2, t2_prof = _read_u8(t2_lbl_path)
        t1_src, t1_prof = _read_u8(t1_src_path)
        chg_src, chg_prof = _read_u8(chg_src_path)
        ignore_mask = (chg_src == 255) | (t2 == 255)
        t2_bin = (t2 > 0).astype(np.uint8)
        t1_bin = (t1_src > 0).astype(np.uint8)
        is_no_road_tile = int(((t1_bin > 0) | (t2_bin > 0)).sum()) == 0
        local_min_change_component_pixels = (
            args.min_change_component_pixels if not is_no_road_tile else min(args.min_change_component_pixels, 128)
        )
        local_min_target_pixels = (
            args.min_target_pixels if not is_no_road_tile else min(args.min_target_pixels, 256)
        )
        local_min_target_ratio = (
            args.min_target_ratio if not is_no_road_tile else min(args.min_target_ratio, 0.0010)
        )
        road_width_px = _estimate_road_width_px(t1_bin, fallback_width_px)
        stats["estimated_width_px_sum"] += road_width_px

        chosen_mode = None
        t1_fake = None
        chg_fake = None

        # Style scheduling: force diverse transform patterns to appear uniformly.
        style_slot = int(i % 6)
        local_quad_erase_prob = args.quad_erase_prob
        local_affine_jitter_prob = args.affine_jitter_prob
        local_curve_edit_prob = args.curve_edit_prob
        local_large_edit_prob = args.large_edit_prob
        local_mega_edit_prob = args.mega_edit_prob
        local_add_rect_strip_prob = args.add_rect_strip_prob
        local_remove_bite_prob = args.remove_bite_prob
        local_thin_edit_prob = args.thin_edit_prob

        if style_slot == 0:
            local_quad_erase_prob = max(local_quad_erase_prob, 1.0)
        elif style_slot == 1:
            local_affine_jitter_prob = max(local_affine_jitter_prob, 1.0)
        elif style_slot == 2:
            local_curve_edit_prob = max(local_curve_edit_prob, 0.95)
        elif style_slot == 3:
            local_add_rect_strip_prob = max(local_add_rect_strip_prob, 0.65)
            local_remove_bite_prob = max(local_remove_bite_prob, 0.65)
        elif style_slot == 4:
            local_mega_edit_prob = max(local_mega_edit_prob, 0.35)
            local_large_edit_prob = max(local_large_edit_prob, 0.95)
        else:
            local_thin_edit_prob = max(local_thin_edit_prob, 0.40)

        for _ in range(args.max_tries_per_sample):
            if is_no_road_tile:
                # For pure non-road tiles, force add-style synthesis (draw roads on T1_fake).
                mode = "add"
            else:
                mode = _choose_mode(
                    rng=rng,
                    mode_names=mode_names,
                    mode_probs=mode_probs,
                    balance_c12=args.balance_c12,
                    balance_ratio_high=args.balance_ratio_high,
                    cls1_pixels=int(stats["class_pixels"]["cls1_new"]),
                    cls2_pixels=int(stats["class_pixels"]["cls2_removed"]),
                )
            base = t1_bin.copy()
            if local_affine_jitter_prob > 0 and rng.random() < local_affine_jitter_prob:
                base = _apply_affine_jitter(
                    base,
                    rng,
                    rotate_deg=args.affine_max_rotate_deg,
                    shift_px=args.affine_max_shift_px,
                    scale_low=affine_scale_low,
                    scale_high=affine_scale_high,
                )
            # Core extreme transform:
            # For remove/both, always keep exactly one random quadrant and erase others.
            if mode in ("remove", "both"):
                base = _apply_quadrant_erase(base, rng)
            elif local_quad_erase_prob > 0 and rng.random() < local_quad_erase_prob:
                base = _apply_quadrant_erase(base, rng)
            if mode == "remove":
                base = _apply_remove(
                    base,
                    rng,
                    road_width_px,
                    width_low,
                    width_high,
                    args.min_draw_thickness_px,
                    local_thin_edit_prob,
                    thin_width_low,
                    thin_width_high,
                    args.min_thin_draw_thickness_px,
                    args.min_edit_component_pixels,
                    local_remove_bite_prob,
                    local_curve_edit_prob,
                    local_large_edit_prob,
                    local_mega_edit_prob,
                    mega_width_mult_low,
                    mega_width_mult_high,
                    mega_length_mult_low,
                    mega_length_mult_high,
                )
            elif mode == "add":
                base = _apply_add(
                    base,
                    rng,
                    road_width_px,
                    width_low,
                    width_high,
                    args.min_draw_thickness_px,
                    local_thin_edit_prob,
                    thin_width_low,
                    thin_width_high,
                    args.min_thin_draw_thickness_px,
                    args.min_edit_component_pixels,
                    args.add_branch_prob,
                    args.add_thickness_mult,
                    add_width_low,
                    add_width_high,
                    add_branch_width_low,
                    add_branch_width_high,
                    args.min_add_width_px,
                    local_add_rect_strip_prob,
                    local_curve_edit_prob,
                    local_large_edit_prob,
                    local_mega_edit_prob,
                    mega_width_mult_low,
                    mega_width_mult_high,
                    mega_length_mult_low,
                    mega_length_mult_high,
                )
            else:
                base = _apply_remove(
                    base,
                    rng,
                    road_width_px,
                    width_low,
                    width_high,
                    args.min_draw_thickness_px,
                    local_thin_edit_prob,
                    thin_width_low,
                    thin_width_high,
                    args.min_thin_draw_thickness_px,
                    args.min_edit_component_pixels,
                    local_remove_bite_prob,
                    local_curve_edit_prob,
                    local_large_edit_prob,
                    local_mega_edit_prob,
                    mega_width_mult_low,
                    mega_width_mult_high,
                    mega_length_mult_low,
                    mega_length_mult_high,
                )
                base = _apply_add(
                    base,
                    rng,
                    road_width_px,
                    width_low,
                    width_high,
                    args.min_draw_thickness_px,
                    local_thin_edit_prob,
                    thin_width_low,
                    thin_width_high,
                    args.min_thin_draw_thickness_px,
                    args.min_edit_component_pixels,
                    args.add_branch_prob,
                    args.add_thickness_mult,
                    add_width_low,
                    add_width_high,
                    add_branch_width_low,
                    add_branch_width_high,
                    args.min_add_width_px,
                    local_add_rect_strip_prob,
                    local_curve_edit_prob,
                    local_large_edit_prob,
                    local_mega_edit_prob,
                    mega_width_mult_low,
                    mega_width_mult_high,
                    mega_length_mult_low,
                    mega_length_mult_high,
                )
            base = _apply_map_noise(base, rng)
            base, c, rm_comp, rm_pix, rm_nobg, rm_nounch, rm_holey = _suppress_small_change_components(
                base, t2_bin, ignore_mask, local_min_change_component_pixels
            )
            if _quality_ok(
                c,
                args.min_change_pixels,
                args.max_change_ratio,
                mode=mode,
                min_target_pixels=local_min_target_pixels,
                min_target_ratio=local_min_target_ratio,
            ):
                chosen_mode = mode
                t1_fake = base
                chg_fake = c
                stats["small_change_components_removed"] += int(rm_comp)
                stats["small_change_pixels_removed"] += int(rm_pix)
                stats["small_change_no_bg_touch_removed"] += int(rm_nobg)
                stats["small_change_no_unch_touch_removed"] += int(rm_nounch)
                stats["small_change_holey_removed"] += int(rm_holey)
                break

        if t1_fake is None:
            # Fallback: stronger perturbation on original T1.
            base = t1_bin.copy()
            if local_affine_jitter_prob > 0 and rng.random() < local_affine_jitter_prob:
                base = _apply_affine_jitter(
                    base,
                    rng,
                    rotate_deg=args.affine_max_rotate_deg,
                    shift_px=args.affine_max_shift_px,
                    scale_low=affine_scale_low,
                    scale_high=affine_scale_high,
                )
            # Keep fallback consistent with main path.
            base = _apply_quadrant_erase(base, rng)
            base = _apply_remove(
                base,
                rng,
                road_width_px,
                width_low,
                width_high,
                args.min_draw_thickness_px,
                local_thin_edit_prob,
                thin_width_low,
                thin_width_high,
                args.min_thin_draw_thickness_px,
                args.min_edit_component_pixels,
                local_remove_bite_prob,
                local_curve_edit_prob,
                local_large_edit_prob,
                local_mega_edit_prob,
                mega_width_mult_low,
                mega_width_mult_high,
                mega_length_mult_low,
                mega_length_mult_high,
            )
            base = _apply_add(
                base,
                rng,
                road_width_px,
                width_low,
                width_high,
                args.min_draw_thickness_px,
                local_thin_edit_prob,
                thin_width_low,
                thin_width_high,
                args.min_thin_draw_thickness_px,
                args.min_edit_component_pixels,
                args.add_branch_prob,
                args.add_thickness_mult,
                add_width_low,
                add_width_high,
                add_branch_width_low,
                add_branch_width_high,
                args.min_add_width_px,
                local_add_rect_strip_prob,
                local_curve_edit_prob,
                local_large_edit_prob,
                local_mega_edit_prob,
                mega_width_mult_low,
                mega_width_mult_high,
                mega_length_mult_low,
                mega_length_mult_high,
            )
            t1_fake = _apply_map_noise(base, rng)
            t1_fake, chg_fake, rm_comp, rm_pix, rm_nobg, rm_nounch, rm_holey = _suppress_small_change_components(
                t1_fake, t2_bin, ignore_mask, local_min_change_component_pixels
            )
            if is_no_road_tile and int(np.isin(chg_fake, [1, 2]).sum()) == 0:
                # Ensure non-road sources still produce synthetic road change.
                forced = np.zeros_like(t1_fake, dtype=np.uint8)
                _draw_free_add_when_no_road(
                    forced,
                    rng,
                    nominal_width_px=max(18.0, road_width_px),
                    min_add_width_px=max(args.min_add_width_px, 12),
                    curve_prob=max(args.curve_edit_prob, 0.7),
                    large_prob=max(args.large_edit_prob, 0.8),
                    mega_edit_prob=max(args.mega_edit_prob, 0.15),
                    mega_width_mult_low=mega_width_mult_low,
                    mega_width_mult_high=mega_width_mult_high,
                    mega_length_mult_low=mega_length_mult_low,
                    mega_length_mult_high=mega_length_mult_high,
                )
                k = np.ones((3, 3), np.uint8)
                forced = cv2.morphologyEx((forced > 0).astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1)
                t1_fake = forced
                chg_fake = _recompute_change(t1_fake, t2_bin, ignore_mask)
            stats["small_change_components_removed"] += int(rm_comp)
            stats["small_change_pixels_removed"] += int(rm_pix)
            stats["small_change_no_bg_touch_removed"] += int(rm_nobg)
            stats["small_change_no_unch_touch_removed"] += int(rm_nounch)
            stats["small_change_holey_removed"] += int(rm_holey)
            chosen_mode = "both"

        # Strict guard: never write a synthetic tile with zero cls1/cls2 change.
        if int(np.isin(chg_fake, [1, 2]).sum()) == 0:
            base = _apply_quadrant_erase(t1_bin.copy(), rng)
            base = _apply_remove(
                base,
                rng,
                road_width_px,
                width_low,
                width_high,
                args.min_draw_thickness_px,
                max(local_thin_edit_prob, 0.40),
                thin_width_low,
                thin_width_high,
                args.min_thin_draw_thickness_px,
                args.min_edit_component_pixels,
                max(local_remove_bite_prob, 0.65),
                max(local_curve_edit_prob, 0.90),
                max(local_large_edit_prob, 0.95),
                max(local_mega_edit_prob, 0.35),
                mega_width_mult_low,
                mega_width_mult_high,
                mega_length_mult_low,
                mega_length_mult_high,
            )
            base = _apply_add(
                base,
                rng,
                road_width_px,
                width_low,
                width_high,
                args.min_draw_thickness_px,
                max(local_thin_edit_prob, 0.40),
                thin_width_low,
                thin_width_high,
                args.min_thin_draw_thickness_px,
                args.min_edit_component_pixels,
                args.add_branch_prob,
                args.add_thickness_mult,
                add_width_low,
                add_width_high,
                add_branch_width_low,
                add_branch_width_high,
                args.min_add_width_px,
                max(local_add_rect_strip_prob, 0.65),
                max(local_curve_edit_prob, 0.90),
                max(local_large_edit_prob, 0.95),
                max(local_mega_edit_prob, 0.35),
                mega_width_mult_low,
                mega_width_mult_high,
                mega_length_mult_low,
                mega_length_mult_high,
            )
            base = _apply_map_noise(base, rng)
            base, c, rm_comp2, rm_pix2, rm_nobg2, rm_nounch2, rm_holey2 = _suppress_small_change_components(
                base, t2_bin, ignore_mask, local_min_change_component_pixels
            )
            if _quality_ok(
                c,
                args.min_change_pixels,
                args.max_change_ratio,
                mode="both",
                min_target_pixels=local_min_target_pixels,
                min_target_ratio=local_min_target_ratio,
            ) and int(np.isin(c, [1, 2]).sum()) > 0:
                t1_fake = base
                chg_fake = c
                chosen_mode = "both"
                stats["small_change_components_removed"] += int(rm_comp2)
                stats["small_change_pixels_removed"] += int(rm_pix2)
                stats["small_change_no_bg_touch_removed"] += int(rm_nobg2)
                stats["small_change_no_unch_touch_removed"] += int(rm_nounch2)
                stats["small_change_holey_removed"] += int(rm_holey2)
            else:
                # Discard invalid no-change synthetic and continue.
                continue

        synth_tile = f"{src_tid.tile}_syn_{i:05d}"
        synth_lines.append(f"{src_tid.region}/{synth_tile}")

        stats["num_synth_written"] += 1
        stats["mode_counts"][chosen_mode] += 1
        stats["class_pixels"]["cls1_new"] += int((chg_fake == 1).sum())
        stats["class_pixels"]["cls2_removed"] += int((chg_fake == 2).sum())

        if args.dry_run:
            continue

        # Write t1 fake + change fake
        t1_out = args.out_root / "labels" / "t1" / src_tid.region / f"{synth_tile}.tif"
        chg_out = args.out_root / "labels" / "change" / src_tid.region / f"{synth_tile}.tif"
        _save_u8(t1_out, t1_fake, t1_prof, compress=args.compress)
        _save_u8(chg_out, chg_fake, chg_prof, compress=args.compress)

        # Copy/link unchanged inputs for this synthetic sample
        t2_img_out = args.out_root / "images" / "t2" / src_tid.region / f"{synth_tile}.tif"
        t2_lbl_out = args.out_root / "labels" / "t2" / src_tid.region / f"{synth_tile}.tif"
        _copy_or_link(t2_img_path, t2_img_out, mode=args.copy_mode)
        _copy_or_link(t2_lbl_path, t2_lbl_out, mode=args.copy_mode)

    if stats["num_synth_written"] > 0:
        stats["estimated_width_px_mean"] = stats["estimated_width_px_sum"] / float(stats["num_synth_written"])
    else:
        stats["estimated_width_px_mean"] = 0.0
    del stats["estimated_width_px_sum"]

    if args.dry_run:
        print("[dry-run] nothing written.")
    else:
        out_split_path.write_text("\n".join(synth_lines) + "\n")
        out_summary_path.write_text(json.dumps(stats, indent=2))
        print(f"[done] wrote split: {out_split_path}")
        print(f"[done] wrote summary: {out_summary_path}")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
