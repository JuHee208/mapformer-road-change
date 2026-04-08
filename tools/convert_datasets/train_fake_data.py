#!/usr/bin/env python3
"""Build synthetic T1-fake road dataset with explicit methods.

Methods
1) qdrop_new: split T1 into 4 quadrants, keep one random quadrant only.
   - T1 road(1)->0 in 3 quadrants, so cls1(new) is induced vs fixed T2.
2) near_add_removed: add road-like strokes near existing T1 roads.
   - T1 0->1 additions induce cls2(removed) vs fixed T2.
3) noroad_add_removed: for tiles with no road in (T1|T2), draw roads on T1.
   - also induces cls2(removed), but skip if T2 image is all-zero.
4) endpoint_trim_new: trim from road endpoints (1->0) to make cls1(new).
5) endpoint_extend_removed: extend from road endpoints (0->1) to make cls2(removed).
6) endpoint_both_mix: endpoint trim + endpoint extend in one tile (cls1 and cls2 together).
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import rasterio


@dataclass
class TileId:
    region: str
    tile: str

    @property
    def id(self) -> str:
        return f"{self.region}/{self.tile}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--src-split-dir", type=Path, required=True)
    p.add_argument("--train-split-name", type=str, default="train_drop_black1000.txt")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--out-split-name", type=str, default="train_fake_data.txt")
    p.add_argument("--num-synth", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method-probs", type=str, default="0.22,0.18,0.15,0.15,0.15,0.15",
                   help="qdrop_new,near_add_removed,noroad_add_removed,endpoint_trim_new,endpoint_extend_removed,endpoint_both_mix")
    p.add_argument("--balance-c12", action="store_true", default=True,
                   help="Bias method sampling to keep cls1/cls2 pixels balanced.")
    p.add_argument("--balance-ratio-high", type=float, default=1.10,
                   help="Upper bound for cls1/cls2 ratio. If exceeded, sampling is biased to weaker side.")
    p.add_argument("--max-tries-per-sample", type=int, default=8)
    p.add_argument("--min-change-pixels", type=int, default=64)
    p.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    p.add_argument("--compress", type=str, default="LZW")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_ids(path: Path) -> List[TileId]:
    ids = []
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        r, t = s.split("/", 1)
        ids.append(TileId(r, t))
    return ids


def read_u8(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.uint8)
        prof = ds.profile.copy()
    return arr, prof


def is_black_t2_img(path: Path) -> bool:
    with rasterio.open(path) as ds:
        for i in range(1, ds.count + 1):
            if np.any(ds.read(i) != 0):
                return False
    return True


def save_u8(path: Path, arr: np.ndarray, profile: dict, compress: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="uint8", count=1, height=arr.shape[0], width=arr.shape[1], compress=compress)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.uint8), 1)


def copy_or_link(src: Path, dst: Path, mode: str):
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


def recompute_change(t1_fake_bin: np.ndarray, t2_bin: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    chg = np.zeros_like(t2_bin, dtype=np.uint8)
    chg[(t1_fake_bin == 0) & (t2_bin == 1)] = 1  # new
    chg[(t1_fake_bin == 1) & (t2_bin == 0)] = 2  # removed
    chg[(t1_fake_bin == 1) & (t2_bin == 1)] = 3  # unchanged
    chg[ignore_mask] = 255
    return chg


def method_qdrop_new(t1_bin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = t1_bin.shape
    h2, w2 = h // 2, w // 2
    quads = [
        (slice(0, h2), slice(0, w2)),
        (slice(0, h2), slice(w2, w)),
        (slice(h2, h), slice(0, w2)),
        (slice(h2, h), slice(w2, w)),
    ]
    keep = int(rng.integers(0, 4))
    out = np.zeros_like(t1_bin, dtype=np.uint8)
    ys, xs = quads[keep]
    out[ys, xs] = (t1_bin[ys, xs] > 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return out


def _random_road_pixel(mask: np.ndarray, rng: np.random.Generator):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    k = int(rng.integers(0, ys.size))
    return int(xs[k]), int(ys[k])


def _draw_curve(canvas: np.ndarray, p0, p1, p2, thickness: int):
    pts = []
    for t in np.linspace(0.0, 1.0, 15):
        x = (1 - t) * (1 - t) * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        y = (1 - t) * (1 - t) * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        pts.append([int(round(x)), int(round(y))])
    cv2.polylines(canvas, [np.array(pts, np.int32)], False, 1, thickness=max(2, thickness), lineType=cv2.LINE_8)


def _skeletonize(bin_mask: np.ndarray) -> np.ndarray:
    img = (bin_mask > 0).astype(np.uint8)
    skel = np.zeros_like(img, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, kernel)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _skeleton_endpoints(road_bin: np.ndarray):
    sk = _skeletonize(road_bin)
    if int(sk.sum()) == 0:
        return []
    skf = sk.astype(np.float32)
    nb = cv2.boxFilter(skf, ddepth=cv2.CV_32F, ksize=(3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT) - skf
    ys, xs = np.where((sk > 0) & (np.abs(nb - 1.0) < 1e-4))
    return [(int(x), int(y), sk) for x, y in zip(xs, ys)]


def _estimate_width_px(road_bin: np.ndarray, x: int, y: int, fallback: float = 14.0) -> float:
    dist = cv2.distanceTransform((road_bin > 0).astype(np.uint8), cv2.DIST_L2, 5)
    v = float(dist[y, x]) if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1] else 0.0
    if v <= 0:
        return fallback
    return float(np.clip(2.0 * v, 6.0, 48.0))


def _endpoint_dir(sk: np.ndarray, x: int, y: int):
    h, w = sk.shape
    pts = []
    for yy in range(max(0, y - 1), min(h, y + 2)):
        for xx in range(max(0, x - 1), min(w, x + 2)):
            if xx == x and yy == y:
                continue
            if sk[yy, xx] > 0:
                pts.append((xx, yy))
    if not pts:
        ang = float(np.random.default_rng().uniform(0, 2 * np.pi))
        return float(np.cos(ang)), float(np.sin(ang))
    nx, ny = pts[0]
    vx, vy = float(x - nx), float(y - ny)
    n = float(np.hypot(vx, vy))
    if n < 1e-6:
        return 1.0, 0.0
    return vx / n, vy / n


def method_near_add_removed(t1_bin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = (t1_bin > 0).astype(np.uint8)
    add = np.zeros_like(out, dtype=np.uint8)
    n_ops = int(rng.integers(1, 4))
    h, w = out.shape
    for _ in range(n_ops):
        a = _random_road_pixel(out, rng)
        if a is None:
            break
        x0, y0 = a
        angle = float(rng.uniform(0, 2 * np.pi))
        dx, dy = np.cos(angle), np.sin(angle)
        length = float(rng.uniform(45, 170))
        thick = int(rng.integers(8, 28))
        if rng.random() < 0.55:
            x1 = int(np.clip(round(x0 + dx * length), 0, w - 1))
            y1 = int(np.clip(round(y0 + dy * length), 0, h - 1))
            cv2.line(add, (x0, y0), (x1, y1), 1, thickness=thick, lineType=cv2.LINE_8)
        else:
            bend = float(rng.uniform(-0.9, 0.9) * length)
            px, py = -dy, dx
            p1 = (x0 + dx * 0.55 * length + px * bend, y0 + dy * 0.55 * length + py * bend)
            p2 = (x0 + dx * length, y0 + dy * length)
            _draw_curve(add, (x0, y0), p1, p2, thickness=thick)
    add = cv2.morphologyEx((add > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    out[(add > 0) & (out == 0)] = 1
    return out


def method_noroad_add_removed(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    # Aggressive mode for no-road tiles:
    # draw many thick long strokes + branches to form clear synthetic roads.
    n_ops = int(rng.integers(4, 9))
    edge_pick = [
        lambda: (int(rng.integers(0, w)), 0),
        lambda: (int(rng.integers(0, w)), h - 1),
        lambda: (0, int(rng.integers(0, h))),
        lambda: (w - 1, int(rng.integers(0, h))),
        lambda: (int(rng.integers(0, w)), int(rng.integers(0, h))),
    ]
    for _ in range(n_ops):
        x0, y0 = edge_pick[int(rng.integers(0, len(edge_pick)))]()
        angle = float(rng.uniform(0, 2 * np.pi))
        dx, dy = np.cos(angle), np.sin(angle)
        length = float(rng.uniform(170, 430))
        thick = int(rng.integers(18, 52))
        if rng.random() < 0.5:
            x1 = int(np.clip(round(x0 + dx * length), 0, w - 1))
            y1 = int(np.clip(round(y0 + dy * length), 0, h - 1))
            cv2.line(out, (x0, y0), (x1, y1), 1, thickness=thick, lineType=cv2.LINE_8)
            # Branch from the middle.
            if rng.random() < 0.9:
                mx = int(round((x0 + x1) * 0.5))
                my = int(round((y0 + y1) * 0.5))
                b_ang = angle + float(rng.uniform(-1.2, 1.2))
                bx, by = np.cos(b_ang), np.sin(b_ang)
                blen = float(rng.uniform(100, 260))
                bx1 = int(np.clip(round(mx + bx * blen), 0, w - 1))
                by1 = int(np.clip(round(my + by * blen), 0, h - 1))
                bth = max(12, int(round(thick * float(rng.uniform(0.55, 0.85)))))
                cv2.line(out, (mx, my), (bx1, by1), 1, thickness=bth, lineType=cv2.LINE_8)
        else:
            bend = float(rng.uniform(-1.4, 1.4) * length)
            px, py = -dy, dx
            p1 = (x0 + dx * 0.55 * length + px * bend, y0 + dy * 0.55 * length + py * bend)
            p2 = (x0 + dx * length, y0 + dy * length)
            _draw_curve(out, (x0, y0), p1, p2, thickness=thick)
            # Curve branch.
            if rng.random() < 0.8:
                mx = int(np.clip(round((x0 + p2[0]) * 0.5), 0, w - 1))
                my = int(np.clip(round((y0 + p2[1]) * 0.5), 0, h - 1))
                b_ang = angle + float(rng.uniform(-1.3, 1.3))
                bx, by = np.cos(b_ang), np.sin(b_ang)
                blen = float(rng.uniform(90, 240))
                bbend = float(rng.uniform(-1.0, 1.0) * blen)
                bpx, bpy = -by, bx
                bp1 = (mx + bx * 0.55 * blen + bpx * bbend, my + by * 0.55 * blen + bpy * bbend)
                bp2 = (mx + bx * blen, my + by * blen)
                bth = max(12, int(round(thick * float(rng.uniform(0.5, 0.8)))))
                _draw_curve(out, (mx, my), bp1, bp2, thickness=bth)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    out = cv2.morphologyEx((out > 0).astype(np.uint8), cv2.MORPH_CLOSE, k5, iterations=2)
    out = cv2.dilate(out, k3, iterations=1)
    return out


def method_endpoint_trim_new(t1_bin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = (t1_bin > 0).astype(np.uint8)
    eps = _skeleton_endpoints(out)
    if not eps:
        return out
    n_ops = int(rng.integers(1, min(4, len(eps)) + 1))
    picks = rng.choice(len(eps), size=n_ops, replace=False)
    for pi in np.atleast_1d(picks):
        x, y, sk = eps[int(pi)]
        dx, dy = _endpoint_dir(sk, x, y)  # outward from endpoint
        # Trim inward, opposite direction.
        dx, dy = -dx, -dy
        length = float(rng.uniform(40, 150))
        wpx = _estimate_width_px(out, x, y, fallback=14.0)
        thick = int(np.clip(round(wpx * rng.uniform(0.9, 1.5)), 8, 44))
        x1 = int(np.clip(round(x + dx * length), 0, out.shape[1] - 1))
        y1 = int(np.clip(round(y + dy * length), 0, out.shape[0] - 1))
        cut = np.zeros_like(out, dtype=np.uint8)
        cv2.line(cut, (x, y), (x1, y1), 1, thickness=thick, lineType=cv2.LINE_8)
        out[(cut > 0) & (out > 0)] = 0
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return out


def method_endpoint_extend_removed(t1_bin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = (t1_bin > 0).astype(np.uint8)
    eps = _skeleton_endpoints(out)
    if not eps:
        return method_near_add_removed(out, rng)
    add = np.zeros_like(out, dtype=np.uint8)
    n_ops = int(rng.integers(1, min(4, len(eps)) + 1))
    picks = rng.choice(len(eps), size=n_ops, replace=False)
    for pi in np.atleast_1d(picks):
        x, y, sk = eps[int(pi)]
        dx, dy = _endpoint_dir(sk, x, y)  # outward
        length = float(rng.uniform(55, 190))
        wpx = _estimate_width_px(out, x, y, fallback=14.0)
        thick = int(np.clip(round(wpx * rng.uniform(0.9, 1.5)), 8, 44))
        if rng.random() < 0.5:
            x1 = int(np.clip(round(x + dx * length), 0, out.shape[1] - 1))
            y1 = int(np.clip(round(y + dy * length), 0, out.shape[0] - 1))
            cv2.line(add, (x, y), (x1, y1), 1, thickness=thick, lineType=cv2.LINE_8)
        else:
            bend = float(rng.uniform(-1.0, 1.0) * length)
            px, py = -dy, dx
            p1 = (x + dx * 0.55 * length + px * bend, y + dy * 0.55 * length + py * bend)
            p2 = (x + dx * length, y + dy * length)
            _draw_curve(add, (x, y), p1, p2, thickness=thick)
    add = cv2.morphologyEx((add > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    out[(add > 0) & (out == 0)] = 1
    return out


def method_endpoint_both_mix(t1_bin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Force both classes in one tile by combining endpoint trim and extend.
    base = method_endpoint_trim_new(t1_bin, rng)
    out = method_endpoint_extend_removed(base, rng)
    return out


def _choose_method_balanced(
    rng: np.random.Generator,
    method_names: np.ndarray,
    base_probs: np.ndarray,
    cls1_pixels: int,
    cls2_pixels: int,
    enabled: bool,
    ratio_high: float,
) -> str:
    if not enabled:
        return str(rng.choice(method_names, p=base_probs))

    # methods predominantly producing cls1(new)
    cls1_methods = {"qdrop_new", "endpoint_trim_new"}
    # methods predominantly producing cls2(removed)
    cls2_methods = {"near_add_removed", "noroad_add_removed", "endpoint_extend_removed"}
    # mixed producer
    both_methods = {"endpoint_both_mix"}

    probs = base_probs.copy()
    if cls1_pixels == 0 and cls2_pixels == 0:
        return str(rng.choice(method_names, p=probs))

    ratio = cls1_pixels / float(max(cls2_pixels, 1))
    need_cls1 = ratio < (1.0 / ratio_high)
    need_cls2 = ratio > ratio_high
    if not (need_cls1 or need_cls2):
        return str(rng.choice(method_names, p=probs))

    for i, m in enumerate(method_names):
        if m in both_methods:
            probs[i] *= 1.20
            continue
        if need_cls1:
            probs[i] *= 2.60 if m in cls1_methods else 0.45
        elif need_cls2:
            probs[i] *= 2.60 if m in cls2_methods else 0.45

    s = float(probs.sum())
    if s <= 0:
        probs = base_probs
        s = float(probs.sum())
    probs = probs / s
    return str(rng.choice(method_names, p=probs))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    split_path = args.src_split_dir / args.train_split_name
    ids = read_ids(split_path)
    if args.overwrite and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    probs = np.array([float(x.strip()) for x in args.method_probs.split(",")], dtype=np.float64)
    probs = probs / probs.sum()
    method_names = np.array([
        "qdrop_new",
        "near_add_removed",
        "noroad_add_removed",
        "endpoint_trim_new",
        "endpoint_extend_removed",
        "endpoint_both_mix",
    ])

    lines = []
    stats = {
        "num_train": len(ids),
        "num_synth_target": args.num_synth,
        "num_synth_written": 0,
        "method_counts": {k: 0 for k in method_names},
        "class_pixels": {"cls1_new": 0, "cls2_removed": 0},
    }

    # one-pass shuffled selection, no duplicate source tile
    order = np.arange(len(ids))
    rng.shuffle(order)

    for idx in order:
        if stats["num_synth_written"] >= args.num_synth:
            break
        src = ids[int(idx)]
        t2_img = args.src_root / "images" / "t2" / src.region / f"{src.tile}.tif"
        t1_p = args.src_root / "labels" / "t1" / src.region / f"{src.tile}.tif"
        t2_p = args.src_root / "labels" / "t2" / src.region / f"{src.tile}.tif"
        chg_p = args.src_root / "labels" / "change" / src.region / f"{src.tile}.tif"

        t1, t1_prof = read_u8(t1_p)
        t2, t2_prof = read_u8(t2_p)
        chg_src, chg_prof = read_u8(chg_p)
        ignore = (chg_src == 255) | (t2 == 255)
        t1_bin = (t1 > 0).astype(np.uint8)
        t2_bin = (t2 > 0).astype(np.uint8)
        has_union_road = int(((t1_bin > 0) | (t2_bin > 0)).sum()) > 0
        black_t2_img = is_black_t2_img(t2_img)

        accepted = False
        for _ in range(args.max_tries_per_sample):
            method = _choose_method_balanced(
                rng=rng,
                method_names=method_names,
                base_probs=probs,
                cls1_pixels=int(stats["class_pixels"]["cls1_new"]),
                cls2_pixels=int(stats["class_pixels"]["cls2_removed"]),
                enabled=bool(args.balance_c12),
                ratio_high=float(args.balance_ratio_high),
            )
            if method == "noroad_add_removed":
                if has_union_road or black_t2_img:
                    continue
                t1_fake = method_noroad_add_removed(t1_bin.shape, rng)
            elif method == "qdrop_new":
                t1_fake = method_qdrop_new(t1_bin, rng)
            elif method == "endpoint_trim_new":
                if int(t1_bin.sum()) == 0:
                    continue
                t1_fake = method_endpoint_trim_new(t1_bin, rng)
            elif method == "endpoint_extend_removed":
                if int(t1_bin.sum()) == 0:
                    continue
                t1_fake = method_endpoint_extend_removed(t1_bin, rng)
            elif method == "endpoint_both_mix":
                if int(t1_bin.sum()) == 0:
                    continue
                t1_fake = method_endpoint_both_mix(t1_bin, rng)
            else:
                # near_add_removed needs existing road seed
                if int(t1_bin.sum()) == 0:
                    continue
                t1_fake = method_near_add_removed(t1_bin, rng)

            chg = recompute_change(t1_fake, t2_bin, ignore)
            n_change = int(np.isin(chg, [1, 2]).sum())
            if n_change < args.min_change_pixels:
                continue

            i = stats["num_synth_written"]
            syn_tile = f"{src.tile}_syn_{i:05d}"
            lines.append(f"{src.region}/{syn_tile}")

            save_u8(args.out_root / "labels" / "t1" / src.region / f"{syn_tile}.tif", t1_fake, t1_prof, args.compress)
            save_u8(args.out_root / "labels" / "change" / src.region / f"{syn_tile}.tif", chg, chg_prof, args.compress)
            copy_or_link(t2_img, args.out_root / "images" / "t2" / src.region / f"{syn_tile}.tif", args.copy_mode)
            copy_or_link(t2_p, args.out_root / "labels" / "t2" / src.region / f"{syn_tile}.tif", args.copy_mode)

            stats["num_synth_written"] += 1
            stats["method_counts"][method] += 1
            stats["class_pixels"]["cls1_new"] += int((chg == 1).sum())
            stats["class_pixels"]["cls2_removed"] += int((chg == 2).sum())
            accepted = True
            break

        if not accepted:
            continue

    out_split_dir = args.out_root / "splits"
    out_split_dir.mkdir(parents=True, exist_ok=True)
    out_split = out_split_dir / args.out_split_name
    out_sum = out_split_dir / (Path(args.out_split_name).stem + "_summary.json")
    out_split.write_text("\n".join(lines) + ("\n" if lines else ""))
    out_sum.write_text(json.dumps(stats, indent=2))

    print(f"[done] wrote split: {out_split}")
    print(f"[done] wrote summary: {out_sum}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
