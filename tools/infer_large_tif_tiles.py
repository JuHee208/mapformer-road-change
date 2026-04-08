#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from rasterio.windows import Window

# Allow running as: python tools/infer_large_tif_tiles.py ...
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def init_segmentor_robust(config, checkpoint, device):
    if isinstance(config, str):
        cfg = mmcv.Config.fromfile(config)
    else:
        cfg = config
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    if checkpoint:
        ckpt = load_checkpoint(model, checkpoint, map_location="cpu")
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        if isinstance(meta, dict):
            if "CLASSES" in meta:
                model.CLASSES = meta["CLASSES"]
            if "PALETTE" in meta:
                model.PALETTE = meta["PALETTE"]
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer a large TIF by tiling into fixed-size patches (default: 512x512)."
    )
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint path")
    parser.add_argument("--t2-image", required=True, help="Large T2 image TIF path")
    parser.add_argument(
        "--t1-map",
        required=True,
        help="Large T1 map TIF path (semantic pre-map; 0/1 expected)",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding stride. If omitted, uses tile-size (no overlap).",
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    parser.add_argument(
        "--memmap-dir",
        default=None,
        help="Optional directory for disk-backed accumulators. "
             "If omitted, memmap is enabled automatically for large rasters.",
    )
    parser.add_argument(
        "--memmap-threshold-gb",
        type=float,
        default=8.0,
        help="Auto-enable memmap when estimated accumulator size exceeds this value.",
    )
    parser.add_argument(
        "--write-chunk-rows",
        type=int,
        default=512,
        help="Number of rows to write per chunk when materializing final TIFFs.",
    )
    parser.add_argument(
        "--keep-memmap",
        action="store_true",
        help="Keep temporary memmap files after inference.",
    )
    parser.add_argument(
        "--gt-change",
        default=None,
        help="Optional GT change label TIF path (0/1/2/3[/255]) for evaluation",
    )
    parser.add_argument(
        "--gt-t2-road",
        default=None,
        help="Optional GT t2 road mask TIF path (0/1) for semantic/road evaluation",
    )
    parser.add_argument(
        "--eval-json",
        default=None,
        help="Optional output JSON path for evaluation summary",
    )
    return parser.parse_args()


def normalize_img(img_hwc, mean, std, to_rgb):
    img = img_hwc.astype(np.float32)
    if to_rgb and img.shape[2] == 3:
        # mmcv Normalize(to_rgb=True): convert BGR -> RGB
        img = img[..., ::-1]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    return (img - mean) / std


def load_t2(path):
    with rasterio.open(path) as ds:
        arr = ds.read()  # C,H,W
        profile = ds.profile.copy()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image (C,H,W), got shape={arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError(f"Expected >=3 channels, got {arr.shape[0]}")
    arr = arr[:3]  # keep first 3 channels
    return np.transpose(arr, (1, 2, 0)), profile


def load_t1_map(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return arr.astype(np.uint8)


def _safe_metrics(tp, fp, fn, tn):
    eps = 1e-9
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return dict(
        TP=int(tp),
        FP=int(fp),
        FN=int(fn),
        TN=int(tn),
        Precision=float(p),
        Recall=float(r),
        IoU=float(iou),
        F1=float(f1),
        Accuracy=float(acc),
    )


def _eval_change(pred, gt):
    valid = gt != 255
    pred_v = pred[valid]
    gt_v = gt[valid]

    per_class = {}
    for c in [0, 1, 2, 3]:
        tp = np.logical_and(pred_v == c, gt_v == c).sum()
        fp = np.logical_and(pred_v == c, gt_v != c).sum()
        fn = np.logical_and(pred_v != c, gt_v == c).sum()
        tn = np.logical_and(pred_v != c, gt_v != c).sum()
        per_class[str(c)] = _safe_metrics(tp, fp, fn, tn)

    pred_pos = np.logical_or(pred_v == 1, pred_v == 2)
    gt_pos = np.logical_or(gt_v == 1, gt_v == 2)
    tp = np.logical_and(pred_pos, gt_pos).sum()
    fp = np.logical_and(pred_pos, np.logical_not(gt_pos)).sum()
    fn = np.logical_and(np.logical_not(pred_pos), gt_pos).sum()
    tn = np.logical_and(np.logical_not(pred_pos), np.logical_not(gt_pos)).sum()
    binary_change = _safe_metrics(tp, fp, fn, tn)
    return per_class, binary_change


def _eval_sem_road(sem_pred, gt_road):
    valid = gt_road != 255
    pred_v = sem_pred[valid]
    gt_v = gt_road[valid]
    out = {}
    ious = []
    for c in [0, 1]:
        tp = np.logical_and(pred_v == c, gt_v == c).sum()
        fp = np.logical_and(pred_v == c, gt_v != c).sum()
        fn = np.logical_and(pred_v != c, gt_v == c).sum()
        tn = np.logical_and(pred_v != c, gt_v != c).sum()
        m = _safe_metrics(tp, fp, fn, tn)
        out[str(c)] = m
        ious.append(m["IoU"])
    out["mIoU_2"] = float(np.mean(ious))
    return out


def _max_overlap(tile_size, stride):
    return int(math.ceil(tile_size / float(stride)) ** 2)


def _pick_count_dtype(max_count):
    if max_count <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_count <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def _estimate_accumulator_bytes(h, w, vote_dtype, count_dtype, score_dtype=np.float32):
    return (
        4 * h * w * np.dtype(vote_dtype).itemsize +   # bc votes
        2 * h * w * np.dtype(vote_dtype).itemsize +   # sem votes
        h * w * np.dtype(score_dtype).itemsize +      # score sum
        h * w * np.dtype(count_dtype).itemsize        # score count
    )


def _allocate_array(shape, dtype, path=None):
    if path is None:
        return np.zeros(shape, dtype=dtype)
    arr = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[...] = 0
    arr.flush()
    return arr


def _sync_if_cuda(device):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def main():
    args = parse_args()
    total_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = init_segmentor_robust(args.config, args.checkpoint, device=args.device)
    cfg = model.cfg
    norm_cfg = None
    for t in cfg.data.test.pipeline:
        if t.get("type") == "MultiScaleFlipAug":
            for tt in t.get("transforms", []):
                if tt.get("type") == "Normalize":
                    norm_cfg = tt
                    break
    if norm_cfg is None:
        raise RuntimeError("Normalize config not found in test pipeline.")

    img_hwc, profile = load_t2(args.t2_image)
    t1_map = load_t1_map(args.t1_map)
    if img_hwc.shape[:2] != t1_map.shape[:2]:
        raise ValueError(
            f"Shape mismatch: t2={img_hwc.shape[:2]} vs t1_map={t1_map.shape[:2]}"
        )

    h, w = t1_map.shape[:2]
    tile = args.tile_size
    stride = args.stride if args.stride is not None else tile
    if stride <= 0 or stride > tile:
        raise ValueError(f"Invalid stride={stride}. Must satisfy 1 <= stride <= tile_size({tile}).")

    max_votes = _max_overlap(tile, stride)
    vote_dtype = _pick_count_dtype(max_votes)
    count_dtype = _pick_count_dtype(max_votes)
    estimated_bytes = _estimate_accumulator_bytes(h, w, vote_dtype, count_dtype)
    use_memmap = (
        args.memmap_dir is not None or
        estimated_bytes > args.memmap_threshold_gb * (1024 ** 3)
    )
    memmap_dir = None
    if use_memmap:
        memmap_dir = Path(args.memmap_dir) if args.memmap_dir else (out_dir / "_memmap")
        if memmap_dir.exists():
            shutil.rmtree(memmap_dir)
        memmap_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[info] using memmap accumulators at {memmap_dir} "
            f"(estimated core accumulator size: {estimated_bytes / (1024 ** 3):.2f} GB, "
            f"vote dtype: {np.dtype(vote_dtype).name})"
        )

    # Overlap-aware accumulators
    # bc classes: 0,1,2,3 ; sem classes: 0,1
    bc_votes = _allocate_array(
        (4, h, w),
        vote_dtype,
        None if memmap_dir is None else (memmap_dir / "bc_votes.dat"),
    )
    sem_votes = _allocate_array(
        (2, h, w),
        vote_dtype,
        None if memmap_dir is None else (memmap_dir / "sem_votes.dat"),
    )
    score_sum = _allocate_array(
        (h, w),
        np.float32,
        None if memmap_dir is None else (memmap_dir / "score_sum.dat"),
    )
    score_cnt = _allocate_array(
        (h, w),
        count_dtype,
        None if memmap_dir is None else (memmap_dir / "score_cnt.dat"),
    )

    tile_count = 0
    infer_start = time.perf_counter()
    forward_elapsed = 0.0
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                tile_count += 1
                y2 = min(y + tile, h)
                x2 = min(x + tile, w)
                ph = y2 - y
                pw = x2 - x

                img_patch = np.zeros((tile, tile, 3), dtype=img_hwc.dtype)
                map_patch = np.zeros((tile, tile), dtype=np.uint8)
                img_patch[:ph, :pw] = img_hwc[y:y2, x:x2]
                map_patch[:ph, :pw] = t1_map[y:y2, x:x2]

                img_patch = normalize_img(
                    img_patch,
                    mean=norm_cfg["mean"],
                    std=norm_cfg["std"],
                    to_rgb=norm_cfg.get("to_rgb", True),
                )

                img_tensor = (
                    torch.from_numpy(np.transpose(img_patch, (2, 0, 1)))
                    .unsqueeze(0)
                    .to(args.device)
                )
                map_tensor = torch.from_numpy(map_patch).unsqueeze(0).to(args.device)

                img_meta = dict(
                    filename=None,
                    ori_filename=None,
                    ori_shape=(tile, tile, 3),
                    img_shape=(tile, tile, 3),
                    pad_shape=(tile, tile, 3),
                    scale_factor=1.0,
                    flip=False,
                    flip_direction=None,
                    img_norm_cfg=norm_cfg,
                )

                _sync_if_cuda(args.device)
                forward_start = time.perf_counter()
                pred = model.simple_test(
                    img=img_tensor,
                    img_metas=[img_meta],
                    gt_semantic_seg_pre=map_tensor,
                    rescale=True,
                )
                _sync_if_cuda(args.device)
                forward_elapsed += time.perf_counter() - forward_start

                bc_patch = pred["bc"][:ph, :pw]
                sem_patch = pred["sem"][:ph, :pw]
                score_patch = pred["bc_score_change"][:ph, :pw].astype(np.float32)

                for c in range(4):
                    bc_votes[c, y:y2, x:x2] += (bc_patch == c).astype(vote_dtype)
                for c in range(2):
                    sem_votes[c, y:y2, x:x2] += (sem_patch == c).astype(vote_dtype)

                score_sum[y:y2, x:x2] += score_patch
                score_cnt[y:y2, x:x2] += 1
    infer_elapsed = time.perf_counter() - infer_start

    pred_profile = profile.copy()
    pred_profile.update(count=1, dtype=rasterio.uint8, compress="lzw")
    score_profile = profile.copy()
    score_profile.update(count=1, dtype=rasterio.float32, compress="lzw")
    bc_path = out_dir / "bc_pred.tif"
    sem_path = out_dir / "sem_pred.tif"
    score_path = out_dir / "bc_score_change.tif"
    with rasterio.open(bc_path, "w", **pred_profile) as ds_bc, \
            rasterio.open(sem_path, "w", **pred_profile) as ds_sem, \
            rasterio.open(score_path, "w", **score_profile) as ds_score:
        for row_start in range(0, h, args.write_chunk_rows):
            row_end = min(row_start + args.write_chunk_rows, h)
            window = Window(0, row_start, w, row_end - row_start)
            bc_chunk = np.argmax(
                np.asarray(bc_votes[:, row_start:row_end, :]),
                axis=0).astype(np.uint8)
            sem_chunk = np.argmax(
                np.asarray(sem_votes[:, row_start:row_end, :]),
                axis=0).astype(np.uint8)
            score_cnt_chunk = np.asarray(
                score_cnt[row_start:row_end, :],
                dtype=np.float32)
            score_chunk = (
                np.asarray(score_sum[row_start:row_end, :], dtype=np.float32)
                / np.maximum(score_cnt_chunk, 1.0)
            ).astype(np.float32)
            ds_bc.write(bc_chunk, 1, window=window)
            ds_sem.write(sem_chunk, 1, window=window)
            ds_score.write(score_chunk, 1, window=window)

    del bc_votes, sem_votes, score_sum, score_cnt
    gc.collect()

    print(f"[done] saved: {bc_path}")
    print(f"[done] saved: {sem_path}")
    print(f"[done] saved: {score_path}")

    summary = {}
    out_bc = None
    out_sem = None
    if args.gt_change is not None:
        with rasterio.open(bc_path) as ds:
            out_bc = ds.read(1).astype(np.uint8)
        with rasterio.open(args.gt_change) as ds:
            gt_change = ds.read(1).astype(np.uint8)
        if gt_change.shape != out_bc.shape:
            raise ValueError(
                f"GT change shape mismatch: pred={out_bc.shape}, gt={gt_change.shape}"
            )
        per_class, binary_change = _eval_change(out_bc, gt_change)
        summary["change_per_class"] = per_class
        summary["binary_change"] = binary_change
        print(
            "[eval][binary_change] "
            f"P={binary_change['Precision']:.4f} "
            f"R={binary_change['Recall']:.4f} "
            f"F1={binary_change['F1']:.4f} "
            f"IoU={binary_change['IoU']:.4f}"
        )

    if args.gt_t2_road is not None:
        if out_sem is None:
            with rasterio.open(sem_path) as ds:
                out_sem = ds.read(1).astype(np.uint8)
        with rasterio.open(args.gt_t2_road) as ds:
            gt_road = ds.read(1).astype(np.uint8)
        if gt_road.shape != out_sem.shape:
            raise ValueError(
                f"GT road shape mismatch: pred={out_sem.shape}, gt={gt_road.shape}"
            )
        sem_eval = _eval_sem_road(out_sem, gt_road)
        summary["sem_road"] = sem_eval
        print(
            "[eval][sem_road] "
            f"mIoU_2={sem_eval['mIoU_2']:.4f} "
            f"road_F1={sem_eval['1']['F1']:.4f} "
            f"road_IoU={sem_eval['1']['IoU']:.4f}"
        )

    if summary:
        out_json = Path(args.eval_json) if args.eval_json else (out_dir / "eval_summary.json")
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"[done] saved: {out_json}")

    total_elapsed = time.perf_counter() - total_start
    runtime_info = {
        "height": int(h),
        "width": int(w),
        "tile_size": int(tile),
        "stride": int(stride),
        "overlap": int(tile - stride),
        "num_tiles": int(tile_count),
        "infer_sec": float(infer_elapsed),
        "forward_only_sec": float(forward_elapsed),
        "total_sec": float(total_elapsed),
        "tiles_per_sec": float(tile_count / max(infer_elapsed, 1e-9)),
        "forward_tiles_per_sec": float(tile_count / max(forward_elapsed, 1e-9)),
        "image_mpix": float((h * w) / 1e6),
        "image_mpix_per_sec": float(((h * w) / 1e6) / max(infer_elapsed, 1e-9)),
        "forward_image_mpix_per_sec": float(((h * w) / 1e6) / max(forward_elapsed, 1e-9)),
    }
    runtime_path = out_dir / "infer_runtime.json"
    runtime_path.write_text(json.dumps(runtime_info, indent=2))
    print(
        "[timing] "
        f"forward_only_sec={runtime_info['forward_only_sec']:.3f} "
        f"infer_sec={runtime_info['infer_sec']:.3f} "
        f"total_sec={runtime_info['total_sec']:.3f} "
        f"num_tiles={runtime_info['num_tiles']} "
        f"tiles_per_sec={runtime_info['tiles_per_sec']:.3f}"
    )
    print(f"[done] saved: {runtime_path}")

    if memmap_dir is not None and not args.keep_memmap:
        shutil.rmtree(memmap_dir, ignore_errors=True)
        print(f"[done] removed temporary memmap dir: {memmap_dir}")


if __name__ == "__main__":
    main()
