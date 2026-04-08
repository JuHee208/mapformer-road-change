#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import rasterio

from mmseg.core.evaluation.scd_metrics import scd_eval_metrics


def read1(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.3f}"


def table_header_1() -> str:
    return "\n".join([
        "+-------+--------------+-----------+-------+-------+-------+",
        "| BC    | BC_precision | BC_recall | SC    | SCS   |  mIoU |",
        "+-------+--------------+-----------+-------+-------+-------+",
    ])


def table_header_2() -> str:
    return "\n".join([
        "per class results:",
        "+-------+------------+-------+",
        "| Class | background | road  |",
        "+-------+------------+-------+",
    ])


def table_header_3() -> str:
    return "\n".join([
        "| iter | mIoU_4 | macroF1 | IoU_12 | F1_12 | P_change | R_change | F1_change | RoadT2_IoU | RoadT2_F1 |",
        "+------+--------+---------+--------+-------+----------+----------+-----------+------------+-----------+",
    ])


def table_header_4() -> str:
    return "\n".join([
        "[Per-class Change Metrics] (0:bg,1:new,2:removed,3:unchanged):",
        "+--------+-------+-------+-------+-------+",
        "| metric | cls0  | cls1  | cls2  | cls3  |",
        "+--------+-------+-------+-------+-------+",
    ])


def table_header_5() -> str:
    return "\n".join([
        "[Binary Change View] (positive = cls1|cls2):",
        "+-------------+-----------+--------+-------+-------+",
        "| metric      | Precision | Recall | F1    | IoU   |",
        "+-------------+-----------+--------+-------+-------+",
    ])


def table_header_6() -> str:
    return "\n".join([
        "[Road Inference Metrics] (positive = cls1|cls3):",
        "+---------------------+-------+-----------+--------+-------+",
        "|        target       |  IoU  | Precision | Recall |   F1  |",
        "+---------------------+-------+-----------+--------+-------+",
    ])


def table_header_7() -> str:
    return "\n".join([
        "[sem road]",
        "+---------------------+-------+-----------+--------+-------+",
        "|         target      |  IoU  | Precision | Recall |   F1  |",
        "+---------------------+-------+-----------+--------+-------+",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="dir containing bc_pred.tif, sem_pred.tif, bc_score_change.tif")
    ap.add_argument("--gt-change", required=True)
    ap.add_argument("--gt-t2-road", required=True)
    ap.add_argument("--title", default="large_infer_eval")
    ap.add_argument("--out-log", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    bc_pred = read1(pred_dir / "bc_pred.tif").astype(np.uint8)
    sem_pred = read1(pred_dir / "sem_pred.tif").astype(np.uint8)
    score = read1(pred_dir / "bc_score_change.tif").astype(np.float32)

    gt_change = read1(Path(args.gt_change)).astype(np.uint8)
    gt_road = read1(Path(args.gt_t2_road)).astype(np.uint8)

    h = min(bc_pred.shape[0], sem_pred.shape[0], score.shape[0], gt_change.shape[0], gt_road.shape[0])
    w = min(bc_pred.shape[1], sem_pred.shape[1], score.shape[1], gt_change.shape[1], gt_road.shape[1])
    bc_pred = bc_pred[:h, :w]
    sem_pred = sem_pred[:h, :w]
    score = score[:h, :w]
    gt_change = gt_change[:h, :w]
    gt_road = gt_road[:h, :w]

    metrics = scd_eval_metrics(
        [{"bc": bc_pred, "sem": sem_pred, "bc_score_change": score}],
        [gt_change],
        [gt_road],
        num_semantic_classes=2,
        ignore_index_bc=255,
        ignore_index_sem=255,
    )

    sc_pc = metrics["SC_per_class"]
    iou_pc = metrics["IoU_per_class"]
    c_iou = metrics["Change_IoU_per_class"]
    c_pre = metrics["Change_Precision_per_class"]
    c_rec = metrics["Change_Recall_per_class"]
    c_f1 = metrics["Change_F1_per_class"]

    lines = []
    lines.append(f"2) {args.title}")
    lines.append(table_header_1())
    lines.append(
        f"| {fmt(metrics['BC'])} | {fmt(metrics['BC_precision']):>12} | {fmt(metrics['BC_recall']):>9} | "
        f"{fmt(metrics['SC'])} | {fmt(metrics['SCS'])} | {fmt(metrics['mIoU'])} |"
    )
    lines.append("+-------+--------------+-----------+-------+-------+-------+")
    lines.append("")
    lines.append(table_header_2())
    lines.append(f"|  IoU  |    {fmt(iou_pc[0])}   | {fmt(iou_pc[1])} |")
    lines.append(f"|  SC   |    {fmt(sc_pc[0])}   | {fmt(sc_pc[1])} |")
    lines.append("+-------+------------+-------+")
    lines.append("")
    lines.append(table_header_3())
    lines.append(
        f"|   -  |  {fmt(metrics['mIoU_4'])} |  {fmt(metrics['macroF1'])}  |  {fmt(metrics['IoU_12'])} | "
        f"{fmt(metrics['F1_12'])} |  {fmt(metrics['P_change'])}   |  {fmt(metrics['R_change'])}   |   "
        f"{fmt(metrics['F1_change'])}   |    {fmt(metrics['RoadT2_IoU'])}   |    {fmt(metrics['RoadT2_F1'])}  |"
    )
    lines.append("+------+--------+---------+--------+-------+----------+----------+-----------+------------+-----------+")
    lines.append("")
    lines.append(table_header_4())
    lines.append(f"| IoU    | {fmt(c_iou[0])} | {fmt(c_iou[1])} | {fmt(c_iou[2])} | {fmt(c_iou[3])} |")
    lines.append(f"| Prec   | {fmt(c_pre[0])} | {fmt(c_pre[1])} | {fmt(c_pre[2])} | {fmt(c_pre[3])} |")
    lines.append(f"| Recall | {fmt(c_rec[0])} | {fmt(c_rec[1])} | {fmt(c_rec[2])} | {fmt(c_rec[3])} |")
    lines.append(f"| F1     | {fmt(c_f1[0])} | {fmt(c_f1[1])} | {fmt(c_f1[2])} | {fmt(c_f1[3])} |")
    lines.append("+--------+-------+-------+-------+-------+")
    lines.append("")
    lines.append(table_header_5())
    lines.append(
        f"| change(1|2) |   {fmt(metrics['P_change'])}   | {fmt(metrics['R_change'])}  | "
        f"{fmt(metrics['F1_change'])} | {fmt(metrics['BC'])} |"
    )
    lines.append("+-------------+-----------+--------+-------+-------+")
    lines.append("")
    lines.append(table_header_6())
    lines.append(
        f"| road_t2 (cls1|cls3) | {fmt(metrics['RoadT2_IoU'])} |   {fmt(metrics['RoadT2_P'])}   |  "
        f"{fmt(metrics['RoadT2_R'])} | {fmt(metrics['RoadT2_F1'])} |"
    )
    lines.append("+---------------------+-------+-----------+--------+-------+")
    lines.append("")
    lines.append(table_header_7())
    lines.append(
        f"|    semantic_road    | {fmt(iou_pc[1])} |   {fmt(metrics['IoU_per_class'][1] / (metrics['IoU_per_class'][1] + 1e-9) if False else np.nan)}   |"
    )
    # sem-road precision/recall/f1 from confusion computed directly
    valid = gt_road != 255
    p = sem_pred[valid]
    g = gt_road[valid]
    tp = np.logical_and(p == 1, g == 1).sum()
    fp = np.logical_and(p == 1, g != 1).sum()
    fn = np.logical_and(p != 1, g == 1).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    lines[-1] = (
        f"|    semantic_road    | {fmt(iou_pc[1])} |   {fmt(prec)}   |  {fmt(rec)} | {fmt(f1)} |"
    )
    lines.append("+---------------------+-------+-----------+--------+-------+")

    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_log}")


if __name__ == "__main__":
    main()
