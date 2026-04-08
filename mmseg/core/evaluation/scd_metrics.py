import numpy as np
from collections import OrderedDict
try:
    from sklearn.metrics import average_precision_score
except Exception:  # pragma: no cover
    average_precision_score = None


def _safe_div(num, den, eps=1e-6):
    return num / (den + eps)


def _binary_stats(pred_pos, gt_pos):
    tp = np.logical_and(pred_pos, gt_pos).sum(dtype=np.float64)
    fp = np.logical_and(pred_pos, ~gt_pos).sum(dtype=np.float64)
    fn = np.logical_and(~pred_pos, gt_pos).sum(dtype=np.float64)
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return iou, precision, recall, f1


def _confusion(pred, gt, n_classes):
    valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)
    gt_v = gt[valid].astype(np.int64)
    pred_v = pred[valid].astype(np.int64)
    idx = gt_v * n_classes + pred_v
    hist = np.bincount(idx, minlength=n_classes * n_classes)
    return hist.reshape(n_classes, n_classes).astype(np.float64)


def scd_eval_metrics(
        results,
        gt_bc_maps,
        gt_sem_maps,
        num_semantic_classes,
        ignore_index_bc,
        ignore_index_sem):
    assert len(results) == len(gt_bc_maps) == len(gt_sem_maps)

    total_bc_tp = 0.0
    total_bc_fp = 0.0
    total_bc_fn = 0.0

    total_sc_intersect = np.zeros((num_semantic_classes,), dtype=np.float64)
    total_sc_union = np.zeros((num_semantic_classes,), dtype=np.float64)
    total_sem_intersect = np.zeros((num_semantic_classes,), dtype=np.float64)
    total_sem_union = np.zeros((num_semantic_classes,), dtype=np.float64)

    # For 4-class change metrics (LSMD style)
    n_change_classes = 4
    total_change_conf = np.zeros((n_change_classes, n_change_classes), dtype=np.float64)
    all_change_scores = []
    all_change_labels = []

    has_multiclass_change = False

    for i in range(len(results)):
        pred_bc = results[i]['bc']
        pred_sem = results[i]['sem']
        gt_bc = gt_bc_maps[i]
        gt_sem = gt_sem_maps[i]

        mask_bc = (gt_bc != ignore_index_bc)
        if mask_bc.sum() == 0:
            continue

        pred_bc_masked = pred_bc[mask_bc]
        gt_bc_masked = gt_bc[mask_bc]
        pred_sem_masked = pred_sem[mask_bc]
        gt_sem_masked = gt_sem[mask_bc]

        # Binary change view (positive = class 1|2 for 4-class change, else class 1)
        if np.nanmax(gt_bc_masked) > 1:
            has_multiclass_change = True
            gt_change_pos = np.isin(gt_bc_masked, [1, 2])
            pred_change_pos = np.isin(pred_bc_masked, [1, 2])
        else:
            gt_change_pos = (gt_bc_masked == 1)
            pred_change_pos = (pred_bc_masked == 1)

        # Collect probability scores for AUPRC when available.
        score_map = results[i].get('bc_score_change', None)
        if score_map is not None:
            score_masked = score_map[mask_bc]
            all_change_scores.append(score_masked.reshape(-1))
            all_change_labels.append(gt_change_pos.astype(np.uint8).reshape(-1))

        tp = np.logical_and(pred_change_pos, gt_change_pos).sum(dtype=np.float64)
        fp = np.logical_and(pred_change_pos, ~gt_change_pos).sum(dtype=np.float64)
        fn = np.logical_and(~pred_change_pos, gt_change_pos).sum(dtype=np.float64)
        total_bc_tp += tp
        total_bc_fp += fp
        total_bc_fn += fn

        # SC on GT-changed pixels (binary change view)
        change_mask = gt_change_pos
        if change_mask.any():
            pred_sc = pred_sem_masked[change_mask]
            gt_sc = gt_sem_masked[change_mask]
            valid_sc = (gt_sc != ignore_index_sem)
            pred_sc = pred_sc[valid_sc]
            gt_sc = gt_sc[valid_sc]
            if gt_sc.size > 0:
                intersect_sc = pred_sc[pred_sc == gt_sc]
                intersect_area_sc = np.histogram(
                    intersect_sc, bins=num_semantic_classes,
                    range=(-0.5, num_semantic_classes - 0.5))[0]
                pred_area_sc = np.histogram(
                    pred_sc, bins=num_semantic_classes,
                    range=(-0.5, num_semantic_classes - 0.5))[0]
                gt_area_sc = np.histogram(
                    gt_sc, bins=num_semantic_classes,
                    range=(-0.5, num_semantic_classes - 0.5))[0]
                union_area_sc = pred_area_sc + gt_area_sc - intersect_area_sc
                total_sc_intersect += intersect_area_sc
                total_sc_union += union_area_sc

        # Semantic mIoU over valid BC mask
        valid_sem = (gt_sem_masked != ignore_index_sem)
        pred_sem_valid = pred_sem_masked[valid_sem]
        gt_sem_valid = gt_sem_masked[valid_sem]
        if gt_sem_valid.size > 0:
            intersect_sem = pred_sem_valid[pred_sem_valid == gt_sem_valid]
            intersect_area_sem = np.histogram(
                intersect_sem, bins=num_semantic_classes,
                range=(-0.5, num_semantic_classes - 0.5))[0]
            pred_area_sem = np.histogram(
                pred_sem_valid, bins=num_semantic_classes,
                range=(-0.5, num_semantic_classes - 0.5))[0]
            gt_area_sem = np.histogram(
                gt_sem_valid, bins=num_semantic_classes,
                range=(-0.5, num_semantic_classes - 0.5))[0]
            union_area_sem = pred_area_sem + gt_area_sem - intersect_area_sem
            total_sem_intersect += intersect_area_sem
            total_sem_union += union_area_sem

        # 4-class change confusion
        if np.nanmax(gt_bc_masked) > 1:
            total_change_conf += _confusion(
                pred=pred_bc_masked,
                gt=gt_bc_masked,
                n_classes=n_change_classes)

    eps = 1e-6
    ret_metrics = OrderedDict()

    # Existing metrics (kept for compatibility)
    bc_iou = _safe_div(total_bc_tp, total_bc_tp + total_bc_fp + total_bc_fn, eps=eps)
    bc_precision = _safe_div(total_bc_tp, total_bc_tp + total_bc_fp, eps=eps)
    bc_recall = _safe_div(total_bc_tp, total_bc_tp + total_bc_fn, eps=eps)

    sc_per_class = _safe_div(total_sc_intersect, total_sc_union, eps=eps)
    sem_iou_per_class = _safe_div(total_sem_intersect, total_sem_union, eps=eps)
    ret_metrics['BC'] = float(bc_iou)
    ret_metrics['SC'] = float(np.nanmean(sc_per_class))
    ret_metrics['mIoU'] = float(np.nanmean(sem_iou_per_class))
    ret_metrics['BC_recall'] = float(bc_recall)
    ret_metrics['BC_precision'] = float(bc_precision)
    ret_metrics['SCS'] = float(0.5 * (ret_metrics['BC'] + ret_metrics['SC']))
    ret_metrics['SC_per_class'] = sc_per_class
    ret_metrics['IoU_per_class'] = sem_iou_per_class

    # Extended metrics for 4-class change setting
    if has_multiclass_change:
        tp_c = np.diag(total_change_conf)
        pred_c = total_change_conf.sum(axis=0)
        gt_c = total_change_conf.sum(axis=1)
        union_c = pred_c + gt_c - tp_c

        iou_c = _safe_div(tp_c, union_c, eps=eps)
        prec_c = _safe_div(tp_c, pred_c, eps=eps)
        rec_c = _safe_div(tp_c, gt_c, eps=eps)
        f1_c = _safe_div(2 * prec_c * rec_c, prec_c + rec_c, eps=eps)

        ret_metrics['Change_IoU_per_class'] = iou_c
        ret_metrics['Change_Precision_per_class'] = prec_c
        ret_metrics['Change_Recall_per_class'] = rec_c
        ret_metrics['Change_F1_per_class'] = f1_c

        ret_metrics['mIoU_4'] = float(np.nanmean(iou_c))
        ret_metrics['macroF1'] = float(np.nanmean(f1_c))
        ret_metrics['IoU_12'] = float(np.nanmean(iou_c[[1, 2]]))
        ret_metrics['F1_12'] = float(np.nanmean(f1_c[[1, 2]]))
        # Backward-compatible alias for existing reporting names.
        ret_metrics['IoU_23'] = ret_metrics['IoU_12']
        ret_metrics['F1_23'] = ret_metrics['F1_12']

        # Binary change view (positive = cls1|cls2)
        pred_change = pred_c[1] + pred_c[2]
        gt_change = gt_c[1] + gt_c[2]
        tp_change = total_change_conf[1, 1] + total_change_conf[1, 2] + total_change_conf[2, 1] + total_change_conf[2, 2]
        fp_change = max(pred_change - tp_change, 0.0)
        fn_change = max(gt_change - tp_change, 0.0)
        p_change = _safe_div(tp_change, tp_change + fp_change, eps=eps)
        r_change = _safe_div(tp_change, tp_change + fn_change, eps=eps)
        f1_change = _safe_div(2 * p_change * r_change, p_change + r_change, eps=eps)
        ret_metrics['P_change'] = float(p_change)
        ret_metrics['R_change'] = float(r_change)
        ret_metrics['F1_change'] = float(f1_change)
        if average_precision_score is not None and all_change_scores:
            try:
                y_score = np.concatenate(all_change_scores)
                y_true = np.concatenate(all_change_labels)
                if np.unique(y_true).size >= 2:
                    ret_metrics['AUPRC_change'] = float(average_precision_score(y_true, y_score))
                else:
                    ret_metrics['AUPRC_change'] = float('nan')
            except Exception:
                ret_metrics['AUPRC_change'] = float('nan')
        else:
            ret_metrics['AUPRC_change'] = float('nan')

        # Road inference view (target = cls1|cls3 in T2)
        pred_road = pred_c[1] + pred_c[3]
        gt_road = gt_c[1] + gt_c[3]
        tp_road = (total_change_conf[1, 1] + total_change_conf[1, 3] +
                   total_change_conf[3, 1] + total_change_conf[3, 3])
        fp_road = max(pred_road - tp_road, 0.0)
        fn_road = max(gt_road - tp_road, 0.0)
        iou_road = _safe_div(tp_road, tp_road + fp_road + fn_road, eps=eps)
        p_road = _safe_div(tp_road, tp_road + fp_road, eps=eps)
        r_road = _safe_div(tp_road, tp_road + fn_road, eps=eps)
        f1_road = _safe_div(2 * p_road * r_road, p_road + r_road, eps=eps)
        ret_metrics['RoadT2_IoU'] = float(iou_road)
        ret_metrics['RoadT2_P'] = float(p_road)
        ret_metrics['RoadT2_R'] = float(r_road)
        ret_metrics['RoadT2_F1'] = float(f1_road)

    return ret_metrics
