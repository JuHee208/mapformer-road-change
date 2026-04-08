#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train_re_lsmd_imbalance_ablation.sh wce_dice_sep_hnm
# Optional env:
#   GPU=2 MAX_ITERS=32000 EVAL_INTERVAL=2000 WORKERS=4 BATCH=6

MODE="${1:-baseline}"
GPU="${GPU:-3}"
MAX_ITERS="${MAX_ITERS:-32000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
BATCH="${BATCH:-6}"
WORKERS="${WORKERS:-4}"

CFG="configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml"
WORK_ROOT="runs/cross_modal_scd/re_lsmd"
COMMON_OPTS=(
  "load_from=./runs/cross_modal_scd/hrscd/mapformer_f2/best_SCS.pth"
  "resume_from=None"
  "data.train.0.data_root=./data/re_lsmd/tiles512_posaware_b2"
  "data.train.0.synth_data_root=./data/re_lsmd/tiles512_posaware_b2_synth_final"
  "data.train.0.split_dir=./data/re_lsmd/splits_512_posaware_b2"
  "data.train.0.split=train_final_dropblack1000_withaug_synth2000_clean"
  "data.val.data_root=./data/re_lsmd/tiles512_posaware_b2"
  "data.val.split_dir=./data/re_lsmd/splits_512_posaware_b2"
  "data.val.split=val"
  "data.test.data_root=./data/re_lsmd/tiles512_posaware_b2"
  "data.test.split_dir=./data/re_lsmd/splits_512_posaware_b2"
  "data.test.split=test"
  "data.samples_per_gpu=${BATCH}"
  "data.workers_per_gpu=${WORKERS}"
  "runner.max_iters=${MAX_ITERS}"
  "evaluation.interval=${EVAL_INTERVAL}"
  "checkpoint_config.interval=500"
  "evaluation.save_best=F1_change"
  "val_vis.interval=${EVAL_INTERVAL}"
  "model.decode_head.bc_head.loss_decode.class_weight=[1.0,4.0,4.0,1.0]"
)

case "${MODE}" in
  baseline)
    WORK_DIR="${WORK_ROOT}/ablation_baseline_cw1441"
    EXP_OPTS=()
    ;;
  dice_only)
    WORK_DIR="${WORK_ROOT}/ablation_dice_only"
    EXP_OPTS=(
      "model.decode_head.bc_head.dice_loss.type=DiceLoss"
      "model.decode_head.bc_head.dice_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.ignore_index=255"
    )
    ;;
  focal_only)
    WORK_DIR="${WORK_ROOT}/ablation_focal_only"
    EXP_OPTS=(
      "model.decode_head.bc_head.focal_loss.type=FocalLoss"
      "model.decode_head.bc_head.focal_loss.use_sigmoid=True"
      "model.decode_head.bc_head.focal_loss.gamma=2.0"
      "model.decode_head.bc_head.focal_loss.alpha=0.25"
      "model.decode_head.bc_head.focal_loss.loss_weight=0.5"
    )
    ;;
  sep_only)
    WORK_DIR="${WORK_ROOT}/ablation_sep_only"
    EXP_OPTS=(
      "model.decode_head.bc_head.separable_loss_weight=0.2"
    )
    ;;
  sep_hnm)
    WORK_DIR="${WORK_ROOT}/ablation_sep_hnm"
    EXP_OPTS=(
      "model.decode_head.bc_head.separable_loss_weight=0.2"
      "model.decode_head.bc_head.hard_negative_ratio=0.3"
      "model.decode_head.bc_head.hard_negative_min_kept=4096"
    )
    ;;
  wce_sep)
    WORK_DIR="${WORK_ROOT}/ablation_wce_sep"
    EXP_OPTS=(
      "model.decode_head.bc_head.separable_loss_weight=0.2"
    )
    ;;
  wce_sep_hnm)
    WORK_DIR="${WORK_ROOT}/ablation_wce_sep_hnm"
    EXP_OPTS=(
      "model.decode_head.bc_head.separable_loss_weight=0.2"
      "model.decode_head.bc_head.hard_negative_ratio=0.3"
      "model.decode_head.bc_head.hard_negative_min_kept=4096"
    )
    ;;
  wce_dice)
    WORK_DIR="${WORK_ROOT}/ablation_wce_dice"
    EXP_OPTS=(
      "model.decode_head.bc_head.dice_loss.type=DiceLoss"
      "model.decode_head.bc_head.dice_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.ignore_index=255"
    )
    ;;
  wce_dice_sep)
    WORK_DIR="${WORK_ROOT}/ablation_wce_dice_sep"
    EXP_OPTS=(
      "model.decode_head.bc_head.dice_loss.type=DiceLoss"
      "model.decode_head.bc_head.dice_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.ignore_index=255"
      "model.decode_head.bc_head.separable_loss_weight=0.2"
    )
    ;;
  wce_dice_sep_hnm)
    WORK_DIR="${WORK_ROOT}/ablation_wce_dice_sep_hnm"
    EXP_OPTS=(
      "model.decode_head.bc_head.dice_loss.type=DiceLoss"
      "model.decode_head.bc_head.dice_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.ignore_index=255"
      "model.decode_head.bc_head.separable_loss_weight=0.2"
      "model.decode_head.bc_head.hard_negative_ratio=0.3"
      "model.decode_head.bc_head.hard_negative_min_kept=4096"
    )
    ;;
  focal_dice)
    WORK_DIR="${WORK_ROOT}/ablation_focal_dice"
    EXP_OPTS=(
      "model.decode_head.bc_head.focal_loss.type=FocalLoss"
      "model.decode_head.bc_head.focal_loss.use_sigmoid=True"
      "model.decode_head.bc_head.focal_loss.gamma=2.0"
      "model.decode_head.bc_head.focal_loss.alpha=0.25"
      "model.decode_head.bc_head.focal_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.type=DiceLoss"
      "model.decode_head.bc_head.dice_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.ignore_index=255"
    )
    ;;
  focal_dice_sep_hnm)
    WORK_DIR="${WORK_ROOT}/ablation_focal_dice_sep_hnm"
    EXP_OPTS=(
      "model.decode_head.bc_head.focal_loss.type=FocalLoss"
      "model.decode_head.bc_head.focal_loss.use_sigmoid=True"
      "model.decode_head.bc_head.focal_loss.gamma=2.0"
      "model.decode_head.bc_head.focal_loss.alpha=0.25"
      "model.decode_head.bc_head.focal_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.type=DiceLoss"
      "model.decode_head.bc_head.dice_loss.loss_weight=0.5"
      "model.decode_head.bc_head.dice_loss.ignore_index=255"
      "model.decode_head.bc_head.separable_loss_weight=0.2"
      "model.decode_head.bc_head.hard_negative_ratio=0.3"
      "model.decode_head.bc_head.hard_negative_min_kept=4096"
    )
    ;;
  focal_sep)
    WORK_DIR="${WORK_ROOT}/ablation_focal_sep"
    EXP_OPTS=(
      "model.decode_head.bc_head.focal_loss.type=FocalLoss"
      "model.decode_head.bc_head.focal_loss.use_sigmoid=True"
      "model.decode_head.bc_head.focal_loss.gamma=2.0"
      "model.decode_head.bc_head.focal_loss.alpha=0.25"
      "model.decode_head.bc_head.focal_loss.loss_weight=0.5"
      "model.decode_head.bc_head.separable_loss_weight=0.2"
    )
    ;;
  focal_sep_hnm)
    WORK_DIR="${WORK_ROOT}/ablation_focal_sep_hnm"
    EXP_OPTS=(
      "model.decode_head.bc_head.focal_loss.type=FocalLoss"
      "model.decode_head.bc_head.focal_loss.use_sigmoid=True"
      "model.decode_head.bc_head.focal_loss.gamma=2.0"
      "model.decode_head.bc_head.focal_loss.alpha=0.25"
      "model.decode_head.bc_head.focal_loss.loss_weight=0.5"
      "model.decode_head.bc_head.separable_loss_weight=0.2"
      "model.decode_head.bc_head.hard_negative_ratio=0.3"
      "model.decode_head.bc_head.hard_negative_min_kept=4096"
    )
    ;;
  *)
    echo "Unknown MODE: ${MODE}"
    echo "Supported:"
    echo "  baseline"
    echo "  dice_only | focal_only"
    echo "  sep_only | sep_hnm"
    echo "  wce_sep | wce_sep_hnm"
    echo "  wce_dice | wce_dice_sep | wce_dice_sep_hnm"
    echo "  focal_sep | focal_sep_hnm"
    echo "  focal_dice | focal_dice_sep_hnm"
    exit 1
    ;;
esac

echo "[run] MODE=${MODE} GPU=${GPU}"
echo "[run] WORK_DIR=${WORK_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" WANDB_MODE=disabled python tools/train.py \
  "${CFG}" \
  --work-dir "${WORK_DIR}" \
  --options "${COMMON_OPTS[@]}" "${EXP_OPTS[@]}"
