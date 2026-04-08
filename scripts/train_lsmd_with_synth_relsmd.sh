#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
WANDB_MODE="${WANDB_MODE:-disabled}" \
python tools/train.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune_with_synth_relsmd.yaml \
  "$@"
