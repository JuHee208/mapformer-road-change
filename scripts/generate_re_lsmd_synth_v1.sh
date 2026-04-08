#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python tools/convert_datasets/generate_lsmd_t1fake_synth.py \
  --src-root data/re_lsmd/tiles512_posaware_b2 \
  --src-split-dir data/re_lsmd/splits_512_posaware_b2 \
  --train-split-name train.txt \
  --out-root data/re_lsmd/tiles512_posaware_b2_synth_v1 \
  --out-split-name train_synth_v1.txt \
  --synth-ratio 0.30 \
  --seed 42 \
  --mode-probs 0.5,0.5,0.0 \
  --min-change-pixels 32 \
  --min-target-pixels 64 \
  --min-target-ratio 0.0005 \
  --max-change-ratio 0.20 \
  --target-road-width-m 7.0 \
  --pixel-size-m 0.12 \
  --width-scale-range 0.8,1.2 \
  --thin-edit-prob 0.00 \
  --thin-width-scale-range 0.15,0.45 \
  --min-draw-thickness-px 6 \
  --min-thin-draw-thickness-px 4 \
  --max-synth-per-source 1 \
  --min-t2-road-pixels 1 \
  --min-road-union-pixels 1 \
  --add-branch-prob 0.50 \
  --add-thickness-mult 1.35 \
  --add-width-scale-range 0.65,1.25 \
  --add-branch-width-scale-range 0.65,1.25 \
  --min-add-width-px 10 \
  --add-rect-strip-prob 0.08 \
  --remove-bite-prob 0.10 \
  --curve-edit-prob 0.75 \
  --large-edit-prob 0.80 \
  --min-edit-component-pixels 160 \
  --min-change-component-pixels 260 \
  --copy-mode hardlink \
  "$@"
