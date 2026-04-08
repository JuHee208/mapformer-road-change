#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python tools/convert_datasets/generate_lsmd_t1fake_synth.py \
  --src-root data/re_lsmd/tiles512_posaware_b2 \
  --src-split-dir data/re_lsmd/splits_512_posaware_b2 \
  --train-split-name train_drop_black1000.txt \
  --out-root data/re_lsmd/tiles512_posaware_b2_synth_v2_removed_heavy \
  --out-split-name train_synth_v2_removed_heavy.txt \
  --num-synth 2500 \
  --seed 42 \
  --mode-probs 0.15,0.80,0.05 \
  --min-change-pixels 64 \
  --min-target-pixels 520 \
  --min-target-ratio 0.0020 \
  --max-change-ratio 0.30 \
  --target-road-width-m 7.0 \
  --pixel-size-m 0.12 \
  --width-scale-range 1.00,1.50 \
  --thin-edit-prob 0.00 \
  --min-draw-thickness-px 10 \
  --min-thin-draw-thickness-px 4 \
  --max-synth-per-source 1 \
  --min-t2-road-pixels 0 \
  --min-road-union-pixels 0 \
  --max-no-road-source-ratio 0.35 \
  --exclude-black-t2-image \
  --add-branch-prob 0.65 \
  --add-thickness-mult 1.75 \
  --add-width-scale-range 1.00,1.70 \
  --add-branch-width-scale-range 1.00,1.60 \
  --min-add-width-px 14 \
  --add-rect-strip-prob 0.10 \
  --remove-bite-prob 0.10 \
  --curve-edit-prob 0.85 \
  --large-edit-prob 0.90 \
  --mega-edit-prob 0.25 \
  --mega-width-mult-range 2.1,3.2 \
  --mega-length-mult-range 2.1,3.8 \
  --quad-erase-prob 0.08 \
  --affine-jitter-prob 0.12 \
  --affine-max-rotate-deg 20 \
  --affine-max-shift-px 24 \
  --affine-scale-range 0.92,1.08 \
  --min-edit-component-pixels 220 \
  --min-change-component-pixels 384 \
  --copy-mode hardlink \
  "$@"

