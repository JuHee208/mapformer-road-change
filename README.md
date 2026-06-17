# MapFormer for Road Change Detection

This repository extends MapFormer toward road change detection on LSMD-style data. It keeps the original HRSCD and DynamicEarthNet code paths, and adds LSMD dataset support, LSMD finetuning configs, synthetic-data workflows, and large-area tiled inference utilities.

The codebase is built on top of the original MapFormer implementation from the ICCV 2023 paper "MapFormer: Boosting Change Detection by Using Semantic Pre-change Information", plus MMSegmentation- and Open-CD-style components.

## Scope

The main research focus in this repository is road change detection with semantic pre-map input.

Included:

- LSMD dataset loader and cross-modal SCD configs
- LSMD preprocessing and tiling scripts
- Synthetic LSMD training-data generation scripts
- Large GeoTIFF inference utilities
- Original HRSCD and DynamicEarthNet experiments

Not included:

- Datasets
- Pretrained checkpoints
- Training runs, previews, or local experiment artifacts

## Environment

- Python 3.9+
- See [requirements.txt](requirements.txt)

Install with:

```bash
pip install -r requirements.txt
```

## Main LSMD Pipeline

The main LSMD training config is:

- [configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml](configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml)

The dataset loader is:

- [mmseg/datasets/ccd/lsmd.py](mmseg/datasets/ccd/lsmd.py)

The default training launcher is:

- [scripts/train_lsmd_default.sh](scripts/train_lsmd_default.sh)

## Repository Layout

Top-level directories you will use most often:

- [configs](configs): experiment configs grouped by task and dataset
- [mmseg](mmseg): dataset definitions, model code, training hooks, and evaluation logic
- [opencd](opencd): change-detection framework components used by this codebase
- [tools](tools): Python entrypoints for training, testing, preprocessing, and large-area inference
- [scripts](scripts): shell launchers for common experiments and ablations
- [data](data): local datasets, generated splits, and tiled training data; not included in this repo
- [model_ckpt](model_ckpt): pretrained backbones and saved checkpoints; not included in this repo
- [runs](runs): training logs, checkpoints, validation outputs, and offline wandb logs; not included in this repo
- [outputs](outputs): large-area inference outputs, mosaics, and export files
- `tmp_*`: temporary previews, debugging outputs, and one-off analysis folders

For the LSMD road-change pipeline in particular:

- [configs/cross_modal_scd/lsmd](configs/cross_modal_scd/lsmd): LSMD experiment configs
- [mmseg/datasets/ccd/lsmd.py](mmseg/datasets/ccd/lsmd.py): LSMD dataset loader
- [tools/convert_datasets](tools/convert_datasets): LSMD reprojection, split generation, tiling, and synthesis utilities
- [tools/infer_large_tif_tiles.py](tools/infer_large_tif_tiles.py): tiled inference on large GeoTIFF scenes

## LSMD Data Layout

### Raw region layout

Each region directory is expected to contain:

- one T2 image
- two road ground-truth rasters, one for T1 and one for T2
- one change-label raster

The preprocessing scripts identify files by name patterns such as `road_gt` and `change`.

Example:

```text
data/lsmd_5186_final/
  anyang/
    anyang_2024.tif
    road_gt_anyang_2023.tif
    road_gt_anyang_2024.tif
    road_change_label.tif
  gangnam/
    ...
  jungnang/
    ...
```

### Tiled training layout

The LSMD dataset class expects the tiled dataset layout below:

```text
data/lsmd_5186_final/tiles512_posaware_b2/
  images/
    t2/
      <region>/
        r000_c000.tif
        ...
  labels/
    t1/
      <region>/
        r000_c000.tif
        ...
    t2/
      <region>/
        r000_c000.tif
        ...
    change/
      <region>/
        r000_c000.tif
        ...
  splits/
    train.txt
    val.txt
    test.txt
```

Each split entry is formatted as:

```text
<region>/rXXX_cYYY
```

## Data Preparation

### 1. Optional reprojection to EPSG:5186

If your LSMD rasters are not already aligned on the target grid, use:

- [tools/convert_datasets/reproject_lsmd_to_5186.py](tools/convert_datasets/reproject_lsmd_to_5186.py)

Example:

```bash
python tools/convert_datasets/reproject_lsmd_to_5186.py \
  --src-root data/lsmd \
  --dst-root data/lsmd_5186 \
  --copy-splits
```

### 2. Build spatial splits

For block-buffer or positive-aware LSMD splits, use:

- [tools/convert_datasets/make_lsmd_posaware_splits.py](tools/convert_datasets/make_lsmd_posaware_splits.py)
- [tools/convert_datasets/make_lsmd_bottom_splits.py](tools/convert_datasets/make_lsmd_bottom_splits.py)
- [tools/convert_datasets/make_block_buffer_splits.py](tools/convert_datasets/make_block_buffer_splits.py)

Example:

```bash
python tools/convert_datasets/make_lsmd_posaware_splits.py \
  --data-root data/lsmd_5186_final \
  --out-dir data/lsmd_5186_final/splits_512_posaware_b2 \
  --trainval-regions anyang gangnam \
  --test-regions jungnang
```

### 3. Create 512 tiles

Use:

- [tools/convert_datasets/create_lsmd_tiles_from_splits.py](tools/convert_datasets/create_lsmd_tiles_from_splits.py)

Example:

```bash
python tools/convert_datasets/create_lsmd_tiles_from_splits.py \
  --data-root data/lsmd_5186_final \
  --split-dir data/lsmd_5186_final/splits_512_posaware_b2 \
  --out-dir data/lsmd_5186_final/tiles512_posaware_b2 \
  --tile-size 512
```

### 4. Optional synthetic training data

Synthetic-data workflows and launchers are included:

- [tools/convert_datasets/generate_lsmd_t1fake_synth.py](tools/convert_datasets/generate_lsmd_t1fake_synth.py)
- [scripts/generate_lsmd_synth_v1.sh](scripts/generate_lsmd_synth_v1.sh)
- [scripts/generate_re_lsmd_synth_v1.sh](scripts/generate_re_lsmd_synth_v1.sh)
- [scripts/generate_re_lsmd_synth_v2.sh](scripts/generate_re_lsmd_synth_v2.sh)

## Pretrained Backbone

The LSMD configs expect:

- `./model_ckpt/mit_b2.pth`

Several non-LSMD configs also use:

- `./model_ckpt/mit_b2_20220624-66e8bf70.pth`

## Training

The default launcher is:

```bash
bash scripts/train_lsmd_default.sh
```

That script is equivalent to:

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled \
python tools/train.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml
```

For a reproducible run with an explicit output directory:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
python tools/train.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml \
  --work-dir runs/cross_modal_scd/lsmd/mapformer_t2map_finetune
```

If you want to override dataset paths or dataloader settings without editing the YAML:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
python tools/train.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml \
  --work-dir runs/cross_modal_scd/lsmd/mapformer_t2map_finetune \
  --options \
    data.train.0.data_root=./data/lsmd_5186_final/tiles512_posaware_b2 \
    data.train.0.split_dir=./data/lsmd_5186_final/splits_512_posaware_b2 \
    data.val.data_root=./data/lsmd_5186_final/tiles512_posaware_b2 \
    data.val.split_dir=./data/lsmd_5186_final/splits_512_posaware_b2 \
    data.test.data_root=./data/lsmd_5186_final/tiles512_posaware_b2 \
    data.test.split_dir=./data/lsmd_5186_final/splits_512_posaware_b2 \
    data.samples_per_gpu=6 \
    data.workers_per_gpu=4
```

Synthetic-data variants:

```bash
bash scripts/train_lsmd_with_synth.sh
bash scripts/train_lsmd_with_synth_relsmd.sh
```

## Evaluation

Evaluate a saved checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml \
  path/to/checkpoint.pth \
  --eval BC BC_precision BC_recall SC SCS mIoU \
  --samples-per-gpu 1
```

If you also want prediction files written under the run directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml \
  runs/cross_modal_scd/lsmd/mapformer_t2map_finetune/latest.pth \
  --eval BC BC_precision BC_recall SC SCS mIoU \
  --samples-per-gpu 1 \
  --show-dir runs/cross_modal_scd/lsmd/mapformer_t2map_finetune/test_vis
```

Useful evaluation outputs:

- metrics are printed to the terminal log
- `results.pkl` is saved under `work_dir/test/<timestamp>/` when `--out` is not specified
- painted outputs are saved under `--show-dir` when requested

## Large-Area Inference

For inference on large GeoTIFFs, use:

- [tools/infer_large_tif_tiles.py](tools/infer_large_tif_tiles.py)
- [tools/eval_large_infer_to_log.py](tools/eval_large_infer_to_log.py)

Example:

```bash
python tools/infer_large_tif_tiles.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml \
  path/to/checkpoint.pth \
  --t2-image path/to/t2_image.tif \
  --t1-map path/to/t1_map.tif \
  --out-dir outputs/infer_example
```

## Other Included Benchmarks

The repository also keeps the original or related experiment paths for:

- HRSCD
- DynamicEarthNet

These configs live under:

- [configs/conditional_bcd](configs/conditional_bcd)
- [configs/conditional_scd](configs/conditional_scd)
- [configs/cross_modal_bcd](configs/cross_modal_bcd)
- [configs/cross_modal_scd](configs/cross_modal_scd)

## License Status

This fork does not currently declare a standalone open-source license.

Before publishing or redistributing this repository, verify redistribution rights for:

- the upstream MapFormer codebase
- MMSegmentation and Open-CD derived components
- any additional scripts or assets you added during LSMD road-change research

Until that is clarified, treat this repository as code prepared for publication review rather than as a fully relicensed package.

## Provenance

This repository is based on:

- the original MapFormer codebase
- MMSegmentation
- FHD/Open-CD style change-detection components

Please keep the upstream attribution when reusing or redistributing the code.
