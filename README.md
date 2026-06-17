# MapFormer for Road Change Detection

This repository extends MapFormer toward road change detection on LSMD-style data. It keeps the original HRSCD and DynamicEarthNet code paths, and adds LSMD dataset support, LSMD finetuning configs, synthetic-data workflows, and large-area tiled inference utilities.

The codebase is built on top of the original MapFormer implementation from the ICCV 2023 paper "MapFormer: Boosting Change Detection by Using Semantic Pre-change Information", plus MMSegmentation- and Open-CD-style components.

## 소개

이 저장소는 MapFormer를 기반으로 도로 변화탐지 실험을 수행하기 위해 정리한 코드입니다. 기본적으로는 원본 MapFormer의 HRSCD, DynamicEarthNet 실험 경로를 유지하고, 여기에 LSMD 계열 데이터셋용 데이터 로더, 파인튜닝 설정, synthetic 데이터 생성 스크립트, 대면적 GeoTIFF 추론 도구를 추가한 형태입니다.

실제로 많이 보게 되는 흐름은 아래와 같습니다.

1. `data/` 아래에 원본 데이터와 전처리 결과를 둡니다.
2. `tools/convert_datasets/` 아래 스크립트로 split 생성, reprojection, tile 생성을 수행합니다.
3. `configs/cross_modal_scd/lsmd/` 아래 실험 설정 파일을 선택합니다.
4. `python tools/train.py ... --work-dir runs/...` 형태로 학습합니다.
5. 학습 결과 체크포인트와 로그는 `runs/` 아래 실험별 폴더에 저장됩니다.
6. test, validation visualization, loss curve, large-area inference 결과도 같은 실험 폴더 또는 `outputs/` 아래에 정리됩니다.

중요한 폴더 구분은 다음과 같습니다.

- `model_ckpt/`: 학습 시작 전에 넣어두는 pretrained backbone 가중치
- `runs/`: 학습을 돌린 뒤 생성되는 실제 결과물 폴더
- `outputs/`: 대면적 추론 결과나 후처리 결과
- `data/`: 원본 데이터, 분할 파일, 타일 데이터

즉, 학습된 모델은 `model_ckpt/` 가 아니라 `runs/` 아래에 저장됩니다. `model_ckpt/mit_b2.pth` 같은 파일은 초기 backbone 가중치이고, 실제 fine-tuning 결과는 예를 들어 `runs/cross_modal_scd/lsmd/mapformer_t2map_finetune/best_SCS.pth` 같은 경로에 생깁니다.

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

학습 결과물 정리 기준은 아래처럼 보면 됩니다.

- `runs/<task>/<dataset>/<experiment_name>/iter_XXXX.pth`: 주기적으로 저장된 checkpoint
- `runs/<task>/<dataset>/<experiment_name>/best_*.pth`: 특정 metric 기준 best checkpoint
- `runs/<task>/<dataset>/<experiment_name>/*.log`: 텍스트 로그
- `runs/<task>/<dataset>/<experiment_name>/*.log.json`: 파싱 가능한 학습 로그
- `runs/<task>/<dataset>/<experiment_name>/val_vis/`: validation 시각화 결과
- `runs/<task>/<dataset>/<experiment_name>/loss_curves/`: loss curve 이미지나 로그 기반 그래프
- `runs/<task>/<dataset>/<experiment_name>/test/` 또는 `test_vis/`: 테스트 결과 저장 폴더

현재 로컬 기준으로는 다음과 같은 실험 폴더들이 확인됩니다.

- `runs/cross_modal_scd/lsmd/mapformer_t2map_finetune/`
- `runs/cross_modal_scd/lsmd/mapformer_t2map_finetune_stage1/`
- `runs/cross_modal_scd/lsmd/mapformer_t2map_finetune_stage1_focal/`
- `runs/cross_modal_scd/re_lsmd/mapformer_t2map_ft_recover_bs6w4_32k/`

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

`model_ckpt/` 는 시작용 가중치 폴더입니다. 여기 들어가는 것은 pretrained backbone 이고, 학습 완료 후 best model 이 자동으로 이 폴더에 복사되지는 않습니다.

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

## 학습 결과 저장 위치

학습을 실행하면 결과는 `--work-dir` 로 지정한 경로 아래에 저장됩니다. `--work-dir` 를 생략하면 config 또는 기본 runner 설정에 따라 work directory 가 결정됩니다. 공개용으로는 항상 `--work-dir` 를 명시하는 편이 좋습니다.

예를 들어 아래처럼 학습하면:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
python tools/train.py \
  configs/cross_modal_scd/lsmd/mapformer_t2map_finetune.yaml \
  --work-dir runs/cross_modal_scd/lsmd/mapformer_t2map_finetune
```

주요 결과물은 보통 다음처럼 생깁니다.

```text
runs/cross_modal_scd/lsmd/mapformer_t2map_finetune/
  iter_500.pth
  iter_1000.pth
  ...
  best_SCS.pth
  best_F1_change.pth
  best_IoU_12.pth
  20260222_054331.log
  20260222_054331.log.json
  val_vis/
  loss_curves/
```

파일 의미는 다음과 같습니다.

- `iter_XXXX.pth`: 해당 iteration 시점 checkpoint
- `best_*.pth`: validation metric 기준 최고 성능 checkpoint
- `*.log`: 사람이 읽는 로그
- `*.log.json`: loss curve 재생성이나 분석에 쓰기 좋은 structured log

즉, 논문이나 결과 보고용으로 다시 사용할 모델은 보통 `best_SCS.pth`, `best_F1_change.pth` 같은 파일을 보면 됩니다.

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
