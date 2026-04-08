import os.path as osp
import numpy as np

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from ..builder import DATASETS
from ..pipelines import ComposeWithVisualization
from .custom_ccd import CustomDatasetCCD


@DATASETS.register_module()
class LSMDDatasetCCD(CustomDatasetCCD):
    """LSMD dataset for conditional/cross-modal SCD.

    Expected tiled structure under ``data_root``:
      - images/t2/<region>/<tile>.tif
      - labels/t1/<region>/<tile>.tif
      - labels/t2/<region>/<tile>.tif
      - labels/change/<region>/<tile>.tif
      - splits/<split>.txt  # each line: <region>/<tile_id_without_suffix>
    """

    CLASSES = ["background", "road"]
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(
        self,
        pipeline,
        data_root,
        split,
        split_dir=None,
        synth_data_root=None,
        synth_tag="_syn_",
        ann_dir=None,
        img_suffix=".tif",
        seg_map_suffix=".tif",
        test_mode=False,
        ignore_index_bc=255,
        ignore_index_sem=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        if_visualize=False,
    ):
        self.pipeline = ComposeWithVisualization(pipeline, if_visualize=if_visualize)
        self.data_root = data_root
        self.img_dir = osp.join(data_root, "images", "t2")
        self.ann_dir = osp.join(data_root, "labels")
        self.synth_data_root = synth_data_root
        self.synth_tag = synth_tag
        if synth_data_root is not None:
            self.synth_img_dir = osp.join(synth_data_root, "images", "t2")
            self.synth_ann_dir = osp.join(synth_data_root, "labels")
        else:
            self.synth_img_dir = None
            self.synth_ann_dir = None
        self.split = split
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix

        if split_dir is None:
            split_file = osp.join(data_root, "splits", split + ".txt")
        else:
            split_file = osp.join(split_dir, split + ".txt")
        with open(split_file, "r") as f:
            self.tile_ids = [s.strip() for s in f.readlines() if s.strip()]

        self.img_infos = self.load_img_infos()
        self.test_mode = test_mode
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)

    def load_img_infos(self):
        img_infos = []
        for tile_id in self.tile_ids:
            if "/" not in tile_id:
                raise ValueError(f"Invalid tile id in split: {tile_id}")
            region, tile = tile_id.split("/", 1)
            tile_name = tile + self.img_suffix
            # Mixed split support:
            # - regular IDs use data_root
            # - synthetic IDs (containing synth_tag) can be resolved from synth_data_root
            if self.synth_img_dir is not None and self.synth_tag and (self.synth_tag in tile):
                img_dir = self.synth_img_dir
                ann_dir = self.synth_ann_dir
            else:
                img_dir = self.img_dir
                ann_dir = self.ann_dir
            img_info = dict(
                filename=osp.join(img_dir, region, tile_name),
                ann=dict(
                    seg_map=osp.join(ann_dir, "change", region, tile + self.seg_map_suffix),
                    seg_map_pre=osp.join(ann_dir, "t1", region, tile + self.seg_map_suffix),
                    seg_map_post=osp.join(ann_dir, "t2", region, tile + self.seg_map_suffix),
                ),
            )
            img_infos.append(img_info)
        print_log(f"Loaded {len(img_infos)} image pairs", logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results["seg_fields"] = []
        if self.custom_classes:
            results["label_map"] = self.label_map

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_bc_maps(self, efficient_test=False):
        gt_bc_maps = []
        for img_info in self.img_infos:
            bc_map_file = img_info["ann"]["seg_map"]
            gt_bc_map = mmcv.imread(bc_map_file, flag="unchanged", backend="tifffile")
            gt_bc_maps.append(gt_bc_map.astype(np.uint8))
        return gt_bc_maps

    def get_gt_sem_maps(self, efficient_test=False):
        gt_sem_maps = []
        for img_info in self.img_infos:
            seg_map_post = img_info["ann"]["seg_map_post"]
            gt_seg_map_post = mmcv.imread(seg_map_post, flag="unchanged", backend="tifffile")
            gt_seg_map_post = gt_seg_map_post.astype(np.uint8)
            if self.reduce_zero_label:
                gt_seg_map_post[gt_seg_map_post == 0] = self.ignore_index_sem
                gt_seg_map_post = gt_seg_map_post - 1
                gt_seg_map_post[gt_seg_map_post == self.ignore_index_sem - 1] = self.ignore_index_sem
            gt_sem_maps.append(gt_seg_map_post.astype(np.uint8))
        return gt_sem_maps
