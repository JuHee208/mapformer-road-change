import os.path as osp
from functools import reduce

import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import scd_eval_metrics
from ..builder import DATASETS
from ..custom_cd import CustomDatasetCD
from ..pipelines import ComposeWithVisualization


@DATASETS.register_module()
class CustomDatasetCCD(CustomDatasetCD):
    '''
    Base class for datasets for Conditional CD.
    '''
    def __init__(self,
                 pipeline,
                 img1_dir,
                 img2_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index_bc=255,
                 ignore_index_sem=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 if_visualize=False,
                 ):
        self.pipeline = ComposeWithVisualization(pipeline, if_visualize=if_visualize)
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img1_dir):
                self.img1_dir = osp.join(self.data_root, self.img1_dir)
                self.img2_dir = osp.join(self.data_root, self.img2_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img1_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)


    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric: Dummy argument for compatibility.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        gt_bc_maps = self.get_gt_bc_maps(efficient_test)
        gt_sem_maps = self.get_gt_sem_maps(efficient_test)

        if self.CLASSES is None:
            num_semantic_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_sem_maps]))
        else:
            num_semantic_classes = len(self.CLASSES)

        ret_metrics = scd_eval_metrics(
            results=results,
            gt_bc_maps=gt_bc_maps,
            gt_sem_maps=gt_sem_maps,
            num_semantic_classes=num_semantic_classes,
            ignore_index_bc=self.ignore_index_bc,
            ignore_index_sem=self.ignore_index_sem
        )

        if self.CLASSES is None:
            class_names = tuple(range(num_semantic_classes))
        else:
            class_names = self.CLASSES

        SCD_metrics = ['BC', 'BC_precision', 'BC_recall', 'SC', 'SCS', 'mIoU']
        summary_table = PrettyTable(field_names=SCD_metrics)
        summary_table.add_row([np.round(ret_metrics[m], decimals=3) for m in SCD_metrics])

        print_log('Summary:', logger=logger)
        print_log('\n' + summary_table.get_string(), logger=logger)

        classwise_table = PrettyTable(field_names=['Class'] + list(class_names))
        classwise_table.add_row(['IoU'] + list(np.round(ret_metrics['IoU_per_class'], decimals=3)))
        classwise_table.add_row(['SC'] + list(np.round(ret_metrics['SC_per_class'], decimals=3)))

        print_log('per class results:', logger=logger)
        print_log('\n' + classwise_table.get_string(), logger=logger)

        # Extended report for 4-class change setting (LSMD finetune)
        if 'Change_IoU_per_class' in ret_metrics:
            ch_iou = ret_metrics['Change_IoU_per_class']
            ch_p = ret_metrics.get('Change_Precision_per_class', None)
            ch_r = ret_metrics.get('Change_Recall_per_class', None)
            ch_f1 = ret_metrics.get('Change_F1_per_class', None)

            ext_fields = ['iter', 'mIoU_4', 'macroF1', 'IoU_12', 'F1_12',
                          'P_change', 'R_change', 'F1_change', 'AUPRC_change',
                          'RoadT2_IoU', 'RoadT2_F1']
            ext_table = PrettyTable(field_names=ext_fields)
            ext_table.add_row([
                '-',  # iter is tracked in eval_reports/summary.csv
                np.round(ret_metrics.get('mIoU_4', np.nan), 3),
                np.round(ret_metrics.get('macroF1', np.nan), 3),
                np.round(ret_metrics.get('IoU_12', np.nan), 3),
                np.round(ret_metrics.get('F1_12', np.nan), 3),
                np.round(ret_metrics.get('P_change', np.nan), 3),
                np.round(ret_metrics.get('R_change', np.nan), 3),
                np.round(ret_metrics.get('F1_change', np.nan), 3),
                np.round(ret_metrics.get('AUPRC_change', np.nan), 3),
                np.round(ret_metrics.get('RoadT2_IoU', np.nan), 3),
                np.round(ret_metrics.get('RoadT2_F1', np.nan), 3),
            ])
            print_log('[Validation Summary - Extended]:', logger=logger)
            print_log('\n' + ext_table.get_string(), logger=logger)

            ch_table = PrettyTable(field_names=['metric', 'cls0', 'cls1', 'cls2', 'cls3'])
            ch_table.add_row(['IoU'] + list(np.round(ch_iou, 3)))
            if ch_p is not None:
                ch_table.add_row(['Prec'] + list(np.round(ch_p, 3)))
            if ch_r is not None:
                ch_table.add_row(['Recall'] + list(np.round(ch_r, 3)))
            if ch_f1 is not None:
                ch_table.add_row(['F1'] + list(np.round(ch_f1, 3)))
            print_log('[Per-class Change Metrics] (0:bg,1:new,2:removed,3:unchanged):', logger=logger)
            print_log('\n' + ch_table.get_string(), logger=logger)

            bc_table = PrettyTable(field_names=['metric', 'Precision', 'Recall', 'F1', 'AUPRC'])
            bc_table.add_row([
                'change(1|2)',
                np.round(ret_metrics.get('P_change', np.nan), 3),
                np.round(ret_metrics.get('R_change', np.nan), 3),
                np.round(ret_metrics.get('F1_change', np.nan), 3),
                np.round(ret_metrics.get('AUPRC_change', np.nan), 3),
            ])
            print_log('[Binary Change View] (positive = cls1|cls2):', logger=logger)
            print_log('\n' + bc_table.get_string(), logger=logger)

            road_table = PrettyTable(field_names=['target', 'IoU', 'Precision', 'Recall', 'F1'])
            road_table.add_row([
                'road_t2 (cls1|cls3)',
                np.round(ret_metrics.get('RoadT2_IoU', np.nan), 3),
                np.round(ret_metrics.get('RoadT2_P', np.nan), 3),
                np.round(ret_metrics.get('RoadT2_R', np.nan), 3),
                np.round(ret_metrics.get('RoadT2_F1', np.nan), 3),
            ])
            print_log('[Road Inference Metrics] (positive = cls1|cls3):', logger=logger)
            print_log('\n' + road_table.get_string(), logger=logger)

        return ret_metrics
