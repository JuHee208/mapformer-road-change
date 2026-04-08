import os.path as osp
import csv
import json

import torch.distributed as dist
import numpy as np
import mmcv
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.
    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        save_best = kwargs.get('save_best', None)
        if isinstance(save_best, (list, tuple)):
            self._save_best_list = list(save_best)
            kwargs['save_best'] = self._save_best_list[0] if self._save_best_list else None
        else:
            self._save_best_list = None
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self._best_scores = {}

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        self._dump_eval_report(runner)
        if self.save_best:
            self._save_ckpt_multi(runner, key_score=key_score)

    def _dump_eval_report(self, runner):
        out = runner.log_buffer.output
        report_dir = osp.join(runner.work_dir, 'eval_reports')
        mmcv.mkdir_or_exist(report_dir)
        iter_id = int(runner.iter) + 1

        def _serialize(v):
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        payload = {'iter': iter_id}
        for k, v in out.items():
            payload[k] = _serialize(v)

        with open(osp.join(report_dir, f'iter_{iter_id:06d}.json'), 'w') as f:
            json.dump(payload, f, indent=2)

        csv_path = osp.join(report_dir, 'summary.csv')
        columns = [
            'iter',
            'mIoU_4', 'macroF1', 'IoU_12', 'F1_12', 'IoU_23', 'F1_23',
            'P_change', 'R_change', 'F1_change', 'AUPRC_change',
            'RoadT2_IoU', 'RoadT2_P', 'RoadT2_R', 'RoadT2_F1',
            'BC', 'BC_precision', 'BC_recall', 'SC', 'SCS', 'mIoU'
        ]
        row = {c: payload.get(c, '') for c in columns}
        write_header = not osp.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_ckpt_multi(self, runner, key_score=None):
        """Save multiple best checkpoints if configured."""
        if not self._save_best_list:
            if key_score is None:
                metrics = runner.log_buffer.output
                indicator = getattr(self, 'key_indicator', None)
                key_score = metrics.get(indicator, None) if indicator is not None else None
            self._save_ckpt(runner, key_score)
            return
        metrics = runner.log_buffer.output
        rule = getattr(self, 'rule', 'greater')
        for metric in self._save_best_list:
            if metric not in metrics:
                continue
            try:
                cur = float(metrics[metric])
            except Exception:
                continue
            best = self._best_scores.get(metric, None)
            if best is None:
                is_better = True
            else:
                is_better = cur > best if rule == 'greater' else cur < best
            if is_better:
                self._best_scores[metric] = cur
                if getattr(runner, 'rank', 0) == 0:
                    runner.save_checkpoint(
                        runner.work_dir,
                        filename_tmpl=f'best_{metric}.pth',
                        save_optimizer=True,
                        meta=runner.meta,
                        create_symlink=False)
                    if metric == self._save_best_list[0]:
                        self.best_ckpt_path = osp.join(runner.work_dir, f'best_{metric}.pth')
                if runner.meta is not None:
                    runner.meta.setdefault('hook_msgs', {})
                    runner.meta['hook_msgs'][f'best_{metric}'] = cur


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.
    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        save_best = kwargs.get('save_best', None)
        if isinstance(save_best, (list, tuple)):
            self._save_best_list = list(save_best)
            kwargs['save_best'] = self._save_best_list[0] if self._save_best_list else None
        else:
            self._save_best_list = None
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self._best_scores = {}

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            self._dump_eval_report(runner)

            if self.save_best:
                self._save_ckpt_multi(runner, key_score=key_score)

    def _dump_eval_report(self, runner):
        out = runner.log_buffer.output
        report_dir = osp.join(runner.work_dir, 'eval_reports')
        mmcv.mkdir_or_exist(report_dir)
        iter_id = int(runner.iter) + 1

        def _serialize(v):
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        payload = {'iter': iter_id}
        for k, v in out.items():
            payload[k] = _serialize(v)

        with open(osp.join(report_dir, f'iter_{iter_id:06d}.json'), 'w') as f:
            json.dump(payload, f, indent=2)

        csv_path = osp.join(report_dir, 'summary.csv')
        columns = [
            'iter',
            'mIoU_4', 'macroF1', 'IoU_12', 'F1_12', 'IoU_23', 'F1_23',
            'P_change', 'R_change', 'F1_change', 'AUPRC_change',
            'RoadT2_IoU', 'RoadT2_P', 'RoadT2_R', 'RoadT2_F1',
            'BC', 'BC_precision', 'BC_recall', 'SC', 'SCS', 'mIoU'
        ]
        row = {c: payload.get(c, '') for c in columns}
        write_header = not osp.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_ckpt_multi(self, runner, key_score=None):
        """Save multiple best checkpoints if configured."""
        if not self._save_best_list:
            if key_score is None:
                metrics = runner.log_buffer.output
                indicator = getattr(self, 'key_indicator', None)
                key_score = metrics.get(indicator, None) if indicator is not None else None
            self._save_ckpt(runner, key_score)
            return
        metrics = runner.log_buffer.output
        rule = getattr(self, 'rule', 'greater')
        for metric in self._save_best_list:
            if metric not in metrics:
                continue
            try:
                cur = float(metrics[metric])
            except Exception:
                continue
            best = self._best_scores.get(metric, None)
            if best is None:
                is_better = True
            else:
                is_better = cur > best if rule == 'greater' else cur < best
            if is_better:
                self._best_scores[metric] = cur
                if runner.rank == 0:
                    runner.save_checkpoint(
                        runner.work_dir,
                        filename_tmpl=f'best_{metric}.pth',
                        save_optimizer=True,
                        meta=runner.meta,
                        create_symlink=False)
                    if metric == self._save_best_list[0]:
                        self.best_ckpt_path = osp.join(runner.work_dir, f'best_{metric}.pth')
                if runner.meta is not None:
                    runner.meta.setdefault('hook_msgs', {})
                    runner.meta['hook_msgs'][f'best_{metric}'] = cur
