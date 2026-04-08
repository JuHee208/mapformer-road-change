import random
import warnings
import os
import os.path as osp
import json
from dataclasses import dataclass, field

import numpy as np
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import Hook, build_optimizer, build_runner
from mmcv.utils import print_log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from mmseg.core.evaluation.eval_hooks import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
from mmseg.apis import single_gpu_test
from PIL import Image


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # Keep frozen backbone in eval mode during stage-1 finetuning.
    freeze_cfg = cfg.get('finetune_freeze', None)
    if freeze_cfg and freeze_cfg.get('enable', False) \
            and freeze_cfg.get('backbone', False) \
            and freeze_cfg.get('backbone_eval', True):
        class FrozenBackboneEvalHook(Hook):
            def _set_backbone_eval(self, runner):
                model_ref = runner.model.module if hasattr(runner.model, 'module') else runner.model
                backbone = getattr(model_ref, 'backbone', None)
                if backbone is not None:
                    backbone.eval()

            def before_train_epoch(self, runner):
                self._set_backbone_eval(runner)

            def before_train_iter(self, runner):
                self._set_backbone_eval(runner)

        runner.register_hook(FrozenBackboneEvalHook(), priority='HIGH')

    # Optional loss plot hook
    loss_plot_cfg = cfg.get('loss_plot', None)
    if loss_plot_cfg and loss_plot_cfg.get('enable', False):
        interval = loss_plot_cfg.get('interval', 50)
        out_file = loss_plot_cfg.get('out_file', None)

        class LossPlotHook(Hook):
            def __init__(self, interval, out_file, work_dir, resume=False):
                self.interval = interval
                self.out_file = out_file
                self.work_dir = work_dir
                self.iters = []
                self.loss_total = []
                self.loss_sem = []
                self.loss_bc = []
                self.loss_contrast = []
                if resume:
                    self._load_history()

            def _load_history(self):
                if not self.work_dir or not osp.isdir(self.work_dir):
                    return
                log_files = sorted(
                    (osp.join(self.work_dir, f) for f in os.listdir(self.work_dir)
                     if f.endswith('.log.json')),
                    key=lambda p: osp.getmtime(p),
                )
                if not log_files:
                    return
                # Merge records from all log files in time order
                records = {}
                for lf in log_files:
                    try:
                        with open(lf, 'r') as f:
                            for line in f:
                                try:
                                    d = json.loads(line)
                                except Exception:
                                    continue
                                if 'iter' not in d:
                                    continue
                                it = int(d['iter'])
                                records[it] = d
                    except Exception:
                        continue
                if not records:
                    return
                for it in sorted(records.keys()):
                    d = records[it]
                    self.iters.append(it)
                    self.loss_total.append(d.get('loss', None))
                    self.loss_sem.append(d.get('decode.sem.loss_seg', None))
                    self.loss_bc.append(d.get('decode.bc.loss_seg', None))
                    # contrastive: sum any logged contrastive terms
                    contrast_sum = 0.0
                    contrast_found = False
                    for k, v in d.items():
                        if 'contrastive_loss' in k:
                            try:
                                contrast_sum += float(v)
                                contrast_found = True
                            except Exception:
                                pass
                    self.loss_contrast.append(contrast_sum if contrast_found else None)

            def after_train_iter(self, runner):
                # Only save on rank 0 in distributed
                if hasattr(runner, 'rank') and runner.rank != 0:
                    return
                cur_iter = runner.iter + 1
                if cur_iter % self.interval != 0:
                    return
                loss = None
                sem_loss = None
                bc_loss = None
                contrast_loss = None
                # Prefer direct runner outputs if available
                if hasattr(runner, 'outputs') and runner.outputs is not None:
                    out_loss = runner.outputs.get('loss', None)
                    if out_loss is not None:
                        try:
                            loss = float(out_loss.item() if hasattr(out_loss, 'item') else out_loss)
                        except Exception:
                            loss = None
                log_out = runner.log_buffer.output
                # Prefer log_vars from outputs if available (more reliable per-iter)
                if hasattr(runner, 'outputs') and runner.outputs is not None:
                    log_vars = runner.outputs.get('log_vars', None)
                    if isinstance(log_vars, dict) and log_vars:
                        log_out = log_vars
                if loss is None:
                    loss = log_out.get('loss', None)
                # sem / bc losses if present
                sem_loss = log_out.get('decode.sem.loss_seg', None)
                bc_loss = log_out.get('decode.bc.loss_seg', None)
                # contrastive: sum any contrastive terms if present
                contrast_sum = 0.0
                contrast_found = False
                for k, v in log_out.items():
                    if 'contrastive_loss' in k:
                        try:
                            contrast_sum += float(v)
                            contrast_found = True
                        except Exception:
                            pass
                if contrast_found:
                    contrast_loss = contrast_sum

                if loss is None:
                    loss = 0.0
                    for k, v in log_out.items():
                        if 'loss' in k:
                            try:
                                loss += float(v)
                            except Exception:
                                pass
                try:
                    loss = float(loss)
                except Exception:
                    return
                self.iters.append(cur_iter)
                self.loss_total.append(loss)
                self.loss_sem.append(float(sem_loss) if sem_loss is not None else None)
                self.loss_bc.append(float(bc_loss) if bc_loss is not None else None)
                self.loss_contrast.append(float(contrast_loss) if contrast_loss is not None else None)
                vis_dir = self.out_file or osp.join(runner.work_dir, 'loss_curves')
                mmcv.mkdir_or_exist(vis_dir)

                def _plot_and_save(y, name, title):
                    if y is None or len(y) == 0:
                        return
                    plt.figure(figsize=(6, 4))
                    plt.plot(self.iters, y, label=name)
                    plt.xlabel('iter')
                    plt.ylabel('loss')
                    plt.title(title)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(osp.join(vis_dir, f'{name}.png'))
                    plt.close()

                _plot_and_save(self.loss_total, 'loss_total', 'Training Loss (Total)')
                if any(v is not None for v in self.loss_sem):
                    _plot_and_save(self.loss_sem, 'loss_sem', 'Training Loss (Sem)')
                if any(v is not None for v in self.loss_bc):
                    _plot_and_save(self.loss_bc, 'loss_bc', 'Training Loss (BC)')
                if any(v is not None for v in self.loss_contrast):
                    _plot_and_save(self.loss_contrast, 'loss_contrast', 'Training Loss (Contrast)')

        resume_flag = bool(getattr(cfg, 'resume_from', None))
        runner.register_hook(LossPlotHook(interval, out_file, runner.work_dir, resume=resume_flag), priority='LOW')

    # register hooks
    for logging_hook in cfg.log_config.hooks:
        if logging_hook.type == 'WandbLoggerHook':
            logging_hook.init_kwargs.config = cfg.to_dict()
            logging_hook.init_kwargs.name = cfg.run_name
            logging_hook.init_kwargs.dir = cfg.work_dir
            logging_hook.init_kwargs.tags = cfg.work_dir.split('/')[:-1]
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook_class = DistEvalHook if distributed else EvalHook
        eval_hook = eval_hook_class(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook, priority='LOW') # https://github.com/open-mmlab/mmcv/issues/1261

        # Optional visualization of a few validation samples
        val_vis_cfg = cfg.get('val_vis', None)
        if val_vis_cfg and val_vis_cfg.get('enable', False):
            vis_interval = val_vis_cfg.get('interval', eval_cfg.get('interval', 500))
            num_samples = val_vis_cfg.get('num_samples', 5)
            out_dir = val_vis_cfg.get('out_dir', None)

            class ValVisHook(Hook):
                def __init__(self, dataloader, interval, num_samples, out_dir):
                    self.dataloader = dataloader
                    self.interval = interval
                    self.num_samples = num_samples
                    self.out_dir = out_dir

                def _extract_img(self, data):
                    img = data.get('img', None)
                    if img is None:
                        return None, None
                    if hasattr(img, 'data'):
                        img = img.data
                    if isinstance(img, list):
                        img = img[0]
                    if torch.is_tensor(img):
                        img = img[0].detach().cpu()  # (C,H,W)
                    meta = data.get('img_metas', None)
                    if meta is not None and hasattr(meta, 'data'):
                        meta = meta.data
                    if isinstance(meta, list):
                        meta = meta[0]
                    return img, meta

                def _denorm(self, img, meta):
                    if meta is None:
                        return img
                    if hasattr(meta, 'data'):
                        meta = meta.data
                    # unwrap nested lists
                    while isinstance(meta, list) and len(meta) > 0:
                        meta = meta[0]
                    norm = meta.get('img_norm_cfg', None)
                    if norm is None:
                        return img
                    # handle multi-image concatenation (e.g., 6 channels)
                    if img is not None and img.shape[0] > 3:
                        img = img[:3]
                    mean = torch.tensor(norm['mean']).view(-1, 1, 1)
                    std = torch.tensor(norm['std']).view(-1, 1, 1)
                    img = img * std + mean
                    if not norm.get('to_rgb', True):
                        img = img[[2, 1, 0], ...]
                    return img

                def _load_gt_maps(self, idx):
                    """Load BC/SEM GT maps directly from dataset annotations."""
                    img_infos = getattr(self.dataloader.dataset, 'img_infos', None)
                    if not img_infos or idx >= len(img_infos):
                        return None, None
                    ann = img_infos[idx].get('ann', None)
                    if ann is None:
                        return None, None

                    def _read(path):
                        if path is None:
                            return None
                        try:
                            arr = mmcv.imread(path, flag='unchanged', backend='tifffile')
                            return arr.squeeze().astype(np.uint8)
                        except Exception:
                            return None

                    gt_bc = _read(ann.get('seg_map', None))
                    gt_sem = _read(ann.get('seg_map_post', None))
                    return gt_bc, gt_sem

                def _resize_label(self, label, target_hw):
                    if label is None:
                        return None
                    if label.shape == target_hw:
                        return label
                    try:
                        return mmcv.imresize(
                            label.astype(np.uint8),
                            target_hw[::-1],
                            interpolation='nearest')
                    except Exception:
                        return label

                def _safe_save_image(self, image, save_path):
                    try:
                        image.save(save_path)
                    except Exception as exc:
                        print_log(
                            f'ValVisHook: failed to save {save_path}: {exc}',
                            logger='mmseg')

                def _save_bc_map(self, arr_u8, save_path):
                    if arr_u8 is None:
                        return
                    arr_u8 = arr_u8.astype(np.uint8)
                    ignore_mask = (arr_u8 == 255)
                    # Always use fixed 4-class BC palette:
                    # 0 bg(black), 1 new(red), 2 removed(blue), 3 unchanged(green)
                    palette_bc = np.array([
                        [0, 0, 0],        # 0 bg
                        [255, 0, 0],      # 1 new
                        [0, 0, 255],      # 2 removed
                        [0, 255, 0],      # 3 unchanged
                    ], dtype=np.uint8)
                    hbc, wbc = arr_u8.shape
                    bc_rgb = np.full((hbc, wbc, 3), 128, dtype=np.uint8)
                    for cls_idx, color in enumerate(palette_bc):
                        bc_rgb[arr_u8 == cls_idx] = color
                    bc_rgb[ignore_mask] = np.array([127, 127, 127], dtype=np.uint8)
                    self._safe_save_image(Image.fromarray(bc_rgb), save_path)

                def _save_sem_map(self, arr_u8, save_path):
                    if arr_u8 is None:
                        return
                    arr_u8 = arr_u8.astype(np.uint8)
                    ignore_mask = (arr_u8 == 255)
                    if arr_u8.max() <= 1:
                        vis = (arr_u8 * 255).astype(np.uint8)
                        vis[ignore_mask] = 127
                        self._safe_save_image(
                            Image.fromarray(vis, mode='L'),
                            save_path)
                    else:
                        h, w = arr_u8.shape
                        sem_rgb = np.full((h, w, 3), 128, dtype=np.uint8)
                        palette = [
                            (220, 20, 60),   # artificial
                            (0, 128, 0),     # agricultural
                            (34, 139, 34),   # forest
                            (0, 191, 255),   # wetland
                            (0, 0, 255),     # water
                        ]
                        for cls_idx, color in enumerate(palette):
                            sem_rgb[arr_u8 == cls_idx] = color
                        sem_rgb[ignore_mask] = np.array([127, 127, 127], dtype=np.uint8)
                        self._safe_save_image(Image.fromarray(sem_rgb), save_path)

                def after_train_iter(self, runner):
                    if hasattr(runner, 'rank') and runner.rank != 0:
                        return
                    cur_iter = runner.iter + 1
                    if cur_iter % self.interval != 0:
                        return
                    vis_root = self.out_dir or osp.join(runner.work_dir, 'val_vis')
                    mmcv.mkdir_or_exist(vis_root)
                    iter_dir = osp.join(vis_root, f'iter_{cur_iter:06d}')
                    mmcv.mkdir_or_exist(iter_dir)

                    # switch to eval mode for visualization
                    model = runner.model
                    model.eval()
                    with torch.no_grad():
                        count = 0
                        pr_scores = []
                        pr_labels = []
                        # sample random indices
                        indices = list(range(len(self.dataloader.dataset)))
                        random.shuffle(indices)
                        indices = indices[:self.num_samples]
                        for idx in indices:
                            gt_bc_raw, gt_sem_raw = self._load_gt_maps(idx)
                            data = self.dataloader.dataset[idx]
                            # mimic dataloader output for single sample
                            for k, v in data.items():
                                if hasattr(v, 'data'):
                                    data[k] = v.data
                                if isinstance(v, list):
                                    data[k] = v
                            data = self.dataloader.collate_fn([data])
                            img, meta = self._extract_img(data)
                            if img is not None:
                                img = self._denorm(img, meta)
                                # If bitemporal images are concatenated, take I2 (last 3 channels)
                                if img.shape[0] >= 6:
                                    img_show = img[-3:, ...]
                                else:
                                    img_show = img[:3, ...]
                                img_show = img_show.permute(1, 2, 0).clamp(0, 255).byte().numpy()
                                self._safe_save_image(
                                    Image.fromarray(img_show),
                                    osp.join(iter_dir, f'sample_{count}_img.png'))
                            # forward
                            # get probability map for BC
                            model_ref = model.module if hasattr(model, 'module') else model
                            img_tensor = data['img']
                            if hasattr(img_tensor, 'data'):
                                img_tensor = img_tensor.data
                            if isinstance(img_tensor, list):
                                img_tensor = img_tensor[0]
                            # If bitemporal images are concatenated, use I2 (last 3 channels)
                            if img_tensor is not None and img_tensor.shape[1] >= 6:
                                img_tensor = img_tensor[:, -3:, ...]
                            # ensure tensor is on the same device as model
                            img_tensor = img_tensor.to(next(model_ref.parameters()).device)
                            gt_pre = data.get('gt_semantic_seg_pre', None)
                            if hasattr(gt_pre, 'data'):
                                gt_pre = gt_pre.data
                            if isinstance(gt_pre, list):
                                gt_pre = gt_pre[0]
                            if gt_pre is not None:
                                gt_pre = gt_pre.to(next(model_ref.parameters()).device)
                            img_metas = data.get('img_metas', None)
                            def _unwrap_meta(m):
                                if hasattr(m, 'data'):
                                    m = m.data
                                while isinstance(m, list) and len(m) > 0:
                                    m = m[0]
                                return m

                            if hasattr(img_metas, 'data'):
                                img_metas = img_metas.data
                            if isinstance(img_metas, list):
                                img_metas = [_unwrap_meta(m) for m in img_metas]
                            else:
                                img_metas = [_unwrap_meta(img_metas)]
                            out = model_ref.inference(
                                img=img_tensor,
                                img_meta=img_metas,
                                rescale=False,
                                gt_semantic_seg_pre=gt_pre)
                            bc_soft = out['bc']
                            if bc_soft.shape[1] >= 4:
                                bc_prob = (bc_soft[:, 1, :, :] + bc_soft[:, 2, :, :]).detach().cpu().numpy()
                            elif bc_soft.shape[1] >= 2:
                                bc_prob = bc_soft[:, 1, :, :].detach().cpu().numpy()
                            else:
                                bc_prob = bc_soft[:, 0, :, :].detach().cpu().numpy()
                            gt_change = data.get('gt_semantic_seg', None)
                            if hasattr(gt_change, 'data'):
                                gt_change = gt_change.data
                            if isinstance(gt_change, list):
                                gt_change = gt_change[0]
                            # fallback: derive change map from pre/post semantics
                            if gt_change is None:
                                gt_pre_tmp = data.get('gt_semantic_seg_pre', None)
                                gt_post_tmp = data.get('gt_semantic_seg_post', None)
                                if hasattr(gt_pre_tmp, 'data'):
                                    gt_pre_tmp = gt_pre_tmp.data
                                if hasattr(gt_post_tmp, 'data'):
                                    gt_post_tmp = gt_post_tmp.data
                                if isinstance(gt_pre_tmp, list):
                                    gt_pre_tmp = gt_pre_tmp[0]
                                if isinstance(gt_post_tmp, list):
                                    gt_post_tmp = gt_post_tmp[0]
                                if gt_pre_tmp is not None and gt_post_tmp is not None:
                                    # gt_change: 1 if class differs, 0 otherwise, 255 ignore if either is ignore
                                    gt_change = (gt_post_tmp != gt_pre_tmp).long()
                                    ignore_mask = (gt_pre_tmp == 255) | (gt_post_tmp == 255)
                                    gt_change[ignore_mask] = 255
                            if gt_change is not None:
                                gt_change = gt_change.detach().cpu().numpy()
                                # normalize to (B,H,W)
                                if gt_change.ndim == 2:
                                    gt_change = gt_change[None, ...]
                                elif gt_change.ndim == 4:
                                    # (B,1,H,W) -> (B,H,W)
                                    gt_change = gt_change[:, 0, ...]
                                # handle batch-wise PR with ignore mask
                                bs = min(bc_prob.shape[0], gt_change.shape[0])
                                for b in range(bs):
                                    gt_b = gt_change[b]
                                    prob_b = bc_prob[b]
                                    if gt_b.shape != prob_b.shape:
                                        # resize GT to prob shape for PR
                                        try:
                                            gt_b = mmcv.imresize(
                                                gt_b.astype('uint8'),
                                                prob_b.shape[::-1],
                                                interpolation='nearest')
                                        except Exception:
                                            continue
                                    mask = (gt_b != 255)
                                    if mask.any():
                                        gt_bin = gt_b[mask]
                                        # 4-class change: positive = cls1|cls2
                                        if gt_bin.max() > 1:
                                            gt_bin = np.isin(gt_bin, [1, 2]).astype(np.uint8)
                                        else:
                                            gt_bin = (gt_bin == 1).astype(np.uint8)
                                        pr_scores.append(prob_b[mask].reshape(-1))
                                        pr_labels.append(gt_bin.reshape(-1))

                            # drop gt_semantic_seg for inference path
                            if 'gt_semantic_seg' in data:
                                data_inf = dict(data)
                                data_inf.pop('gt_semantic_seg', None)
                            else:
                                data_inf = data
                            results = model(return_loss=False, **data_inf)
                            if isinstance(results, list):
                                results = results[0]
                            bc = results.get('bc', None)
                            sem = results.get('sem', None)

                            def _to_numpy(x):
                                if x is None:
                                    return None
                                if isinstance(x, list):
                                    x = x[0]
                                if torch.is_tensor(x):
                                    x = x.detach().cpu().numpy()
                                return x

                            bc = _to_numpy(bc)
                            sem = _to_numpy(sem)

                            if bc is not None:
                                if bc.ndim == 3:
                                    # (C,H,W) -> argmax over C if needed
                                    if bc.shape[0] > 1:
                                        bc_map = bc.argmax(axis=0)
                                    else:
                                        bc_map = bc[0]
                                else:
                                    bc_map = bc
                                bc_u8 = bc_map.astype('uint8')
                                self._save_bc_map(
                                    bc_u8,
                                    osp.join(iter_dir, f'sample_{count}_bc.png'))
                                gt_bc = self._resize_label(gt_bc_raw, bc_u8.shape)
                                self._save_bc_map(
                                    gt_bc,
                                    osp.join(iter_dir, f'sample_{count}_bc_gt.png'))

                            if sem is not None:
                                if sem.ndim == 3:
                                    # (C,H,W) -> class map
                                    sem_u8 = sem.argmax(axis=0).astype('uint8')
                                else:
                                    sem_u8 = sem.astype('uint8')
                                # For binary semantics, force grayscale:
                                # 0=black(background), 1=white(road)
                                if sem_u8.max() <= 1:
                                    self._save_sem_map(
                                        sem_u8,
                                        osp.join(iter_dir, f'sample_{count}_sem.png'))
                                else:
                                    if hasattr(model_ref, 'PALETTE') and model_ref.PALETTE is not None:
                                        palette = model_ref.PALETTE
                                    else:
                                        # fallback palette (5 classes)
                                        palette = [
                                            (220, 20, 60),   # artificial
                                            (0, 128, 0),     # agricultural
                                            (34, 139, 34),   # forest
                                            (0, 191, 255),   # wetland
                                            (0, 0, 255),     # water
                                        ]
                                    h, w = sem_u8.shape
                                    # start with gray for unknown/ignore pixels
                                    sem_rgb = np.full((h, w, 3), 128, dtype=np.uint8)
                                    for idx, color in enumerate(palette):
                                        sem_rgb[sem_u8 == idx] = color
                                    self._safe_save_image(
                                        Image.fromarray(sem_rgb),
                                        osp.join(iter_dir, f'sample_{count}_sem.png'))
                                gt_sem = self._resize_label(gt_sem_raw, sem_u8.shape)
                                self._save_sem_map(
                                    gt_sem,
                                    osp.join(iter_dir, f'sample_{count}_sem_gt.png'))
                            count += 1
                        # PR curve
                        if pr_scores and pr_labels:
                            import numpy as _np
                            scores = _np.concatenate(pr_scores)
                            labels = _np.concatenate(pr_labels)
                            precision, recall, _ = precision_recall_curve(labels, scores)
                            plt.figure(figsize=(5, 4))
                            plt.plot(recall, precision)
                            plt.xlabel('Recall')
                            plt.ylabel('Precision')
                            plt.title('PR Curve (BC)')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            try:
                                plt.savefig(osp.join(iter_dir, 'pr_curve.png'))
                            except Exception as exc:
                                print_log(
                                    f'ValVisHook: failed to save '
                                    f"{osp.join(iter_dir, 'pr_curve.png')}: {exc}",
                                    logger='mmseg')
                            plt.close()
                        else:
                            # write debug note if PR not generated
                            try:
                                with open(osp.join(iter_dir, 'pr_curve_skip.txt'), 'w') as f:
                                    f.write('PR curve skipped: no valid (non-ignored) GT pixels or shape mismatch.\n')
                            except Exception:
                                pass
                    model.train()

            runner.register_hook(
                ValVisHook(val_dataloader, vis_interval, num_samples, out_dir),
                priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from, strict=False)
    runner.run(data_loaders, cfg.workflow)

    # test
    if cfg.data.test:
        print_log('============================== Testing ==============================', logger=logger)
        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        test_dataloader = build_dataloader(
            test_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        # Load best checkpoint if available; otherwise skip test to avoid crash
        best_ckpt = None
        try:
            best_ckpt = getattr(eval_hook, 'best_ckpt_path', None)
        except Exception:
            best_ckpt = None
        if isinstance(best_ckpt, str) and best_ckpt:
            runner.load_checkpoint(best_ckpt)
        else:
            print_log('No best checkpoint found; skipping final test.', logger=logger)
            return
        results = single_gpu_test(
            runner.model,
            test_dataloader,
            show=False,
            efficient_test=False)
        eval_res = test_dataset.evaluate(results, logger=runner.logger)
