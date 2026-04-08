import os
import sys
sys.path.append(os.getcwd())


import argparse
import copy
import os.path as osp
import time
import setproctitle

import mmcv
import torch
import torch.nn as nn
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

torch.multiprocessing.set_sharing_strategy('file_system')
#torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--exp-name', default='Train!', help='Exp name')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():

    args = parse_args()
    setproctitle.setproctitle(args.exp_name)
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = list(range(1)) if args.gpus is None else list(range(args.gpus))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # auto version subdir (v1, v2, ...) under work_dir unless resuming
    if cfg.work_dir is not None and cfg.resume_from is None:
        base_dir = osp.abspath(cfg.work_dir)
        if osp.isdir(base_dir):
            import re
            v_re = re.compile(r'^v(\\d+)$')
            existing = []
            for name in os.listdir(base_dir):
                m = v_re.match(name)
                if m:
                    try:
                        existing.append(int(m.group(1)))
                    except Exception:
                        pass
            next_v = (max(existing) + 1) if existing else 1
            cfg.work_dir = osp.join(base_dir, f'v{next_v}')

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    def _safe_cfg_text(cfg_obj):
        try:
            return cfg_obj.pretty_text
        except Exception:
            return cfg_obj.text if hasattr(cfg_obj, 'text') else str(cfg_obj)

    logger.info(f'Config:\n{_safe_cfg_text(cfg)}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # Optional: reinitialize only selected heads/maps for finetuning.
    reinit_cfg = cfg.get('finetune_reinit', None)
    if reinit_cfg and reinit_cfg.get('enable', False):
        reinit_logs = []

        def _try_reset(module, name):
            if module is None:
                reinit_logs.append(f'[skip] {name}: not found')
                return
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                reinit_logs.append(f'[ok] {name}: reset_parameters()')
                return
            touched = 0
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                        touched += 1
            reinit_logs.append(f'[ok] {name}: reset {touched} sub-layers')

        decode_head = getattr(model, 'decode_head', None)
        bc_head = getattr(decode_head, 'bc_head', None) if decode_head is not None else None
        sem_head = getattr(decode_head, 'sem_head', None) if decode_head is not None else None

        if reinit_cfg.get('map_encoder_layer1', False):
            map_encoder = getattr(bc_head, 'map_encoder', None) if bc_head is not None else None
            _try_reset(getattr(map_encoder, 'layer1', None), 'decode_head.bc_head.map_encoder.layer1')
        if reinit_cfg.get('bc_head_all', False):
            _try_reset(bc_head, 'decode_head.bc_head(all)')
        if reinit_cfg.get('sem_classifier', False):
            _try_reset(getattr(sem_head, 'conv_seg', None), 'decode_head.sem_head.conv_seg')
        if reinit_cfg.get('bc_classifier', False):
            _try_reset(getattr(bc_head, 'conv_seg', None), 'decode_head.bc_head.conv_seg')

        if reinit_logs:
            logger.info('Finetune reinit summary:\n  ' + '\n  '.join(reinit_logs))

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]

    if mmcv.is_list_of(cfg.workflow, list):
        cfg.workflow = [tuple(w) for w in cfg.workflow]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=_safe_cfg_text(cfg),
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
