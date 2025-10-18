# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Provide args as argument to main()
# - Snapshot source code
# - Build UDA model instead of regular one

import argparse
import copy
import os
import os.path as osp
import sys
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models.builder import build_train_model
from mmseg.utils import collect_env, get_root_logger
from mmseg.utils.collect_env import gen_code_archive

"""
Dưới đây là **luồng chạy ngắn gọn** của file:

1. Parse CLI
* `parse_args(args)`: đọc `config`, `--work-dir`, `--options`, `--gpus/--gpu-ids`, `--seed`, `--launcher`,…
* Bổ sung `os.environ['LOCAL_RANK']` nếu thiếu.

2. Nạp & ghi đè cấu hình
* `cfg = Config.fromfile(args.config)`
* `cfg.merge_from_dict(args.options)` nếu có `--options`
* Bật `torch.backends.cudnn.benchmark` nếu `cfg.cudnn_benchmark=True`

3. Xác định `work_dir`
* Ưu tiên: `--work-dir` > `cfg.work_dir` > `./work_dirs/<tên_config>`
* Gán `cfg.model.train_cfg.work_dir = cfg.work_dir`
* Ánh xạ `load_from`, `resume_from`, `gpu_ids`/`gpus` vào `cfg`

4. Khởi tạo phân tán (nếu có)
* Nếu `launcher != 'none'` → `init_dist(args.launcher, **cfg.dist_params)`
* `distributed = True/False`

5. Chuẩn bị logging & snapshot
* Tạo thư mục `work_dir`
* `cfg.dump()` lưu bản config đã merge
* `gen_code_archive(work_dir)` snapshot mã nguồn
* Tạo logger, ghi **env info** + toàn bộ `cfg.pretty_text`

6. Seed & deterministic
* Lấy `seed` từ CLI hoặc `cfg`, rồi `set_random_seed`
* Ghi `seed` và `exp_name` vào `cfg`/`meta`

7. Build model
* `model = build_train_model(cfg, …)`
* Nếu `cfg` có `uda` → build **UDA wrapper** (DAFormer/HRDA…)
* Ngược lại → build **segmentor thường** (EncoderDecoder,…)
* `model.init_weights()`

8. Build datasets & workflow
* `datasets = [build_dataset(cfg.data.train)]`
* Nếu `cfg.workflow` có train+val → clone `cfg.data.val` (pipeline theo train) và append

9. Gắn metadata cho checkpoint
* Thêm `mmseg_version`, `config`, `CLASSES`, `PALETTE` vào `cfg.checkpoint_config.meta`
* `model.CLASSES = datasets[0].CLASSES`
* `meta.update(...)`

10. Chạy train
* `train_segmentor(model, datasets, cfg, distributed, validate=not args.no_validate, timestamp, meta)`
  → Dựng runner/optimizer/scheduler, đăng ký hooks (log, ckpt, eval), chạy train/val theo `workflow`, lưu **log + checkpoint** vào `work_dir`.
"""

def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
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
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(args):
    args = parse_args(args)

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
    cfg.model.train_cfg.work_dir = cfg.work_dir
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # snapshot source code
    gen_code_archive(cfg.work_dir)
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
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]

    model = build_train_model(  # dựng mô hình từ cfg.model (backbone, decode head, DAFormer head, …).
        cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(  # Chèn meta vào checkpoint_config
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES # Set model.CLASSES để các hook/visualizer dễ truy cập.
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    """
    Hàm “orchestrator” sẽ:
    dựng runner/optimizer/scheduler theo cfg,
    đăng ký hooks (log, ckpt, eval),
    nếu validate=True thì chạy validation theo workflow,
    lưu checkpoint + log + meta vào work_dir."""
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main(sys.argv[1:])
