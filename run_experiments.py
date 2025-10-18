# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
"""
Nhận 1 trong 2 đầu vào: --config <đường_dẫn_config.py> hoặc --exp <mã_thí_nghiệm> (tạo nhiều config từ experiments.py).
Sinh ra file config con (JSON) có name, work_dir, git_rev, _base_… đặt dưới configs/generated/....
Gọi hàm huấn luyện tools.train.main([...]) cho từng cấu hình.
Tùy chọn --debug để rút ngắn chu kỳ log/val, và thêm tham số debug cho UDA (ví dụ DACS).
Hiện chỉ hỗ trợ --machine local.

Script này sinh ra config con (JSON) kế thừa từ config gốc, đóng gói metadata
(name/work_dir/git_rev), có chế độ debug, rồi gọi tools/train.py để huấn luyện — hỗ trợ chạy 1 file
config hoặc nhiều config từ experiments.py.

Nó không phải để huấn luyện trực tiếp,
mà để tự động sinh config JSON từ nhiều file cấu hình hoặc kịch bản thí nghiệm, rồi gọi train.py để huấn luyện từng cái.
"""


import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime

import torch
from experiments import generate_experiment_cfgs
from mmcv import Config, get_git_hash
from tools import train

# chạy lệnh shell, in stdout
def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

# bọc rsync (chưa dùng trong đoạn main này)
def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)


if __name__ == '__main__':
    # hoặc --exp hoặc --config (không được cả hai/không cái nào). 
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--exp',
        type=int,
        default=None,
        help='Experiment id as defined in experiment.py',
    )
    group.add_argument(
        '--config',
        default=None,
        help='Path to config file',
    )
    # --machine hiện chỉ cho phép local.
    parser.add_argument(
        '--machine', type=str, choices=['local'], default='local')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    assert (args.config is None) != (args.exp is None), \
        'Either config or exp has to be defined.'

    GEN_CONFIG_DIR = 'configs/generated/'
    JOB_DIR = 'jobs'    # # (khai báo nhưng chưa dùng ở phần dưới)
    cfgs, config_files = [], []

    # Training with Predefined Config
    if args.config is not None:
        cfg = Config.fromfile(args.config)  # đọc .py thành đối tượng cfg (mmcv.Config)

        # Specify Name and Work Directory
        # Tạo tên thí nghiệm và tên instance duy nhất
        exp_name = f'{args.machine}-{cfg["exp"]}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                      f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        
        # Sinh một "child config" ở dạng JSON (mỏng) kế thừa từ file gốc:
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),  # trỏ ngược 2 cấp do nằm trong configs/generated/...
            'name': unique_name,
            'work_dir': os.path.join('work_dirs', exp_name, unique_name),
            'git_rev': get_git_hash()
        }

        # Ghi ra JSON: configs/generated/<exp_name>/<unique_name>.json
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        exp_name = f'{args.machine}-exp{args.exp}'
        cfgs = generate_experiment_cfgs(args.exp)   # trả về list dict cấu hình (do bạn định nghĩa trong experiments.py)
        # Generate Configs
        for i, cfg in enumerate(cfgs):
            if args.debug:
                # rút ngắn & bật debug: log/val ngắn, debug ảnh UDA (nếu tên có "dacs")
                cfg.setdefault('log_config', {})['interval'] = 10
                cfg['evaluation'] = dict(interval=200, metric='mIoU')
                if 'dacs' in cfg['name']:
                    cfg.setdefault('uda', {})['debug_img_interval'] = 10
                    # cfg.setdefault('uda', {})['print_grad_magnitude'] = True
            
            # Generate Config File
            # Tạo tên duy nhất + work_dir + git_rev
            cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                          f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
            cfg['work_dir'] = os.path.join('work_dirs', exp_name, cfg['name'])
            cfg['git_rev'] = get_git_hash()

            # Điều chỉnh đường dẫn _base_ do file JSON sẽ nằm ở configs/generated/...
            cfg['_base_'] = ['../../' + e for e in cfg['_base_']]

            # Ghi JSON ra configs/generated/<exp_name>/<name>.json
            cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
            os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
            assert not os.path.isfile(cfg_out_file)
            with open(cfg_out_file, 'w') as of:
                json.dump(cfg, of, indent=4)
            config_files.append(cfg_out_file)

    if args.machine == 'local':
        for i, cfg in enumerate(cfgs):
            print('Run job {}'.format(cfg['name']))
            train.main([config_files[i]])   # GỌI HÀM train trong tools/train.py, truyền vào file JSON vừa sinh
            torch.cuda.empty_cache()    #  # giải phóng VRAM giữa các job
    else:
        raise NotImplementedError(args.machine)
