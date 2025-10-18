# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='DACS',    # gọi class DACS qua registry @UDA.register_module()
    alpha=0.99,         #  EMA momentum (teacher update)
    pseudo_threshold=0.968, # ngưỡng tin cậy cho pseudo-label
    pseudo_weight_ignore_top=0,  # bỏ trọng số pseudo ở mép trên (px)
    pseudo_weight_ignore_bottom=0,  # bỏ trọng số pseudo ở mép dưới (px)
    imnet_feature_dist_lambda=0,    # 0 = tắt ImageNet Feature Distance (FD)
    imnet_feature_dist_classes=None,    # nếu bật FD, ràng buộc theo class nào
    imnet_feature_dist_scale_min_ratio=None,    # điều kiện downscale label khi tạo mask cho FD
    mix='class',    # bật ClassMix (trộn theo mask class)
    blur=True,  # augmentation blur trong strong_transform
    color_jitter_strength=0.2,  # độ mạnh jitter
    color_jitter_probability=0.2,   # xác suất jitter
    debug_img_interval=1000,    # mỗi N iter lưu ảnh debug mix
    print_grad_magnitude=False, # in chuẩn độ lớn gradient (debug)
)
use_ddp_wrapper = True
