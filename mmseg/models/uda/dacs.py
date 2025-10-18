# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs
"""
DACS: Domain Adaptive Semantic Segmentation via Cross-Domain Mixing and
Self-Training
Mở rộng cơ chế train của MMSegmentation để:
1. Kết hợp domain source (label) và target (no label) trong cùng một bước huận luyện
2. Sử dụng mô hình (teacher) để tạo nhãn giả cho domain target
3. Trộn ảnh source và target (class-mix) -> giúp giảm domain gap;
4. Thêm regularization bằng feature distance loss (FDist) để mô hình không quên đặc trưng ImageNet.

ClassMix:
Giả sử bạn có:

Ảnh source có nhãn thật ys
Ảnh target có pseudo label y^t
Ta tạo Mask M cho từng ảnh:
Mc = 1 nếu pixel đó thuộc class có trong ảnh source
Mc = 0 nếu pixel đó lấy từ target
Sau đó trộn ảnh và nhãn:
xmix = M * xs + (1 - M) * xt
ymix = M * ys + (1 - M) * y^t
"""

"""
1. UDA - Unsupervised Domain Adaptation
Đây là bài toán học thích nghi miền không giám sát, mục tiêu là:
Huấn luyện một mô hình phân đoạn ảnh có nhãn trong source domain (ví dụ GTA5 - ảnh mô phỏng),
nhưng áp dụng tốt cho target domain (ví dụ Cityscapes - ảnh thật), mà target không có nhãn.
→ UDA giúp mô hình giảm domain gap giữa hai miền dữ liệu.
2. Self-training
Là chiến lược tự huấn luyện (self-training) dựa trên pseudo label.
Cụ thể:
Dùng mô hình (hoặc bản EMA - Exponential Moving Average) để dự đoán nhãn cho ảnh target → gọi là pseudo label.
Chọn những pixel có độ tin cậy cao (probability > threshold).
Huấn luyện lại mô hình (student) bằng các pseudo label đó, cùng với dữ liệu thật (source).
→ Vòng lặp này gọi là self-training loop, và là nền tảng của DACS / DAFormer.
3. ImageNet Feature Distance (FD)
Đây là thành phần bổ sung quan trọng trong mô hình DAFormer / DACS.
Khi mô hình được huấn luyện chỉ bằng dữ liệu source + pseudo target, nó dễ bị “catastrophic forgetting” — quên mất những đặc trưng phổ quát đã học từ pretraining ImageNet.
Vì vậy, họ thêm một Loss gọi là Feature Distance Loss (FD) để giữ ổn định đặc trưng ImageNet trong backbone.

dacs.py
👉 Một thuật toán huấn luyện domain adaptation (UDA)
👉 Sử dụng self-training bằng pseudo label
👉 Và bổ sung ImageNet Feature Distance loss để tránh quên đặc trưng pretrain.
"""


"""
                  ┌───────────────────────────────────────────┐
                  │         UDA Self-Training Framework        │
                  └───────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Các thành phần chính                       │
├────────────────────────────────────────────────────────────────────┤
│ 1️⃣ Source Supervised Branch                                        │
│   - Input: Ảnh nguồn (x_s), nhãn thật (y_s)                        │
│   - Loss: Cross Entropy Loss                                        │
│                                                                    │
│ 2️⃣ Target Pseudo-Label Branch                                     │
│   - Input: Ảnh đích (x_t)                                          │
│   - Mô hình EMA Teacher sinh pseudo-label                          │
│   - Chỉ lấy pixel có độ tin cậy cao (p_max > threshold)            │
│   - Loss: Cross Entropy (trọng số theo confidence)                 │
│                                                                    │
│ 3️⃣ ClassMix                                                      │
│   - Trộn ảnh source và target theo mask từng class                 │
│   - Mixed Image = M⊙x_s + (1−M)⊙x_t                               │
│   - Mixed Label = M⊙y_s + (1−M)⊙ŷ_t                                │
│   - Loss: Cross Entropy (supervised + pseudo)                      │
│                                                                    │
│ 4️⃣ ImageNet Feature Distance (FD)                                │
│   - So sánh feature backbone của student và backbone pretrained    │
│     từ ImageNet:                                                   │
│       L_FD = λ·‖F_student − F_ImageNet‖₂                           │
│   - Vai trò: Giữ ổn định đặc trưng pretrained                     │
│                                                                    │
│ 5️⃣ EMA Teacher Update                                            │
│   - θ_ema ← αθ_ema + (1−α)θ_student                               │
│   - Teacher sinh pseudo label ổn định hơn                         │
└────────────────────────────────────────────────────────────────────┘

Tổng loss toàn bộ hệ thống:
   L_total = L_source + L_mix + λ_FD·L_FD

"""
"""
Class-Mix: class nào ưu tiên thấp hơn thì bỏ qua, ghép cùng kích thước theo config.
Họ tạo 1 mask duy nhất cho mỗi ảnh (get_class_masks()),
Mask đó là hợp (union) của tất cả class được chọn từ source,
Pixel nào không nằm trong mask → lấy từ target.

Nếu thuộc class được chọn → dùng pixel source
Nếu không → dùng pixel target

class mask -> Lấy từ source. Còn lại lấy từ target.
"""

# https://www.notion.so/vuongcris4/DACS-DAformer-28f16baeeae98002979cfff32ad12e33


import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

# Registry modue để config gọi tới class mà k cần import trực tiếp.
# LUỒNG UDADecorator
# https://www.notion.so/vuongcris4/Lu-ng-UDADecorator-28f16baeeae98089ad63d4017100e12c
"""
- **`self.model` (Student):** Là một object `EncoderDecoder`. Nó được tạo bởi lớp cha `UDADecorator`.
- **`self.ema_model` (Teacher):** Là một object `EncoderDecoder` khác. Nó được tạo bởi chính `DACS`.
- **`self.imnet_model` (Reference):** Là một object `EncoderDecoder` thứ ba. Nó cũng được tạo bởi `DACS`.
"""
@UDA.register_module()
class DACS(UDADecorator):
    """
    DACS Is-A UDADecorator
    """

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']   # từ config daformer đưa xuống.
        self.alpha = cfg['alpha']       # EMA, teacher update
        self.pseudo_threshold = cfg['pseudo_threshold'] # ngưỡng tin cậy cho pseudo-label
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda'] # Trọng số FD
        self.fdist_classes = cfg['imnet_feature_dist_classes']  # list thing classes
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio'] # > r, trung bình onehot labels phải > r
        self.enable_fdist = self.fdist_lambda > 0   # trọng số FD > 0 thì bật FD
        self.mix = cfg['mix']   # class-mix
        self.blur = cfg['blur'] # augmentation blur trong strong_transform
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}

        # Build teacher (EMA) + (tuỳ chọn) ImageNet model cho FD
        ema_cfg = deepcopy(cfg['model'])    # ← chính là model từ _base_/models/...
        self.ema_model = build_segmentor(ema_cfg)   # toàn bộ cấu hình segmentor (MiT-B3 + DAFormer head) lấy từ

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))  # MiT-B3, ImageNet pretrained
        else:
            self.imnet_model = None

    # teacher
    def get_ema_model(self):
        return get_module(self.ema_model)

    # backbone encoder
    def get_imnet_model(self):
        return get_module(self.imnet_model)

    # load pretrained
    def _init_ema_weights(self):
        # teacher model không bao giờ được huấn luyện bằng backpropagation.
        for param in self.get_ema_model().parameters():
            param.detach_()
        # Lấy ra danh sách tất cả các tham số (trọng số, bias...) của student model (mp) và teacher model (mcp). 
        # Cả hai model có cùng kiến trúc nên hai danh sách này sẽ có cùng độ dài và thứ tự tương ứng.
        mp = list(self.get_model().parameters())    # student
        mcp = list(self.get_ema_model().parameters())   # teacher
        # sao chép trọng số từ student sang teacher
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha) # iter = 0 -> alpha_teacher = 0, iter = lớn -> alpha_teacher = 1
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)   # viết tắt để gọi phương thức forward() của class DACS. (gọi forward_train().)
        optimizer.step()    # cập nhật trọng số của self.model

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        "Dòng cuối cùng chỉ đơn giản là định dạng lại đầu ra theo đúng chuẩn mà engine huấn luyện mong đợi, bao gồm log_vars để hiển thị và num_samples để tính trung bình các chỉ số."
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    # TÍNH FD
    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2 # feature map của student model trừ đi feature map của ImageNet model.
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')

        # (Batch, Channels, Height, Width). Tại mỗi pixel (h, w), ta có một vector đặc trưng (feature vector) với độ dài là Channels.
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)    # Tính toán chuẩn L2 (L2-norm), hay còn gọi là khoảng cách Euclid.
        # pw_feat_dist là một tensor mới có kích thước (Batch, Height, Width). Mỗi giá trị trong tensor này là một con số, 
        # đại diện cho "mức độ khác biệt" của feature tại pixel tương ứng. "pw" ở đây có thể hiểu là "pixel-wise" (từng pixel).

        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')

            # loại bỏ chiều 1
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]    # mask là một tensor boolean (True/False) có kích thước (Batch, Height, Width), nó đánh dấu True ở những pixel thuộc các lớp chúng ta quan tâm.
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)


    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log


    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
