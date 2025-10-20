# Obtained from: https://github.com/NVlabs/SegFormer
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

# https://www.notion.so/vuongcris4/SegformerEncoder-29116baeeae9804d8111ee16b2f10065
# OverlapPatchEmbed -> Block (Attention + Mlp) -> LayerNorm + Residual Connection
class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x) # (B, N, C_in) -> → (B, N, C_hidden)
        x = self.dwconv(x, H, W) # (B, N, C_hidden) -> (B, N, C_hidden)    # DWConv giúp tăng cường đặc trưng không gian trong từng kênh
        x = self.act(x) # Gelu
        x = self.drop(x)
        x = self.fc2(x) # (B, N, C_hidden) → (B, N, C_out)
        x = self.drop(x)
        return x    # (B, N, C_out)

"""
Trong mô hình gốc Vision Transformer (ViT):
Attention có chi phí O(N²), vì Q phải tương tác với tất cả K token (N = HxW).
Trong MiT (SegFormer backbone):
👉 họ dùng Spatial-Reduction Attention (SRA) — chính là efficient version này.
"""
# giúp mô hình nhìn toàn ảnh với chi phí rẻ hơn ViT gốc
class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio    # Nén sr_ratio lần
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    """
    VD: sr_ratio = 4 → nén K,V còn N/4
    C=8, num_heads=2 ⇒ head_dim = 4
    """ 
    # VD input x.shape = [B,N,C] = [1, 3136, 64],    num_heads = 8, head_dim = C/num_heads = 8
    def forward(self, x, H, W): # x = (B, N=H*W, C)     with C = num_heads * head_dim
        # [1, 3136, 64]
        B, N, C = x.shape # B là batch size, N là số token (H*W), C là số chiều embedding (dim)
        # [1, 8, 3136, 8] -> [B, N, num_heads, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous() # (B, num_heads, N, head_dim).  q là Query thu được bằng chiếu tuyến tính học được từ x 

        if self.sr_ratio > 1:   # x_ dùng để sinh ra K / R, V / R
            # [1, 64, 56, 56]
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)    # (B,C,H,W)
            # [1, 64, 14, 14] -> [1, 64, 196] -> [1, 196, 64]
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()    # Conv2d stride=sr_ratio ⇒ (B, C, H/R, W/R) -> (B, N/R^2, C)
            x_ = self.norm(x_)  # (B, N/R^2, C)
            # (B, N/R^2, C) -> (B, N/R^2, 2C) -> reshape(B, -1(Nk), 2(K_V), num_heads, head_dim) -> (2(K_V), B, num_heads, Nk, head_dim), Với Nk = N/R^2
            # [1, 196, 64] -> [1, 196, 128] -> [1, 196, 2, 8, 8] -> [2, 1, 8, 196, 8]
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()    # (2, B, num_heads, Nk, head_dim)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1] # each: (B, num_heads, Nk, head_dim).     with Nk = N/R

        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, Nk) = (B, num_heads, N, Nk)
        # [1, 8, 196, 8] @ [1, 8, 8, 196] = [1, 8, 196, 196]
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale  # (B, num_heads, N, N/R^2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # dropout

        # (B, num_heads, N, N/R^2) @ (B, num_heads, Nk, head_dim) = (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)    # (B,N,C), Context vectors
        x = self.proj(x)    # Mỗi hàng x[b, i, :] là context vector (ngữ cảnh tổng hợp) của token thứ i trong batch b.
        x = self.proj_drop(x)

        return x    # (B, N, C)


class Block(nn.Module):

    def __init__(self,
                 dim,   # số kênh đầu vào (C).
                 num_heads, # số head trong Multi-head Attention.
                 mlp_ratio=4.,  # tỉ lệ ẩn trong MLP (thường 4×dim).
                 qkv_bias=False,    # điều chỉnh khi tính Q, K, V.
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):   # hệ số giảm chiều không gian trong Efficient Attention (giảm độ phức tạp).
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp( # MLP gồm 2 lớp fully connected với kích thước ẩn là mlp_hidden_dim, nếu k cho out_features thì mặc định out = in
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # (B, N, C)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))   # (B, N, C)

        return x

# biến ảnh thành chuỗi token có tính cục bộ
class OverlapPatchEmbed(nn.Module):
    # input x: tensor ảnh (B, C, H, W), trong đó B là batch size, C là số kênh (ví dụ: 3 cho RGB), H và W là chiều cao và chiều rộng của ảnh.
    # output: x_embed (B, N==H*W, C==embed_dim), H, W
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # (224, 224)
        patch_size = to_2tuple(patch_size)  # (7, 7)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,   # 3 kênh RGB
            embed_dim,  # số chiều embedding (C)
            kernel_size=patch_size, # mỗi patch 7x7
            stride=stride,  # overlap vì stride nhỏ hơn kernel
            padding=(patch_size[0] // 2, patch_size[1] // 2))   # giữ biên khi convolution
        self.norm = nn.LayerNorm(embed_dim)

    # input x: tensor ảnh (B, C, H, W), trong đó B là batch size, C là số kênh (ví dụ: 3 cho RGB), H và W là chiều cao và chiều rộng của ảnh.
    # output: x_embed (B, N==H*W, C==embed_dim), H, W
    def forward(self, x):   # x = (1, 3, 224, 224)
        x = self.proj(x)    # [1, 64, 56, 56]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()   # [1, 64, 3136] -> [1, 3136, 64]    # 3136 tokens, mỗi token d = 64, mỗi token biểu diễn thông tin của một vùng 7×7 pixel, 3 channels
        x = self.norm(x)    

        return x, H, W  # H, W giảm stride lần


@BACKBONES.register_module()
class MixVisionTransformer(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 style=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze_patch_embed=False):
        super().__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(  # input x, ouutput x_embed (B, N==H*W, C==embed_dim), H, W
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
        Block(
            dim=embed_dims[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i],
            norm_layer=norm_layer,
            sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) \
        #     if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained is None:
            logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # Mỗi stage: OverlapPatchEmbed (downsample) → danh sách Block (Attention + MLP) → LayerNorm → reshape về B,C,H,W.
    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)  # x_embed (B, N==H*W, C==embed_dim), H, W
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x) # (B, N, C)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # B,H,W,C -> B,C,H,W
        outs.append(x)  # F1

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)  # F2

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x    # trả về 4 feature maps (stage 1→4) cho decoder.

# Depth-Wise Conv
class DWConv(nn.Module):
    """
    Đây là depthwise convolution:
    in_channels = out_channels = dim
    kernel_size = 3     stride = 1
    padding = 1 → giữ nguyên kích thước HxW
    groups = dim → nghĩa là mỗi kênh được convolution riêng biệt (thay vì trộn giữa các kênh như conv thông thường)
    👉 Vì thế, mỗi kênh đầu ra chỉ phụ thuộc vào một kênh đầu vào tương ứng, không có cross-channel mixing.
    Nó chỉ học đặc trưng không gian trong từng kênh → hiệu quả và rẻ.
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)   # 3x3, stride=1, padding=1, có dim filter

    def forward(self, x, H, W):
        B, N, C = x.shape   # B, N, C từ attention trước đó
        x = x.transpose(1, 2).contiguous().view(B, C, H, W) # (B, C, H, W)
        x = self.dwconv(x)  # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2).contiguous()   # (B, C, N) -> (B, N, C)

        return x


@BACKBONES.register_module()
class mit_b0(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b1(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1], 
            **kwargs)


@BACKBONES.register_module()
class mit_b2(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b3(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b4(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


"""
"backbone": {
            "type": "mit_b5",
            "style": "pytorch",
            "drop_path_rate": 0.1,
            "ema_drop_path_rate": 0.0
        },
"""
@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,   # giảm 4 lần kích thước ảnh, chỉ kí hiệu, k ảnh hướng tới code
            embed_dims=[64, 128, 320, 512], # Số kênh đầu ra (C) cho 4 stage:  Càng sâu → càng nhiều kênh, mô tả đặc trưng trừu tượng hơn.
            num_heads=[1, 2, 5, 8], # Số multi-head attention heads ở mỗi stage. Stage 1 dùng 1 head (feature còn thô, ít kênh). Stage 4 dùng 8 head (feature giàu kênh hơn). → Mỗi head học quan hệ không gian riêng biệt.
            mlp_ratios=[4, 4, 4, 4],    # Trong MLP: hidden_dim = in_dim × mlp_ratio = 4×C.  → Tăng năng lực biểu diễn của MLP mà vẫn giữ ổn định.
            qkv_bias=True,  # qkv_bias=True. Thêm bias vào các lớp Linear tạo Q, K, V trong attention. Cải thiện hiệu năng một chút, phổ biến trong ViT/Transformer.
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],     # OverlapPatchEmbed → [Block × depth] → LayerNorm
            sr_ratios=[8, 4, 2, 1], # Hệ số Spatial Reduction trong Efficient Self-Attention:
            **kwargs)
