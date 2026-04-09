# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import logging
import math
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from scipy import ndimage

import ml_collections
from torch.nn import Dropout, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger(__name__)

# ================================================================
# 1. config
# ================================================================
def get_b16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_skip = 3
    config.n_classes = 4
    config.activation = 'softmax'
    return config


def get_r50_b16_config():
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
}

# ================================================================
# 2. ResNetV2
# ================================================================
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert 0 < pad < 3
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ================================================================
# 3. Embeddings
# ================================================================
def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0],
                          img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(in_channels, config.hidden_size,
                                       kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)      # B, C, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, features


# ================================================================
# 4. 你的 Global_Block 系列
# ================================================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Self_Attention_Qformer(nn.Module):
    """
    这里是你Qformer里用的那种注意力，我修了残差那一行
    """
    def __init__(self, dim, num_heads=8, attn_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.cross_key = nn.Linear(dim, dim)
        self.cross_value = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        # q: [B, Nq, C]
        q_res = q  # ← 保存原始的 q，用来做残差
        B, N_q, C = q.shape

        q = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)

        # 这里一定要和原始 q_res 加，不是和多头后的 q 加
        k_new = self.cross_key(x + q_res)
        v_new = self.cross_value(x + q_res)
        return k_new, v_new


class GlobalCrossAttention(nn.Module):
    def __init__(self, dim, Nq, num_heads=8, poolstep=4, attn_drop_ratio=0.):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=poolstep, stride=poolstep)
        self.query_tokens = nn.Parameter(torch.zeros(1, Nq, dim))
        trunc_normal_(self.query_tokens, std=0.02)

        self.key_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )
        self.value_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )
        self.attn = Self_Attention_Qformer(dim, num_heads, attn_drop_ratio)

    def forward(self, featuremap):
        B, C, H, W = featuremap.shape
        q = self.query_tokens.expand(B, -1, -1)
        pooled = self.avg_pool(featuremap)
        k = self.key_proj(pooled).flatten(2).transpose(1, 2)
        v = self.value_proj(pooled).flatten(2).transpose(1, 2)
        k_new, v_new = self.attn(q, k, v)
        return k_new, v_new


class GlobalWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, global_k=None, global_v=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if global_k is not None and global_v is not None:
            B = global_k.shape[0]
            nW = B_ // B
            gk = global_k.unsqueeze(1).expand(B, nW, -1, -1).reshape(B_, -1, C)
            gv = global_v.unsqueeze(1).expand(B, nW, -1, -1).reshape(B_, -1, C)
            gk = gk.view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            gv = gv.view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat([k, gk], dim=2)
            v = torch.cat([v, gv], dim=2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn[:, :, :, :N] = attn[:, :, :, :N] + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalBlockViT(nn.Module):
    def __init__(self, dim, grid_size, num_heads=8, window_size=4,
                 Nq=5, poolstep=4, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.H = grid_size
        self.W = grid_size
        self.window_size = window_size  # 显式写出来
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.global_cross = GlobalCrossAttention(dim, Nq, num_heads, poolstep)
        self.window_attn = GlobalWindowAttention(dim,
                                                 window_size=to_2tuple(window_size),
                                                 num_heads=num_heads)
        self.mlp = MlpBlock(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.H * self.W, "token数和设定的网格不一致"

        shortcut = x
        x = self.norm1(x)
        x_2d = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()
        global_k, global_v = self.global_cross(x_2d)

        x_windows = window_partition(x.view(B, self.H, self.W, C), self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        x_windows, attn = self.window_attn(x_windows, global_k=global_k, global_v=global_v)

        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, self.H, self.W)
        x = x.view(B, N, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


# ================================================================
# 5. Encoder 用新的 Block
# ================================================================
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x); x = self.act_fn(x); x = self.dropout(x)
        x = self.fc2(x); x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis, grid_size):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = GlobalBlockViT(
            dim=config.hidden_size,
            grid_size=grid_size,
            num_heads=config.transformer["num_heads"],
            window_size=4,
            Nq=5,
            poolstep=4,
            drop_path=0.
        )
        self.ffn = Mlp(config)
        self.vis = vis

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, attn_map = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, attn_map

    def load_from(self, weights, n_block):
        # 我们只加载 MLP 和 LN
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            mlp_weight_0 = np2th(weights[pjoin(ROOT, "MlpBlock_3", "Dense_0", "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, "MlpBlock_3", "Dense_1", "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, "MlpBlock_3", "Dense_0", "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, "MlpBlock_3", "Dense_1", "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, "LayerNorm_0", "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, "LayerNorm_0", "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, "LayerNorm_2", "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, "LayerNorm_2", "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, img_size):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()

        # 计算 token grid
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"][0]
        else:
            grid_size = img_size // config.patches["size"][0]

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, grid_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


# ================================================================
# 6. Transformer wrapper
# ================================================================
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, img_size)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


# ================================================================
# 7. Decoder & Head
# ================================================================
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size, stride=stride,
                         padding=padding, bias=not use_batchnorm)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, 3, padding=1)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if config.n_skip != 0:
            skip_channels = config.skip_channels
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.shape
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < len(features)) else None
            x = decoder_block(x, skip)
        return x


# ================================================================
# 8. VisionTransformer seg model
# ================================================================
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=512, num_classes=4, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.n_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits


def config_transunetHA(img_size=512, n_classes=4):
    name = 'R50-ViT-B_16'
    cfg = CONFIGS[name]
    if name.find('R50') != -1:
        cfg.patches.grid = (img_size // 16, img_size // 16)
    cfg.n_classes = n_classes
    return VisionTransformer(cfg, img_size=img_size, num_classes=n_classes, zero_head=True, vis=False)


if __name__ == "__main__":
    from thop import profile
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = config_transunet(img_size=512, n_classes=4).to(device)
    img = torch.randn(1, 3, 512, 512).to(device)

    flops, params = profile(net, inputs=(img,))
    print("flops:", flops / 1e9, "G")
    print("params:", params / 1e6, "M")

    t0 = time.time()
    with torch.no_grad():
        out = net(img)
    print("infer_time:", time.time() - t0)
    print("out:", out.shape)
