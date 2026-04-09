# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

logger = logging.getLogger(__name__)


# ------------------------------------------------
# ① 一个超轻的全局块，只做全局池化 + 广播回去
# ------------------------------------------------
class LiteGlobalBlock(nn.Module):
    """
    输入/输出: [B, L, C]
    - 把特征还原成 [B,C,H,W]
    - 自适应池化成一个很小的特征 (p×p)，压成一个全局token
    - 用线性+sigmoid生成通道权重，广播回所有token
    这样就有全局感受野，但参数量很小
    """
    def __init__(self, dim, input_resolution, n_global=4, pool=4):
        super().__init__()
        self.H, self.W = input_resolution
        self.dim = dim
        self.pool = pool

        # 把全局token做一点变换
        # C -> C//2 -> C 这一点点参数
        mid = max(dim // 2, 16)
        self.to_global = nn.Sequential(
            nn.Linear(dim, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, dim, bias=False),
        )
        # 生成权重并广播
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, f"LiteGlobalBlock: L={L}, H*W={H*W} 不一致"

        x_map = x.transpose(1, 2).view(B, C, H, W)  # [B,C,H,W]

        # 全局池化成很小的图
        g = F.adaptive_avg_pool2d(x_map, (self.pool, self.pool))  # [B,C,p,p]
        g = g.flatten(2).mean(-1)  # [B,C]
        g = self.to_global(g)      # [B,C]
        w = self.proj(g).unsqueeze(1)  # [B,1,C]

        out = x * w + x
        return out


# ------------------------------------------------
# 下面是你原来那一大坨 BiFormer / BRA 模块
# 我基本不动，只是在系统里插入 LiteGlobalBlock
# ------------------------------------------------

class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: torch.Tensor, r_weight: torch.Tensor, kv: torch.Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
        )
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4,
                 kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo",
                 param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=5, auto_pad=False):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0
        self.scale = qk_scale or self.qk_dim ** -0.5

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1,
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else (lambda x: torch.zeros_like(x))

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        self.router = TopkRouting(qk_dim=self.qk_dim, qk_scale=self.scale, topk=self.topk,
                                  diff_routing=self.diff_routing, param_routing=self.param_routing)
        if self.soft_routing:
            mul_weight = 'soft'
        elif self.diff_routing:
            mul_weight = 'hard'
        else:
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        if self.kv_downsample_mode == 'ada_avgpool':
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        else:
            raise ValueError
        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.size()
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0

        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
        q, kv = self.qkv(x)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])

        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        r_weight, r_idx = self.router(q_win, k_win)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)
        out = out + lepe
        out = self.wo(out)

        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()
        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x


class AttentionLePE(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1,
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else (lambda x: torch.zeros_like(x))

    def forward(self, x):
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, input_resolution, drop_path=0.,
                 layer_scale_init_value=-1, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None,
                 kv_downsample_mode='ada_avgpool', topk=4, param_attention="qkvo",
                 param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False, side_dwconv=5,
                 before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim
        self.input_resolution = input_resolution

        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv,
                                       padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if topk > 0:
            self.attn = BiLevelRoutingAttention(
                dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                qk_scale=qk_scale, kv_per_win=kv_per_win,
                kv_downsample_ratio=kv_downsample_ratio,
                kv_downsample_kernel=kv_downsample_kernel,
                kv_downsample_mode=kv_downsample_mode,
                topk=topk, param_attention=param_attention,
                param_routing=param_routing, diff_routing=diff_routing,
                soft_routing=soft_routing, side_dwconv=side_dwconv,
                auto_pad=auto_pad
            )
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        else:
            self.attn = nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)

        if self.pre_norm:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop_path(self.attn(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))

        x = x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        # 保留 shortcut 的结构
        x = x  # 这里你的原逻辑就是返回 x，shortcut 在上面 attention 里已经加了
        return x


# --------------------- 解码相关 ---------------------
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        x = rearrange(
            x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, embed_dim, num_heads,
                 drop_path_rate=0., layer_scale_init_value=-1, topks=[8, 8, -1, -1],
                 qk_dims=[96, 192, 384, 768], n_win=7,
                 kv_per_wins=[2, 2, -1, -1], kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], kv_downsample_mode='ada_avgpool',
                 param_attention='qkvo', param_routing=False, diff_routing=False,
                 soft_routing=False, pre_norm=True, mlp_ratios=[4, 4, 4, 4],
                 mlp_dwconv=False, side_dwconv=5, qk_scale=None, before_attn_dwconv=3,
                 auto_pad=False, norm_layer=nn.LayerNorm, upsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum([depth]))]
        cur = 0
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                input_resolution=input_resolution,
                drop_path=dp_rates[cur + i],
                layer_scale_init_value=layer_scale_init_value,
                num_heads=num_heads,
                n_win=n_win,
                qk_dim=qk_dims,
                qk_scale=qk_scale,
                kv_per_win=kv_per_wins,
                kv_downsample_ratio=kv_downsample_ratios,
                kv_downsample_kernel=kv_downsample_kernels,
                kv_downsample_mode=kv_downsample_mode,
                topk=topks,
                param_attention=param_attention,
                param_routing=param_routing,
                diff_routing=diff_routing,
                soft_routing=soft_routing,
                mlp_ratio=mlp_ratios,
                mlp_dwconv=mlp_dwconv,
                side_dwconv=side_dwconv,
                before_attn_dwconv=before_attn_dwconv,
                pre_norm=pre_norm,
                auto_pad=auto_pad
            )
            for i in range(depth)
        ])
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SCCSA(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(SCCSA, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out


# ------------------------------------------------
# BRAUnetSystem 里面插入 LiteGlobalBlock
# ------------------------------------------------
class BRAUnetSystem(nn.Module):
    def __init__(self, img_size=256, depth=[3, 4, 8, 3], depths_decoder=[2, 2, 2, 2],
                 in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., use_checkpoint_stages=[],
                 norm_layer=nn.LayerNorm,
                 n_win=7, kv_downsample_mode='identity',
                 kv_per_wins=[2, 2, -1, -1], topks=[8, 8, -1, -1],
                 side_dwconv=5, layer_scale_init_value=-1, qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True, pe=None, pe_stages=[0], before_attn_dwconv=3,
                 auto_pad=False,
                 kv_downsample_kernels=[4, 2, 1, 1], kv_downsample_ratios=[4, 2, 1, 1],
                 mlp_ratios=[4, 4, 4, 4], param_attention='qkvo',
                 final_upsample="expand_first", mlp_dwconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim[0]
        patches_resolution = [img_size // 4, img_size // 4]
        self.num_layers = len(depth)
        self.patches_resolution = patches_resolution
        self.final_upsample = final_upsample

        self.sccsa1 = SCCSA(in_channels=embed_dim[1], out_channels=embed_dim[1])
        self.sccsa2 = SCCSA(in_channels=embed_dim[2], out_channels=embed_dim[2])
        self.sccsa3 = SCCSA(in_channels=embed_dim[3], out_channels=embed_dim[3])

        # 下采样
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dim[i + 1])
                )
            )

        # 编码 stage
        self.stages = nn.ModuleList()
        # 这里新加：每个 stage 一个 lite global
        self.lite_globals = nn.ModuleList()

        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(
                    dim=embed_dim[i],
                    input_resolution=(patches_resolution[0] // (2 ** i),
                                      patches_resolution[1] // (2 ** i)),
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    topk=topks[i],
                    num_heads=nheads[i],
                    n_win=n_win,
                    qk_dim=qk_dims[i],
                    qk_scale=qk_scale,
                    kv_per_win=kv_per_wins[i],
                    kv_downsample_ratio=kv_downsample_ratios[i],
                    kv_downsample_kernel=kv_downsample_kernels[i],
                    kv_downsample_mode=kv_downsample_mode,
                    param_attention=param_attention,
                    param_routing=param_routing,
                    diff_routing=diff_routing,
                    soft_routing=soft_routing,
                    mlp_ratio=mlp_ratios[i],
                    mlp_dwconv=mlp_dwconv,
                    side_dwconv=side_dwconv,
                    before_attn_dwconv=before_attn_dwconv,
                    pre_norm=pre_norm,
                    auto_pad=auto_pad
                ) for j in range(depth[i])]
            )
            self.stages.append(stage)

            # 对应这个尺度的全局块（轻量）
            h_i = patches_resolution[0] // (2 ** i)
            w_i = patches_resolution[1] // (2 ** i)
            self.lite_globals.append(
                LiteGlobalBlock(
                    dim=embed_dim[i],
                    input_resolution=(h_i, w_i),
                    n_global=4,
                    pool=4 if h_i >= 32 else 2
                )
            )

            cur += depth[i]

        # 解码部分
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * embed_dim[self.num_layers - 1 - i_layer],
                                      embed_dim[self.num_layers - 1 - i_layer]) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=embed_dim[self.num_layers - 1 - i_layer],
                    dim_scale=2,
                    norm_layer=norm_layer
                )
            else:
                layer_up = BasicLayer_up(
                    dim=embed_dim[self.num_layers - 1 - i_layer],
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths_decoder[i_layer],
                    embed_dim=embed_dim[self.num_layers - 1 - i_layer],
                    num_heads=nheads[(self.num_layers - 1 - i_layer)],
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=-1,
                    topks=topks[3 - i_layer],
                    qk_dims=qk_dims[3 - i_layer],
                    n_win=n_win,
                    kv_per_wins=kv_per_wins[3 - i_layer],
                    kv_downsample_kernels=[3 - i_layer],
                    kv_downsample_ratios=[3 - i_layer],
                    kv_downsample_mode=kv_downsample_mode,
                    param_attention=param_attention,
                    param_routing=param_routing,
                    diff_routing=diff_routing,
                    soft_routing=soft_routing,
                    pre_norm=pre_norm,
                    mlp_ratios=mlp_ratios[3 - i_layer],
                    mlp_dwconv=mlp_dwconv,
                    side_dwconv=side_dwconv,
                    qk_scale=qk_scale,
                    before_attn_dwconv=before_attn_dwconv,
                    auto_pad=auto_pad,
                    norm_layer=nn.LayerNorm,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(embed_dim[0])
        if self.final_upsample == "expand_first":
            self.up4 = FinalPatchExpand_X4(
                input_resolution=(img_size // 4, img_size // 4),
                dim_scale=4,
                dim=embed_dim[0]
            )
        self.output = nn.Conv2d(in_channels=embed_dim[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # encoder
    def forward_features(self, x):
        x_downsample = []
        # 前三层
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = x.flatten(2).transpose(1, 2)    # [B, L, C]
            x = self.stages[i](x)
            # 新增：插入轻量 global
            x = self.lite_globals[i](x)

            x_downsample.append(x)
            B, L, C = x.shape
            x = x.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
        # 最后一层
        x = self.downsample_layers[3](x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)
        x = self.stages[3](x)
        x = self.lite_globals[3](x)  # 也过一下全局
        return x, x_downsample

    # decoder
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            elif inx == 1:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                B, L, C = x.shape
                x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
                x = self.sccsa3(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            elif inx == 2:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                B, L, C = x.shape
                x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
                x = self.sccsa2(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                B, L, C = x.shape
                x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
                x = self.sccsa1(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W
        if self.final_upsample == "expand_first":
            x = self.up4(x)
            x = x.view(B, 4 * H, 4 * W, -1).permute(0, 3, 1, 2)
            x = self.output(x)
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x


# 这个是外面包一层的 BRAUnet
class BRAUnetHA(nn.Module):
    def __init__(self, img_size=512, in_chans=3, num_classes=1, n_win=8):
        super(BRAUnetHA, self).__init__()
        self.bra_unet = BRAUnetSystem(
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            head_dim=32,
            n_win=n_win,
            embed_dim=[96, 192, 384, 768],
            depth=[2, 2, 8, 2],
            depths_decoder=[2, 8, 2, 2],
            mlp_ratios=[3, 3, 3, 3],
            drop_path_rate=0.2,
            topks=[2, 4, 8, -2],
            qk_dims=[96, 192, 384, 768]
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.bra_unet(x)
        return logits

    def load_from(self):
        pretrained_path = '/home/xxx/biformer_base_best.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = self.bra_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(
                            k, full_dict[k].shape, model_dict[k].shape))
                        del full_dict[k]
            msg = self.bra_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")


if __name__ == '__main__':
    from thop import profile
    import time

    net = BRAUnet(num_classes=2).cuda()
    img = torch.rand(1, 3, 512, 512).cuda()
    flops, params = profile(net, inputs=(img,))
    print('flops:%f' % flops)
    print('params:%f' % params)
    print('flops: %.1f G, params: %.1f M' % (flops / 1e9, params / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(img)
    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))
    print(out.shape)
