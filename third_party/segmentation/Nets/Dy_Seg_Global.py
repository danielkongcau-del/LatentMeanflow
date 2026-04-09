# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('../..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x




class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.down = nn.Conv2d(in_channels=dim,
                              out_channels=2 * dim,
                              kernel_size=2,
                              stride=2,
                              padding=0)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution

        B, HW, C = x.shape

        x = self.norm(x.reshape(B, C, -1).permute(0, 2, 1)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.reshape(B, 2 * C, -1).permute(0, 2, 1)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class UpMoEModule(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, up_channel, skip_channels, out_channels, drop_path_rate=0.1, bilinear=True, ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(up_channel, up_channel, kernel_size=2, stride=2)

        self.module = DecoderModule(
            in_channels=(skip_channels + up_channel),
            out_channels=out_channels,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x1, x2):
        # x1 is skip connection
        # x2 is output of encoder
        if x1.shape[2] == x2.shape[2] and x1.shape[3] == x2.shape[3]:
            pass
        else:
            x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.module(x)


class DecoderModule(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0., ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.moe = ExpertModule(out_channels, drop_path_rate)

    def forward(self, x):
        x = self.proj(x)
        x = self.moe(x)
        return x


class ExpertModule(nn.Module):
    def __init__(self, out_channels, drop_path_rate=0.):
        super().__init__()
        self.Expert = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Dropout(drop_path_rate)
        )

    def forward(self, x):
        return self.Expert(x)


class InverseEmbeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, in_channels=64, out_channels=64, patch_size=4, bilinear=True, drop_path_rate=0.):
        super(InverseEmbeddings, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)

        self.module = ExpertModule(out_channels, drop_path_rate)

    def forward(self, x):
        # x = self.dropout(self.norm(self.up(x)))
        x = self.up(x)
        x = self.module(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)
# =========================
# Step1: 自注意力模块，用于 Qformer 跨模态
# =========================
class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 注意力权重 dropout
        self.attn_drop = nn.Dropout(attn_drop_ratio)

        # 用于输出的新 Key/Value 投影层
        self.cross_key = nn.Linear(dim, dim)
        self.cross_value = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        """
        q: [B, Nq, C] 来自 Qformer 的 Query
        k: [B, Nk, C] 来自池化后的全局特征
        v: [B, Nv, C] 来自池化后的全局特征
        return: 更新后的 (K, V)，表示调制后的全局信息
        """
        q_res = q  # 残差连接用

        B, N_q, C = q.shape
        B, N_k, C = k.shape
        B, N_v, C = v.shape

        # reshape 为多头形式
        q = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,Nq,D]
        k = k.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,Nk,D]
        v = v.reshape(B, N_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,Nv,D]

        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 聚合信息
        x = (attn @ v)  # [B,H,Nq,D]
        x = x.transpose(1, 2).reshape(B, N_q, C)  # [B,Nq,C]

        # 得到更新后的 K,V (相当于全局记忆 token)
        k_new = self.cross_key(x + q_res)  # [B, Nq, C]
        v_new = self.cross_value(x + q_res)  # [B, Nq, C]

        return k_new, v_new

# =========================
# Step2: QFormer 模块
# =========================
class GlobalCrossAttention(nn.Module):
    def __init__(self, dim, Nq, num_heads=8, num_patches=56 * 56, poolstep=8, attn_drop_ratio=0.,
                 ape=True, input_resolution=56):
        super().__init__()
        self.ape = ape
        if self.ape:
            # 可学习的绝对位置编码
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=attn_drop_ratio)

        # 平均池化，获得压缩的全局特征
        self.avg_pool = nn.AvgPool2d(kernel_size=poolstep, stride=poolstep)

        # Qformer 的 Query token
        self.query_tokens = nn.Parameter(torch.zeros(1, Nq, dim))
        trunc_normal_(self.query_tokens, std=0.02)

        # key/value 的卷积投影
        self.key_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        self.value_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

        # 调用自定义的 Self_Attention
        self.attention = Self_Attention(dim, num_heads=num_heads, attn_drop_ratio=attn_drop_ratio)

    def forward(self, featuremap):
        """
        featuremap: [B,C,H,W] Swin backbone 的输入特征图
        return: 调制后的全局 K,V
        """
        B, C, H, W = featuremap.shape

        # Step1: 生成 Q (来自 query_tokens)
        query = self.query_tokens.expand(B, -1, -1)  # [B, Nq, C]
        query = self.pos_drop(query)

        # Step2: Key / Value 来自池化的特征
        pooled = self.avg_pool(featuremap)  # [B,C,H/k,W/k]
        key = self.key_proj(pooled).flatten(2).transpose(1, 2)  # [B, hw, C]
        value = self.value_proj(pooled).flatten(2).transpose(1, 2)  # [B, hw, C]

        # Step3: 自注意力，得到调制后的全局 K,V
        k_new, v_new = self.attention(query, key, value)  # [B,Nq,C], [B,Nq,C]

        return k_new, v_new


# =========================
# Step3: 改造 WindowAttention，支持全局 token 融合
# =========================
class GlobalWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 相对位置 index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2,Wh,Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2,Wh*Ww,Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww,Wh*Ww,2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # qkv投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, global_k=None, global_v=None):
        """
        x: [B*nW, N, C] 每个窗口的局部 token
        global_k, global_v: [B,Nq,C] Qformer 输出的全局 token
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [3, B_, H, N, D]

        # ===== 拼接全局 K,V =====
        if global_k is not None and global_v is not None:
            # 1) 把 global token 复制到每个窗口
            # global_k_rep: [BnW, Nq, C]
            # 假设 B_ = B*nW, 所以要 repeat
            B = global_k.shape[0]
            nW = B_ // B
            global_k = global_k.unsqueeze(1).expand(B, nW, -1, -1).reshape(B_, -1, C)  # [B_,Nq,C]
            global_v = global_v.unsqueeze(1).expand(B, nW, -1, -1).reshape(B_, -1, C)

            # 2) 把 local k/v 和 global k/v 都 reshape 成 multi-head 形式
            # 假设已经得到 local k, v 的多头形式：
            # local_k: [BnW, num_heads, N, head_dim]
            # local_v: [BnW, num_heads, N, head_dim]
            global_k = global_k.reshape(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            global_v = global_v.reshape(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # 3) 拼接 token 维度
            k = torch.cat([k, global_k], dim=2)  # [B_,H,N+Nq,D]
            v = torch.cat([v, global_v], dim=2)

        # 注意力计算
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_,H,N,N+Nq]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [H,N,N]
        attn[:, :, :, :N] = attn[:, :, :, :N] + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class GlobalBlock(nn.Module):
    """Swin Block + Global CrossAttention + WindowAttention"""
    def __init__(self,Nq, poolstep, dim, input_resolution, num_heads,
                 window_size=7, mlp_ratio=4., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # GlobalCrossAttention（Qformer）模块
        self.global_cross_attn = GlobalCrossAttention(
            dim=dim, num_heads=num_heads,
            input_resolution=input_resolution[0],
            Nq = Nq,
            poolstep=poolstep,
        )

        # WindowAttention 模块，支持 global token 融合
        self.window_attn = GlobalWindowAttention(
            dim=dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads
        )

        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        x: [B, H*W, C]
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x_2d = x.view(B, H, W, C).permute(0,3,1,2).contiguous()  # [B,C,H,W] 给 GlobalCrossAttention

        # 1️⃣ 生成全局 token
        global_k, global_v = self.global_cross_attn(x_2d)  # [B,Nq,C]

        # 2️⃣ 窗口划分
        x_windows = window_partition(x.view(B,H,W,C), self.window_size)  # [nW*B, Wh, Ww, C]
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)  # [nW*B, N, C]

        # 3️⃣ WindowAttention + 全局 token
        # x_windows = self.window_attn(x_windows, global_k=global_k, global_v=global_v)

        # 4️⃣ merge windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)  # [B,H,W,C]
        x = x.view(B, H*W, C)

        # 5️⃣ 残差 + FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, poolstep, Nq,dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.,drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            GlobalBlock(poolstep = poolstep,Nq = Nq,
                        dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer,
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_before_downsample = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_before_downsample

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class Dy_Seg_Encoder(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, poolstep=[8,4,2,1],Nq = 4,img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(poolstep=poolstep[i_layer],
                                Nq = Nq,
                                dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,

                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint
                               )

            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def change_feature_shape(self, x, H, W):
        B, L, C = x.shape

        assert L == H * W, f"input feature has wrong size,L: {L}, H:{H}, W: {W} "
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        stages_feature = []
        for layer in self.layers:
            x, x_before_downsampling = layer(x)

            x_reshape = self.change_feature_shape(x_before_downsampling, layer.input_resolution[0],
                                                  layer.input_resolution[1])
            stages_feature.append(x_reshape)

        return stages_feature

    def forward(self, x):
        x = self.forward_features(x)
        return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops



class Dy_Seg_Global(nn.Module):
    def __init__(self, img_size, window_size, embed_dim,
                 n_classes, drop_path_rate=0.1, poolstep=[8,4,2,1], Nq=4,depths=[2,2,6,2]):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.in_chans = 3
        self.patch_size = 4

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        self.encoder = Dy_Seg_Encoder(
            poolstep = poolstep,
            Nq=Nq,
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[4, 8, 16, 32],
            window_size=window_size,
            mlp_ratio=4.,

            drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,

        )
        self.num_features = 512
        self.up1 = UpMoEModule(embed_dim*8, embed_dim*4, embed_dim*4, dpr[0], )
        self.up2 = UpMoEModule(embed_dim*4, embed_dim*2, embed_dim*2, dpr[1], )
        self.up3 = UpMoEModule(embed_dim*2, embed_dim, embed_dim, dpr[2], )

        self.invemb = InverseEmbeddings(patch_size=4, in_channels=embed_dim, out_channels=embed_dim, drop_path_rate=dpr[3])
        self.outc = Out(embed_dim, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)

    def forward(self, x):
        x_list = self.encoder(x)
        # for i in x_list:
        #     print(i.shape)
        x = self.up1(x_list[-1], x_list[-2])
        x = self.up2(x, x_list[-3])
        x = self.up3(x, x_list[-4])
        x = self.invemb(x)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    from thop import profile
    import time

    # print(torch.__version__)
    # net =Dy_Seg_Global(
    #         poolstep=[8,4,2,1],
    #         Nq=5,
    #         img_size=512,
    #         window_size=16,
    #         embed_dim=64,
    #         n_classes=3,
    #         drop_path_rate=0.1,
    #         depths=[2, 2, 6, 2],  #depths=[2, 2, 18, 2],
    #      ).cuda()
    # print(net)
    # image = torch.rand(1, 3, 512, 512).cuda()
    # f, p = profile(net, inputs=(image,))
    # print('flops:%f' % f)
    # print('params:%f' % p)
    # print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))
    # s = time.time()
    # with torch.no_grad():
    #     out = net(image, )
    # print('infer_time:', time.time() - s)
    # print("FPS:%f" % (1 / (time.time() - s)))
    # print(out.shape)
    print(torch.__version__)
    net = Dy_Seg_Global(
            poolstep=[8,4,2,1],
            Nq=5,
            img_size=512,
            window_size=16,
            embed_dim=64,
            n_classes=3,
            drop_path_rate=0.1,
            depths=[2, 2, 6, 2],  #depths=[2, 2, 18, 2],
         ).cuda()

    import torchsummary

    # torchsummary.summary(net)
    print(net)
    image = torch.rand(1, 3, 512, 512).cuda()
    # time_step=torch.tensor([999] * 1, device="cuda")
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # f, p = profile(net, inputs=(image, time_step))

    f, p = profile(net, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)
    # 模拟输入数据，例如一批图像
    batch_size = 2  # 每批处理的图像数量
    num_batches = 80  # 总共处理的批次数量
    input_data = torch.randn(batch_size, 3, 512, 512).cuda()  # 模拟输入图像数据


    # 模拟一个简单的前向传播
    def process_image(image):
        return net(image)


    # 将模型切换到评估模式
    net.eval()
    # 计时开始
    start_time = time.time()
    # 模拟处理多批图像
    with torch.no_grad():
        for batch in range(num_batches):
            output = process_image(input_data)  # 调用模型处理图像
    # 计时结束
    end_time = time.time()
    # 计算吞吐量（每秒处理的图像数量）
    total_images_processed = batch_size * num_batches
    throughput = total_images_processed / (end_time - start_time)
    print(f"模型吞吐量: {throughput:.2f} 图像/秒")