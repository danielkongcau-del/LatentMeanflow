import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, to_2tuple, DropPath


# -------------------- 基础卷积模块 --------------------
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    """ Depthwise Convolution """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups is None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, bias=True)

    def forward(self, x):
        return self.depthwise(x)


# -------------------- U 形卷积编码器（按 512 输入） --------------------
class UEncoder(nn.Module):
    def __init__(self):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.res2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.res3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.res4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.res5 = DoubleConv(256, 512)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []

        x = self.res1(x)          # 512 -> 512
        features.append(x)        # [0] (B,32,512,512)
        x = self.pool1(x)         # 256

        x = self.res2(x)          # 256
        features.append(x)        # [1] (B,64,256,256)
        x = self.pool2(x)         # 128

        x = self.res3(x)          # 128
        features.append(x)        # [2] (B,128,128,128)
        x = self.pool3(x)         # 64

        x = self.res4(x)          # 64
        features.append(x)        # [3] (B,256,64,64)
        x = self.pool4(x)         # 32

        x = self.res5(x)          # 32
        features.append(x)        # [4] (B,512,32,32)
        x = self.pool5(x)         # 16

        features.append(x)        # [5] (B,512,16,16)
        return features


# ============================================================
# 你的 Global_Block 系列：Self_Attention / GlobalCross / Window
# ============================================================
class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.cross_key = nn.Linear(dim, dim)
        self.cross_value = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        q_res = q
        B, N_q, C = q.shape

        q = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N_q, C)

        k_new = self.cross_key(x + q_res)
        v_new = self.cross_value(x + q_res)
        return k_new, v_new


class GlobalCrossAttention(nn.Module):
    def __init__(self, dim, Nq, num_heads=8, num_patches=56 * 56, poolstep=8,
                 attn_drop_ratio=0., ape=True, input_resolution=56):
        super().__init__()
        self.ape = ape
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=attn_drop_ratio)
        self.avg_pool = nn.AvgPool2d(kernel_size=poolstep, stride=poolstep)

        self.query_tokens = nn.Parameter(torch.zeros(1, Nq, dim))
        trunc_normal_(self.query_tokens, std=0.02)

        self.key_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        self.value_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

        self.attention = Self_Attention(dim, num_heads=num_heads, attn_drop_ratio=attn_drop_ratio)

    def forward(self, featuremap):
        B, C, H, W = featuremap.shape

        query = self.query_tokens.expand(B, -1, -1)
        query = self.pos_drop(query)

        pooled = self.avg_pool(featuremap)
        key = self.key_proj(pooled).flatten(2).transpose(1, 2)
        value = self.value_proj(pooled).flatten(2).transpose(1, 2)

        k_new, v_new = self.attention(query, key, value)
        return k_new, v_new


class GlobalWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
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
        attn = (q @ k.transpose(-2, -1))

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
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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


class GlobalBlock(nn.Module):
    """
    输入: [B, H*W, C]
    """
    def __init__(self, Nq, poolstep, dim, input_resolution, num_heads,
                 window_size=8, mlp_ratio=4., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.global_cross_attn = GlobalCrossAttention(
            dim=dim, num_heads=num_heads,
            input_resolution=input_resolution[0],
            Nq=Nq,
            poolstep=poolstep,
        )

        self.window_attn = GlobalWindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        global_k, global_v = self.global_cross_attn(x_2d)

        x_windows = window_partition(x.view(B, H, W, C), self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        x_windows = self.window_attn(x_windows, global_k=global_k, global_v=global_v)

        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GlobalBlock2D(nn.Module):
    """
    把 (B,C,H,W) 包一层喂进 GlobalBlock
    """
    def __init__(self, Nq, poolstep, dim, H, W, num_heads, window_size=8):
        super().__init__()
        self.H = H
        self.W = W
        self.block = GlobalBlock(
            Nq=Nq,
            poolstep=poolstep,
            dim=dim,
            input_resolution=(H, W),
            num_heads=num_heads,
            window_size=window_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # B,HW,C
        x_seq = self.block(x_seq)
        x = x_seq.transpose(1, 2).view(B, C, H, W)
        return x


# -------------------- 解码块 --------------------
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
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
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


# -------------------- Inter / Multi-scale Attention --------------------
class MLP_simple(nn.Module):
    def __init__(self, dim):
        super(MLP_simple, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiScaleAtten(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

    def forward(self, x):
        B, num_blocks, _, _, C = x.shape
        qkv = self.qkv_linear(x).reshape(
            B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head
        ).permute(4, 0, 1, 2, 5, 3, 6).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
        atten_value = self.proj(atten_value)
        return atten_value


class InterTransBlock(nn.Module):
    def __init__(self, dim):
        super(InterTransBlock, self).__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = MultiScaleAtten(dim)
        self.FFN = MLP_simple(dim)

    def forward(self, x):
        h = x
        x = self.SlayerNorm_1(x)
        x = self.Attention(x)
        x = h + x

        h = x
        x = self.SlayerNorm_2(x)
        x = self.FFN(x)
        x = h + x
        return x


class SpatialAwareTrans(nn.Module):
    def __init__(self, dim=128, num=1):   # 你这里原来就是想要 4 个尺度一起搞
        super(SpatialAwareTrans, self).__init__()
        self.ini_win_size = 2
        self.channels = [128, 256, 512, 512]
        self.dim = dim
        self.depth = 4
        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num

        for i in range(self.depth):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))
        for i in range(self.depth):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.group_attention = nn.Sequential(*[InterTransBlock(dim) for _ in range(self.num)])
        self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]

    def forward(self, x):
        # x: list of 4 tensors (B,C,H,W) -> 128,64,32,16
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(
                0, 1, 3, 2, 4, 5
            ).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            x[j] = item
        x = tuple(x)
        x = torch.cat(x, dim=-2)

        for i in range(self.num):
            x = self.group_attention[i](x)

        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)

        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(
                0, 1, 3, 2, 4, 5
            ).contiguous().reshape(B, num_blocks * win_size, num_blocks * win_size, C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item
        return x


# -------------------- Transformer 编码器：用 GlobalBlock2D --------------------
class TransEncoder(nn.Module):
    def __init__(self):
        super(TransEncoder, self).__init__()
        # 四个尺度
        self.size = [128, 64, 32, 16]
        self.channels = [128, 256, 512, 512]
        self.poolsteps = [8, 4, 2, 1]
        self.num_heads = [4, 4, 8, 8]  # 可以自己调，反正能整除

        # 四个 stage 的 global block
        self.stage1 = GlobalBlock2D(Nq=5, poolstep=self.poolsteps[0], dim=self.channels[0],
                                    H=self.size[0], W=self.size[0],
                                    num_heads=self.num_heads[0], window_size=8)
        self.stage2 = GlobalBlock2D(Nq=5, poolstep=self.poolsteps[1], dim=self.channels[1],
                                    H=self.size[1], W=self.size[1],
                                    num_heads=self.num_heads[1], window_size=8)
        self.stage3 = GlobalBlock2D(Nq=5, poolstep=self.poolsteps[2], dim=self.channels[2],
                                    H=self.size[2], W=self.size[2],
                                    num_heads=self.num_heads[2], window_size=8)
        self.stage4 = GlobalBlock2D(Nq=5, poolstep=self.poolsteps[3], dim=self.channels[3],
                                    H=self.size[3], W=self.size[3],
                                    num_heads=self.num_heads[3], window_size=8)

        # 下采样，把上一层变成下一层的通道
        self.downlayers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.downlayers.append(
                ConvBNReLU(self.channels[i], self.channels[i] * 2, 2, 2, padding=0)
            )

        self.squeelayers = nn.ModuleList()
        for i in range(len(self.channels) - 2):
            self.squeelayers.append(
                nn.Conv2d(self.channels[i] * 4, self.channels[i] * 2, 1, 1)
            )
        self.squeeze_final = nn.Conv2d(self.channels[-1] * 3, self.channels[-1], 1, 1)

    def forward(self, x):
        # x 是 UEncoder 的 features: [512,256,128,64,32,16]
        _, _, feature0, feature1, feature2, feature3 = x  # 128,64,32,16

        # 128x128, 128c -> GlobalBlock
        feature0_trans = self.stage1(feature0)
        feature0_trans_down = self.downlayers[0](feature0_trans)  # 64x64, 256c

        # 64x64
        feature1_in = torch.cat((feature1, feature0_trans_down), dim=1)  # 256 + 256 = 512
        feature1_in = self.squeelayers[0](feature1_in)                   # -> 256
        feature1_trans = self.stage2(feature1_in)
        feature1_trans_down = self.downlayers[1](feature1_trans)         # -> 32x32, 512

        # 32x32
        feature2_in = torch.cat((feature2, feature1_trans_down), dim=1)  # 512 + 512 = 1024
        feature2_in = self.squeelayers[1](feature2_in)                   # -> 512
        feature2_trans = self.stage3(feature2_in)
        feature2_trans_down = self.downlayers[2](feature2_trans)         # -> 16x16, 1024

        # 16x16
        feature3_in = torch.cat((feature3, feature2_trans_down), dim=1)  # 512 + 1024 = 1536
        feature3_in = self.squeeze_final(feature3_in)                    # -> 512
        feature3_trans = self.stage4(feature3_in)

        return [feature0_trans, feature1_trans, feature2_trans, feature3_trans]


# -------------------- 并行编码器：卷积 + Transformer --------------------
class ParallEncoder(nn.Module):
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()
        self.Encoder2 = TransEncoder()
        self.num_module = 4
        self.fusion_list = [128, 256, 512, 512]
        self.inter_trans = SpatialAwareTrans(dim=128)

        self.squeelayers = nn.ModuleList()
        for i in range(self.num_module):
            self.squeelayers.append(
                nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], 1, 1)
            )

    def forward(self, x):
        skips = []
        features = self.Encoder1(x)               # 6 个尺度
        feature_trans = self.Encoder2(features)   # 4 个尺度: 128,64,32,16
        feature_trans = self.inter_trans(feature_trans)

        # 原分支的高分辨率特征
        skips.extend(features[:2])

        for i in range(self.num_module):
            skip = self.squeelayers[i](torch.cat((feature_trans[i], features[i + 2]), dim=1))
            skips.append(skip)
        return skips
        # 顺序: [512x512(32), 256x256(64), 128x128(128), 64x64(256), 32x32(512), 16x16(512)]


# -------------------- 总模型 --------------------
class ScaleFormerHA(nn.Module):
    def __init__(self, num_classes):
        super(ScaleFormerHA, self).__init__()
        self.p_encoder = ParallEncoder()

        self.decoder1 = DecoderBlock(512 + 512, 512)  # 16->32
        self.decoder2 = DecoderBlock(512 + 256, 256)  # 32->64
        self.decoder3 = DecoderBlock(256 + 128, 128)  # 64->128
        self.decoder4 = DecoderBlock(128 + 64, 64)    # 128->256
        self.decoder_final = DecoderBlock(64 + 32, 32)  # 256->512

        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder_skips = self.p_encoder(x)
        # encoder_skips: [512,256,128,64,32,16]

        x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])  # 16 -> 32
        x2_up = self.decoder2(x1_up,        encoder_skips[-3])      # 32 -> 64
        x3_up = self.decoder3(x2_up,        encoder_skips[-4])      # 64 -> 128
        x4_up = self.decoder4(x3_up,        encoder_skips[-5])      # 128 -> 256
        x_final = self.decoder_final(x4_up, encoder_skips[0])       # 256 -> 512

        logits = self.segmentation_head(x_final)
        return logits


if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)

    net = ScaleFormer(num_classes=2, ).cuda()
    print(net)
    image = torch.rand(1, 3, 512, 512).cuda()
    f, p = profile(net, inputs=(image,))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))
    s = time.time()
    with torch.no_grad():
        out = net(image, )
    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))
    print(out.shape)
