from math import sqrt
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple


# =============================
# 小工具
# =============================
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# =============================
# 你的注意力机制相关模块
# =============================
class Self_Attention(nn.Module):
    """
    Qformer 用的自注意力：用可学习的 query 去看池化后的全局特征
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
        """
        q: [B, Nq, C]
        k: [B, Nk, C]
        v: [B, Nv, C]
        """
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
    """
    从整张特征图里池化出一组全局 token，再用 query 读它
    """
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

        pooled = self.avg_pool(featuremap)   # [B,C,H/p,W/p]
        key = self.key_proj(pooled).flatten(2).transpose(1, 2)
        value = self.value_proj(pooled).flatten(2).transpose(1, 2)

        k_new, v_new = self.attention(query, key, value)
        return k_new, v_new


class GlobalWindowAttention(nn.Module):
    """
    窗口注意力 + 全局 token 融合
    """
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

        # 计算窗口内的相对位置索引
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
        """
        x: [B*nW, N, C]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 拼全局 token
        if global_k is not None and global_v is not None:
            B = global_k.shape[0]
            nW = B_ // B
            global_k = global_k.unsqueeze(1).expand(B, nW, -1, -1).reshape(B_, -1, C)
            global_v = global_v.unsqueeze(1).expand(B, nW, -1, -1).reshape(B_, -1, C)

            global_k = global_k.view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            global_v = global_v.view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            k = torch.cat([k, global_k], dim=2)
            v = torch.cat([v, global_v], dim=2)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 只给局部窗口部分加相对位置偏置
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
    """
    x: (B, H, W, C)
    """
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
    最终的注意力块：全局 → 窗口 + 全局 → FFN
    输入: [B, H*W, C]
    """
    def __init__(self, Nq, poolstep, dim, input_resolution, num_heads,
                 window_size=8, mlp_ratio=4., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size  # 这里我们用 8，能整除 128/64/32/16

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

        # 做一次全局 token
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        global_k, global_v = self.global_cross_attn(x_2d)

        # 窗口划分 + 融合全局 token
        x_windows = window_partition(x.view(B, H, W, C), self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        x_windows = self.window_attn(x_windows, global_k=global_k, global_v=global_v)

        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================
# 编码器 MiT —— 按 SegFormer 原始配置
# =============================
class MiT(nn.Module):
    def __init__(
        self,
        *,
        Nq=5,
        poolstep=(8, 4, 2, 1),
        # 输入 512 下的 1/4,1/8,1/16,1/32
        input_resolution=((128, 128), (64, 64), (32, 32), (16, 16)),
        # 原始 SegFormer-B0 的 dims
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers,
        use_checkpoint=True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # overlap patch embedding 配置，和论文一样：4x4 stride4，然后3x3 stride2...
        stage_kernel_stride_pad = (
            (4, 4, 0),  # 512 -> 128
            (3, 2, 1),  # 128 -> 64
            (3, 2, 1),  # 64  -> 32
            (3, 2, 1),  # 32  -> 16
        )

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for stage_idx, (
            (dim_in, dim_out),
            (kernel, stride, padding),
            num_layer_per_stage,
            ff_exp,
            num_head,
            _,
        ) in enumerate(zip(
            dim_pairs,
            stage_kernel_stride_pad,
            num_layers,
            ff_expansion,
            heads,
            reduction_ratio,
        )):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            blocks = nn.ModuleList()
            for _ in range(num_layer_per_stage):
                blocks.append(
                    GlobalBlock(
                        Nq=Nq,
                        poolstep=poolstep[stage_idx],
                        dim=dim_out,
                        input_resolution=input_resolution[stage_idx],
                        num_heads=num_head,
                        window_size=8,      # 按 128/64/32/16 整除
                        mlp_ratio=ff_exp,
                        drop_path=0.,
                    )
                )

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                blocks
            ]))

    def forward(self, x, return_layer_outputs=False):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, return_layer_outputs)
        else:
            return self._forward(x, return_layer_outputs)

    def _forward(self, x, return_layer_outputs=False):
        B, C, H, W = x.shape
        cur_H, cur_W = H, W
        layer_outputs = []

        for (get_overlap_patches, overlap_embed, blocks) in self.stages:
            x_unfold = get_overlap_patches(x)                # (B, C*k*k, L)
            num_patches = x_unfold.shape[-1]
            ratio = int((cur_H * cur_W / num_patches) ** 0.5)
            new_H = cur_H // ratio
            new_W = cur_W // ratio

            x = x_unfold.view(B, -1, new_H, new_W)
            x = overlap_embed(x)                             # (B, dim_out, new_H, new_W)

            B_, C_, H_, W_ = x.shape
            x_seq = x.flatten(2).transpose(1, 2)             # (B, H_*W_, C_)

            for blk in blocks:
                x_seq = blk(x_seq)

            x = x_seq.transpose(1, 2).view(B_, C_, H_, W_)

            layer_outputs.append(x)
            cur_H, cur_W = H_, W_

        return x if not return_layer_outputs else layer_outputs


# =============================
# SegFormer 解码头（和原版一样：1x1+上采样+concat）
# =============================
class Segformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes=4,
        poolstep=(8, 4, 2, 1),
        input_resolution=((128, 128), (64, 64), (32, 32), (16, 16)),
        # 原始 SegFormer-B0
        dims=(32, 64, 160, 256),
        heads=(1, 2, 5, 8),
        ff_expansion=(4, 4, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=(2, 2, 2, 2),
        channels=3,
        decoder_dim=256,
        use_checkpoint=True,
    ):
        super().__init__()

        dims, heads, ff_expansion, reduction_ratio, num_layers = map(
            partial(cast_tuple, depth=4),
            (dims, heads, ff_expansion, reduction_ratio, num_layers)
        )

        self.mit = MiT(
            Nq=5,
            poolstep=poolstep,
            input_resolution=input_resolution,
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
        )

        # 把四个尺度都映射到 decoder_dim，然后上采样回 512
        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1),
                nn.Upsample(scale_factor=2 ** (i + 2), mode="bilinear", align_corners=False)
            ) for i, dim in enumerate(dims)
        ])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
        feats = self.mit(x, return_layer_outputs=True)   # 4 个 feature: 128,64,32,16
        fused = [proj(f) for f, proj in zip(feats, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        out = self.to_segmentation(fused)
        return out


def segformermodelHA(num_classes=2, use_checkpoint=True):
    return Segformer(
        num_classes=num_classes,
        dims=(64, 128, 256, 512),
        heads=(1, 1, 1, 1),
        ff_expansion=(4, 4, 4, 4),
        reduction_ratio=(1, 1, 1, 1),
        num_layers=4,
        channels=3,
        decoder_dim=256,
        use_checkpoint=True,
    )


if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)

    net = segformermodel(num_classes=2, ).cuda()
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
